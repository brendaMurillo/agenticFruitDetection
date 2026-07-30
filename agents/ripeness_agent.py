"""
Ripeness Agent
--------------
Job: given a fruit crop + its type, estimate ripeness as a percentage.

Scale (both implementations share this scale, so results are comparable
regardless of mode):
    0%   = fully unripe
    100% = peak ripeness (ideal to eat right now)
    100-150% = past peak; increasing overripeness/spoilage

LocalRipenessAgent: HSV color-shift heuristic. Reliable for fruits that
  visibly change color as they ripen. Flags low confidence for oranges
  and watermelons, whose exterior color is a weak ripeness signal.

AIRipenessAgent: sends the crop to Claude's vision API along with a
  written rubric (color/texture anchors at 0/50/100/125/150%) so its
  judgment is grounded in specific visual criteria rather than a vague
  guess.
"""

import json

import numpy as np

from segmentation import encode_image_b64

# ---- Local heuristic profiles ---------------------------------------------

FRUIT_PROFILES = {
    "banana":     {"unripe_hue": (35, 85),  "ripe_hue": (18, 34),  "reliable": True},
    "mango":      {"unripe_hue": (40, 85),  "ripe_hue": (5, 30),   "reliable": True, "spot_sat_max": 110},
    "strawberry": {"unripe_hue": (35, 85),  "ripe_hue": (170, 8),  "reliable": True},
    "tomato":     {"unripe_hue": (35, 85),  "ripe_hue": (170, 8),  "reliable": True},
    "apple":      {"unripe_hue": (35, 85),  "ripe_hue": (170, 10), "reliable": True},
    "avocado":    {"darkness_based": True,                          "reliable": True},
    "orange":     {"unripe_hue": (40, 55),  "ripe_hue": (5, 25),   "reliable": False, "spot_sat_max": 110},
    "watermelon": {"unripe_hue": (45, 85),  "ripe_hue": (35, 50),  "reliable": False},
}

# ---- Rubric fed to the AI agent --------------------------------------------

RIPENESS_RUBRICS = {
    "banana": "0%: fully green, hard. 50%: green with yellow patches. 100%: fully "
              "yellow, at most 1-2 tiny brown flecks -- peak fresh-eating ripeness. "
              "125%: yellow mostly covered in brown spots. 150%: skin mostly "
              "brown/black, very soft -- past fresh eating, good for baking.",
    "mango": "0%: fully green, hard. 50%: green-yellow, slight red blush starting. "
             "100%: mostly yellow/orange/red with slight give when pressed -- peak "
             "ripeness. 125%: very soft-looking, wrinkling, dark spots. 150%: "
             "heavily wrinkled/darkened, likely overripe.",
    "strawberry": "0%: white/pale green shoulders. 50%: half red, half white/green. "
                  "100%: fully deep red, glossy -- peak ripeness. 125%: very dark "
                  "red, dull, maybe a soft spot. 150%: mushy, dark red/purple, "
                  "possible mold.",
    "tomato": "0%: fully green, hard. 50%: green-orange 'breaker' stage. 100%: "
              "fully red, glossy, firm-looking -- peak ripeness. 125%: deep red, "
              "wrinkled or soft-looking. 150%: very soft, splitting skin, possible mold.",
    "apple": "0%: green/hard-looking, no blush, matte skin. 50%: partial color "
             "developing. 100%: full rich color, glossy, firm -- peak ripeness. "
             "125%: wrinkling or soft/bruised patches visible. 150%: heavily "
             "wrinkled, soft, visible browning/bruising.",
    "avocado": "0%: bright glossy green, smooth, taut skin. 50%: color slightly "
               "darkening. 100%: dark green to near-black, skin slightly "
               "yielding-looking -- peak ripeness for eating. 125%: very dark, "
               "visible soft dents. 150%: shriveled, sunken, visibly overripe.",
    "orange": "Color is a WEAK ripeness signal for oranges -- they can be ripe "
              "while greenish and stay orange well past peak. Base your estimate "
              "mainly on skin texture: 100% = taut, smooth, uniform color. "
              "125%+ = wrinkled, dull, or soft spots visible. Say so explicitly "
              "if the photo doesn't give you enough to go on.",
    "watermelon": "Exterior rind color is NOT a reliable ripeness signal. A "
                  "visible yellow/cream 'ground spot' (where it rested on soil) is "
                  "a positive ripeness sign if you can see it. Otherwise treat any "
                  "score here as a low-confidence guess and say so.",
}


def _hue_frac(hue, sat, mask_bool, lo, hi, min_sat=40):
    if lo <= hi:
        band = (hue >= lo) & (hue <= hi)
    else:  # wraps around 0 (reds)
        band = (hue >= lo) | (hue <= hi)
    band &= (sat >= min_sat) & mask_bool
    total = mask_bool.sum()
    return band.sum() / total if total > 0 else 0.0


class LocalRipenessAgent:
    def assess(self, fruit_data, fruit_type):
        import cv2

        profile = FRUIT_PROFILES[fruit_type]
        crop, roi_mask = fruit_data["crop"], fruit_data["mask"]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask_bool = roi_mask > 0

        if profile.get("darkness_based"):
            vals = v[mask_bool]
            if vals.size == 0:
                return {"ripeness_percent": 0.0, "cues": [], "reliability_note": "no pixels detected"}
            mean_v = vals.mean()
            # Rough anchors: V~180 bright/unripe (0%), V~100 peak-ripe (100%),
            # V~50 shriveled/overripe (150%). Recalibrate these against your
            # own lighting if avocado scores look off.
            if mean_v >= 100:
                base = np.clip((180 - mean_v) / (180 - 100) * 100, 0, 100)
            else:
                base = np.clip(100 + (100 - mean_v) / (100 - 50) * 50, 100, 150)
            return {
                "ripeness_percent": float(base),
                "cues": [f"mean brightness {mean_v:.0f}/255 (darker = riper)"],
                "reliability_note": "",
            }

        unripe_lo, unripe_hi = profile["unripe_hue"]
        ripe_lo, ripe_hi = profile["ripe_hue"]

        # Judge color only on the non-dark ("bright") part of the skin --
        # dark/spotted pixels don't reliably match either hue band, so
        # counting them in the color fraction just dilutes it, making a
        # heavily-spotted (genuinely riper) fruit look LESS ripe-colored
        # than a clean one. Darkness is tracked separately below and always
        # adds to the score, instead of being diluted away or gated behind
        # a threshold.
        bright_mask = (v >= 90) & mask_bool
        total = mask_bool.sum()

        # For most fruits, low brightness alone is a good decay signal. But
        # for fruits whose HEALTHY ripe color is itself dark-leaning (like
        # mango's red blush, which can be nearly as dark as real bruising),
        # brightness alone false-positives on vivid, healthy coloring. Where
        # a profile sets spot_sat_max, also require low saturation -- real
        # decay is dull/muddy, whereas healthy blush stays vivid/saturated.
        spot_sat_max = profile.get("spot_sat_max")
        if spot_sat_max is not None:
            dark_mask = (v < 90) & (s < spot_sat_max) & mask_bool
            dark_frac = dark_mask.sum() / total if total > 0 else 0.0
        else:
            dark_frac = 1 - (bright_mask.sum() / total) if total > 0 else 0.0

        unripe_frac = _hue_frac(h, s, bright_mask, unripe_lo, unripe_hi)
        ripe_frac = _hue_frac(h, s, bright_mask, ripe_lo, ripe_hi)
        denom = unripe_frac + ripe_frac
        color_base = (ripe_frac / denom * 100) if denom > 0 else 50.0

        # Any real spotting is treated as a hard signal of "past peak," not
        # just a nudge on top of the color read. Below 3% dark pixels, go by
        # color alone (0-100%). From 3-10%, land in the "just past peak,
        # still good to eat" range. Above 10%, land solidly in "overripe /
        # best used in baking" territory. These thresholds intentionally
        # override the color reading once triggered -- real spotting means
        # real ripening regardless of how much yellow is left.
        dark_pct = dark_frac * 100
        if dark_pct < 3:
            base = min(100, color_base)
        elif dark_pct < 10:
            base = 108 + (dark_pct - 3) / 7 * (130 - 108)
        else:
            base = 130 + min(1.0, (dark_pct - 10) / 40) * (150 - 130)

        note = "" if profile.get("reliable", True) else (
            "Low confidence: this fruit's skin color doesn't change much as it "
            "ripens -- treat this number as a rough guess."
        )
        cues = [f"{ripe_frac * 100:.0f}% ripe-colored (of non-dark skin)", f"{dark_frac * 100:.0f}% dark/spotted"]
        return {"ripeness_percent": float(np.clip(base, 0, 150)), "cues": cues, "reliability_note": note}


class AIRipenessAgent:
    def __init__(self, model="claude-haiku-4-5"):
        import anthropic  # lazy import: only needed for AI mode
        self.client = anthropic.Anthropic()
        self.model = model

    def assess(self, fruit_data, fruit_type):
        b64 = encode_image_b64(fruit_data["crop"])
        rubric = RIPENESS_RUBRICS.get(fruit_type, "Use general color/texture judgment.")
        prompt = (
            f"This is a photo of a {fruit_type}. Estimate its ripeness using this "
            f"rubric:\n{rubric}\n\n"
            "Ripeness can go above 100% for overripe fruit (cap at 150%). "
            "Respond with ONLY raw JSON (no markdown fences): "
            '{"ripeness_percent": <number 0-150>, "cues": ["<short visual '
            'observation>", ...], "reliability_note": "<empty string, or a short '
            'caveat if this fruit type is hard to judge from color alone>"}'
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
            data["ripeness_percent"] = float(data.get("ripeness_percent", 50))
        except (json.JSONDecodeError, ValueError, TypeError):
            data = {"ripeness_percent": 50.0, "cues": [], "reliability_note": f"could not parse model response: {text[:80]}"}
        return data
