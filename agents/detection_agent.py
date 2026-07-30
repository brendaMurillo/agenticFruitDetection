"""
Detection Agent
---------------
Job: figure out WHAT fruit each segmented region is.

LocalDetectionAgent: free/instant/offline. Uses color + shape heuristics
  computed during segmentation (dominant hue, aspect ratio, size). Works
  well when the two fruits are visually distinct (e.g. banana vs
  avocado), but can genuinely confuse fruits that share both a rough
  color AND a round shape -- apple, tomato, and orange all look similar
  to a hue+shape heuristic. If you know what you're photographing,
  passing --fruit1/--fruit2 on the command line skips this agent
  entirely and is always more reliable than auto-detection.

AIDetectionAgent: sends the cropped photo to Claude's vision API and
  asks it to name the fruit. Much better at telling visually-similar
  round fruits apart since it isn't limited to a hand-written rule
  table. Needs ANTHROPIC_API_KEY, internet, and costs a small amount
  per call.
"""

import json

from segmentation import encode_image_b64

KNOWN_FRUITS = [
    "apple", "banana", "mango", "strawberry",
    "tomato", "avocado", "orange", "watermelon",
]


class LocalDetectionAgent:
    def identify(self, fruit_data):
        hue = fruit_data["mean_hue"]
        val = fruit_data["mean_val"]
        aspect = fruit_data["aspect_ratio"]
        area = fruit_data["area_px"]

        # Very elongated -> banana, regardless of exact ripeness color
        if aspect > 2.1 and 15 <= hue <= 85:
            return {"fruit_type": "banana", "confidence_note": "shape: elongated"}

        # Moderately elongated + green/yellow/orange -> mango
        if 1.35 <= aspect <= 2.1 and hue <= 90:
            return {"fruit_type": "mango", "confidence_note": "shape: oval, warm-ish hue"}

        # Deep green, glossy, fairly compact/small -> avocado
        if 40 <= hue <= 95 and area < 40000 and aspect < 1.6 and val < 160:
            return {"fruit_type": "avocado", "confidence_note": "dark green, compact"}

        # Bright green, large -> whole watermelon
        if 45 <= hue <= 85 and area >= 60000:
            return {"fruit_type": "watermelon", "confidence_note": "large + green"}

        # Reds (wrap around hue 0/179)
        if hue <= 10 or hue >= 165:
            if area < 8000:
                return {"fruit_type": "strawberry", "confidence_note": "small + red"}
            elif area < 30000:
                return {"fruit_type": "tomato", "confidence_note": "medium + red, round"}
            else:
                return {"fruit_type": "apple", "confidence_note": "large + red, round"}

        # Orange/yellow, round, not elongated
        if 10 < hue <= 30:
            return {"fruit_type": "orange", "confidence_note": "round, orange hue"}

        return {
            "fruit_type": "apple",
            "confidence_note": "low-confidence fallback -- consider specifying --fruit1/--fruit2 manually",
        }


class AIDetectionAgent:
    def __init__(self, model="claude-haiku-4-5"):
        import anthropic  # lazy import: only needed for AI mode
        self.client = anthropic.Anthropic()
        self.model = model

    def identify(self, fruit_data):
        b64 = encode_image_b64(fruit_data["crop"])
        prompt = (
            "Identify the single fruit in this photo. Respond with ONLY raw JSON "
            "(no markdown fences), in exactly this shape: "
            '{"fruit_type": "<one of: ' + ", ".join(KNOWN_FRUITS) + '>", '
            '"confidence_note": "<one short phrase on what visual feature you used>"}'
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
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
        except json.JSONDecodeError:
            data = {"fruit_type": "unknown", "confidence_note": f"could not parse model response: {text[:80]}"}
        if data.get("fruit_type") not in KNOWN_FRUITS:
            data["fruit_type"] = data.get("fruit_type", "unknown")
        return data
