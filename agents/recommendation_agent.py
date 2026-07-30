"""
Recommendation Agent
---------------------
Job: turn (fruit type, ripeness %) into a short, actionable, ACCURATE tip.

Key food-science fact this bakes in: not all fruit keeps ripening after
it's picked.
  - Climacteric fruits (banana, mango, avocado, tomato, apple) keep
    ripening off the plant -- "wait a couple more days" is real advice.
  - Non-climacteric fruits (strawberry, orange/citrus, watermelon) do
    NOT ripen further once picked -- telling someone to "wait" for one
    of these to ripen more would just be wrong. The honest advice is
    closer to "what you see is what you get."

LocalRecommendationAgent: rule-based lookup using the table below.
AIRecommendationAgent: text-only Claude call, given the same facts as
  context so its phrasing is natural but still grounded (it isn't free
  to invent ripening advice that contradicts fruit biology).
"""

FRUIT_INFO = {
    "banana": {
        "climacteric": True, "wait_days": "2-4 days",
        "speed_tip": "a paper bag (add an apple or another banana to speed it up)",
        "overripe_use": "perfect for banana bread, smoothies, or pancakes",
    },
    "mango": {
        "climacteric": True, "wait_days": "2-4 days",
        "speed_tip": "a paper bag at room temperature",
        "overripe_use": "great blended into a smoothie or pureed for salsa",
    },
    "avocado": {
        "climacteric": True, "wait_days": "2-4 days",
        "speed_tip": "a paper bag with a banana or apple nearby",
        "overripe_use": "still great mashed for guacamole, unless it's stringy/mushy all the way through",
    },
    "tomato": {
        "climacteric": True, "wait_days": "1-3 days",
        "speed_tip": "room temperature, stem-side down, out of direct sun",
        "overripe_use": "good for sauce, soup, or salsa",
    },
    "apple": {
        "climacteric": True, "wait_days": "2-5 days",
        "speed_tip": "room temperature (most store-bought apples are already fully ripe, though)",
        "overripe_use": "good for applesauce, baking, or pie filling",
    },
    "strawberry": {
        "climacteric": False,
        "note": "Strawberries don't ripen further once picked -- what you see is what you get.",
        "overripe_use": "use immediately in a smoothie or jam; check for mold first",
    },
    "orange": {
        "climacteric": False,
        "note": "Citrus doesn't ripen further once picked, and color isn't a reliable "
                "ripeness cue for oranges anyway -- taste is the real test.",
        "overripe_use": "if any part feels soft or moldy it's past use; otherwise juice it",
    },
    "watermelon": {
        "climacteric": False,
        "note": "Watermelon doesn't ripen further once picked. A hollow sound when "
                "tapped and a creamy yellow ground spot are better ripeness cues than "
                "a photo can capture.",
        "overripe_use": "if it's actually gone soft/fermented rather than just ripe, it's best composted",
    },
}


class LocalRecommendationAgent:
    def recommend(self, fruit_type, ripeness_percent, cues=None):
        info = FRUIT_INFO[fruit_type]

        if not info["climacteric"]:
            if ripeness_percent > 115:
                return f"{info['note']} At this ripeness it's past its best -- {info['overripe_use']}."
            return info["note"]

        if ripeness_percent < 40:
            return (f"Still quite unripe. Give it about {info['wait_days']} at room "
                     f"temperature -- {info['speed_tip']} helps it along.")
        elif ripeness_percent < 85:
            return (f"Getting there -- probably a day or so from peak ripeness. "
                     f"{info['speed_tip'].capitalize()} if you want it sooner.")
        elif ripeness_percent <= 108:
            return "Perfectly ripe -- great to eat now!"
        elif ripeness_percent <= 130:
            return "Very ripe, just past peak fresh-eating -- still good, best used soon."
        else:
            return f"Overripe for fresh eating -- {info['overripe_use']}."


class AIRecommendationAgent:
    def __init__(self, model="claude-haiku-4-5"):
        import anthropic  # lazy import: only needed for AI mode
        self.client = anthropic.Anthropic()
        self.model = model

    def recommend(self, fruit_type, ripeness_percent, cues=None):
        info = FRUIT_INFO[fruit_type]
        facts = (
            f"Ripens further after picking: {info['climacteric']}. "
            + (f"Speeds up with: {info['speed_tip']}. " if info.get("speed_tip") else "")
            + (f"Note: {info['note']} " if info.get("note") else "")
            + f"If overripe, best use: {info['overripe_use']}."
        )
        cues_str = "; ".join(cues) if cues else "none given"
        prompt = (
            f"A {fruit_type} was just measured at {ripeness_percent:.0f}% ripeness "
            f"(0%=unripe, 100%=peak ripeness, >100%=past peak). Visual cues: {cues_str}. "
            f"Facts to stay consistent with: {facts}\n\n"
            "Write ONE short, friendly, actionable sentence (max ~30 words) telling "
            "the person what to do. Do not contradict the facts above. Respond with "
            "plain text only, no JSON, no markdown."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
