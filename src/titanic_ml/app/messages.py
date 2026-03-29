"""Flavour messages for prediction results.

Randomly selected messages add personality to predictions. Messages are
organised by outcome (survived / not survived) and confidence tier.

To add a new message, append a string to the appropriate list below.
Character-specific Easter eggs are checked first and override the
confidence-based pool when the input data matches.

Confidence tiers:
    HIGH   — 90 %+
    MEDIUM — 70 – 89 %
    LOW    — 50 – 69 %
"""

import random
from typing import Any, Dict, Optional

# ── Survived messages ──────────────────────────────────────────────

SURVIVED_HIGH = [
    "Smooth sailing — the lifeboats were practically reserved for you.",
    "First-class treatment pays off. Welcome aboard the rescue ship.",
    "The sea was no match for you. You'd make the Carpathia proud.",
    "You could've swum to New York and still made it.",
    "Even the iceberg stepped aside for you.",
]

SURVIVED_MEDIUM = [
    "The odds are in your favour — better grab a lifeboat early just in case.",
    "You'd survive, but you'd definitely be writing a strongly worded letter to White Star Line.",
    "Not the smoothest voyage, but you live to tell the tale.",
    "You'd make it, though your luggage might not be so lucky.",
    "Survival looks likely — just stay away from the bow at midnight.",
]

SURVIVED_LOW = [
    "It's a coin flip, but luck is on your side... barely.",
    "You'd survive by the skin of your teeth. Dramatic, but effective.",
    "The model gives you a pass, but it wasn't confident about it.",
    "You'd live, but only because someone else gave up their seat.",
    "Squeaking by — the kind of survival that makes a great dinner story.",
]

# ── Did not survive messages ──────────────────────────────────────

NOT_SURVIVED_HIGH = [
    "The iceberg sends its regards.",
    "This is not your voyage. The Atlantic has spoken.",
    "Even with a crystal ball, this one was written in the stars.",
    "The ship went down, and unfortunately, so did you.",
    "Some things are inevitable. This was one of them.",
]

NOT_SURVIVED_MEDIUM = [
    "Things aren't looking great. Should've listened to mum and stayed home.",
    "The odds aren't in your favour. Maybe try a different decade.",
    "Not great, not terrible... actually, no — it's pretty terrible.",
    "You drew the short straw. The very short straw. Won't even help you float...",
    "The Atlantic Ocean: 1. You: 0.",
]

NOT_SURVIVED_LOW = [
    "It could go either way, but the ship isn't betting on you.",
    "The margin is thin, and it's not tilting your way.",
    "Almost made it. Well, not really...",
    "A slightly different cabin and this could have gone the other way. Capitalism!",
]

# ── Character Easter eggs ─────────────────────────────────────────
# Keyed by (age, sex, pclass). Override the confidence-based pool.

CHARACTER_EASTER_EGGS: Dict[tuple, list] = {
    # Jack Dawson
    (20, "male", "3"): [
        "There was room on that door, and the model knows it.",
        "King of the world? The iceberg sends its regards.",
        "He could draw you like one of his French girls, but swimming - not his thing it seems.",
        "Jack: great artist, handy with the ladies, questionable survivalist.",
        "Doors are much like blankets, women tend to hog the whole thing :)",
    ],
    # Rose DeWitt Bukater
    (17, "female", "1"): [
        "She'll never let go... and neither will that survival rate.",
        "She promised she'd never let go. The model agrees.",
        "First class, strong will, and a door all to herself. Classic Rose.",
    ],
}


def get_flavour_message(
    survived: bool,
    confidence_pct: float,
    input_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a randomly selected flavour message for the prediction.

    Checks for character Easter eggs first. Falls back to confidence-tiered
    message pools.

    Args:
        survived: Whether the passenger survived.
        confidence_pct: Model confidence as a percentage (0-100).
        input_data: Original form data dict (keys: age, sex, pclass, etc.).

    Returns:
        A flavour message string.
    """
    # Check for character Easter eggs
    if input_data:
        key = (
            int(float(input_data.get("Age", input_data.get("age", 0)))),
            str(input_data.get("Sex", input_data.get("sex", ""))),
            str(input_data.get("Pclass", input_data.get("pclass", ""))),
        )
        if key in CHARACTER_EASTER_EGGS:
            return str(random.choice(CHARACTER_EASTER_EGGS[key]))  # noqa: S311  # nosec B311

    # Select pool based on outcome + confidence tier
    if survived:
        if confidence_pct >= 90:
            pool = SURVIVED_HIGH
        elif confidence_pct >= 70:
            pool = SURVIVED_MEDIUM
        else:
            pool = SURVIVED_LOW
    else:
        if confidence_pct >= 90:
            pool = NOT_SURVIVED_HIGH
        elif confidence_pct >= 70:
            pool = NOT_SURVIVED_MEDIUM
        else:
            pool = NOT_SURVIVED_LOW

    return str(random.choice(pool))  # noqa: S311  # nosec B311
