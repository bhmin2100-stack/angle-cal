from __future__ import annotations

import math


SCIENTIFIC_NOTATION_THRESHOLD = 100_000.0


def format_number(
    value: float,
    significant_digits: int = 5,
    scientific_threshold: float = SCIENTIFIC_NOTATION_THRESHOLD,
) -> str:
    """Format measurements without scientific notation until the threshold is exceeded."""
    number = float(value)
    digits = max(1, int(significant_digits))
    if not math.isfinite(number):
        return str(number)
    if abs(number) > float(scientific_threshold):
        return f"{number:.{digits - 1}e}"
    if number == 0:
        return "0"

    magnitude = math.floor(math.log10(abs(number)))
    decimal_places = digits - magnitude - 1
    rounded = round(number, decimal_places)
    text = f"{rounded:.{max(0, decimal_places)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0"} else text
