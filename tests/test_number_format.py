import pytest

from angle_cal.number_format import format_number


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (0, 5, "0"),
        (12.3456, 4, "12.35"),
        (12_345.6, 4, "12350"),
        (100_000, 5, "100000"),
        (-100_000, 5, "-100000"),
        (0.00000123456, 4, "0.000001235"),
    ],
)
def test_format_number_keeps_plain_decimal_through_one_hundred_thousand(value, digits, expected):
    assert format_number(value, digits) == expected


def test_format_number_uses_scientific_notation_only_above_threshold():
    assert format_number(100_000.01, 5) == "1.0000e+05"
    assert format_number(-2_500_000, 3) == "-2.50e+06"
