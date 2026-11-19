"""Every number as words for a voice that cannot read digits (pilot.system.spoken_digits)."""

import re
from datetime import date

import pytest

from pilot.system.spoken_digits import ordinal_any, words_for_voice

TODAY = date(2026, 11, 19)

CASES = [
    # The two Pocket got wrong on 19 November, and the rest of that day's comparison.
    ("Siobhan Nakamura called at 3:15.", "Siobhan Nakamura called at three fifteen."),
    ("That was in March 2026.", "That was in March twenty twenty-six."),
    ("It's 10:42.", "It's ten forty-two."),
    ("The rate went up to 3.5 percent.", "The rate went up to three point five percent."),
    ("The battery is at 64 percent.", "The battery is at sixty-four percent."),
    # Times
    ("The call is at 7pm.", "The call is at seven p m."),
    ("Meet at 9:05 a.m.", "Meet at nine oh five a m."),
    ("The train leaves at 23:40.", "The train leaves at eleven forty p m."),
    ("Lunch is 12:00–13:00.", "Lunch is twelve to one p m."),
    # Dates and years
    ("The deadline is 2026-12-01.", "The deadline is the first of December."),
    ("On Thu 19 Nov at 3pm.", "On Thursday the nineteenth at three p m."),
    ("Born on the 3rd of December 1999.", "Born on the third of December nineteen ninety-nine."),
    ("Sales grew between 2020-2026.", "Sales grew between twenty twenty to twenty twenty-six."),
    ("It opened in 2005.", "It opened in two thousand and five."),
    # Money
    ("It costs $1,299.99.", "It costs one thousand two hundred and ninety-nine dollars and ninety-nine cents."),
    ("They raised $2.5m and €40k.", "They raised two point five million dollars and forty thousand euros."),
    ("The fee is £5.", "The fee is five pounds."),
    # Percentages, decimals, ranges
    ("It is 3-5% over.", "It is three to five percent over."),
    ("Read pages 12-15.", "Read pages twelve to fifteen."),
    ("About 1.5 hours.", "About one point five hours."),
    ("It was -3°C, then 72°F.", "It was minus three degrees Celsius, then seventy-two degrees Fahrenheit."),
    # Ordinals and counts
    ("She finished 100th, he was 21st.", "She finished one hundredth, he was twenty-first."),
    (
        "We keep 2048 files for 4 200 000 users.",
        "We keep two thousand and forty-eight files for four million two hundred thousand users.",
    ),
    ("Agent 007 is in room 4B.", "Agent zero zero seven is in room four B."),
    # Phone-like digit groups, versions and addresses
    ("Call 555-0142.", "Call five five five, zero one four two."),
    ("Ring +44 20 7946 0018 today.", "Ring plus four four, two zero, seven nine four six, zero zero one eight today."),
    ("The office is (212) 555-0199.", "The office is two one two, five five five, zero one nine nine."),
    (
        "Version 1.2.0 is on 192.168.1.1.",
        "Version one point two point zero is on one nine two dot one six eight dot one dot one.",
    ),
]


@pytest.mark.parametrize(("written", "said"), CASES)
def test_every_number_reaches_the_voice_as_words(written, said):
    assert words_for_voice(written, TODAY) == said


@pytest.mark.parametrize(("written", "_said"), CASES)
def test_no_digit_is_left_and_the_conversion_is_idempotent(written, _said):
    once = words_for_voice(written, TODAY)
    assert not re.search(r"\d", once)
    assert words_for_voice(once, TODAY) == once


def test_text_without_numbers_is_unchanged():
    text = "I found two files called budget; which one do you mean?"
    assert words_for_voice(text, TODAY) == text


def test_ordinals_of_any_size():
    assert [ordinal_any(n) for n in (3, 31, 40, 100, 121, 1000)] == [
        "third",
        "thirty-first",
        "fortieth",
        "one hundredth",
        "one hundred and twenty-first",
        "one thousandth",
    ]
