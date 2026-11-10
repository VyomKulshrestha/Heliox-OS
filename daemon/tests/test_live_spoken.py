"""Heliox Live understands slips that are spoken: the transcript of a speech recogniser writes dates and numbers as
words, drops punctuation and capitals, and the detectors parse written forms.

Precision first: a long list of ordinary spoken sentences must raise nothing, and every card that comes from
speech quotes the words that were said and reads like a person talking."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from pilot.live.commitments import find_commitments
from pilot.live.detectors import SPEECH_KINDS, TextContext, detect
from pilot.live.engine import say
from pilot.live.spoken import (
    cardinal_words,
    date_words,
    normalise,
    number_words,
    ordinal_words,
    time_words,
    to_written,
    year_words,
)

TUESDAY = date(2026, 11, 10)  # 12 November 2026 is a Thursday
NOW = datetime(2026, 11, 9, 11, 0)  # a Monday


def spoken(text: str) -> list:
    return detect(text, TextContext(today=TUESDAY, source="speech", hour=11), kinds=SPEECH_KINDS)


def typed(text: str) -> list:
    return detect(text, TextContext(today=TUESDAY, app="slack.exe", title="#team - Slack", hour=11))


def aloud(text: str) -> list[str]:
    return [say(c.title, c.message, "speech") for c in spoken(text)]


# ── the normaliser, as a pure function ──


@pytest.mark.parametrize(
    ("said", "written"),
    [
        # dates
        ("monday the twelfth of november", "monday the 12th of november"),
        ("monday november twelfth", "monday november 12th"),
        ("monday the 12th of november", "monday the 12th of november"),
        ("monday twelfth nov", "monday 12th nov"),
        ("twelfth of november", "12th of november"),
        ("the thirty first of november", "the 31st of november"),
        ("the thirty-first of november", "the 31st of november"),
        ("november twenty second", "november 22nd"),
        ("tuesday the fourteenth", "tuesday the 14th"),
        ("friday the third", "friday the 3rd"),
        ("monday the twelve of november", "monday the 12 of november"),
        ("see you november thirty at ten", "see you november 30 at 10"),
        ("on the twelfth", "on the 12th"),
        ("by the twentieth", "by the 20th"),
        ("the fifth of may", "the 5th of may"),
        ("friday may fifth", "friday may 5th"),
        # years
        ("the twelfth of november twenty twenty six", "the 12th of november 2026"),
        ("the twelfth of november two thousand and twenty six", "the 12th of november 2026"),
        ("november twelfth nineteen ninety nine", "november 12th 1999"),
        ("november twelfth twenty oh five", "november 12th 2005"),
        # times
        ("at ten", "at 10"),
        ("ten a m", "10 am"),
        ("ten a.m.", "10 am"),
        ("ten AM", "10 am"),
        ("ten p m", "10 pm"),
        ("ten thirty a m", "10:30 am"),
        ("at ten thirty", "at 10:30"),
        ("at ten oh five", "at 10:05"),
        ("at ten forty five p m", "at 10:45 pm"),
        ("at ten o'clock", "at 10:00"),
        ("ten o'clock", "10:00"),
        ("half past two", "2:30"),
        ("half past two p m", "2:30 pm"),
        ("quarter past two", "2:15"),
        ("a quarter to five", "4:45"),
        ("quarter to five", "4:45"),
        ("ten past four", "4:10"),
        ("at twenty to five", "at 4:40"),
        ("at fourteen thirty p m", "at 14:30 pm"),
        # cardinals in sums and percentages
        ("twelve times fifteen is one seventy", "12 × 15 = 170"),
        ("twelve times fifteen is a hundred and eighty", "12 × 15 = 180"),
        ("two thousand five hundred plus three hundred is three thousand", "2500 + 300 = 3000"),
        ("twenty five hundred minus one fifty five is two thousand three hundred and forty five", "2500 - 155 = 2345"),
        ("two hundred divided by four is fifty", "200 ÷ 4 = 50"),
        ("two point five times four is ten", "2.5 × 4 = 10"),
        ("twelve multiplied by fifteen equals one eighty", "12 × 15 = 180"),
        ("12 times 15 is 170", "12 × 15 = 170"),
        ("twenty percent of three hundred is seventy", "20% of 300 is 70"),
        ("twenty per cent of three hundred", "20% of 300"),
        ("twenty percent of three hundred, so seventy", "20% of 300, so 70"),
        ("a hundred percent", "100%"),
        ("two point five percent", "2.5%"),
        # homophones, only where the number reading is certain
        ("ten plus ate is eighteen", "10 + 8 = 18"),
        ("four plus for is eight", "4 + 4 = 8"),
        ("for times to is eight", "4 × 2 = 8"),
        ("won plus won is two", "1 + 1 = 2"),
        ("for percent of two hundred is eight", "4% of 200 is 8"),
        ("meet at for p m", "meet at 4 pm"),
        ("ate o'clock", "8:00"),
        # relative days lose their comma in speech
        ("tomorrow thursday the twelfth", "tomorrow, thursday the 12th"),
        ("today monday", "today, monday"),
    ],
)
def test_spoken_forms_become_the_written_forms_the_detectors_parse(said, written):
    assert to_written(said) == written


@pytest.mark.parametrize(
    "text",
    [
        "one of the two of us",
        "for the fourth time",
        "the second before you hit send",
        "may I have two",
        "march on",
        "wait a second",
        "he came second in the race",
        "we may second the motion",
        "you may first read it",
        "on the second floor",
        "the third floor",
        "twenty one pilots",
        "I want to go to the park for two hours",
        "it takes four to six weeks",
        "one by one",
        "two for one",
        "ten to one odds",
        "from nine to five",
        "I have twelve and fifteen",
        "the number fifteen",
        "sixty seconds",
        "to be or not to be",
        "I need for p m to be released",
    ],
)
def test_words_that_are_not_dates_times_or_sums_stay_words(text):
    assert to_written(text) == text


@pytest.mark.parametrize(
    ("text", "written"),
    [
        ("I ate at eight", "I ate at 8"),  # a number after "at", which no detector reads as a slip
        ("I'll see you at ten past the hour", "I'll see you at 10 past the hour"),
        ("at ten percent", "at 10%"),
    ],
)
def test_the_only_conversion_in_these_look_alikes_is_a_plain_number(text, written):
    assert to_written(text) == written


def test_the_written_form_of_text_that_needs_nothing_is_the_text_itself():
    text = "Remind me that the launch call is Monday 12 November at 10:30 am, 20% of 300 is 60."
    assert to_written(text) == text
    assert not normalise(text).changed


def test_what_was_said_is_recoverable_for_any_stretch_of_the_written_text():
    said = "remind me that the launch is monday the twelfth of november at ten thirty a m please"
    result = normalise(said)
    assert result.text == "remind me that the launch is monday the 12th of november at 10:30 am please"
    start = result.text.index("monday")
    assert result.said(start, result.text.index(" at 10")) == "monday the twelfth of november"
    assert result.said(result.text.index("10:30"), result.text.index(" please")) == "ten thirty a m"
    assert result.said(0, len(result.text)) == said


def test_recogniser_punctuation_and_capitals_do_not_matter():
    for said in (
        "Monday, the twelfth of November, at ten.",
        "MONDAY THE TWELFTH OF NOVEMBER AT TEN",
        "Monday the Twelfth of November.",
    ):
        [card] = spoken(said)
        assert card.title == "The twelfth of November is a Thursday"
        assert card.message == "You said Monday."


# ── words for numbers, dates and times ──


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (7, "seven"),
        (12, "twelve"),
        (21, "twenty-one"),
        (60, "sixty"),
        (100, "one hundred"),
        (170, "one hundred and seventy"),
        (180, "one hundred and eighty"),
        (1000, "one thousand"),
        (2345, "two thousand three hundred and forty-five"),
        (2005, "two thousand and five"),
        (-3, "minus three"),
        (1_250_000, "one million two hundred and fifty thousand"),
    ],
)
def test_cardinal_words(n, words):
    assert cardinal_words(n) == words


def test_dates_numbers_and_times_in_words():
    assert [ordinal_words(n) for n in (1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 20, 21, 22, 30, 31)] == [
        "first", "second", "third", "fourth", "fifth", "eighth", "ninth", "eleventh", "twelfth", "thirteenth",
        "twentieth", "twenty-first", "twenty-second", "thirtieth", "thirty-first",
    ]  # fmt: skip
    assert date_words(12, "November") == "the twelfth of November"
    assert date_words(31, "November", article=False) == "thirty-first of November"
    assert [year_words(y) for y in (2026, 2005, 2000, 1999, 2100)] == [
        "twenty twenty-six", "two thousand and five", "two thousand", "nineteen ninety-nine", "two thousand one hundred",
    ]  # fmt: skip
    assert number_words("170") == "one hundred and seventy"
    assert number_words("-3.5") == "minus three point five"
    assert number_words("1,200") == "one thousand two hundred"
    assert [time_words(10), time_words(10, 30), time_words(2, 5), time_words(14, 30)] == [
        "ten", "ten thirty", "two oh five", "fourteen thirty",
    ]  # fmt: skip


# ── what Heliox says, and what the card quotes ──


@pytest.mark.parametrize(
    ("said", "title", "reads_aloud", "evidence"),
    [
        (
            "Remind me that the launch call is Monday the twelfth of November at ten",
            "The twelfth of November is a Thursday",
            "The twelfth of November is a Thursday. You said Monday.",
            "Monday the twelfth of November",
        ),
        (
            "launch is monday november twelfth",
            "The twelfth of November is a Thursday",
            "The twelfth of November is a Thursday. You said Monday.",
            "monday november twelfth",
        ),
        (
            "launch is on monday twelfth nov twenty twenty six",
            "The twelfth of November is a Thursday",
            "The twelfth of November is a Thursday. You said Monday.",
            "monday twelfth nov twenty twenty six",
        ),
        (
            "the party is on the thirty first of november",
            "There's no thirty-first of November",
            "There's no thirty-first of November. November has thirty days.",
            "thirty first of november",
        ),
        (
            "see you february twenty ninth twenty twenty seven",
            "There's no twenty-ninth of February twenty twenty-seven",
            "There's no twenty-ninth of February twenty twenty-seven. Twenty twenty-seven isn't a leap year.",
            "february twenty ninth twenty twenty seven",
        ),
        (
            "tomorrow thursday the twelfth",
            "Tomorrow is Wednesday",
            "Tomorrow is Wednesday. You said Thursday.",
            "tomorrow thursday",
        ),
        (
            "today is wednesday",
            "Today is Tuesday",
            "Today is Tuesday. You said Wednesday.",
            "today is wednesday",
        ),
        (
            "tomorrow thursday the twelfth of november",
            "The twelfth of November isn't tomorrow",
            "The twelfth of November isn't tomorrow. Tomorrow is Wednesday the eleventh of November.",
            "tomorrow thursday",
        ),
        (
            "twenty percent of three hundred is seventy",
            "Twenty percent of three hundred is sixty",
            "Twenty percent of three hundred is sixty. You said seventy.",
            "twenty percent of three hundred is seventy",
        ),
        (
            "twelve times fifteen is one seventy",
            "Twelve times fifteen is one hundred and eighty",
            "Twelve times fifteen is one hundred and eighty. You said one seventy.",
            "twelve times fifteen is one seventy",
        ),
        (
            "two hundred divided by four is fifty five",
            "Two hundred divided by four is fifty",
            "Two hundred divided by four is fifty. You said fifty five.",
            "two hundred divided by four is fifty five",
        ),
        (
            "fifteen minus nine is five",
            "Fifteen minus nine is six",
            "Fifteen minus nine is six. You said five.",
            "fifteen minus nine is five",
        ),
        (
            "ten plus ate is nineteen",
            "Ten plus eight is eighteen",
            "Ten plus eight is eighteen. You said nineteen.",
            "ten plus ate is nineteen",
        ),
        (
            "two point five times four is nine",
            "Two point five times four is ten",
            "Two point five times four is ten. You said nine.",
            "two point five times four is nine",
        ),
        (
            "meet at fourteen thirty p m",
            "Fourteen thirty p m is two thirty p m",
            "Fourteen thirty p m is two thirty p m. It mixes the twenty-four hour and twelve hour clocks.",
            "fourteen thirty p m",
        ),
        (
            "be there at thirteen a m",
            "There's no thirteen a m",
            "There's no thirteen a m. It mixes the twenty-four hour and twelve hour clocks.",
            "thirteen a m",
        ),
        (
            "at twenty four thirty",
            "There's no twenty-four thirty",
            "There's no twenty-four thirty. A day ends at twenty-three fifty-nine.",
            "twenty four thirty",
        ),
    ],
)
def test_a_spoken_slip_is_caught_quoted_in_the_words_said_and_read_like_a_person(said, title, reads_aloud, evidence):
    [card] = spoken(said)
    assert card.title == title
    assert say(card.title, card.message, "speech") == reads_aloud
    assert card.evidence == evidence
    assert evidence in said
    assert "×" not in reads_aloud and "%" not in reads_aloud and '"' not in reads_aloud


def test_the_spoken_fix_is_the_sentence_as_said_with_the_right_weekday():
    [card] = spoken("the call is monday the twelfth of november")
    assert card.fix == "thursday the twelfth of november" and card.fix_kind == "replace"
    [card] = spoken("tomorrow thursday the twelfth of november")
    assert card.fix == "thursday" and card.fix_kind == "replace"
    [card] = spoken("twelve times fifteen is one seventy")
    assert card.fix == "" and card.fix_kind == ""  # nothing to paste into speech


def test_a_spoken_year_is_read_so_a_correct_future_date_is_not_a_slip():
    assert spoken("the launch is thursday the twelfth of november twenty twenty six") == []
    [card] = spoken("the launch is thursday the twelfth of november twenty twenty seven")
    assert card.title == "The twelfth of November is a Friday"


def test_the_same_slip_typed_still_reads_the_way_it_always_did():
    [card] = typed("Launch call is Monday 12 Nov at 10am")
    assert (card.title, card.message) == ("12 November is a Thursday", "You wrote Monday.")
    [card] = typed("12 x 15 = 170")
    assert card.title == "12 × 15 is 180" and card.message == "You wrote 170."
    [card] = typed("20% of 300 is 70")
    assert card.title == "20% of 300 is 60"
    [card] = typed("There is a launch on 31 November")
    assert (card.title, card.message) == ("There's no 31 November", "November has 30 days.")
    [card] = typed("Tomorrow, Thursday the 12th.")
    assert (card.title, card.message) == ("Tomorrow is Wednesday", "You wrote Thursday.")


def test_a_speech_card_for_written_forms_is_worded_for_speech_too():
    # a recogniser sometimes writes digits: the card still reads aloud without symbols
    [card] = spoken("monday 12 nov")
    assert say(card.title, card.message, "speech") == "The twelfth of November is a Thursday. You said Monday."
    [card] = spoken("12 x 15 = 170")
    assert say(card.title, card.message, "speech").startswith("Twelve times fifteen is one hundred and eighty.")
    [card] = spoken("20% of 300 is 70")
    assert say(card.title, card.message, "speech") == "Twenty percent of three hundred is sixty. You said 70."


def test_why_explains_a_spoken_card_in_words_too():
    [card] = spoken("the call is monday the twelfth of november")
    assert card.detail == "If you meant Monday, that's the ninth of November."
    [card] = spoken("tomorrow thursday the twelfth")
    assert card.detail == "The next Thursday is the twelfth of November."


# ── precision: ordinary speech raises nothing ──

QUIET = [
    "I'll see you at ten",
    "one of the two of us",
    "for the fourth time",
    "the second before you hit send",
    "may I have two",
    "march on",
    "I ate at eight",
    "wait a second",
    "we may second the motion",
    "on the second floor",
    "i'll be there at ten thirty",
    "call me at half past two",
    "meet me at a quarter to five",
    "let's meet at ten a m on thursday",
    "the call is thursday the twelfth of november at ten",
    "the call is thursday november twelfth",
    "the call is on thursday twelfth nov",
    "the launch is thursday the twelfth of november twenty twenty six",
    "tomorrow is wednesday the eleventh of november",
    "today is tuesday",
    "tomorrow wednesday at ten",
    "yesterday monday we shipped it",
    "tomorrow friday we ship",
    "twelve times fifteen is one hundred and eighty",
    "twelve times fifteen is one eighty",
    "twenty percent of three hundred is sixty",
    "twenty five percent of two hundred is fifty",
    "ten percent of two fifty is twenty five",
    "two plus two is four",
    "ten plus ate is eighteen",
    "for times four is sixteen",
    "two hundred divided by four is fifty",
    "two thousand five hundred plus three hundred is two thousand eight hundred",
    "a hundred percent sure",
    "there are thirty days in june",
    "thirty days hath september",
    "february has twenty eight days",
    "it is the twenty first century",
    "for two hours",
    "it takes four to six weeks",
    "ten to one odds",
    "from nine to five",
    "I have three hundred and sixty five days",
    "the fifth of may is a wednesday",
    "friday the thirteenth of november",
    "sunday the first time I saw it",
    "I may march on monday",
    "to be or not to be",
    "one by one",
    "you need to be there at eleven fifteen",
    "i'll be home at seven",
    "the second of all",
    "my manager is away today",
    "the budget is twelve hundred",
    "she scored four to two",
    "Monday, the ninth of November, at ten.",
]


def test_there_are_at_least_forty_ordinary_spoken_sentences_and_each_stays_quiet():
    assert len(QUIET) >= 40
    for sentence in QUIET:
        assert spoken(sentence) == [], sentence


def test_ordinary_spoken_sentences_raise_no_commitment_offer_either():
    ctx = TextContext(today=NOW.date(), source="speech")
    for sentence in QUIET:
        assert find_commitments(sentence, ctx, NOW) == [], sentence


def test_quiet_sentences_also_stay_quiet_with_a_recogniser_that_adds_capitals_and_commas():
    for sentence in QUIET:
        dressed = sentence[:1].upper() + sentence[1:] + "."
        assert spoken(dressed) == [], dressed


# ── promises, spoken ──


def offers(text: str, now: datetime = NOW) -> list[str]:
    ctx = TextContext(today=now.date(), source="speech")
    return [c.fix for c in find_commitments(text, ctx, now)]


@pytest.mark.parametrize(
    ("said", "fix", "title"),
    [
        (
            "i'll send priya the deck by friday",
            "Remind me on Friday at 10am to send priya the deck",
            "Want a reminder on Friday at ten a m?",
        ),
        (
            "let me get back to you tomorrow morning",
            "Remind me tomorrow at 10am to get back to them",
            "Want a reminder tomorrow at ten a m?",
        ),
        (
            "i will call the bank on the twelfth",
            "Remind me on Thursday at 10am to call the bank",
            "Want a reminder on Thursday at ten a m?",
        ),
        (
            "ok so i will review the pr by three p m",
            "Remind me at 2pm to review the pr",
            "Want a reminder today at two p m?",
        ),
        (
            "I’ll update the doc on the twelfth of november at ten a m",
            "Remind me on Thursday at 9am to update the doc",
            "Want a reminder on Thursday at nine a m?",
        ),
        (
            "i'll ping sam tonight",
            "Remind me at 7pm to ping sam",
            "Want a reminder today at seven p m?",
        ),
        (
            "i can share the numbers by end of day",
            "Remind me at 4pm to share the numbers",
            "Want a reminder today at four p m?",
        ),
        (
            "i'm going to call mum on the eighteenth of november",
            "Remind me on 18 November at 10am to call mum",
            "Want a reminder on the eighteenth of November at ten a m?",
        ),
    ],
)
def test_a_spoken_promise_gets_the_same_offer_as_a_typed_one(said, fix, title):
    ctx = TextContext(today=NOW.date(), source="speech")
    [offer] = find_commitments(said, ctx, NOW)
    assert offer.fix == fix and offer.title == title
    assert offer.evidence in said


def test_a_spoken_promise_quotes_the_words_and_ignores_the_next_clause():
    ctx = TextContext(today=NOW.date(), source="speech")
    [offer] = find_commitments("ok i'll send priya the deck by friday and then i'll call bob tomorrow", ctx, NOW)
    assert offer.fix == "Remind me on Friday at 10am to send priya the deck"
    assert offer.message == "You said you'd send priya the deck by friday."
    [typed_offer] = find_commitments(
        "I'll send Priya the deck by Friday and then I'll call Bob tomorrow.",
        TextContext(today=NOW.date(), app="slack.exe", title="#team - Slack"),
        NOW,
    )
    assert typed_offer.fix == "Remind me on Friday at 10am to send Priya the deck"


@pytest.mark.parametrize(
    "said",
    [
        "did you send the deck by friday",
        "i'll think about it",
        "he'll send it by friday",
        "i sent the deck on friday",
        "let me know by friday",
        "if i finish early i'll send it friday",
        "i'll send it",
        "i'll meet you on the second floor",
        "i'll take the second one",
        "i will be out on the twelfth",
        "i'll have to think about it",
    ],
)
def test_spoken_questions_states_and_vague_promises_are_left_alone(said):
    assert offers(said) == []


def test_typed_promises_with_a_day_of_the_month_work_too():
    ctx = TextContext(today=NOW.date(), app="slack.exe", title="#team - Slack")
    [offer] = find_commitments("I'll call the bank on the 12th.", ctx, NOW)
    assert offer.fix == "Remind me on Thursday at 10am to call the bank"
    assert find_commitments("I'll meet you on the 2nd floor.", ctx, NOW) == []
