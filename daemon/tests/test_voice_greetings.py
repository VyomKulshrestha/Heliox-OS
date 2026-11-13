"""Greetings of other languages before the name wake Heliox; the name must still follow. The sentences are what
Whisper small wrote for synthetic speech on 13 November (espeak-ng; Windows' Helena voice in Spanish), not a
recording of a person."""

from __future__ import annotations

import pytest

from pilot.system.addressing import vocative_wake
from pilot.system.voice import loose_wake_prefix


@pytest.mark.parametrize(
    ("heard", "command"),
    [
        ("Oye Heliox, Recúrdame Maana a las 9 la reuníene.", "Recúrdame Maana a las 9 la reuníene."),
        ("Hallo Heliox. Erinnere mich morgen um 9 Uhr an das Mädchen.", "Erinnere mich morgen um 9 Uhr an das"),
        ("Hola Heliox, ¿qué hora es?", "¿qué hora es?"),
        ("Ciao Heliox, che ore sono?", "che ore sono?"),
        ("Salut Heliox, quelle heure est-il ?", "quelle heure est-il ?"),
        ("Oi Heliox, que horas são?", "que horas são?"),
        ("Namaste Heliox, kal ka mausam batao.", "kal ka mausam batao."),
        ("Hey, Heliox可是,明天早上有點提醒我開會。", ""),  # another script before or after: the name is found
    ],
)
def test_a_greeting_of_another_language_before_the_name_wakes(heard, command):
    wake = vocative_wake(heard)
    assert wake is not None and wake.exact
    assert wake.command.startswith(command[:12])
    assert loose_wake_prefix(heard)


@pytest.mark.parametrize(
    "heard",
    [
        "Hola, ¿cómo estás?",  # a greeting alone
        "Hallo Alex, wie geht's?",  # not the name
        "Ciao Helen, a domani.",
        "Oye, the helicopter is loud.",
    ],
)
def test_a_greeting_without_the_name_wakes_nothing(heard):
    assert vocative_wake(heard) is None
