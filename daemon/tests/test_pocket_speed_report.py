"""The Pocket speed report's arithmetic (tests/platform_real/pocket_speed_report.py); the run itself is CI's."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent / "platform_real" / "pocket_speed_report.py"


def load():
    spec = importlib.util.spec_from_file_location("pocket_speed_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_streams_without_gaps_only_when_every_piece_is_made_faster_than_it_plays():
    report = load()
    fast = [{"first_s": 0.1, "made_s": 0.3, "audio_s": 1.0}, {"first_s": 0.2, "made_s": 0.9, "audio_s": 2.0}]
    assert report.summarise(fast) == {
        "pieces": 2,
        "first_median_s": 0.15,
        "first_worst_s": 0.2,
        "rtf_median": 0.375,
        "rtf_worst": 0.45,
        "streams_without_gaps": True,
    }
    slow = [*fast, {"first_s": 0.3, "made_s": 2.2, "audio_s": 2.0}]
    assert report.summarise(slow)["streams_without_gaps"] is False


def test_an_empty_run_says_nothing_streams():
    assert load().summarise([])["streams_without_gaps"] is False
