"""Tests for the on-device wake-word calibration module (voice_calibration.py).

Covers the pure edit-distance function, the near-miss/promotion logic, and
JSON round-tripping via tmp_path — no real audio/mic involved anywhere.
"""

from pilot.system.voice_calibration import (
    VoiceCalibrationStore,
    WakeWordCalibrator,
    WakeWordVariant,
    levenshtein,
)


class TestLevenshtein:
    def test_identical_strings(self):
        assert levenshtein("hey heliox", "hey heliox") == 0

    def test_empty_strings(self):
        assert levenshtein("", "") == 0
        assert levenshtein("abc", "") == 3
        assert levenshtein("", "abc") == 3

    def test_single_substitution(self):
        assert levenshtein("hey heliox", "hey heliocks") != 0
        assert levenshtein("cat", "bat") == 1

    def test_known_distance(self):
        # Classic textbook example
        assert levenshtein("kitten", "sitting") == 3

    def test_insertion_and_deletion(self):
        assert levenshtein("hey heliox", "heliox") == 4  # "hey " deleted (4 chars)


class TestVoiceCalibrationStore:
    def test_load_returns_empty_dict_when_no_file_exists(self, tmp_path):
        store = VoiceCalibrationStore(data_dir=tmp_path / "voice_calibration")
        assert store.load() == {}

    def test_save_and_load_round_trip(self, tmp_path):
        store = VoiceCalibrationStore(data_dir=tmp_path / "voice_calibration")
        variants = {
            "hey heliocks": WakeWordVariant(text="hey heliocks", confirmed_count=3),
        }
        store.save(variants)

        reloaded = store.load()
        assert "hey heliocks" in reloaded
        assert reloaded["hey heliocks"].confirmed_count == 3

    def test_reset_deletes_the_file(self, tmp_path):
        store = VoiceCalibrationStore(data_dir=tmp_path / "voice_calibration")
        store.save({"x": WakeWordVariant(text="x")})
        assert store.load() != {}

        store.reset()
        assert store.load() == {}

    def test_load_tolerates_corrupted_file(self, tmp_path):
        data_dir = tmp_path / "voice_calibration"
        data_dir.mkdir(parents=True)
        (data_dir / "wake_variants.json").write_text("{not valid json", encoding="utf-8")

        store = VoiceCalibrationStore(data_dir=data_dir)
        assert store.load() == {}


class TestWakeWordCalibrator:
    def _calibrator(self, tmp_path) -> WakeWordCalibrator:
        store = VoiceCalibrationStore(data_dir=tmp_path / "voice_calibration")
        return WakeWordCalibrator(["hey heliox"], store=store)

    def test_check_near_miss_finds_close_leading_prefix(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        # "hey iliox" is 2 edits from "hey heliox" (dropped "he" sound) -
        # a plausible accent mistranscription, within NEAR_MISS_MAX_DISTANCE.
        # Trailing command words must NOT bleed into the comparison window.
        result = calibrator.check_near_miss("hey iliox turn on the lights")
        assert result == "hey iliox"

    def test_check_near_miss_ignores_exact_matches(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        # An exact match belongs to voice.py's fast path, not this fallback.
        assert calibrator.check_near_miss("hey heliox turn on the lights") is None

    def test_check_near_miss_rejects_unrelated_transcript(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        assert calibrator.check_near_miss("what is the weather today") is None

    def test_promotion_requires_multiple_confirmations(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        near_miss = "hey heliocks"

        for _ in range(calibrator.PROMOTION_THRESHOLD - 1):
            calibrator.record_pending(near_miss)
            calibrator.confirm_pending_if_followed_by_hit()

        # Not yet promoted - one short of the threshold.
        assert calibrator.match_promoted_variant(f"{near_miss} turn on the lights") is None

        calibrator.record_pending(near_miss)
        calibrator.confirm_pending_if_followed_by_hit()

        assert calibrator.match_promoted_variant(f"{near_miss} turn on the lights") == near_miss

    def test_pending_expires_after_confirm_window_without_a_hit(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        calibrator.record_pending("hey heliocks")

        for _ in range(calibrator.CONFIRM_WINDOW_ITERATIONS):
            calibrator.tick()

        # Expired - a hit now shouldn't confirm the stale candidate.
        calibrator.confirm_pending_if_followed_by_hit()
        variants = calibrator.get_variants()
        assert all(v.text != "hey heliocks" or v.confirmed_count == 0 for v in variants)

    def test_confirm_with_no_pending_candidates_is_a_no_op(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        calibrator.confirm_pending_if_followed_by_hit()
        assert calibrator.get_variants() == []

    def test_unpromoted_variant_does_not_match(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        calibrator.record_pending("hey heliocks")
        calibrator.confirm_pending_if_followed_by_hit()  # only 1 confirmation
        assert calibrator.match_promoted_variant("hey heliocks turn on the lights") is None

    def test_reset_clears_variants_pending_and_the_store(self, tmp_path):
        calibrator = self._calibrator(tmp_path)
        for _ in range(calibrator.PROMOTION_THRESHOLD):
            calibrator.record_pending("hey heliocks")
            calibrator.confirm_pending_if_followed_by_hit()
        assert calibrator.get_variants() != []

        calibrator.record_pending("still pending")
        calibrator.reset()

        assert calibrator.get_variants() == []
        assert calibrator.match_promoted_variant("hey heliocks turn on the lights") is None
        # A hit right after reset shouldn't confirm the now-cleared pending candidate.
        calibrator.confirm_pending_if_followed_by_hit()
        assert calibrator.get_variants() == []

        # The on-device store itself was cleared too, not just in-memory state.
        fresh_calibrator = self._calibrator(tmp_path)
        assert fresh_calibrator.get_variants() == []
