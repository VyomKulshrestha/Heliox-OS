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


def test_the_processor_model_is_read_from_linux_cpuinfo_on_x86_and_arm():
    report = load()
    x86 = "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 7763 64-Core Processor\nprocessor\t: 1\n"
    assert report.name_from_cpuinfo(x86) == "AMD EPYC 7763 64-Core Processor"
    arm = "processor\t: 0\nBogoMIPS\t: 50.00\nCPU implementer\t: 0x41\nHardware\t: Neoverse-N1\n"
    assert report.name_from_cpuinfo(arm) == "Neoverse-N1"
    assert report.name_from_cpuinfo("processor\t: 0\n") == ""


def test_the_report_names_the_processor_and_its_architecture_on_each_system():
    report = load()
    readers = {"Darwin": lambda: "Apple M1 (Virtual)", "Windows": lambda: "  AMD EPYC   7763  ", "Linux": lambda: ""}
    mac = report.processor_facts("Darwin", "arm64", readers, fallback=lambda: "arm")
    assert mac == {"processor": "Apple M1 (Virtual)", "architecture": "arm64"}
    windows = report.processor_facts("Windows", "AMD64", readers, fallback=lambda: "")
    assert windows == {"processor": "AMD EPYC 7763", "architecture": "AMD64"}
    linux = report.processor_facts("Linux", "x86_64", readers, fallback=lambda: "x86_64")
    assert linux["processor"] == "x86_64"  # nothing in cpuinfo: platform's own answer

    def broken() -> str:
        raise OSError("no registry")

    unknown = report.processor_facts("Windows", "AMD64", {"Windows": broken}, fallback=lambda: "")
    assert unknown == {"processor": "unknown", "architecture": "AMD64"}
    machine = {**mac, "cores": 3, "memory_gb": 7.0, "load_s": 6.1, "rss_mb": 1100}
    assert report.heading(machine).startswith("### Pocket TTS on Apple M1 (Virtual) (arm64), 3 cores, 7.0 GB")
