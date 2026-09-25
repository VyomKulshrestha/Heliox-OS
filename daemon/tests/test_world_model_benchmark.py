from benchmarks.world_model_suite import benchmark


def test_shipped_world_model_benchmark_beats_baselines_and_preserves_directions() -> None:
    report = benchmark(iterations=2)

    assert report["training_samples"] == 25_200
    assert report["validation_samples"] == 5_400
    assert report["learned_action_types"] == 12
    assert report["validation"]["disk_improvement_percent"] > 0
    assert report["validation"]["process_improvement_percent"] > 0
    assert report["direction_checks_passed"] == len(report["direction_checks"])
    assert report["inference"]["iterations"] == 24
