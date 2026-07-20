import pytest

from scripts.plot_cod_throughput import DEFAULT_INPUTS, corpus_metrics, load_runs


def test_cod_throughput_publication_records_are_matched():
    runs = load_runs(list(DEFAULT_INPUTS))
    metrics = corpus_metrics(runs)

    assert metrics["speedups"][("numpy", "cpu")]["all"] == pytest.approx(4.591094)
    assert metrics["speedups"][("torch", "cpu")]["all"] == pytest.approx(4.972382)
    assert metrics["speedups"][("torch", "cuda")]["all"] == pytest.approx(13.167573)
    assert metrics["torch_cuda_acceleration"] == pytest.approx(2.587790)

    cpu_results = {
        (result["id"], result["mode"]): result
        for result in runs[("torch", "cpu")]["results"]
    }
    cuda_results = {
        (result["id"], result["mode"]): result
        for result in runs[("torch", "cuda")]["results"]
    }
    cuda_faster = sum(
        cpu_results[key]["braggcalculator_seconds"]
        > cuda_results[key]["braggcalculator_seconds"]
        for key in cpu_results
    )
    assert cuda_faster == 74
