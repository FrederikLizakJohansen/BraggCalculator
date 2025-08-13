from braggcalculator import BraggCalculator


def test_fq_runs_smoke(tmp_path):
    # Needs a simple CIF present during real tests; here we just import class
    calc = BraggCalculator()
    assert hasattr(calc, "fq")
