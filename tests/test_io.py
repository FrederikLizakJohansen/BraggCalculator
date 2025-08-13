import pytest
from braggcalculator.io import to_pmg_structure


def test_io_rejects_unknown():
    with pytest.raises(TypeError):
        to_pmg_structure(object())
