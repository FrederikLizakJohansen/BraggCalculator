import pytest
from braggcalculator import BraggCalculator


@pytest.fixture
def calc():
    return BraggCalculator()
