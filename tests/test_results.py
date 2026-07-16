import numpy as np

from braggcalculator import BraggCalculator, ReflectionTable


def test_reflection_table_exposes_consistent_indexed_quantities(nacl):
    calculator = BraggCalculator().load(nacl)
    table = calculator.reflection_table()
    assert isinstance(table, ReflectionTable)
    assert table.hkl.shape == (len(table), 3)
    np.testing.assert_allclose(table.q, 2 * np.pi / table.d_spacing)
    np.testing.assert_allclose(
        table.q,
        4 * np.pi * np.sin(np.radians(table.two_theta) / 2) / calculator.wavelength,
    )
    positions, intensity = calculator.iq()
    np.testing.assert_allclose(table.two_theta, positions)
    np.testing.assert_allclose(table.intensity, intensity)
