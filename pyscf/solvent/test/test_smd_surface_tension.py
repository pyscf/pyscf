# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""The Python SMD surface-tension gradient differentiates its energy model."""
import unittest
import numpy as np
from pyscf.solvent import smd_experiment
from pyscf.solvent.grad import smd_experiment as smd_grad


class KnownValues(unittest.TestCase):
    def check_gradient(self, water):
        # A glycine-like geometry exercises H-C/O, C-C/N, N-C, and O-C/N terms.
        symbols = ['N', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H']
        coords = np.array([[0, 0, 0], [1.45, 0, 0], [2.0, 1.3, 0],
                           [3.2, 1.5, 0], [1.2, 2.2, 0], [-.4, .8, .4],
                           [-.4, -.8, .4], [1.8, -.5, .8], [1.8, -.5, -.8],
                           [.4, 1.9, 0]])
        options = dict(n=1.4, alpha=.3, beta=.5, water=water)
        grad = smd_grad.atomic_surface_tension(symbols, coords, **options)
        np.testing.assert_allclose(grad.sum(axis=1), 0, atol=1e-12)
        eps = 1e-5
        for atom in range(len(symbols)):
            for axis in range(3):
                with self.subTest(atom=atom, axis=axis):
                    plus, minus = coords.copy(), coords.copy()
                    plus[atom, axis] += eps
                    minus[atom, axis] -= eps
                    fp = smd_experiment.atomic_surface_tension(symbols, plus, **options)
                    fm = smd_experiment.atomic_surface_tension(symbols, minus, **options)
                    np.testing.assert_allclose(grad[:, atom, axis], (fp-fm)/(2*eps), atol=2e-6)
        order = np.arange(len(symbols))[::-1]
        actual = smd_experiment.atomic_surface_tension([symbols[i] for i in order], coords[order]+2.1, **options)
        expected = smd_experiment.atomic_surface_tension(symbols, coords, **options)[order]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_water_gradient(self):
        self.check_gradient(True)

    def test_nonwater_gradient(self):
        self.check_gradient(False)

    def test_isolated_element_tensions(self):
        symbols = ['F', 'Si', 'S', 'Cl', 'Br', 'P']
        coords = np.arange(18).reshape(6, 3)*10.
        value = smd_experiment.atomic_surface_tension(symbols, coords, 1.4, .3, .5)
        np.testing.assert_allclose(value, [38.18, 0, -9.10, 9.82, -8.72, 0])
        grad = smd_grad.atomic_surface_tension(symbols, coords, 1.4, .3, .5)
        np.testing.assert_array_equal(grad, np.zeros((6, 6, 3)))


if __name__ == '__main__':
    unittest.main()
