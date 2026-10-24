# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Fourier interpolation recovers a nearest-neighbor two-orbital Hamiltonian."""
import unittest
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lo import interpolation


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.M(atom='He 0 0 0', a=np.eye(3)*3,
                         basis=[[0, [.8, 1]], [0, [.3, 1]]], verbose=0)

    def model(self, kpts):
        phase = kpts @ self.cell.lattice_vectors()[0]
        h = np.zeros((len(kpts), 2, 2), dtype=complex)
        h[:, 0, 0] = 1 + .3*np.cos(phase)
        h[:, 1, 1] = 2 - .4*np.cos(phase)
        h[:, 0, 1] = .2j*np.sin(phase)
        h[:, 1, 0] = h[:, 0, 1].conj()
        return h

    def test_off_grid_interpolation(self):
        band = self.cell.get_abs_kpts(np.array([[.13, 0, 0], [-.22, 0, 0]]))
        nk = 4
        kpts = self.cell.make_kpts([nk, 1, 1])
        actual = interpolation.fourier_interpolate(self.cell, kpts, band, self.model(kpts))
        np.testing.assert_allclose(actual, self.model(band), atol=1e-12)
        rs, weights = interpolation.get_WigerSeitz_Rs(self.cell.lattice_vectors(), [nk, 1, 1])
        self.assertAlmostEqual(np.sum(weights**2), nk, places=12)

    def test_wannier_interpolation_at_input_kpoints(self):
        kpts = self.cell.make_kpts([3, 1, 1])
        mf = scf.KRHF(self.cell, kpts, exxdiv=None)
        mf.mo_energy, mf.mo_coeff = mf.eig(mf.get_hcore(), mf.get_ovlp())
        energy, coeff = interpolation.wannier_interpolation(mf, kpts, mf.mo_coeff)
        np.testing.assert_allclose(energy, mf.mo_energy, atol=1e-10)
        for c in coeff:
            np.testing.assert_allclose(c.conj().T @ c, np.eye(2), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
