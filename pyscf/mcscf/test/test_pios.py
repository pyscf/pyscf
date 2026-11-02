# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Pi-space selection must preserve the occupied space and AO metric."""
import contextlib
import io
import unittest
import numpy as np
from pyscf import gto, scf
from pyscf.mcscf import PiOS


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        theta = np.arange(6) * np.pi / 3
        ring = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(6)))
        cls.mol = gto.M(atom=[('C', p) for p in ring * 1.4] +
                            [('H', p) for p in ring * 2.48],
                        basis='sto-3g', verbose=0)
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-11)

    def check_space(self, npiocc, npivirt):
        options = {} if npiocc is None else dict(nPiOcc=npiocc, nPiVirt=npivirt)
        # PiOS prints its diagnostic tables and changes NumPy print settings.
        with contextlib.redirect_stdout(io.StringIO()), np.printoptions():
            core, active, external, nelec, coeff = PiOS.MakePiOS(
                self.mol, self.mf, [1, 2, 3, 4, 5, 6], **options)
        npiocc = 3 if npiocc is None else npiocc
        npivirt = 3 if npivirt is None else npivirt
        self.assertEqual((active, nelec), (npiocc + npivirt, 2 * npiocc))
        self.assertEqual(core + npiocc, self.mol.nelectron // 2)
        self.assertEqual(core + active + external, self.mol.nao_nr())
        overlap = self.mf.get_ovlp()
        np.testing.assert_allclose(coeff.T @ overlap @ coeff,
                                   np.eye(coeff.shape[1]), atol=2e-10)
        occupied = coeff[:, :core + npiocc]
        np.testing.assert_allclose(2 * occupied @ occupied.T,
                                   self.mf.make_rdm1(), atol=2e-9)
        # For a planar molecule, the complete pi space has only p_z AO components.
        if active == 6:
            pi_coeff = coeff[:, core:core + active]
            in_plane = [i for i, label in enumerate(self.mol.ao_labels()) if 'pz' not in label]
            np.testing.assert_allclose(pi_coeff[in_plane], 0., atol=2e-8)

    def test_benzene_complete_pi_space(self):
        self.check_space(None, None)

    def test_benzene_frontier_pi_space(self):
        self.check_space(1, 1)


if __name__ == '__main__':
    unittest.main()
