# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""SGX forces agree with energy derivatives on a small molecule."""
import unittest
import numpy as np
from pyscf import gto, scf, dft
from pyscf.sgx.sgx import sgx_fit


class KnownValues(unittest.TestCase):
    def check_gradient(self, method, fit_ovlp=False, optk=False, symm_fit=False, dfj=True, debug=False):
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr',
                    basis='sto-3g', verbose=0)
        mf = sgx_fit(method(mol))
        if isinstance(mf, dft.rks.KohnShamDFT):
            mf.xc = 'pbe0'
            mf.grids.level = 2
        mf.with_df.grids_level_f = 2
        mf.with_df.dfj = dfj
        mf.with_df.debug = debug
        mf.with_df.fit_ovlp = fit_ovlp
        mf.with_df.optk = optk
        mf.with_df._symm_ovlp_fit = symm_fit
        mf.with_df.use_opt_grids = True
        mf.conv_tol = 1e-12
        mf.kernel()
        grad = mf.nuc_grad_method().set(sgx_grid_response=True, grid_response=True)
        force = grad.kernel()
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-10)
        scanner = mf.as_scanner()
        coords = mol.atom_coords()
        plus, minus = coords.copy(), coords.copy()
        plus[1, 2] += 1e-4
        minus[1, 2] -= 1e-4
        ep = scanner(mol.set_geom_(plus, unit='Bohr', inplace=False))
        em = scanner(mol.set_geom_(minus, unit='Bohr', inplace=False))
        self.assertAlmostEqual(force[1, 2], (ep-em)/2e-4, places=5)

    def test_rhf(self):
        self.check_gradient(scf.RHF)

    def test_uhf_overlap_fit(self):
        self.check_gradient(scf.UHF, fit_ovlp=True)

    def test_rks_optimized_exchange(self):
        self.check_gradient(dft.RKS, optk=True)

    def test_uks_symmetric_fit(self):
        self.check_gradient(dft.UKS, fit_ovlp=True, optk=True, symm_fit=True)

    def test_rhf_seminumerical_coulomb(self):
        self.check_gradient(scf.RHF, dfj=False)

    def test_uhf_reference_integrals(self):
        self.check_gradient(scf.UHF, dfj=False, debug=True)


if __name__ == '__main__':
    unittest.main()
