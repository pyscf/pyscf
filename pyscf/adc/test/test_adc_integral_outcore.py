# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""UHF and ROHF ADC spectra agree across integral representations."""
import unittest
import numpy as np
from pyscf import adc, gto, scf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
        cls.uhf = scf.UHF(mol).run(conv_tol=1e-12)
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6',
                    basis={'Li': ('sto3g', [[0, [1, 1]]]), 'H': 'sto3g'},
                    charge=1, spin=1, verbose=0)
        cls.rohf = scf.ROHF(mol).run(conv_tol=1e-12)
        assert cls.uhf.converged and cls.rohf.converged

    def make_adc(self, mf, method='adc(3)', excitation='ee'):
        obj = adc.ADC(mf)
        obj.method = method
        obj.method_type = excitation
        obj.conv_tol = 1e-10
        obj.tol_residual = 1e-7
        return obj

    def check_spectra(self, ref, out, compare_projector=False):
        er, vr, pr, _ = ref.kernel(nroots=2)
        eo, vo, po, _ = out.kernel(nroots=2)
        np.testing.assert_allclose(eo, er, atol=1e-8)
        np.testing.assert_allclose(po, pr, atol=1e-7)
        self.assertAlmostEqual(out.e_corr, ref.e_corr, places=10)
        if compare_projector:
            # Degenerate spin partners may rotate; compare their joint projector.
            np.testing.assert_allclose(vo @ vo.T, vr @ vr.T, atol=1e-6)

    def check_storage(self, mf, method='adc(3)', excitation='ee',
                      compare_projector=False):
        ref = self.make_adc(mf, method, excitation)
        out = self.make_adc(mf, method, excitation)
        out.incore_complete = False
        out.max_memory = 0
        self.check_spectra(ref, out, compare_projector)

    def test_uhf_adc2x(self):
        self.check_storage(self.uhf, 'adc(2)-x', compare_projector=True)

    def test_uhf_adc3(self):
        self.check_storage(self.uhf, compare_projector=True)

    def test_rohf_excitation(self):
        self.check_storage(self.rohf)

    def test_rohf_ionization(self):
        self.check_storage(self.rohf, excitation='ip')

    def test_rohf_attachment(self):
        self.check_storage(self.rohf, excitation='ea')

    def test_rohf_lossless_natural_auxiliary_basis(self):
        ref = self.make_adc(self.rohf).density_fit('weigend')
        out = self.make_adc(self.rohf).density_fit('weigend')
        out.if_naf = True
        # Remove only the null space of the auxiliary pair-density metric.
        out.thresh_naf = 1e-12
        self.check_spectra(ref, out)


if __name__ == '__main__':
    unittest.main()
