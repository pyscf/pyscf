# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Open-shell natural virtual orbitals preserve the metric and corrected MP2 energy."""
import unittest
import numpy as np
from pyscf import adc, gto, scf


class KnownValues(unittest.TestCase):
    def test_open_shell_fno(self):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', charge=1, spin=1,
                    basis='sto-3g', verbose=0)
        mf = scf.ROHF(mol).run(conv_tol=1e-12)
        canonical = adc.ADC(mf)
        canonical.method = 'adc(2)'
        canonical.kernel_gs()
        overlap = mf.get_ovlp()
        for spin_adapted in (False, True):
            with self.subTest(open_shell_fno=spin_adapted):
                natural = adc.ADC2FNO(mf)
                natural.if_osfno = spin_adapted
                natural.kernel_gs(pct_occ=.9)
                self.assertTrue(any(len(frozen) for frozen in natural.frozen))
                for coeff in natural.mo_coeff:
                    np.testing.assert_allclose(coeff.T@overlap@coeff, np.eye(mol.nao), atol=1e-10)
                reduced = adc.UADC(mf, natural.frozen, natural.mo_coeff, natural.mo_occ,
                                   natural.mo_energy, natural.f_ov)
                reduced.method = 'adc(2)'
                reduced.kernel_gs()
                self.assertAlmostEqual(natural.correct_corr(reduced.e_corr), canonical.e_corr, places=9)


if __name__ == '__main__':
    unittest.main()
