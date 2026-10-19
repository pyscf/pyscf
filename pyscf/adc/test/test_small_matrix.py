# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Check ADC roots against explicit small secular matrices and analysis output."""
import io
import unittest
import numpy as np
from pyscf import adc, gto, lib, scf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
        cls.mf = scf.UHF(mol).run(conv_tol=1e-12)

    def check_matrix(self, method, excitation, restricted=False, core=True):
        obj = adc.ADC(self.mf.to_rhf() if restricted else self.mf)
        obj.conv_tol = 1e-10
        obj.tol_residual = 1e-7
        obj.method = method
        obj.method_type = excitation
        if excitation == 'ip' and core:
            obj.ncvs = 1
        energy, vectors, strength, transitions = obj.kernel(nroots=2)
        excited = obj._adc_es
        eris = excited.transform_integrals()
        intermediates = excited.get_imds(eris)
        diagonal = excited.get_diag(intermediates, eris)
        matvec = excited.matvec(intermediates, eris)
        if method == 'adc(2)' and not restricted:
            matrix = np.column_stack([matvec(v) for v in np.eye(len(diagonal))])
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
            np.testing.assert_allclose(energy, np.linalg.eigvalsh(matrix)[:2], atol=1e-7)
        residual = np.column_stack([matvec(v) for v in vectors.T]) - vectors * energy
        # Davidson discards correction vectors with squared norm below 1e-14.
        np.testing.assert_allclose(residual, 0., atol=obj.tol_residual)
        if not restricted:
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(2), atol=1e-8)
        with io.StringIO() as out:
            with lib.temporary_env(excited, stdout=out, verbose=4,
                                   evec_print_tol=1e-4, spec_factor_print_tol=1e-4):
                excited.analyze()
            self.assertIn('Eigenvector analysis', out.getvalue())
        # Reporting must not reorder roots or mutate their amplitudes.
        np.testing.assert_array_equal(excited.E, energy)
        np.testing.assert_array_equal(excited.U, vectors)

    def test_ee_adc2(self):
        self.check_matrix('adc(2)', 'ee')

    def test_ee_adc3(self):
        self.check_matrix('adc(3)', 'ee')

    def test_core_ip_adc2(self):
        self.check_matrix('adc(2)', 'ip')

    def test_core_ip_adc3(self):
        self.check_matrix('adc(3)', 'ip')

    def test_ea_adc2(self):
        self.check_matrix('adc(2)', 'ea')

    def test_ea_adc3(self):
        self.check_matrix('adc(3)', 'ea')

    def test_valence_ip_adc2(self):
        self.check_matrix('adc(2)', 'ip', core=False)

    def test_valence_ip_adc3(self):
        self.check_matrix('adc(3)', 'ip', core=False)

    def test_restricted_ee_adc3(self):
        self.check_matrix('adc(3)', 'ee', restricted=True)

    def test_restricted_core_ip_adc3(self):
        self.check_matrix('adc(3)', 'ip', restricted=True)


if __name__ == '__main__':
    unittest.main()
