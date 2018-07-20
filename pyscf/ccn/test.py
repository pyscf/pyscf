#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import cc

from pyscf import gto, scf, lib
from pyscf.cc import gccsd, eom_gccsd

import numpy
from numpy import testing
import unittest


class H2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.verbose = 0
        cls.mol.atom = "H 0 0 0; H 0.74 0 0"
        cls.mol.basis = 'ccpvdz'
        cls.mol.build()

        cls.mf = scf.GHF(cls.mol)
        cls.mf.kernel()

        cls.ccsd = gccsd.GCCSD(cls.mf)
        cls.ccsd.kernel()
        cls.ccsd.conv_tol_normt = 1e-8
        cls.ccsd.solve_lambda()

        cls.nroots = 2

        cls.eomip = eom_gccsd.EOMIP(cls.ccsd)
        cls.eomip.conv_tol = 1e-12
        cls.eomip.kernel(nroots=cls.nroots)
        cls.eomea = eom_gccsd.EOMEA(cls.ccsd)
        cls.eomea.conv_tol = 1e-12
        cls.eomea.kernel(nroots=cls.nroots)

    def test_iter_s(self):
        """CCS iterations."""
        e1, t1 = cc.kernel_ground_state_s(self.ccsd)

        testing.assert_allclose(e1, 0, atol=1e-8)
        # TODO: atol=1e-8 does not work
        testing.assert_allclose(t1, 0, atol=1e-6)

    def test_iter_sd(self):
        """CCSD iterations."""
        e2, t1, t2 = cc.kernel_ground_state_sd(self.ccsd)

        testing.assert_allclose(e2, self.ccsd.e_corr, atol=1e-8)
        testing.assert_allclose(t1, self.ccsd.t1, atol=1e-8)
        testing.assert_allclose(t2, self.ccsd.t2, atol=1e-7)

    def test_iter_sdt(self):
        """CCSDT iterations (there are no triple excitations for a 2-electron system)."""
        e3, t1, t2, t3 = cc.kernel_ground_state_sdt(self.ccsd)

        testing.assert_allclose(e3, self.ccsd.e_corr, atol=1e-8)
        testing.assert_allclose(t1, self.ccsd.t1, atol=1e-8)
        testing.assert_allclose(t2, self.ccsd.t2, atol=1e-7)
        testing.assert_allclose(t3, 0, atol=1e-8)

    def test_lambda_sd(self):
        """CCSD lambda iterations."""
        l1, l2 = cc.kernel_lambda_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2)

        testing.assert_allclose(l1, self.ccsd.l1, atol=1e-8)
        testing.assert_allclose(l2, self.ccsd.l2, atol=1e-8)

    def test_ip_sd(self):
        """CCSD EOM iterations (IP)."""
        values, vectors = cc.kernel_ip_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2, nroots=self.nroots)

        testing.assert_allclose(values, self.eomip.e, atol=1e-12)

    def test_ea_sd(self):
        """CCSD EOM iterations (EA)."""
        values, vectors = cc.kernel_ea_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2, nroots=self.nroots)

        testing.assert_allclose(values, self.eomea.e, atol=1e-12)


class OTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Oxygen atom test vs ORCA-MRCC data.

        ORCA version 3.0.3

        Input example:

        ! cc-pvdz UHF TightSCF

        %mrcc
          method "CCSDT"
          ETol 10
        end

        %pal nprocs 4
        end

        * xyzfile 0 1 initial.xyz

        ORCA reference energies:

        HF    -74.6652538779
        CCS   -74.841686696943
        CCSD  -74.819248718982
        CCSDT -74.829163218204
        """
        cls.mol = gto.Mole()
        cls.mol.verbose = 0
        cls.mol.atom = "O 0 0 0"
        cls.mol.basis = 'cc-pvdz'
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-11
        cls.mf.kernel()
        testing.assert_allclose(cls.mf.e_tot, -74.6652538779, atol=1e-4)
        cls.mf = scf.addons.convert_to_ghf(cls.mf)

        cls.ccsd = gccsd.GCCSD(cls.mf, frozen=2)
        cls.ccsd.kernel()
        cls.ccsd.conv_tol_normt = 1e-8
        cls.ccsd.solve_lambda()

        cls.nroots = 2

        cls.eomip = eom_gccsd.EOMIP(cls.ccsd)
        cls.eomip.conv_tol = 1e-12
        cls.eomip.kernel(nroots=cls.nroots, koopmans=True)
        cls.eomea = eom_gccsd.EOMEA(cls.ccsd)
        cls.eomea.conv_tol = 1e-12
        cls.eomea.kernel(nroots=cls.nroots, koopmans=True)

    def test_iter_s(self):
        """CCS iterations."""
        e1, t1 = cc.kernel_ground_state_s(self.ccsd)
        # # TODO: MRCC energy is way off expected
        # testing.assert_allclose(self.mf.e_tot + e1, -74.841686696943, atol=1e-4)

        import pyscf.cc
        mf = scf.RHF(self.mol).set(conv_tol = 1e-11).run()
        cc1 = pyscf.cc.UCCSD(mf, frozen=1)
        old_update_amps = cc1.update_amps
        def update_amps(t1, t2, eris):
            t1, t2 = old_update_amps(t1, t2, eris)
            return t1, (t2[0]*0, t2[1]*0, t2[2]*0)
        cc1.update_amps = update_amps
        cc1.kernel()
        testing.assert_allclose(self.mf.e_tot + e1, cc1.e_tot, atol=1e-4)

    def test_iter_sd(self):
        """CCSD iterations."""
        e2, t1, t2 = cc.kernel_ground_state_sd(self.ccsd)
        testing.assert_allclose(self.mf.e_tot + e2, -74.819248718982, atol=1e-4)

    def _test_iter_sdt(self):
        """CCSDT iterations."""
        e3, t1, t2, t3 = cc.kernel_ground_state_sdt(self.ccsd)
        testing.assert_allclose(self.mf.e_tot + e3, -74.829163218204, atol=1e-4)

    def test_lambda_sd(self):
        """CCSD lambda iterations."""
        l1, l2 = cc.kernel_lambda_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2)

        testing.assert_allclose(l1, self.ccsd.l1, atol=1e-8)
        testing.assert_allclose(l2, self.ccsd.l2, atol=1e-7)

    def test_ip_sd(self):
        """CCSD EOM iterations (IP)."""
        values, vectors = cc.kernel_ip_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2, nroots=self.nroots)

        testing.assert_allclose(values, self.eomip.e, atol=1e-12)

    def test_ea_sd(self):
        """CCSD EOM iterations (EA)."""
        values, vectors = cc.kernel_ea_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2, nroots=self.nroots)

        testing.assert_allclose(values, self.eomea.e, atol=1e-12)


class H2OTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        H20 molecule test vs ORCA-MRCC data.

        ORCA reference energies:

        HF    -75.97354725
        CCS   --
        CCSD  -76.185805898396
        CCSDT -76.189327633478
        """
        cls.mol = gto.Mole()
        cls.mol.verbose = 0
        cls.mol.atom = "O 0 0 0; H  0.758602  0.000000  0.504284; H  0.758602  0.000000  -0.504284"
        cls.mol.unit = "angstrom"

        cls.mol.basis = 'cc-pvdz'
        cls.mol.build()

        cls.mf = scf.GHF(cls.mol)
        cls.mf.conv_tol = 1e-11
        cls.mf.kernel()
        testing.assert_allclose(cls.mf.e_tot, -75.97354725, atol=1e-4)

        cls.ccsd = gccsd.GCCSD(cls.mf, frozen=2)
        cls.ccsd.kernel()
        cls.ccsd.conv_tol_normt = 1e-8
        cls.ccsd.solve_lambda()

        cls.nroots = 4

        cls.eomip = eom_gccsd.EOMIP(cls.ccsd)
        cls.eomip.conv_tol = 1e-12
        cls.eomip.kernel(nroots=cls.nroots)
        cls.eomea = eom_gccsd.EOMEA(cls.ccsd)
        cls.eomea.conv_tol = 1e-12
        cls.eomea.kernel(nroots=cls.nroots)

    def test_iter_sd(self):
        """CCSD iterations."""
        e2, t1, t2 = cc.kernel_ground_state_sd(self.ccsd)

        testing.assert_allclose(self.mf.e_tot + e2, -76.185805898396, atol=1e-4)

    def test_iter_d(self):
        """CCD iterations."""
        e2, t2 = cc.kernel_ground_state_d(self.ccsd)
        # # TODO: MRCC energy is way off expected
        # testing.assert_allclose(self.mf.e_tot + e2, -76.177931897355, atol=1e-4)

        import pyscf.cc
        cc1 = scf.RHF(self.mol).run(conv_tol = 1e-11).apply(pyscf.cc.CCSD)
        cc1.frozen = 1
        old_update_amps = cc1.update_amps
        def update_amps(t1, t2, eris):
            t1, t2 = old_update_amps(t1, t2, eris)
            return t1*0, t2
        cc1.update_amps = update_amps
        cc1.kernel()
        testing.assert_allclose(self.mf.e_tot + e2, cc1.e_tot, atol=1e-4)

    def _test_iter_sdt(self):
        """CCSDT iterations."""
        e3, t1, t2, t3 = cc.kernel_ground_state_sdt(self.ccsd)

        testing.assert_allclose(self.mf.e_tot + e3, -76.189327633478, atol=1e-4)

    def test_lambda_sd(self):
        """CCSD lambda iterations."""
        l1, l2 = cc.kernel_lambda_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2)

        testing.assert_allclose(l1, self.ccsd.l1, atol=1e-8)
        testing.assert_allclose(l2, self.ccsd.l2, atol=1e-8)

    def test_ip_sd(self):
        """CCSD EOM iterations (IP)."""
        values, vectors = cc.kernel_ip_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2, nroots=self.nroots)

        testing.assert_allclose(values, self.eomip.e, atol=1e-12)

    def test_ea_sd(self):
        """CCSD EOM iterations (EA)."""
        values, vectors = cc.kernel_ea_sd(self.ccsd, self.ccsd.t1, self.ccsd.t2, nroots=self.nroots)

        testing.assert_allclose(values, self.eomea.e, atol=1e-12)

class utilTests(unittest.TestCase):
    def test_p(self):
        numpy.random.seed(2)
        a = numpy.random.random((3,3,3))
        self.assertAlmostEqual(lib.finger(cc.p('a.c', a)), -1.1768882755852079, 12)
        self.assertAlmostEqual(lib.finger(cc.p('.ab', a)), -1.9344875839983993, 12)
        self.assertAlmostEqual(lib.finger(cc.p('abc', a)), -0.0055534783760265282, 14)
        self.assertAlmostEqual(abs(cc.p('a.a', a) - a).max(), 0, 12)
