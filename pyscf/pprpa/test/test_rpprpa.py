#!/usr/bin/env python
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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

import unittest
from pyscf import gto, dft, lib
from pyscf.pprpa.rpprpa_direct import RppRPADirect
from pyscf.pprpa.rpprpa_davidson import RppRPADavidson
from pyscf.pprpa.upprpa_direct import UppRPADirect

def setUpModule():
    global mol, rmf, umf
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [
        ["O", (0.0, 0.0, 0.0)],
        ["H", (0.0, -0.7571, 0.5861)],
        ["H", (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    # create a (N-2)-electron system for charged-neutral H2O
    mol.charge = 2
    mol.build()
    rmf = mol.RKS(xc = "b3lyp").run()
    umf = mol.UKS(xc = "b3lyp").run()

def tearDownModule():
    global mol, rmf, umf
    mol.stdout.close()
    del mol, rmf, umf

class KnownValues(unittest.TestCase):
    def test_rpprpa_correlation_energy(self):
        pp = RppRPADirect(rmf, nocc_act=None, nvir_act=10)
        ec = pp.get_correlation()
        etot, ehf, ec = pp.energy_tot()
        self.assertAlmostEqual(ec, -0.0450238550202-0.0159221841934, 8)

        pp = RppRPADirect(rmf)
        ec = pp.get_correlation()
        etot, ehf, ec = pp.energy_tot()
        self.assertAlmostEqual(ec, -0.11242089840288827)

    def test_upprpa_correlation_energy(self):
        pp = UppRPADirect(umf)
        ec = pp.get_correlation()
        etot, ehf, ec = pp.energy_tot()
        self.assertAlmostEqual(ec, -0.11242089840288827)

    def test_rpprpa_direct(self):
        pp = RppRPADirect(rmf, nocc_act=None, nvir_act=10)
        pp.pp_state = 10
        pp.kernel("s")
        pp.kernel("t")
        self.assertAlmostEqual(pp.ec_s, -0.0450238550202, 8)
        self.assertAlmostEqual(pp.ec_t, -0.0159221841934, 8)

        ref = [0.92727944, 1.18456136, 1.25715794, 1.66117646, 1.72616615]
        self.assertAlmostEqual(abs(pp.exci_s[10:15] - ref).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(pp.exci_s), -21.19709249893, 8)

        ref = [1.16289368, 1.24603019, 1.64840497, 1.69022118, 1.74312208]
        self.assertAlmostEqual(abs(pp.exci_t[6:11] - ref).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(pp.exci_t), -19.45490052780, 8)
        pp.analyze()

    def test_upprpa_direct(self):
        pp = UppRPADirect(umf, nocc_act=None, nvir_act=10)
        pp.pp_state = 10
        pp.kernel()
        self.assertAlmostEqual(pp.ec[0], -0.0159221841934/3, 8)
        self.assertAlmostEqual(pp.ec[1], -0.0159221841934/3, 8)
        self.assertAlmostEqual(pp.ec[2], -0.0450238550202-0.0159221841934/3, 7)
        self.assertAlmostEqual(lib.fp(pp.exci[0]), -19.45490020712, 8)
        self.assertAlmostEqual(lib.fp(pp.exci[1]), -19.45490025537, 8)
        self.assertAlmostEqual(lib.fp(pp.exci[2]), -35.07654958367, 8)
        pp.analyze()

    def test_rpprpa_davidson(self):
        pp = RppRPADavidson(rmf, nocc_act=None, nvir_act=10, nroot=5)
        ref = [0.92727944, 1.18456136, 1.25715794, 1.66117646, 1.72616615]
        pp.kernel("s")
        self.assertAlmostEqual(abs(pp.exci_s - ref).max(), 0, 7)
        ref = [1.16289368, 1.24603019, 1.64840497, 1.69022118, 1.74312208]
        pp.kernel("t")
        self.assertAlmostEqual(abs(pp.exci_t - ref).max(), 0, 7)
        pp.analyze()

    def test_hhrpa(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.atom = [
            ["O", (0.0, 0.0, 0.0)],
            ["H", (0.0, -0.7571, 0.5861)],
            ["H", (0.0, 0.7571, 0.5861)]]
        mol.basis = 'def2-svp'
        mol.charge = -2
        mol.build()
        mf = mol.RKS(xc = "b3lyp").run()
        pp = RppRPADirect(mf, nvir_act=10, nelec="n+2")
        pp.hh_state = 10
        pp.kernel("s")
        pp.kernel("t")
        self.assertAlmostEqual(pp.ec_s, -0.0837534201778, 8)
        self.assertAlmostEqual(pp.ec_t, -0.0633019675263, 8)
        pp.analyze()

if __name__ == "__main__":
    print('Full Tests for ppRPA')
    unittest.main()
