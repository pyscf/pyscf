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

import unittest
from functools import reduce
import numpy
from pyscf import gto, scf, lib, fci
from pyscf.mcscf import newton_casscf, CASSCF, addons

def setUpModule():
    global mol, mf, mc, sa, mol_N2, mf_N2, mc_N2
    mol = gto.Mole()
    mol.verbose = lib.logger.DEBUG
    mol.output = '/dev/null'
    mol.atom = [
        ['H', ( 5.,-1.    , 1.   )],
        ['H', ( 0.,-5.    ,-2.   )],
        ['H', ( 4.,-0.5   ,-3.   )],
        ['H', ( 0.,-4.5   ,-1.   )],
        ['H', ( 3.,-0.5   ,-0.   )],
        ['H', ( 0.,-3.    ,-1.   )],
        ['H', ( 2.,-2.5   , 0.   )],
        ['H', ( 1., 1.    , 3.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    b = 1.4
    mol_N2 = gto.Mole()
    mol_N2.build(
    verbose = lib.logger.DEBUG,
    output = '/dev/null',
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    symmetry = 1
    )
    mf_N2 = scf.RHF (mol_N2).run (conv_tol=1e-10)
    solver1 = fci.FCI(mol_N2)
    solver1.spin = 0
    solver1.nroots = 2
    solver2 = fci.FCI(mol_N2, singlet=False)
    solver2.wfnsym = 'A1u'
    solver2.spin = 2
    mc_N2 = CASSCF(mf_N2, 4, 4)
    mc_N2 = addons.state_average_mix_(mc_N2, [solver1, solver2],
                                         (0.25,0.25,0.5)).newton ()
    mc_N2.conv_tol = 1e-9
    mc_N2.kernel()
    mf = scf.RHF(mol)
    mf.max_cycle = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    mc = newton_casscf.CASSCF(mf, 4, 4)
    mc.fcisolver = fci.direct_spin1.FCI(mol)
    mc.conv_tol = 1e-9
    mc.kernel()
    sa = CASSCF(mf, 4, 4)
    sa.fcisolver = fci.direct_spin1.FCI (mol)
    sa = sa.state_average ([0.5,0.5]).newton ()
    sa.kernel()

def tearDownModule():
    global mol, mf, mc, sa, mol_N2, mf_N2, mc_N2
    mol.stdout.close()
    mol_N2.stdout.close()
    del mol, mf, mc, sa, mol_N2, mf_N2, mc_N2


class KnownValues(unittest.TestCase):
    def test_gen_g_hop(self):
        numpy.random.seed(1)
        mo = numpy.random.random(mf.mo_coeff.shape)
        ci0 = numpy.random.random((6,6))
        ci0/= numpy.linalg.norm(ci0)
        gall, gop, hop, hdiag = newton_casscf.gen_g_hop(mc, mo, ci0, mc.ao2mo(mo))
        self.assertAlmostEqual(lib.fp(gall), 21.288022525148595, 8)
        self.assertAlmostEqual(lib.fp(hdiag), -15.618395788969842, 8)
        x = numpy.random.random(gall.size)
        u, ci1 = newton_casscf.extract_rotation(mc, x, 1, ci0)
        self.assertAlmostEqual(lib.fp(gop(u, ci1)), -412.9441873541524, 8)
        self.assertAlmostEqual(lib.fp(hop(x)), 24.04569925660985, 8)

    def test_get_grad(self):
        # mc.e_tot may be converged to -3.6268060853430573
        self.assertAlmostEqual(mc.e_tot, -3.626383757091541, 7)
        self.assertAlmostEqual(abs(mc.get_grad()).max(), 0, 5)

    def test_sa_gen_g_hop(self):
        numpy.random.seed(1)
        mo = numpy.random.random(mf.mo_coeff.shape)
        ci0 = numpy.random.random((2,36))
        ci0/= numpy.linalg.norm(ci0, axis=1)[:,None]
        ci0 = list (ci0.reshape ((2,6,6)))
        gall, gop, hop, hdiag = newton_casscf.gen_g_hop(sa, mo, ci0, sa.ao2mo(mo))
        self.assertAlmostEqual(lib.fp(gall), 32.46973284682045, 8)
        self.assertAlmostEqual(lib.fp(hdiag), -70.61862254321517, 8)
        x = numpy.random.random(gall.size)
        u, ci1 = newton_casscf.extract_rotation(sa, x, 1, ci0)
        self.assertAlmostEqual(lib.fp(gop(u, ci1)), -49.017079186126, 8)
        self.assertAlmostEqual(lib.fp(hop(x)), 136.2077988624156, 8)

    def test_sa_get_grad(self):
        self.assertAlmostEqual(sa.e_tot, -3.62638372957158, 7)
        # MRH 06/24/2020: convergence thresh of scf may not have consistent
        # meaning in SA problems
        self.assertAlmostEqual(abs(sa.get_grad()).max(), 0, 4)

    def test_sa_mix(self):
        e = mc_N2.e_states
        self.assertAlmostEqual(mc_N2.e_tot, -108.80340952016508, 7)
        self.assertAlmostEqual(mc_N2.e_average, -108.80340952016508, 7)
        self.assertAlmostEqual(numpy.dot(e,[.25,.25,.5]), -108.80340952016508, 7)
        dm1 = mc_N2.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 0.52172669549357464, 4)
        self.assertAlmostEqual(lib.fp(dm1[1]), 0.53366776017869022, 4)
        self.assertAlmostEqual(lib.fp(dm1[0]+dm1[1]), 1.0553944556722636, 4)

        mc_N2.cas_natorb()



if __name__ == "__main__":
    print("Full Tests for mcscf.addons")
    unittest.main()
