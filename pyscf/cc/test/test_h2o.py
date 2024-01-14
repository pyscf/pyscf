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
import numpy
from functools import reduce

from pyscf import gto, lib
from pyscf import scf
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import uccsd
from pyscf.cc import gccsd
from pyscf.cc import rccsd
from pyscf.cc import dfccsd

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol_grad = 1e-8
    ehf = mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_ccsd(self):
        mcc = cc.ccsd.CC(mf)
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        eris = mcc.ao2mo()
        emp2, t1, t2 = mcc.init_amps(eris)
        self.assertAlmostEqual(abs(t2).sum(), 4.9556571211255909, 5)
        self.assertAlmostEqual(emp2, -0.2040199672883385, 8)
        t1, t2 = cc.ccsd.update_amps(mcc, t1, t2, eris)
        self.assertAlmostEqual(abs(t1).sum(), 0.0475038989126, 6)
        self.assertAlmostEqual(abs(t2).sum(), 5.4018238455030, 5)
        self.assertAlmostEqual(cc.ccsd.energy(mcc, t1, t2, eris),
                               -0.208967840546667, 8)
        t1, t2 = cc.ccsd.update_amps(mcc, t1, t2, eris)
        self.assertAlmostEqual(cc.ccsd.energy(mcc, t1, t2, eris),
                               -0.212173678670510, 8)
        self.assertAlmostEqual(abs(t1).sum(), 0.05470123093500083, 6)
        self.assertAlmostEqual(abs(t2).sum(), 5.5605208386554716, 5)

        mcc.kernel()
        self.assertTrue(numpy.allclose(mcc.t2,mcc.t2.transpose(1,0,3,2)))
        self.assertAlmostEqual(mcc.ecc, -0.2133432312951, 8)
        self.assertAlmostEqual(abs(mcc.t2).sum(), 5.63970279799556984, 5)

    def test_ccsd_frozen(self):
        mcc = cc.ccsd.CC(mf, frozen=range(1))
        mcc.conv_tol = 1e-10
        mcc.kernel()
        self.assertAlmostEqual(mcc.ecc, -0.21124878189922872, 7)
        self.assertAlmostEqual(abs(mcc.t2).sum(), 5.4996425901189347, 5)

    def test_ccsd_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        pmol.build()
        mf = scf.RHF(pmol).set(conv_tol_grad=1e-8).run()
        mcc = cc.ccsd.CC(mf, frozen=range(1))
        mcc.conv_tol = 1e-10
        mcc.kernel()
        self.assertAlmostEqual(mcc.ecc, -0.21303885376969361, 8)

    def test_h2o_non_hf_orbital_high_cost(self):
        nmo = mf.mo_energy.size
        nocc = mol.nelectron // 2
        nvir = nmo - nocc
        u = numpy.eye(nmo)
        numpy.random.seed(1)
        u[:nocc,:nocc] = numpy.linalg.svd(numpy.random.random((nocc,nocc)))[0]
        u[nocc:,nocc:] = numpy.linalg.svd(numpy.random.random((nvir,nvir)))[0]
        mycc = cc.ccsd.CCSD(mf)
        mycc.conv_tol = 1e-12
        mycc.diis_start_energy_diff = 1e2
        mycc.max_cycle = 400
        mycc.mo_coeff = mo_coeff=numpy.dot(mf.mo_coeff,u)
        self.assertAlmostEqual(mycc.kernel()[0], -0.21334323320620596, 8)

## FIXME
#    def test_h2o_without_scf(self):
#        mycc = cc.ccsd.CCSD(mf)
#        nmo = mf.mo_energy.size
#        nocc = mol.nelectron // 2
#        nvir = nmo - nocc
#        numpy.random.seed(1)
#        u = numpy.eye(nmo) + numpy.random.random((nmo,nmo))*.2
#        u, w, vh = numpy.linalg.svd(u)
#        u = numpy.dot(u, vh)
#
#        mo1 = numpy.dot(mf.mo_coeff, u)
#        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
#
#        mycc.diis_start_energy_diff = 1e2
#        mycc.max_cycle = 1000
#        mycc.conv_tol = 1e-12
#        mycc.mo_coeff = mo1
#        self.assertAlmostEqual(mf.energy_tot(dm1)+mycc.kernel()[0],
#                               ehf-0.21334323320620596, 8)

    def test_ccsd_lambda(self):
        mcc = cc.ccsd.CC(mf)
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        mcc.kernel()
        mcc.solve_lambda()
        self.assertAlmostEqual(numpy.linalg.norm(mcc.l1), 0.01326267012100099, 6)
        self.assertAlmostEqual(numpy.linalg.norm(mcc.l2), 0.21257559872380857, 5)

    def test_ccsd_rdm(self):
        mcc = cc.ccsd.CC(mf)
        mcc.max_memory = 0
        mcc.conv_tol = 1e-9
        mcc.conv_tol_normt = 1e-7
        mcc.kernel()
        mcc.solve_lambda()
        dm1 = mcc.make_rdm1()
        dm2 = mcc.make_rdm2()
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 4.4227785269355078, 3)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 20.074587448789089, 3)

        nmo = mf.mo_coeff.shape[1]
        eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo)
        h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), mf.get_hcore(), mf.mo_coeff))
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mf.energy_nuc()
        self.assertAlmostEqual(e1, mcc.e_tot, 7)

    def test_scanner(self):
        cc_scanner = scf.RHF(mol).apply(cc.CCSD).as_scanner()
        cc_scanner.conv_tol = 1e-8
        self.assertAlmostEqual(cc_scanner(mol), -76.240108935038691, 6)
        geom = '''
        O   0.   0.       .1
        H   0.   -0.757   0.587
        H   0.   0.757    0.587'''
        cc_scanner = cc_scanner.as_scanner()
        self.assertAlmostEqual(cc_scanner(geom), -76.228972886940639, 6)

    def test_init(self):
        self.assertTrue(isinstance(cc.CCSD(mf), ccsd.CCSD))
        self.assertTrue(isinstance(cc.CCSD(mf.density_fit()), dfccsd.RCCSD))
        self.assertTrue(isinstance(cc.CCSD(mf.newton()), ccsd.CCSD))
        self.assertTrue(isinstance(cc.CCSD(mf.density_fit().newton()), dfccsd.RCCSD))
        self.assertTrue(isinstance(cc.CCSD(mf.newton().density_fit()), ccsd.CCSD))
        self.assertTrue(not isinstance(cc.CCSD(mf.newton().density_fit()), dfccsd.RCCSD))
        self.assertTrue(isinstance(cc.CCSD(mf.density_fit().newton().density_fit()), dfccsd.RCCSD))

        self.assertTrue(isinstance(cc.UCCSD(mf), uccsd.UCCSD))
#        self.assertTrue(isinstance(cc.UCCSD(mf.density_fit()), dfccsd.UCCSD))
        self.assertTrue(isinstance(cc.UCCSD(mf.newton()), uccsd.UCCSD))
#        self.assertTrue(isinstance(cc.UCCSD(mf.density_fit().newton()), dfccsd.UCCSD))
        self.assertTrue(isinstance(cc.UCCSD(mf.newton().density_fit()), uccsd.UCCSD))
#        self.assertTrue(not isinstance(cc.UCCSD(mf.newton().density_fit()), dfccsd.UCCSD))
#        self.assertTrue(isinstance(cc.UCCSD(mf.density_fit().newton().density_fit()), dfccsd.UCCSD))
        self.assertTrue(isinstance(cc.CCSD(scf.GHF(mol)), gccsd.GCCSD))

        umf = scf.convert_to_uhf(mf, scf.UHF(mol))
        self.assertTrue(isinstance(cc.CCSD(umf), uccsd.UCCSD))
#        self.assertTrue(isinstance(cc.CCSD(umf.density_fit()), dfccsd.UCCSD))
        self.assertTrue(isinstance(cc.CCSD(umf.newton()), uccsd.UCCSD))
#        self.assertTrue(isinstance(cc.CCSD(umf.density_fit().newton()), dfccsd.UCCSD))
        self.assertTrue(isinstance(cc.CCSD(umf.newton().density_fit()), uccsd.UCCSD))
#        self.assertTrue(not isinstance(cc.CCSD(umf.newton().density_fit()), dfccsd.UCCSD))
#        self.assertTrue(isinstance(cc.CCSD(umf.density_fit().newton().density_fit()), dfccsd.UCCSD))

        self.assertTrue(isinstance(cc.CCSD(mf, mo_coeff=mf.mo_coeff*1j), rccsd.RCCSD))


    # May be related to issue #509
    def test_race_condition_skip(self):
        current_nthreads = lib.num_threads()
        lib.num_threads(32)

        mycc = cc.ccsd.CCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        ecc = mycc.kernel()[0]
        self.assertAlmostEqual(ecc, -0.2133432312951, 8)

        lib.num_threads(current_nthreads)

if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
