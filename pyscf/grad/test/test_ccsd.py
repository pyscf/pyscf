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
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import ao2mo
from pyscf import cc
from pyscf import grad
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad import uccsd as uccsd_grad

mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = '631g'
mol.build()
mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-8
mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_ccsd_grad(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        l1, l2 = mycc.solve_lambda(eris=eris)
        g1 = ccsd_grad.kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), -0.036999389889460096, 6)

        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mf0 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc0 = cc.ccsd.CCSD(mf0).run(conv_tol=1e-10)
        mol.set_geom_('H 0 0 0; H 0 0 1.704', unit='Bohr')
        mf1 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc1= cc.ccsd.CCSD(mf1).run(conv_tol=1e-10)
        mol.set_geom_('H 0 0 0; H 0 0 1.705', unit='Bohr')
        mycc2 = cc.ccsd.CCSD(scf.RHF(mol))
        g_scanner = mycc2.nuc_grad_method().as_scanner().as_scanner()
        g1 = g_scanner(mol)[1]
        self.assertTrue(g_scanner.converged)
        self.assertAlmostEqual(g1[0,2], (mycc1.e_tot-mycc0.e_tot)*500, 6)

    def test_frozen(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.frozen = [0,1,10,11,12]
        mycc.diis_start_cycle = 1
        mycc.max_memory = 1
        g1 = mycc.nuc_grad_method().kernel(atmlst=range(mol.natm))
        self.assertAlmostEqual(lib.finger(g1), 0.10599503839207361, 6)

    def test_uccsd_grad(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.spin = 2
        mol.basis = '631g'
        mol.build(0, 0)
        mf = scf.UHF(mol)
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        mycc = cc.UCCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        l1, l2 = mycc.solve_lambda(eris=eris)
        g_scan = mycc.nuc_grad_method().as_scanner()
        g1 = g_scan(mol.atom)[1]
        self.assertAlmostEqual(lib.finger(g1), -0.22892720804519961, 6)

        cc_scanner = mycc.as_scanner()
        mol.set_geom_([
            [8 , (0. , 0.     , 0.001)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]], unit='Ang')
        e1 = cc_scanner(mol)
        mol.set_geom_([
            [8 , (0. , 0.     ,-0.001)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]], unit='Ang')
        e2 = cc_scanner(mol)
        self.assertAlmostEqual(g1[0,2], (e1-e2)/.002*lib.param.BOHR, 5)

    def test_rdm2_mo2ao(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.conv_tol = 1e-10
        mycc.diis_start_cycle = 1
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        l1, l2 = mycc.solve_lambda(eris=eris)
        fdm2 = lib.H5TmpFile()
        d2 = cc.ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)

        nao = mycc.mo_coeff.shape[0]
        ref = cc.ccsd_rdm._make_rdm2(mycc, None, d2, with_dm1=False)
        ref = lib.einsum('ijkl,pi,qj,rk,sl->pqrs', ref, mycc.mo_coeff,
                         mycc.mo_coeff, mycc.mo_coeff, mycc.mo_coeff)
        ref = ref + ref.transpose(0,1,3,2)
        ref = ref + ref.transpose(1,0,2,3)
        ref = ao2mo.restore(4, ref, nao) * .5
        rdm2 = ccsd_grad._rdm2_mo2ao(mycc, d2, mycc.mo_coeff)
        ccsd_grad._rdm2_mo2ao(mycc, d2, mycc.mo_coeff, fdm2)
        self.assertAlmostEqual(abs(ref-rdm2).max(), 0, 10)
        self.assertAlmostEqual(abs(ref-fdm2['dm2'].value).max(), 0, 10)
        self.assertAlmostEqual(lib.finger(rdm2), -0.32532303057849454, 6)

    def test_uccsd_rdm2_mo2ao(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.spin = 2
        mol.basis = '631g'
        mol.build(0, 0)
        mf = scf.UHF(mol)
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        mycc = cc.UCCSD(mf)
        mycc.diis_start_cycle = 1
        mycc.conv_tol = 1e-10
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        l1, l2 = mycc.solve_lambda(eris=eris)
        fdm2 = lib.H5TmpFile()
        d2 = cc.uccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)

        nao = mycc.mo_coeff[0].shape[0]
        ref = cc.uccsd_rdm._make_rdm2(mycc, None, d2, with_dm1=False)
        aa = lib.einsum('ijkl,pi,qj,rk,sl->pqrs', ref[0], mycc.mo_coeff[0],
                        mycc.mo_coeff[0], mycc.mo_coeff[0], mycc.mo_coeff[0])
        ab = lib.einsum('ijkl,pi,qj,rk,sl->pqrs', ref[1], mycc.mo_coeff[0],
                        mycc.mo_coeff[0], mycc.mo_coeff[1], mycc.mo_coeff[1])
        bb = lib.einsum('ijkl,pi,qj,rk,sl->pqrs', ref[2], mycc.mo_coeff[1],
                        mycc.mo_coeff[1], mycc.mo_coeff[1], mycc.mo_coeff[1])
        aa = aa + aa.transpose(0,1,3,2)
        aa = aa + aa.transpose(1,0,2,3)
        aa = ao2mo.restore(4, aa, nao) * .5
        bb = bb + bb.transpose(0,1,3,2)
        bb = bb + bb.transpose(1,0,2,3)
        bb = ao2mo.restore(4, bb, nao) * .5
        ab = ab + ab.transpose(0,1,3,2)
        ab = ab + ab.transpose(1,0,2,3)
        ab = ao2mo.restore(4, ab, nao) * .5
        ref = (aa+ab, bb+ab.T)
        rdm2 = uccsd_grad._rdm2_mo2ao(mycc, d2, mycc.mo_coeff)
        self.assertAlmostEqual(abs(ref[0]-rdm2[0]).max(), 0, 10)
        self.assertAlmostEqual(abs(ref[1]-rdm2[1]).max(), 0, 10)
        uccsd_grad._rdm2_mo2ao(mycc, d2, mycc.mo_coeff, fdm2)
        self.assertAlmostEqual(abs(ref[0]-fdm2['dm2aa+ab'].value).max(), 0, 10)
        self.assertAlmostEqual(abs(ref[1]-fdm2['dm2bb+ab'].value).max(), 0, 10)
        self.assertAlmostEqual(lib.finger(rdm2[0]), -1.6247203743431637, 7)
        self.assertAlmostEqual(lib.finger(rdm2[1]), -0.44062825991527471, 7)


if __name__ == "__main__":
    print("Tests for CCSD gradients")
    unittest.main()

