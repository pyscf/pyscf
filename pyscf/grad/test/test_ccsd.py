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

def setUpModule():
    global mol, mf
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
        g1 = ccsd_grad.Gradients(mycc).kernel(t1, t2, l1, l2)
#[[ 0   0                1.00950925e-02]
# [ 0   2.28063426e-02  -5.04754623e-03]
# [ 0  -2.28063426e-02  -5.04754623e-03]]
        self.assertAlmostEqual(lib.fp(g1), -0.036999389889460096, 6)

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
        mycc2.frozen = [] # test has_frozen_orbitals
        g_scanner = mycc2.nuc_grad_method().as_scanner().as_scanner()
        g1 = g_scanner(mol)[1]
        self.assertTrue(g_scanner.converged)
        self.assertAlmostEqual(g1[0,2], (mycc1.e_tot-mycc0.e_tot)*500, 6)

    def test_ccsd_frozen(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.frozen = [0,1,10,11,12]
        mycc.diis_start_cycle = 1
        mycc.max_memory = 1
        g1 = mycc.nuc_grad_method().kernel(atmlst=range(mol.natm))
#[[ -7.81105940e-17   3.81840540e-15   1.20415540e-02]
# [  1.73095055e-16  -7.94568837e-02  -6.02077699e-03]
# [ -9.49844615e-17   7.94568837e-02  -6.02077699e-03]]
        self.assertAlmostEqual(lib.fp(g1), 0.10599503839207361, 6)

        mycc = mf.CCSD()
        mycc.frozen = [0,1]
        g_scan = mycc.nuc_grad_method().as_scanner()
        e, g1 = g_scan(mol)
        self.assertAlmostEqual(e, -76.07649382891177, 7)
        self.assertAlmostEqual(lib.fp(g1), -0.03152584, 6)

        mycc.frozen = 2
        g1 = mycc.nuc_grad_method().kernel()
        self.assertAlmostEqual(e, -76.07649382891177, 7)
        self.assertAlmostEqual(lib.fp(g1), -0.03152584, 6)

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
        self.assertAlmostEqual(abs(ref-fdm2['dm2'][:]).max(), 0, 10)
        self.assertAlmostEqual(lib.fp(rdm2), -0.32532303057849454, 6)

    def test_with_x2c_scanner(self):
        with lib.light_speed(20.):
            mycc = cc.ccsd.CCSD(mf.x2c())
            mycc.frozen = [0,1,10,11,12]
            gscan = mycc.nuc_grad_method().as_scanner().as_scanner()
            e, g1 = gscan(mol)

            cs = mycc.as_scanner()
            e1 = cs(gto.M(
                atom=[[8 , (0. , 0.     , 0.)],
                      [1 , (0. , -0.757 , 0.5871)],
                      [1 , (0. ,  0.757 , 0.587)]],
                basis='631g'))
            e2 = cs(gto.M(
                atom=[[8 , (0. , 0.     , 0.)],
                      [1 , (0. , -0.757 , 0.5869)],
                      [1 , (0. ,  0.757 , 0.587)]],
                basis='631g'))
            self.assertAlmostEqual(g1[1,2], (e1-e2)/0.0002*lib.param.BOHR, 5)

    def test_with_qmmm_scanner(self):
        from pyscf import qmmm
        mol = gto.Mole()
        mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                       H                 -0.00000000   -0.84695236    0.59109389
                       H                 -0.00000000    0.89830571    0.52404783 '''
        mol.verbose = 0
        mol.basis = '6-31g'
        mol.build()

        coords = [(0.5,0.6,0.1)]
        #coords = [(0.0,0.0,0.0)]
        charges = [-0.1]
        mf = qmmm.add_mm_charges(scf.RHF(mol), coords, charges)
        ccs = cc.ccsd.CCSD(mf).as_scanner()
        e1 = ccs(''' O                  0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        e2 = ccs(''' O                 -0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        ref = (e1 - e2)/0.002 * lib.param.BOHR
        g = ccs.nuc_grad_method().kernel()
        self.assertAlmostEqual(g[0,0], ref, 5)

    def test_symmetrize(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True)
        g = mol.RHF.run().CCSD().run().Gradients().kernel()
        self.assertAlmostEqual(lib.fp(g), 0.10105388861195158, 6)


if __name__ == "__main__":
    print("Tests for CCSD gradients")
    unittest.main()
