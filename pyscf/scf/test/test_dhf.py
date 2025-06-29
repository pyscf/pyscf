#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf import lib

def setUpModule():
    global mol, mf, h4
    mol = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = '''
            O     0    0        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis = '631g',
    )

    mf = scf.dhf.UHF(mol)
    mf.conv_tol_grad = 1e-5
    mf.kernel()

    h4 = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = '''
            H     0    0        1
            H     1    1        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis = ('sto3g', [[1,[0.3,1]]]),
    )

def tearDownModule():
    global mol, mf, h4
    mol.stdout.close()
    h4.stdout.close()
    del mol, mf, h4


class KnownValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = scf.dhf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 14.859714177083553, 8)

    def test_init_guess_huckel(self):
        dm = scf.dhf.DHF(mol).get_init_guess(mol, key='huckel')
        self.assertAlmostEqual(lib.fp(dm), (-0.6090467376579871-0.08968155321478456j), 8)

    def test_init_guess_mod_huckel(self):
        dm = scf.dhf.DHF(mol).get_init_guess(mol, key='mod_huckel')
        self.assertAlmostEqual(lib.fp(dm), (-0.5563045659111319-0.0897593233637678j), 8)

    def test_init_guess_sap(self):
        dm = scf.dhf.DHF(mol).get_init_guess(mol, key='sap')
        self.assertAlmostEqual(lib.fp(dm), (-0.3165252968705663-0.21867900769448959j), 8)

    def test_get_hcore(self):
        h = mf.get_hcore()
        self.assertAlmostEqual(numpy.linalg.norm(h), 129.81389477933607, 7)

    def test_get_ovlp(self):
        s = mf.get_ovlp()
        self.assertAlmostEqual(numpy.linalg.norm(s), 6.9961451281502809, 9)

    def test_1e(self):
        mf = scf.dhf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.888778707255078, 7)

    def test_analyze(self):
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 2.2858506185320837, 6)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.038520455193833, 6)

    def test_energy_tot(self):
        e = mf.energy_tot(mf.make_rdm1())
        self.assertAlmostEqual(e, mf.e_tot, 9)

    def test_get_grad(self):
        g = mf.get_grad(mf.mo_coeff, mf.mo_occ)
        self.assertAlmostEqual(abs(g).max(), 0, 5)

    if scf.dhf.zquatev:
        def test_rhf(self):
            mol = gto.M(
                verbose = 5,
                output = '/dev/null',
                atom = '''
                    O     0    0        0
                    H     0    -0.757   0.587
                    H     0    0.757    0.587''',
                basis = '631g',
            )
            mf = scf.dhf.RHF(mol)
            mf.with_ssss = False
            mf.conv_tol_grad = 1e-5
            self.assertAlmostEqual(mf.kernel(), -76.03852477545016, 8)

            mf.ssss_approx = None
            mf.conv_tol_grad = 1e-5
            self.assertAlmostEqual(mf.kernel(), -76.03852480744785, 8)

    def test_get_veff(self):
        n4c = mol.nao_2c() * 2
        numpy.random.seed(1)
        dm = numpy.random.random((n4c,n4c))+numpy.random.random((n4c,n4c))*1j
        dm = dm + dm.T.conj()
        v = mf.get_veff(mol, dm)
        self.assertAlmostEqual(lib.fp(v), (-21.613084684028077-28.50754366262467j), 8)

        mf1 = mf.copy()
        mf1.direct_scf = False
        v1 = mf1.get_veff(mol, dm)
        self.assertAlmostEqual(abs(v-v1).max(), 0, 9)

    def test_get_jk(self):
        n2c = h4.nao_2c()
        n4c = n2c * 2
        c1 = .5 / lib.param.LIGHT_SPEED
        eri0 = numpy.zeros((n4c,n4c,n4c,n4c), dtype=numpy.complex128)
        eri0[:n2c,:n2c,:n2c,:n2c] = h4.intor('int2e_spinor')
        eri0[n2c:,n2c:,:n2c,:n2c] = h4.intor('int2e_spsp1_spinor') * c1**2
        eri0[:n2c,:n2c,n2c:,n2c:] = eri0[n2c:,n2c:,:n2c,:n2c].transpose(2,3,0,1)
        ssss = h4.intor('int2e_spsp1spsp2_spinor') * c1**4
        eri0[n2c:,n2c:,n2c:,n2c:] = ssss
        eri0bak = eri0.copy()

        # to setup mf.opt
        mf = h4.DHF().build()
        mf._coulomb_level = 'SSSS'

        numpy.random.seed(1)
        dm = numpy.random.random((2,n4c,n4c))+numpy.random.random((2,n4c,n4c))*1j
        dm = dm + dm.transpose(0,2,1).conj()
        vj0 = numpy.einsum('ijkl,lk->ij', eri0, dm[0])
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm[0])
        vj, vk = mf.get_jk(h4, dm[0], hermi=1)
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

        vj0 = numpy.einsum('ijkl,xlk->xij', ssss, dm[:,n2c:,n2c:])
        vk0 = numpy.einsum('ijkl,xjk->xil', ssss, dm[:,n2c:,n2c:])
        vj, vk = scf.dhf._call_veff_ssss(h4, dm, hermi=1)
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

        eri0[n2c:,n2c:,n2c:,n2c:] = 0
        vj0 = numpy.einsum('ijkl,xlk->xij', eri0, dm)
        vk0 = numpy.einsum('ijkl,xjk->xil', eri0, dm)
        vj, vk = scf.dhf.get_jk(h4, dm, hermi=1, coulomb_allow='SSLL')
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

        eri0[n2c:,n2c:,:n2c,:n2c] = 0
        eri0[:n2c,:n2c,n2c:,n2c:] = 0
        vj0 = numpy.einsum('ijkl,lk->ij', eri0, dm[0])
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm[0])
        vj, vk = scf.dhf.get_jk(h4, dm[0], hermi=1, coulomb_allow='LLLL')
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

        eri0 = eri0bak
        numpy.random.seed(1)
        dm = numpy.random.random((2,n4c,n4c))+numpy.random.random((2,n4c,n4c))*1j
        vj0 = numpy.einsum('ijkl,lk->ij', eri0, dm[0])
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm[0])
        vj, vk = mf.get_jk(h4, dm[0], hermi=0)
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

        vj0 = numpy.einsum('ijkl,xlk->xij', ssss, dm[:,n2c:,n2c:])
        vk0 = numpy.einsum('ijkl,xjk->xil', ssss, dm[:,n2c:,n2c:])
        vj, vk = scf.dhf._call_veff_ssss(h4, dm, hermi=0)
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

        eri0[n2c:,n2c:,n2c:,n2c:] = 0
        vj0 = numpy.einsum('ijkl,xlk->xij', eri0, dm)
        vk0 = numpy.einsum('ijkl,xjk->xil', eri0, dm)
        vj, vk = scf.dhf.get_jk(h4, dm, hermi=0, coulomb_allow='SSLL')
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

        eri0[n2c:,n2c:,:n2c,:n2c] = 0
        eri0[:n2c,:n2c,n2c:,n2c:] = 0
        vj0 = numpy.einsum('ijkl,lk->ij', eri0, dm[0])
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm[0])
        vj, vk = scf.dhf.get_jk(h4, dm[0], hermi=0, coulomb_allow='LLLL')
        self.assertAlmostEqual(abs(vj0 - vj).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk).max(), 0, 12)

    def test_get_jk_with_gaunt_breit_high_cost(self):
        n2c = h4.nao_2c()
        n4c = n2c * 2
        c1 = .5 / lib.param.LIGHT_SPEED
        eri0 = numpy.zeros((n4c,n4c,n4c,n4c), dtype=numpy.complex128)
        eri0[:n2c,:n2c,:n2c,:n2c] = h4.intor('int2e_spinor')
        eri0[n2c:,n2c:,:n2c,:n2c] = h4.intor('int2e_spsp1_spinor') * c1**2
        eri0[:n2c,:n2c,n2c:,n2c:] = eri0[n2c:,n2c:,:n2c,:n2c].transpose(2,3,0,1)
        eri0[n2c:,n2c:,n2c:,n2c:] = h4.intor('int2e_spsp1spsp2_spinor') * c1**4

        numpy.random.seed(1)
        dm = numpy.random.random((2,n4c,n4c))+numpy.random.random((2,n4c,n4c))*1j
        dm = dm + dm.transpose(0,2,1).conj()

        eri1 = eri0.copy()
        eri0 -= _fill_gaunt(h4, h4.intor('int2e_ssp1ssp2_spinor') * c1**2)
        vj0 = numpy.einsum('ijkl,xlk->xij', eri0, dm)
        vk0 = numpy.einsum('ijkl,xjk->xil', eri0, dm)

        mf = scf.dhf.DHF(h4)
        mf.with_gaunt = True
        vj1, vk1 = mf.get_jk(h4, dm, hermi=1)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(vk0, vk1))

        eri1 += _fill_gaunt(h4, h4.intor('int2e_breit_ssp1ssp2_spinor', comp=1) * c1**2)
        vj0 = numpy.einsum('ijkl,xlk->xij', eri1, dm)
        vk0 = numpy.einsum('ijkl,xjk->xil', eri1, dm)

        mf = scf.dhf.DHF(h4)
        mf.with_breit = True
        vj1, vk1 = mf.get_jk(h4, dm, hermi=1)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(vk0, vk1))

    def test_gaunt(self):
        erig = _fill_gaunt(h4, h4.intor('int2e_ssp1ssp2_spinor'))

        n4c = erig.shape[0]
        numpy.random.seed(1)
        dm = numpy.random.random((2,n4c,n4c))+numpy.random.random((2,n4c,n4c))*1j
        c1 = .5 / lib.param.LIGHT_SPEED
        vj0 = -numpy.einsum('ijkl,xlk->xij', erig, dm) * c1**2
        vk0 = -numpy.einsum('ijkl,xjk->xil', erig, dm) * c1**2
        vj1, vk1 = scf.dhf._call_veff_gaunt_breit(h4, dm, hermi=0)
        self.assertAlmostEqual(abs(vj0 - vj1).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk1).max(), 0, 12)

        dm = dm + dm.transpose(0,2,1).conj()
        c1 = .5 / lib.param.LIGHT_SPEED
        vj0 = -numpy.einsum('ijkl,xlk->xij', erig, dm) * c1**2
        vk0 = -numpy.einsum('ijkl,xjk->xil', erig, dm) * c1**2
        vj1, vk1 = scf.dhf._call_veff_gaunt_breit(h4, dm)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(vk0, vk1))

#    def test_breit(self):
#        mol = gto.M(atom='Cl',
#                    basis={'Cl': gto.parse('''
#                                           Cl  S 5.5    1.0
#                                           Cl  P 9.053563477       1.0''')},
#                    charge=9)
#        mf = mol.DHF().set(with_breit=True)
#        mf.run()
#        self.assertTrue(mf.e_tot, -234.888983310961, 8)
#
#        mf.with_ssss = False
#        mf.run()
#        self.assertTrue(mf.e_tot, -234.888999687936, 8)

    def test_breit_high_cost(self):
        erig = _fill_gaunt(h4, h4.intor('int2e_breit_ssp1ssp2_spinor', comp=1))

        n4c = erig.shape[0]
        numpy.random.seed(1)
        dm = numpy.random.random((n4c,n4c))+numpy.random.random((n4c,n4c))*1j
        c1 = .5 / lib.param.LIGHT_SPEED
        vj0 = -numpy.einsum('ijkl,xlk->xij', erig, dm) * c1**2
        vk0 = -numpy.einsum('ijkl,xjk->xil', erig, dm) * c1**2
        vj1, vk1 = scf.dhf._call_veff_gaunt_breit(h4, dm, hermi=0)
        self.assertAlmostEqual(abs(vj0 - vj1).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0 - vk1).max(), 0, 12)

        dm = dm + dm.T.conj()
        c1 = .5 / lib.param.LIGHT_SPEED
        vj0 = numpy.einsum('ijkl,lk->ij', erig, dm) * c1**2
        vk0 = numpy.einsum('ijkl,jk->il', erig, dm) * c1**2
        vj1, vk1 = scf.dhf._call_veff_gaunt_breit(h4, dm, with_breit=True)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(vk0, vk1))

    def test_time_rev_matrix(self):
        s = mol.intor_symmetric('int1e_ovlp_spinor')
        ts = scf.dhf.time_reversal_matrix(mol, s)
        self.assertTrue(numpy.allclose(s, ts))

    def test_get_occ(self):
        mo_energy = mf.mo_energy.copy()
        n2c = mo_energy.size // 2
        mo_energy[n2c] -= lib.param.LIGHT_SPEED**2*2
        mo_energy[n2c+6] -= lib.param.LIGHT_SPEED**2*2
        occ = mf.get_occ(mo_energy)
        self.assertEqual(list(occ[n2c:]),
                [0.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,1.,
                 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    def test_x2c(self):
        mfx2c = mf.x2c().run()
        self.assertAlmostEqual(mfx2c.e_tot, -76.032703699443999, 9)

    def test_h2_sto3g(self):
        # There was a bug of cache size in lib/vhf/r_direct.c for minimal
        # system
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto3g', verbose=0)
        e = mol.DHF().kernel()
        self.assertAlmostEqual(e, -1.066122658859047, 12)

def _fill_gaunt(mol, erig):
    n2c = erig.shape[0]
    n4c = n2c * 2

    tao = numpy.asarray(mol.time_reversal_map())
    idx = abs(tao)-1 # -1 for C indexing convention
    sign_mask = tao<0

    eri0 = numpy.zeros((n4c,n4c,n4c,n4c), dtype=numpy.complex128)
    eri0[:n2c,n2c:,:n2c,n2c:] = erig # ssp1ssp2

    eri2 = erig[idx[:,None],idx] # sps1ssp2
    eri2[sign_mask,:] *= -1
    eri2[:,sign_mask] *= -1
    eri2 = -eri2.transpose(1,0,2,3)
    eri0[n2c:,:n2c,:n2c,n2c:] = eri2

    eri2 = erig[:,:,idx[:,None],idx] # ssp1sps2
    eri2[:,:,sign_mask,:] *= -1
    eri2[:,:,:,sign_mask] *= -1
    eri2 = -eri2.transpose(0,1,3,2)
    #self.assertTrue(numpy.allclose(eri0, eri2))
    eri0[:n2c,n2c:,n2c:,:n2c] = eri2

    eri2 = erig[idx[:,None],idx]
    eri2 = eri2[:,:,idx[:,None],idx] # sps1sps2
    eri2 = eri2.transpose(1,0,3,2)
    eri2[sign_mask,:] *= -1
    eri2[:,sign_mask] *= -1
    eri2[:,:,sign_mask,:] *= -1
    eri2[:,:,:,sign_mask] *= -1
    eri0[n2c:,:n2c,n2c:,:n2c] = eri2
    return eri0


if __name__ == "__main__":
    print("Full Tests for dhf")
    unittest.main()
