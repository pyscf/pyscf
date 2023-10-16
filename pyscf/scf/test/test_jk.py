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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.scf import jk

def setUpModule():
    global mol, mf
    mol = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = '''
    O     0    0        0
    H     0    -0.757   0.587
    H     0    0.757    0.587''',
        basis = '631g',
        cart = True,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-10)

def tearDownModule():
    global mol, mf
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_range_separated_Coulomb(self):
        '''test range-separated Coulomb'''
        with mol.with_range_coulomb(0.2):
            dm = mf.make_rdm1()
            vk0 = jk.get_jk(mol, dm, 'ijkl,jk->il', hermi=0)
            vk1 = jk.get_jk(mol, (dm,dm), ['ijkl,jk->il','ijkl,li->kj'], hermi=1)
            self.assertAlmostEqual(abs(vk1[0]-vk0).max(), 0, 9)
            self.assertAlmostEqual(abs(vk1[1]-vk0).max(), 0, 9)
            self.assertAlmostEqual(lib.fp(vk0), 0.87325708945599279, 9)

            vk = scf.hf.get_jk(mol, dm)[1]
            self.assertAlmostEqual(abs(vk-vk0).max(), 0, 12)
        vk = scf.hf.get_jk(mol, dm)[1]
        self.assertTrue(abs(vk-vk0).max() > 0.1)

    def test_shls_slice(self):
        dm = mf.make_rdm1()
        ao_loc = mol.ao_loc_nr()
        shls_slice = [0, 2, 1, 4, 2, 5, 0, 4]
        locs = [ao_loc[i] for i in shls_slice]
        i0, i1, j0, j1, k0, k1, l0, l1 = locs

        vs = jk.get_jk(mol, (dm[j0:j1,k0:k1], dm[l0:l1,k0:k1]),
                       ['ijkl,jk->il', 'ijkl,lk->ij'], hermi=0,
                       intor='int2e_ip1', shls_slice=shls_slice)
        self.assertEqual(vs[0].shape, (3,2,6))
        self.assertEqual(vs[1].shape, (3,2,5))

    def test_shls_slice1(self):
        mol = gto.M(atom='H 0 -.5 0; H 0 .5 0', basis='cc-pvdz')
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        mol1 = gto.M(atom='He 2 0 0', basis='6-31g')
        nao1 = mol1.nao_nr()
        dm1 = numpy.random.random((nao1,nao1))
        eri0 = gto.conc_mol(mol, mol1).intor('int2e_sph').reshape([nao+nao1]*4)

        j1part = jk.get_jk((mol1,mol1,mol,mol), dm1[:1,:1], scripts='ijkl,ji->kl', intor='int2e',
                           shls_slice=(0,1,0,1,0,mol.nbas,0,mol.nbas))
        j1ref = numpy.einsum('ijkl,ji->kl', eri0[nao:nao+1,nao:nao+1,:nao,:nao], dm1[:1,:1])
        self.assertAlmostEqual(abs(j1part - j1ref).max(), 0, 12)

        k1part = jk.get_jk((mol1,mol,mol,mol1), dm1[:,:1], scripts='ijkl,li->kj', intor='int2e',
                           shls_slice=(0,1,0,1,0,mol.nbas,0,mol1.nbas))
        k1ref = numpy.einsum('ijkl,li->kj', eri0[nao:nao+1,:1,:nao,nao:], dm1[:,:1])
        self.assertAlmostEqual(abs(k1part - k1ref).max(), 0, 12)

        j1part = jk.get_jk(mol, dm[:1,1:2], scripts='ijkl,ji->kl', intor='int2e',
                           shls_slice=(1,2,0,1,0,mol.nbas,0,mol.nbas))
        j1ref = numpy.einsum('ijkl,ji->kl', eri0[1:2,:1,:nao,:nao], dm[:1,1:2])
        self.assertAlmostEqual(abs(j1part - j1ref).max(), 0, 12)

        k1part = jk.get_jk(mol, dm[:,1:2], scripts='ijkl,li->kj', intor='int2e',
                           shls_slice=(1,2,0,1,0,mol.nbas,0,mol.nbas))
        k1ref = numpy.einsum('ijkl,li->kj', eri0[:1,1:2,:nao,:nao], dm[:,1:2])
        self.assertAlmostEqual(abs(k1part - k1ref).max(), 0, 12)

    def test_mols(self):
        pmol = mol.copy(deep=False)
        mols = (mol, pmol, pmol, mol)
        dm = mf.make_rdm1()
        vj0 = jk.get_jk(mols, dm, 'ijkl,lk->ij')
        vj1 = scf.hf.get_jk(mol, dm)[0]
        self.assertAlmostEqual(abs(vj1-vj0).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(vj0), 28.36214139459754, 6)

    def test_vk_s8(self):
        mol = gto.M(atom='H 0 -.5 0; H 0 .5 0; H 1.1 0.2 0.2; H 0.6 0.5 0.4',
                    basis='cc-pvdz')
        ao_loc = mol.ao_loc_nr()
        eri0 = mol.intor('int2e')
        nao = mol.nao
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))

        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm)
        self.assertAlmostEqual(abs(vk0-get_vk_s4(mol, dm)).max(), 0, 12)
        self.assertAlmostEqual(abs(vk0-get_vk_s8(mol, dm)).max(), 0, 12)

        shls_slice = (2,4,0,2,0,8,0,5)
        i0,i1,j0,j1,k0,k1,l0,l1 = [ao_loc[x] for x in shls_slice]
        vk0 = numpy.einsum('ijkl,jk->il', eri0[i0:i1,j0:j1,k0:k1,l0:l1], dm[j0:j1,k0:k1])
        vk1 = numpy.einsum('ijkl,jl->ik', eri0[i0:i1,j0:j1,k0:k1,l0:l1], dm[j0:j1,l0:l1])
        vk2 = numpy.einsum('ijkl,ik->jl', eri0[i0:i1,j0:j1,k0:k1,l0:l1], dm[i0:i1,k0:k1])
        vk3 = numpy.einsum('ijkl,il->jk', eri0[i0:i1,j0:j1,k0:k1,l0:l1], dm[i0:i1,l0:l1])
        vk = jk.get_jk(mol,
                       [dm[j0:j1,k0:k1], dm[j0:j1,l0:l1], dm[i0:i1,k0:k1], dm[i0:i1,l0:l1]],
                       scripts=['ijkl,jk->il', 'ijkl,jl->ik', 'ijkl,ik->jl', 'ijkl,il->jk'],
                       shls_slice=shls_slice)
        self.assertAlmostEqual(abs(vk0-vk[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(vk1-vk[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(vk2-vk[2]).max(), 0, 12)
        self.assertAlmostEqual(abs(vk3-vk[3]).max(), 0, 12)


def get_vk_s4(mol, dm):
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vk = numpy.zeros((nao,nao))
    bas_groups = list(lib.prange(0, mol.nbas, 3))
    for ip, (ish0, ish1) in enumerate(bas_groups):
        for jp, (jsh0, jsh1) in enumerate(bas_groups[:ip]):
            for kp, (ksh0, ksh1) in enumerate(bas_groups):
                for lp, (lsh0, lsh1) in enumerate(bas_groups[:kp]):
                    shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
                    i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
                    dms = [dm[j0:j1,k0:k1],
                           dm[i0:i1,k0:k1],
                           dm[j0:j1,l0:l1],
                           dm[i0:i1,l0:l1]]
                    scripts = ['ijkl,jk->il',
                               'ijkl,ik->jl',
                               'ijkl,jl->ik',
                               'ijkl,il->jk']
                    kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice)
                    vk[i0:i1,l0:l1] += kparts[0]
                    vk[j0:j1,l0:l1] += kparts[1]
                    vk[i0:i1,k0:k1] += kparts[2]
                    vk[j0:j1,k0:k1] += kparts[3]

                lsh0, lsh1 = ksh0, ksh1
                shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
                i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
                kparts = jk.get_jk(mol,
                                   [dm[j0:j1,k0:k1], dm[i0:i1,k0:k1]],
                                   scripts=['ijkl,jk->il', 'ijkl,ik->jl'],
                                   shls_slice=shls_slice)
                vk[i0:i1,l0:l1] += kparts[0]
                vk[j0:j1,l0:l1] += kparts[1]

        jsh0, jsh1 = ish0, ish1
        for kp, (ksh0, ksh1) in enumerate(bas_groups):
            for lp, (lsh0, lsh1) in enumerate(bas_groups[:kp]):
                shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
                i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
                kparts = jk.get_jk(mol,
                                   [dm[j0:j1,k0:k1], dm[j0:j1,l0:l1]],
                                   scripts=['ijkl,jk->il', 'ijkl,jl->ik'],
                                   shls_slice=shls_slice)
                vk[i0:i1,l0:l1] += kparts[0]
                vk[j0:j1,k0:k1] += kparts[1]

            lsh0, lsh1 = ksh0, ksh1
            shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
            i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
            kparts = jk.get_jk(mol,
                               [dm[j0:j1,k0:k1]],
                               scripts=['ijkl,jk->il'],
                               shls_slice=shls_slice)
            vk[i0:i1,l0:l1] += kparts[0]
    return vk

def get_vk_s8(mol, dm):
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vk = numpy.zeros((nao,nao))
    bas_groups = list(lib.prange(0, mol.nbas, 3))
    for ip, (ish0, ish1) in enumerate(bas_groups):
        for jp, (jsh0, jsh1) in enumerate(bas_groups[:ip]):
            for kp, (ksh0, ksh1) in enumerate(bas_groups[:ip]):
                for lp, (lsh0, lsh1) in enumerate(bas_groups[:kp]):
                    shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
                    i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
                    dms = [dm[j0:j1,k0:k1],
                           dm[j0:j1,l0:l1],
                           dm[i0:i1,k0:k1],
                           dm[i0:i1,l0:l1],
                           dm[l0:l1,i0:i1],
                           dm[l0:l1,j0:j1],
                           dm[k0:k1,i0:i1],
                           dm[k0:k1,j0:j1]]
                    scripts = ['ijkl,jk->il',
                               'ijkl,jl->ik',
                               'ijkl,ik->jl',
                               'ijkl,il->jk',
                               'ijkl,li->kj',
                               'ijkl,lj->ki',
                               'ijkl,ki->lj',
                               'ijkl,kj->li']
                    kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice)
                    vk[i0:i1,l0:l1] += kparts[0]
                    vk[i0:i1,k0:k1] += kparts[1]
                    vk[j0:j1,l0:l1] += kparts[2]
                    vk[j0:j1,k0:k1] += kparts[3]
                    vk[k0:k1,j0:j1] += kparts[4]
                    vk[k0:k1,i0:i1] += kparts[5]
                    vk[l0:l1,j0:j1] += kparts[6]
                    vk[l0:l1,i0:i1] += kparts[7]

                # ip > jp, ip > kp, kp == lp
                lsh0, lsh1 = ksh0, ksh1
                shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
                i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
                dms = [dm[j0:j1,k0:k1],
                       dm[i0:i1,k0:k1],
                       dm[k0:k1,j0:j1],
                       dm[k0:k1,i0:i1]]
                scripts = ['ijkl,jk->il',
                           'ijkl,ik->jl',
                           'ijkl,kj->li',
                           'ijkl,ki->lj']
                kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice)
                vk[i0:i1,l0:l1] += kparts[0]
                vk[j0:j1,l0:l1] += kparts[1]
                vk[l0:l1,i0:i1] += kparts[2]
                vk[l0:l1,j0:j1] += kparts[3]

            # ip == kp and ip > jp and kp > lp
            kp, ksh0, ksh1 = ip, ish0, ish1
            for lp, (lsh0, lsh1) in enumerate(bas_groups[:kp]):
                shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
                i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
                dms = [dm[j0:j1,k0:k1],
                       dm[i0:i1,k0:k1],
                       dm[j0:j1,l0:l1],
                       dm[i0:i1,l0:l1]]
                scripts = ['ijkl,jk->il',
                           'ijkl,ik->jl',
                           'ijkl,jl->ik',
                           'ijkl,il->jk']
                kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice)
                vk[i0:i1,l0:l1] += kparts[0]
                vk[j0:j1,l0:l1] += kparts[1]
                vk[i0:i1,k0:k1] += kparts[2]
                vk[j0:j1,k0:k1] += kparts[3]

        # ip == jp and ip >= kp
        jsh0, jsh1 = ish0, ish1
        for kp, (ksh0, ksh1) in enumerate(bas_groups[:ip+1]):
            for lp, (lsh0, lsh1) in enumerate(bas_groups[:kp]):
                shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
                i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
                dms = [dm[j0:j1,k0:k1],
                       dm[j0:j1,l0:l1],
                       dm[k0:k1,j0:j1],
                       dm[l0:l1,j0:j1]]
                scripts = ['ijkl,jk->il',
                           'ijkl,jl->ik',
                           'ijkl,kj->li',
                           'ijkl,lj->ki']
                kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice)
                vk[i0:i1,l0:l1] += kparts[0]
                vk[i0:i1,k0:k1] += kparts[1]
                vk[l0:l1,i0:i1] += kparts[2]
                vk[k0:k1,i0:i1] += kparts[3]

        # ip == jp and ip > kp and kp == lp
        for kp, (ksh0, ksh1) in enumerate(bas_groups[:ip]):
            lsh0, lsh1 = ksh0, ksh1
            shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
            i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
            dms = [dm[j0:j1,k0:k1],
                   dm[l0:l1,i0:i1]]
            scripts = ['ijkl,jk->il',
                       'ijkl,li->kj']
            kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice)
            vk[i0:i1,l0:l1] += kparts[0]
            vk[k0:k1,j0:j1] += kparts[1]

        # ip == jp == kp == lp
        kp, ksh0, ksh1 = ip, ish0, ish1
        lsh0, lsh1 = ksh0, ksh1
        shls_slice = (ish0, ish1, jsh0, jsh1, ksh0, ksh1, lsh0, lsh1)
        i0, i1, j0, j1, k0, k1, l0, l1 = [ao_loc[x] for x in shls_slice]
        kparts = jk.get_jk(mol,
                           [dm[j0:j1,k0:k1]],
                           scripts=['ijkl,jk->il'],
                           shls_slice=shls_slice)
        vk[i0:i1,l0:l1] += kparts[0]
    return vk

if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()
