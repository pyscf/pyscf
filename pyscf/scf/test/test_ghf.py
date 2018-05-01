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

import unittest
import numpy
import scipy.linalg
from functools import reduce

from pyscf import gto
from pyscf import lib
from pyscf import scf

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)
mf = scf.GHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

molsym = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = 'cc-pvdz',
    symmetry = 'c2v'
)
mfsym = scf.GHF(molsym)

def tearDownModule():
    global mol, mf, molsym, mfsym
    mol.stdout.close()
    molsym.stdout.close()
    del mol, mf, molsym, mfsym

def spin_square(mol, mo):
    s = mol.intor('int1e_ovlp')
    nao = s.shape[0]
    sx = numpy.zeros((nao*2,nao*2))
    sy = numpy.zeros((nao*2,nao*2), dtype=numpy.complex128)
    sz = numpy.zeros((nao*2,nao*2))
    s1 = numpy.zeros((nao*2,nao*2))
    sx[:nao,nao:] = .5 * s
    sx[nao:,:nao] = .5 * s
    sy[:nao,nao:] =-.5j* s
    sy[nao:,:nao] = .5j* s
    sz[:nao,:nao] = .5 * s
    sz[nao:,nao:] =-.5 * s
    sx = reduce(numpy.dot, (mo.T.conj(), sx, mo))
    sy = reduce(numpy.dot, (mo.T.conj(), sy, mo))
    sz = reduce(numpy.dot, (mo.T.conj(), sz, mo))
    ss = numpy.einsum('ij,kl->ijkl', sx, sx) + 0j
    ss+= numpy.einsum('ij,kl->ijkl', sy, sy)
    ss+= numpy.einsum('ij,kl->ijkl', sz, sz)
    nmo = mo.shape[1]
    dm2 = numpy.einsum('ij,kl->ijkl', numpy.eye(nmo), numpy.eye(nmo))
    dm2-= numpy.einsum('jk,il->ijkl', numpy.eye(nmo), numpy.eye(nmo))
    ss = numpy.einsum('ijkl,ijkl', ss, dm2).real

    s1[:nao,:nao] = s
    s1[nao:,nao:] = s
    s1 = reduce(numpy.dot, (mo.T.conj(), s1, mo))
    ss+= s1.trace().real * .75
    return ss

class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = mf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 14.00554247575052, 9)

    def test_init_guess_chk(self):
        dm = scf.ghf.GHF(mol).get_init_guess(mol, key='chkfile')
        self.assertAlmostEqual(lib.finger(dm), 1.8117584283411752, 9)

        dm = mf.get_init_guess(mol, key='chkfile')
        self.assertAlmostEqual(lib.finger(dm), 3.2111753674560535, 9)

    def test_ghf_complex(self):
        dm = mf.init_guess_by_1e(mol) + 0j
        nao = dm.shape[0] // 2
        numpy.random.seed(12)
        dm[:nao,nao:] = numpy.random.random((nao,nao)) * .1j
        dm[nao:,:nao] = dm[:nao,nao:].T.conj()
        mf.kernel(dm)
        self.assertAlmostEqual(mf.e_tot, -76.0267656731, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()*2
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = mf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 556.53059717681901, 9)

    def test_get_jk(self):
        nao = mol.nao_nr()*2
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        vj, vk = mf.get_jk(mol, d)
        self.assertEqual(vj.shape, (2,nao,nao))
        self.assertEqual(vk.shape, (2,nao,nao))
        self.assertAlmostEqual(lib.finger(vj), 246.24944977538354, 9)
        self.assertAlmostEqual(lib.finger(vk), 37.840557968925779, 9)

        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao)) + 1j*numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao)) + 1j*numpy.random.random((nao,nao))
        d = (d1+d1.T.conj(), d2+d2.T.conj())
        vj, vk = mf.get_jk(mol, d)
        self.assertEqual(vj.shape, (2,nao,nao))
        self.assertEqual(vk.shape, (2,nao,nao))
        self.assertAlmostEqual(lib.finger(vj), 254.68614111766146+0j, 9)
        self.assertAlmostEqual(lib.finger(vk), 53.629159066971539-2.1298002812909353j, 9)

        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        vj, vk = mf.get_jk(mol, d)
        self.assertEqual(vj.shape, (2,nao,nao))
        self.assertEqual(vk.shape, (2,nao,nao))
        self.assertAlmostEqual(lib.finger(vj), -388.17756605981504, 9)
        self.assertAlmostEqual(lib.finger(vk), -84.276190743451622, 9)

    def test_spin_square(self):
        nao = mol.nao_nr()
        s = mol.intor('int1e_ovlp')
        w, v = numpy.linalg.eigh(s)
        x = numpy.dot(v/w**.5, v.T)
        moa = x[:,:5]
        mob = x[:,2:7]
        mo = scipy.linalg.block_diag(moa, mob)
        ssref = scf.uhf.spin_square((moa,mob), s)[0]
        ss = mf.spin_square(mo, s)[0]
        self.assertAlmostEqual(ssref, ss, 9)
        ssref = spin_square(mol, mo)
        self.assertAlmostEqual(ssref, ss, 9)

        numpy.random.seed(1)
        mo = numpy.random.random((nao*2,10))*.1
        ss, s = mf.spin_square(mo)
        self.assertAlmostEqual(ss, 2.043727425109497, 9)
        ssref = spin_square(mol, mo)
        self.assertAlmostEqual(ssref, ss, 9)

        mo = mo + 1j*numpy.random.random((nao*2,10))*.1
        ss, s = mf.spin_square(mo)
        self.assertAlmostEqual(ss, 3.9543837879512358, 9)
        ssref = spin_square(mol, mo)
        self.assertAlmostEqual(ssref, ss, 9)

    def test_canonicalize(self):
        mo = mf.mo_coeff + 0j
        nocc = numpy.count_nonzero(mf.mo_occ > 0)
        nvir = mf.mo_occ.size - nocc
        numpy.random.seed(1)
        t = numpy.random.random((nocc,nvir))+1j*numpy.random.random((nocc,nvir))
        u, w, vh = numpy.linalg.svd(t)
        mo[:,:nocc] = numpy.dot(mo[:,:nocc], u)
        mo[:,nocc:] = numpy.dot(mo[:,nocc:], vh)
        mo_e, mo = mf.canonicalize(mo, mf.mo_occ)
        self.assertAlmostEqual(abs(mo_e-mf.mo_energy).max(), 0, 7)

    def test_get_occ(self):
        mfsym.irrep_nelec['B1'] = 1
        occ = mfsym.get_occ(mf.mo_energy, mf.mo_coeff+0j)
        self.assertAlmostEqual(lib.finger(occ), 0.49368251542877073, 9)
        mfsym.irrep_nelec['A2'] = 5
        occ = mfsym.get_occ(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(lib.finger(occ), -1.3108338866693456, 9)

    def test_analyze(self):
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 4.0049440587033116, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 2.05844441822, 5)
        (pop, chg), dip = mf.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 3.2031790129016922, 6)

    def test_det_ovlp(self):
        s, x = mf.det_ovlp(mf.mo_coeff, mf.mo_coeff, mf.mo_occ, mf.mo_occ)
        self.assertAlmostEqual(s, 1.000000000, 9)

    def test_spin_square(self):
        self.assertAlmostEqual(mf.spin_square()[0], 0, 9)

    def test_get_veff(self):
        pmol = gto.Mole()
        pmol.atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587'''
        pmol.basis = '6-31g'
        pmol.cart = True

        mf1 = scf.ghf.GHF(pmol)
        mf1.direct_scf = True
        mf1.max_memory = 0
        nao = pmol.nao_nr()*2
        numpy.random.seed(1)
        dm = numpy.random.random((2,3,nao,nao)) - .5 + 0.1j
        vhf2 = mf1.get_veff(pmol, dm[0,0], hermi=0)
        self.assertEqual(vhf2.ndim, 2)

        vhf3 = mf1.get_veff(pmol, dm[0], hermi=0)
        self.assertEqual(vhf3.ndim, 3)
        self.assertAlmostEqual(abs(vhf3[0]-vhf2).max(), 0, 12)

        vhf4 = mf1.get_veff(pmol, dm, hermi=0)
        self.assertEqual(vhf4.ndim, 4)
        self.assertAlmostEqual(lib.finger(vhf4),
                               -5.1441200982786057-5.5331447834480718j, 12)
        self.assertAlmostEqual(abs(vhf4[0]-vhf3).max(), 0, 12)

        vj = mf1.get_j(pmol, dm[0,0], hermi=0)
        vk = mf1.get_k(pmol, dm[0,0], hermi=0)
        self.assertAlmostEqual(abs(vj-vk-vhf2).max(), 0, 12)


if __name__ == "__main__":
    print("Full Tests for GHF")
    unittest.main()

