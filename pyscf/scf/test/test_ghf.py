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

import unittest
import tempfile
import numpy
import scipy.linalg
from functools import reduce

from pyscf import gto
from pyscf import lib
from pyscf import scf
scf.hf.MUTE_CHKFILE = True
from pyscf import ao2mo

def setUpModule():
    global mol, mf, molsym, mfsym, mol1, mf_r, mf_u
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
    mf.chkfile = tempfile.NamedTemporaryFile().name
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
    mfsym = scf.GHF(molsym).run(conv_tol=1e-10)

    mol1 = gto.M(atom=mol.atom, basis='631g', spin=2, verbose=0)
    mf_r = scf.RHF(mol1).run(conv_tol=1e-10, chkfile=tempfile.NamedTemporaryFile().name)
    mf_u = scf.RHF(mol1).run(conv_tol=1e-10, chkfile=tempfile.NamedTemporaryFile().name)

def tearDownModule():
    global mol, mf, molsym, mfsym, mol1, mf_r, mf_u
    mol.stdout.close()
    molsym.stdout.close()
    del mol, mf, molsym, mfsym, mol1, mf_r, mf_u

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

class KnownValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = mf.get_init_guess(mol, key='minao')
        self.assertEqual(dm.shape, (48,48))
        self.assertAlmostEqual(lib.fp(dm[:24,:24])*2, 2.5912875957299684, 9)
        self.assertAlmostEqual(lib.fp(dm[24:,24:])*2, 2.5912875957299684, 9)

    def test_init_guess_atom(self):
        dm = mf.get_init_guess(mol, key='atom')
        self.assertEqual(dm.shape, (48,48))
        self.assertAlmostEqual(lib.fp(dm[:24,:24])*2, 2.7821827416174094, 7)
        self.assertAlmostEqual(lib.fp(dm[24:,24:])*2, 2.7821827416174094, 7)

    def test_init_guess_chk(self):
        dm = mol.GHF(chkfile=tempfile.NamedTemporaryFile().name).get_init_guess(mol, key='chkfile')
        self.assertEqual(dm.shape, (48,48))
        self.assertAlmostEqual(lib.fp(dm), 1.8117584283411752, 5)

        dm = mf.get_init_guess(mol, key='chkfile')
        self.assertEqual(dm.shape, (48,48))
        self.assertAlmostEqual(lib.fp(dm), 1.3594274771226789, 5)

        dm = scf.ghf.init_guess_by_chkfile(mol1, mf_r.chkfile, project=True)
        self.assertEqual(dm.shape, (26,26))
        self.assertAlmostEqual(lib.fp(dm), -3.742519160521582, 5)

        dm = scf.ghf.init_guess_by_chkfile(mol1, mf_u.chkfile)
        self.assertEqual(dm.shape, (26,26))
        self.assertAlmostEqual(lib.fp(dm), -3.742519160521582, 5)

    def test_init_guess_huckel(self):
        dm = scf.GHF(mol).get_init_guess(mol, key='huckel')
        self.assertAlmostEqual(lib.fp(dm), 1.0574099243527206, 7)

    def test_init_guess_mod_huckel(self):
        dm = scf.GHF(mol).get_init_guess(mol, key='mod_huckel')
        self.assertAlmostEqual(lib.fp(dm), 1.1466144161440015, 7)

    def test_init_guess_sap(self):
        dm = mf.get_init_guess(mol, key='sap')
        self.assertEqual(dm.shape, (48,48))
        self.assertAlmostEqual(lib.fp(dm[:24,:24])*2, 4.2267871571567195, 7)
        self.assertAlmostEqual(lib.fp(dm[24:,24:])*2, 4.2267871571567195, 7)

    def test_ghf_complex(self):
        mf1 = scf.GHF(mol)
        dm = mf1.init_guess_by_1e(mol) + 0j
        nao = dm.shape[0] // 2
        numpy.random.seed(12)
        dm[:nao,nao:] = numpy.random.random((nao,nao)) * .1j
        dm[nao:,:nao] = dm[:nao,nao:].T.conj()
        mf1.kernel(dm)
        self.assertAlmostEqual(mf1.e_tot, mf.e_tot, 9)
        self.assertAlmostEqual(mf1.e_tot, -76.02676567312, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()*2
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao)) + numpy.random.random((nao,nao)) * 1j
        d2 = numpy.random.random((nao,nao)) + numpy.random.random((nao,nao)) * 1j
        d = numpy.array((d1+d1.conj().T, d2+d2.conj().T))
        eri = numpy.zeros((nao, nao, nao, nao))
        n = nao // 2
        eri[:n,:n,:n,:n] = \
        eri[n:,n:,n:,n:] = \
        eri[:n,:n,n:,n:] = \
        eri[n:,n:,:n,:n] = ao2mo.restore(1, mf._eri, n)
        vj = numpy.einsum('ijkl,xji->xkl', eri, d)
        vk = numpy.einsum('ijkl,xjk->xil', eri, d)
        ref = vj - vk
        v = mf.get_veff(mol, d)
        self.assertAlmostEqual(abs(ref - v).max(), 0, 12)
        self.assertAlmostEqual(numpy.linalg.norm(v), 560.3785699368684, 9)

        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = numpy.array((d1+d1.conj().T, d2+d2.conj().T))
        vj = numpy.einsum('ijkl,xji->xkl', eri, d)
        vk = numpy.einsum('ijkl,xjk->xil', eri, d)
        ref = vj - vk
        v = mf.get_veff(mol, d)
        self.assertAlmostEqual(abs(ref - v).max(), 0, 12)
        self.assertAlmostEqual(numpy.linalg.norm(v), 607.5489445471493, 9)

    def test_get_jk(self):
        nao = mol.nao_nr()*2
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        vj, vk = mf.get_jk(mol, d)
        self.assertEqual(vj.shape, (2,nao,nao))
        self.assertEqual(vk.shape, (2,nao,nao))
        self.assertAlmostEqual(lib.fp(vj), 246.24944977538354, 9)
        self.assertAlmostEqual(lib.fp(vk), 53.22954677121744, 9)

        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao)) + 1j*numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao)) + 1j*numpy.random.random((nao,nao))
        d = (d1+d1.T.conj(), d2+d2.T.conj())
        vj, vk = mf.get_jk(mol, d)
        self.assertEqual(vj.shape, (2,nao,nao))
        self.assertEqual(vk.shape, (2,nao,nao))
        self.assertAlmostEqual(lib.fp(vj), 254.68614111766146+0j, 9)
        self.assertAlmostEqual(lib.fp(vk), 62.08832587927003-8.640597547171135j, 9)

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

        # Test the call to spin_square() in ghf.analyze()
        mf.analyze()

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
        self.assertAlmostEqual(abs(mo_e-mf.mo_energy).max(), 0, 6)

        e, c = mfsym.canonicalize(mfsym.mo_coeff, mfsym.mo_occ)
        self.assertAlmostEqual(abs(e - mfsym.mo_energy).max(), 0, 6)

    def test_get_occ(self):
        mf1 = mfsym.copy()
        mf1.irrep_nelec = {}
        mf1.irrep_nelec['B1'] = 1
        occ = mf1.get_occ(mf.mo_energy, mf.mo_coeff+0j)
        self.assertAlmostEqual(lib.fp(occ), 0.49368251542877073, 9)
        mf1.irrep_nelec['A2'] = 5
        occ = mf1.get_occ(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(lib.fp(occ), -1.3108338866693456, 9)

        order = numpy.argsort(numpy.random.random(mfsym.mo_occ.size))
        mo_e = mfsym.mo_energy[order]
        mo = numpy.array(mfsym.mo_coeff[:,order])
        self.assertTrue(numpy.allclose(mfsym.get_occ(mo_e, mo),
                                       mfsym.mo_occ[order]))

    def test_get_occ_extreme_case(self):
        mol = gto.M(atom='He', verbose=7, output='/dev/null')
        mf = scf.GHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, -2.8077839575399737, 12)

        mol.charge = 2
        mf = scf.GHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, 0, 12)
        mol.stdout.close()

    def test_analyze(self):
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 4.0049440587033116, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 2.05844441822, 5)
        (pop, chg), dip = mf.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 3.2031790129016922, 6)

        (pop, chg), dip = mfsym.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 4.0049440587033116, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 2.05844441822, 5)
        (pop, chg), dip = mfsym.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 3.2031790129016922, 6)

    def test_get_grad(self):
        g = mf.get_grad(mf.mo_coeff, mf.mo_occ)
        self.assertAlmostEqual(abs(g).max(), 0, 6)

    def test_det_ovlp(self):
        s, x = mf.det_ovlp(mf.mo_coeff, mf.mo_coeff, mf.mo_occ, mf.mo_occ)
        self.assertAlmostEqual(s, 1.000000000, 9)

    def test_spin_square1(self):
        self.assertAlmostEqual(mf.spin_square()[0], 0, 9)

    def test_get_veff1(self):
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

        vhf4 = mf1.get_veff(pmol, dm, hermi=1)
        self.assertEqual(vhf4.ndim, 4)
        self.assertAlmostEqual(lib.fp(vhf4), 17.264430281812047-5.533144783448073j, 12)
        vhf4 = mf1.get_veff(pmol, dm, hermi=0)
        self.assertEqual(vhf4.ndim, 4)
        self.assertAlmostEqual(lib.fp(vhf4), 4.95789766380037-5.2816995816903045j, 12)
        self.assertAlmostEqual(abs(vhf4[0]-vhf3).max(), 0, 12)

        vj = mf1.get_j(pmol, dm[0,0], hermi=0)
        vk = mf1.get_k(pmol, dm[0,0], hermi=0)
        self.assertAlmostEqual(abs(vj-vk-vhf2).max(), 0, 12)

    def test_guess_orbspin(self):
        self.assertTrue(numpy.all(scf.ghf.guess_orbspin(mf.mo_coeff) == -1))
        mf1 = scf.addons.convert_to_ghf(mf_r)
        self.assertEqual(list(scf.ghf.guess_orbspin(mf1.mo_coeff)),
                         [0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
        mf1 = scf.addons.convert_to_ghf(mf_u)
        self.assertEqual(list(scf.ghf.guess_orbspin(mf1.mo_coeff)),
                         [0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])

    def test_get_irrep_nelec(self):
        fock = mfsym.get_fock()
        s1e = mfsym.get_ovlp()
        e, c = mfsym.eig(fock, s1e)
        mo_occ = mfsym.get_occ(e, c)
        irrep_nelec = mfsym.get_irrep_nelec(molsym, c, mo_occ)
        self.assertEqual(irrep_nelec['A1'], 6)
        self.assertEqual(irrep_nelec['A2'], 0)
        self.assertEqual(irrep_nelec['B1'], 2)
        self.assertEqual(irrep_nelec['B2'], 2)

        mf1 = mfsym.copy()
        mf1.irrep_nelec = irrep_nelec
        mf1.irrep_nelec['A1'] = 2
        mf1.irrep_nelec['A2'] = 2
        self.assertRaises(ValueError, mf1.build)
        mf1.irrep_nelec.pop('A2')
        mf1.irrep_nelec['A1'] = 8
        self.assertRaises(ValueError, mf1.build)
        mf1.irrep_nelec['A1'] = (4,4)
        mf1.irrep_nelec['A1g'] = 4
        self.assertRaises(ValueError, mf1.build)

    def test_scanner(self):
        mf_scanner = mf.as_scanner()
        e = mf_scanner(molsym)
        self.assertAlmostEqual(e, mfsym.e_tot, 9)

    def test_ecp_soc(self):
        mol = gto.M(
            verbose = 0,
            atom ='I 0 0 0; H 1.599 0 0',
            basis = {'H':'cc-pvdz', 'I':'stuttgart_dz'},
            ecp = {'I': '''
I nelec 46
I ul
2      1.000000        0.000000
I S
2      3.380230        83.107547
2      1.973454        5.099343
I P
2      2.925323        27.299020       -54.598040
2      3.073557        55.607847       55.607847
2      1.903188        0.778322        -1.556643
2      1.119689        1.751128        1.751128
I D
2      1.999036        8.234552        -8.234552
2      1.967767        12.488097       8.325398
2      0.998982        2.177334        -2.177334
2      0.972272        3.167401        2.111601
I F
2      2.928812        -11.777154      7.851436
2      2.904069        -15.525522      -7.762761
2      0.287352        -0.148550       0.099033
2      0.489380        -0.273682       -0.136841'''},)
        mf = mol.GHF(with_soc=True).run()
        # from DIRAC19. See issue #744
        ref = -11.76034990661
        self.assertAlmostEqual(mf.e_tot, ref, 9)


if __name__ == "__main__":
    print("Full Tests for GHF")
    unittest.main()
