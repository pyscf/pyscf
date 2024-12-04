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

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import numint

def setUpModule():
    numint.SWITCH_SIZE = 0
    global mol, mf, h4, mf_h4, mol1, h2o, nao, he2, mf_he2
    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [('h', (0,0,i*3)) for i in range(12)]
        mol.basis = 'ccpvtz'
        mol.build()
        mf = dft.RKS(mol)
        mf.grids.atom_grid = {"H": (50, 110)}
        mf.prune = None
        mf.grids.build(with_non0tab=True)
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()

        h4 = gto.Mole()
        h4.verbose = 0
        h4.atom = 'H 0 0 0; H 0 0 9; H 0 9 0; H 0 9 9'
        h4.basis = 'ccpvtz'
        h4.build()
        mf_h4 = dft.RKS(h4)
        mf_h4.grids.atom_grid = {"H": (50, 110)}
        mf_h4.grids.build(with_non0tab=True)

        mol1 = gto.Mole()
        mol1.verbose = 0
        mol1.atom = [('h', (0,0,i*3)) for i in range(4)]
        mol1.basis = 'ccpvtz'
        mol1.build()

        h2o = gto.Mole()
        h2o.verbose = 5
        h2o.output = '/dev/null'
        h2o.atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ]

        h2o.basis = {"H": '6-31g', "O": '6-31g',}
        h2o.build()

        he2 = gto.Mole()
        he2.verbose = 0
        he2.atom = 'He 0.0 0.0 0.0; He 0.0 0.0 20.0'
        he2.basis = 'aug-pc-4'
        he2.build()
        mf_he2 = dft.RKS(he2)
        mf_he2.grids.level = 0
        mf_he2.grids.build(with_non0tab=True)

def tearDownModule():
    numint.SWITCH_SIZE = 800
    global mol, mf, h4, mf_h4, mol1, h2o
    h2o.stdout.close()
    del mol, mf, h4, mf_h4, mol1, h2o

def _not_sparse(mask):
    return False

def _sparse(mask):
    return True

def _fill_zero_blocks(mat, ao_loc, mask):
    nrow, nbas = mask.shape
    BLKSIZE = dft.numint.BLKSIZE
    for ib in range(nbas):
        for ig in range(nrow):
            if not mask[ig,ib]:
                mat[ig*BLKSIZE:(ig+1)*BLKSIZE,ao_loc[ib]:ao_loc[ib+1]] = 0
    return mat

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_dot_ao_dm(self):
        dm = mf_h4.get_init_guess(key='minao')
        ao_loc = h4.ao_loc_nr()
        ao = mf_h4._numint.eval_ao(h4, mf_h4.grids.coords).copy() + 0j
        nao = ao.shape[1]
        v1 = dft.numint._dot_ao_dm(h4, ao, dm, mf_h4.grids.non0tab, (0,h4.nbas), ao_loc)
        v2 = dft.numint._dot_ao_dm(h4, ao, dm, None, None, None)
        self.assertAlmostEqual(abs(v1-v2).max(), 0, 9)

        dm = mf_he2.get_init_guess(key='minao')
        ao_loc = he2.ao_loc_nr()
        ao = mf_he2._numint.eval_ao(he2, mf_he2.grids.coords).copy()
        v1 = dft.numint._dot_ao_dm(he2, ao, dm, mf_he2.grids.non0tab, (0,he2.nbas), ao_loc)
        v2 = dft.numint._dot_ao_dm(he2, ao, dm, None, None, None)
        self.assertAlmostEqual(abs(v1-v2).max(), 0, 9)

    def test_dot_ao_dm_high_cost(self):
        non0tab = mf._numint.make_mask(mol, mf.grids.coords)
        ao = dft.numint.eval_ao(mol, mf.grids.coords)
        numpy.random.seed(1)
        nao = ao.shape[1]
        ao_loc = mol.ao_loc_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        pair_mask = mol.get_overlap_cond() < -numpy.log(numint.CUTOFF)
        res0 = lib.dot(ao, dm)
        res1 = dft.numint._dot_ao_dm_sparse(
            ao, dm, dft.gen_grid.NBINS, non0tab, pair_mask, ao_loc=ao_loc)
        res0 = _fill_zero_blocks(res0, ao_loc, non0tab)
        res1 = _fill_zero_blocks(res1, ao_loc, non0tab)
        rho0 = dft.numint._contract_rho(res0, ao)
        rho1 = dft.numint._contract_rho_sparse(res1, ao, non0tab, ao_loc)
        self.assertAlmostEqual(abs(rho0 - rho1).max(), 0, 9)

    def test_dot_ao_ao(self):
        dm = mf_h4.get_init_guess(key='minao')
        ao_loc = h4.ao_loc_nr()
        ao = mf_h4._numint.eval_ao(h4, mf_h4.grids.coords).copy() + 0j
        nao = h4.nao_nr()
        v1 = dft.numint._dot_ao_ao(h4, ao, ao, mf_h4.grids.non0tab, (0,h4.nbas), ao_loc)
        v2 = dft.numint._dot_ao_ao(h4, ao, ao, None, None, None)
        self.assertAlmostEqual(abs(v1-v2).max(), 0, 9)

        dm = mf_he2.get_init_guess(key='minao')
        ao_loc = he2.ao_loc_nr()
        ao = mf_he2._numint.eval_ao(he2, mf_he2.grids.coords).copy()
        nao = he2.nao_nr()
        v1 = dft.numint._dot_ao_ao(he2, ao, ao, mf_he2.grids.non0tab, (0,he2.nbas), ao_loc)
        v2 = dft.numint._dot_ao_ao(he2, ao, ao, None, None, None)
        self.assertAlmostEqual(abs(v1-v2).max(), 0, 9)

    def test_scale_ao(self):
        ao = numpy.random.rand(8, 20).T
        wv = numpy.random.rand(20)
        self.assertAlmostEqual(abs(numpy.einsum('pi,p->pi', ao, wv) -
                                   numint._scale_ao(ao, wv)).max(), 0, 12)
        ao = numpy.random.rand(3, 8, 20).transpose(0,2,1)
        wv = numpy.random.rand(3, 20) + numpy.random.rand(3, 20) * 1j
        self.assertAlmostEqual(abs(numpy.einsum('npi,np->pi', ao, wv) -
                                   numint._scale_ao(ao, wv)).max(), 0, 12)
        ao = (numpy.random.rand(8, 20) + numpy.random.rand(8, 20) * 1j).T
        wv = numpy.random.rand(20)
        self.assertAlmostEqual(abs(numpy.einsum('pi,p->pi', ao, wv) -
                                   numint._scale_ao(ao, wv)).max(), 0, 12)
        ao = (numpy.random.rand(3, 8, 20) +
              numpy.random.rand(3, 8, 20) * 1j).transpose(0,2,1)
        wv = numpy.random.rand(3, 20) + numpy.random.rand(3, 20) * 1j
        self.assertAlmostEqual(abs(numpy.einsum('npi,np->pi', ao, wv) -
                                   numint._scale_ao(ao, wv)).max(), 0, 12)

    def test_dot_ao_ao_high_cost(self):
        non0tab = mf.grids.make_mask(mol, mf.grids.coords)
        ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
        nao = ao.shape[1]
        ao_loc = mol.ao_loc_nr()
        cutoff = mf.grids.cutoff * 1e2
        cutoff2 = numint.CUTOFF * 1e2
        nbins = numint.NBINS * 2 - int(numint.NBINS * numpy.log(cutoff)
                                       / numpy.log(mf.grids.cutoff))
        pair_mask = mol.get_overlap_cond() < -numpy.log(cutoff2)
        wv = numpy.ones(ao.shape[1])
        res0 = lib.dot(ao[0].T, ao[1])
        res1 = dft.numint._dot_ao_ao(mol, ao[0], ao[1], non0tab,
                                     shls_slice=(0,mol.nbas), ao_loc=ao_loc)
        res2 = dft.numint._dot_ao_ao_sparse(ao[0], ao[1], wv, nbins,
                                            non0tab, pair_mask, ao_loc,
                                            hermi=0)
        self.assertAlmostEqual(abs(res0 - res1).max(), 0, 9)
        self.assertAlmostEqual(abs(res0 - res2).max(), 0, 9)

    def test_eval_rho(self):
        numpy.random.seed(10)
        ngrids = 500
        coords = numpy.random.random((ngrids,3))*20
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ao = dft.numint.eval_ao(mol, coords, deriv=2)

        e, mo_coeff = numpy.linalg.eigh(dm)
        mo_occ = numpy.ones(nao)
        mo_occ[-2:] = -1
        dm = numpy.einsum('pi,i,qi->pq', mo_coeff, mo_occ, mo_coeff)

        rho0 = numpy.zeros((6,ngrids))
        rho0[0] = numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[0].conj())
        rho0[1] = numpy.einsum('pi,ij,pj->p', ao[1], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[1].conj())
        rho0[2] = numpy.einsum('pi,ij,pj->p', ao[2], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[2].conj())
        rho0[3] = numpy.einsum('pi,ij,pj->p', ao[3], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[3].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[4].conj()) + numpy.einsum('pi,ij,pj->p', ao[4], dm, ao[0].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[7].conj()) + numpy.einsum('pi,ij,pj->p', ao[7], dm, ao[0].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[9].conj()) + numpy.einsum('pi,ij,pj->p', ao[9], dm, ao[0].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[1], dm, ao[1].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[2], dm, ao[2].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[3], dm, ao[3].conj())
        rho0[4]+= rho0[5]*2
        rho0[5] *= .5

        ni = dft.numint.NumInt()
        rho1 = ni.eval_rho (mol, ao, dm, xctype='MGGA')
        rho2 = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, xctype='MGGA')
        self.assertAlmostEqual(abs(rho0 - rho1).max(), 0, 9)
        self.assertAlmostEqual(abs(rho0 - rho2).max(), 0, 9)

    def test_eval_mat(self):
        numpy.random.seed(10)
        ngrids = 500
        coords = numpy.random.random((ngrids,3))*20
        rho = numpy.random.random((6,ngrids))
        vxc = numpy.random.random((4,ngrids))
        weight = numpy.random.random(ngrids)
        ao = dft.numint.eval_ao(mol, coords, deriv=2)

        mat0 = numpy.einsum('pi,p,pj->ij', ao[0].conj(), weight*vxc[0], ao[0])
        mat1 = dft.numint.eval_mat(mol, ao[0], weight, rho, vxc[0], xctype='LDA')
        self.assertAlmostEqual(abs(mat0 - mat1).max(), 0, 9)
        # UKS
        mat2 = dft.numint.eval_mat(mol, ao[0], weight, rho, [vxc[0]]*2, xctype='LDA', spin=1)
        self.assertAlmostEqual(abs(mat0 - mat2).max(), 0, 9)

        vrho, vsigma = vxc[:2]
        wv = weight * vsigma * 2
        mat0  = numpy.einsum('pi,p,pj->ij', ao[0].conj(), weight*vrho, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[1]*wv, ao[1]) + numpy.einsum('pi,p,pj->ij', ao[1].conj(), rho[1]*wv, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[2]*wv, ao[2]) + numpy.einsum('pi,p,pj->ij', ao[2].conj(), rho[2]*wv, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[3]*wv, ao[3]) + numpy.einsum('pi,p,pj->ij', ao[3].conj(), rho[3]*wv, ao[0])
        mat1 = dft.numint.eval_mat(mol, ao, weight, rho, vxc[:4], xctype='GGA')
        self.assertAlmostEqual(abs(mat0 - mat1).max(), 0, 9)
        # UKS
        ngrids = weight.size
        vxc_1 = [vxc[0], numpy.vstack((vxc[1], numpy.zeros(ngrids))).T]
        mat2 = dft.numint.eval_mat(mol, ao, weight, [rho[:4]]*2, vxc_1, xctype='GGA', spin=1)
        self.assertAlmostEqual(abs(mat0 - mat2).max(), 0, 9)

        vrho, vsigma, _, vtau = vxc
        vxc = (vrho, vsigma, None, vtau)
        wv = weight * vtau * .5
        mat2  = numpy.einsum('pi,p,pj->ij', ao[1].conj(), wv, ao[1])
        mat2 += numpy.einsum('pi,p,pj->ij', ao[2].conj(), wv, ao[2])
        mat2 += numpy.einsum('pi,p,pj->ij', ao[3].conj(), wv, ao[3])
        mat0 += mat2
        mat1 = dft.numint.eval_mat(mol, ao, weight, rho, vxc, xctype='MGGA')
        self.assertAlmostEqual(abs(mat0 - mat1).max(), 0, 9)
        # UKS
        ngrids = weight.size
        vxc_1 = [vxc[0],
                 numpy.vstack((vxc[1], numpy.zeros(ngrids))).T,
                 None,
                 numpy.vstack((vxc[3], numpy.zeros(ngrids))).T]
        mat2 = dft.numint.eval_mat(mol, ao, weight, [rho]*2, vxc_1, xctype='MGGA', spin=1)
        self.assertAlmostEqual(abs(mat0 - mat2).max(), 0, 9)

    def test_rks_vxc(self):
        numpy.random.seed(10)
        nao = mol.nao_nr()
        dms = numpy.random.random((2,nao,nao))
        v = mf._numint.nr_vxc(mol, mf.grids, 'B88,', dms, spin=0, hermi=0)[2]
        self.assertAlmostEqual(lib.fp(v), -0.70124686853021512, 8)

        v = mf._numint.nr_vxc(mol, mf.grids, 'HF', dms, spin=0, hermi=0)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)
        v = mf._numint.nr_vxc(mol, mf.grids, '', dms, spin=0, hermi=0)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)

    def test_uks_vxc(self):
        numpy.random.seed(10)
        nao = h2o.nao_nr()
        dms = numpy.random.random((2,nao,nao))
        grids = dft.gen_grid.Grids(h2o)
        v = mf._numint.nr_vxc(h2o, grids, 'B88,', dms, spin=1)[2]
        self.assertAlmostEqual(lib.fp(v), -7.7508525240447348, 8)

        v = mf._numint.nr_vxc(h2o, grids, 'HF', dms, spin=1)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)
        v = mf._numint.nr_vxc(h2o, grids, '', dms, spin=1)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)

    def test_uks_vxc_high_cost(self):
        numpy.random.seed(10)
        nao = mol.nao_nr()
        dms = numpy.random.random((2,nao,nao))
        v = mf._numint.nr_vxc(mol, mf.grids, 'B88,', dms, spin=1)[2]
        self.assertAlmostEqual(lib.fp(v), -0.73803886056633594, 8)

        v = mf._numint.nr_vxc(mol, mf.grids, 'HF', dms, spin=1)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)
        v = mf._numint.nr_vxc(mol, mf.grids, '', dms, spin=1)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)

    def test_rks_fxc(self):
        numpy.random.seed(10)
        nao = mol1.nao_nr()
        dm0 = numpy.random.random((nao,nao))
        _, mo_coeff = numpy.linalg.eigh(dm0)
        mo_occ = numpy.ones(nao)
        mo_occ[-2:] = -1
        dm0 = numpy.einsum('pi,i,qi->pq', mo_coeff, mo_occ, mo_coeff)
        dms = numpy.random.random((2,nao,nao))
        ni = dft.numint.NumInt()
        grids = dft.Grids(mol1)
        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_fxc(mol1, grids, 'B88,', dm0, dms, spin=0, hermi=0)
        self.assertAlmostEqual(lib.fp(v), -7.571122737701957, 8)

        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = ni.nr_fxc(mol1, grids, 'B88,', dm0, dms, spin=0, hermi=0)
        self.assertAlmostEqual(lib.fp(v), -7.571122737701957, 8)

        # test cache_kernel
        rvf = ni.cache_xc_kernel(mol1, grids, 'B88,', mo_coeff, mo_occ, spin=0)
        v1 = dft.numint.nr_fxc(mol1, grids, 'B88,', dm0, dms, spin=0, hermi=0,
                               rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)

        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_fxc(mol1, grids, 'LDA,', dm0, dms[0], spin=0, hermi=0)
        self.assertAlmostEqual(lib.fp(v), -3.0008266036125315, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = ni.nr_fxc(mol1, grids, 'LDA,', dm0, dms[0], spin=0, hermi=0)
        self.assertAlmostEqual(lib.fp(v), -3.0008266036125315, 8)

        # test cache_kernel
        rvf = ni.cache_xc_kernel(mol1, grids, 'LDA,', mo_coeff, mo_occ, spin=0)
        v1 = dft.numint.nr_fxc(mol1, grids, 'LDA,', dm0, dms[0], spin=0, hermi=0,
                               rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)

        v = ni.nr_fxc(mol1, grids, 'HF', dm0, dms, spin=0, hermi=0)
        self.assertAlmostEqual(abs(v).max(), 0, 9)
        v = ni.nr_fxc(mol1, grids, '', dm0, dms, spin=0, hermi=0)
        self.assertAlmostEqual(abs(v).max(), 0, 9)

        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = dft.numint.nr_fxc(mol1, grids, 'm06l,', dm0, dms)
        self.assertAlmostEqual(lib.fp(v), -11.138947264441164, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = dft.numint.nr_fxc(mol1, grids, 'm06l,', dm0, dms)
        self.assertAlmostEqual(lib.fp(v), -11.138947264441164, 8)

        rvf = ni.cache_xc_kernel(mol1, grids, 'm06l,', mo_coeff, mo_occ, spin=0)
        v1 = dft.numint.nr_fxc(mol1, grids, 'm06l,', dm0, dms,
                               rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)

    def test_rks_fxc_st(self):
        numpy.random.seed(10)
        nao = mol1.nao_nr()
        dm0 = numpy.random.random((nao,nao))
        _, mo_coeff = numpy.linalg.eigh(dm0)
        mo_occ = numpy.ones(nao)
        mo_occ[-2:] = -1
        dm0 = numpy.einsum('pi,i,qi->pq', mo_coeff, mo_occ, mo_coeff)
        dms = numpy.random.random((2,nao,nao))
        ni = dft.numint.NumInt()
        grids = dft.Grids(mol1)
        rvf = ni.cache_xc_kernel(mol1, grids, 'B88,', [mo_coeff,mo_coeff],
                                 [mo_occ*.5]*2, spin=1)
        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_rks_fxc_st(mol1, grids, 'B88,', dm0, dms, singlet=True)
        self.assertAlmostEqual(lib.fp(v), -7.571122737701957*2, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = ni.nr_rks_fxc_st(mol1, grids, 'B88,', dm0, dms, singlet=True)
        self.assertAlmostEqual(lib.fp(v), -7.571122737701957*2, 8)
        v1 = ni.nr_rks_fxc_st(mol1, grids, 'B88,', dm0, dms, singlet=True,
                              rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v1), -7.571122737701957*2, 8)

        v = ni.nr_rks_fxc_st(mol1, grids, 'B88,', dm0, dms, singlet=False)
        v1 = ni.nr_rks_fxc_st(mol1, grids, 'B88,', dm0, dms, singlet=False,
                              rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v1), -7.571122737701957*2, 8)

        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            rvf = ni.cache_xc_kernel(mol1, grids, 'LDA,', [mo_coeff,mo_coeff],
                                     [mo_occ*.5]*2, spin=1)
            v = ni.nr_rks_fxc_st(mol1, grids, 'LDA,', dm0, dms[0], singlet=True)
            v1 = ni.nr_rks_fxc_st(mol1, grids, 'LDA,', dm0, dms[0], singlet=True,
                                  rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v1), -3.0008266036125315*2, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v1 = ni.nr_rks_fxc_st(mol1, grids, 'LDA,', dm0, dms[0], singlet=True,
                                  rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(lib.fp(v1), -3.0008266036125315*2, 8)

        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_rks_fxc_st(mol1, grids, 'LDA,', dm0, dms[0], singlet=False)
        self.assertAlmostEqual(lib.fp(v), -3.0008266036125315*2, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = ni.nr_rks_fxc_st(mol1, grids, 'LDA,', dm0, dms[0], singlet=False)
        self.assertAlmostEqual(lib.fp(v), -3.0008266036125315*2, 8)
        v1 = ni.nr_rks_fxc_st(mol1, grids, 'LDA,', dm0, dms[0], singlet=False,
                              rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)

        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_rks_fxc_st(mol1, grids, 'm06l,', dm0, dms, singlet=True)
            rvf = ni.cache_xc_kernel(mol1, grids, 'm06l,', [mo_coeff,mo_coeff],
                                     [mo_occ*.5]*2, spin=1)
            v1 = ni.nr_rks_fxc_st(mol1, grids, 'm06l,', dm0, dms, singlet=True,
                                  rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v1), -11.138947264441164*2, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v1 = ni.nr_rks_fxc_st(mol1, grids, 'm06l,', dm0, dms, singlet=True,
                                  rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(lib.fp(v1), -11.138947264441164*2, 8)

        v = ni.nr_rks_fxc_st(mol1, grids, 'm06l,', dm0, dms, singlet=False)
        rvf = ni.cache_xc_kernel(mol1, grids, 'm06l,', [mo_coeff,mo_coeff],
                                 [mo_occ*.5]*2, spin=1)
        v1 = ni.nr_rks_fxc_st(mol1, grids, 'm06l,', dm0, dms, singlet=False,
                              rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v1), -11.138947264441164*2, 8)

    def test_uks_fxc(self):
        numpy.random.seed(10)
        nao = mol1.nao_nr()
        dm0 = numpy.random.random((2,nao,nao))
        e, mo_coeff = numpy.linalg.eigh(dm0)
        mo_occ = numpy.ones((2,nao))
        mo_occ[:,-2:] = -1
        dm0 = numpy.einsum('xpi,xi,xqi->xpq', mo_coeff, mo_occ, mo_coeff)
        dms = numpy.random.random((2,nao,nao))
        ni = dft.numint.NumInt()
        grids = dft.Grids(mol1)
        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_fxc(mol1, grids, 'B88,', dm0, dms, spin=1)
        self.assertAlmostEqual(lib.fp(v), -10.316735149305348, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = ni.nr_fxc(mol1, grids, 'B88,', dm0, dms, spin=1)
        self.assertAlmostEqual(lib.fp(v), -10.316735149305348, 8)
        # test cache_kernel
        rvf = ni.cache_xc_kernel(mol1, grids, 'B88,', mo_coeff, mo_occ, spin=1)
        v1 = dft.numint.nr_fxc(mol1, grids, 'B88,', dm0, dms, hermi=0, spin=1,
                               rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)

        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_fxc(mol1, grids, 'LDA,', dm0, dms[0], spin=1)
        self.assertAlmostEqual(lib.fp(v), -5.646254460347009, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = ni.nr_fxc(mol1, grids, 'LDA,', dm0, dms[0], spin=1)
        self.assertAlmostEqual(lib.fp(v), -5.646254460347009, 8)
        # test cache_kernel
        rvf = ni.cache_xc_kernel(mol1, grids, 'LDA,', mo_coeff, mo_occ, spin=1)
        v1 = dft.numint.nr_fxc(mol1, grids, 'LDA,', dm0, dms[0], hermi=0, spin=1,
                               rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)

        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            v = ni.nr_fxc(mol1, grids, 'm06l', dm0, dms[0], spin=1)
        self.assertAlmostEqual(lib.fp(v), -7.004513546383883, 8)
        with lib.temporary_env(numint, _sparse_enough=_sparse):
            v = ni.nr_fxc(mol1, grids, 'm06l', dm0, dms[0], spin=1)
        self.assertAlmostEqual(lib.fp(v), -7.004513546383883, 8)
        # test cache_kernel
        rvf = ni.cache_xc_kernel(mol1, grids, 'm06l', mo_coeff, mo_occ, spin=1)
        v1 = dft.numint.nr_fxc(mol1, grids, 'm06l', dm0, dms[0], hermi=0, spin=1,
                               rho0=rvf[0], vxc=rvf[1], fxc=rvf[2])
        self.assertAlmostEqual(abs(v-v1).max(), 0, 8)

    def test_vv10nlc(self):
        numpy.random.seed(10)
        rho = numpy.random.random((4,20))
        coords = (numpy.random.random((20,3))-.5)*3
        vvrho = numpy.random.random((4,60))
        vvweight = numpy.random.random(60)
        vvcoords = (numpy.random.random((60,3))-.5)*3
        nlc_pars = .8, .3
        v = dft.numint._vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars)
        self.assertAlmostEqual(lib.fp(v[0]), 0.15894647203764295, 9)
        self.assertAlmostEqual(lib.fp(v[1]), 0.20500922537924576, 9)

    def test_nr_uks_vxc_vv10(self):
        method = dft.UKS(h2o)
        dm = method.get_init_guess()
        grids = dft.gen_grid.Grids(h2o)
        grids.atom_grid = {'H': (20, 50), 'O': (20,50)}
        ni = dft.numint.NumInt()
        n, e, v = dft.numint.nr_nlc_vxc(ni, h2o, grids, 'wB97M_V', dm[0]*2, hermi=0)
        self.assertAlmostEqual(n, 9.987377839393485, 8)
        self.assertAlmostEqual(e, 0.04237199619089385, 8)
        self.assertAlmostEqual(lib.fp([v, v]), 0.02293399033256055, 8)

    def test_uks_gga_wv1(self):
        numpy.random.seed(1)
        rho0 = [numpy.random.random((4,5))]*2
        rho1 = [numpy.random.random((4,5))]*2
        weight = numpy.ones(5)

        exc, vxc, fxc, kxc = dft.libxc.eval_xc('b88,', rho0, 1, 0, 3)
        wva, wvb = dft.numint._uks_gga_wv1(rho0, rho1, vxc, fxc, weight)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('b88,', rho0[0]+rho0[1], 0, 0, 3)
        wv = dft.numint._rks_gga_wv1(rho0[0]+rho0[1], rho1[0]+rho1[1], vxc, fxc, weight)
        self.assertAlmostEqual(abs(1 - wv/wva).max(), 0, 12)
        self.assertAlmostEqual(abs(1 - wv/wvb).max(), 0, 12)

    def test_uks_gga_wv2(self):
        numpy.random.seed(1)
        rho0 = [numpy.random.random((4,5))]*2
        rho1 = [numpy.random.random((4,5))]*2
        weight = numpy.ones(5)

        exc, vxc, fxc, kxc = dft.libxc.eval_xc('b88,', rho0, 1, 0, 3)
        wva, wvb = dft.numint._uks_gga_wv2(rho0, rho1, fxc, kxc, weight)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('b88,', rho0[0]+rho0[1], 0, 0, 3)
        wv = dft.numint._rks_gga_wv2(rho0[0]+rho0[1], rho1[0]+rho1[1], fxc, kxc, weight)
        self.assertAlmostEqual(abs(1 - wv/wva).max(), 0, 12)
        self.assertAlmostEqual(abs(1 - wv/wvb).max(), 0, 12)

    def test_complex_dm(self):
        mf = dft.RKS(h2o)
        mf.xc = 'b3lyp5'
        nao = h2o.nao
        numpy.random.seed(1)
        dm = (numpy.random.random((nao,nao)) +
              numpy.random.random((nao,nao))*1j)
        dm = dm + dm.conj().T
        v = mf.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(v), 30.543789621782576-0.23207622637751305j, 9)

    def test_rsh_omega(self):
        rho0 = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        ni = dft.numint.NumInt()
        ni.omega = 0.4
        omega = 0.2
        exc, vxc, fxc, kxc = ni.eval_xc('ITYH,', rho0, 0, 0, 1, omega)
        self.assertAlmostEqual(float(exc), -0.6359945579326314, 7)
        self.assertAlmostEqual(float(vxc[0]), -0.8712041561251518, 7)
        self.assertAlmostEqual(float(vxc[1]), -0.003911167644579979, 7)

        exc, vxc, fxc, kxc = ni.eval_xc('ITYH,', rho0, 0, 0, 1)
        self.assertAlmostEqual(float(exc), -0.5406095865415561, 7)
        self.assertAlmostEqual(float(vxc[0]), -0.772123720263471, 7)
        self.assertAlmostEqual(float(vxc[1]), -0.00301639097170439, 7)

    def test_cache_xc_kernel(self):
        mf = dft.RKS(h2o)
        mf.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        mf.run()

        mf.xc = 'WB97XD'
        mf.omega = 0.9
        with lib.temporary_env(numint, _sparse_enough=_not_sparse):
            rho, vxc, fxc = mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ)
        self.assertAlmostEqual(rho[0].dot(mf.grids.weights), 10, 4)
        self.assertAlmostEqual(numpy.einsum('g,g,ig->', mf.grids.weights, rho[0], rho), 81.04275692925363, 5)
        self.assertAlmostEqual(numpy.einsum('g,xg,xyg->', mf.grids.weights, rho, fxc), -6.194969637088992, 5)

        with lib.temporary_env(numint, _sparse_enough=_sparse):
            rho, vxc, fxc = mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ)
        self.assertAlmostEqual(rho[0].dot(mf.grids.weights), 10, 4)
        self.assertAlmostEqual(numpy.einsum('g,g,ig->', mf.grids.weights, rho[0], rho), 81.04275692925363, 5)
        self.assertAlmostEqual(numpy.einsum('g,xg,xyg->', mf.grids.weights, rho, fxc), -6.194969637088992, 5)

        if hasattr(dft, 'xcfun'):
            mf.xc = 'camb3lyp'
            mf.omega = 0.9
            rho1, vxc1, fxc1 = mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ)

            mf.xc = 'camb3lyp'
            mf._numint.libxc = dft.xcfun
            mf.omega = 0.9
            rho2, vxc2, fxc2 = mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ)

            self.assertAlmostEqual(abs(fxc1*rho1[0] - fxc2*rho2[0]).max(), 0, 4)

if __name__ == "__main__":
    print("Test numint")
    unittest.main()
