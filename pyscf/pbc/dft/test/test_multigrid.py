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

from pyscf import lib
from pyscf.pbc import gto, scf, dft, df
from pyscf.pbc import tools
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import multigrid
multigrid.R_RATIO_SUBLOOP = 0.6

def setUpModule():
    global cell_orth, cell_nonorth, cell_he, mydf
    global kpts, nao, dm, dm1, vj_uks_orth, he_nao, dm_he
    numpy.random.seed(2)
    cell_orth = gto.M(
        verbose = 7,
        output = '/dev/null',
        a = numpy.eye(3)*3.5668,
        atom = '''C     0.      0.      0.
                  C     1.8     1.8     1.8   ''',
        basis = 'gth-dzv',
        pseudo = 'gth-pade',
        precision = 1e-9,
        mesh = [48] * 3,
    )
    cell_nonorth = gto.M(
        a = numpy.eye(3)*3.5668 + numpy.random.random((3,3)),
        atom = '''C     0.      0.      0.
                  C     0.8917  0.8917  0.8917''',
        basis = 'gth-dzv',
        pseudo = 'gth-pade',
        precision = 1e-9,
        mesh = [44,43,42],
    )

    cell_he = gto.M(atom='He 0 0 0',
                    basis=[[0, ( 1, 1, .1), (.5, .1, 1)],
                           [1, (.8, 1)]],
                    unit='B',
                    precision = 1e-9,
                    mesh=[18]*3,
                    a=numpy.eye(3)*5)

    kptsa = numpy.random.random((2,3))
    kpts = kptsa.copy()
    kpts[1] = -kpts[0]
    nao = cell_orth.nao_nr()
    dm = numpy.random.random((len(kpts),nao,nao)) * .2
    dm1 = dm + numpy.eye(nao)
    dm = dm1 + dm1.transpose(0,2,1)
    mydf = df.FFTDF(cell_orth)
    vj_uks_orth = mydf.get_jk(dm1, with_k=False)[0]

    he_nao = cell_he.nao
    dm_he = numpy.random.random((len(kpts), he_nao, he_nao))
    dm_he = dm_he + dm_he.transpose(0,2,1)
    dm_he = dm_he * .2 + numpy.eye(he_nao)

def tearDownModule():
    global cell_orth, cell_nonorth, cell_he, mydf
    del cell_orth, cell_nonorth, cell_he, mydf

class KnownValues(unittest.TestCase):
    def test_orth_get_pp(self):
        ref = df.FFTDF(cell_orth).get_pp()
        out = multigrid.MultiGridFFTDF(cell_orth).get_pp()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

        # test small memory
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        mydf.max_memory = 10
        out = mydf.get_pp(max_memory=2)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_nonorth_get_pp(self):
        ref = df.FFTDF(cell_nonorth).get_pp()
        out = multigrid.MultiGridFFTDF(cell_nonorth).get_pp()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

        # test small memory
        mydf = multigrid.MultiGridFFTDF(cell_nonorth)
        mydf.max_memory = 10
        out = mydf.get_pp(max_memory=2)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_orth_get_nuc_kpts(self):
        ref = df.FFTDF(cell_orth).get_nuc(kpts)
        out = multigrid.MultiGridFFTDF(cell_orth).get_nuc(kpts)
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_orth_get_j_kpts(self):
        ref = df.FFTDF(cell_orth).get_jk(dm, kpts=kpts, with_k=False)[0]
        out = multigrid.MultiGridFFTDF(cell_orth).get_jk(dm, kpts=kpts)[0]
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

#        mydf = multigrid.MultiGridFFTDF(cell_orth)
#        self.assertRaises(ValueError, mydf.get_jk, dm1, hermi=0, kpts=kpts, with_k=False)

    def test_nonorth_get_j_kpts(self):
        ref = df.FFTDF(cell_nonorth).get_jk(dm, kpts=kpts, with_k=False)[0]
        out = multigrid.MultiGridFFTDF(cell_nonorth, kpts=kpts).get_jk(dm)[0]
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_nonorth_get_j(self):
        ref = df.FFTDF(cell_nonorth).get_jk(dm[0], with_k=False)[0]
        out = multigrid.MultiGridFFTDF(cell_nonorth).get_jk(dm)[0]
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_orth_rks_lda_kpts(self):
        xc = 'lda,'
        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        n, exc0, ref = ni.nr_rks(cell_orth, mydf.grids, xc, dm, 1, kpts=kpts)
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        n, exc1, vxc = multigrid.nr_rks(mydf, xc, dm, kpts=kpts)
        self.assertEqual(vxc.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 7)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 7)

    def test_multigrid_kuks(self):
        mf = dft.KUKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, numpy.array((dm_he,dm_he)), kpts=kpts)
        out = multigrid.multigrid_fftdf(mf).get_veff(cell_he, (dm_he,dm_he), kpts=kpts)
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)
        self.assertAlmostEqual(abs(ref.exc-out.exc).max(), 0, 8)
        self.assertAlmostEqual(abs(ref.ecoul-out.ecoul).max(), 0, 8)

    def test_multigrid_krks(self):
        mf = dft.KRKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, dm_he, kpts=kpts)
        out = multigrid.multigrid_fftdf(mf).get_veff(cell_he, dm_he, kpts=kpts)
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)
        self.assertAlmostEqual(abs(ref.exc-out.exc).max(), 0, 8)
        self.assertAlmostEqual(abs(ref.ecoul-out.ecoul).max(), 0, 8)

    def test_multigrid_kroks(self):
        mf = dft.KROKS(cell_he)
        mf.xc = 'lda,'
        nao = cell_he.nao
        mo = dm_he
        mo_occ = numpy.ones((2,nao))
        dm1 = numpy.einsum('kpi,ki,kqi->kpq', mo, mo_occ, mo)
        dm1 = lib.tag_array(numpy.array([dm1,dm1]), mo_coeff=mo,
                            mo_occ=mo_occ*2)
        ref = mf.get_veff(cell_he, dm1, kpts=kpts)
        out = multigrid.multigrid_fftdf(mf).get_veff(cell_he, dm1, kpts=kpts)
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.exc-out.exc).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.ecoul-out.ecoul).max(), 0, 7)

    def test_multigrid_uks(self):
        mf = dft.UKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, numpy.array((dm_he[0],dm_he[0])))
        out = multigrid.multigrid_fftdf(mf).get_veff(cell_he, (dm_he[0], dm_he[0]))
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.exc-out.exc).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.ecoul-out.ecoul).max(), 0, 7)

    def test_multigrid_rks(self):
        mf = dft.RKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, dm_he[0])
        out = multigrid.multigrid_fftdf(mf).get_veff(cell_he, dm_he[0])
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.exc-out.exc).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.ecoul-out.ecoul).max(), 0, 7)

    def test_multigrid_roks(self):
        mf = dft.ROKS(cell_he)
        mf.xc = 'lda,'
        mo = dm_he[0]
        nao = cell_he.nao
        mo_occ = numpy.ones(nao)
        dm1 = numpy.einsum('pi,i,qi->pq', mo, mo_occ, mo)
        dm1 = lib.tag_array(numpy.array([dm1,dm1]), mo_coeff=mo,
                            mo_occ=mo_occ*2)
        ref = mf.get_veff(cell_he, dm1)
        out = multigrid.multigrid_fftdf(mf).get_veff(cell_he, dm1)
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.exc-out.exc).max(), 0, 7)
        self.assertAlmostEqual(abs(ref.ecoul-out.ecoul).max(), 0, 7)

    def test_orth_rks_gga_kpts(self):
        xc = 'b88,'
        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        n, exc0, ref = ni.nr_rks(cell_orth, mydf.grids, xc, dm, hermi=1, kpts=kpts)
        ref += mydf.get_jk(dm, hermi=1, with_k=False, kpts=kpts)[0]
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        n, exc1, vxc = multigrid.nr_rks(mydf, xc, dm, hermi=1, kpts=kpts, with_j=True)
        self.assertEqual(vxc.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 7)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(ref), -0.05697304864467462+0.6990367789096609j, 7)

    def test_eval_rhoG_orth_kpts(self):
        numpy.random.seed(9)
        dm = numpy.random.random(dm1.shape) + numpy.random.random(dm1.shape) * 1j
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        rhoG = multigrid.multigrid._eval_rhoG(mydf, dm, hermi=0, kpts=kpts, deriv=0,
                                              rhog_high_order=True)
        self.assertTrue(rhoG.dtype == numpy.complex128)

        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        ao_kpts = ni.eval_ao(cell_orth, mydf.grids.coords, kpts, deriv=0)
        ref = ni.eval_rho(cell_orth, ao_kpts, dm, hermi=0, xctype='LDA')
        rhoR = tools.ifft(rhoG[0], cell_orth.mesh)
        rhoR *= numpy.prod(cell_orth.mesh)/cell_orth.vol
        self.assertAlmostEqual(abs(rhoR-ref).max(), 0, 8)

    def test_eval_rhoG_orth_gga(self):
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        rhoG = multigrid.multigrid._eval_rhoG(mydf, dm, hermi=1, kpts=kpts, deriv=1,
                                              rhog_high_order=True)

        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        ao_kpts = ni.eval_ao(cell_orth, mydf.grids.coords, kpts, deriv=1)
        ref = ni.eval_rho(cell_orth, ao_kpts, dm, xctype='GGA')
        rhoR = tools.ifft(rhoG[0], cell_orth.mesh)
        rhoR *= numpy.prod(cell_orth.mesh)/cell_orth.vol
        self.assertAlmostEqual(abs(rhoR-ref).max(), 0, 8)

    def test_eval_rhoG_nonorth_gga(self):
        mydf = multigrid.MultiGridFFTDF(cell_nonorth)
        rhoG = multigrid.multigrid._eval_rhoG(mydf, dm, hermi=1, kpts=kpts, deriv=1,
                                              rhog_high_order=True)

        mydf = df.FFTDF(cell_nonorth)
        ni = dft.numint.KNumInt()
        ao_kpts = ni.eval_ao(cell_nonorth, mydf.grids.coords, kpts, deriv=1)
        ref = ni.eval_rho(cell_nonorth, ao_kpts, dm, xctype='GGA')
        rhoR = tools.ifft(rhoG[0], cell_nonorth.mesh)
        rhoR *= numpy.prod(cell_nonorth.mesh)/cell_nonorth.vol
        self.assertAlmostEqual(abs(rhoR-ref).max(), 0, 7)

    def test_gen_rhf_response(self):
        numpy.random.seed(9)
        dm1 = numpy.random.random(dm_he.shape)
        dm1 = dm1 + dm1.transpose(0,2,1)
        dm1[1] = dm1[0]
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()

        mf = dft.KRKS(cell_he)
        mf.with_df = multigrid.MultiGridFFTDF(cell_he)
        mf.kpts = kpts

        mf.xc = 'lda,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, mf.xc, dm_he, dm1,
                                    hermi=1, kpts=kpts)
        vj = mydf.get_jk(dm1, with_k=False, kpts=kpts)[0]
        ref += vj
        v = multigrid.multigrid._gen_rhf_response(mf, dm_he, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

        mf.xc = 'b88,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, mf.xc, dm_he, dm1,
                                    hermi=1, kpts=kpts)
        ref += vj
        v = multigrid.multigrid._gen_rhf_response(mf, dm_he, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 6)

    def test_nr_rks_fxc(self):
        numpy.random.seed(9)
        dm1 = numpy.random.random(dm_he.shape) + numpy.random.random(dm_he.shape)*1j
        dm1 = dm1 + dm1.transpose(0,2,1)
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.NumInt()
        mg_df = multigrid.MultiGridFFTDF(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1,
                                   hermi=1)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 9)

        xc = 'b88,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1,
                                    hermi=1)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 6)

    def test_nr_rks_fxc_hermi0(self):
        numpy.random.seed(9)
        dm1 = numpy.random.random(dm_he.shape) + numpy.random.random(dm_he.shape)*1j
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.NumInt()
        mg_df = multigrid.MultiGridFFTDF(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1, hermi=0)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=0)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 9)

        xc = 'b88,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1, hermi=0)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=0)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 6)

    # FIXME: is the discrepancy due to problems in precision or threshold estimation?
    def test_nr_rks_fxc_st(self):
        numpy.random.seed(9)
        dm1 = numpy.random.random(dm_he.shape) + numpy.random.random(dm_he.shape)*1j
        dm1[1] = dm1[0]
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()
        mg_df = multigrid.MultiGridFFTDF(cell_he)

        mf = dft.KRKS(cell_he)
        mf.with_df = mg_df
        mf.kpts = kpts

        xc = 'lda,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, xc, dm_he, dm1,
                                       singlet=True, kpts=kpts)
        v = multigrid.nr_rks_fxc_st(mg_df, xc, dm_he, dm1, singlet=True, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 4)

        mf.xc = 'b88,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, mf.xc, dm_he, dm1,
                                       singlet=True, kpts=kpts)
        v = multigrid.multigrid._gen_rhf_response(mf, dm_he, singlet=True)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 5)

        mf.xc = 'lda,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, mf.xc, dm_he, dm1,
                                       singlet=False, kpts=kpts)
        v = multigrid.multigrid._gen_rhf_response(mf, dm_he, singlet=False)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 4)

        xc = 'b88,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, xc, dm_he, dm1,
                                       singlet=False, kpts=kpts)
        v = multigrid.nr_rks_fxc_st(mg_df, xc, dm_he, dm1, singlet=False, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 5)

    def test_gen_uhf_response(self):
        numpy.random.seed(9)
        dm1 = numpy.random.random(dm_he.shape)
        dm1 = dm1 + dm1.transpose(0,2,1)
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.NumInt()

        mf = dft.UKS(cell_he)
        mf.with_df = multigrid.MultiGridFFTDF(cell_he)

        mf.xc = 'lda,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, mf.xc, dm_he, dm1, hermi=1)
        vj = mydf.get_jk(dm1, with_k=False)[0]
        ref += vj[0] + vj[1]
        v = multigrid.multigrid._gen_uhf_response(mf, dm_he, with_j=True, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 7)

        mf.xc = 'b88,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, mf.xc, dm_he, dm1, hermi=1)
        ref += vj[0] + vj[1]
        v = multigrid.multigrid._gen_uhf_response(mf, dm_he, with_j=True, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 7)

    # FIXME: is the discrepancy due to problems in precision or threshold estimation?
    def test_nr_uks_fxc(self):
        numpy.random.seed(9)
        dm1 = numpy.random.random(dm_he.shape) + numpy.random.random(dm_he.shape)*1j
        dm1 = dm1 + dm1.transpose(0,2,1)
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()
        mg_df = multigrid.MultiGridFFTDF(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 4)

        xc = 'b88,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 4)

    def test_orth_uks_fxc_hermi0(self):
        numpy.random.seed(9)
        dm1 = numpy.random.random(dm_he.shape) + numpy.random.random(dm_he.shape)*1j
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()
        mg_df = multigrid.MultiGridFFTDF(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 9)

        xc = 'b88,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

    def test_rcut_vs_ke_cut(self):
        xc = 'lda,'
        with lib.temporary_env(multigrid.multigrid, TASKS_TYPE='rcut'):
            mg_df = multigrid.MultiGridFFTDF(cell_orth)
            n1, exc1, v1 = multigrid.nr_rks(mg_df, xc, dm1, kpts=kpts)
            self.assertEqual(len(mg_df.tasks), 3)
        with lib.temporary_env(multigrid.multigrid, TASKS_TYPE='ke_cut'):
            mg_df = multigrid.MultiGridFFTDF(cell_orth)
            n2, exc2, v2 = multigrid.nr_rks(mg_df, xc, dm1, kpts=kpts)
            self.assertEqual(len(mg_df.tasks), 6)
        self.assertAlmostEqual(n1, n2, 6)
        self.assertAlmostEqual(exc1, exc2, 7)
        self.assertAlmostEqual(abs(v1-v2).max(), 0, 7)

    def test_multigrid_krks_high_cost(self):
        cell = gto.M(
            a = numpy.eye(3)*3.5668,
            atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            #basis = 'sto3g',
            #basis = 'ccpvdz',
            basis = 'gth-dzvp',
            #basis = 'gth-szv',
            pseudo = 'gth-pade'
        )
        mesh = [21] * 3
        multigrid.multi_grids_tasks(cell, mesh, 5)

        nao = cell.nao_nr()
        numpy.random.seed(1)
        kpts = cell.make_kpts([3,1,1])

        dm = numpy.random.random((len(kpts),nao,nao)) * .2
        dm += numpy.eye(nao)
        dm = dm + dm.transpose(0,2,1)

        mf = dft.KRKS(cell)
        ref = mf.get_veff(cell, dm, kpts=kpts)
        out = multigrid.multigrid(mf).get_veff(cell, dm, kpts=kpts)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)


if __name__ == '__main__':
    print("Full Tests for multigrid")
    unittest.main()
