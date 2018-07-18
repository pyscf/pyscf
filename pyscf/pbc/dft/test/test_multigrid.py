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

numpy.random.seed(2)
cell_orth = gto.M(
    verbose = 7,
    output = '/dev/null',
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     1.8     1.8     1.8   ''',
    basis = 'gth-dzv',
    pseudo = 'gth-pade',
    mesh = [48] * 3,
)
cell_nonorth = gto.M(
    a = numpy.eye(3)*3.5668 + numpy.random.random((3,3)),
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917''',
    basis = 'gth-dzv',
    pseudo = 'gth-pade',
    mesh = [44,43,42],
)

cell_he = gto.M(atom='He 0 0 0',
                basis=[[0, ( 1, 1, .1), (.5, .1, 1)],
                       [1, (.8, 1)]],
                unit='B',
                mesh=[8]*3,
                a=numpy.eye(3)*5)

kpts = numpy.random.random((2,3))
nao = cell_orth.nao_nr()
dm = numpy.random.random((len(kpts),nao,nao)) * .2
dm1 = dm + numpy.eye(nao)
dm = dm1 + dm1.transpose(0,2,1)

dm_he = numpy.random.random((len(kpts), cell_he.nao, cell_he.nao))
dm_he = dm_he + dm_he.transpose(0,2,1)

def tearDownModule():
    global cell_orth, cell_nonorth, cell_he
    del cell_orth, cell_nonorth, cell_he

class KnownValues(unittest.TestCase):
    def test_orth_get_pp(self):
        ref = df.FFTDF(cell_orth).get_pp()
        out = multigrid.MultiGridFFTDF(cell_orth).get_pp()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 9)

    def test_nonorth_get_pp(self):
        ref = df.FFTDF(cell_nonorth).get_pp()
        out = multigrid.MultiGridFFTDF(cell_nonorth).get_pp()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 9)

    def test_orth_get_nuc_kpts(self):
        ref = df.FFTDF(cell_orth).get_nuc(kpts)
        out = multigrid.MultiGridFFTDF(cell_orth).get_nuc(kpts)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_orth_get_j_kpts(self):
        ref = df.FFTDF(cell_orth).get_jk(dm, kpts=kpts, with_k=False)[0]
        out = multigrid.MultiGridFFTDF(cell_orth).get_jk(dm, kpts=kpts)[0]
        self.assertAlmostEqual(abs(ref-out).max(), 0, 9)

    def test_nonorth_get_j_kpts(self):
        ref = df.FFTDF(cell_nonorth).get_jk(dm, kpts=kpts, with_k=False)[0]
        out = multigrid.MultiGridFFTDF(cell_nonorth, kpts=kpts).get_jk(dm)[0]
        self.assertAlmostEqual(abs(ref-out).max(), 0, 9)

    def test_nonorth_get_j(self):
        ref = df.FFTDF(cell_nonorth).get_jk(dm[0], with_k=False)[0]
        out = multigrid.MultiGridFFTDF(cell_nonorth).get_jk(dm)[0]
        self.assertAlmostEqual(abs(ref-out).max(), 0, 9)

    def test_orth_rks_lda_kpts(self):
        xc = 'lda,'
        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        n, exc0, ref = ni.nr_rks(cell_orth, mydf.grids, xc, dm, 0, kpts=kpts)
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        n, exc1, vxc, vj = mydf.rks_j_xc(dm, xc, kpts=kpts, j_in_xc=False, with_j=False)
        self.assertAlmostEqual(float(abs(ref-vxc).max()), 0, 9)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 9)

    def test_multigrid_kuks(self):
        mf = dft.KUKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, numpy.array((dm_he,dm_he)), kpts=kpts)
        out = multigrid.multigrid(mf).get_veff(cell_he, (dm_he,dm_he), kpts=kpts)
        self.assertAlmostEqual(float(abs(ref-out).max()), 0, 9)

    def test_multigrid_krks(self):
        mf = dft.KRKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, dm_he, kpts=kpts)
        out = multigrid.multigrid(mf).get_veff(cell_he, dm_he, kpts=kpts)
        self.assertAlmostEqual(float(abs(ref-out).max()), 0, 9)

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
        out = multigrid.multigrid(mf).get_veff(cell_he, dm1, kpts=kpts)
        self.assertAlmostEqual(float(abs(ref-out).max()), 0, 9)

    def test_multigrid_uks(self):
        mf = dft.UKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, numpy.array((dm_he[0],dm_he[0])))
        out = multigrid.multigrid(mf).get_veff(cell_he, (dm_he[0], dm_he[0]))
        self.assertAlmostEqual(float(abs(ref-out).max()), 0, 9)

    def test_multigrid_rks(self):
        mf = dft.RKS(cell_he)
        mf.xc = 'lda,'
        ref = mf.get_veff(cell_he, dm_he[0])
        out = multigrid.multigrid(mf).get_veff(cell_he, dm_he[0])
        self.assertAlmostEqual(float(abs(ref-out).max()), 0, 9)

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
        out = multigrid.multigrid(mf).get_veff(cell_he, dm1)
        self.assertAlmostEqual(float(abs(ref-out).max()), 0, 9)

    def test_orth_rks_gga_kpts(self):
        xc = 'b88,'
        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        n, exc0, ref = ni.nr_rks(cell_orth, mydf.grids, xc, dm, 0, kpts=kpts)
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        n, exc1, vxc, vj = mydf.rks_j_xc(dm, xc, kpts=kpts, j_in_xc=False, with_j=False)
        self.assertAlmostEqual(float(abs(ref-vxc).max()), 0, 9)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 8)

    def test_orth_uks_lda_hermi0(self):
        xc = 'lda,'
        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.NumInt()
        n, exc0, ref = ni.nr_uks(cell_orth, mydf.grids, xc, dm1, 0)
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        n, exc1, vxc, vj = mydf.uks_j_xc(dm1, xc, hermi=0, j_in_xc=False, with_j=False)
        self.assertAlmostEqual(float(abs(ref-vxc).max()), 0, 9)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 9)

    def test_orth_uks_gga_hermi0(self):
        xc = 'b88,'
        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.NumInt()
        n, exc0, ref = ni.nr_uks(cell_orth, mydf.grids, xc, dm1, 0)
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        n, exc1, vxc, vj = mydf.uks_j_xc(dm1, xc, hermi=0, j_in_xc=False, with_j=False)
        self.assertAlmostEqual(float(abs(ref-vxc).max()), 0, 9)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 8)

    def test_eval_rhoG_orth_kpts(self):
        numpy.random.seed(9)
        dm = numpy.random.random(dm1.shape) + numpy.random.random(dm1.shape) * 1j
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        rhoG = multigrid._eval_rhoG(mydf, dm, hermi=0, kpts=kpts, deriv=0)
        self.assertTrue(rhoG.dtype == numpy.complex128)

        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        ao_kpts = ni.eval_ao(cell_orth, mydf.grids.coords, kpts, deriv=0)
        ref = ni.eval_rho(cell_orth, ao_kpts, dm, hermi=0, xctype='LDA')
        rhoR = tools.ifft(rhoG[0], cell_orth.mesh).real
        rhoR *= numpy.prod(cell_orth.mesh)/cell_orth.vol
        self.assertAlmostEqual(abs(rhoR-ref).max(), 0, 7)

    def test_eval_rhoG_orth_gga(self):
        mydf = multigrid.MultiGridFFTDF(cell_orth)
        rhoG = multigrid._eval_rhoG(mydf, dm, hermi=1, kpts=kpts, deriv=1)

        mydf = df.FFTDF(cell_orth)
        ni = dft.numint.KNumInt()
        ao_kpts = ni.eval_ao(cell_orth, mydf.grids.coords, kpts, deriv=1)
        ref = ni.eval_rho(cell_orth, ao_kpts, dm, xctype='GGA')
        rhoR = tools.ifft(rhoG[0], cell_orth.mesh).real
        rhoR *= numpy.prod(cell_orth.mesh)/cell_orth.vol
        self.assertAlmostEqual(abs(rhoR-ref).max(), 0, 6)

    def test_eval_rhoG_nonorth_gga(self):
        mydf = multigrid.MultiGridFFTDF(cell_nonorth)
        rhoG = multigrid._eval_rhoG(mydf, dm, hermi=1, kpts=kpts, deriv=1)

        mydf = df.FFTDF(cell_nonorth)
        ni = dft.numint.KNumInt()
        ao_kpts = ni.eval_ao(cell_nonorth, mydf.grids.coords, kpts, deriv=1)
        ref = ni.eval_rho(cell_nonorth, ao_kpts, dm, xctype='GGA')
        rhoR = tools.ifft(rhoG[0], cell_nonorth.mesh).real
        rhoR *= numpy.prod(cell_nonorth.mesh)/cell_nonorth.vol
        self.assertAlmostEqual(abs(rhoR-ref).max(), 0, 5)

if __name__ == '__main__':
    print("Full Tests for multigrid")
    unittest.main()
