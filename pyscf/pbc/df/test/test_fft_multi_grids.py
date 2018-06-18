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
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import df
from pyscf.pbc.df import fft_jk
from pyscf.pbc.df import fft_multi_grids

fft_multi_grids.RMAX_FACTOR = 0.5
fft_multi_grids.RMAX_RATIO = 0.75
fft_multi_grids.RHOG_HIGH_DERIV = False

cell = gto.M(
    a = numpy.eye(3)*4,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917 ''',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    verbose = 0,
)
nao = cell.nao_nr()
kpts = cell.make_kpts([2,1,1])
numpy.random.seed(1)
dm = (numpy.random.random((len(kpts),nao,nao)) +
      numpy.random.random((len(kpts),nao,nao)) * 1j)
dm = dm + dm.transpose(0,2,1).conj()
dmb= (numpy.random.random((len(kpts),nao,nao)) +
      numpy.random.random((len(kpts),nao,nao)) * 1j)
dmb = dmb + dmb.transpose(0,2,1).conj()
dmab = numpy.asarray((dm,dmb))

class KnownValues(unittest.TestCase):
    def test_get_pp(self):
        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        v1 = mydf.get_pp(kpts)
        self.assertAlmostEqual(lib.finger(v1), 1.698462975912101, 9)
        self.assertTrue(v1.dtype == numpy.complex128)

    def test_get_nuc(self):
        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        v1 = mydf.get_nuc(kpts)
        self.assertAlmostEqual(lib.finger(v1), 1.4458185035880309, 9)
        self.assertTrue(v1.dtype == numpy.complex128)

    def test_get_j(self):
        mydf = df.FFTDF(cell)
        ref = fft_jk.get_j_kpts(mydf, dm, kpts=kpts)
        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        v = mydf.get_j_kpts(dm, kpts=kpts)
        self.assertAlmostEqual(abs(ref-v).max(), 0, 8)
        self.assertAlmostEqual(lib.finger(v), 0.21963596743261393, 8)

    def test_rks_lda(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        ref_j = fft_jk.get_j_kpts(mydf, dm, kpts=kpts)
        n, exc, ref = mydf._numint.nr_rks(cell, mydf.grids, 'lda,', dm, 0, kpts)
        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.rks_j_xc(dm, 'lda,', kpts=kpts, with_j=True, j_in_xc=False)
        self.assertAlmostEqual(abs(ref_j-vj).max(), 0, 8)
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 6)
        self.assertAlmostEqual(lib.finger(vxc), -0.10369726976205595, 6)
        self.assertAlmostEqual(exc, -3.0494590778748032, 8)
        self.assertAlmostEqual(n, 8.0749444279646347, 8)

    def test_rks_gga(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        n, exc, ref = mydf._numint.nr_rks(cell, mydf.grids, 'b88,', dm, 0, kpts)

        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.rks_j_xc(dm, 'b88,', kpts=kpts, with_j=True, j_in_xc=False)
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 4)
        self.assertAlmostEqual(lib.finger(vxc), -0.16275189251143374, 4)
        self.assertAlmostEqual(lib.finger(vj), 0.21963596743261393, 8)
        self.assertAlmostEqual(exc, -3.3730554511051336, 8)
        self.assertAlmostEqual(n, 8.0749444279646347, 8)

        fft_multi_grids.RHOG_HIGH_DERIV = True
        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.rks_j_xc(dm, 'b88,', kpts=kpts, with_j=True, j_in_xc=False)
        fft_multi_grids.RHOG_HIGH_DERIV = False
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 5)
        self.assertAlmostEqual(lib.finger(vxc), -0.16273163782921141, 8)
        self.assertAlmostEqual(exc, -3.3730552963576814, 8)
        self.assertAlmostEqual(n, 8.0749444279646347, 8)

    def test_rks_mgga(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        n, exc, ref = mydf._numint.nr_rks(cell, mydf.grids, 'tpss,', dm, 0, kpts)

        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.rks_j_xc(dm, 'tpss,', kpts=kpts, with_j=True, j_in_xc=False)
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 2)
        self.assertAlmostEqual(lib.finger(vxc), -0.12904037508240063, 8)
        self.assertAlmostEqual(lib.finger(vj), 0.21963596743261393, 8)
        self.assertAlmostEqual(exc, -3.4188140995616179, 8)
        self.assertAlmostEqual(n, 8.0749444279646347, 8)

    def test_rks_lda_plus_j(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        ref_j = fft_jk.get_j_kpts(mydf, dm, kpts=kpts)
        n, exc, ref = mydf._numint.nr_rks(cell, mydf.grids, 'lda,', dm, 0, kpts)

        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.rks_j_xc(dm, 'lda,', kpts=kpts, with_j=False, j_in_xc=True)
        self.assertAlmostEqual(abs(ref_j+ref-vxc).max(), 0, 6)
        self.assertAlmostEqual(lib.finger(vxc), 0.11593869767055653+0j, 8)
        self.assertAlmostEqual(exc, 2.3601737480485134, 8)
        self.assertAlmostEqual(n, 8.0749444279646347, 8)

    def test_rks_gga_plus_j(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        ref_j = fft_jk.get_j_kpts(mydf, dm, kpts=kpts)
        n, exc, ref = mydf._numint.nr_rks(cell, mydf.grids, 'b88,', dm, 0, kpts)

        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.rks_j_xc(dm, 'b88,', kpts=kpts, with_j=False, j_in_xc=True)
        self.assertAlmostEqual(abs(ref_j+ref-vxc).max(), 0, 4)
        self.assertAlmostEqual(lib.finger(vxc), 0.056907756241836749+0j, 4)
        self.assertAlmostEqual(exc, 2.036577374818183, 6)
        self.assertAlmostEqual(n, 8.0749444279646347, 8)


    def test_uks_lda(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        ref_j = fft_jk.get_j_kpts(mydf, dmab, kpts=kpts)
        n, exc, ref = mydf._numint.nr_uks(cell, mydf.grids, 'lda,', dmab, 0, kpts)
        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.uks_j_xc(dmab, 'lda,', kpts=kpts, with_j=True, j_in_xc=False)
        self.assertAlmostEqual(abs(ref_j-vj).max(), 0, 8)
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 6)
        self.assertAlmostEqual(lib.finger(vxc), 0.56051003830739932, 6)
        self.assertAlmostEqual(exc, -8.1508688099749218, 8)
        self.assertAlmostEqual(sum(n), 16.77639832814981, 8)

    def test_uks_gga(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        n, exc, ref = mydf._numint.nr_uks(cell, mydf.grids, 'b88,', dmab, 0, kpts)

        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.uks_j_xc(dmab, 'b88,', kpts=kpts, with_j=True, j_in_xc=False)
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 4)
        self.assertAlmostEqual(lib.finger(vxc), 0.62762734582165547, 4)
        self.assertAlmostEqual(lib.finger(vj), -0.80873190911595572, 8)
        self.assertAlmostEqual(exc, -8.8140921361508582, 8)
        self.assertAlmostEqual(sum(n), 16.77639832814981, 8)

        fft_multi_grids.RHOG_HIGH_DERIV = True
        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.uks_j_xc(dmab, 'b88,', kpts=kpts, with_j=True, j_in_xc=False)
        fft_multi_grids.RHOG_HIGH_DERIV = False
        self.assertAlmostEqual(abs(ref-vxc).max(), 0, 4)
        self.assertAlmostEqual(lib.finger(vxc), 0.62762734582165547, 4)
        self.assertAlmostEqual(exc, -8.8140921361508582, 6)
        self.assertAlmostEqual(sum(n), 16.77639832814981, 6)

    ## FIXME: small errors in rho that leads to big difference
    #def test_uks_mgga(self):
    #    mydf = df.FFTDF(cell)
    #    mydf.grids.build()
    #    n, exc, ref = mydf._numint.nr_uks(cell, mydf.grids, 'tpss', dmab, 0, kpts)

    #    mydf = fft_multi_grids.MultiGridFFTDF(cell)
    #    n, exc, vxc, vj = mydf.uks_j_xc(dmab, 'tpss', kpts=kpts, with_j=True, j_in_xc=False)
    #    self.assertAlmostEqual(abs(ref-vxc).max(), 0, 2)
    #    self.assertAlmostEqual(lib.finger(vxc), 0.69310396591433887, 8)
    #    self.assertAlmostEqual(lib.finger(vj), -0.80873190911595572, 8)
    #    self.assertAlmostEqual(exc, -9.1512747624120561, 8)
    #    self.assertAlmostEqual(sum(n), 16.77639832814981, 8)

    def test_uks_lda_plus_j(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        ref_j = fft_jk.get_j_kpts(mydf, dmab, kpts=kpts)
        n, exc, ref = mydf._numint.nr_uks(cell, mydf.grids, 'lda,', dmab, 0, kpts)

        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.uks_j_xc(dmab, 'lda,', kpts=kpts, with_j=False, j_in_xc=True)
        self.assertAlmostEqual(abs(ref_j+ref-vxc).max(), 0, 6)
        self.assertAlmostEqual(lib.finger(vxc), -0.24822187080855951+0j, 4)
        self.assertAlmostEqual(exc, 3.3968809873729704, 6)
        self.assertAlmostEqual(sum(n), 16.77639832814981, 8)

    def test_uks_gga_plus_j(self):
        mydf = df.FFTDF(cell)
        mydf.grids.build()
        ref_j = fft_jk.get_j_kpts(mydf, dmab, kpts=kpts)
        n, exc, ref = mydf._numint.nr_uks(cell, mydf.grids, 'b88,', dmab, 0, kpts)

        mydf = fft_multi_grids.MultiGridFFTDF(cell)
        n, exc, vxc, vj = mydf.uks_j_xc(dmab, 'b88,', kpts=kpts, with_j=False, j_in_xc=True)
        self.assertAlmostEqual(abs(ref_j+ref-vxc).max(), 0, 4)
        self.assertAlmostEqual(lib.finger(vxc), -0.18113908287908242+0j, 8)
        self.assertAlmostEqual(exc, 2.7336576611968653, 8)
        self.assertAlmostEqual(sum(n), 16.77639832814981, 8)


if __name__ == '__main__':
    print("Full Tests for fft_multi_grids")
    unittest.main()

