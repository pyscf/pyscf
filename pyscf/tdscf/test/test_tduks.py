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
import copy
from pyscf import lib, gto, scf, dft
from pyscf import tdscf

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom = [
    ['H' , (0. , 0. , .917)],
    ['F' , (0. , 0. , 0.)], ]
mol.spin = 2
mol.basis = '631g'
mol.build()

mf_lda = dft.UKS(mol).run(xc='lda')
mf_bp86 = dft.UKS(mol).run(xc='b88,p86')
mf_b3lyp = dft.UKS(mol).run(xc='b3lyp')

class KnownValues(unittest.TestCase):
    def test_nohbrid_lda(self):
        td = tdscf.uks.TDDFTNoHybrid(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -36.999992353507778, 6)

    def test_nohbrid_b88p86(self):
        td = tdscf.uks.TDDFTNoHybrid(mf_bp86)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -36.024387689354342, 6)

    def test_tddft_lda(self):
        td = tdscf.uks.TDDFT(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -36.999992353507778, 6)

    def test_tddft_b88p86(self):
        td = tdscf.uks.TDDFT(mf_bp86)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -36.024387689354342, 6)

    def test_tddft_b3lyp(self):
        td = tdscf.uks.TDDFT(mf_b3lyp)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -36.755073461699247, 6)

    def test_tda_b3lyp(self):
        td = tdscf.TDA(mf_b3lyp)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -36.645878790257441, 6)

    def test_tda_lda(self):
        td = tdscf.TDA(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -37.011261125964992, 6)

    def test_ab_hf(self):
        mf = scf.UHF(mol).run()
        a, b = tdscf.TDDFT(mf).get_ab()
        ftda = tdscf.uhf.gen_tda_operation(mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(mf)[0]
        nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
        nvir_a = numpy.count_nonzero(mf.mo_occ[0] == 0)
        nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)
        nvir_b = numpy.count_nonzero(mf.mo_occ[1] == 0)
        numpy.random.seed(2)
        xa, ya = numpy.random.random((2,nocc_a,nvir_a))
        xb, yb = numpy.random.random((2,nocc_b,nvir_b))
        x = numpy.hstack((xa.ravel(), xb.ravel()))
        y = numpy.hstack((ya.ravel(), yb.ravel()))
        xy = numpy.hstack((x, y))
        ax_a = numpy.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= numpy.einsum('iajb,jb->ia', a[1], xb)
        ax_b = numpy.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= numpy.einsum('iajb,jb->ia', a[2], xb)
        ax = numpy.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x])).max(), 0, 9)

        ay_a = numpy.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= numpy.einsum('iajb,jb->ia', a[1], yb)
        ay_b = numpy.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= numpy.einsum('iajb,jb->ia', a[2], yb)
        ay = numpy.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = numpy.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= numpy.einsum('iajb,jb->ia', b[1], xb)
        bx_b = numpy.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= numpy.einsum('iajb,jb->ia', b[2], xb)
        bx = numpy.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = numpy.einsum('iajb,jb->ia', b[0], ya)
        by_a+= numpy.einsum('iajb,jb->ia', b[1], yb)
        by_b = numpy.einsum('jbia,jb->ia', b[1], ya)
        by_b+= numpy.einsum('iajb,jb->ia', b[2], yb)
        by = numpy.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = numpy.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf([xy])
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_lda(self):
        mf = mf_lda
        a, b = tdscf.TDDFT(mf).get_ab()
        ftda = tdscf.uhf.gen_tda_operation(mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(mf)[0]
        nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
        nvir_a = numpy.count_nonzero(mf.mo_occ[0] == 0)
        nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)
        nvir_b = numpy.count_nonzero(mf.mo_occ[1] == 0)
        numpy.random.seed(2)
        xa, ya = numpy.random.random((2,nocc_a,nvir_a))
        xb, yb = numpy.random.random((2,nocc_b,nvir_b))
        x = numpy.hstack((xa.ravel(), xb.ravel()))
        y = numpy.hstack((ya.ravel(), yb.ravel()))
        xy = numpy.hstack((x, y))
        ax_a = numpy.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= numpy.einsum('iajb,jb->ia', a[1], xb)
        ax_b = numpy.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= numpy.einsum('iajb,jb->ia', a[2], xb)
        ax = numpy.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x])).max(), 0, 9)

        ay_a = numpy.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= numpy.einsum('iajb,jb->ia', a[1], yb)
        ay_b = numpy.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= numpy.einsum('iajb,jb->ia', a[2], yb)
        ay = numpy.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = numpy.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= numpy.einsum('iajb,jb->ia', b[1], xb)
        bx_b = numpy.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= numpy.einsum('iajb,jb->ia', b[2], xb)
        bx = numpy.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = numpy.einsum('iajb,jb->ia', b[0], ya)
        by_a+= numpy.einsum('iajb,jb->ia', b[1], yb)
        by_b = numpy.einsum('jbia,jb->ia', b[1], ya)
        by_b+= numpy.einsum('iajb,jb->ia', b[2], yb)
        by = numpy.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = numpy.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf([xy])
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_b3lyp(self):
        mf = mf_b3lyp
        a, b = tdscf.TDDFT(mf).get_ab()
        ftda = tdscf.uhf.gen_tda_operation(mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(mf)[0]
        nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
        nvir_a = numpy.count_nonzero(mf.mo_occ[0] == 0)
        nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)
        nvir_b = numpy.count_nonzero(mf.mo_occ[1] == 0)
        numpy.random.seed(2)
        xa, ya = numpy.random.random((2,nocc_a,nvir_a))
        xb, yb = numpy.random.random((2,nocc_b,nvir_b))
        x = numpy.hstack((xa.ravel(), xb.ravel()))
        y = numpy.hstack((ya.ravel(), yb.ravel()))
        xy = numpy.hstack((x, y))
        ax_a = numpy.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= numpy.einsum('iajb,jb->ia', a[1], xb)
        ax_b = numpy.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= numpy.einsum('iajb,jb->ia', a[2], xb)
        ax = numpy.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x])).max(), 0, 9)

        ay_a = numpy.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= numpy.einsum('iajb,jb->ia', a[1], yb)
        ay_b = numpy.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= numpy.einsum('iajb,jb->ia', a[2], yb)
        ay = numpy.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = numpy.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= numpy.einsum('iajb,jb->ia', b[1], xb)
        bx_b = numpy.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= numpy.einsum('iajb,jb->ia', b[2], xb)
        bx = numpy.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = numpy.einsum('iajb,jb->ia', b[0], ya)
        by_a+= numpy.einsum('iajb,jb->ia', b[1], yb)
        by_b = numpy.einsum('jbia,jb->ia', b[1], ya)
        by_b+= numpy.einsum('iajb,jb->ia', b[2], yb)
        by = numpy.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = numpy.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf([xy])
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_nto(self):
        mf = scf.UHF(mol).run()
        td = tdscf.TDA(mf).run()
        w, nto = td.get_nto(state=1)
        self.assertAlmostEqual(w[0][0], 8.1446175495227284e-05, 9)
        self.assertAlmostEqual(w[1][0], 0.99983845975349361, 9)
        self.assertAlmostEqual(lib.finger(w[0]), 0.00012472118674926, 9)
        self.assertAlmostEqual(lib.finger(w[1]), 0.99983845975349361, 9)

        pmol = copy.copy(mol)
        pmol.symmetry = True
        pmol.build(0, 0)
        mf = scf.UHF(mol).run()
        td = tdscf.TDA(mf).run()
        w, nto = td.get_nto(state=0)
        self.assertAlmostEqual(w[0][0], 8.1446175495227284e-05, 9)
        self.assertAlmostEqual(w[1][0], 0.99983845975349361, 9)
        self.assertAlmostEqual(lib.finger(w[0]), 0.00012472118674926, 9)
        self.assertAlmostEqual(lib.finger(w[1]), 0.99983845975349361, 9)

    def test_analyze(self):
        mf = scf.UHF(mol).run()
        td = tdscf.TDHF(mf).run()
        f = td.oscillator_strength(gauge='length')
        self.assertAlmostEqual(lib.finger(f), -0.00034747191841310503, 7)
        f = td.oscillator_strength(gauge='velocity', order=2)
        self.assertAlmostEqual(lib.finger(f), -0.0043239860814561762, 7)
        td.analyze()


if __name__ == "__main__":
    print("Full Tests for TD-UKS")
    unittest.main()
