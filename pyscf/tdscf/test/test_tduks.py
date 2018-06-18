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
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]
mol.spin = 2
mol.basis = '631g'
mol.build()

mf_lda = dft.UKS(mol).set(xc='lda', conv_tol=1e-12)
mf_lda.grids.prune = None
mf_lda = mf_lda.newton().run()
mf_bp86 = dft.UKS(mol).set(xc='b88,p86', conv_tol=1e-12)
mf_bp86.grids.prune = None
mf_bp86 = mf_bp86.newton().run()
mf_b3lyp = dft.UKS(mol).set(xc='b3lyp', conv_tol=1e-12)
mf_b3lyp.grids.prune = None
mf_b3lyp = mf_b3lyp.newton().run()

def diagonalize(a, b, nroots=4):
    a_aa, a_ab, a_bb = a
    b_aa, b_ab, b_bb = b
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    a_ab = a_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    a_bb = a_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    b_aa = b_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    b_ab = b_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    b_bb = b_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    a = numpy.bmat([[ a_aa  , a_ab],
                    [ a_ab.T, a_bb]])
    b = numpy.bmat([[ b_aa  , b_ab],
                    [ b_ab.T, b_bb]])
    e = numpy.linalg.eig(numpy.bmat([[a        , b       ],
                                     [-b.conj(),-a.conj()]]))[0]
    lowest_e = numpy.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e

class KnownValues(unittest.TestCase):
    def test_nohbrid_lda(self):
        td = tdscf.uks.TDDFTNoHybrid(mf_lda).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 6)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.finger(es[:3]*27.2114), 1.2946309669294163, 6)

    def test_nohbrid_b88p86(self):
        td = tdscf.uks.TDDFTNoHybrid(mf_bp86).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 6)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.finger(es[:3]*27.2114), 1.4624730971221087, 6)

    def test_tddft_lda(self):
        td = tdscf.uks.TDDFT(mf_lda).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es[:3]), 1.2946309669294163, 6)

    def test_tddft_b88p86(self):
        td = tdscf.uks.TDDFT(mf_bp86).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es[:3]), 1.4624730971221087, 6)

    def test_tddft_b3lyp(self):
        td = tdscf.uks.TDDFT(mf_b3lyp).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es[:3]), 1.2984822994759448, 6)

    def test_tda_b3lyp(self):
        td = tdscf.TDA(mf_b3lyp).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es[:3]), 1.4303636271767162, 6)

    def test_tda_lda(self):
        td = tdscf.TDA(mf_lda).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es[:3]), 1.4581538269747121, 6)

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
        self.assertAlmostEqual(w[0][0], 0.00018520143461015, 9)
        self.assertAlmostEqual(w[1][0], 0.99963372674044326, 9)
        self.assertAlmostEqual(lib.finger(w[0]), 0.00027305600430816, 9)
        self.assertAlmostEqual(lib.finger(w[1]), 0.99964370569529093, 9)

        pmol = copy.copy(mol)
        pmol.symmetry = True
        pmol.build(0, 0)
        mf = scf.UHF(mol).run()
        td = tdscf.TDA(mf).run()
        w, nto = td.get_nto(state=0)
        self.assertAlmostEqual(w[0][0], 0.00018520143461016, 9)
        self.assertAlmostEqual(w[1][0], 0.99963372674044326, 9)
        self.assertAlmostEqual(lib.finger(w[0]), 0.00027305600430816, 9)
        self.assertAlmostEqual(lib.finger(w[1]), 0.99964370569529093, 9)

    def test_analyze(self):
        mf = scf.UHF(mol).run()
        td = tdscf.TDHF(mf).run()
        f = td.oscillator_strength(gauge='length')
        self.assertAlmostEqual(lib.finger(f), 0.16147450863004867, 7)
        f = td.oscillator_strength(gauge='velocity', order=2)
        self.assertAlmostEqual(lib.finger(f), 0.19750347627735745, 7)
        td.analyze()


if __name__ == "__main__":
    print("Full Tests for TD-UKS")
    unittest.main()
