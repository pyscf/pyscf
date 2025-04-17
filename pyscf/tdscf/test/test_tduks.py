#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto, scf, dft, symm
from pyscf import tdscf

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
    a = numpy.block([[ a_aa  , a_ab],
                     [ a_ab.T, a_bb]])
    b = numpy.block([[ b_aa  , b_ab],
                     [ b_ab.T, b_bb]])
    abba = numpy.asarray(numpy.block([[a        , b       ],
                                      [-b.conj(),-a.conj()]]))
    e = numpy.linalg.eig(abba)[0]
    lowest_e = numpy.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e

def setUpModule():
    global mol, mol1, mf_uhf, td_hf, mf_lda, mf_bp86, mf_b3lyp, mf_m06l
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = '''
    O     0.   0.       0.
    H     0.   -0.757   0.587
    H     0.   0.757    0.587'''
    mol.spin = 2
    mol.basis = '631g'
    mol.build()

    mol1 = gto.Mole()
    mol1.verbose = 0
    mol1.atom = '''
    O     0.   0.       0.
    H     0.   -0.757   0.587
    H     0.   0.757    0.587'''
    mol1.basis = '631g'
    mol1.build()

    mf_uhf = scf.UHF(mol).run()
    td_hf = tdscf.TDHF(mf_uhf).run(conv_tol=1e-6)

    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mf_lda = dft.UKS(mol).set(xc='lda', conv_tol=1e-12)
        mf_lda.grids.prune = None
        mf_lda = mf_lda.newton().run()
        mf_bp86 = dft.UKS(mol).set(xc='b88,p86', conv_tol=1e-12)
        mf_bp86.grids.prune = None
        mf_bp86 = mf_bp86.newton().run()
        mf_b3lyp = dft.UKS(mol).set(xc='b3lyp5', conv_tol=1e-12)
        mf_b3lyp.grids.prune = None
        mf_b3lyp = mf_b3lyp.newton().run()
        mf_m06l = dft.UKS(mol).run(xc='m06l')

def tearDownModule():
    global mol, mol1, mf_uhf, td_hf, mf_lda, mf_bp86, mf_b3lyp, mf_m06l
    mol.stdout.close()
    del mol, mol1, mf_uhf, td_hf, mf_lda, mf_bp86, mf_b3lyp, mf_m06l


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_nohybrid_lda(self):
        td = tdscf.uks.CasidaTDDFT(mf_lda)
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 6)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 1.294630966929489, 4)

        mf = dft.UKS(mol1).run(xc='lda, vwn_rpa').run()
        td = mf.CasidaTDDFT()
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        ref = [6.94083826, 7.61492553, 8.55550045, 9.36308859, 9.49948318]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

    def test_nohybrid_b88p86(self):
        td = tdscf.uks.CasidaTDDFT(mf_bp86)
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 6)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 1.4624730971221087, 4)

    def test_tddft_lda(self):
        td = tdscf.uks.TDDFT(mf_lda)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 1.2946309669294163, 4)

    def test_tddft_b88p86(self):
        td = tdscf.uks.TDDFT(mf_bp86)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 1.4624730971221087, 4)
        ref = [2.45700922, 2.93224712, 6.19693767, 12.22264487, 13.40445012]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

        mf = dft.UKS(mol1).run(xc='b88,p86').run()
        es = mf.TDDFT().kernel(nstates=5)[0] * 27.2114
        ref = [6.96396398, 7.70954799, 8.59882244, 9.35356454, 9.69774071]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

    def test_tddft_b3lyp(self):
        td = tdscf.uks.TDDFT(mf_b3lyp)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 1.2984822994759448, 4)

    def test_tddft_camb3lyp(self):
        mf = mol1.UKS(xc='camb3lyp').run()
        td = mf.TDDFT()
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 6)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 7.69383202636, 4)

    def test_tda_b3lyp(self):
        td = tdscf.TDA(mf_b3lyp)
        es = td.kernel(nstates=4)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 1.4303636271767162, 4)

    def test_tda_lda(self):
        td = tdscf.TDA(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 1.4581538269747121, 4)
        ref = [2.14644585, 3.27738191, 5.90913787, 12.14980714, 13.15535042]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

        mf = dft.UKS(mol1).run(xc='lda,vwn').run()
        td = mf.TDA()
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        ref = [6.88046608, 7.58244885, 8.49961771, 9.30209259, 9.53368005]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

    def test_tda_rsh(self):
        mol = gto.M(atom='H 0 0 0.6; H 0 0 0', basis = "6-31g")
        mf = dft.UKS(mol)
        mf.xc = 'wb97'
        e = mf.kernel()
        self.assertAlmostEqual(e, -1.14670613191817, 8)
        e_td = mf.TDA().set(nstates=5).kernel()[0]
        ref = [0.51100114, 0.59718449, 0.86558547, 1.02667323, 1.57231767]
        self.assertAlmostEqual(abs(e_td - ref).max(), 0, 6)

        mf.xc = 'hse06'
        e = mf.kernel()
        self.assertAlmostEqual(e, -1.1447666793407982, 8)
        e_td = mf.TDA().set(nstates=5).kernel()[0]
        ref = [0.47713861, 0.60314354, 0.83949946, 1.03802547, 1.55472339]
        self.assertAlmostEqual(abs(e_td - ref).max(), 0, 6)

    def test_tda_m06l(self):
        td = mf_m06l.TDA()
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -20.49388623318, 4)
        ref = [2.74346804, 3.10082138, 6.87321246, 12.8332282, 14.30085068, 14.61913328]
        self.assertAlmostEqual(abs(es - ref[:5]).max(), 0, 4)

    def test_ab_hf(self):
        mf = mf_uhf
        a, b = tdscf.TDDFT(mf_uhf).get_ab()
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

    def test_ab_mgga(self):
        mf = mf_m06l
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
        mf = mf_uhf
        td = tdscf.TDA(mf).run()
        w, nto = td.get_nto(state=1)
        self.assertAlmostEqual(w[0][0], 0.00018520143461015, 7)
        self.assertAlmostEqual(w[1][0], 0.99963372674044326, 7)
        self.assertAlmostEqual(lib.fp(w[0]), 0.00027305600430816, 7)
        self.assertAlmostEqual(lib.fp(w[1]), 0.99964370569529093, 7)

        pmol = mol.copy(deep=False)
        pmol.symmetry = True
        pmol.build(0, 0)
        mf = scf.UHF(pmol).run()
        td = tdscf.TDA(mf).run(nstates=3)
        w, nto = td.get_nto(state=0)
        self.assertAlmostEqual(w[0][0], 0.00018520143461016, 7)
        self.assertAlmostEqual(w[1][0], 0.99963372674044326, 7)
        self.assertAlmostEqual(lib.fp(w[0]), 0.00027305600430816, 7)
        self.assertAlmostEqual(lib.fp(w[1]), 0.99964370569529093, 7)

        w, nto = td.get_nto(state=-1)
        self.assertAlmostEqual(w[0][0], 0.00236940007134660, 7)
        self.assertAlmostEqual(w[1][0], 0.99759687228056182, 7)

    def test_analyze(self):
        f = td_hf.oscillator_strength(gauge='length')
        self.assertAlmostEqual(lib.fp(f), 0.16147450863004867, 5)
        f = td_hf.oscillator_strength(gauge='velocity', order=2)
        self.assertAlmostEqual(lib.fp(f), 0.19750347627735745, 5)

        note_args = []
        def temp_logger_note(rec, msg, *args):
            note_args.append(args)
        with lib.temporary_env(lib.logger.Logger, note=temp_logger_note):
            td_hf.analyze()
        ref = [(),
               (1, 2.057393297642004, 602.62734, 0.1605980834206071),
               (2, 2.2806597448158272, 543.63317, 0.0016221163442707552),
               (3, 6.372445278065303, 194.56302, 0)]
        self.assertAlmostEqual(abs(numpy.hstack(ref) -
                                   numpy.hstack(note_args)).max(), 0, 4)

    def test_init(self):
        hf = scf.UHF(mol)
        ks = scf.UKS(mol)
        kshf = scf.UKS(mol).set(xc='HF')

        self.assertTrue(isinstance(tdscf.TDA(hf), tdscf.uhf.TDA))
        self.assertTrue(isinstance(tdscf.TDA(ks), tdscf.uks.TDA))
        self.assertTrue(isinstance(tdscf.TDA(kshf), tdscf.uks.TDA))

        self.assertTrue(isinstance(tdscf.RPA(hf), tdscf.uhf.TDHF))
        self.assertTrue(isinstance(tdscf.RPA(ks), tdscf.uks.TDDFTNoHybrid))
        self.assertTrue(isinstance(tdscf.RPA(kshf), tdscf.uks.TDDFT))

        self.assertTrue(isinstance(tdscf.TDDFT(hf), tdscf.uhf.TDHF))
        self.assertTrue(isinstance(tdscf.TDDFT(ks), tdscf.uks.TDDFTNoHybrid))
        self.assertTrue(isinstance(tdscf.TDDFT(kshf), tdscf.uks.TDDFT))

        self.assertRaises(RuntimeError, tdscf.uks.dRPA, hf)
        self.assertTrue(isinstance(tdscf.dRPA(kshf), tdscf.uks.dRPA))
        self.assertTrue(isinstance(tdscf.dRPA(ks), tdscf.uks.dRPA))

        self.assertRaises(RuntimeError, tdscf.uks.dTDA, hf)
        self.assertTrue(isinstance(tdscf.dTDA(kshf), tdscf.uks.dTDA))
        self.assertTrue(isinstance(tdscf.dTDA(ks), tdscf.uks.dTDA))

    def test_tda_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build(0, 0)

        mf = dft.UKS(pmol).run()
        td = tdscf.uks.TDA(mf)
        td.wfnsym = 'B2'
        es = td.kernel(nstates=3)[0]
        self.assertAlmostEqual(lib.fp(es), 0.16350926466999033, 6)
        td.analyze()

    def test_tdhf_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build()

        mf = scf.UHF(pmol).run()
        td = tdscf.uhf.TDHF(mf)
        td.wfnsym = 'B2'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.fp(es), 0.11306948533259675, 6)

        note_args = []
        def temp_logger_note(rec, msg, *args):
            note_args.append(args)
        with lib.temporary_env(lib.logger.Logger, note=temp_logger_note):
            td.analyze()
        ref = [(),
               (1, 'B2', 2.0573933276026657, 602.62734, 0.1605980714821934),
               (2, 'B2', 14.851066559488304, 83.48505, 0.001928664835262468),
               (3, 'B2', 16.832235179166293, 73.65878, 0.17021505486468672)]
        self.assertEqual(note_args[1][1], 'B2')
        self.assertEqual(note_args[2][1], 'B2')
        self.assertEqual(note_args[3][1], 'B2')
        self.assertAlmostEqual(abs(numpy.hstack((ref[1][2:], ref[2][2:], ref[3][2:])) -
                                   numpy.hstack((note_args[1][2:], note_args[2][2:], note_args[3][2:]))).max(),
                               0, 4)

    def test_tddft_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build()

        mf = dft.UKS(pmol).run()
        td = tdscf.uks.CasidaTDDFT(mf)
        td.wfnsym = 'B2'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.fp(es), 0.15403661700414412, 6)
        td.analyze()

if __name__ == "__main__":
    print("Full Tests for TD-UKS")
    unittest.main()
