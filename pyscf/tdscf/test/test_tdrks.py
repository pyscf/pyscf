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
from pyscf import lib, gto, scf, dft
from pyscf.tdscf import rhf, rks
from pyscf import tdscf
from pyscf.data import nist

has_xcfun = hasattr(dft, 'xcfun')

def setUpModule():
    global mol, mf, td_hf, mf_lda, mf_bp86, mf_b3lyp, mf_m06l
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol).run()
    td_hf = tdscf.TDHF(mf).run(conv_tol=1e-6)

    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mf_lda = dft.RKS(mol)
        mf_lda.xc = 'lda, vwn'
        mf_lda.grids.prune = None
        mf_lda.run(conv_tol=1e-10)

        mf_bp86 = dft.RKS(mol)
        mf_bp86.xc = 'b88,p86'
        mf_bp86.grids.prune = None
        mf_bp86.run(conv_tol=1e-10)

        mf_b3lyp = dft.RKS(mol)
        mf_b3lyp.xc = 'b3lyp5'
        mf_b3lyp.grids.prune = None
        mf_b3lyp.run(conv_tol=1e-10)

        mf_m06l = dft.RKS(mol).run(xc='m06l', conv_tol=1e-10)

def tearDownModule():
    global mol, mf, td_hf, mf_lda, mf_bp86, mf_b3lyp, mf_m06l
    mol.stdout.close()
    del mol, mf, td_hf, mf_lda, mf_bp86, mf_b3lyp, mf_m06l

def diagonalize(a, b, nroots=4):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = numpy.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e = numpy.linalg.eig(numpy.asarray(h))[0]
    lowest_e = numpy.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_nohbrid_lda(self):
        td = rks.CasidaTDDFT(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -41.100806721759945, 5)
        ref = [9.67249402,  9.67249402, 14.79447862, 30.32465371, 30.32465371]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 5)

    def test_nohbrid_b88p86(self):
        td = rks.CasidaTDDFT(mf_bp86)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -40.4619799852133, 6)
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=5) * 27.2114
        self.assertAlmostEqual(abs(es - ref).max(), 0, 6)

    def test_tddft_lda(self):
        td = rks.TDDFT(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -41.100806721759945, 5)

    def test_tddft_b88p86(self):
        td = rks.TDDFT(mf_bp86)
        td.conv_tol = 1e-5
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -40.4619799852133, 6)

    def test_tddft_b3lyp(self):
        td = rks.TDDFT(mf_b3lyp)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -41.29609453661341, 4)
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=5) * 27.2114
        self.assertAlmostEqual(abs(es - ref).max(), 0, 6)

    def test_tddft_camb3lyp(self):
        mf = mol.RKS(xc='camb3lyp').run()
        td = mf.TDDFT()
        td.conv_tol = 1e-5
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 9.0054057603534, 4)

    def test_tddft_hse06(self):
        mf = mol.RKS(xc='hse06').run()
        td = mf.TDDFT()
        td.conv_tol = 1e-5
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.34554352587555387, 6)

    def test_tda_b3lypg(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lypg'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -41.385520327568869, 4)

    def test_tda_lda(self):
        td = rks.TDA(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -41.201828219760415, 4)

    @unittest.skipIf(has_xcfun, "xcfun library not found.")
    def test_tddft_b3lyp_xcfun(self):
        with lib.temporary_env(dft.numint.NumInt, libxc=dft.xcfun):
            td = rks.TDDFT(mf_b3lyp)
            es = td.kernel(nstates=5)[0] * 27.2114
        ref = [9.88975514, 9.88975514, 15.16643994, 30.55289462, 30.55289462]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 5)

    @unittest.skipIf(has_xcfun, "xcfun library not found.")
    def test_tda_b3lyp_xcfun(self):
        with lib.temporary_env(dft.numint.NumInt, libxc=dft.xcfun):
            td = rks.TDA(mf_b3lyp)
            es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -41.393122257109056, 5)

    @unittest.skipIf(has_xcfun, "xcfun library not found.")
    def test_tda_lda_xcfun(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf._numint.libxc = dft.xcfun
        mf.scf()
        td = rks.TDA(mf)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -41.201828219760415, 5)
        ref = [9.68872769,  9.68872769, 15.07122478]
        self.assertAlmostEqual(abs(es[:3] - ref).max(), 0, 5)

    def test_tda_b3lyp_triplet(self):
        td = rks.TDA(mf_b3lyp)
        td.singlet = False
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -40.020204585289648, 5)
        td.analyze()

    def test_tda_lda_triplet(self):
        td = rks.TDA(mf_lda)
        td.singlet = False
        es = td.kernel(nstates=6)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[[0,1,2,4,5]]), -39.988118769202416, 5)
        ref = [9.0139312, 9.0139312, 12.42444659, 29.38040677, 29.63058493, 29.63058493]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

    def test_tddft_b88p86_triplet(self):
        td = rks.TDDFT(mf_bp86)
        td.singlet = False
        es = td.kernel(nstates=5)[0] * 27.2114
        ref = [9.09315203, 9.09315203, 12.29837275, 29.26724001, 29.26724001]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

        utd = mf_bp86.to_uks().TDDFT()
        nocc = mf_bp86.mol.nelectron//2
        nvir = mf_bp86.mo_energy.size - nocc
        nov = nocc * nvir
        shape = (nov, nov)
        a, b = utd.get_ab()
        a_aa, a_ab, a_bb = a
        b_aa, b_ab, b_bb = b
        a = numpy.block([[a_aa.reshape(shape)  ,a_ab.reshape(shape)],
                         [a_ab.reshape(shape).T,a_bb.reshape(shape)]])
        b = numpy.block([[b_aa.reshape(shape)  ,b_ab.reshape(shape)],
                         [b_ab.reshape(shape).T,b_bb.reshape(shape)]])
        h = numpy.block([[a        , b         ],
                         [-b.conj(),-a.conj()]])
        e = numpy.linalg.eig(numpy.asarray(h))[0]
        ref = numpy.sort(e[e.real>0])[[0,1,4,6,7]] * 27.2114
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

    def test_tda_rsh(self):
        mol = gto.M(atom='H 0 0 0.6; H 0 0 0', basis = "6-31g")
        mf = dft.RKS(mol)
        mf.xc = 'wb97'
        e = mf.kernel()
        self.assertAlmostEqual(e, -1.14670613191817, 8)

        e_td = mf.TDA().set(nstates=5).kernel()[0]
        ref = [0.59718453, 1.02667324, 1.81786290]
        self.assertAlmostEqual(abs(e_td - ref).max(), 0, 6)

        mf.xc = 'wb97 + 1e-9*HF'
        ref = mf.TDA().set(nstates=5).kernel()[0]
        self.assertAlmostEqual(abs(e_td - ref).max(), 0, 8)

        mf.xc = 'hse06'
        e = mf.kernel()
        self.assertAlmostEqual(e, -1.1447666793407982, 8)

        e_td = mf.TDA().set(nstates=5).kernel()[0]
        ref = [0.60314386, 1.03802565, 1.82757364]
        self.assertAlmostEqual(abs(e_td - ref).max(), 0, 6)

        mf.xc = 'hse06 + 1e-9*HF'
        ref = mf.TDA().set(nstates=5).kernel()[0]
        self.assertAlmostEqual(abs(e_td - ref).max(), 0, 8)

    def test_tda_m06l_singlet(self):
        td = mf_m06l.TDA()
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es), -42.50751841202568, 4)
        ref = [10.82698652, 10.82698652, 16.73024993]
        self.assertAlmostEqual(abs(es[:3] - ref).max(), 0, 4)

    def test_ab_hf(self):
        a, b = rhf.get_ab(mf)
        ftda = rhf.gen_tda_operation(mf, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nocc,nvir))
        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 5)

        ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
        ab2 =-numpy.einsum('iajb,jb->ia', b, x)
        ab2-= numpy.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_lda(self):
        mf = mf_lda
        a, b = rhf.get_ab(mf)
        ftda = rhf.gen_tda_operation(mf, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nocc,nvir))
        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
        ab2 =-numpy.einsum('iajb,jb->ia', b, x)
        ab2-= numpy.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_b3lyp(self):
        mf = mf_b3lyp
        a, b = rks.TDDFT(mf).get_ab()
        ftda = rhf.gen_tda_operation(mf, singlet=None)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nocc,nvir))
        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
        ab2 =-numpy.einsum('iajb,jb->ia', b, x)
        ab2-= numpy.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_mgga(self):
        mf = mf_m06l
        a, b = rks.TDDFT(mf).get_ab()
        ftda = rhf.gen_tda_operation(mf, singlet=None)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nocc,nvir))
        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
        ab2 =-numpy.einsum('iajb,jb->ia', b, x)
        ab2-= numpy.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_nto(self):
        mf = scf.RHF(mol).run()
        td = rks.TDA(mf).run(conv_tol=1e-6, nstates=5)
        w, nto = td.get_nto(state=3)
        self.assertAlmostEqual(w[0], 0.98655300613468389, 7)
        self.assertAlmostEqual(lib.fp(w), 0.98625701534112464, 7)

        w, nto = td.get_nto(state=0)
        self.assertAlmostEqual(w[0], 0.99997335352278072, 7)
        self.assertAlmostEqual(lib.fp(w), 0.99998775067586554, 7)

        pmol = mol.copy(deep=False)
        pmol.symmetry = True
        pmol.build(0, 0)
        mf = scf.RHF(pmol).run()
        td = rks.TDA(mf).run(conv_tol=1e-6, nstates=3)
        w, nto = td.get_nto(state=-1)
        self.assertAlmostEqual(w[0], 0.98655300613468389, 7)
        self.assertAlmostEqual(lib.fp(w), 0.98625701534112464, 7)

    def test_analyze(self):
        f = td_hf.oscillator_strength(gauge='length')
        self.assertAlmostEqual(lib.fp(f), -0.13908774016795605, 5)
        f = td_hf.oscillator_strength(gauge='velocity', order=2)
        self.assertAlmostEqual(lib.fp(f), -0.096991134490587522, 5)

        note_args = []
        def temp_logger_note(rec, msg, *args):
            note_args.append(args)
        with lib.temporary_env(lib.logger.Logger, note=temp_logger_note):
            td_hf.analyze()
        ref = [(),
               (1, 11.834865910142547, 104.76181013351982, 0.01075359074556743),
               (2, 11.834865910142618, 104.76181013351919, 0.010753590745567499),
               (3, 16.66308427853695, 74.40651170629978, 0.3740302871966713)]
        self.assertAlmostEqual(abs(numpy.hstack(ref) -
                                   numpy.hstack(note_args)).max(), 0, 4)

        self.assertEqual(td_hf.nroots, td_hf.nstates)
        self.assertAlmostEqual(lib.fp(td_hf.e_tot-mf.e_tot), 0.41508325757603637, 5)

    def test_init(self):
        hf = scf.RHF(mol)
        ks = scf.RKS(mol)
        kshf = scf.RKS(mol).set(xc='HF')

        self.assertTrue(isinstance(tdscf.TDA(hf), tdscf.rhf.TDA))
        self.assertTrue(isinstance(tdscf.TDA(ks), tdscf.rks.TDA))
        self.assertTrue(isinstance(tdscf.TDA(kshf), tdscf.rks.TDA))

        self.assertTrue(isinstance(tdscf.RPA(hf), tdscf.rhf.TDHF))
        self.assertTrue(isinstance(tdscf.RPA(ks), tdscf.rks.TDDFTNoHybrid))
        self.assertTrue(isinstance(tdscf.RPA(kshf), tdscf.rks.TDDFT))

        self.assertTrue(isinstance(tdscf.TDDFT(hf), tdscf.rhf.TDHF))
        self.assertTrue(isinstance(tdscf.TDDFT(ks), tdscf.rks.TDDFTNoHybrid))
        self.assertTrue(isinstance(tdscf.TDDFT(kshf), tdscf.rks.TDDFT))

        self.assertRaises(RuntimeError, tdscf.rks.dRPA, hf)
        self.assertTrue(isinstance(tdscf.dRPA(kshf), tdscf.rks.dRPA))
        self.assertTrue(isinstance(tdscf.dRPA(ks), tdscf.rks.dRPA))

        self.assertRaises(RuntimeError, tdscf.rks.dTDA, hf)
        self.assertTrue(isinstance(tdscf.dTDA(kshf), tdscf.rks.dTDA))
        self.assertTrue(isinstance(tdscf.dTDA(ks), tdscf.rks.dTDA))

        kshf.xc = ''
        self.assertTrue(isinstance(tdscf.dTDA(kshf), tdscf.rks.dTDA))
        self.assertTrue(isinstance(tdscf.dRPA(kshf), tdscf.rks.dRPA))

    def test_tda_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = 'C2v'
        pmol.build(0, 0)

        mf = dft.RKS(pmol).run()
        td = rks.TDA(mf)
        td.wfnsym = 'A2'
        es = td.kernel(nstates=3)[0]
        self.assertTrue(len(es) == 2)  # At most 2 states due to symmetry subspace size
        self.assertAlmostEqual(lib.fp(es), 2.1857694738741071, 5)

        note_args = []
        def temp_logger_note(rec, msg, *args):
            note_args.append(args)
        with lib.temporary_env(lib.logger.Logger, note=temp_logger_note):
            td.analyze()
        ref = [(),
               (1, 'A2', 38.42106241429979, 32.26985141807447, 0.0),
               (2, 'A2', 38.972172173478356, 31.813519911465608, 0.0)]
        self.assertEqual(note_args[1][1], 'A2')
        self.assertEqual(note_args[2][1], 'A2')
        self.assertAlmostEqual(abs(numpy.append(ref[1][2:], ref[2][2:]) -
                                   numpy.append(note_args[1][2:], note_args[2][2:])).max(),
                               0, 4)

    def test_tdhf_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build()

        mf = scf.RHF(pmol).run()
        td = rhf.TDHF(mf)
        td.wfnsym = 'A2'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.fp(es), 2.2541287466157165, 5)
        td.analyze()

    def test_tddft_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build()

        mf = dft.RKS(pmol).run()
        td = rks.CasidaTDDFT(mf)
        td.wfnsym = 'A2'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertTrue(len(es) == 2)  # At most 2 states due to symmetry subspace size
        self.assertAlmostEqual(lib.fp(es), 2.1856920990871753, 5)
        td.analyze()

    def test_scanner(self):
        td_scan = td_hf.as_scanner().as_scanner()
        td_scan.nroots = 3
        td_scan(mol)
        self.assertAlmostEqual(lib.fp(td_scan.e), 0.41508325757603637, 5)

    def test_transition_multipoles(self):
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_dipole()             [2])), 0.39833021312014988, 5)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_quadrupole()         [2])), 0.14862776196563565, 5)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_octupole()           [2])), 2.79058994496489410, 5)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_velocity_dipole()    [2])), 0.24021409469918567, 5)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_magnetic_dipole()    [2])), 0                  , 5)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_magnetic_quadrupole()[2])), 0.16558596265719450, 5)

    def test_dRPA(self):
        td = rks.dRPA(mf_lda)
        td._scf.xc = ''
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]), 0.32727702719009616, 5)
        ref = [10.00343861, 10.00343861, 15.62586305, 30.69238874, 30.69238874]
        self.assertAlmostEqual(abs(es * 27.2114 - ref).max(), 0, 5)

    def test_dTDA(self):
        td = rks.dTDA(mf_lda)
        td._scf.xc = ''
        es = td.kernel(nstates=3)[0]
        self.assertAlmostEqual(lib.fp(es), 0.3237948650800024, 5)

        td = rks.dTDA(mf_lda)
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]), 0.3237948650800024, 5)
        ref = [10.05245288, 10.05245288, 16.03497655, 30.7120363,  30.7120363 ]
        self.assertAlmostEqual(abs(es * 27.2114 - ref).max(), 0, 5)

    def test_reset(self):
        mol1 = gto.M(atom='C')
        td = scf.RHF(mol).newton().TDHF()
        td.reset(mol1)
        self.assertTrue(td.mol is mol1)
        self.assertTrue(td._scf.mol is mol1)
        self.assertTrue(td._scf._scf.mol is mol1)

    @unittest.skipIf(has_xcfun, "xcfun library not found.")
    def test_custom_rsh(self):
        mol = gto.M(atom='H 0 0 0.6; H 0 0 0', basis = "6-31g")
        mf = dft.RKS(mol)
        mf._numint.libxc = dft.xcfun
        mf.xc = "camb3lyp"
        mf.omega = 0.2
        e = mf.kernel()
        self.assertAlmostEqual(e, -1.143272159913611, 8)

        e_td = mf.TDDFT().kernel()[0]
        ref = [16.14837289, 28.01968627, 49.00854076]
        self.assertAlmostEqual(abs(e_td*nist.HARTREE2EV - ref).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for TD-RKS")
    unittest.main()
