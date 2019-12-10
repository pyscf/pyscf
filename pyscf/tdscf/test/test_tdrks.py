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
from pyscf.tdscf import rhf, rks
from pyscf import tdscf

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom = [
    ['H' , (0. , 0. , .917)],
    ['F' , (0. , 0. , 0.)], ]
mol.basis = '631g'
mol.build()

mf = scf.RHF(mol).run()
td_hf = tdscf.TDHF(mf).run()

mf_lda3 = dft.RKS(mol)
mf_lda3.xc = 'lda, vwn_rpa'
mf_lda3.grids.prune = None
mf_lda3.scf()

mf_lda = dft.RKS(mol)
mf_lda.xc = 'lda, vwn'
mf_lda.grids.prune = None
mf_lda.scf()

mf_bp86 = dft.RKS(mol)
mf_bp86.xc = 'b88,p86'
mf_bp86.grids.prune = None
mf_bp86.scf()

mf_b3lyp = dft.RKS(mol)
mf_b3lyp.xc = 'b3lyp'
mf_b3lyp.grids.prune = None
mf_b3lyp.scf()

mf_b3lyp1 = dft.RKS(mol)
mf_b3lyp1.xc = 'b3lyp'
mf_b3lyp1.grids.prune = None
mf_b3lyp1._numint.libxc = dft.xcfun
mf_b3lyp1.scf()

mf_b3pw91g = dft.RKS(mol)
mf_b3pw91g.xc = 'b3pw91g'
mf_b3pw91g.grids.prune = None
mf_b3pw91g.scf()

def tearDownModule():
    global mol, mf, td_hf, mf_lda3, mf_lda, mf_bp86, mf_b3lyp, mf_b3lyp1, mf_b3pw91g
    mol.stdout.close()
    del mol, mf, td_hf, mf_lda3, mf_lda, mf_bp86, mf_b3lyp, mf_b3lyp1, mf_b3pw91g

class KnownValues(unittest.TestCase):
    def test_nohbrid_lda(self):
        td = rks.TDDFTNoHybrid(mf_lda3)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -41.059050077236151, 6)

    def test_nohbrid_b88p86(self):
        td = rks.TDDFTNoHybrid(mf_bp86)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -40.462005239920558, 6)

    def test_tddft_lda(self):
        td = rks.TDDFT(mf_lda3)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -41.059050077236151, 6)

    def test_tddft_b88p86(self):
        td = rks.TDDFT(mf_bp86)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -40.462005239920558, 6)

    #def test_tddft_b3pw91g(self):
    #    td = rks.TDDFT(mf_b3pw91g)
    #    es = td.kernel(nstates=5)[0] * 27.2114
    #    self.assertAlmostEqual(lib.finger(es), -41.218912874291014, 6)

    def test_tddft_b3lyp(self):
        td = rks.TDDFT(mf_b3lyp)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -41.29609453661341, 6)

    def test_tda_b3lypg(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lypg'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -41.385520327568869, 6)

    #def test_tda_b3pw91g(self):
    #    td = rks.TDA(mf_b3pw91g)
    #    es = td.kernel(nstates=5)[0] * 27.2114
    #    self.assertAlmostEqual(lib.finger(es), -41.313632163628363, 6)

    def test_tda_lda(self):
        td = rks.TDA(mf_lda)
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -41.201828219760415, 6)

    def test_tddft_b3lyp_xcfun(self):
        td = rks.TDDFT(mf_b3lyp1)
        es = td.kernel(nstates=5)[0] * 27.2114
        dft.numint.NumInt.libxc = dft.libxc
        self.assertAlmostEqual(abs(es - [9.88975514, 9.88975514, 15.16643994, 30.55289462, 30.55289462]).max(), 0, 6)

    def test_tda_b3lyp_xcfun(self):
        td = rks.TDA(mf_b3lyp1)
        es = td.kernel(nstates=5)[0] * 27.2114
        dft.numint.NumInt.libxc = dft.libxc
        self.assertAlmostEqual(lib.finger(es), -41.393122257109056, 6)

    def test_tda_lda_xcfun(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf._numint.libxc = dft.xcfun
        mf.scf()
        td = rks.TDA(mf)
        es = td.kernel(nstates=5)[0] * 27.2114
        dft.numint.NumInt.libxc = dft.libxc
        self.assertAlmostEqual(lib.finger(es), -41.201828219760415, 6)

    def test_tda_b3lyp_triplet(self):
        td = rks.TDA(mf_b3lyp)
        td.singlet = False
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -40.020204585289648, 6)
        td.analyze()

    def test_tda_lda_triplet(self):
        td = rks.TDA(mf_lda)
        td.singlet = False
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -39.988118769202416, 6)

    def test_ab_hf(self):
        mf = scf.RHF(mol).run()
        a, b = rhf.get_ab(mf)
        fock = mf.get_hcore() + mf.get_veff()
        ftda = rhf.gen_tda_operation(mf, fock, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nocc,nvir))
        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 6)

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

    def test_nto(self):
        mf = scf.RHF(mol).run()
        td = rks.TDA(mf).run()
        w, nto = td.get_nto(state=3)
        self.assertAlmostEqual(w[0], 0.98655300613468389, 9)
        self.assertAlmostEqual(lib.finger(w), 0.98625701534112464, 9)

        w, nto = td.get_nto(state=0)
        self.assertAlmostEqual(w[0], 0.99997335352278072, 9)
        self.assertAlmostEqual(lib.finger(w), 0.99998775067586554, 9)

        pmol = copy.copy(mol)
        pmol.symmetry = True
        pmol.build(0, 0)
        mf = scf.RHF(pmol).run()
        td = rks.TDA(mf).run(nstates=3)
        w, nto = td.get_nto(state=-1)
        self.assertAlmostEqual(w[0], 0.98655300613468389, 9)
        self.assertAlmostEqual(lib.finger(w), 0.98625701534112464, 9)

    def test_analyze(self):
        f = td_hf.oscillator_strength(gauge='length')
        self.assertAlmostEqual(lib.finger(f), -0.13908774016795605, 7)
        f = td_hf.oscillator_strength(gauge='velocity', order=2)
        self.assertAlmostEqual(lib.finger(f), -0.096991134490587522, 7)

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
                                   numpy.hstack(note_args)).max(), 0, 7)

        self.assertEqual(td_hf.nroots, td_hf.nstates)
        self.assertAlmostEqual(lib.finger(td_hf.e_tot-mf.e_tot), 0.41508325757603637, 6)

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

        self.assertRaises(RuntimeError, tdscf.dRPA, hf)
        self.assertTrue(isinstance(tdscf.dRPA(kshf), tdscf.rks.dRPA))
        self.assertTrue(isinstance(tdscf.dRPA(ks), tdscf.rks.dRPA))

        self.assertRaises(RuntimeError, tdscf.dTDA, hf)
        self.assertTrue(isinstance(tdscf.dTDA(kshf), tdscf.rks.dTDA))
        self.assertTrue(isinstance(tdscf.dTDA(ks), tdscf.rks.dTDA))

    def test_tda_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build(0, 0)

        mf = dft.RKS(pmol).run()
        td = rks.TDA(mf)
        td.wfnsym = 'A2'
        es = td.kernel(nstates=3)[0]
        self.assertTrue(len(es) == 2)  # At most 2 states due to symmetry subspace size
        self.assertAlmostEqual(lib.finger(es), 2.1857694738741071, 6)

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
                               0, 7)

    def test_tdhf_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build()

        mf = scf.RHF(pmol).run()
        td = rhf.TDHF(mf)
        td.wfnsym = 'A2'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.finger(es), 2.2541287466157165, 6)
        td.analyze()

    def test_tddft_with_wfnsym(self):
        pmol = mol.copy()
        pmol.symmetry = True
        pmol.build()

        mf = dft.RKS(pmol).run()
        td = rks.TDDFTNoHybrid(mf)
        td.wfnsym = 'A2'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertTrue(len(es) == 2)  # At most 2 states due to symmetry subspace size
        self.assertAlmostEqual(lib.finger(es), 2.1856920990871753, 6)
        td.analyze()

    def test_scanner(self):
        td_scan = td_hf.as_scanner().as_scanner()
        td_scan.nroots = 3
        td_scan(mol)
        self.assertAlmostEqual(lib.finger(td_scan.e), 0.41508325757603637, 6)

    def test_transition_multipoles(self):
        self.assertAlmostEqual(abs(lib.finger(td_hf.transition_dipole()             [2])), 0.39833021312014988, 5)
        self.assertAlmostEqual(abs(lib.finger(td_hf.transition_quadrupole()         [2])), 0.14862776196563565, 5)
        self.assertAlmostEqual(abs(lib.finger(td_hf.transition_octupole()           [2])), 2.79058994496489410, 5)
        self.assertAlmostEqual(abs(lib.finger(td_hf.transition_velocity_dipole()    [2])), 0.24021409469918567, 5)
        self.assertAlmostEqual(abs(lib.finger(td_hf.transition_magnetic_dipole()    [2])), 0                  , 5)
        self.assertAlmostEqual(abs(lib.finger(td_hf.transition_magnetic_quadrupole()[2])), 0.16558596265719450, 5)

    def test_dRPA(self):
        td = rks.dRPA(mf_lda)
        td._scf.xc = ''
        es = td.kernel(nstates=3)[0]
        self.assertAlmostEqual(lib.finger(es), 0.32727702719009616, 6)

    def test_dTDA(self):
        td = rks.dTDA(mf_lda)
        td._scf.xc = ''
        es = td.kernel(nstates=3)[0]
        self.assertAlmostEqual(lib.finger(es), 0.3237948650800024, 6)

        td = rks.dTDA(mf_lda)
        es = td.kernel(nstates=3)[0]
        self.assertAlmostEqual(lib.finger(es), 0.3237948650800024, 6)


if __name__ == "__main__":
    print("Full Tests for TD-RKS")
    unittest.main()
