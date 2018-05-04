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

    def test_tda_lda_triplet(self):
        td = rks.TDA(mf_lda)
        td.singlet = False
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.finger(es), -39.988118769202416, 6)

    def test_ab_hf(self):
        mf = scf.RHF(mol).run()
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
        a, b = rhf.get_ab(mf)
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
        w, nto = td.get_nto(state=1)
        self.assertAlmostEqual(w[0], 0.99997335352278072, 9)
        self.assertAlmostEqual(lib.finger(w), 0.99998775067586554, 9)

        pmol = copy.copy(mol)
        pmol.symmetry = True
        pmol.build(0, 0)
        mf = scf.RHF(mol).run()
        td = rks.TDA(mf).run()
        w, nto = td.get_nto(state=0)
        self.assertAlmostEqual(w[0], 0.99997335352278072, 9)
        self.assertAlmostEqual(lib.finger(w), 0.99998775067586554, 9)

    def test_analyze(self):
        mf = scf.RHF(mol).run()
        td = tdscf.TDHF(mf).run()
        f = td.oscillator_strength(gauge='length')
        self.assertAlmostEqual(lib.finger(f), -0.13908774016795605, 7)
        f = td.oscillator_strength(gauge='velocity', order=2)
        self.assertAlmostEqual(lib.finger(f), -0.096991134490587522, 7)
        td.analyze()


if __name__ == "__main__":
    print("Full Tests for TD-RKS")
    unittest.main()
