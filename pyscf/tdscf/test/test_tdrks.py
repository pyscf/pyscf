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
from pyscf import gto, scf, dft
from pyscf.tdscf import rhf, rks

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['H' , (0. , 0. , .917)],
    ['F' , (0. , 0. , 0.)], ]
mol.basis = '631g'
mol.build()

def finger(a):
    w = numpy.cos(numpy.arange(len(a)))
    return numpy.dot(w, a)

class KnownValues(unittest.TestCase):
    def test_nohbrid_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda, vwn_rpa'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFTNoHybrid(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.059050077236151, 6)

    def test_nohbrid_b88p86(self):
        mf = dft.RKS(mol)
        mf.xc = 'b88,p86'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFTNoHybrid(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -40.462005239920558, 6)

    def test_tddft_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda, vwn_rpa'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.059050077236151, 6)

    def test_tddft_b88p86(self):
        mf = dft.RKS(mol)
        mf.xc = 'b88,p86'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -40.462005239920558, 6)

    def test_tddft_b3pw91(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3pw91'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.218912874291014, 6)

    def test_tddft_b3lyp(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.29609453661341, 6)

    def test_tda_b3lypg(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lypg'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.385520327568869, 6)

    def test_tda_b3pw91(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3pw91'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.313632163628363, 6)

    def test_tda_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -41.201828219760415, 6)

#NOTE b3lyp by libxc is quite different to b3lyp from xcfun
    def test_tddft_b3lyp_xcfun(self):
        dft.numint._NumInt.libxc = dft.xcfun
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDDFT(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        dft.numint._NumInt.libxc = dft.libxc
        self.assertAlmostEqual(finger(es), finger([9.88975514, 9.88975514, 15.16643994, 30.55289462, 30.55289462]), 6)

    def test_tddft_b3lyp_xcfun(self):
        dft.numint._NumInt.libxc = dft.xcfun
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        dft.numint._NumInt.libxc = dft.libxc
        self.assertAlmostEqual(finger(es), -41.393122257109056, 6)

    def test_tda_lda_xcfun(self):
        dft.numint._NumInt.libxc = dft.xcfun
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        dft.numint._NumInt.libxc = dft.libxc
        self.assertAlmostEqual(finger(es), -41.201828219760415, 6)

    def test_tda_b3lyp_triplet(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.singlet = False
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -40.020204585289648, 6)

    def test_tda_lda_triplet(self):
        mf = dft.RKS(mol)
        mf.xc = 'lda,vwn'
        mf.grids.prune = None
        mf.scf()
        td = rks.TDA(mf)
        td.singlet = False
        td.nstates = 5
        es = td.kernel()[0] * 27.2114
        self.assertAlmostEqual(finger(es), -39.988118769202416, 6)

    def test_ab_hf(self):
        mf = scf.RHF(mol).run()
        a, b = rhf.get_ab(mf)
        ftda = rhf.gen_tda_operation(mf, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nvir,nocc))
        ax = numpy.einsum('iajb,bj->ai', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nvir,nocc)).max(), 0, 9)

        ab1 = ax + numpy.einsum('iajb,bj->ai', b, y)
        ab2 =-numpy.einsum('iajb,bj->ai', b, x)
        ab2-= numpy.einsum('iajb,bj->ai', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nvir,nocc)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_lda(self):
        mf = dft.RKS(mol).run(xc='lda,vwn')
        a, b = rhf.get_ab(mf)
        ftda = rhf.gen_tda_operation(mf, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nvir,nocc))
        ax = numpy.einsum('iajb,bj->ai', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nvir,nocc)).max(), 0, 9)

        ab1 = ax + numpy.einsum('iajb,bj->ai', b, y)
        ab2 =-numpy.einsum('iajb,bj->ai', b, x)
        ab2-= numpy.einsum('iajb,bj->ai', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nvir,nocc)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_b3lyp(self):
        mf = dft.RKS(mol).run(xc='b3lyp')
        a, b = rhf.get_ab(mf)
        ftda = rhf.gen_tda_operation(mf, singlet=None)[0]
        ftdhf = rhf.gen_tdhf_operation(mf, singlet=True)[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 2)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nvir,nocc))
        ax = numpy.einsum('iajb,bj->ai', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nvir,nocc)).max(), 0, 9)

        ab1 = ax + numpy.einsum('iajb,bj->ai', b, y)
        ab2 =-numpy.einsum('iajb,bj->ai', b, x)
        ab2-= numpy.einsum('iajb,bj->ai', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nvir,nocc)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for TD-RKS")
    unittest.main()
