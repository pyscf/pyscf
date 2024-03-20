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
from pyscf import tdscf

def diagonalize(a, b, nroots=4,extype=0):
    a_b2a, a_a2b = a
    b_b2a, b_a2b = b
    
    nocc_b,nvir_a = a_b2a.shape[:2]
    nocc_a,nvir_b = a_a2b.shape[:2]
    
    a_b2a = a_b2a.reshape((nocc_b*nvir_a,nocc_b*nvir_a))
    a_a2b = a_a2b.reshape((nocc_a*nvir_b,nocc_a*nvir_b))
    b_b2a = b_b2a.reshape((nocc_b*nvir_a,nocc_a*nvir_b))
    b_a2b = b_a2b.reshape((nocc_a*nvir_b,nocc_b*nvir_a))
    
    if extype == 0:
        tdm = numpy.block([[ a_b2a  , b_b2a],
                           [-b_a2b, -a_a2b]])
    elif extype == 1:
        tdm = numpy.block([[ a_a2b  , b_a2b],
                           [-b_b2a, -a_b2a]])
    
    e = numpy.linalg.eig(tdm)[0]
    lowest_e = numpy.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e

def setUpModule():
    global mol, mf_lda, mf_bp86, mf_b3lyp, mf_tpss
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

    mf_lda = dft.UKS(mol).set(xc='lda', conv_tol=1e-12)
    mf_lda.grids.prune = None
    mf_lda = mf_lda.newton().run()
    mf_bp86 = dft.UKS(mol).set(xc='b88,p86', conv_tol=1e-12)
    mf_bp86.grids.prune = None
    mf_bp86 = mf_bp86.newton().run()
    mf_b3lyp = dft.UKS(mol).set(xc='b3lyp', conv_tol=1e-12)
    mf_b3lyp.grids.prune = None
    mf_b3lyp = mf_b3lyp.newton().run()
    mf_tpss = dft.UKS(mol).set(xc='tpss', conv_tol=1e-12)
    mf_tpss.grids.prune = None
    mf_tpss = mf_tpss.newton().run()

def tearDownModule():
    global mol, mf_lda, mf_bp86, mf_b3lyp, mf_tpss
    mol.stdout.close()
    del mol, mf_lda, mf_bp86, mf_b3lyp, mf_tpss


class KnownValues(unittest.TestCase):
    def test_lda(self):
        td = tdscf.uks_sf.CasidaTDDFT(mf_lda).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 7.948103693468636, 4)
        
        td = tdscf.uks_sf.CasidaTDDFT(mf_lda).set(conv_tol=1e-12)
        es = td.kernel(nstates=4)[0]
        td.extype = 1
        td.collinear_samples = 200
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), -0.6779466524705806, 4)

    def test_b88p86(self):
        td = mf_bp86.TDDFT_SF().set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 8.368198437727425, 4)
        
        td = mf_bp86.TDDFT_SF().set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), -0.6642766007127244, 4)

    def test_tddft_b3lyp(self):
        td = tdscf.uks_sf.TDDFT_SF(mf_b3lyp).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), 8.310118087917216, 4)
        
        td = tdscf.uks_sf.TDDFT_SF(mf_b3lyp).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), 0.7265669784640787, 4)
        
    def test_tddft_tpss(self):
        td = tdscf.TDDFT_SF(mf_tpss).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), 8.00109000769877, 4)
        
        td = tdscf.TDDFT_SF(mf_tpss).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab_sf()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), -0.6715042543955895, 4)
        
    
    def test_tda_lda(self):
        td = tdscf.TDA_SF(mf_lda).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), 7.957676833140654, 4)
        ref = [0.41866563, 0.54355698, 1.00904681, 1.02421774, 1.03323334]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)
        
        td = tdscf.TDA_SF(mf_lda).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), -0.6432887108850486, 4)
        ref = [0.00054127, 0.02671484, 0.09279362, 0.09346453, 0.24262985]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)
        
    def test_tda_bp86(self):
        td = tdscf.uks_sf.TDA_SF(mf_lda).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), 8.85194990513331, 4)
        ref = [0.44926303, 0.57473835, 1.04408458, 1.05905184, 1.06439505]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)
        
        td = tdscf.uks_sf.TDA_SF(mf_lda).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), -0.6430950096838526, 4)
        ref = [4.15172344e-04, 1.92481652e-02, 8.27791805e-02, 9.40247282e-02, 4.74210269e-01]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)
        
    def test_tda_b3lyp(self):
        td = tdscf.TDA_SF(mf_b3lyp).set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), 8.924552911104696, 4)
        ref = [0.45941292, 0.57799581, 1.06629258, 1.06747435, 1.06770292]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)
        
        td = tdscf.TDA_SF(mf_b3lyp).set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0] 
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), -0.6637971023605909, 4)
        ref = [0.000670008, 0.019562623, 0.085627994, 0.090479791, 0.23541800]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 4)

    def test_tda_tpss(self):
        td = mf_tpss.TDA_SF().set(conv_tol=1e-12)
        td.extype = 0
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0] 
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), 8.692299578197659, 4)
        ref = [0.44986526, 0.57071859, 1.05441118, 1.07853214, 1.08234770]
        self.assertAlmostEqual(abs(es[:4] - ref[:4]).max(), 0, 4)
        
        td = mf_tpss.TDA_SF().set(conv_tol=1e-12)
        td.extype = 1
        td.collinear_samples = 200
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(lib.fp(es[:3]* 27.2114), -0.6411392658997148, 4)
        ref = [0.00063663, 0.02329287, 0.08839006, 0.10966013, 0.25380749]
        self.assertAlmostEqual(abs(es[:4] - ref[:4]).max(), 0, 4)

    def test_ab_lda(self):
        mf = mf_lda
        a, b = tdscf.TDDFT_SF(mf).get_ab_sf()
        ftda_sfu = tdscf.uhf.gen_tda_operation_sf(mf,extype=0)[0]
        # ftda_sfd = tdscf.uhf.gen_tda_operation_sf(mf,extype=1)[0]

        nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
        nvir_a = numpy.count_nonzero(mf.mo_occ[0] == 0)
        nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)
        nvir_b = numpy.count_nonzero(mf.mo_occ[1] == 0)
        
        numpy.random.seed(2)
        xb2a = numpy.random.random((nocc_b,nvir_a))
        ya2b = numpy.random.random((nocc_a,nvir_b))

        ax_b2a = numpy.einsum('iajb,jb->ia', a[0], xb2a)
        ax_b2a+= numpy.einsum('iajb,jb->ia', b[0], ya2b)
        ax_a2b =-numpy.einsum('iajb,jb->ia', b[1], xb2a)
        ax_a2b-= numpy.einsum('iajb,jb->ia', a[1], ya2b)
        
        xy_sfu = numpy.hstack((xb2a.ravel(), ya2b.ravel()))
        # xy_sfd = numpy.hstack((xa2b.ravel(), yb2a.ravel()))     

        ax = numpy.hstack((ax_b2a.ravel(), -ax_a2b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda_sfu(xy_sfu)).max(), 0, 9)
        
    def test_ab_bp86(self):
        mf = mf_bp86
        a, b = tdscf.TDDFT_SF(mf).get_ab_sf()
        ftda_sfu = tdscf.uhf.gen_tda_operation_sf(mf,extype=0)[0]
        # ftda_sfd = tdscf.uhf.gen_tda_operation_sf(mf,extype=1)[0]
        
        nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
        nvir_a = numpy.count_nonzero(mf.mo_occ[0] == 0)
        nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)
        nvir_b = numpy.count_nonzero(mf.mo_occ[1] == 0)
        
        numpy.random.seed(2)
        xb2a = numpy.random.random((nocc_b,nvir_a))
        ya2b = numpy.random.random((nocc_a,nvir_b))

        ax_b2a = numpy.einsum('iajb,jb->ia', a[0], xb2a)
        ax_b2a+= numpy.einsum('iajb,jb->ia', b[0], ya2b)
        ax_a2b =-numpy.einsum('iajb,jb->ia', b[1], xb2a)
        ax_a2b-= numpy.einsum('iajb,jb->ia', a[1], ya2b)
        
        xy_sfu = numpy.hstack((xb2a.ravel(), ya2b.ravel()))
        # xy_sfd = numpy.hstack((xa2b.ravel(), yb2a.ravel()))     

        ax = numpy.hstack((ax_b2a.ravel(), -ax_a2b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda_sfu(xy_sfu)).max(), 0, 9)

    def test_ab_b3lyp(self):
        mf = mf_b3lyp
        a, b = tdscf.TDDFT_SF(mf).get_ab_sf()
        ftda_sfu = tdscf.uhf.gen_tda_operation_sf(mf,extype=0)[0]
        # ftda_sfd = tdscf.uhf.gen_tda_operation_sf(mf,extype=1)[0]
        
        nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
        nvir_a = numpy.count_nonzero(mf.mo_occ[0] == 0)
        nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)
        nvir_b = numpy.count_nonzero(mf.mo_occ[1] == 0)
        
        numpy.random.seed(2)
        xb2a = numpy.random.random((nocc_b,nvir_a))
        ya2b = numpy.random.random((nocc_a,nvir_b))

        ax_b2a = numpy.einsum('iajb,jb->ia', a[0], xb2a)
        ax_b2a+= numpy.einsum('iajb,jb->ia', b[0], ya2b)
        ax_a2b =-numpy.einsum('iajb,jb->ia', b[1], xb2a)
        ax_a2b-= numpy.einsum('iajb,jb->ia', a[1], ya2b)
        
        xy_sfu = numpy.hstack((xb2a.ravel(), ya2b.ravel()))
        # xy_sfd = numpy.hstack((xa2b.ravel(), yb2a.ravel()))     

        ax = numpy.hstack((ax_b2a.ravel(), -ax_a2b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda_sfu(xy_sfu)).max(), 0, 9)

    def test_ab_mgga(self):
        mf = mf_tpss
        a, b = tdscf.TDDFT_SF(mf).get_ab_sf()
        ftda_sfu = tdscf.uhf.gen_tda_operation_sf(mf,extype=0)[0]
        # ftda_sfd = tdscf.uhf.gen_tda_operation_sf(mf,extype=1)[0]
        
        nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
        nvir_a = numpy.count_nonzero(mf.mo_occ[0] == 0)
        nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)
        nvir_b = numpy.count_nonzero(mf.mo_occ[1] == 0)
        
        numpy.random.seed(2)
        xb2a = numpy.random.random((nocc_b,nvir_a))
        ya2b = numpy.random.random((nocc_a,nvir_b))

        ax_b2a = numpy.einsum('iajb,jb->ia', a[0], xb2a)
        ax_b2a+= numpy.einsum('iajb,jb->ia', b[0], ya2b)
        ax_a2b =-numpy.einsum('iajb,jb->ia', b[1], xb2a)
        ax_a2b-= numpy.einsum('iajb,jb->ia', a[1], ya2b)
        
        xy_sfu = numpy.hstack((xb2a.ravel(), ya2b.ravel()))
        # xy_sfd = numpy.hstack((xa2b.ravel(), yb2a.ravel()))     

        ax = numpy.hstack((ax_b2a.ravel(), -ax_a2b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda_sfu(xy_sfu)).max(), 0, 9)
        
    def test_init(self):
        ks = scf.UKS(mol)
        self.assertTrue(isinstance(tdscf.TDA_SF(ks), tdscf.uks_sf.TDA_SF))
        self.assertTrue(isinstance(tdscf.TDDFT_SF(ks), tdscf.uks_sf.CasidaTDDFT))


if __name__ == "__main__":
    print("Full Tests for SF-TD-UKS")
    unittest.main()
