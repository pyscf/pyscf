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

import os
import unittest
import numpy
from pyscf import gto, scf
from pyscf import dft
from pyscf import lib

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = 'h 0 0 0; h 1 .5 0; h 0 4 1; h 1 0 .2'
mol.basis = 'aug-ccpvdz'
mol.build()
#dm = scf.RHF(mol).run(conv_tol=1e-14).make_rdm1()
dm = numpy.load(os.path.realpath(os.path.join(__file__, '..', 'dm_h4.npy')))
mf = dft.RKS(mol)
mf.grids.atom_grid = {"H": (50, 110)}
mf.prune = None
mf.grids.build(with_non0tab=False)
nao = mol.nao_nr()
ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')

def tearDownModule():
    global mol, mf, ao, rho
    del mol, mf, ao, rho

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnownValues(unittest.TestCase):
    def test_parse_xc(self):
        hyb, fn_facs = dft.xcfun.parse_xc('.5*HF+.5*B3LYP,VWN*.5')
        self.assertAlmostEqual(hyb[0], .6, 12)
        self.assertEqual([x[0] for x in fn_facs], [0,6,16,3])
        self.assertTrue(numpy.allclose([x[1] for x in fn_facs],
                                       (0.04, 0.36, 0.405, 0.595)))
        hyb, fn_facs = dft.xcfun.parse_xc('HF,')
        self.assertEqual(hyb[0], 1)
        self.assertEqual(fn_facs, [])

        hyb, fn_facs = dft.libxc.parse_xc('B88 - SLATER')
        self.assertEqual(fn_facs, [(106, 1), (1, -1)])
        hyb, fn_facs = dft.libxc.parse_xc('B88 -SLATER*.5')
        self.assertEqual(fn_facs, [(106, 1), (1, -0.5)])

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*B3LYP+0.25*B3LYP')
        self.assertTrue(numpy.allclose(hyb, [.15, 0, 0]))
        hyb = dft.libxc.hybrid_coeff('0.5*B3LYP+0.25*B3LYP')
        self.assertAlmostEqual(hyb, .15, 12)

        hyb, fn_facs = dft.xcfun.parse_xc('CAM_B3LYP')
        self.assertTrue(numpy.allclose(hyb, [0.19, 0.65, 0.33]))

        hyb, fn_facs = dft.xcfun.parse_xc('0.6*CAM_B3LYP+0.4*B3P86')
        self.assertTrue(numpy.allclose(hyb, [.08+0.19*.6, 0.65*.6, 0.33]))
        self.assertTrue(numpy.allclose(fn_facs,
                                       [(9, 0.6), (3, 0.19), (16, 0.486), (0, 0.032), (6, 0.288), (46, 0.324)]))
        rsh = dft.xcfun.rsh_coeff('0.6*CAM_B3LYP+0.4*B3P86')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.39, -0.196)))

        hyb, fn_facs = dft.xcfun.parse_xc('0.4*B3P86+0.6*CAM_B3LYP')
        self.assertTrue(numpy.allclose(hyb, [.08+0.19*.6, 0.65*.6, 0.33]))
        self.assertTrue(numpy.allclose(fn_facs,
                                       [(0, 0.032), (6, 0.288), (46, 0.324), (3, 0.19), (9, 0.6), (16, 0.486)]))
        rsh = dft.xcfun.rsh_coeff('0.4*B3P86+0.6*CAM_B3LYP')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.39, -0.196)))

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*SR-HF(0.3) + .8*HF + .22*LR_HF')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*SR-HF + .22*LR_HF(0.3) + .8*HF')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*SR-HF + .8*HF + .22*LR_HF(0.3)')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*RSH(2.04;0.56;0.3) + 0.5*BP86')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])
        self.assertEqual(fn_facs, [(6, 0.5), (46, 0.5)])

        self.assertRaises(ValueError, dft.xcfun.parse_xc, 'SR_HF(0.3) + LR_HF(.5)')
        self.assertRaises(ValueError, dft.xcfun.parse_xc, 'LR-HF(0.3) + SR-HF(.5)')

        hyb = dft.xcfun.hybrid_coeff('M05')
        self.assertAlmostEqual(hyb, 0.28, 9)

        hyb, fn_facs = dft.xcfun.parse_xc('APBE,')
        self.assertEqual(fn_facs[0][0], 58)

        hyb, fn_facs = dft.xcfun.parse_xc('VWN,')
        self.assertEqual(fn_facs, [(3, 1)])

        hyb, fn_facs = dft.xcfun.parse_xc('TF,')
        self.assertEqual(fn_facs, [(24, 1)])

        ref = [(0, 1), (3, 1)]
        self.assertEqual(dft.xcfun.parse_xc_name('LDA,VWN'), (0,3))
        self.assertEqual(dft.xcfun.parse_xc(('LDA','VWN'))[1], ref)
        self.assertEqual(dft.xcfun.parse_xc((0, 3))[1], ref)
        self.assertEqual(dft.xcfun.parse_xc('0, 3')[1], ref)
        self.assertEqual(dft.xcfun.parse_xc(3)[1], [(3,1)])

        #self.assertEqual(dft.xcfun.parse_xc('M11-L')[1], [(226,1),(75,1)])
        #self.assertEqual(dft.xcfun.parse_xc('M11L' )[1], [(226,1),(75,1)])
        #self.assertEqual(dft.xcfun.parse_xc('M11-L,M11L' )[1], [(226,1),(75,1)])
        #self.assertEqual(dft.xcfun.parse_xc('M11_L,M11-L')[1], [(226,1),(75,1)])
        #self.assertEqual(dft.xcfun.parse_xc('M11L,M11_L' )[1], [(226,1),(75,1)])

        #self.assertEqual(dft.xcfun.parse_xc('Xpbe,')[1], [(123,1)])
        #self.assertEqual(dft.xcfun.parse_xc('pbe,' )[1], [(101,1)])
        hyb, fn_facs = dft.xcfun.parse_xc('PBE*.4+LDA')
        self.assertEqual(fn_facs, [(5, 0.4), (4, 0.4), (0, 1)])
        hyb, fn_facs = dft.xcfun.parse_xc('PBE*.4+VWN')
        self.assertEqual(fn_facs, [(5, 0.4), (4, 0.4), (3, 1)])

        self.assertTrue (dft.xcfun.is_meta_gga('m05'))
        self.assertFalse(dft.xcfun.is_meta_gga('pbe0'))
        self.assertFalse(dft.xcfun.is_meta_gga('tf,'))
        self.assertFalse(dft.xcfun.is_meta_gga('vv10'))
        self.assertTrue (dft.xcfun.is_gga('PBE0'))
        self.assertFalse(dft.xcfun.is_gga('m05'))
        self.assertFalse(dft.xcfun.is_gga('tf,'))
        self.assertTrue (dft.xcfun.is_lda('tf,'))
        self.assertFalse(dft.xcfun.is_lda('vv10'))
        self.assertTrue (dft.xcfun.is_hybrid_xc('m05'))
        self.assertTrue (dft.xcfun.is_hybrid_xc('pbe0,'))
        self.assertFalse(dft.xcfun.is_hybrid_xc('m05,'))
        self.assertFalse(dft.xcfun.is_hybrid_xc('vv10'))
        self.assertTrue (dft.xcfun.is_hybrid_xc(('b3lyp',4,'vv10')))

    def test_nlc_coeff(self):
        self.assertEqual(dft.xcfun.nlc_coeff('vv10'), [5.9, 0.0093])

    def test_lda(self):
        e,v,f,k = dft.xcfun.eval_xc('lda,', rho[0][:3], deriv=3)
        self.assertAlmostEqual(lib.finger(e)   , -0.4720562542635522, 8)
        self.assertAlmostEqual(lib.finger(v[0]), -0.6294083390180697, 8)
        self.assertAlmostEqual(lib.finger(f[0]), -1.1414693830969338, 8)
        self.assertAlmostEqual(lib.finger(k[0]),  4.1402447248393921, 8)

        e,v,f,k = dft.xcfun.eval_xc('lda,', [rho[0][:3]*.5]*2, spin=1, deriv=3)
        self.assertAlmostEqual(lib.finger(e)   , -0.4720562542635522, 8)
        self.assertAlmostEqual(lib.finger(v[0].T[0]), -0.6294083390180697, 8)
        self.assertAlmostEqual(lib.finger(v[0].T[1]), -0.6294083390180697, 8)
        self.assertAlmostEqual(lib.finger(f[0].T[0]), -1.1414693830969338*2, 8)
        self.assertAlmostEqual(lib.finger(f[0].T[2]), -1.1414693830969338*2, 8)
        self.assertAlmostEqual(lib.finger(k[0].T[0]),  4.1402447248393921*4, 7)
        self.assertAlmostEqual(lib.finger(k[0].T[3]),  4.1402447248393921*4, 7)

    def test_lyp(self):
        e,v,f = dft.xcfun.eval_xc(',LYP', rho, deriv=3)[:3]
        self.assertAlmostEqual(numpy.dot(rho[0],e), -62.114576182676615, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[0]),-81.771670866308455, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[1]), 27.485383255125743, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],f[0]), 186.823806251777, 7)
        self.assertAlmostEqual(numpy.dot(rho[0],f[1]), -3391.2428894571085, 6)
        self.assertAlmostEqual(numpy.dot(rho[0],f[2]), 0, 9)

    def test_beckex(self):
        rho =(numpy.array([1.    , 1., 0., 0.]).reshape(-1,1),
              numpy.array([    .8, 1., 0., 0.]).reshape(-1,1))
        e,v,f = dft.xcfun.eval_xc('b88,', rho, spin=1, deriv=3)[:3]
        self.assertAlmostEqual(lib.finger(e)   ,-0.9061911523772116   , 9)
        self.assertAlmostEqual(lib.finger(v[0]),-1.8531364353196298   , 9)
        self.assertAlmostEqual(lib.finger(v[1]),-0.0018308066137967724, 9)
        self.assertAlmostEqual(lib.finger(f[0]),-0.21602284426026866  , 9)
        self.assertAlmostEqual(lib.finger(f[1]), 0.0072053520662545617, 9)
        self.assertAlmostEqual(lib.finger(f[2]), 0.0002275350850255538, 9)

    def test_m05x(self):
        rho =(numpy.array([1., 1., 0., 0., 0., 0.165 ]).reshape(-1,1),
              numpy.array([.8, 1., 0., 0., 0., 0.1050]).reshape(-1,1))
        test_ref = numpy.array([-1.57876583, -2.12127045,-2.11264351,-0.00315462,
                                 0.00000000, -0.00444560, 3.45640232, 4.4349756])
        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('m05,', rho, 1, deriv=3)
        self.assertAlmostEqual(float(exc)*1.8, test_ref[0], 5)
        self.assertAlmostEqual(abs(vxc[0]-test_ref[1:3]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[1]-test_ref[3:6]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[3]-test_ref[6:8]).max(), 0, 5)

        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('m05,', rho[0], 0, deriv=3)
        self.assertAlmostEqual(float(exc), -0.5746231988116002, 5)
        self.assertAlmostEqual(float(vxc[0]), -0.8806121005703862, 6)
        self.assertAlmostEqual(float(vxc[1]), -0.0032300155406846756, 7)
        self.assertAlmostEqual(float(vxc[3]), 0.4474953100487698, 5)

    def test_camb3lyp(self):
        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('camb3lyp', rho, 0, deriv=1)
        # FIXME, xcfun and libxc do not agree on camb3lyp
        # self.assertAlmostEqual(float(exc), -0.5752559666317147, 5)
        # self.assertAlmostEqual(float(vxc[0]), -0.7709812578936763, 5)
        # self.assertAlmostEqual(float(vxc[1]), -0.0029862221286189846, 7)

        self.assertEqual(dft.xcfun.rsh_coeff('camb3lyp'), (0.33, 0.65, -0.46))

    def test_define_xc(self):
        def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
            exc = vxc = fxc = kxc = None
            return exc, vxc, fxc, kxc

        mf = dft.RKS(mol)
        ni = dft.xcfun.define_xc(mf._numint, eval_xc, 'GGA', hyb=0.2)
        ni = dft.xcfun.define_xc(mf._numint, 'b3lyp+vwn', 'GGA', hyb=0.2)
        self.assertRaises(ValueError, dft.xcfun.define_xc, mf._numint, 0.1)

    def test_vs_libxc_rks(self):
        ao = dft.numint.eval_ao(mol, mf.grids.coords[:200], deriv=2)
        rho = dft.numint.eval_rho(mol, ao, dm, xctype='MGGA')
        rhoa = rho[:,:200]
        def check(xc_code, deriv=3, e_place=9, v_place=9, f_place=9, k_place=9):
            exc0, vxc0, fxc0, kxc0 = dft.libxc.eval_xc(xc_code, rhoa, 0, deriv=deriv)
            exc1, vxc1, fxc1, kxc1 = dft.xcfun.eval_xc(xc_code, rhoa, 0, deriv=deriv)
            self.assertAlmostEqual(abs(exc0-exc1).max(), 0, e_place)
            if deriv > 0:
                for v0, v1 in zip(vxc0, vxc1):
                    if v0 is not None and v1 is not None:
                        self.assertAlmostEqual(abs(v0-v1).max(), 0, v_place)
            if deriv > 1:
                for f0, f1 in zip(fxc0, fxc1):
                    if f0 is not None and f1 is not None:
                        self.assertAlmostEqual(abs(f0-f1).max(), 0, f_place)
            if deriv > 2:
                for k0, k1 in zip(kxc0, kxc1):
                    if k0 is not None and k1 is not None:
                        self.assertAlmostEqual(abs(k0-k1).max(), 0, k_place)

        check('lda,')

        check('pw86,')
        check('pbe,', e_place=6, v_place=6, f_place=5, k_place=4)
        #?check('becke,')
        #?check('br,')
        #?check('LDAERF,')
        check('optx,')
        check('OPTXCORR,')
        check('RPBE,')
        check('TF,'  )
        check('PW91,' , e_place=6, v_place=4, f_place=2, k_place=-1)
        check('m05,'  , deriv=1, e_place=6, v_place=6)
        check('m052x,', deriv=1, e_place=6, v_place=6)
        check('m06,'  , deriv=1, e_place=6, v_place=6)
        check('m062x,', deriv=1, e_place=6, v_place=6)
        check('m06l,' , deriv=1, e_place=6, v_place=6)
        check('TPSS,' ,                                  k_place=-4)
        #?check('REVTPSS,', deriv=1)  # xcfun crash
        check('APBE,')
        check('BLOC,' ,                                  k_place=-5)
        check('PBEINT,', e_place=7, v_place=6, f_place=5, k_place=4)

        check(',vwn3')
        check(',vwn5')
        check(',pbe'  , deriv=2)
        #?check(',br')
        #?check(',LDAERF')
        check(',lyp'  , deriv=2)
        check(',SPBE' , deriv=2, e_place=1, v_place=1, f_place=0)
        check(',PW91' , deriv=2,                       f_place=3)
        check(',m052x', deriv=1)
        check(',m05'  , deriv=1)
        check(',m06'  , deriv=1)
        check(',m062x', deriv=1)
        check(',m06l' , deriv=1)
        check(',TPSS' , deriv=1,              v_place=1)
        check(',REVTPSS', deriv=1, e_place=2, v_place=1)
        check(',p86'    , deriv=2, e_place=5, v_place=5, f_place=3)
        check(',APBE'   , deriv=2)
        check(',PBEINT' , deriv=1)
        check(',TPSSLOC', deriv=1, e_place=1, v_place=0)

        #?check('br')
        check('revpbe', deriv=2, e_place=6, v_place=6, f_place=5)
        check('b97'   , deriv=2, e_place=6, v_place=5, f_place=3)
        #?check('b97_1')
        #?check('b97_2')
        check('SVWN')
        check('BLYP'  , deriv=2)
        check('BP86'  , deriv=2, e_place=5, v_place=5, f_place=3)
        check('OLYP'  , deriv=2)
        check('KT1'   , deriv=1)
        check('KT2'   , deriv=1)
        #?check('KT3')
        check('PBE0'   , deriv=2, e_place=6, v_place=6, f_place=5)
        check('B3P86'  , deriv=2, e_place=5, v_place=5, f_place=3)
        check('B3P86G' , deriv=2, e_place=5, v_place=5, f_place=3)
        check('B3PW91' , deriv=2,                       f_place=4)
        check('B3PW91G', deriv=2, e_place=2, v_place=2, f_place=2)
        check('B3LYP'  , deriv=2)
        check('B3LYP5' , deriv=2)
        check('B3LYPG' , deriv=2)
        check('O3LYP'  , deriv=2)
        check('X3LYP'  , deriv=2, e_place=7, v_place=5, f_place=2)
        check('CAMB3LYP', deriv=1)
        check('B97_1'   , deriv=2, e_place=6, v_place=5, f_place=3)
        check('B97_2'   , deriv=2, e_place=6, v_place=5, f_place=3)
        check('TPSSH'   , deriv=1,            v_place=1)

    def test_vs_libxc_uks(self):
        ao = dft.numint.eval_ao(mol, mf.grids.coords[:400], deriv=2)
        rho = dft.numint.eval_rho(mol, ao, dm, xctype='MGGA')
        rhoa = rho[:,:200]
        rhob = rhoa + rho[:,200:400]
        def check(xc_code, deriv=3, e_place=9, v_place=9, f_place=9, k_place=9):
            exc0, vxc0, fxc0, kxc0 = dft.libxc.eval_xc(xc_code, (rhoa, rhob), 1, deriv=deriv)
            exc1, vxc1, fxc1, kxc1 = dft.xcfun.eval_xc(xc_code, (rhoa, rhob), 1, deriv=deriv)
            self.assertAlmostEqual(abs(exc0-exc1).max(), 0, e_place)
            if deriv > 0:
                for v0, v1 in zip(vxc0, vxc1):
                    if v0 is not None and v1 is not None:
                        self.assertAlmostEqual(abs(v0-v1).max(), 0, v_place)
            if deriv > 1:
                for f0, f1 in zip(fxc0, fxc1):
                    if f0 is not None and f1 is not None:
                        self.assertAlmostEqual(abs(f0-f1).max(), 0, f_place)
            if deriv > 2 and kxc0 is not None:
                for k0, k1 in zip(kxc0, kxc1):
                    if k0 is not None and k1 is not None:
                        self.assertAlmostEqual(abs(k0-k1).max(), 0, k_place)

        check('lda,')

        check('pw86,')
        check('pbe,', e_place=6, v_place=6, f_place=5, k_place=4)
        #?check('becke,')
        #?check('br,')
        #?check('LDAERF,')
        check('optx,')
        check('OPTXCORR,')
        check('RPBE,')
        check('TF,'  , e_place=0, v_place=-1, f_place=-2, k_place=-2)
        check('PW91,' , e_place=6, v_place=4, f_place=2, k_place=-1)
        check('m05,'  , deriv=1, e_place=6, v_place=6)
        check('m052x,', deriv=1, e_place=6, v_place=6)
        check('m06,'  , deriv=1, e_place=6, v_place=6)
        check('m062x,', deriv=1, e_place=6, v_place=6)
        check('m06l,' , deriv=1, e_place=6, v_place=6)
        check('TPSS,' ,                                  k_place=-4)
        #?check('REVTPSS,', deriv=1)  # libxc crash
        check('APBE,')
        check('BLOC,' ,                                  k_place=-5)
        check('PBEINT,', e_place=7, v_place=6, f_place=5, k_place=4)

        check(',vwn3', e_place=2, v_place=1, f_place=1, k_place=0)
        check(',vwn5')
        check(',pbe'  , deriv=2)
        #?check(',br')
        #?check(',LDAERF')
        check(',lyp'  , deriv=2)
        check(',SPBE' , deriv=2, e_place=1, v_place=1, f_place=0)
        check(',PW91' , deriv=2,                       f_place=3)
        check(',m052x', deriv=1)
        check(',m05'  , deriv=1)
        check(',m06'  , deriv=1)
        check(',m062x', deriv=1)
        check(',m06l' , deriv=1)
        check(',TPSS' , deriv=1,              v_place=1)
        check(',REVTPSS', deriv=1, e_place=2, v_place=1)
        check(',p86'    , deriv=2, e_place=5, v_place=5, f_place=3)
        check(',APBE'   , deriv=2)
        check(',PBEINT' , deriv=1)
        check(',TPSSLOC', deriv=1, e_place=1, v_place=0)

        #?check('br')
        check('revpbe', deriv=2, e_place=6, v_place=6, f_place=5)
        check('b97'   , deriv=2, e_place=6, v_place=5, f_place=3)
        #?check('b97_1')
        #?check('b97_2')
        check('SVWN')
        check('BLYP'  , deriv=2)
        check('BP86'  , deriv=2, e_place=5, v_place=5, f_place=3)
        check('OLYP'  , deriv=2)
        check('KT1'   , deriv=1)
        check('KT2'   , deriv=1)
        #?check('KT3')
        check('PBE0'   , deriv=2, e_place=6, v_place=6, f_place=5)
        check('B3P86'  , deriv=2, e_place=5, v_place=5, f_place=3)
        check('B3P86G' , deriv=2, e_place=3, v_place=2, f_place=2)
        check('B3PW91' , deriv=2,                       f_place=4)
        check('B3PW91G', deriv=2, e_place=2, v_place=2, f_place=2)
        check('B3LYP'  , deriv=2)
        check('B3LYP5' , deriv=2)
        check('B3LYPG' , deriv=2, e_place=3, v_place=2, f_place=2)
        check('O3LYP'  , deriv=2, e_place=3, v_place=2, f_place=1)
        check('X3LYP'  , deriv=2, e_place=7, v_place=5, f_place=2)
        check('CAMB3LYP', deriv=1)
        check('B97_1'   , deriv=2, e_place=6, v_place=5, f_place=3)
        check('B97_2'   , deriv=2, e_place=6, v_place=5, f_place=3)
        check('TPSSH'   , deriv=1,            v_place=1)


if __name__ == "__main__":
    print("Test xcfun")
    unittest.main()

