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

def setUpModule():
    global mol, mf, ao, rho, dm
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = 'h 0 0 0; h 1 .5 0; h 0 4 1; h 1 0 .2'
    mol.basis = 'aug-ccpvdz'
    mol.build()
    #dm = scf.RHF(mol).run(conv_tol=1e-14).make_rdm1()
    dm = numpy.load(os.path.realpath(os.path.join(__file__, '..', 'dm_h4.npy')))
    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mf = dft.RKS(mol)
        mf.grids.atom_grid = {"H": (50, 110)}
        mf.prune = None
        mf.grids.build(with_non0tab=False, sort_grids=False)
        ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
        rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA', with_lapl=True)

def tearDownModule():
    global mol, mf, ao, rho
    del mol, mf, ao, rho

def fp(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_parse_xc(self):
        hyb, fn_facs = dft.xcfun.parse_xc('.5*HF+.5*B3LYP5,VWN*.5')
        self.assertAlmostEqual(hyb[0], .6, 12)
        self.assertEqual([x[0] for x in fn_facs], [0,6,16,3])
        self.assertTrue(numpy.allclose([x[1] for x in fn_facs],
                                       (0.04, 0.36, 0.405, 0.595)))
        hyb, fn_facs = dft.xcfun.parse_xc('HF,')
        self.assertEqual(hyb[0], 1)
        self.assertEqual(fn_facs, ())

        hyb, fn_facs = dft.libxc.parse_xc('B88 - SLATER')
        self.assertEqual(fn_facs, ((106, 1), (1, -1)))
        hyb, fn_facs = dft.libxc.parse_xc('B88 -SLATER*.5')
        self.assertEqual(fn_facs, ((106, 1), (1, -0.5)))

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*B3LYP+0.25*B3LYP')
        self.assertTrue(numpy.allclose(hyb, (.15, 0, 0)))
        hyb = dft.libxc.hybrid_coeff('0.5*B3LYP+0.25*B3LYP')
        self.assertAlmostEqual(hyb, .15, 12)

        hyb, fn_facs = dft.xcfun.parse_xc('CAM_B3LYP')
        self.assertTrue(numpy.allclose(hyb, (0.19, 0.65, 0.33)))

        hyb, fn_facs = dft.xcfun.parse_xc('0.6*CAM_B3LYP+0.4*B3P86V5')
        self.assertTrue(numpy.allclose(hyb, (.08+0.19*.6, 0.65*.6, 0.33)))
        self.assertTrue(numpy.allclose(fn_facs,
                                       ((8, 0.276), (6, 0.498), (3, 0.19), (16, 0.486), (0, 0.032), (56, 0.324))))
        rsh = dft.xcfun.rsh_coeff('0.6*CAM_B3LYP+0.4*B3P86V5')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.39, -0.196)))

        hyb, fn_facs = dft.xcfun.parse_xc('0.4*B3P86V5+0.6*CAM_B3LYP')
        self.assertTrue(numpy.allclose(hyb, (.08+0.19*.6, 0.65*.6, 0.33)))
        self.assertTrue(numpy.allclose(fn_facs,
                                       ((0, 0.032), (6, 0.498), (56, 0.324), (3, 0.19), (8, 0.276), (16, 0.486))))
        rsh = dft.xcfun.rsh_coeff('0.4*B3P86V5+0.6*CAM_B3LYP')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.39, -0.196)))

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*SR-HF(0.3) + .8*HF + .22*LR_HF')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*SR-HF + .22*LR_HF(0.3) + .8*HF')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*SR-HF + .8*HF + .22*LR_HF(0.3)')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))

        hyb, fn_facs = dft.xcfun.parse_xc('0.5*RSH(2.04;0.56;0.3) + 0.5*BP86')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))
        self.assertEqual(fn_facs, ((6, 0.5), (56, 0.5)))

        self.assertRaises(ValueError, dft.xcfun.parse_xc, 'SR_HF(0.3) + LR_HF(.5)')
        self.assertRaises(ValueError, dft.xcfun.parse_xc, 'LR-HF(0.3) + SR-HF(.5)')

        hyb = dft.xcfun.hybrid_coeff('M05')
        self.assertAlmostEqual(hyb, 0.28, 9)

        hyb, fn_facs = dft.xcfun.parse_xc('APBE,')
        self.assertEqual(fn_facs[0][0], 68)

        hyb, fn_facs = dft.xcfun.parse_xc('VWN,')
        self.assertEqual(fn_facs, ((3, 1),))

        hyb, fn_facs = dft.xcfun.parse_xc('TF,')
        self.assertEqual(fn_facs, ((24, 1),))

        hyb, fn_facs = dft.xcfun.parse_xc("9.999e-5*HF,")
        self.assertEqual(hyb, (9.999e-5, 0, 0))

        ref = ((0, 1), (3, 1))
        self.assertEqual(dft.xcfun.parse_xc_name('LDA,VWN'), (0,3))
        self.assertEqual(dft.xcfun.parse_xc(('LDA','VWN'))[1], ref)
        self.assertEqual(dft.xcfun.parse_xc((0, 3))[1], ref)
        self.assertEqual(dft.xcfun.parse_xc('0, 3')[1], ref)
        self.assertEqual(dft.xcfun.parse_xc(3)[1], ((3,1),))

        #self.assertEqual(dft.xcfun.parse_xc('M11-L')[1], ((226,1),(75,1)))
        #self.assertEqual(dft.xcfun.parse_xc('M11L' )[1], ((226,1),(75,1)))
        #self.assertEqual(dft.xcfun.parse_xc('M11-L,M11L' )[1], ((226,1),(75,1)))
        #self.assertEqual(dft.xcfun.parse_xc('M11_L,M11-L')[1], ((226,1),(75,1)))
        #self.assertEqual(dft.xcfun.parse_xc('M11L,M11_L' )[1], ((226,1),(75,1)))

        #self.assertEqual(dft.xcfun.parse_xc('Xpbe,')[1], ((123,1),))
        #self.assertEqual(dft.xcfun.parse_xc('pbe,' )[1], ((101,1),))
        hyb, fn_facs = dft.xcfun.parse_xc('PBE*.4+LDA')
        self.assertEqual(fn_facs, ((5, 0.4), (4, 0.4), (0, 1)))
        hyb, fn_facs = dft.xcfun.parse_xc('PBE*.4+VWN')
        self.assertEqual(fn_facs, ((5, 0.4), (4, 0.4), (3, 1)))

        self.assertTrue (dft.xcfun.is_meta_gga('m05'))
        self.assertFalse(dft.xcfun.is_meta_gga('pbe0'))
        self.assertFalse(dft.xcfun.is_meta_gga('tf,'))
        #self.assertFalse(dft.xcfun.is_meta_gga('vv10'))
        self.assertTrue (dft.xcfun.is_gga('PBE0'))
        self.assertFalse(dft.xcfun.is_gga('m05'))
        self.assertFalse(dft.xcfun.is_gga('tf,'))
        self.assertTrue (dft.xcfun.is_lda('tf,'))
        #self.assertFalse(dft.xcfun.is_lda('vv10'))
        self.assertTrue (dft.xcfun.is_hybrid_xc('m05'))
        self.assertTrue (dft.xcfun.is_hybrid_xc('pbe0,'))
        self.assertFalse(dft.xcfun.is_hybrid_xc('m05,'))
        #self.assertFalse(dft.xcfun.is_hybrid_xc('vv10'))
        self.assertTrue (dft.xcfun.is_hybrid_xc(('b3lyp', 4, 'vv10')))

    def test_nlc_coeff(self):
        self.assertEqual(dft.xcfun.nlc_coeff('0.5*vv10'), (((5.9, 0.0093), .5),))
        self.assertEqual(dft.xcfun.nlc_coeff('pbe+vv10'), (((5.9, 0.0093), 1),))

    def test_lda(self):
        e,v,f,k = dft.xcfun.eval_xc('lda,', rho[0], deriv=3)
        self.assertAlmostEqual(numpy.dot(rho[0], e)   , -789.1150849798871 , 8)
        self.assertAlmostEqual(numpy.dot(rho[0], v[0]), -1052.1534466398498, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], f[0]), -1762.3340626646932, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], k[0]), 1202284274.6255436 , 3)

        e,v,f,k = dft.xcfun.eval_xc('lda,', [rho[0]*.5]*2, spin=1, deriv=3)
        self.assertAlmostEqual(numpy.dot(rho[0], e)        , -789.1150849798871 , 8)
        self.assertAlmostEqual(numpy.dot(rho[0], v[0].T[0]), -1052.1534466398498, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], v[0].T[1]), -1052.1534466398498, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], f[0].T[0]), -1762.3340626646932*2, 5)
        self.assertAlmostEqual(numpy.dot(rho[0], f[0].T[2]), -1762.3340626646932*2, 5)
        #FIXME: large errors found in 3rd derivatives
        #self.assertAlmostEqual(numpy.dot(rho[0], k[0].T[0]),  1202284274.6255436*4, 3)
        #self.assertAlmostEqual(numpy.dot(rho[0], k[0].T[3]),  1202284274.6255436*4, 3)

    def test_lyp(self):
        e,v,f = dft.xcfun.eval_xc(',LYP', rho, deriv=3)[:3]
        self.assertAlmostEqual(numpy.dot(rho[0],e), -62.114576182676615, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[0]),-81.771670866308455, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[1]), 27.485383255125743, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],f[0]), 186.823806251777, 7)
        self.assertAlmostEqual(numpy.dot(rho[0],f[1]), -3391.2428894571085, 6)
        self.assertAlmostEqual(numpy.dot(rho[0],f[2]), 0, 9)

    #def test_tpss(self):
    #    #FIXME: raised numerical error
    #    rho_a = numpy.array([[3.67808547e-08,-2.02358682e-08, 2.16729780e-07, 2.27036045e-07,-1.47795869e-07,-1.45668997e-09]]).T
    #    e, v = dft.xcfun.eval_xc('tpss,', rho_a, spin=0, deriv=1)[:2]
    #    rho_b = numpy.array([[4.53272893e-06, 4.18968775e-06,-2.83034672e-06, 2.61832978e-06, 5.63360737e-06, 8.97541777e-07]]).T
    #    e, v = dft.xcfun.eval_xc('tpss,', (rho_a, rho_b), spin=1, deriv=1)[:2]

    def test_beckex(self):
        rho =(numpy.array([1.    , 1., 0., 0.]).reshape(-1,1),
              numpy.array([    .8, 1., 0., 0.]).reshape(-1,1))
        e,v,f = dft.xcfun.eval_xc('b88,', rho, spin=1, deriv=3)[:3]
        self.assertAlmostEqual(lib.fp(e)   ,-0.9061911523772116   , 9)
        self.assertAlmostEqual(lib.fp(v[0]),-1.8531364353196298   , 9)
        self.assertAlmostEqual(lib.fp(v[1]),-0.0018308066137967724, 9)
        self.assertAlmostEqual(lib.fp(f[0]),-0.21602284426026866  , 9)
        self.assertAlmostEqual(lib.fp(f[1]), 0.0072053520662545617, 9)
        self.assertAlmostEqual(lib.fp(f[2]), 0.0002275350850255538, 9)

    def test_m05x(self):
        rho =(numpy.array([1., 1., 0., 0., 0., 0.165 ]).reshape(-1,1),
              numpy.array([.8, 1., 0., 0., 0., 0.1050]).reshape(-1,1))
        test_ref = numpy.array([-1.57876583, -2.12127045,-2.11264351,-0.00315462,
                                 0.00000000, -0.00444560, 3.45640232, 4.4349756])
        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('m05,', rho, 1, deriv=3)
        self.assertAlmostEqual(float(exc[0])*1.8, test_ref[0], 5)
        self.assertAlmostEqual(abs(vxc[0]-test_ref[1:3]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[1]-test_ref[3:6]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[3]-test_ref[6:8]).max(), 0, 5)

        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('m05,', rho[0], 0, deriv=3)
        self.assertAlmostEqual(float(exc[0]), -0.5746231988116002, 5)
        self.assertAlmostEqual(float(vxc[0][0]), -0.8806121005703862, 6)
        self.assertAlmostEqual(float(vxc[1][0]), -0.0032300155406846756, 7)
        self.assertAlmostEqual(float(vxc[3][0]), 0.4474953100487698, 5)

    def test_camb3lyp(self):
        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('camb3lyp', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc[0]), -0.5752559666317147, 5)
        self.assertAlmostEqual(float(vxc[0][0]), -0.7709812578936763, 5)
        self.assertAlmostEqual(float(vxc[1][0]), -0.0029862221286189846, 7)

        self.assertEqual(dft.xcfun.rsh_coeff('camb3lyp'), (0.33, 0.65, -0.46))

        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('RSH(0.65;-0.46;0.5) + BECKECAMX', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc[0]), -0.48916154057161476, 9)
        self.assertAlmostEqual(float(vxc[0][0]), -0.6761177630311709, 9)
        self.assertAlmostEqual(float(vxc[1][0]), -0.002949151742087167, 9)

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
        rho = dft.numint.eval_rho(mol, ao, dm, xctype='MGGA', with_lapl=True)
        rhoa = rho[:,:200]
        def check(xc_code, deriv=3, e_place=9, v_place=8, f_place=6, k_place=4):
            xctype = dft.libxc.xc_type(xc_code)
            if xctype == 'LDA':
                nv = 1
            elif xctype == 'GGA':
                nv = 4
            else:
                nv = 6
            exc0, vxc0, fxc0, kxc0 = dft.libxc.eval_xc(xc_code, rhoa[:nv], 0, deriv=deriv)
            exc1, vxc1, fxc1, kxc1 = dft.xcfun.eval_xc(xc_code, rhoa[:nv], 0, deriv=deriv)
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
        check('m05'  , deriv=1, e_place=6, v_place=6)
        #check('m05-2x,', deriv=1, e_place=6, v_place=6)
        check('m06'  , deriv=1, e_place=6, v_place=6)
        check('m06,'  , deriv=1, e_place=6, v_place=6)
        check('m062x,', deriv=1, e_place=6, v_place=6)
        check('m06l,' , deriv=1, e_place=6, v_place=6)
        check('scan,', deriv=2, e_place=8, v_place=7, f_place=-2)
        check('TPSS,' ,                               f_place=-3, k_place=-4)
        #check('REVTPSS,', deriv=1)  # xcfun crash
        check('APBE,')
        #check('BLOC,' , deriv=2)
        check('PBEINT,', e_place=7, v_place=6, f_place=5, k_place=4)

        check(',vwn3')
        check(',vwn5')
        check(',pbe'  , deriv=3)
        #?check(',br')
        #?check(',LDAERF')
        check(',lyp'  , deriv=3,                                  k_place=0)
        check(',SPBE' , deriv=3, e_place=1, v_place=1, f_place=0, k_place=-2)
        check(',PW91' , deriv=3, e_place=5, v_place=3, f_place=0, k_place=-2)
        check(',m052x', deriv=1)
        check(',m05'  , deriv=1)
        check(',m06'  , deriv=1)
        check(',m062x', deriv=1)
        check(',m06l' , deriv=1)
        check(',scan' , deriv=1, e_place=8, v_place=7)
        check(',TPSS' , deriv=1)
        check(',REVTPSS', deriv=1, e_place=2, v_place=1)
        check(',p86'    , deriv=3, e_place=5, v_place=5, f_place=3, k_place=-1)
        check(',APBE'   , deriv=3)
        check(',PBEINT' , deriv=3)
        check(',TPSSLOC', deriv=1)

        #?check('br')
        check('revpbe', deriv=3, e_place=6, v_place=6, f_place=5, k_place=4)
        check('b97'   , deriv=3, e_place=6, v_place=5, f_place=3, k_place=-5)
        #?check('b97_1')
        #?check('b97_2')
        check('SVWN')
        check('BLYP'  , deriv=3,                                  k_place=0)
        check('BP86'  , deriv=3, e_place=5, v_place=5, f_place=3, k_place=-1)
        check('OLYP'  , deriv=3,                                  k_place=0)
        check('KT1'   , deriv=3,                                  k_place=0)
        check('KT2'   , deriv=3,                                  k_place=-1)
        #?check('KT3')
        check('PBE0'   , deriv=3, e_place=6, v_place=6, f_place=5, k_place=-2)
        check('B3P86'  , deriv=3, e_place=5, v_place=5, f_place=3, k_place=-1)
        check('B3P86G' , deriv=3, e_place=5, v_place=5, f_place=3, k_place=-2)
        check('B3PW91' , deriv=3, e_place=5, v_place=3, f_place=0, k_place=-2)
        check('B3LYP'  , deriv=3,                                  k_place=0)
        check('B3LYP5' , deriv=3,                                  k_place=0)
        check('B3LYPG' , deriv=3,                                  k_place=-2)
        check('O3LYP'  , deriv=3,                                  k_place=-2)
        check('X3LYP'  , deriv=3, e_place=7, v_place=5, f_place=2, k_place=0)
        check('CAMB3LYP', deriv=1)
        check('B97_1'   , deriv=2, e_place=6, v_place=5, f_place=3)
        check('B97_2'   , deriv=2, e_place=6, v_place=5, f_place=3)
        check('TPSSH'   , deriv=1)

    def test_vs_libxc_uks(self):
        ao = dft.numint.eval_ao(mol, mf.grids.coords[:400], deriv=2)
        rho = dft.numint.eval_rho(mol, ao, dm, xctype='MGGA', with_lapl=True)
        rhoa = rho[:,:200]
        rhob = rhoa + rho[:,200:400]
        def check(xc_code, deriv=3, e_place=9, v_place=8, f_place=6, k_place=4):
            xctype = dft.libxc.xc_type(xc_code)
            if xctype == 'LDA':
                nv = 1
            elif xctype == 'GGA':
                nv = 4
            else:
                nv = 6
            exc0, vxc0, fxc0, kxc0 = dft.libxc.eval_xc(xc_code, (rhoa[:nv], rhob[:nv]), 1, deriv=deriv)
            exc1, vxc1, fxc1, kxc1 = dft.xcfun.eval_xc(xc_code, (rhoa[:nv], rhob[:nv]), 1, deriv=deriv)
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
        check('m05'  , deriv=1, e_place=6, v_place=6)
        #check('m052x,', deriv=1, e_place=6, v_place=6)
        check('m06'  , deriv=1, e_place=6, v_place=6)
        check('m06,'  , deriv=1, e_place=6, v_place=6)
        check('m062x,', deriv=1, e_place=6, v_place=6)
        check('m06l,' , deriv=1, e_place=6, v_place=6)
        check('scan,', deriv=2, e_place=8, v_place=7, f_place=-4)
        check('TPSS,' ,                               f_place=-4, k_place=-4)
        #check('REVTPSS,', deriv=1)  # xcfun crash
        check('APBE,')
        #check('BLOC,' , deriv=2)
        check('PBEINT,', e_place=7, v_place=6, f_place=5, k_place=4)

        check(',vwn3', e_place=2, v_place=1, f_place=1, k_place=0)
        check(',vwn5')
        check(',pbe'  , deriv=3,                        k_place=0)
        #?check(',br')
        #?check(',LDAERF')
        check(',lyp'  , deriv=3,                                  k_place=-1)
        check(',SPBE' , deriv=3, e_place=1, v_place=1, f_place=0, k_place=-1)
        check(',PW91' , deriv=3, e_place=5, v_place=3, f_place=2, k_place=-2)
        check(',m052x', deriv=1)
        check(',m05'  , deriv=1)
        check(',m06'  , deriv=1)
        check(',m062x', deriv=1)
        check(',m06l' , deriv=1)
        check(',scan' , deriv=1, e_place=8, v_place=7)
        check(',TPSS' , deriv=1)
        check(',REVTPSS', deriv=1, e_place=2, v_place=1)
        check(',p86'    , deriv=3, e_place=5, v_place=5, f_place=3, k_place=-2)
        check(',APBE'   , deriv=3,                                  k_place=-1)
        check(',PBEINT' , deriv=3,                                  k_place=-1)
        check(',TPSSLOC', deriv=1)

        #?check('br')
        check('revpbe', deriv=3, e_place=6, v_place=6, f_place=5, k_place=0)
        check('b97'   , deriv=3, e_place=6, v_place=5, f_place=3, k_place=-5)
        #?check('b97_1')
        #?check('b97_2')
        check('SVWN')
        check('BLYP'  , deriv=3,                                  k_place=-1)
        check('BP86'  , deriv=3, e_place=5, v_place=5, f_place=3, k_place=-3)
        check('OLYP'  , deriv=3,                                  k_place=-1)
        check('KT1'   , deriv=3,                                  k_place=-2)
        check('KT2'   , deriv=3,                                  k_place=-2)
        #?check('KT3')
        check('PBE0'   , deriv=3, e_place=6, v_place=6, f_place=5, k_place=-2)
        check('B3P86'  , deriv=3, e_place=5, v_place=5, f_place=3, k_place=-2)
        check('B3P86G' , deriv=3, e_place=3, v_place=2, f_place=2, k_place=-3)
        check('B3PW91' , deriv=3, e_place=5, v_place=4, f_place=2, k_place=-1)
        check('B3LYP'  , deriv=3,                                  k_place=-1)
        check('B3LYP5' , deriv=3,                                  k_place=-1)
        check('B3LYPG' , deriv=3, e_place=3, v_place=2, f_place=2, k_place=-2)
        check('O3LYP'  , deriv=3, e_place=3, v_place=2, f_place=1, k_place=-2)
        check('X3LYP'  , deriv=3, e_place=7, v_place=5, f_place=2, k_place=-1)
        check('CAMB3LYP', deriv=1, v_place=2)
        check('B97_1'   , deriv=3, e_place=6, v_place=5, f_place=3, k_place=-4)
        check('B97_2'   , deriv=3, e_place=6, v_place=5, f_place=3, k_place=-3)
        check('TPSSH'   , deriv=1)

    def test_m06(self):
        rho = numpy.array([0.11939647, -0.18865577, -0.11633254, 0.01779666, -0.55475521, 0.07092032,
                           0.11850155, -0.19297934, -0.11581427, 0.01373251, -0.57534216, 0.06596468,])
        rho = rho.reshape(2,6,1)
        exc, vxc, fxc, kxc = dft.xcfun.eval_xc(',m06', rho[0], 0, deriv=3)
        exc_ref = numpy.array([-0.02809518])
        vxc_ref = numpy.array([-0.0447466, -0.002186, 0.0154814])
        fxc_ref = numpy.array([-0.66060719, 0.35848677, -1.37754938,
                               -1.63288299, 0.21999345, 1.20222259])
        kxc_ref = numpy.array([-8.91475141e+00, 2.57071018e+01, -4.37413046e+01, 9.92064877e+01,
                               -1.47859948e+01, 1.51506522e+01, -4.83492461e+00, -1.75713898e+01,
                               -2.82029718e+01, 6.66847006e+01])
        self.assertAlmostEqual(exc[0] - exc_ref[0], 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([x for x in vxc if x is not None])-vxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([x for x in fxc if x is not None])-fxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([x for x in kxc if x is not None])-kxc_ref).max(), 0, 7)

        exc, vxc, fxc, kxc = dft.xcfun.eval_xc(',m06', rho, 1, deriv=3)
        exc_ref = numpy.array([-0.03730162])
        vxc_ref = numpy.array([-0.06168318,-0.0610502,-0.00957516,0,-0.01988745, 0.03161512, 0.04391615])
        fxc_ref = numpy.array([ 0.62207817,-1.02032218, 0.59318092,
                               -0.00267104, 0, 0.2768053, 0.26785479, 0, 0.02509479,
                               -1.34211352, 0,-0.0399852, 0, 0,-1.28860549,
                               -1.69697623,-0.03422618,-2.1835459,
                               -0.22523468, 0.26116507, 0.26545614,-0.23488143,
                               1.52817397,-0.13694284, 0, 0,-0.13797703, 1.73484678])
        kxc_ref = numpy.array([ 1.49056834e+01, 4.93235407e+00, 5.13177936e+00, 6.83179846e+00,
                               -1.97455841e+01, 0,-8.12321586e-01,-1.51226035e+00, 0,-1.66826458e+00,-8.99591711e-01, 0, -9.19638526e+00,
                               1.90010442e+01, 0, 1.34796896e-01, 0, 0,-7.73620959e-01,-6.10018848e-01, 0, 2.77414513e-01, 0, 0, 6.16594005e+00,
                               2.67162133e+01, 0,-9.09592410e-01, 0, 0,-9.28025786e-01, 0, 0, 0, 3.76462242e+01,
                               2.74381367e-01,-2.08192677e+00, 5.25100873e-01, 6.10851890e-01,-2.02662181e+00,-3.07707014e+00,
                               -4.45125589e-01, 6.81329714e-01, 0, 0,-4.69145558e-01, -2.42540301e+00,-2.45259140e+00,-5.57504449e-01, 0, 0, 6.02633158e-01, 2.58871134e+00,
                               2.87554064e+00, 2.85563442e-01,-1.94628304e-01, -1.88482952e-01, 2.90513272e-01, 3.07132889e+00,
                               -1.08483575e+00, 1.57821239e+00, 0, 0, 1.59013101e+00, 1.61019571e+00, 0, 0, 0, 0, 1.62235586e+00,-3.60275312e+00,
                               -3.90972925e+01,-1.10619931e-02,-1.12015763e-02, 0, 0, 0,-1.11455329e-02,-1.12861703e-02, -4.81639494e+01,
                               7.24911656e+01, 4.35367189e-02, 4.40860771e-02, 1.00010975e+02])
        self.assertAlmostEqual(exc[0] - exc_ref[0], 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([x for x in vxc if x is not None])-vxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([x for x in fxc if x is not None])-fxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([x for x in kxc if x is not None])-kxc_ref).max(), 0, 6)


if __name__ == "__main__":
    print("Test xcfun")
    unittest.main()
