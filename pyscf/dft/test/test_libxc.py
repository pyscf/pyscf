#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
import ctypes
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
        mf.grids.build(with_non0tab=False)
        nao = mol.nao_nr()
        ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
        rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')

def tearDownModule():
    global mol, mf, ao, rho
    del mol, mf, ao, rho

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_parse_xc(self):
        hyb, fn_facs = dft.libxc.parse_xc('.5*HF+.5*B3LYP5,VWN*.5')
        self.assertAlmostEqual(hyb[0], .6, 12)
        self.assertEqual([x[0] for x in fn_facs], [1,106,131,7])
        self.assertTrue(numpy.allclose([x[1] for x in fn_facs],
                                       (0.04, 0.36, 0.405, 0.595)))
        hyb, fn_facs = dft.libxc.parse_xc('HF,')
        self.assertEqual(hyb[0], 1)
        self.assertEqual(fn_facs, ())

        hyb, fn_facs = dft.libxc.parse_xc('B88 - SLATER')
        self.assertEqual(fn_facs, ((106, 1), (1, -1)))
        hyb, fn_facs = dft.libxc.parse_xc('B88 -SLATER*.5')
        self.assertEqual(fn_facs, ((106, 1), (1, -0.5)))

        hyb = dft.libxc.hybrid_coeff('0.5*B3LYP\n+0.25*B3LYP')
        self.assertAlmostEqual(hyb, .15, 12)

        hyb, fn_facs = dft.libxc.parse_xc('0.6*CAM_B3LYP+0.4*B3P86V5')
        self.assertTrue(numpy.allclose(hyb, (.08, .08, 0)))
        self.assertTrue(numpy.allclose(fn_facs,
                                       ((433, 0.6), (1, 0.032), (106, 0.288), (132, 0.324), (7, 0.076))))
        rsh = dft.libxc.rsh_coeff('0.6*CAM_B3LYP+0.4*B3P86V5')
        rsh1 = dft.libxc.rsh_coeff('CAM_B3LYP')
        hyb = dft.libxc.hybrid_coeff('B3P86V5')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.6*rsh1[1]+.4*hyb, 0.6*rsh1[2])))

        hyb, fn_facs = dft.libxc.parse_xc('0.4*B3P86V5+0.6*CAM_B3LYP')
        self.assertTrue(numpy.allclose(hyb, (.08, .08, 0)))
        self.assertTrue(numpy.allclose(fn_facs,
                                       ((1, 0.032), (106, 0.288), (132, 0.324), (7, 0.076), (433, 0.6))))
        rsh = dft.libxc.rsh_coeff('0.4*B3P86V5+0.6*CAM_B3LYP')
        rsh1 = dft.libxc.rsh_coeff('CAM_B3LYP')
        hyb = dft.libxc.hybrid_coeff('B3P86V5')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.6*rsh1[1]+.4*hyb, 0.6*rsh1[2])))

        hyb, fn_facs = dft.libxc.parse_xc('0.5*SR-HF(0.3) + .8*HF + .22*LR_HF')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))

        hyb, fn_facs = dft.libxc.parse_xc('0.5*SR-HF + .22*LR_HF(0.3) + .8*HF')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))

        hyb, fn_facs = dft.libxc.parse_xc('0.5*SR-HF + .8*HF + .22*LR_HF(0.3)')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))

        hyb, fn_facs = dft.libxc.parse_xc('0.5*RSH(2.04;0.56;0.3) + 0.5*BP86')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))
        self.assertEqual(fn_facs, ((106, 0.5), (132, 0.5)))

        hyb, fn_facs = dft.libxc.parse_xc('0.5*RSH(.3, 2.04, 0.56) + 0.5*BP86')
        self.assertEqual(hyb, (1.3, 1.02, 0.3))
        self.assertEqual(fn_facs, ((106, 0.5), (132, 0.5)))

        rsh = dft.libxc.rsh_coeff('0.5*HSE06+.001*HF')
        self.assertTrue(numpy.allclose(rsh, (0.11, .001, 0.125)))

        rsh = dft.libxc.rsh_coeff('0.5*wb97+0.001*HF')
        self.assertTrue(numpy.allclose(rsh, (0.4, 0.501, -.5)))

        self.assertRaises(ValueError, dft.libxc.parse_xc, 'SR_HF(0.3) + LR_HF(.5)')
        self.assertRaises(ValueError, dft.libxc.parse_xc, 'LR-HF(0.3) + SR-HF(.5)')

        hyb = dft.libxc.hybrid_coeff('0.5*B3LYP+0.2*HF')
        self.assertAlmostEqual(hyb, .3, 12)

        hyb = dft.libxc.hybrid_coeff('M05')
        self.assertAlmostEqual(hyb, 0.28, 9)

        hyb, fn_facs = dft.libxc.parse_xc('APBE,')
        self.assertEqual(fn_facs, ((184, 1),))

        hyb, fn_facs = dft.libxc.parse_xc('LDA0')
        self.assertEqual(fn_facs, ((177, 1),))

        #hyb, fn_facs = dft.libxc.parse_xc('TF,')
        #self.assertEqual(fn_facs, [(50, 1)])

        hyb, fn_facs = dft.libxc.parse_xc("9.999e-5*HF,")
        self.assertEqual(hyb, (9.999e-5, 9.999e-5, 0))

        ref = ((1, 1), (7, 1))
        self.assertEqual(dft.libxc.parse_xc_name('LDA,VWN'), (1,7))
        self.assertEqual(dft.libxc.parse_xc(('LDA','VWN'))[1], ref)
        self.assertEqual(dft.libxc.parse_xc((1, 7))[1], ref)
        self.assertEqual(dft.libxc.parse_xc('1, 7')[1], ref)
        self.assertEqual(dft.libxc.parse_xc(7)[1], ((7,1),))

        self.assertEqual(dft.libxc.parse_xc('M11-L')[1], ((226,1),(75,1)))
        self.assertEqual(dft.libxc.parse_xc('M11L' )[1], ((226,1),(75,1)))
        self.assertEqual(dft.libxc.parse_xc('M11-L,M11L' )[1], ((226,1),(75,1)))
        self.assertEqual(dft.libxc.parse_xc('M11_L,M11-L')[1], ((226,1),(75,1)))
        self.assertEqual(dft.libxc.parse_xc('M11L,M11_L' )[1], ((226,1),(75,1)))

        self.assertEqual(dft.libxc.parse_xc('Xpbe,')[1], ((123,1),))
        self.assertEqual(dft.libxc.parse_xc('pbe,' )[1], ((101,1),))
        self.assertEqual(dft.libxc.parse_xc('gga_x_pbe_gaussian' )[1], ((321,1),))


        hyb, fn_facs = dft.libxc.parse_xc('PBE*.4+LDA')
        self.assertEqual(fn_facs, ((101, 0.4), (130, 0.4), (1, 1)))
        self.assertRaises(KeyError, dft.libxc.parse_xc, 'PBE+VWN')

        self.assertTrue (dft.libxc.is_meta_gga('m05'))
        self.assertFalse(dft.libxc.is_meta_gga('pbe0'))
        self.assertFalse(dft.libxc.is_meta_gga('tf,'))
        self.assertFalse(dft.libxc.is_meta_gga('vv10'))
        self.assertTrue (dft.libxc.is_gga('PBE0'))
        self.assertFalse(dft.libxc.is_gga('m05'))
        self.assertFalse(dft.libxc.is_gga('tf,'))
        self.assertTrue (dft.libxc.is_lda('tf,'))
        self.assertFalse(dft.libxc.is_lda('vv10'))
        self.assertTrue (dft.libxc.is_hybrid_xc('m05'))
        self.assertTrue (dft.libxc.is_hybrid_xc('pbe0,'))
        self.assertTrue (dft.libxc.is_hybrid_xc('m05,'))
        self.assertFalse(dft.libxc.is_hybrid_xc('vv10'))
        self.assertTrue (dft.libxc.is_hybrid_xc((402,'vv10')))
        self.assertTrue (dft.libxc.is_hybrid_xc(('402','vv10')))
        self.assertTrue (dft.libxc.is_nlc('b97mv'))
        self.assertTrue (dft.libxc.is_nlc('lc-vv10'))
        self.assertTrue (dft.libxc.is_nlc('scanl-vv10'))
        self.assertTrue (dft.libxc.is_nlc('b97mv+pbe'))
        self.assertTrue (dft.libxc.is_nlc((402, 'b97mv')))
        self.assertTrue (dft.libxc.is_nlc(('402', 'b97mv')))

    def test_libxc_cam_beta(self):
        rsh_tmp = (ctypes.c_double*3)()
        dft.libxc._itrf.LIBXC_rsh_coeff(1, rsh_tmp)
        beta = rsh_tmp[2]
        self.assertEqual(beta, 0)

        dft.libxc._itrf.LIBXC_rsh_coeff(433, rsh_tmp)
        dft.libxc._itrf.LIBXC_rsh_coeff(1, rsh_tmp)
        beta = rsh_tmp[2]
        self.assertEqual(beta, 0)

        dft.libxc._itrf.LIBXC_is_hybrid(1)
        dft.libxc._itrf.LIBXC_rsh_coeff(1, rsh_tmp)
        beta = rsh_tmp[2]
        self.assertEqual(beta, 0)

    def test_nlc_coeff(self):
        self.assertEqual(dft.libxc.nlc_coeff('0.5*vv10'), (((5.9, 0.0093), .5),))
        self.assertEqual(dft.libxc.nlc_coeff('pbe+vv10'), (((5.9, 0.0093), 1),))

    def test_lda(self):
        e,v,f,k = dft.libxc.eval_xc('lda,', rho[0], deriv=3)
        self.assertAlmostEqual(numpy.dot(rho[0], e)   , -789.1150849798871 , 8)
        self.assertAlmostEqual(numpy.dot(rho[0], v[0]), -1052.1534466398498, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], f[0]), -1762.3340626646932, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], k[0]), 1202284274.6255436 , 3)

        e,v,f,k = dft.libxc.eval_xc('lda,', [rho[0]*.5]*2, spin=1, deriv=3)
        self.assertAlmostEqual(numpy.dot(rho[0], e)        , -789.1150849798871 , 8)
        self.assertAlmostEqual(numpy.dot(rho[0], v[0].T[0]), -1052.1534466398498, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], v[0].T[1]), -1052.1534466398498, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], f[0].T[0]), -1762.3340626646932*2, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], f[0].T[2]), -1762.3340626646932*2, 8)
        self.assertAlmostEqual(numpy.dot(rho[0], k[0].T[0]),  1202284274.6255436*4, 2)
        self.assertAlmostEqual(numpy.dot(rho[0], k[0].T[3]),  1202284274.6255436*4, 2)

    def test_lyp(self):
        e,v,f = dft.libxc.eval_xc(',LYP', rho, deriv=2)[:3]
        self.assertAlmostEqual(numpy.dot(rho[0],e), -62.114576182676615, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[0]),-81.771670866308455, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[1]), 27.485383255125743, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],f[0]), 186.823806251777, 2)
        self.assertAlmostEqual(numpy.dot(rho[0],f[1]), -3391.2428894571085, 6)
        self.assertAlmostEqual(numpy.dot(rho[0],f[2]), 0, 8)
        self.assertAlmostEqual(abs(f[2]).sum(), 0, 3)

    #def test_tpss(self):
    #    #FIXME: raised numerical error
    #    rho_a = numpy.array([[3.67808547e-08,-2.02358682e-08, 2.16729780e-07, 2.27036045e-07,-1.47795869e-07,-1.45668997e-09]]).T
    #    e, v = dft.libxc.eval_xc('tpss,', rho_a, spin=0, deriv=1)[:2]
    #    rho_b = numpy.array([[4.53272893e-06, 4.18968775e-06,-2.83034672e-06, 2.61832978e-06, 5.63360737e-06, 8.97541777e-07]]).T
    #    e, v = dft.libxc.eval_xc('tpss,', (rho_a, rho_b), spin=1, deriv=1)[:2]

    #TDOO: enable this test when https://gitlab.com/libxc/libxc/-/issues/561 is solved
    @unittest.skip('hse03 and hse06 fxc have large numerical errors in Libxc')
    def test_hse06(self):
        ni = dft.numint.NumInt()
        rho = numpy.array([.235, 1.5e-9, 2e-9, 1e-9])[:,None]
        xc = 'hse06'
        fxc1 = ni.eval_xc_eff(xc, rho, deriv=2, xctype='GGA')[2]
        rho = numpy.array([rho*.5]*2)
        fxc2 = ni.eval_xc_eff(xc, rho, deriv=2, xctype='GGA')[2]
        fxc2 = (fxc2[0,:,0] + fxc2[0,:,1])/2
        self.assertAlmostEqual(abs(fxc2 - fxc1).max(), 0, 12)

        rho = numpy.array([.235, 0, 0, 0])[:,None]
        xc = 'hse06'
        fxc1 = ni.eval_xc_eff(xc, rho, deriv=2, xctype='GGA')[2]
        rho = numpy.array([rho*.5]*2)
        fxc2 = ni.eval_xc_eff(xc, rho, deriv=2, xctype='GGA')[2]
        fxc2 = (fxc2[0,:,0] + fxc2[0,:,1])/2
        self.assertAlmostEqual(abs(fxc2 - fxc1).max(), 0, 12)

    def test_define_xc_gga(self):
        def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
            # A fictitious XC functional to demonstrate the usage
            rho0, dx, dy, dz = rho[:4]
            gamma = (dx**2 + dy**2 + dz**2)
            exc = .01 * rho0**2 + .02 * (gamma+.001)**.5
            vrho = .01 * 2 * rho0
            vgamma = .02 * .5 * (gamma+.001)**(-.5)
            vlapl = None
            vtau = None
            vxc = (vrho, vgamma, vlapl, vtau)
            fxc = None  # 2nd order functional derivative
            kxc = None  # 3rd order functional derivative
            return exc, vxc, fxc, kxc

        mf = dft.RKS(mol)
        ni = dft.libxc.define_xc(mf._numint, eval_xc, 'GGA', hyb=0.2)
        numpy.random.seed(1)
        rho = numpy.random.random((4,10))
        exc, vxc = ni.eval_xc(None, rho, 0, deriv=1)[:2]
        self.assertAlmostEqual(lib.fp(exc), 0.0012441814416833327, 9)
        self.assertAlmostEqual(lib.fp(vxc[0]), 0.0065565189784811129, 9)
        self.assertAlmostEqual(lib.fp(vxc[1]), 0.0049270110162854116, 9)

        with self.assertRaises(AssertionError):
            ni.eval_xc_eff(None, rho, deriv=2)

        # Ensure the xc_code in the input has no impact to the result.
        n, exc, vxc = ni.nr_rks(mol, mf.grids, 'm06x', dm)
        self.assertAlmostEqual(n, 4, 5)
        self.assertAlmostEqual(lib.fp(exc), 0.01197588220700074, 6)
        self.assertAlmostEqual(lib.fp(vxc), -0.043974912389152986, 6)

        ni = dft.libxc.define_xc(ni, eval_xc, 'MGGA')
        with self.assertRaises(AssertionError):
            ni.eval_xc_eff(None, rho, deriv=1)

        mf = mf.define_xc_('0.5*B3LYP+0.5*B3LYP')
        exc0, vxc0 = mf._numint.eval_xc(None, rho, 0, deriv=1)[:2]
        exc1, vxc1 = dft.libxc.eval_xc('0.5*B3LYP+0.5*B3LYP', rho, 0, deriv=1)[:2]
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 9)
        self.assertAlmostEqual(abs(vxc0[0]-vxc1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(vxc0[1]-vxc1[1]).max(), 0, 9)

        self.assertRaises(ValueError, dft.libxc.define_xc, mf._numint, 0.1)

        ni = dft.libxc.define_xc(mf._numint, 'PBE')
        exc1, vxc1 = dft.libxc.eval_xc('PBE', rho, 0, deriv=1)[:2]
        exc0, vxc0 = ni.eval_xc(None, rho, 0, deriv=1)[:2]
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 9)
        self.assertAlmostEqual(abs(vxc0[0]-vxc1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(vxc0[1]-vxc1[1]).max(), 0, 9)

    def test_define_xc_mgga(self):
        def eval_mgga_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
            # A fictitious XC functional to demonstrate the usage
            rho0, dx, dy, dz, tau = rho
            gamma = (dx**2 + dy**2 + dz**2)
            exc = .01 * rho0**2 + .02 * (gamma+.001)**.5 + .01 * tau**2
            vrho = .01 * 2 * rho0
            vgamma = .02 * .5 * (gamma+.001)**(-.5)
            vtau = .02 * tau
            vxc = (vrho, vgamma, vtau)
            zeros = numpy.zeros_like(rho0)
            fxc = (zeros, zeros, zeros)
            kxc = None  # 3rd order functional derivative
            return exc, vxc, fxc, kxc

        mf = dft.RKS(mol)
        ni = dft.libxc.define_xc(mf._numint, eval_mgga_xc, 'MGGA')
        numpy.random.seed(1)
        rho = numpy.random.random((5,10))
        exc, vxc = ni.eval_xc(None, rho, 0, deriv=1)[:2]
        self.assertAlmostEqual(lib.fp(exc), 0.016353861820705, 9)
        self.assertAlmostEqual(lib.fp(vxc[0]), 0.0065565189784811129, 9)
        self.assertAlmostEqual(lib.fp(vxc[1]), 0.0049270110162854116, 9)

        with self.assertRaises(AssertionError):
            ni.eval_xc_eff(None, rho, deriv=2)

    def test_m05x(self):
        rho =(numpy.array([1., 1., 0., 0., 0., 0.165 ]).reshape(-1,1),
              numpy.array([.8, 1., 0., 0., 0., 0.1050]).reshape(-1,1))
        #test_ref = numpy.array([-1.57876583, -2.12127045,-2.11264351,-0.00315462,
        #                         0.00000000, -0.00444560, 3.45640232, 4.4349756])  # libxc-4.3.4
        test_ref = numpy.array([-1.57730394, -2.12127045,-2.11297165,-0.00315462,
                                 0.00000000, -0.00446935, 3.45640232, 4.42563831])  # libxc-5.1.2
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('1.38888888889*m05,', rho, 1, deriv=1)
        self.assertAlmostEqual(float(exc[0])*1.8, test_ref[0], 5)
        self.assertAlmostEqual(abs(vxc[0]-test_ref[1:3]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[1]-test_ref[3:6]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[3]-test_ref[6:8]).max(), 0, 5)

        exc, vxc, fxc, kxc = dft.libxc.eval_xc('1.38888888889*m05,', rho[0], 0, deriv=1)
        self.assertAlmostEqual(float(exc[0]), -0.5746231988116002, 5)
        self.assertAlmostEqual(float(vxc[0][0]), -0.8806121005703862, 6)
        self.assertAlmostEqual(float(vxc[1][0]), -0.0032300155406846756, 7)
        self.assertAlmostEqual(float(vxc[3][0]), 0.4474953100487698, 5)

    def test_camb3lyp(self):
        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('camb3lyp', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc[0]), -0.5752559666317147, 7)
        self.assertAlmostEqual(float(vxc[0][0]), -0.7709812578936763, 7)
        self.assertAlmostEqual(float(vxc[1][0]), -0.0029862221286189846, 7)

        self.assertEqual(dft.libxc.rsh_coeff('camb3lyp'), (0.33, 0.65, -0.46))

        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('RSH(0.5,0.65,-0.46) + 0.46*ITYH + .35*B88,', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc[0]), -0.48916154057161476, 9)
        self.assertAlmostEqual(float(vxc[0][0]), -0.6761177630311709, 9)
        self.assertAlmostEqual(float(vxc[1][0]), -0.002949151742087167, 9)

    def test_ityh(self):
        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('ityh,', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc[0]), -0.6359945579326314, 7)
        self.assertAlmostEqual(float(vxc[0][0]), -0.8712041561251518, 7)
        self.assertAlmostEqual(float(vxc[1][0]), -0.003911167644579979, 7)
        self.assertEqual(dft.libxc.rsh_coeff('ityh,'), (0.2, 0.0, 0.0))

    def test_lcwpbe(self):
        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('LC_wPBE', rho, 0, deriv=1)
        exc1, vxc1, fxc1, kxc1 = dft.libxc.eval_xc('RSH(0.4,1.0,-1.0)+wpbeh,pbe', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc[0]), float(exc1[0]), 7)
        self.assertAlmostEqual(float(vxc[0][0]), float(vxc1[0][0]), 7)
        self.assertAlmostEqual(float(vxc[1][0]), float(vxc1[1][0]), 7)

    def test_deriv_order(self):
        self.assertTrue(dft.libxc.test_deriv_order('lda', 3, raise_error=False))
        self.assertTrue(dft.libxc.test_deriv_order('m05', 2, raise_error=False))
        self.assertTrue(dft.libxc.test_deriv_order('camb3lyp', 3, True))
        self.assertTrue(dft.libxc.test_deriv_order('pbe0', 3, True))
        self.assertRaises(KeyError, dft.libxc.test_deriv_order, 'OL2', 3, True)

    def test_xc_type(self):
        self.assertEqual(dft.libxc.xc_type(416), 'GGA')
        self.assertEqual(dft.libxc.xc_type('hf'), 'HF')
        self.assertEqual(dft.libxc.xc_type(',vwn'), 'LDA')
        self.assertEqual(dft.libxc.xc_type('lda+b3lyp'), 'GGA')
        self.assertEqual(dft.libxc.xc_type('wb97x_v'), 'GGA')
        self.assertEqual(dft.libxc.xc_type('wb97m_v'), 'MGGA')
        self.assertEqual(dft.libxc.xc_type('bp86'), 'GGA')

    def test_m06(self):
        rho = numpy.array([0.11939647, -0.18865577, -0.11633254, 0.01779666, -0.55475521, 0.07092032,
                           0.11850155, -0.19297934, -0.11581427, 0.01373251, -0.57534216, 0.06596468,])
        rho = rho.reshape(2,6,1)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc(',m06', rho[0], 0, deriv=3)
        exc_ref = numpy.array([-0.02809518])
        vxc_ref = numpy.array([-0.0447466, -0.002186, 0.0154814])
        fxc_ref = numpy.array([-0.66060719, 0.35848677, -1.37754938,
                               -1.63288299, 0.21999345, 1.20222259])
        kxc_ref = numpy.array([-8.91475141e+00, 2.57071018e+01, -4.37413046e+01, 9.92064877e+01,
                               -1.47859948e+01, 1.51506522e+01, -4.83492461e+00, -1.75713898e+01,
                               -2.82029718e+01, 6.66847006e+01])
        self.assertAlmostEqual(exc[0] - exc_ref[0], 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([vxc[i] for i in [0,1,3]])-vxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([fxc[i] for i in [0,1,2,4,6,9]])-fxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([kxc[i] for i in [0,1,2,3,5,7,10,12,15,19]])-kxc_ref).max(), 0, 7)

        exc, vxc, fxc, kxc = dft.libxc.eval_xc(',m06', rho, 1, deriv=3)
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
        self.assertAlmostEqual(abs(numpy.hstack([vxc[i] for i in [0,1,3]])-vxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([fxc[i] for i in [0,1,2,4,6,9]])-fxc_ref).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.hstack([kxc[i] for i in [0,1,2,3,5,7,10,12,15,19]])-kxc_ref).max(), 0, 6)

    def test_dft_parser(self):
        from pyscf.dft.dft_parser import parse_dft
        self.assertEqual(parse_dft('wb97m-d3bj'), ('wb97m-v', False, 'd3bj'))
        self.assertEqual(parse_dft('b3lyp-d3zerom'), ('b3lyp', '', 'd3zerom'))
        self.assertEqual(parse_dft('wb97x-d3bj'), ('wb97x-v', False, 'd3bj'))
        self.assertEqual(parse_dft('wb97x-d3zero2b'), ('wb97x', '', 'd3zero2b'))

if __name__ == "__main__":
    print("Test libxc")
    unittest.main()
