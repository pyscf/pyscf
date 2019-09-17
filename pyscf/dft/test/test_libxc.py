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
        hyb, fn_facs = dft.libxc.parse_xc('.5*HF+.5*B3LYP,VWN*.5')
        self.assertAlmostEqual(hyb[0], .6, 12)
        self.assertEqual([x[0] for x in fn_facs], [1,106,131,7])
        self.assertTrue(numpy.allclose([x[1] for x in fn_facs],
                                       (0.04, 0.36, 0.405, 0.595)))
        hyb, fn_facs = dft.libxc.parse_xc('HF,')
        self.assertEqual(hyb[0], 1)
        self.assertEqual(fn_facs, [])

        hyb, fn_facs = dft.libxc.parse_xc('B88 - SLATER')
        self.assertEqual(fn_facs, [(106, 1), (1, -1)])
        hyb, fn_facs = dft.libxc.parse_xc('B88 -SLATER*.5')
        self.assertEqual(fn_facs, [(106, 1), (1, -0.5)])

        hyb, fn_facs = dft.libxc.parse_xc('0.5*B3LYP+0.25*B3LYP')
        self.assertTrue(numpy.allclose(hyb, [.15, 0, 0]))
        hyb = dft.libxc.hybrid_coeff('0.5*B3LYP+0.25*B3LYP')
        self.assertAlmostEqual(hyb, .15, 12)

        hyb, fn_facs = dft.libxc.parse_xc('0.6*CAM_B3LYP+0.4*B3P86')
        self.assertTrue(numpy.allclose(hyb, [.08, 0, 0]))
        self.assertTrue(numpy.allclose(fn_facs,
                                       [(433, 0.6), (1, 0.032), (106, 0.288), (132, 0.324), (7, 0.076)]))
        rsh = dft.libxc.rsh_coeff('0.6*CAM_B3LYP+0.4*B3P86')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.39, -0.196)))

        hyb, fn_facs = dft.libxc.parse_xc('0.4*B3P86+0.6*CAM_B3LYP')
        self.assertTrue(numpy.allclose(hyb, [.08, 0, 0]))
        self.assertTrue(numpy.allclose(fn_facs,
                                       [(1, 0.032), (106, 0.288), (132, 0.324), (7, 0.076), (433, 0.6)]))
        rsh = dft.libxc.rsh_coeff('0.4*B3P86+0.6*CAM_B3LYP')
        self.assertTrue(numpy.allclose(rsh, (0.33, 0.39, -0.196)))

        hyb, fn_facs = dft.libxc.parse_xc('0.5*SR-HF(0.3) + .8*HF + .22*LR_HF')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])

        hyb, fn_facs = dft.libxc.parse_xc('0.5*SR-HF + .22*LR_HF(0.3) + .8*HF')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])

        hyb, fn_facs = dft.libxc.parse_xc('0.5*SR-HF + .8*HF + .22*LR_HF(0.3)')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])

        hyb, fn_facs = dft.libxc.parse_xc('0.5*RSH(2.04;0.56;0.3) + 0.5*BP86')
        self.assertEqual(hyb, [1.3, 1.02, 0.3])
        self.assertEqual(fn_facs, [(106, 0.5), (132, 0.5)])

        self.assertRaises(ValueError, dft.libxc.parse_xc, 'SR_HF(0.3) + LR_HF(.5)')
        self.assertRaises(ValueError, dft.libxc.parse_xc, 'LR-HF(0.3) + SR-HF(.5)')

        hyb = dft.libxc.hybrid_coeff('M05')
        self.assertAlmostEqual(hyb, 0.28, 9)

        hyb, fn_facs = dft.libxc.parse_xc('APBE,')
        self.assertEqual(fn_facs, [(184, 1)])

        #hyb, fn_facs = dft.libxc.parse_xc('TF,')
        #self.assertEqual(fn_facs, [(50, 1)])

        ref = [(1, 1), (7, 1)]
        self.assertEqual(dft.libxc.parse_xc_name('LDA,VWN'), (1,7))
        self.assertEqual(dft.libxc.parse_xc(('LDA','VWN'))[1], ref)
        self.assertEqual(dft.libxc.parse_xc((1, 7))[1], ref)
        self.assertEqual(dft.libxc.parse_xc('1, 7')[1], ref)
        self.assertEqual(dft.libxc.parse_xc(7)[1], [(7,1)])

        self.assertEqual(dft.libxc.parse_xc('M11-L')[1], [(226,1),(75,1)])
        self.assertEqual(dft.libxc.parse_xc('M11L' )[1], [(226,1),(75,1)])
        self.assertEqual(dft.libxc.parse_xc('M11-L,M11L' )[1], [(226,1),(75,1)])
        self.assertEqual(dft.libxc.parse_xc('M11_L,M11-L')[1], [(226,1),(75,1)])
        self.assertEqual(dft.libxc.parse_xc('M11L,M11_L' )[1], [(226,1),(75,1)])

        self.assertEqual(dft.libxc.parse_xc('Xpbe,')[1], [(123,1)])
        self.assertEqual(dft.libxc.parse_xc('pbe,' )[1], [(101,1)])
        hyb, fn_facs = dft.libxc.parse_xc('PBE*.4+LDA')
        self.assertEqual(fn_facs, [(101, 0.4), (130, 0.4), (1, 1)])
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
        self.assertFalse(dft.libxc.is_hybrid_xc('m05,'))
        self.assertFalse(dft.libxc.is_hybrid_xc('vv10'))
        self.assertTrue (dft.libxc.is_hybrid_xc((402,'vv10')))
        self.assertTrue (dft.libxc.is_hybrid_xc(('402','vv10')))

    def test_libxc_cam_beta_bug(self):
        '''As a detector for libxc-3.0.0. libxc-3.0.1 fixed this bug
        '''
        import ctypes
        rsh_tmp = (ctypes.c_double*3)()
        dft.libxc._itrf.LIBXC_rsh_coeff(1, rsh_tmp)
        beta = rsh_tmp[2]
        self.assertEqual(beta, 0)

        dft.libxc._itrf.LIBXC_rsh_coeff(433, rsh_tmp)
        dft.libxc._itrf.LIBXC_rsh_coeff(1, rsh_tmp)
        beta = rsh_tmp[2]
        self.assertEqual(beta, 0) # libxc-3.0.0 produces -0.46

        dft.libxc._itrf.LIBXC_is_hybrid(1)
        dft.libxc._itrf.LIBXC_rsh_coeff(1, rsh_tmp)
        beta = rsh_tmp[2]
        self.assertEqual(beta, 0)

    def test_nlc_coeff(self):
        self.assertEqual(dft.libxc.nlc_coeff('0.5*vv10'), [5.9, 0.0093])

    def test_lda(self):
        e,v,f,k = dft.libxc.eval_xc('lda,', rho[0][:3], deriv=3)
        self.assertAlmostEqual(lib.finger(e)   , -0.4720562542635522, 8)
        self.assertAlmostEqual(lib.finger(v[0]), -0.6294083390180697, 8)
        self.assertAlmostEqual(lib.finger(f[0]), -1.1414693830969338, 8)
        self.assertAlmostEqual(lib.finger(k[0]),  4.1402447248393921, 8)

        e,v,f,k = dft.libxc.eval_xc('lda,', [rho[0][:3]*.5]*2, spin=1, deriv=3)
        self.assertAlmostEqual(lib.finger(e)   , -0.4720562542635522, 8)
        self.assertAlmostEqual(lib.finger(v[0].T[0]), -0.6294083390180697, 8)
        self.assertAlmostEqual(lib.finger(v[0].T[1]), -0.6294083390180697, 8)
        self.assertAlmostEqual(lib.finger(f[0].T[0]), -1.1414693830969338*2, 8)
        self.assertAlmostEqual(lib.finger(f[0].T[2]), -1.1414693830969338*2, 8)
        self.assertAlmostEqual(lib.finger(k[0].T[0]),  4.1402447248393921*4, 7)
        self.assertAlmostEqual(lib.finger(k[0].T[3]),  4.1402447248393921*4, 7)

    def test_lyp(self):
        e,v,f = dft.libxc.eval_xc(',LYP', rho, deriv=2)[:3]
        self.assertAlmostEqual(numpy.dot(rho[0],e), -62.114576182676615, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[0]),-81.771670866308455, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[1]), 27.485383255125743, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],f[0]), 186.823806251777, 2)
        self.assertAlmostEqual(numpy.dot(rho[0],f[1]), -3391.2428894571085, 6)
        self.assertAlmostEqual(numpy.dot(rho[0],f[2]), 0, 8)
        self.assertAlmostEqual(abs(f[2]).sum(), 0, 3)

    def test_define_xc(self):
        def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
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
        self.assertAlmostEqual(lib.finger(exc), 0.0012441814416833327, 9)
        self.assertAlmostEqual(lib.finger(vxc[0]), 0.0065565189784811129, 9)
        self.assertAlmostEqual(lib.finger(vxc[1]), 0.0049270110162854116, 9)

        mf = mf.define_xc_('0.5*B3LYP+0.5*B3LYP')
        exc0, vxc0 = mf._numint.eval_xc(None, rho, 0, deriv=1)[:2]
        exc1, vxc1 = dft.libxc.eval_xc('0.5*B3LYP+0.5*B3LYP', rho, 0, deriv=1)[:2]
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 9)
        self.assertAlmostEqual(abs(vxc0[0]-vxc1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(vxc0[1]-vxc1[1]).max(), 0, 9)

        self.assertRaises(ValueError, dft.libxc.define_xc, mf._numint, 0.1)

# libxc-4.2.3 does not support m05x
    def test_m05x(self):
        rho =(numpy.array([1., 1., 0., 0., 0., 0.165 ]).reshape(-1,1),
              numpy.array([.8, 1., 0., 0., 0., 0.1050]).reshape(-1,1))
        test_ref = numpy.array([-1.57876583, -2.12127045,-2.11264351,-0.00315462,
                                 0.00000000, -0.00444560, 3.45640232, 4.4349756])
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('m05,', rho, 1, deriv=1)
        self.assertAlmostEqual(float(exc)*1.8, test_ref[0], 5)
        self.assertAlmostEqual(abs(vxc[0]-test_ref[1:3]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[1]-test_ref[3:6]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[3]-test_ref[6:8]).max(), 0, 5)

        exc, vxc, fxc, kxc = dft.libxc.eval_xc('m05,', rho[0], 0, deriv=1)
        self.assertAlmostEqual(float(exc), -0.5746231988116002, 5)
        self.assertAlmostEqual(float(vxc[0]), -0.8806121005703862, 6)
        self.assertAlmostEqual(float(vxc[1]), -0.0032300155406846756, 7)
        self.assertAlmostEqual(float(vxc[3]), 0.4474953100487698, 5)

    def test_camb3lyp(self):
        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('camb3lyp', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc), -0.5752559666317147, 5)
        self.assertAlmostEqual(float(vxc[0]), -0.7709812578936763, 5)
        self.assertAlmostEqual(float(vxc[1]), -0.0029862221286189846, 7)

    def test_deriv_order(self):
        self.assertTrue(dft.libxc.test_deriv_order('lda', 3, raise_error=False))
        self.assertTrue(not dft.libxc.test_deriv_order('m05', 2, raise_error=False))
        self.assertRaises(NotImplementedError, dft.libxc.test_deriv_order, 'pbe0', 3, True)
        self.assertRaises(KeyError, dft.libxc.test_deriv_order, 'OL2', 3, True)

    def test_xc_type(self):
        self.assertEqual(dft.libxc.xc_type(416), 'GGA')
        self.assertEqual(dft.libxc.xc_type('hf'), 'HF')
        self.assertEqual(dft.libxc.xc_type(',vwn'), 'LDA')
        self.assertEqual(dft.libxc.xc_type('lda+b3lyp'), 'GGA')
        self.assertEqual(dft.libxc.xc_type('wb97m_v'), 'MGGA')
        self.assertEqual(dft.libxc.xc_type('bp86'), 'GGA')


if __name__ == "__main__":
    print("Test libxc")
    unittest.main()


