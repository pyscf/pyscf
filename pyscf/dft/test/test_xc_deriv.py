#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft import xc_deriv

def eval_xc(xctype, ng):
    np.random.seed(1)
    outbuf = np.random.rand(220,ng)
    exc = outbuf[0]
    if xctype == 'R-LDA':
        vxc = [outbuf[1]]
        fxc = [outbuf[2]]
        kxc = [outbuf[3]]
    elif xctype == 'R-GGA':
        vxc = [outbuf[1], outbuf[2]]
        fxc = [outbuf[3], outbuf[4], outbuf[5]]
        kxc = [outbuf[6], outbuf[7], outbuf[8], outbuf[9]]
    elif xctype == 'U-LDA':
        vxc = [outbuf[1:3].T]
        fxc = [outbuf[3:6].T]
        kxc = [outbuf[6:10].T]
    elif xctype == 'U-GGA':
        vxc = [outbuf[1:3].T, outbuf[3:6].T]
        fxc = [outbuf[6:9].T, outbuf[9:15].T, outbuf[15:21].T]
        kxc = [outbuf[21:25].T, outbuf[25:34].T, outbuf[34:46].T, outbuf[46:56].T]
    elif xctype == 'R-MGGA':
        vxc = [outbuf[1], outbuf[2], None, outbuf[4]]
        fxc = [
            # v2rho2, v2rhosigma, v2sigma2,
            outbuf[5], outbuf[6], outbuf[7],
            # v2lapl2, v2tau2,
            None, outbuf[9],
            # v2rholapl, v2rhotau,
            None, outbuf[11],
            # v2lapltau, v2sigmalapl, v2sigmatau,
            None, None, outbuf[14]]
        # v3lapltau2 might not be strictly 0
        # outbuf[18] = 0
        kxc = [
            # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
            outbuf[15], outbuf[16], outbuf[17], outbuf[18],
            # v3rho2lapl, v3rho2tau,
            None, outbuf[20],
            # v3rhosigmalapl, v3rhosigmatau,
            None, outbuf[22],
            # v3rholapl2, v3rholapltau, v3rhotau2,
            None, None, outbuf[25],
            # v3sigma2lapl, v3sigma2tau,
            None, outbuf[27],
            # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
            None, None, outbuf[30],
            # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
            None, None, None, outbuf[34]]
    elif xctype == 'U-MGGA':
        vxc = [outbuf[1:3].T, outbuf[3:6].T, None, outbuf[8:10].T]
        # v2lapltau might not be strictly 0
        # outbuf[39:43] = 0
        fxc = [
            # v2rho2, v2rhosigma, v2sigma2,
            outbuf[10:13].T, outbuf[13:19].T, outbuf[19:25].T,
            # v2lapl2, v2tau2,
            None, outbuf[28:31].T,
            # v2rholapl, v2rhotau,
            None, outbuf[35:39].T,
            # v2lapltau, v2sigmalapl, v2sigmatau,
            None, None, outbuf[49:55].T]
        # v3lapltau2 might not be strictly 0
        # outbuf[204:216] = 0
        kxc = [
            # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
            outbuf[55:59].T, outbuf[59:68].T, outbuf[68:80].T, outbuf[80:90].T,
            # v3rho2lapl, v3rho2tau,
            None, outbuf[96:102].T,
            # v3rhosigmalapl, v3rhosigmatau,
            None, outbuf[114:126].T,
            # v3rholapl2, v3rholapltau, v3rhotau2,
            None, None, outbuf[140:146].T,
            # v3sigma2lapl, v3sigma2tau,
            None, outbuf[158:170].T,
            # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
            None, None, outbuf[191:200].T,
            # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
            None, None, None, outbuf[216:220].T]
    return exc, vxc, fxc, kxc

def v6to5(v6):
    if v6.ndim == 2:
        v5 = v6[[0,1,2,3,5]]
    else:
        v5 = v6[:,[0,1,2,3,5]]
    return v5

def v5to6(v5):
    if v5.ndim == 2:
        v6 = np.zeros((6, v5.shape[1]))
        v6[[0,1,2,3,5]] = v5
    else:
        v6 = np.zeros((2, 6, v5.shape[2]))
        v6[:,[0,1,2,3,5]] = v5
    return v6

def eval_xc_eff(xc_code, rho, deriv, mod):
    xctype = mod.xc_type(xc_code)
    rhop = np.asarray(rho)

    if xctype == 'LDA':
        spin_polarized = rhop.ndim >= 2
    else:
        spin_polarized = rhop.ndim == 3

    if spin_polarized:
        assert rhop.shape[0] == 2
        spin = 1
        if rhop.ndim == 3 and rhop.shape[1] == 5:  # MGGA
            ngrids = rhop.shape[2]
            rhop = np.empty((2, 6, ngrids))
            rhop[0,:4] = rho[0][:4]
            rhop[1,:4] = rho[1][:4]
            rhop[:,4] = 0
            rhop[0,5] = rho[0][4]
            rhop[1,5] = rho[1][4]
    else:
        spin = 0
        if rhop.ndim == 2 and rhop.shape[0] == 5:  # MGGA
            ngrids = rho.shape[1]
            rhop = np.empty((6, ngrids))
            rhop[:4] = rho[:4]
            rhop[4] = 0
            rhop[5] = rho[4]

    exc, vxc, fxc, kxc = mod.eval_xc(xc_code, rhop, spin, 0, deriv)
    if deriv > 2:
        kxc = xc_deriv.transform_kxc(rhop, fxc, kxc, xctype, spin)
    if deriv > 1:
        fxc = xc_deriv.transform_fxc(rhop, vxc, fxc, xctype, spin)
    if deriv > 0:
        vxc = xc_deriv.transform_vxc(rhop, vxc, xctype, spin)
    return exc, vxc, fxc, kxc

def setUpModule():
    global rho
    rho = np.array(
        [[[ 0.17283732, 0.17272921, 0.17244017, 0.17181541, 0.17062690],
          [-0.01025988,-0.02423402,-0.04315779,-0.06753381,-0.09742367],
          [ 0.00219947, 0.00222727, 0.00226589, 0.00231774, 0.00238570],
          [ 0.00151577, 0.00153381, 0.00155893, 0.00159277, 0.00163734],
          [ 0.00323925, 0.00386831, 0.00520072, 0.00774571, 0.01218266]],
         [[ 0.17443331, 0.17436845, 0.17413969, 0.17359613, 0.17251427],
          [-0.00341093,-0.01727580,-0.03605226,-0.06023877,-0.08989467],
          [ 0.00357202, 0.00361952, 0.00368537, 0.00377355, 0.00388873],
          [ 0.00233614, 0.00236578, 0.00240683, 0.00246173, 0.00253334],
          [ 0.00473343, 0.00533707, 0.00663345, 0.00912920, 0.01350123],]]
    )

class KnownValues(unittest.TestCase):
    def test_gga_deriv1(self):
        ng = 7
        xctype = 'GGA'
        np.random.seed(8)
        rho = np.random.rand(2,4,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = numint._rks_gga_wv0(rho[0], vxc, weight)
        ref[0] *= 2
        v1  = xc_deriv.transform_vxc(rho[0], vxc, xctype, spin=0)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = np.array(numint._uks_gga_wv0(rho, vxc, weight))
        ref[:,0] *= 2
        v1  = xc_deriv.transform_vxc(rho, vxc, xctype, spin=1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_gga_deriv2(self):
        ng = 7
        xctype = 'GGA'
        np.random.seed(8)
        rho = np.random.rand(2,4,ng)
        rho1 = np.random.rand(2,4,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = numint._rks_gga_wv1(rho[0], rho1[0], vxc, fxc, weight)
        ref[0] *= 2
        v1 = xc_deriv.transform_fxc(rho[0], vxc, fxc, xctype, spin=0)
        v1 = np.einsum('xg,xyg->yg', rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = np.array(numint._uks_gga_wv1(rho, rho1, vxc, fxc, weight))
        ref[:,0] *= 2
        v1  = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin=1)
        v1 = np.einsum('axg,axbyg->byg', rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_gga_deriv3(self):
        ng = 7
        xctype = 'GGA'
        np.random.seed(8)
        rho = np.random.rand(2,4,ng)
        rho1 = np.random.rand(2,4,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = numint._rks_gga_wv2(rho[0], rho1[0], fxc, kxc, weight)
        ref[0] *= 2
        v1 = xc_deriv.transform_kxc(rho[0], fxc, kxc, xctype, spin=0)
        v1 = np.einsum('xg,yg,xyzg->zg', rho1[0], rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = np.array(numint._uks_gga_wv2(rho, rho1, fxc, kxc, weight))
        ref[:,0] *= 2
        v1  = xc_deriv.transform_kxc(rho, fxc, kxc, xctype, spin=1)
        v1 = np.einsum('axg,byg,axbyczg->czg', rho1, rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_mgga_deriv1(self):
        ng = 7
        xctype = 'MGGA'
        np.random.seed(8)
        rho = np.random.rand(2,5,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = v6to5(numint._rks_mgga_wv0(v5to6(rho[0]), vxc, weight))
        ref[0] *= 2
        ref[4] *= 4
        v1  = xc_deriv.transform_vxc(rho[0], vxc, xctype, spin=0)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = v6to5(np.array(numint._uks_mgga_wv0(v5to6(rho), vxc, weight)))
        ref[:,0] *= 2
        ref[:,4] *= 4
        v1  = xc_deriv.transform_vxc(rho, vxc, xctype, spin=1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_mgga_deriv2(self):
        ng = 7
        xctype = 'MGGA'
        np.random.seed(8)
        rho = np.random.rand(2,5,ng)
        rho1 = np.random.rand(2,5,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = v6to5(numint._rks_mgga_wv1(v5to6(rho[0]), v5to6(rho1[0]), vxc, fxc, weight))
        ref[0] *= 2
        ref[4] *= 4
        v1 = xc_deriv.transform_fxc(rho[0], vxc, fxc, xctype, spin=0)
        v1 = np.einsum('xg,xyg->yg', rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = v6to5(np.array(numint._uks_mgga_wv1(v5to6(rho), v5to6(rho1), vxc, fxc, weight)))
        ref[:,0] *= 2
        ref[:,4] *= 4
        v1  = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin=1)
        v1 = np.einsum('axg,axbyg->byg', rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_mgga_deriv3(self):
        ng = 7
        xctype = 'MGGA'
        np.random.seed(8)
        rho = np.random.rand(2,5,ng)
        rho1 = np.random.rand(2,5,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = v6to5(numint._rks_mgga_wv2(v5to6(rho[0]), v5to6(rho1[0]), fxc, kxc, weight))
        ref[0] *= 2
        ref[4] *= 4
        v1 = xc_deriv.transform_kxc(rho[0], fxc, kxc, xctype, spin=0)
        v1 = np.einsum('xg,yg,xyzg->zg', rho1[0], rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = v6to5(np.array(numint._uks_mgga_wv2(v5to6(rho), v5to6(rho1), fxc, kxc, weight)))
        ref[:,0] *= 2
        ref[:,4] *= 4
        v1  = xc_deriv.transform_kxc(rho, fxc, kxc, xctype, spin=1)
        v1 = np.einsum('axg,byg,axbyczg->czg', rho1, rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_ud2ts(self):
        c = np.array([[.5,  .5],    # vrho = (va + vb) / 2
                      [.5, -.5]])   # vs   = (va - vb) / 2
        np.random.seed(8)
        v_ud = np.random.rand(2,4,7)
        f_ud = np.random.rand(2,4,2,4,7)
        k_ud = np.random.rand(2,4,2,4,2,4,7)
        v_ts = np.einsum('ra,axg->rxg', c, v_ud)
        f_ts = np.einsum('ra,axbyg->rxbyg', c, f_ud)
        f_ts = np.einsum('sb,rxbyg->rxsyg', c, f_ts)
        k_ts = np.einsum('ra,axbyczg->rxbyczg', c, k_ud)
        k_ts = np.einsum('sb,rxbyczg->rxsyczg', c, k_ts)
        k_ts = np.einsum('tc,rxsyczg->rxsytzg', c, k_ts)
        self.assertAlmostEqual(abs(xc_deriv.ud2ts(v_ud) - v_ts).max(), 0, 12)
        self.assertAlmostEqual(abs(xc_deriv.ud2ts(f_ud) - f_ts).max(), 0, 12)
        self.assertAlmostEqual(abs(xc_deriv.ud2ts(k_ud) - k_ts).max(), 0, 12)

    def test_libxc_lda_deriv3(self):
        rho1 = rho[:,0].copy()
        ref = eval_xc_eff('LDA,', rho1, 3, dft.libxc)

        xc1 = dft.libxc.eval_xc_eff('LDA,', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 51.36053114469969, 9)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('LDA,', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -13.323225829690143, 9)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('LDA,', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -6.912554696220437, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

        rho1 = rho[1,0].copy()
        ref = eval_xc_eff('LDA,', rho1, 3, dft.libxc)

        xc1 = dft.libxc.eval_xc_eff('LDA,', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 20.21333987261437, 9)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('LDA,', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -5.269784014086463, 9)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('LDA,', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -2.7477984980958627, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

    def test_libxc_gga_deriv3(self):
        rho1 = rho[:,:4].copy()
        ref = eval_xc_eff('PBE', rho1, 3, dft.libxc)

        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 61.29042037001073, 3)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -13.896034377219816, 4)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -7.616226587554259, 6)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

        rho1 = rho[1,:4].copy()
        ni = numint.NumInt()
        ref = eval_xc_eff('PBE', rho1, 3, dft.libxc)

        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 26.08081046374974, 3)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -5.559303849017572, 4)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -3.0715856471099032, 6)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

    def test_libxc_mgga_deriv3(self):
        rho1 = rho
        ref = eval_xc_eff('M06', rho1, 3, dft.libxc)

        xc1 = dft.libxc.eval_xc_eff('M06', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 3461867.985594323, 1)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('M06', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -19196.865088253828, 3)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('M06', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), 90.99262909378264, 6)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

        rho1 = rho[1]
        ni = numint.NumInt()
        ref = eval_xc_eff('M06', rho1, 3, dft.libxc)

        xc1 = dft.libxc.eval_xc_eff('M06', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 2506574.915698602, 1)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('M06', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -9308.64852580393, 3)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.libxc.eval_xc_eff('M06', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), 19.977512805950784, 7)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

    @unittest.skipIf(dft.libxc.max_deriv_order('pbe,') <= 3, 'libxc order')
    def test_libxc_gga_deriv4(self):
        rho1 = rho[:,:4].copy()
        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=4)
        self.assertAlmostEqual(xc1.sum(), -1141.356286780069, 1)

        rho1 = rho[1,:4].copy()
        xc1 = dft.libxc.eval_xc_eff('PBE', rho1, deriv=4)
        self.assertAlmostEqual(xc1.sum(), -615.116081052867, 1)

    @unittest.skipIf(not hasattr(dft, 'xcfun'), 'xcfun order')
    def test_xcfun_lda_deriv3(self):
        rho1 = rho[:,0].copy()
        ref = eval_xc_eff('LDA,', rho1, 3, dft.xcfun)

        xc1 = dft.xcfun.eval_xc_eff('LDA,', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 51.36053114469969, 9)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('LDA,', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -13.323225829690143, 9)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('LDA,', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -6.912554696220437, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

        rho1 = rho[1,0].copy()
        ref = eval_xc_eff('LDA,', rho1, 3, dft.xcfun)

        xc1 = dft.xcfun.eval_xc_eff('LDA,', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 20.21333987261437, 9)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('LDA,', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -5.269784014086463, 9)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('LDA,', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -2.7477984980958627, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

    @unittest.skipIf(not hasattr(dft, 'xcfun'), 'xcfun order')
    def test_xcfun_gga_deriv3(self):
        rho1 = rho[:,:4].copy()
        ref = eval_xc_eff('PBE', rho1, 3, dft.xcfun)

        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 61.29042037001073, 9)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -13.896034377219816, 9)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -7.616226587554259, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

        rho1 = rho[1,:4].copy()
        ref = eval_xc_eff('PBE', rho1, 3, dft.xcfun)

        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 26.08081046374974, 9)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -5.559303849017572, 9)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), -3.0715856471099032, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

    @unittest.skipIf(not hasattr(dft, 'xcfun'), 'xcfun order')
    def test_xcfun_mgga_deriv3(self):
        rho1 = rho
        ref = eval_xc_eff('M06', rho1, 3, dft.xcfun)

        xc1 = dft.xcfun.eval_xc_eff('M06', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 3461867.985594323, 5)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('M06', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -19196.865088253828, 5)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('M06', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), 90.99262909378264, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

        rho1 = rho[1]
        ref = eval_xc_eff('M06', rho1, 3, dft.xcfun)

        xc1 = dft.xcfun.eval_xc_eff('M06', rho1, deriv=3)
        self.assertAlmostEqual(xc1.sum(), 2506574.915698602, 5)
        self.assertAlmostEqual(abs(ref[3] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('M06', rho1, deriv=2)
        self.assertAlmostEqual(xc1.sum(), -9308.64852580393, 5)
        self.assertAlmostEqual(abs(ref[2] - xc1).max(), 0, 9)

        xc1 = dft.xcfun.eval_xc_eff('M06', rho1, deriv=1)
        self.assertAlmostEqual(xc1.sum(), 19.977512805950784, 9)
        self.assertAlmostEqual(abs(ref[1] - xc1).max(), 0, 9)

    @unittest.skipIf(not (hasattr(dft, 'xcfun') and dft.xcfun.MAX_DERIV_ORDER > 3), 'xcfun order')
    def test_xcfun_gga_deriv4(self):
        rho1 = rho[:,:4].copy()
        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=4)
        self.assertAlmostEqual(xc1.sum(), -1141.356286780069, 9)

        rho1 = rho[1,:4].copy()
        xc1 = dft.xcfun.eval_xc_eff('PBE', rho1, deriv=4)
        self.assertAlmostEqual(xc1.sum(), -615.116081052867, 9)

    @unittest.skipIf(not (hasattr(dft, 'xcfun') and dft.xcfun.MAX_DERIV_ORDER > 3), 'xcfun order')
    def test_xcfun_gga_deriv4_finite_diff(self):
        xctype = 'GGA'
        deriv = 4
        nvar = 4
        delta = 1e-6

        spin = 1
        rhop = rho[:,:nvar].copy()
        xcp = dft.xcfun.eval_xc1('pbe,', rhop, spin, deriv=deriv)
        lxc = xc_deriv.transform_xc(rhop, xcp, xctype, spin,4)
        for s in (0, 1):
            for t in range(nvar):
                rhop = rho[:,:nvar].copy()
                rhop[s,t] += delta * .5
                xcp = dft.xcfun.eval_xc1('pbe,', rhop, spin, deriv=deriv-1)
                kxc0 = xc_deriv.transform_xc(rhop, xcp, xctype, spin, deriv-1)
                rhop[s,t] -= delta
                xcp = dft.xcfun.eval_xc1('pbe,', rhop, spin, deriv=deriv-1)
                kxc1 = xc_deriv.transform_xc(rhop, xcp, xctype, spin, deriv-1)
                self.assertAlmostEqual(abs((kxc0-kxc1)/delta - lxc[s,t]).max(), 0, 7)

        spin = 0
        rhop = rho[0,:nvar].copy()
        xcp = dft.xcfun.eval_xc1('b88,', rhop, spin, deriv=deriv)
        lxc = xc_deriv.transform_xc(rhop, xcp, xctype, spin,4)
        for t in range(nvar):
            rhop = rho[0,:nvar].copy()
            rhop[t] += delta * .5
            xcp = dft.xcfun.eval_xc1('b88,', rhop, spin, deriv=deriv-1)
            kxc0 = xc_deriv.transform_xc(rhop, xcp, xctype, spin, deriv-1)
            rhop[t] -= delta
            xcp = dft.xcfun.eval_xc1('b88,', rhop, spin, deriv=deriv-1)
            kxc1 = xc_deriv.transform_xc(rhop, xcp, xctype, spin, deriv-1)
            self.assertAlmostEqual(abs((kxc0-kxc1)/delta - lxc[t]).max(), 0, 7)

    def test_diagonal_indices(self):
        nabla_idx = [1, 2, 3]

        def equiv_diagonal_indices(idx, order):
            # this function is equivalent to xc_deriv._diagonal_indices
            # it is less efficient but probably more intuitive
            len_idx = len(idx)
            indices = np.arange(len_idx**(2 * order)).reshape([len_idx] * (2 * order))
            # diagonalize all pairs
            # e.g. [(0, 1), (2, 3), (4, 5)] -> [(2, 3), (4, 5), diag01] ->
            #      [(4, 5), diag01, diag23] -> [diag01, diag23, diag45]    (when order = 3)
            for _ in range(order):
                indices = indices.diagonal(axis1=0, axis2=1)
            # retrive diagonal indices [diag01, diag23, diag45] from original [(0, 1), (2, 3), (4, 5)]
            indices = np.unravel_index(indices.reshape(-1), shape=[len_idx] * (2 * order))
            # map to original indices
            return tuple([np.asarray(idx)[i] for i in indices])

        # case of order = 2, corresponds to 4th/5th xc derivative
        # this order is more like n_pairs, related to the order of xc derivative but not the same
        order = 2
        indices = xc_deriv._diagonal_indices(nabla_idx, order)
        equiv_indices = equiv_diagonal_indices(nabla_idx, order)
        for (idx, equiv_idx) in zip(indices, equiv_indices):
            self.assertTrue((idx == equiv_idx).all())

        # case of order = 4, corresponds to 8th/9th xc derivative
        order = 4
        indices = xc_deriv._diagonal_indices(nabla_idx, order)
        equiv_indices = equiv_diagonal_indices(nabla_idx, order)
        for (idx, equiv_idx) in zip(indices, equiv_indices):
            self.assertTrue((idx == equiv_idx).all())

if __name__ == "__main__":
    print("Test xc_deriv")
    unittest.main()
