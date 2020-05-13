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

import unittest
import numpy
from pyscf import gto, lib, scf
from pyscf.prop import gtensor
from pyscf.data import nist

def make_dia_gc2e(gobj, dm0, gauge_orig, sso_qed_fac=1):
    mol = gobj.mol
    dma, dmb = dm0
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    alpha2 = nist.ALPHA ** 2
    #sso_qed_fac = (nist.G_ELECTRON - 1)
    nao = dma.shape[0]

    # int2e_ip1v_r1 = (ij|\frac{\vec{r}_{12}}{r_{12}^3} \vec{r}_1|kl)
    if gauge_orig is None:
        gc2e_ri = mol.intor('int2e_ip1v_r1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
    else:
        with mol.with_common_origin(gauge_orig):
            gc2e_ri = mol.intor('int2e_ip1v_rc1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
    ej = numpy.zeros((3,3))
    ek = numpy.zeros((3,3))
    if isinstance(gobj.para_soc2e, str) and 'SSO' in gobj.dia_soc2e.upper():
        # spin-density should be contracted to electron 1 (associated to operator r_i)
        ej += sso_qed_fac * numpy.einsum('xyijkl,ji,lk->xy', gc2e_ri, dma-dmb, dma+dmb)
        ek += sso_qed_fac * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dma, dma)
        ek -= sso_qed_fac * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dmb, dmb)
    if isinstance(gobj.para_soc2e, str) and 'SOO' in gobj.dia_soc2e.upper():
        # spin-density should be contracted to electron 2
        ej += 2 * numpy.einsum('xyijkl,ji,lk->xy', gc2e_ri, dma+dmb, dma-dmb)
        ek += 2 * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dma, dma)
        ek -= 2 * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dmb, dmb)
    gc2e = ej - ek
    gc2e -= numpy.eye(3) * gc2e.trace()
    gc2e *= (alpha2/8) / effspin / muB

    # giao2e1 = ([GIAO-i j] + [i GIAO-j]|\frac{\vec{r}_{12}}{r_{12}^3} x p1|kl)
    # giao2e2 = (ij|\frac{\vec{r}_{12}}{r_{12}^3} x p1|[GIAO-k l] + [k GIAO-l])
    if gauge_orig is None:
        giao2e1 = mol.intor('int2e_ipvg1_xp1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
        giao2e2 = mol.intor('int2e_ipvg2_xp1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
        giao2e = giao2e1 + giao2e2.transpose(1,0,2,3,4,5)
        ej = numpy.zeros((3,3))
        ek = numpy.zeros((3,3))
        if isinstance(gobj.para_soc2e, str) and 'SSO' in gobj.dia_soc2e.upper():
            ej += sso_qed_fac * numpy.einsum('xyijkl,ji,lk->xy', giao2e, dma-dmb, dma+dmb)
            ek += sso_qed_fac * numpy.einsum('xyijkl,jk,li->xy', giao2e, dma, dma)
            ek -= sso_qed_fac * numpy.einsum('xyijkl,jk,li->xy', giao2e, dmb, dmb)
        if isinstance(gobj.para_soc2e, str) and 'SOO' in gobj.dia_soc2e.upper():
            ej += 2 * numpy.einsum('xyijkl,ji,lk->xy', giao2e, dma+dmb, dma-dmb)
            ek += 2 * numpy.einsum('xyijkl,jk,li->xy', giao2e, dma, dma)
            ek -= 2 * numpy.einsum('xyijkl,jk,li->xy', giao2e, dmb, dmb)
        gc2e -= (ej - ek) * (alpha2/4) / effspin / muB

    if gobj.mb:  # correction of order c^{-2} from MB basis, does it exist?
        vj, vk = gobj._scf.get_jk(mol, dm0)
        vhf = vj[0] + vj[1] - vk
        gc_mb = numpy.einsum('ij,ji', vhf[0], dma)
        gc_mb-= numpy.einsum('ij,ji', vhf[1], dmb)
        gc2e += gc_mb * (alpha2/4) / effspin / muB * numpy.eye(3)

    return gc2e

def make_para_soc2e(gobj, dm0, dm10, sso_qed_fac=1):
    mol = gobj.mol
    alpha2 = nist.ALPHA ** 2
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    #sso_qed_fac = (nist.G_ELECTRON - 1)

    dm0a, dm0b = dm0
    dm10a, dm10b = dm10
    nao = dm0a.shape[0]

# hso2e is the imaginary part of SSO
# SSO term of JCP 122, 034107 (2005); DOI:10.1063/1.1829047 Eq (3) = 1/4c^2 hso2e
#
# Different approximations for the spin operator part are used in
# JCP 122, 034107 (2005) Eq (15) and JCP 115, 11080 (2001) Eq (34).  The formulae of the
# so-called spin-averaging in JCP 122, 034107 (2005) Eq (15) is not well documented
# and its effects are not fully tested.  Approximation of JCP 115, 11080 (2001) Eq (34)
# are adopted here.
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(3,nao,nao,nao,nao)
    ej = numpy.zeros((3,3))
    ek = numpy.zeros((3,3))
    if isinstance(gobj.para_soc2e, str) and 'SSO' in gobj.para_soc2e.upper():
        ej += sso_qed_fac * numpy.einsum('yijkl,ji,xlk->xy', hso2e, dm0a-dm0b, dm10a+dm10b)
        ej += sso_qed_fac * numpy.einsum('yijkl,xji,lk->xy', hso2e, dm10a-dm10b, dm0a+dm0b)
        ek += sso_qed_fac * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0a, dm10a)
        ek -= sso_qed_fac * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0b, dm10b)
        ek += sso_qed_fac * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10a, dm0a)
        ek -= sso_qed_fac * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10b, dm0b)
    if isinstance(gobj.para_soc2e, str) and 'SOO' in gobj.para_soc2e.upper():
        ej += 2 * numpy.einsum('yijkl,ji,xlk->xy', hso2e, dm0a+dm0b, dm10a-dm10b)
        ej += 2 * numpy.einsum('yijkl,xji,lk->xy', hso2e, dm10a+dm10b, dm0a-dm0b)
        ek += 2 * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0a, dm10a)
        ek -= 2 * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0b, dm10b)
        ek += 2 * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10a, dm0a)
        ek -= 2 * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10b, dm0b)
# ~ <H^{01},MO^1> = - Tr(Im[H^{01}],Im[MO^1])
    gpara2e = -(ej - ek)
    gpara2e *= (alpha2/4) / effspin / muB
    return gpara2e


mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'
mol.atom = '''
    H  0. , 0. , .917
    F  0. , 0. , 0.'''
mol.basis = 'ccpvdz'
mol.spin = 2
mol.build()

nrhf = scf.UHF(mol)
nrhf.conv_tol_grad = 1e-6
nrhf.conv_tol = 1e-12
nrhf.kernel()

nao = mol.nao_nr()
numpy.random.seed(1)
dm0 = numpy.random.random((2,nao,nao))
dm0 = dm0 + dm0.transpose(0,2,1)
dm1 = numpy.random.random((2,3,nao,nao))
dm1 = dm1 - dm1.transpose(0,1,3,2)


class KnowValues(unittest.TestCase):
    def test_nr_common_gauge_dia_gc2e(self):
        g = gtensor.uhf.GTensor(nrhf)
        g.dia_soc2e = 'SSO+SOO'
        g.para_soc2e = 'SSO+SOO'
        g.mb = True
        ref = make_dia_gc2e(g, dm0, (1.2, .3, .5), 1)
        dat = g.make_dia_gc2e(dm0, (1.2, .3, .5), 1)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_nr_giao_dia_gc2e(self):
        g = gtensor.uhf.GTensor(nrhf)
        g.dia_soc2e = 'SSO+SOO'
        g.para_soc2e = 'SSO+SOO'
        g.mb = True
        ref = make_dia_gc2e(g, dm0, None, 1)
        dat = g.make_dia_gc2e(dm0, None, 1)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_nr_para_soc2e(self):
        g = gtensor.uhf.GTensor(nrhf)
        ref = make_para_soc2e(g, dm0, dm1, 1)
        dat = g.make_para_soc2e(dm0, dm1, 1)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_nr_uhf(self):
        g = gtensor.uhf.GTensor(nrhf)
        g.dia_soc2e = 'SSO+SOO'
        g.para_soc2e = 'SSO+SOO'
        g.so_eff_charge = True
        g.cphf = False
        g.mb = True
        dat = g.kernel()
        self.assertAlmostEqual(numpy.linalg.norm(dat), 3.46802309158, 7)


if __name__ == "__main__":
    print("Full Tests for HF g-tensor")
    unittest.main()
