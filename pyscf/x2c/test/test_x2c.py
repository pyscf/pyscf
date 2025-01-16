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

import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.x2c import x2c

def setUpModule():
    global mol
    mol = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = '''
            O     0    0        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_sfx2c1e(self):
        myx2c = scf.RHF(mol).sfx2c1e()
        myx2c.with_x2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.081765429967618, 9)

        myx2c.with_x2c.xuncontract = True
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075429077955874, 9)

        myx2c.with_x2c.approx = 'ATOM1E'
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075429682026396, 9)

    def test_sfx2c1e_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        myx2c = scf.RHF(pmol).sfx2c1e()
        myx2c.with_x2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.081452837461342, 9)

    def test_x2c1e(self):
        myx2c = x2c.UHF(mol)
        myx2c.with_x2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.08176796102066, 9)

        myx2c.with_x2c.xuncontract = True
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075431226329414, 9)

        myx2c.with_x2c.approx = 'ATOM1E'
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.07543183416206, 9)

    def test_picture_change(self):
        c = lib.param.LIGHT_SPEED
        myx2c = x2c.UHF(mol)
        myx2c.with_x2c.xuncontract = False

        def tv(with_x2c):
            xmol = with_x2c.get_xmol()[0]
            t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
            #v = xmol.intor_symmetric('int1e_nuc_spinor')
            w = xmol.intor_symmetric('int1e_spnucsp_spinor')
            return t, 'int1e_nuc_spinor', w

        t, v, w = tv(myx2c.with_x2c)
        h1 = myx2c.with_x2c.picture_change((v, w*(.5/c)**2-t), t)
        href = myx2c.with_x2c.get_hcore()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 9)

        myx2c.with_x2c.xuncontract = True
        t, v, w = tv(myx2c.with_x2c)
        h1 = myx2c.with_x2c.picture_change((v, w*(.5/c)**2-t), t)
        href = myx2c.with_x2c.get_hcore()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 9)

        myx2c.with_x2c.basis = 'unc-sto3g'
        t, v, w = tv(myx2c.with_x2c)
        h1 = myx2c.with_x2c.picture_change((v, w*(.5/c)**2-t), t)
        href = myx2c.with_x2c.get_hcore()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 9)

    def test_sfx2c1e_picture_change(self):
        c = lib.param.LIGHT_SPEED
        myx2c = scf.RHF(mol).sfx2c1e()
        myx2c.with_x2c.xuncontract = False

        def tv(with_x2c):
            xmol = with_x2c.get_xmol()[0]
            t = xmol.intor_symmetric('int1e_kin')
            w = xmol.intor_symmetric('int1e_pnucp')
            return t, 'int1e_nuc', w

        t, v, w = tv(myx2c.with_x2c)
        h1 = myx2c.with_x2c.picture_change((v, w*(.5/c)**2-t), t)
        href = myx2c.with_x2c.get_hcore()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 9)

        myx2c.with_x2c.xuncontract = True
        t, v, w = tv(myx2c.with_x2c)
        h1 = myx2c.with_x2c.picture_change((v, w*(.5/c)**2-t), t)
        href = myx2c.with_x2c.get_hcore()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 9)

        myx2c.with_x2c.basis = 'unc-sto3g'
        t, v, w = tv(myx2c.with_x2c)
        h1 = myx2c.with_x2c.picture_change((v, w*(.5/c)**2-t), t)
        href = myx2c.with_x2c.get_hcore()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 9)

    def test_lindep_xbasis(self):
        mol = gto.M(atom='C', basis='''
C     S 
    0.823600000E+04    0.677199997E-03
    0.123500000E+04    0.427849998E-02
    0.280800000E+03    0.213575999E-01
    0.792700000E+02    0.821857997E-01
    0.255900000E+02    0.235071499E+00
    0.899700000E+01    0.434261298E+00
    0.331900000E+01    0.345733299E+00
    0.905900000E+00    0.392976999E-01
    0.364300000E+00   -0.895469997E-02
    0.128500000E+00    0.237739999E-02
C     S 
    0.823600000E+04   -0.144499989E-03
    0.123500000E+04   -0.915599933E-03
    0.280800000E+03   -0.460309966E-02
    0.792700000E+02   -0.182283987E-01
    0.255900000E+02   -0.558689959E-01
    0.899700000E+01   -0.126988891E+00
    0.331900000E+01   -0.170104988E+00
    0.905900000E+00    0.140976590E+00
    0.364300000E+00    0.598675956E+00
    0.128500000E+00    0.394868571E+00
C     S 
    0.905900000E+00    0.100000000E+01
C     S 
    0.128500000E+00    0.100000000E+01
C     P 
    0.187100000E+02    0.140738004E-01
    0.413300000E+01    0.869016023E-01
    0.120000000E+01    0.290201608E+00
    0.382700000E+00    0.500903913E+00
    0.120900000E+00    0.343523809E+00
C     P 
    0.382700000E+00    0.100000000E+01
C     P 
    0.120900000E+00    0.100000000E+01
C     D 
    0.109700000E+01    0.100000000E+01
C     D 
    0.318000000E+00    0.100000000E+01
C     F 
    0.761000000E+00    0.100000000E+01
''')
        xmol, c = x2c.X2C(mol).get_xmol(mol)
        self.assertEqual(xmol.nbas, 18)
        self.assertEqual(xmol.nao, 42)
        self.assertAlmostEqual(lib.fp(c), -5.480689638416739, 12)

    def test_get_hcore(self):
        myx2c = scf.RHF(mol).sfx2c1e()
        myx2c.with_x2c.get_xmat = lambda xmol: numpy.zeros((xmol.nao, xmol.nao))
        h1 = myx2c.with_x2c.get_hcore()
        ref = mol.intor('int1e_nuc')
        self.assertAlmostEqual(abs(h1 - ref).max(), 0, 12)

        with_x2c = x2c.X2C(mol)
        with_x2c.get_xmat = lambda xmol: numpy.zeros((xmol.nao_2c(), xmol.nao_2c()))
        h1 = with_x2c.get_hcore()
        ref = mol.intor('int1e_nuc_spinor')
        self.assertAlmostEqual(abs(h1 - ref).max(), 0, 12)

    def test_ghf(self):
        # Test whether the result of .X2C() is a solution of .GHF().x2c()
        mol = gto.M(atom='C', basis='ccpvdz-dk')
        ref = mol.X2C().run()
        c = numpy.vstack(mol.sph2spinor_coeff())
        mo1 = c.dot(ref.mo_coeff)
        dm = ref.make_rdm1(mo1, ref.mo_occ)
        mf = mol.GHF().x2c1e()
        mf.max_cycle = 1
        mf.kernel(dm0=dm)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 9)
        self.assertAlmostEqual(abs(mf.dip_moment() - ref.dip_moment()).max(), 0, 9)

    def test_undo_x2c(self):
        mf = mol.RHF().x2c().density_fit()
        self.assertEqual(mf.__class__.__name__, 'DFsfX2C1eRHF')
        mf = mf.undo_x2c()
        self.assertEqual(mf.__class__.__name__, 'DFRHF')

        mf = mol.GHF().x2c().density_fit()
        self.assertEqual(mf.__class__.__name__, 'DFX2C1eGHF')
        mf = mf.undo_x2c()
        self.assertEqual(mf.__class__.__name__, 'DFGHF')

    # issue 2605
    def test_kappa_spinor(self):
        mol = gto.M(
            atom='''He 0.   0.7  .1
                    He 0.5 -0.2 -.1''',
            basis=[[0, [0.5547, 1.]],
                   [1, [11., 0.94, .31], [4.68, 0.22, .80]],
                   [1, -2, [1.9, 1.]],
                   [1, 1, [1.53, 1.]],
                   [1, 0, [0.53, 1.]]],
        )
        pmol, c = x2c.SpinorX2CHelper(mol).get_xmol()
        self.assertEqual(c.shape, (pmol.nao_2c(), mol.nao_2c()))
        ref = mol.intor('int1e_ovlp_spinor')
        dat = c.T.dot(pmol.intor('int1e_ovlp_spinor')).dot(c)
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 12)


if __name__ == "__main__":
    print("Full Tests for x2c")
    unittest.main()
