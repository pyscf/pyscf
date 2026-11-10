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
import scipy.linalg
import unittest
import unittest.mock
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.data import nist
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

    def test_x2c1e_hcore_lindep_fallback(self):
        # Deterministically cover the linear-dependency fallback of
        # _x2c1e_get_hcore on every platform: force the generalized eigh to
        # fail the way a non-positive-definite metric does, and verify the
        # fallback reproduces the direct-path hcore.
        mol1 = gto.M(atom='C', basis='cc-pvdz', verbose=0)
        x2c_obj = x2c.X2C(mol1)
        href = x2c_obj.get_hcore()
        eigh0 = scipy.linalg.eigh
        def eigh_no_generalized(a, b=None, *args, **kwargs):
            if b is not None:
                raise scipy.linalg.LinAlgError('forced generalized failure')
            return eigh0(a, *args, **kwargs)
        with unittest.mock.patch.object(scipy.linalg, 'eigh',
                                        eigh_no_generalized):
            hfb = x2c_obj.get_hcore()
        self.assertAlmostEqual(abs(href - hfb).max(), 0, 7)

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
        x2c_obj = x2c.X2C(mol)
        xmol, c = x2c_obj.get_xmol(mol)
        self.assertEqual(xmol.nbas, 18)
        self.assertEqual(xmol.nao, 42)
        self.assertAlmostEqual(lib.fp(c), -5.480689638416739, 12)

        hcore = x2c_obj.get_hcore()
        s = mol.intor_symmetric('int1e_ovlp_spinor')
        e_ref = scipy.linalg.eigvalsh(hcore, s)

        mol = gto.M(atom='C', basis=(mol.basis, [[0, [0.128500001, 1]]]))
        x2c_obj = x2c.X2C(mol)
        xmol, c = x2c_obj.get_xmol(mol)
        self.assertEqual(xmol.nbas, 19)
        self.assertEqual(xmol.nao, 43)
        hcore = x2c_obj.get_hcore()
        s = mol.intor_symmetric('int1e_ovlp_spinor')
        d, t = scipy.linalg.eigh(s)
        idx = d > 1e-8
        t = t[:,idx] / numpy.sqrt(d[idx])
        tht = t.T.conj().dot(hcore.dot(t))
        e = scipy.linalg.eigvalsh(tht)
        self.assertAlmostEqual(abs(e - e_ref).max(), 0, 6)

    def test_get_hcore(self):
        myx2c = scf.RHF(mol).sfx2c1e()
        myx2c.with_x2c.get_xmat = lambda xmol: numpy.zeros((xmol.nao, xmol.nao))
        h1 = myx2c.with_x2c.get_hcore()
        ref = mol.intor('int1e_nuc')
        self.assertAlmostEqual(abs(h1 - ref).max(), 0, 11)

        with_x2c = x2c.X2C(mol)
        with_x2c.get_xmat = lambda xmol: numpy.zeros((xmol.nao_2c(), xmol.nao_2c()))
        h1 = with_x2c.get_hcore()
        ref = mol.intor('int1e_nuc_spinor')
        self.assertAlmostEqual(abs(h1 - ref).max(), 0, 11)

    def test_ghf(self):
        # Test whether the result of spinor X2C is a solution of .GHF().x2c()
        mol = gto.M(atom='C', basis='ccpvdz-dk')
        ref = mol.DHF().x2c().run()
        c = numpy.vstack(mol.sph2spinor_coeff())
        mo1 = c.dot(ref.mo_coeff)
        dm = ref.make_rdm1(mo1, ref.mo_occ)
        mf = mol.GHF().x2c1e()
        mf.max_cycle = 1
        mf.kernel(dm0=dm)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 9)
        self.assertAlmostEqual(abs(mf.dip_moment() - ref.dip_moment()).max(), 0, 9)

    def test_ghf_atom(self):
        # Test whether the result of spinor X2C is a solution of .GHF().x2c()
        mol = gto.M(atom='C', basis='ccpvdz-dk')
        mf_1e = mol.GHF().x2c1e()
        mf_1e.kernel()
        mf_atom1e = mol.GHF().x2c1e()
        mf_atom1e.with_x2c.approx = 'ATOM1E'
        mf_atom1e.kernel()
        self.assertAlmostEqual(mf_1e.e_tot, mf_atom1e.e_tot, 9)
        self.assertAlmostEqual(abs(mf_1e.mo_energy - mf_atom1e.mo_energy).max(), 0, 9)

        with lib.temporary_env(lib.param, LIGHT_SPEED=15.):
            mol = gto.M(atom='Ne 0 1 -1; Ne 0 8 8', basis='ccpvdz')
            mf_1e = mol.GHF().x2c1e()
            mf_1e.kernel()
            mf_atom1e = mol.GHF().x2c1e()
            mf_atom1e.with_x2c.approx = 'ATOM1E'
            mf_atom1e.kernel()
            self.assertAlmostEqual(mf_1e.e_tot, -267.39699993561, 8)
            self.assertAlmostEqual(mf_1e.e_tot, mf_atom1e.e_tot, 8)
            self.assertAlmostEqual(abs(mf_1e.mo_energy - mf_atom1e.mo_energy).max(), 0, 6)

    def test_gks(self):
        mol = gto.M(atom='C', basis='ccpvdz-dk')
        ref = mol.DKS(xc='b3lyp').x2c().run()
        c = numpy.vstack(mol.sph2spinor_coeff())
        mo1 = c.dot(ref.mo_coeff)
        dm = ref.make_rdm1(mo1, ref.mo_occ)
        mf = mol.GKS(xc='b3lyp').x2c1e()
        mf.max_cycle = 1
        mf.kernel(dm0=dm)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 9)
        self.assertAlmostEqual(abs(mf.dip_moment() - ref.dip_moment()).max(), 0, 9)

        with lib.temporary_env(lib.param, LIGHT_SPEED=15.):
            mol = gto.M(atom='Ne 0 1 -1; Ne 0 8 8', basis='ccpvdz')
            mf_1e = mol.DKS().x2c1e()
            mf_1e.kernel()
            mf_atom1e = mol.DKS().x2c1e()
            mf_atom1e.with_x2c.approx = 'ATOM1E'
            mf_atom1e.kernel()
            self.assertAlmostEqual(mf_1e.e_tot, -266.688128052731, 8)
            self.assertAlmostEqual(mf_1e.e_tot, mf_atom1e.e_tot, 8)
            self.assertAlmostEqual(abs(mf_1e.mo_energy - mf_atom1e.mo_energy).max(), 0, 6)

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

    # issue 3295
    def test_to_ks(self):
        def check(mf, cls):
            assert isinstance(mf, x2c._X2C_SCF) and mf.istype(cls)

        mol = gto.M(atom='He')
        for cls in ('RHF', 'UHF', 'RKS', 'UKS'):
            mf = getattr(mol, cls)().sfx2c1e()
            check(mf.to_rks(), 'RKS')
            check(mf.to_uks(), 'UKS')
            check(mf.to_gks(), 'GKS')
            check(mf.to_rhf(), 'RHF')
            check(mf.to_uhf(), 'UHF')
            check(mf.to_ghf(), 'GHF')

        mf = mol.GHF().sfx2c1e()
        check(mf.to_ghf(), 'GHF')
        check(mf.to_gks(), 'GKS')
        mf = mol.GKS().sfx2c1e()
        check(mf.to_ghf(), 'GHF')
        check(mf.to_gks(), 'GKS')

    def test_soscf(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
        H     0.   0.    0.
        H     0.  -0.7   0.7
        H     0.   0.7   0.7'''
        mol.basis = '6-31g'
        mol.spin = 1
        mol.build()
        mf = mol.UKS(xc='lda,').run()
        mf1 = mol.UKS(xc='lda,').newton().run()
        self.assertAlmostEqual(mf.e_tot, mf1.e_tot, 9)


    def test_dipole_spin_representation(self):
        mol = gto.M(atom='F 0 0 0; H 0 0 .9', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).sfx2c1e().run(conv_tol=1e-11)
        ref = mf.dip_moment(unit='AU', verbose=0)
        numpy.testing.assert_allclose(mf.dip_moment(verbose=0), ref*nist.AU2DEBYE, atol=1e-10)
        for spin in (mf.to_uhf(), mf.to_ghf()):
            numpy.testing.assert_allclose(spin.dip_moment(unit='AU', verbose=0), ref, atol=1e-10)
        uncorrected = scf.hf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)
        numpy.testing.assert_allclose(mf.dip_moment(unit='AU', picture_change=False, verbose=0),
                                   uncorrected, atol=1e-10)
        moved = mol.set_geom_(mol.atom_coords()+[.3, -.2, .1], unit='Bohr', inplace=False)
        shifted = scf.RHF(moved).sfx2c1e()
        numpy.testing.assert_allclose(shifted.dip_moment(dm=mf.make_rdm1(), unit='AU', verbose=0),
                                   ref, atol=1e-9)


    def check_picture_change(self, spin_orbital=False, atomic=False):
        self.mol = gto.M(atom='F 0 0 0; H 0 0 .9', basis='sto-3g', verbose=0)
        cls = x2c.SpinOrbitalX2CHelper if spin_orbital else x2c.SpinorX2CHelper
        helper = cls(self.mol)
        helper.xuncontract = False
        if atomic:
            helper.approx = 'ATOM1E'
        c = lib.param.LIGHT_SPEED
        if spin_orbital:
            s = scipy.linalg.block_diag(*([self.mol.intor('int1e_ovlp')]*2))
            t = scipy.linalg.block_diag(*([self.mol.intor('int1e_kin')]*2))
            v = scipy.linalg.block_diag(*([self.mol.intor('int1e_nuc')]*2))
            w = x2c._sigma_dot(self.mol.intor('int1e_spnucsp'))
        else:
            s = self.mol.intor('int1e_ovlp_spinor')
            t = self.mol.intor('int1e_spsp_spinor')*.5
            v = self.mol.intor('int1e_nuc_spinor')
            w = self.mol.intor('int1e_spnucsp_spinor')
        metric = helper.picture_change((s, t/(2*c*c)))
        numpy.testing.assert_allclose(metric, s, atol=1e-10)
        potential = (v, w/(4*c*c)-t)
        h = helper.picture_change(potential, t)
        numpy.testing.assert_allclose(h, helper.get_hcore(), atol=1e-9)
        if not atomic:
            h4 = numpy.block([[v, t], [t, potential[1]]])
            s4 = scipy.linalg.block_diag(s, t/(2*c*c))
            spectrum = scipy.linalg.eigvalsh(h4, s4)[len(s):]
            numpy.testing.assert_allclose(scipy.linalg.eigvalsh(h, s), spectrum, atol=1e-9)
        # Vector-valued operators obey the same linear transformation.
        odd = numpy.array([t, t*.3, -t*.4])
        vector = helper.picture_change(odd_operator=odd)
        scalar = helper.picture_change(odd_operator=t)
        numpy.testing.assert_allclose(vector, numpy.array([scalar, scalar*.3, -scalar*.4]), atol=1e-11)

    def test_picture_change_spinor(self):
        self.check_picture_change()

    def test_picture_change_spin_orbital(self):
        self.check_picture_change(spin_orbital=True)

    def test_picture_change_atomic_spinor(self):
        self.check_picture_change(atomic=True)

    def test_picture_change_atomic_spin_orbital(self):
        self.check_picture_change(spin_orbital=True, atomic=True)

if __name__ == "__main__":
    print("Full Tests for x2c")
    unittest.main()
