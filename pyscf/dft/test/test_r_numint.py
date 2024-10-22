#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import r_numint, xc_deriv
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '6-31g'
    mol.build()

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_eval_rho(self):
        n2c = mol.nao_2c()
        numpy.random.seed(10)
        ngrids = 100
        coords = numpy.random.random((ngrids,3))*2
        coords = coords[70:75]
        dm = numpy.random.random((n2c,n2c))
        dm = dm + dm.T.conj()
        aoLa, aoLb, aoSa, aoSb = r_numint.eval_ao(mol, coords, deriv=1)

        rho0a = numpy.einsum('pi,ij,pj->p', aoLa[0], dm, aoLa[0].conj())
        rho0b = numpy.einsum('pi,ij,pj->p', aoLb[0], dm, aoLb[0].conj())
        rho0 = rho0a + rho0b

        aoL = numpy.array([aoLa[0],aoLb[0]])
        m0 = numpy.einsum('api,ji,bpj,xab->xp', aoL.conj(), dm, aoL, lib.PauliMatrices)

        ni = r_numint.RNumInt()
        rho1 = ni.eval_rho(mol, (aoLa[0], aoLb[0]), dm, xctype='LDA')
        self.assertAlmostEqual(abs(rho1[0].imag).max(), 0, 9)
        self.assertAlmostEqual(abs(rho0-rho1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(m0 - rho1[1:4]).max(), 0, 9)

    def test_rsh_omega(self):
        rho0 = numpy.array([[1., 1., 0.1, 0.1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [.1, .1, 0.01, .01]]).reshape(4, 4, 1)
        ni = r_numint.RNumInt()
        ni.omega = 0.4
        omega = 0.2
        exc, vxc, fxc, kxc = ni.eval_xc_eff('ITYH,', rho0, deriv=1, omega=omega)
        vxc = xc_deriv.ud2ts(vxc)
        self.assertAlmostEqual(exc[0], -0.6376259665301467, 7)
        #self.assertAlmostEqual(float(vxc[0][0,0]), -0.8688965017309331, 7)  # libxc-4.3.4
        #self.assertAlmostEqual(float(vxc[0][0,1]), -0.04641346660681983, 7)  # libxc-4.3.4
        self.assertAlmostEqual(vxc[0,0,0], -0.8701119430520298, 7)  # libxc-5.1.2
        self.assertAlmostEqual(vxc[1,0,0], -0.032830587216387985, 7)  # libxc-5.1.2
        # vsigma of GGA may be problematic?
        #?self.assertAlmostEqual(float(vxc[1][0,0]), 0, 7)
        #?self.assertAlmostEqual(float(vxc[1][0,1]), 0, 7)

        exc, vxc, fxc, kxc = ni.eval_xc_eff('ITYH,', rho0, deriv=1)
        vxc = xc_deriv.ud2ts(vxc)
        self.assertAlmostEqual(exc[0], -0.542221740505985, 7)
        #self.assertAlmostEqual(float(vxc[0][0,0]), -0.7699824959456474, 7)  # libxc-4.3.4
        #self.assertAlmostEqual(float(vxc[0][0,1]), -0.04529004028228567, 7)  # libxc-4.3.4
        self.assertAlmostEqual(vxc[0,0,0], -0.7710640750107329, 7)  # libxc-5.1.2
        self.assertAlmostEqual(vxc[1,0,0], -0.032344719811289835, 7)  # libxc-5.1.2
        # vsigma of GGA may be problematic?
        #?self.assertAlmostEqual(float(vxc[1][0,0]), 0, 7)
        #?self.assertAlmostEqual(float(vxc[1][0,1]), 0, 7)


    def test_vxc1(self):
        mol = gto.M(atom=[
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            basis='6311g*')
        mf = dft.dks.DKS(mol)
        mf.grids.atom_grid = {"H": (30, 194), "O": (30, 194),}
        mf.grids.prune = None
        mf.grids.build()
        dm = mf.get_init_guess(key='minao')
        mf._numint.collinear = 'col'
        res = mf._numint.get_vxc(mol, mf.grids, mf.xc, dm, spin=0)
        self.assertAlmostEqual(res[1], -8.631807003163278, 11)

        mf._numint.collinear = 'ncol'
        res = mf._numint.get_vxc(mol, mf.grids, mf.xc, dm, spin=0)
        self.assertAlmostEqual(res[1], -8.631807003163278, 11)

    def test_vxc_col(self):
        ni = r_numint.RNumInt()
        ni.collinear = 'c'
        mf = mol.DKS()
        dm = mf.get_init_guess(mol, 'minao')
        n, e, v = ni.get_vxc(mol, mf.grids, 'B88,', dm)
        self.assertAlmostEqual(n, 9.984666945, 5)
        self.assertAlmostEqual(e, -8.8304689765, 6)
        self.assertAlmostEqual(lib.fp(v), 0.05882536730070975+0.37093946710262393j, 8)

    def test_vxc_ncol(self):
        ni = r_numint.RNumInt()
        ni.collinear = 'n'
        mf = mol.DKS()
        dm = mf.get_init_guess(mol, 'minao')
        n, e, v = ni.get_vxc(mol, mf.grids, 'LDA,', dm)
        self.assertAlmostEqual(n, 9.984666945, 5)
        self.assertAlmostEqual(e, -7.9613152012, 6)
        self.assertAlmostEqual(lib.fp(v), 0.09939730962851079+0.35765389167296235j, 8)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_vxc_mcol(self):
        ni = r_numint.RNumInt()
        ni.collinear = 'm'
        ni.spin_samples = 14
        mf = mol.DKS()
        dm = mf.get_init_guess(mol, 'minao')
        n, e, v = ni.get_vxc(mol, mf.grids, 'LDA,', dm)
        self.assertAlmostEqual(n, 9.984666945, 5)
        self.assertAlmostEqual(e, -7.9613152012, 6)
        self.assertAlmostEqual(lib.fp(v), 0.09939730962851079+0.35765389167296235j, 8)

        n, e, v = ni.get_vxc(mol, mf.grids, 'B88,', dm)
        self.assertAlmostEqual(n, 9.984666945, 5)
        self.assertAlmostEqual(e, -8.8304689765, 6)
        self.assertAlmostEqual(lib.fp(v), 0.058825367300710196+0.37093946710262404j, 8)

    def test_fxc_col(self):
        ni = r_numint.RNumInt()
        ni.collinear = 'c'
        mf = mol.DKS()
        dm = mf.get_init_guess(mol, 'minao')
        numpy.random.seed(10)
        dm1 = numpy.random.random(dm.shape)
        v = ni.get_fxc(mol, mf.grids, 'B88,', dm, dm1)
        self.assertAlmostEqual(lib.fp(v), -0.6265667390145981+0.5244965520637875j, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_fxc_mcol(self):
        ni = r_numint.RNumInt()
        ni.collinear = 'm'
        ni.spin_samples = 14
        mf = mol.DKS()
        dm = mf.get_init_guess(mol, 'minao')
        numpy.random.seed(10)
        dm1 = numpy.random.random(dm.shape)
        v = ni.get_fxc(mol, mf.grids, 'LDA,', dm, dm1)
        self.assertAlmostEqual(lib.fp(v), -1.9850498137636299+1.4541338353513784j, 6)

        v = ni.get_fxc(mol, mf.grids, 'M06', dm, dm1)
        self.assertAlmostEqual(lib.fp(v), -1.0864202540818795+0.06981358086231704j, 6)

    def test_get_rho(self):
        ni = r_numint.RNumInt()
        ni.collinear = 'c'
        mf = mol.DKS()
        grids = mf.grids.build()
        dm = mf.get_init_guess(mol, 'minao')
        rho = ni.get_rho(mol, dm, grids)
        self.assertAlmostEqual(lib.fp(rho), -361.4682369790235, 8)

        ni.collinear = 'm'
        ni.spin_samples = 50
        rho = ni.get_rho(mol, dm, grids)
        self.assertAlmostEqual(lib.fp(rho), -361.4682369790235, 8)


if __name__ == "__main__":
    print("Test r_numint")
    unittest.main()
