#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import r_numint

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
    def test_eval_rho(self):
        n2c = mol.nao_2c()
        numpy.random.seed(10)
        ngrids = 100
        coords = numpy.random.random((ngrids,3))*20
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
        rho1, m1 = ni.eval_rho(mol, (aoLa[0], aoLb[0]), dm, xctype='LDA')
        self.assertAlmostEqual(abs(rho1.imag).max(), 0, 9)
        self.assertAlmostEqual(abs(rho0-rho1).max(), 0, 9)
        self.assertTrue(numpy.allclose(m0, m1))

    def test_eval_mat(self):
        numpy.random.seed(10)
        ngrids = 100
        coords = numpy.random.random((ngrids,3))*20
        rho = numpy.random.random((ngrids))
        m = numpy.random.random((3,ngrids)) * .05
        vxc = [numpy.random.random((2,ngrids)).T, None, None, None]
        weight = numpy.random.random(ngrids)
        aoLa, aoLb, aoSa, aoSb = r_numint.eval_ao(mol, coords, deriv=1)

        s = numpy.linalg.norm(m, axis=0)
        m_pauli = numpy.einsum('xp,xij,p->pij', m, lib.PauliMatrices, 1./(s+1e-300))
        aoL = numpy.array([aoLa[0],aoLb[0]])

        mat0 = numpy.einsum('pi,p,pj->ij', aoLa[0].conj(), weight*vxc[0][:,0], aoLa[0])
        mat0+= numpy.einsum('pi,p,pj->ij', aoLb[0].conj(), weight*vxc[0][:,0], aoLb[0])
        mat0+= numpy.einsum('api,p,pab,bpj->ij', aoL.conj(), weight*vxc[0][:,1], m_pauli, aoL)
        mat1 = r_numint.eval_mat(mol, (aoLa[0], aoLb[0]), weight, (rho, m), vxc, xctype='LDA')
        self.assertTrue(numpy.allclose(mat0, mat1))

#    def test_vxc(self):
#    def test_fxc(self):

if __name__ == "__main__":
    print("Test r_numint")
    unittest.main()

