#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf import lib

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [('h', (0,0,i*3)) for i in range(12)]
mol.basis = 'ccpvtz'
mol.build()
mf = dft.RKS(mol)
mf.grids.atom_grid = {"H": (50, 110)}
mf.prune = None
mf.grids.build()
nao = mol.nao_nr()
ao_loc = mol.ao_loc_nr()

def finger(a):
    return numpy.dot(numpy.cos(numpy.arange(a.size)), a.ravel())

class KnowValues(unittest.TestCase):
    def test_make_mask(self):
        non0 = dft.numint.make_mask(mol, mf.grids.coords)
        self.assertEqual(non0.sum(), 19622)
        self.assertAlmostEqual(finger(non0),
                               1.3143191152489875, 9)
        self.assertAlmostEqual(finger(numpy.cos(non0)),
                               -0.55017664009781386, 9)

    def test_dot_ao_dm(self):
        non0tab = dft.numint.make_mask(mol, mf.grids.coords)
        ao = dft.numint.eval_ao(mol, mf.grids.coords)
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        res0 = lib.dot(ao, dm)
        res1 = dft.numint._dot_ao_dm(mol, ao, dm, non0tab,
                                     shls_slice=(0,mol.nbas), ao_loc=ao_loc)
        self.assertTrue(numpy.allclose(res0, res1))

    def test_dot_ao_ao(self):
        non0tab = dft.numint.make_mask(mol, mf.grids.coords)
        ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
        res0 = lib.dot(ao[0].T, ao[1])
        res1 = dft.numint._dot_ao_ao(mol, ao[0], ao[1], non0tab,
                                     shls_slice=(0,mol.nbas), ao_loc=ao_loc)
        self.assertTrue(numpy.allclose(res0, res1))

    def test_eval_rho(self):
        numpy.random.seed(10)
        ngrids = 500
        coords = numpy.random.random((ngrids,3))*20
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ao = dft.numint.eval_ao(mol, coords, deriv=2)

        rho0 = numpy.zeros((6,ngrids))
        rho0[0] = numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[0].conj())
        rho0[1] = numpy.einsum('pi,ij,pj->p', ao[1], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[1].conj())
        rho0[2] = numpy.einsum('pi,ij,pj->p', ao[2], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[2].conj())
        rho0[3] = numpy.einsum('pi,ij,pj->p', ao[3], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[3].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[4].conj()) + numpy.einsum('pi,ij,pj->p', ao[4], dm, ao[0].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[7].conj()) + numpy.einsum('pi,ij,pj->p', ao[7], dm, ao[0].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[9].conj()) + numpy.einsum('pi,ij,pj->p', ao[9], dm, ao[0].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[1], dm, ao[1].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[2], dm, ao[2].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[3], dm, ao[3].conj())
        rho0[4]+= rho0[5]*2
        rho0[5] *= .5

        rho1 = dft.numint.eval_rho(mol, ao, dm, xctype='MGGA')
        self.assertTrue(numpy.allclose(rho0, rho1))

    def test_eval_mat(self):
        numpy.random.seed(10)
        ngrids = 500
        coords = numpy.random.random((ngrids,3))*20
        rho = numpy.random.random((6,ngrids))
        vxc = numpy.random.random((4,ngrids))
        weight = numpy.random.random(ngrids)
        ao = dft.numint.eval_ao(mol, coords, deriv=2)

        mat0 = numpy.einsum('pi,p,pj->ij', ao[0].conj(), weight*vxc[0], ao[0])
        mat1 = dft.numint.eval_mat(mol, ao[0], weight, rho, vxc, xctype='LDA')
        self.assertTrue(numpy.allclose(mat0, mat1))

        vrho, vsigma = vxc[:2]
        wv = weight * vsigma * 2
        mat0  = numpy.einsum('pi,p,pj->ij', ao[0].conj(), weight*vrho, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[1]*wv, ao[1]) + numpy.einsum('pi,p,pj->ij', ao[1].conj(), rho[1]*wv, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[2]*wv, ao[2]) + numpy.einsum('pi,p,pj->ij', ao[2].conj(), rho[2]*wv, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[3]*wv, ao[3]) + numpy.einsum('pi,p,pj->ij', ao[3].conj(), rho[3]*wv, ao[0])
        mat1 = dft.numint.eval_mat(mol, ao, weight, rho, vxc, xctype='GGA')
        self.assertTrue(numpy.allclose(mat0, mat1))

if __name__ == "__main__":
    print("Test numint")
    unittest.main()

