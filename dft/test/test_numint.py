#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import gen_grid
from pyscf.dft import radi
dft.numint.BLKSIZE = 12

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [('h', (0,0,i*3)) for i in range(12)]
mol.grids = {"H": (50, 110)}
mol.basis = 'ccpvtz'
mol.build()
mf = dft.RKS(mol)
mf.grids.setup_grids_()
nao = mol.nao_nr()

class KnowValues(unittest.TestCase):
    def test_make_mask(self):
        non0 = dft.numint.make_mask(mol, mf.grids.coords)
        self.assertEqual(non0.sum(), 71955)
        self.assertAlmostEqual(numpy.dot(non0.ravel(),
                                         numpy.cos(numpy.arange(non0.size))),
                               107.69227300597809, 9)
        self.assertAlmostEqual(numpy.dot(numpy.cos(non0).ravel(),
                                         numpy.cos(numpy.arange(non0.size))),
                               -48.536221663504378, 9)

    def test_dot_ao_dm(self):
        non0tab = dft.numint.make_mask(mol, mf.grids.coords)
        ao = dft.numint.eval_ao(mol, mf.grids.coords)
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        res0 = lib.dot(ao, dm)
        res1 = dft.numint._dot_ao_dm(mol, ao, dm, nao,
                                     mf.grids.weights.size, non0tab)
        self.assertTrue(numpy.allclose(res0, res1))

    def test_dot_ao_ao(self):
        non0tab = dft.numint.make_mask(mol, mf.grids.coords)
        ao = dft.numint.eval_ao(mol, mf.grids.coords, isgga=True)
        res0 = lib.dot(ao[0].T, ao[1])
        res1 = dft.numint._dot_ao_ao(mol, ao[0], ao[1], nao,
                                     mf.grids.weights.size, non0tab)
        self.assertTrue(numpy.allclose(res0, res1))

if __name__ == "__main__":
    print("Test numint")
    unittest.main()

