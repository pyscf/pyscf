import unittest
import numpy as np

from pyscf import gto
from pyscf.dft import rks

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import rks as pbcrks
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint


mol = gto.Mole()
mol.unit = 'B'
L = 60
mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
# these are some exponents which are not hard to integrate
mol.basis = { 'He': [[0, (0.8, 1.0)],
                     [0, (1.0, 1.0)],
                     [0, (1.2, 1.0)]] }
mol.verbose = 5
mol.output = '/dev/null'
mol.build()

m = rks.RKS(mol)
m.xc = 'LDA,VWN_RPA'
e0 =  (m.scf()) # -2.64096172441

def make_grids(n):
    pseudo = None
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])
    cell.nimgs = [0,0,0]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()
    grids = gen_grid.UniformGrids(cell)
    grids.setup_grids_()
    return cell, grids

class KnowValues(unittest.TestCase):
    def test_eval_ao(self):
        cell, grids = make_grids(30)
        ao1 = numint.eval_ao(cell, grids.coords)
        w = np.arange(ao1.size) * .01
        self.assertAlmostEqual(np.dot(w,ao1.ravel()), (44072.276638371265+0j), 8)

#    def test_eval_ao_gga(self):
#        cell, grids = make_grids(30)
#        ao1 = numint.eval_ao(cell, grids.coords, isgga=True)
#        w = np.arange(ao1.size)
#        self.assertAlmostEqual(np.dot(w,ao1), 0, 8)
#
#    def test_eval_rho(self):
#        self.assertAlmostEqual(e1, 0, 8)
#
    def test_eval_mat(self):
        cell, grids = make_grids(30)
        ng = grids.weights.size
        np.random.seed(1)
        rho = np.random.random(ng)
        rho *= 1/np.linalg.norm(rho)
        vrho = np.random.random(ng)
        ao1 = numint.eval_ao(cell, grids.coords)
        mat1 = numint.eval_mat(mol, ao1, grids.weights, rho, vrho)
        w = np.arange(mat1.size) * .01
        self.assertAlmostEqual(np.dot(w,mat1.ravel()), (.14777107967912118+0j), 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()

