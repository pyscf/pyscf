import unittest
import numpy as np

from pyscf import gto
from pyscf.scf import hf

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf


def make_cell1(L, n):
    mol = gto.Mole()
    mol.unit = 'B'
    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
# these are some exponents which are not hard to integrate
    mol.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    mol.verbose = 0
    mol.build()

    pseudo = None
    cell = pbcgto.Cell()
    # cell.output = '/dev/null'
    cell.verbose = 5
    cell.unit = mol.unit
    cell.h = ((L,0,0),(0,L,0),(0,0,L))
    cell.gs = [n,n,n]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()
    return mol, cell

def make_cell2(L, n):
    mol = gto.M(atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                        ['He', (L/2.   ,L/2.,L/2.+.5)]],
                basis = { 'He': [[0, (0.8, 1.0)],
                                 [0, (1.0, 1.0)],
                                 [0, (1.2, 1.0)]] },
                unit = 'B',
                verbose = 0)

    cell = pbcgto.Cell()
               #output = '/dev/null',
    cell.build(unit = mol.unit,
               h = ((L,0,0),(0,L,0),(0,0,L)),
               gs = [n,n,n],
               atom = mol.atom,
               basis = mol.basis)
    return mol, cell

def finger(mat):
    w = np.cos(np.arange(mat.size))
    return np.dot(mat.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_numint20(self):
        mol, cell = make_cell1(20, 20)
        mf = pbchf.RHF(cell)
        self.assertAlmostEqual(mf.scf(), -2.4607103378280013, 8)

#    def test_numint(self):
#        mol, cell = make_cell1(60, 120)
#        mf = pbchf.RHF(cell)
## python 2.6, numpy 1.6.2, mkl 10.3 got -2.5877571064424578
#        self.assertAlmostEqual(mf.scf(), -2.58766850182551, 8)

    def test_kinetic20(self):
        mol, cell = make_cell2(20, 20)
        tao = pbchf.get_t(cell)
        self.assertAlmostEqual(finger(tao), -2.0092386485871367, 9)

#    def test_kinetic80(self):
#        mol, cell = make_cell2(20, 80)
#        tao = pbchf.get_t(cell)
#        tao2 = mol.intor_symmetric('cint1e_kin_sph')
#        self.assertTrue(np.allclose(tao, tao2))

    def test_overlap20(self):
        mol, cell = make_cell2(20, 20)
        sao = pbchf.get_ovlp(cell)
        self.assertAlmostEqual(finger(sao), -0.8503588722723614, 9)

#    def test_overlap80(self):
#        mol, cell = make_cell2(20, 80)
#        sao = pbchf.get_ovlp(cell)
#        sao2 = mol.intor_symmetric('cint1e_ovlp_sph')
#        self.assertTrue(np.allclose(sao, sao2))

    def test_j20(self):
        mol, cell = make_cell2(20, 20)
        mf = hf.RHF(mol)
        mf.kernel()
        dm = mf.make_rdm1()
        jao = pbchf.get_j(cell, dm)
        self.assertAlmostEqual(finger(jao), -2.6245387435710494, 9)

#    def test_j80(self):
#        mol, cell = make_cell2(40, 100)
#        mf = hf.RHF(mol)
#        mf.kernel()
#        dm = mf.make_rdm1()
#        jao = pbchf.get_j(cell, dm)
#        jao2 = mf.get_j(mol, dm)
## should *not* match, since G=0 component is removed
#        print abs(jao-jao2).sum()
#        #self.assertTrue(np.allclose(jao, jao2))

    def test_nuc_el20(self):
        mol, cell = make_cell2(20, 20)
        neao = pbchf.get_nuc(cell)
        self.assertAlmostEqual(finger(neao), 3.93800680639802, 9)

    def test_nuc_el_withpp20(self):
        mol, cell = make_cell2(20, 20)
        cell.pseudo = 'gth-lda'
        cell.build(0, 0)
        vppao = pbchf.get_pp(cell)
        self.assertAlmostEqual(finger(vppao), 3.9555741129610578, 9)

#    def test_e_coul(self):
#        mol, cell = make_cell1(40, 100)
#        mf = hf.RHF(mol)
#        mf.kernel()
#        dm = mf.make_rdm1()
#        ecoul0 =(np.einsum('ij,ij', dm, mf.get_j(dm)) * .5
#               + np.einsum('ij,ij', dm, mol.intor_symmetric('cint1e_nuc_sph')))
#
#        ew_cut = (40,40,40)
#        ew_eta = 0.1
#        ew = pbchf.ewald(cell, ew_eta, ew_cut)
#        jao = pbchf.get_j(cell, dm)
#        neao = pbchf.get_nuc(cell)
#        ecoul1 = np.einsum("ij,ij", dm, neao + .5*jao) + ew
#        self.assertAlmostEqual(ecoul0, ecoul1, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
