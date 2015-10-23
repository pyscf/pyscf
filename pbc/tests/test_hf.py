import numpy as np

from pyscf import gto
from pyscf.scf import hf

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf

def test_hf(pseudo=None):
    # The molecular calculation
    mol = gto.Mole()
    mol.unit = 'B'
    L = 60
    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    # these are some exponents which are not hard to integrate
    mol.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    mol.build()

    m = hf.RHF(mol)
    print "Molecular HF energy"
    print (m.scf()) # -2.63502450321874

    # The periodic calculation
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([80,80,80])
    cell.nimgs = [0,0,0]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()

    mf = pbchf.RHF(cell)

    print (mf.scf()) # -2.58766850182551: doesn't look good, but this is due
                     # to interaction of the exchange hole with its periodic
                     # image, which can only be removed with *very* large boxes.

    # Now try molecular type integrals for the exchange operator, 
    # and periodic integrals for Coulomb. This effectively
    # truncates the exchange operator. 
    mf.mol_ex = True 
    print (mf.scf()) # -2.63493445685: much better!

if __name__ == '__main__':
    test_hf()
    #test_hf('gth-lda')
