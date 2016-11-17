import numpy as np

from pyscf import gto
from pyscf.scf import hf
from pyscf import ao2mo

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf
from pyscf.pbc import ao2mo as pbcao2mo

def test_moints():
    # not yet working

    # The molecular calculation
    mol = gto.Mole()
    mol.unit = 'B'
    L = 60
    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    mol.basis = 'cc-pvdz'
    mol.build()

    # The periodic calculation
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.a = np.diag([L,L,L])
    cell.gs = np.array([40,40,40])

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.build()

    #mf = hf.RHF(mol)
    mf = pbchf.RHF(cell)

    print (mf.scf()) 

    print "mo coeff shape", mf.mo_coeff.shape
    nmo = mf.mo_coeff.shape[1]
    print mf.mo_coeff

    eri_mo = pbcao2mo.get_mo_eri(cell, 
             [mf.mo_coeff, mf.mo_coeff], [mf.mo_coeff, mf.mo_coeff])
    
    eri_ao = pbcao2mo.get_ao_eri(cell)
    eri_mo2 = ao2mo.incore.general(np.real(eri_ao), 
              (mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mf.mo_coeff), compact=False)
    print eri_mo.shape
    print eri_mo2.shape
    for i in range(nmo*nmo):
        for j in range(nmo*nmo):
            print i, j, np.real(eri_mo[i,j]), eri_mo2[i,j]


    print ("ERI dimension")
    print (eri_mo.shape), nmo
    Ecoul = 0.
    Ecoul2 = 0.
    nocc = 1

    print "diffs"
    for i in range(nocc):
        for j in range(nocc):
            Ecoul += 2*eri_mo[i*nmo+i,j*nmo+j] - eri_mo[i*nmo+j,i*nmo+j]
            Ecoul2 += 2*eri_mo2[i*nmo+i,j*nmo+j] - eri_mo2[i*nmo+j,i*nmo+j]
    print Ecoul, Ecoul2

if __name__ == '__main__':
    test_moints()
