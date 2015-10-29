import numpy as np
import pyscf.gto
import pyscf.pbc as pbc
import pyscf.pbc.gto as pbcgto
import eris
import pyscf.scf.hf as hf
import pyscf.pbc.scf.hf as pbchf
import pyscf.ao2mo

def test_eris():
    L=10

    pseudo = None
    cell = pbcgto.Cell()
    cell.basis = { 'He': [[0, (0.8, 1.0)],
                          [0, (1.0, 1.0)],
                          [0, (1.2, 1.0)]] }
    cell.atom.extend([['He', (L/2.,L/2.,L/2.)], ])

    cell.output = '/dev/null'
    cell.verbose = 5
    cell.unit = 'B'
    cell.h = ((L,0,0),(0,L,0),(0,0,L))
    cell.gs = [40,40,40]
    cell.pseudo = pseudo
    cell.build()

    mf = pbchf.RHF(cell)
    mf.scf()

    mo_coeff = mf.mo_coeff
    # Generate MO integrals by direct evaluation on grid
    # (i.e. DF by k-space grid)
    eri_mo = eris.get_mo_eri(cell, 
             [mf.mo_coeff, mf.mo_coeff], [mf.mo_coeff, mf.mo_coeff])
    
    # Generate integrals by integral transformation
    # of periodic AO integrals
    eri_ao = eris.get_ao_eri(cell)
    eri_mo2 = pyscf.ao2mo.incore.general(np.real(np.ascontiguousarray(eri_ao)), 
              (mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mf.mo_coeff), compact=False)

    nmo=mo_coeff.shape[1]
    print "ERI dimension", eri_mo.shape, nmo
    Ej = Ek = 0.
    Ej2 = Ek2 = 0.
    nocc = 1

    for i in range(nocc):
        for j in range(nocc):
            Ej += 2*eri_mo[i*nmo+i,j*nmo+j]
            Ek += - eri_mo[i*nmo+j,i*nmo+j]
            Ej2 += 2*eri_mo2[i*nmo+i,j*nmo+j] 
            Ek2 += - eri_mo2[i*nmo+j,i*nmo+j]

    # Note that this is not equal to the molecular Coulomb/Exchange
    # energy, since the G=0 component is missing (except in the limit L \to \infty
    print "Coulomb/Exchange HF energy from ao2mo transformation",  Ej, Ek
    print "Coulomb/Exchange HF energy from MO evaluation on grid", Ej2, Ek2
    print "Diff (should be zero)", Ej-Ej2,Ek - Ek2

