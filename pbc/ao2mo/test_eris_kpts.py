import numpy as np
import pyscf.gto
import pyscf.pbc as pbc
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf.kscf as kscf
import pyscf.ao2mo
import eris
import scipy.linalg
pi=np.pi

def test_eris_kpts():
    L=5

    pseudo = None
    cell = pbcgto.Cell()
    cell.basis = { 'He': [[0, (1.0, 1.0)]] }
    cell.atom.extend([['He', (L/2.,L/2.,L/2.)], ])


    cell.output = '/dev/null'
    cell.verbose = 5
    cell.unit = 'B'
    cell.h = ((L,0,0),(0,L,0),(0,0,L))
    cell.gs = [20,20,20]
    cell.pseudo = pseudo
    cell.build()
    
    invhT=scipy.linalg.inv(cell._h.T)
    kGvs=[]

    ncells=3
    for i in range(ncells):
        kGvs.append(i*1./ncells*2*pi*np.dot(invhT,(1,0,0)))

    kpts=np.vstack(kGvs) 
    mf = kscf.KRKS(cell, cell.gs, cell.ew_eta, cell.ew_cut, kpts)
    mf.scf()

    mo_eris=eris.get_mo_eri_kpts(cell, kpts, mf.mo_coeff)
