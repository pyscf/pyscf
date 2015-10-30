import numpy as np
import pyscf.pbc.scf.hf as pbchf
import pyscf.pbc.gto as pbcgto

def test_band():

    L = 1
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([10,10,10])

    cell.atom.extend([['He', (L/2.,L/2.,L/2.)]])
    cell.basis = { 'He': [[0, (1.0, 1.0)]] }

    cell.build()

    mf = pbchf.RHF(cell)
    mf.scf()

    auxcell = cell.copy()
    auxcell.gs = np.array([1,1,1])
    auxcell.build()

    # print auxcell.Gv.shape
    # print auxcell.Gv
    # print "---", auxcell.Gv[:,-1]

    for i in range(1,10):
        kpt = 1./i * auxcell.Gv[:,-1]
        #print kpt
        print pbchf.get_eig_kpt(mf, kpt)[0]
