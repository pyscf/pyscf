import numpy as np
import scipy.linalg
import pyscf.pbc.scf.hf as pbchf
import pyscf.pbc.scf.kscf as pbckscf
import pyscf.pbc.gto as pbcgto

pi = np.pi

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

    for i in range(1,10):
        kpt = 1./i * auxcell.Gv[-1,:]
        print pbchf.get_eig_kpt(mf, kpt)[0]

def test_band_kscf():
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

    invhT = scipy.linalg.inv(np.asarray(cell._h).T)

    ncells = 2
    kGvs = []
    for i in range(ncells):
        kGvs.append(i*1./ncells*2*pi*np.dot(invhT,(1,0,0)))
    kpts = np.vstack(kGvs)

    kmf = pbckscf.KRKS(cell, cell.gs, cell.ew_eta, cell.ew_cut, kpts)
    kmf.init_guess = "atom"
    print kmf.scf()

    for i in range(1,10):
        band_kpt = 1./i * auxcell.Gv[-1,:]
        print pbchf.get_eig_kpt(kmf, band_kpt)[0]

