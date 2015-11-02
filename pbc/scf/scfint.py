'''
SCF (Hartree-Fock and DFT) tools for periodic systems at a *single* k-point,
    using analytical GTO integrals instead of PWs.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.

'''

import numpy as np
import scipy.linalg
import pyscf.scf
import pyscf.scf.hf
import pyscf.dft
import pyscf.gto
import pyscf.lib
import pyscf.lib.parameters as param
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo

from pyscf.lib import logger

def get_lattice_Ls(cell, nimgs):
    '''Get the (unitful) lattice translation vectors for nearby images.'''
    #nimgs = cell.nimgs
    Ts = [[i,j,k] for i in range(-nimgs[0],nimgs[0]+1)
                  for j in range(-nimgs[1],nimgs[1]+1)
                  for k in range(-nimgs[2],nimgs[2]+1)
                  if i**2+j**2+k**2 <= 1./3*np.dot(nimgs,nimgs)]
    Ts = np.array(Ts)
    Ls = np.dot(cell._h, Ts.T).T
    return Ls

def get_hcore(cell, kpt=None):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.'''
    if kpt is None:
        kpt = np.zeros([3,1])

    # TODO: these are still on grid
    if cell.pseudo is None:
        hcore = pyscf.pbc.scf.hf.get_nuc(cell, kpt)
    else:
        hcore = (pyscf.pbc.scf.hf.get_pp(cell, kpt) + 
                 get_jvloc_G0(cell, kpt))
    hcore += get_t(cell, kpt)

    # hcore = get_t(cell, kpt)
    return hcore

def get_jvloc_G0(cell, kpt=None):
    '''Get the (separately) divergent Hartree + Vloc G=0 contribution.'''

    return 1./cell.vol * np.sum(pseudo.get_alphas(cell)) * get_ovlp(cell, kpt)

def get_int1e(intor, cell, kpt=None):
    '''Get the one-electron integral defined by `intor` using lattice sums.'''
    if kpt is None:
        kpt = np.zeros([3,1])

    # if pyscf.lib.norm(kpt) == 0.:
    #     dtype = np.float64
    # else:
    dtype = np.complex128

    int1e = np.zeros((cell.nao_nr(),cell.nao_nr()), dtype=dtype)

    Ls = get_lattice_Ls(cell, cell.nimgs)

# Just change the basis position, keep all other envrionments
    cellL = cell.copy()
    ptr_coord = cellL._atm[:,pyscf.gto.PTR_COORD]
    _envL = cellL._env
    for L in Ls:
        _envL[ptr_coord+0] = cell._env[ptr_coord+0] + L[0]
        _envL[ptr_coord+1] = cell._env[ptr_coord+1] + L[1]
        _envL[ptr_coord+2] = cell._env[ptr_coord+2] + L[2]
        int1e += (np.exp(1j*np.dot(kpt.T,L)) *
                  pyscf.gto.intor_cross(intor, cell, cellL))
    return int1e

def get_ovlp(cell, kpt=None):
    '''Get the overlap AO matrix.'''
    if kpt is None:
        kpt = np.zeros([3,1])

    s = get_int1e('cint1e_ovlp_sph', cell, kpt) 
    return s

def get_t(cell, kpt=None):
    '''Get the kinetic energy AO matrix.'''
    if kpt is None:
        kpt = np.zeros([3,1])
    
    t = get_int1e('cint1e_kin_sph', cell, kpt) 
    return t

def test_periodic_ints():
    from pyscf import gto
    from pyscf.lib.parameters import BOHR
    import pyscf.pbc.gto as pgto
    import scf

    B = BOHR

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    Lunit = 3
    Ly = Lz = Lunit
    Lx = Lunit

    h = np.diag([Lx,Ly,Lz])
    
    # mol.atom.extend([['He', (2*B, 0.5*Ly*B, 0.5*Lz*B)],
    #                  ['He', (3*B, 0.5*Ly*B, 0.5*Lz*B)]])
    #mol.atom.extend([['H', (0, 0, 0)],
    #                  ['H', (1, 0, 0)]])


    #mol.atom
    mol.build(
        verbose = 0,
        atom = '''H     0    0.       0.
        H     1    0.       0.
        ''',
        basis={'H':'sto-3g'})
    #    basis={'H':[[0,(1.0,1.0)]]})

    # these are some exponents which are 
    # not hard to integrate
    # mol.basis = { 'He': [[0, (1.0, 1.0)], [0, [2.0, 1.0]]] }
    #mol.basis = { 'H': [[0, (1.0, 1.0)], [0, [2.0, 1.0]]] }
    #mol.unit='A'
    #mol.build()

    cell = pgto.Cell()
    cell.__dict__ = mol.__dict__ # hacky way to make a cell
    cell.h = h
    #cell.vol = scipy.linalg.det(cell.h)
    cell.nimgs = gto.get_nimgs(cell, 1.e-6)
    # print "NIMG",  
    print "NIMGS", cell.nimgs
    cell.pseudo = None
    cell.output = None
    cell.verbose = 0
    cell.build()

    gs = np.array([20,20,20])

    # Analytic
    sA=get_ovlp(cell)
    tA=get_t(cell)

    # Grid
    sG=scf.get_ovlp(cell, gs)
    tG=scf.get_t(cell, gs)

    # These differences should be 0 up to grid integration error
    print "Diff", np.linalg.norm(sA-sG) # 1.05796568891e-06
    print "Diff", np.linalg.norm(tA-tG) # 4.82330435721e-06

    print sA
    print tA
