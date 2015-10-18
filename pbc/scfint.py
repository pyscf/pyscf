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
import cell as cl
import pbc
import pp

from pyscf.lib import logger

def get_lattice_Ls(cell):
    '''Get the (unitful) lattice translation vectors for nearby images.'''
    nimgs = cell.nimgs
    Ts = [[i,j,k] for i in range(-nimgs[0],nimgs[0]+1)
                  for j in range(-nimgs[1],nimgs[1]+1)
                  for k in range(-nimgs[2],nimgs[2]+1)
                  if i**2+j**2+k**2 <= 1./3*np.dot(nimgs,nimgs)]
    Ts = np.array(Ts)
    Ls = np.dot(cell.h, Ts.T).T
    return Ls

def get_hcore(mf, cell, kpt=None):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.'''
    if kpt is None:
        kpt = np.zeros([3,1])

    if cell.pseudo is None:
        hcore = get_nuc(cell, kpt)
    else:
        hcore = get_pp(cell, kpt)
    hcore += get_t(cell, kpt)
    return hcore

def get_int1e(intor, cell, kpt=None):
    '''Get the one-electron integral defined by `intor` using lattice sums.'''
    if kpt is None:
        kpt = np.zeros([3,1])

    int1e = np.zeros((cell.nbas,cell.nbas))
    
    Ls = get_lattice_Ls(cell)
    for L in Ls:
        cellL = cell.copy()
        atomL = list()
        for atom, coord in cell.atom:
            atomL.append([atom, tuple(list(coord) + L)])
        cellL.atom = atomL
        cellL.build()
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

