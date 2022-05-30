#!/usr/bin/env python
#
# written by Zhihao Cui <zcui@caltech.edu>
#            Qiming Sun <osirpt.sun@gmail.com>
#

'''
HF method for 2D Hubbard model.
'''

import numpy as np
from pyscf import pbc

#
# Define problem
#

t=1.0
U=12 # t/U=0.077 use for FM case

Nx=2
Ny=2
Nele=4
Nkx=15
Nky=15
Nk=Nkx*Nky
Nsite=Nx*Ny

#
# Create an empty cell in real space (attribute cell.a), and assign certain
# number of particles to the cell (cell.nelectron).
#
cell = pbc.gto.M(
     unit='B',
     a=[[Nx*1.0,0.0   ,0.0],
        [0.0   ,Ny*1.0,0.0],
        [0.0   ,0.0   ,1.0]],
     verbose=4,
)
cell.nelectron=Nele
kpts = cell.make_kpts([Nkx,Nky,1])


#
# Functions to generate 1-particle Hamiltonian Hcore
#

### generate TB Hamiltonian at a specific k point.

def gen_H_tb(t,Nx,Ny,kvec):
    H = np.zeros((Nx,Ny,Nx,Ny),dtype=np.complex128)
    for i in range(Nx):
        for j in range(Ny):
            if i == Nx-1:
                H[i,j,0   ,j] += np.exp(-1j*np.dot(np.array(kvec),np.array([Nx,0])))
            else:
                H[i,j,i+1 ,j] += 1

            if i == 0:
                H[i,j,Nx-1,j] += np.exp(-1j*np.dot(np.array(kvec),np.array([-Nx,0])))
            else:
                H[i,j,i-1 ,j] += 1

            if j == Ny-1:
                H[i,j,i,0   ] += np.exp(-1j*np.dot(np.array(kvec),np.array([0,Ny])))
            else:
                H[i,j,i,j+1] += 1

            if j == 0:
                H[i,j,i,Ny-1] += np.exp(-1j*np.dot(np.array(kvec),np.array([0,-Ny])))
            else:
                H[i,j,i,j-1] += 1
    return -t*H.reshape(Nx*Ny,Nx*Ny)

### get H_tb at a series of kpoints.

def get_H_tb_array(kpts,Nx,Ny,t):
    H_tb_array=[]
    for kpt in kpts:
        H_tb = gen_H_tb(t, Nx, Ny, kpt[:2])
        H_tb_array.append(H_tb)
    return np.array(H_tb_array)


#
# Initialize k-points UHF calculation and overwriting the hcore, overlap and
# JK generator function.  Unlike the example in examples/scf/40-hf_with_given_hamiltonian.py
# k-point SCF object does not support to compute J/K matrix through the ._eri
# attribute.  get_veff function is defined here to mimic the contractions of
# two-particle integrals with the density matrices.
#
# Ignore finite-size exchange correction for this model system.
#
kmf = pbc.scf.KUHF(cell, kpts, exxdiv=None)

def get_veff(cell, dm, *args):
    weight = 1./Nk
    j_a = np.diag(weight * np.einsum('kii->i', dm[0]) * U)
    k_a = np.diag(weight * np.einsum('kii->i', dm[0]) * U)
    j_b = np.diag(weight * np.einsum('kii->i', dm[1]) * U)
    k_b = np.diag(weight * np.einsum('kii->i', dm[1]) * U)
    j = j_a + j_b
    veff_a = np.array([j-k_a]*Nk)
    veff_b = np.array([j-k_b]*Nk)
    return (veff_a,veff_b)

# Hcore: a (Nk,Nsite,Nsite) array
H_tb_array=get_H_tb_array(kpts,Nx,Ny,t)
kmf.get_hcore = lambda *args: H_tb_array
kmf.get_ovlp = lambda *args: np.array([np.eye(Nsite)]*Nk)
kmf.get_veff = get_veff

kmf = pbc.scf.addons.smearing_(kmf, sigma=0.2, method='gaussian')

#
# Prepare an initial guess.  If not given, dm=0 will be used as initial guess
#
dm_a = np.array([np.eye(Nsite)]*Nk)
dm_b = dm_a * 0
dm = [dm_a, dm_b]
kmf.kernel(dm)

