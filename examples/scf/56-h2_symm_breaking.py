#!/usr/bin/env python
# Author: James D Whitfield <jdwhitfield@gmail.com>
'''
Scan H2 molecule dissociation curve comparing UHF and RHF solutions per the 
example of Szabo and Ostlund section 3.8.7

The initial guess is obtained by mixing the HOMO and LUMO and is implemented
as a function that can be used in other applications.

See also 16-h2_scan.py, 30-scan_pes.py, 32-break_spin_symm.py
'''

import numpy
import pyscf
from pyscf import scf
from pyscf import gto

erhf = []
euhf = []
dm = None


def init_guess_mixed(mol,mixing_parameter=numpy.pi/4):
    ''' Generate density matrix with broken spatial and spin symmetry by mixing
    HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.
    
    psi_1a = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
    psi_1b = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
        
    psi_2a = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
    psi_2b =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo

    Returns: 
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    # opt: q, mixing parameter 0 < q < 2 pi
    
    #based on init_guess_by_1e
    h1e = scf.hf.get_hcore(mol)
    s1e = scf.hf.get_ovlp(mol)
    mo_energy, mo_coeff = rhf.eig(h1e, s1e)
    mo_occ = rhf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx=0
    lumo_idx=1

    for i in range(len(mo_occ)-1):
        if mo_occ[i]>0 and mo_occ[i+1]<0:
            homo_idx=i
            lumo_idx=i+1

    psi_homo=mo_coeff[:, homo_idx]
    psi_lumo=mo_coeff[:, lumo_idx]
    
    Ca=numpy.zeros_like(mo_coeff)
    Cb=numpy.zeros_like(mo_coeff)


    #mix homo and lumo of alpha and beta coefficients
    q=mixing_parameter

    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:,k] = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
            Cb[:,k] = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
            continue
        if k==lumo_idx:
            Ca[:,k] = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            Cb[:,k] =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            continue
        Ca[:,k]=mo_coeff[:,k]
        Cb[:,k]=mo_coeff[:,k]

    dm =scf.UHF(mol).make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
    return dm 


for b in numpy.arange(0.7, 4.01, 0.1):
    mol = gto.M(atom=[["H", 0., 0., 0.],
                      ["H", 0., 0., b ]], basis='sto-3g', verbose=0)
    rhf = scf.RHF(mol)
    uhf = scf.UHF(mol)
    erhf.append(rhf.kernel(dm))
    euhf.append(uhf.kernel(init_guess_mixed(mol)))
    dm = rhf.make_rdm1()

print('R     E(RHF)      E(UHF)')
for i, b in enumerate(numpy.arange(0.7, 4.01, 0.1)):
    print('%.2f  %.8f  %.8f' % (b, erhf[i], euhf[i]))
