#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Access AO integrals

Mole.intor and Mole.intor_by_shell functions can generate AO integrals.
Calling Mole.intor with the integral function name returns a integral matrix
for all basis functions defined in Mole.  If the integral operator has many
compenents eg gradients.

Mole.intor_by_shell function generates the integrals for the given shell
indices.

See pyscf/gto/moleintor.py file for the complete list of supported integrals.
'''

import numpy
from pyscf import gto, scf

mol = gto.M(
    verbose = 0,
    atom = 'C 0 0 0; O 0 0 1.5',
    basis = 'ccpvdz'
)
mf = scf.RHF(mol)
mf.kernel()
dm = mf.make_rdm1()

#
# Overlap, kinetic, nuclear attraction
#
s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')
# Overlap, kinetic, nuclear attraction gradients (against electron coordinates
# on bra)
s1 = mol.intor('int1e_ipovlp')  # (3,N,N) array, 3 for x,y,z components
t1 = mol.intor('int1e_ipkin' )  # (3,N,N) array, 3 for x,y,z components
v1 = mol.intor('int1e_ipnuc' )  # (3,N,N) array, 3 for x,y,z components

mol.set_common_origin([0,0,0])  # Set gauge origin before computing dipole integrals
print('Dipole %s' % numpy.einsum('xij,ji->x', mol.intor('int1e_r'), dm))

#
# AO overlap between two molecules
#
mol1 = gto.M(
    verbose = 0,
    atom = 'H 0 1 0; H 1 0 0',
    basis = 'ccpvdz'
)
s = gto.intor_cross('int1e_ovlp', mol, mol1)
print('overlap shape (%d, %d)' % s.shape)


#
# 2e integrals.  Keyword aosym is to specify the permutation symmetry in the
# AO integral matrix.  s8 means 8-fold symmetry, s2kl means 2-fold symmetry
# for the symmetry between kl in (ij|kl)
#
eri = mol.intor('int2e', aosym='s8')


#
# 2e gradient integrals (against electronic coordinates) on bra of first atom.
# aosym=s2kl indicates that the permutation symmetry is used on k,l indices of
# (ij|kl). The resultant eri is a 3-dimension array (3, N*N, N*(N+1)/2) where
# N is the number of AO orbitals.
#
eri = mol.intor('int2e_ip1', aosym='s2kl')


#
# Settting aosym=s1 (the default flag) leads to a 3-dimension (3, N*N, N*N)
# eri array.
#
nao = mol.nao_nr()
eri = mol.intor('int2e_ip1').reshape(3,nao,nao,nao,nao)


#
# 2e integral gradients on a specific atom
#
atm_id = 1  # second atom
bas_start, bas_end, ao_start, ao_end = mol.aoslice_by_atom()[atm_id]
tot_bra = ao_end - ao_start
nao = mol.nao_nr()
eri1 = numpy.empty((3,tot_bra,nao,nao,nao))
pi = 0
for i in range(mol.nbas):
    if mol.bas_atom(i) == atm_id:
        pj = 0
        for j in range(mol.nbas):
            pk = 0
            for k in range(mol.nbas):
                pl = 0
                for l in range(mol.nbas):
                    shls = (i, j, k, l)
                    buf = mol.intor_by_shell('int2e_ip1_sph', shls)
                    comp_3, di, dj, dk, dl = buf.shape
                    eri1[:,pi:pi+di,pj:pj+dj,pk:pk+dk,pl:pl+dl] = buf
                    pl += dl
                pk += dk
            pj += dj
        pi += di
print('integral shape %s' % str(eri1.shape))
# This integral block can be generated using mol.intor
eri1 = mol.intor('int2e_ip1_sph', shls_slice=(bas_start, bas_end,
                                              0, mol.nbas,
                                              0, mol.nbas,
                                              0, mol.nbas))


#
# Generate a sub-block of AO integrals.  The sub-block (ij|kl) contains the
# shells 2:5 for basis i, 0:2 for j, 0:4 for k and 1:3 for l
#
sub_eri = mol.intor('int2e', shls_slice=(2,5,0,2,0,4,1,3))

# This statement is equivalent to
dims = []
for i in range(mol.nbas):
    l = mol.bas_angular(i)
    nc = mol.bas_nctr(i)
    dims.append((l * 2 + 1) * nc)
nao_i = sum(dims[2:5])
nao_j = sum(dims[0:2])
nao_k = sum(dims[0:4])
nao_l = sum(dims[1:3])
sub_eri = numpy.empty((nao_i,nao_j,nao_k,nao_l))
pi = 0
for i in range(2,5):
    pj = 0
    for j in range(0,2):
        pk = 0
        for k in range(0,4):
            pl = 0
            for l in range(1,3):
                shls = (i, j, k, l)
                buf = mol.intor_by_shell('int2e_sph', shls)
                di, dj, dk, dl = buf.shape
                sub_eri[pi:pi+di,pj:pj+dj,pk:pk+dk,pl:pl+dl] = buf
                pl += dl
            pk += dk
        pj += dj
    pi += di
sub_eri = sub_eri.reshape(nao_i*nao_j,nao_k*nao_l)

#
# 2-electron integrals over different molecules. E.g. a,c of (ab|cd) on one molecule
# and b on another molecule and d on the third molecule.
#
mol1 = mol
mol2 = gto.M(atom='He', basis='ccpvdz')
mol3 = gto.M(atom='O', basis='sto-3g')

mol123 = mol1 + mol2 + mol3
eri = mol123.intor('int2e', shls_slice=(0, mol1.nbas,
                                        mol1.nbas, mol1.nbas+mol2.nbas,
                                        0, mol1.nbas,
                                        mol1.nbas+mol2.nbas, mol123.nbas))


#
# Generate all AO integrals for a sub-system.
#
mol = gto.M(atom=[['H', 0,0,i] for i in range(10)])
atom_idx = [0,2,4]  # atoms in the sub-system
sub_mol = mol.copy()
sub_mol._bas = mol._bas[atom_idx]
sub_eri = sub_mol.intor('int2e', aosym='s1')

# This statement is equivalent to
sub_nao = 0
for i in range(mol.nbas):
    if mol.bas_atom(i) in atom_idx:
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        sub_nao += (l * 2 + 1) * nc
sub_eri = numpy.empty((sub_nao,sub_nao,sub_nao,sub_nao))
pi = 0
for i in range(mol.nbas):
    if mol.bas_atom(i) in atom_idx:
        pj = 0
        for j in range(mol.nbas):
            if mol.bas_atom(j) in atom_idx:
                pk = 0
                for k in range(mol.nbas):
                    if mol.bas_atom(k) in atom_idx:
                        pl = 0
                        for l in range(mol.nbas):
                            if mol.bas_atom(l) in atom_idx:
                                shls = (i, j, k, l)
                                buf = mol.intor_by_shell('int2e_sph', shls)
                                di, dj, dk, dl = buf.shape
                                sub_eri[pi:pi+di,pj:pj+dj,pk:pk+dk,pl:pl+dl] = buf
                                pl += dl
                        pk += dk
                pj += dj
        pi += di
sub_eri = sub_eri.reshape(sub_nao**2,sub_nao**2)
