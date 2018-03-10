#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Access AO integrals

Mole.intor and Mole.intor_by_shell functions can generate AO integrals.
Calling Mole.intor with the integral function name returns a integral matrix
for all basis functions defined in Mole.  If the integral operator has many
compenents eg gradients,  keyword argument comp=* needs to be specified to
tell the function how many components the integrals have.
Mole.intor_by_shell function generates the integrals for the given shell
indices.  Keyword argument comp=* is also required when the integral operator
has multiple components.

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

# Overlap, kinetic, nuclear attraction
s = mol.intor('cint1e_ovlp_sph')
t = mol.intor('cint1e_kin_sph')
v = mol.intor('cint1e_nuc_sph')
# Overlap, kinetic, nuclear attraction gradients (against electron coordinates)
s1 = mol.intor('cint1e_ipovlp_sph', comp=3)
t1 = mol.intor('cint1e_ipkin_sph' , comp=3)
v1 = mol.intor('cint1e_ipnuc_sph' , comp=3)

print('Dipole %s' % numpy.einsum('xij,ij->x',
                                 mol.intor('cint1e_r_sph', comp=3), dm))

#
# AO overlap between two molecules
#
mol1 = gto.M(
    verbose = 0,
    atom = 'H 0 1 0; H 1 0 0',
    basis = 'ccpvdz'
)
s = gto.intor_cross('cint1e_ovlp_sph', mol, mol1)
print('overlap shape (%d, %d)' % s.shape)

#
# 2e integrals.  Keyword aosym is to specify the permutation symmetry in the
# AO integral matrix.  s8 means 8-fold symmetry, s2kl means 2-fold symmetry
# for the symmetry between kl in (ij|kl)
#
eri = mol.intor('cint2e_sph', aosym='s8')
#
# 2e gradient integrals on first atom only
#
eri = mol.intor('cint2e_ip1_sph', aosym='s2kl')

#
# 2e integral gradients on certain atom
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
                    buf = mol.intor_by_shell('cint2e_ip1_sph', shls, comp=3)
                    di, dj, dk, dl = buf.shape[1:]
                    eri1[:,pi:pi+di,pj:pj+dj,pk:pk+dk,pl:pl+dl] = buf
                    pl += dl
                pk += dk
            pj += dj
        pi += di
print('integral shape %s' % str(eri1.shape))

#
# Generate a sub-block of AO integrals.  The sub-block (ij|kl) contains the
# shells 2:5 for basis i, 0:2 for j, 0:4 for k and 1:3 for l
#
sub_eri = mol.intor('int2e_sph', shls_slice=(2,5,0,2,0,4,1,3))
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
# Generate all AO integrals for a sub-system.
#
mol = gto.M(atom=[['H', 0,0,i] for i in range(10)])
atom_idx = [0,2,4]  # The disjoint atoms
sub_mol = mol.copy()
sub_mol._bas = mol._bas[atom_idx]
sub_eri = sub_mol.intor('int2e_sph', aosym='s1')

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
