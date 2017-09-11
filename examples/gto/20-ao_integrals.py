#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf

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
# 2e integrals (New in PySCF-1.1).  keyword aosym is required to specify the
# permutation symmetry in the AO integral matrix.  s8 means 8-fold symmetry,
# s2kl means 2-fold symmetry for the symmetry between kl only
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
                    dl += dl
                pk += dk
            pj += dj
        pi += di
print('integral shape %s' % str(eri1.shape))
