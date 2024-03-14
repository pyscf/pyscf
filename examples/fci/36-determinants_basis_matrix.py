#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
One-particle and two-particle operators represented in Slater determinant basis.
'''

import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring
from pyscf.fci import fci_slow

np.random.seed(1)
norb = 7
nelec = (4,4)
h1 = np.random.random((norb,norb))
eri = np.random.random((norb,norb,norb,norb))
# Restore permutation symmetry
h1 = h1 + h1.T
eri = eri + eri.transpose(1,0,2,3)
eri = eri + eri.transpose(0,1,3,2)
eri = eri + eri.transpose(2,3,0,1)

link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), nelec[0])
na = nb = cistring.num_strings(norb, nelec[0])
idx_a = idx_b = np.arange(na)

# One-particle operator is straightforward. Using the link_index we can
# propagate determinants in ket to determinants in bra for operator
#    O_{pq} * a_p^+ a_q
mat1 = np.zeros((na,nb,na,nb))
for str0, tab in enumerate(link_indexa):
    for p, q, str1, sign in tab:
        # alpha spin
        mat1[str1,idx_b,str0,idx_b] += sign * h1[p,q]
        # beta spin
        mat1[idx_a,str1,idx_a,str0] += sign * h1[p,q]
mat1 = mat1.reshape(na*nb,na*nb)

# Two-particle operator is a little bit complicated. The code below requires a
# representation of the operator in this form
#    O_{pqrs} * a_p^+ a_q a_r^+ a_s
# However, the regular ERI tensor is
#    eri_{pqrs} * a_p^+ a_r^+ a_s a_q
# We need to use the absorb_h1e function to transform the ERI tensor first
h2 = fci_slow.absorb_h1e(h1*0, eri, norb, nelec)
t1 = np.zeros((norb,norb,na,nb,na,nb))
for str0, tab in enumerate(link_indexa):
    for a, i, str1, sign in tab:
        # alpha spin
        t1[a,i,str1,idx_b,str0,idx_b] += sign
        # beta spin
        t1[a,i,idx_a,str1,idx_a,str0] += sign
t1 = lib.einsum('psqr,qrABab->psABab', h2, t1)
mat2 = np.zeros((na,nb,na,nb))
for str0, tab in enumerate(link_indexa):
    for a, i, str1, sign in tab:
        # alpha spin
        mat2[str1] += sign * t1[a,i,str0]
        # beta spin
        mat2[:,str1] += sign * t1[a,i,:,str0]
mat2 = mat2.reshape(na*nb,na*nb)

H_fci = mat1 + mat2 * .5
H_ref = fci.direct_spin1.pspace(h1, eri, norb, nelec, np=1225)[1]
print('Check', abs(H_fci - H_ref).max())
