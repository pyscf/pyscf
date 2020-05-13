#!/usr/bin/env python

'''
Convert the k-sampled MO/integrals to corresponding Gamma-point supercell
MO/integrals.

Zhihao Cui <zcui@caltech.edu>

See also the original implementation at
https://github.com/zhcui/local-orbital-and-cdft/blob/master/k2gamma.py
'''

import numpy as np
from pyscf.pbc import gto, dft
from pyscf.pbc import tools
from pyscf.pbc.tools import k2gamma
cell = gto.Cell()
cell.atom = '''
H 0.  0.  0.
H 0.5 0.3 0.4
'''

cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4.
cell.unit='B'
cell.build()

kmesh = [2, 2, 1]
kpts = cell.make_kpts(kmesh)

print("Transform k-point integrals to supercell integral")
scell, phase = k2gamma.get_phase(cell, kpts)
NR, Nk = phase.shape
nao = cell.nao
s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
s = scell.pbc_intor('int1e_ovlp')
s1 = np.einsum('Rk,kij,Sk->RiSj', phase, s_k, phase.conj())
print(abs(s-s1.reshape(s.shape)).max())

s = scell.pbc_intor('int1e_ovlp').reshape(NR,nao,NR,nao)
s1 = np.einsum('Rk,RiSj,Sk->kij', phase.conj(), s, phase)
print(abs(s1-s_k).max())

kmf = dft.KRKS(cell, kpts)
ekpt = kmf.run()

mf = k2gamma.k2gamma(kmf, kmesh)
c_g_ao = mf.mo_coeff

# The following is to check whether the MO is correctly coverted:

print("Supercell gamma MO in AO basis from conversion:")
scell = tools.super_cell(cell, kmesh)
mf_sc = dft.RKS(scell)

s = mf_sc.get_ovlp()
mf_sc.run()
sc_mo = mf_sc.mo_coeff

nocc = scell.nelectron // 2
print("Supercell gamma MO from direct calculation:")
print(np.linalg.det(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc])))
print(np.linalg.svd(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc]))[1])

