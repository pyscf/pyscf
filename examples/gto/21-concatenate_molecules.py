#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf

'''
Concatenate two molecule enviroments

We need the integrals from different bra and ket space, eg to prepare
initial guess from different geometry or basis, or compute the
transition properties between two states.  To access These integrals, we
need concatenate the enviroments of two Mole object.

This method can be used to generate the 3-center integrals for RI integrals.
'''

mol1 = gto.M(
    verbose = 0,
    atom = 'C 0 0 0; O 0 0 1.5',
    basis = 'ccpvdz'
)
mol2 = gto.M(
    verbose = 0,
    atom = 'H 0 1 0; H 1 0 0',
    basis = 'ccpvdz'
)

atm3, bas3, env3 = gto.conc_env(mol1._atm, mol1._bas, mol1._env,
                                mol2._atm, mol2._bas, mol2._env)
nao1 = mol1.nao_nr()
nao2 = mol2.nao_nr()
s12 = numpy.empty((nao1,nao2))
pi = 0
for i in range(mol1.nbas):
    pj = 0
    for j in range(mol1.nbas, mol1.nbas+mol2.nbas):
        shls = (i, j)
        buf = gto.moleintor.getints_by_shell('cint1e_ovlp_sph',
                                             shls, atm3, bas3, env3)
        di, dj = buf.shape
        s12[pi:pi+di,pj:pj+dj] = buf
        pj += dj
    pi += di
print('<mol1|mol2> overlap shape %s' % str(s12.shape))

#
# 3-center and 2-center 2e integrals for density fitting
#
mol = mol1
auxmol = gto.M(
    verbose = 0,
    atom = 'C 0 0 0; O 0 0 1.5',
    basis = 'weigend'
)
nao = mol.nao_nr()
naoaux = auxmol.nao_nr()

atm, bas, env = \
        gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                          auxmol._atm, auxmol._bas, auxmol._env)
eri3c = numpy.empty((nao,nao,naoaux))
pi = 0
for i in range(mol.nbas):
    pj = 0
    for j in range(mol.nbas):
        pk = 0
        for k in range(mol.nbas, mol.nbas+auxmol.nbas):
            shls = (i, j, k)
            buf = gto.moleintor.getints_by_shell('cint3c2e_sph',
                                                 shls, atm, bas, env)
            di, dj, dk = buf.shape
            eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
            pk += dk
        pj += dj
    pi += di

eri2c = numpy.empty((naoaux,naoaux))
pi = 0
for i in range(mol.nbas, mol.nbas+auxmol.nbas):
    pj = 0
    for j in range(mol.nbas, mol.nbas+auxmol.nbas):
        shls = (i, j)
        buf = gto.moleintor.getints_by_shell('cint2c2e_sph',
                                             shls, atm, bas, env)
        di, dj = buf.shape
        eri2c[pi:pi+di,pj:pj+dj] = buf
        pj += dj
    pi += di

#
# Density fitting Hartree-Fock
#
def get_vhf(mol, dm, *args, **kwargs):
    naux = eri2c.shape[0]
    rho = numpy.einsum('ijp,ij->p', eri3c, dm)
    rho = numpy.linalg.solve(eri2c, rho)
    jmat = numpy.einsum('p,ijp->ij', rho, eri3c)
    kpj = numpy.einsum('ijp,jk->ikp', eri3c, dm)
    pik = numpy.linalg.solve(eri2c, kpj.reshape(-1,naux).T).reshape(-1,nao,nao)
    kmat = numpy.einsum('pik,kjp->ij', pik, eri3c)
    return jmat - kmat * .5

mf = scf.RHF(mol)
mf.verbose = 0
mf.get_veff = get_vhf
print('E(DF-HF) = %.12f, ref = %.12f' % (mf.kernel(), scf.density_fit(mf).kernel()))

