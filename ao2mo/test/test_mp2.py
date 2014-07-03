#!/usr/bin/env python

import os
import ctypes
import tempfile
import numpy
import time

from pyscf import lib
from pyscf import ao2mo


# the MO integral for MP2 is (ov|ov). The most efficient integral
# transformation is
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)
def rmp2_energy_incore(mf, verbose=None):
    mol = mf.mol
    if verbose is None:
        verbose = mf.verbose
    log = lib.logger.Logger(mf.fout, verbose)

    tcpu0 = time.clock()
    log.debug('transform (ia|jb)')
    nocc = mol.nelectron / 2
    co = mf.mo_coeff[:,:nocc]
    cv = mf.mo_coeff[:,nocc:]
    g = ao2mo.incore.general(mf._eri, (co,cv,co,cv))
    tcpu0, dt = time.clock(), time.clock()-tcpu0
    log.debug('integral transformation CPU time: %8.2f', dt)

    eia = mf.mo_energy[:nocc].reshape(nocc,1) - mf.mo_energy[nocc:]
    nvir = eia.shape[1]
    emp2 = 0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                ia = i * nvir + a
                ja = j * nvir + a
                for b in range(nvir):
                    ib = i * nvir + b
                    jb = j * nvir + b
                    emp2 += g[ia,jb] * (g[ia,jb]*2-g[ib,ja]) \
                            / (eia[i,a]+eia[j,b])

    log.log('RMP2 energy = %.15g', emp2)
    tcpu0, dt = time.clock(), time.clock()-tcpu0
    log.debug('MP2 CPU time: %8.2f', dt)

    return emp2


def rmp2_energy(mol, mo_coeff, mo_energy, nocc, verbose=None):
    if verbose is None:
        verbose = mol.verbose
    log = lib.logger.Logger(mol.fout, verbose)

    tcpu0 = time.clock()
    log.debug('transform (ia|jb)')
    co = mo_coeff[:,:nocc]
    cv = mo_coeff[:,nocc:]
    g = ao2mo.direct.general_iofree(mol, (co,cv,co,cv), verbose=verbose)
    tcpu0, dt = time.clock(), time.clock()-tcpu0
    log.debug('integral transformation CPU time: %8.2f', dt)

    eia = mo_energy[:nocc].reshape(nocc,1) - mo_energy[nocc:]
    nvir = eia.shape[1]
    emp2 = 0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                ia = i * nvir + a
                ja = j * nvir + a
                for b in range(nvir):
                    ib = i * nvir + b
                    jb = j * nvir + b
                    emp2 += g[ia,jb] * (g[ia,jb]*2-g[ib,ja]) \
                            / (eia[i,a]+eia[j,b])

    log.log('RMP2 energy = %.15g', emp2)
    tcpu0, dt = time.clock(), time.clock()-tcpu0
    log.debug('MP2 CPU time: %8.2f', dt)

    return emp2


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    mol.max_memory = 100
    mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    rhf = scf.RHF(mol)
    print rhf.scf()

    print rmp2_energy(mol, rhf.mo_coeff, rhf.mo_energy, mol.nelectron/2)
    print rmp2_energy_incore(rhf)

    from pyscf import lib._vhf as _vhf
    mo_coeff, mo_energy, nocc = rhf.mo_coeff, rhf.mo_energy, mol.nelectron/2
    n = mo_energy.size
    g = _vhf.restore_full_eri(rhf._eri, n)
    g = numpy.dot(g.reshape(n*n*n,n),mo_coeff)
    g = numpy.dot(mo_coeff.T,g.reshape(n,n*n*n))
    g = numpy.transpose(g.reshape(n,n,n,n), (2,3,0,1))
    g = numpy.dot(g.reshape(n*n*n,n),mo_coeff)
    g = numpy.dot(mo_coeff.T,g.reshape(n,n*n*n))
    g = g.reshape(n,n,n,n)
    eia = mo_energy[:nocc].reshape(nocc,1) - mo_energy[nocc:]
    nvir = eia.shape[1]
    emp2 = 0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    emp2 += g[i,nocc+a,j,nocc+b] \
                            * (g[i,nocc+a,j,nocc+b]*2-g[j,nocc+a,i,nocc+b]) \
                            / (eia[i,a]+eia[j,b])
    print emp2

