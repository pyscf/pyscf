#!/usr/bin/env python
import os
import time
import re
import numpy
from pyscf import lib
from pyscf import gto, scf, dft, mcscf, mp, cc, lo

def sort_mo(casscf, idx, mo_coeff):
    mol = casscf.mol
    corth = lo.orth.orth_ao(mol)
    casorb = corth[:,idx]

    nmo = mo_coeff.shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    assert(ncas == casorb.shape[1])

    mo1 = reduce(numpy.dot, (casorb.T, casscf._scf.get_ovlp(), mo_coeff))
    sdiag = numpy.einsum('pi,pi->i', mo1, mo1)

    nocc = ncore + nelecas[0]
    casidx = numpy.hstack((numpy.argsort(sdiag[:nocc])[ncore:],
                           nocc+numpy.argsort(-sdiag[nocc:])[:ncas-nelecas[0]]))
    notcas = [i for i in range(nmo) if i not in casidx]
    mo = numpy.hstack((mo_coeff[:,notcas[:ncore]],
                       mo_coeff[:,casidx],
                       mo_coeff[:,notcas[ncore:]]))
    return mo

mol = gto.Mole()
mol.verbose = 0
log = lib.logger.Logger(mol.stdout, 5)
with open('/proc/cpuinfo') as f:
    for line in f:
        if 'model name' in line:
            log.note(line[:-1])
            break
with open('/proc/meminfo') as f:
    log.note(f.readline()[:-1])
log.note('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))

for bas in ('3-21g', '6-31g*', 'cc-pVTZ', 'ANO-Roos-TZ'):
    mol.atom = 'N 0 0 0; N 0 0 1.1'
    mol.basis = bas
    mol.build(0, 0)
    cpu0 = time.clock(), time.time()

    mf = scf.RHF(mol)
    mf.kernel()
    cpu0 = log.timer('N2 %s RHF'%bas, *cpu0)

    mymp2 = mp.MP2(mf)
    mymp2.kernel()
    cpu0 = log.timer('N2 %s MP2'%bas, *cpu0)

    mymc = mcscf.CASSCF(mf, 4, 4)
    idx = mol.search_ao_label('2p[xy]')
    mo = sort_mo(mymc, idx, mf.mo_coeff)
    mymc.kernel(mo)
    cpu0 = log.timer('N2 %s CASSCF'%bas, *cpu0)

    mycc = cc.CCSD(mf)
    mycc.kernel()
    cpu0 = log.timer('N2 %s CCSD'%bas, *cpu0)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    cpu0 = log.timer('N2 %s B3LYP'%bas, *cpu0)

    mf = scf.density_fit(mf)
    mf.kernel()
    cpu0 = log.timer('N2 %s density-fit RHF'%bas, *cpu0)
