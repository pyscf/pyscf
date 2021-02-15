#!/usr/bin/env python
import os
import time
import pyscf
from pyscf.tools.mo_mapping import mo_comps

log = pyscf.lib.logger.Logger(verbose=5)
with open('/proc/cpuinfo') as f:
    for line in f:
        if 'model name' in line:
            log.note(line[:-1])
            break
with open('/proc/meminfo') as f:
    log.note(f.readline()[:-1])
log.note('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))

for bas in ('3-21g', '6-31g*', 'cc-pVTZ', 'ANO-Roos-TZ'):
    mol = pyscf.M(atom = 'N 0 0 0; N 0 0 1.1',
                  basis = bas)
    cpu0 = time.clock(), time.time()

    mf = mol.RHF().run()
    cpu0 = log.timer('N2 %s RHF'%bas, *cpu0)

    mymp2 = mf.MP2().run()
    cpu0 = log.timer('N2 %s MP2'%bas, *cpu0)

    mymc = mf.CASSCF(4, 4)
    idx_2pz = mo_comps('2p[xy]', mol, mf.mo_coeff).argsort()[-4:]
    mo = mymc.sort_mo(idx_2pz, base=0)
    mymc.kernel(mo)
    cpu0 = log.timer('N2 %s CASSCF'%bas, *cpu0)

    mycc = mf.CCSD().run()
    cpu0 = log.timer('N2 %s CCSD'%bas, *cpu0)

    mf = mol.RKS().run(xc='b3lyp')
    cpu0 = log.timer('N2 %s B3LYP'%bas, *cpu0)

    mf = mf.density_fit().run()
    cpu0 = log.timer('N2 %s density-fit RHF'%bas, *cpu0)
