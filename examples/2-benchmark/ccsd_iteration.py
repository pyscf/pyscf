#!/usr/bin/env python

import os
import time
import pyscf

log = pyscf.lib.logger.Logger(verbose=5)
with open('/proc/cpuinfo') as f:
    for line in f:
        if 'model name' in line:
            log.note(line[:-1])
            break
with open('/proc/meminfo') as f:
    log.note(f.readline()[:-1])
log.note('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))


for n in (30, 50):
    mol = gto.M(atom=['H 0 0 %f' % i for i in range(n)],
                basis='ccpvqz')
    mf = mol.RHF().run()
    mycc = mf.CCSD()
    eris = mycc.ao2mo()
    _, t1, t2 = mycc.get_init_guess()
    cpu0 = time.clock(), time.time()
    mycc.update_amps(t1, t2, eris)
    log.timer('H%d cc-pVQZ CCSD iteration' % n, *cpu0)
