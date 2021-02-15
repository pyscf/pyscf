#!/usr/bin/env python

import os
import time
import pyscf
from pyscf.tools import c60struct

log = pyscf.lib.logger.Logger(verbose=5)
with open('/proc/cpuinfo') as f:
    for line in f:
        if 'model name' in line:
            log.note(line[:-1])
            break
with open('/proc/meminfo') as f:
    log.note(f.readline()[:-1])
log.note('OMP_NUM_THREADS=%s\n', os.environ.get('OMP_NUM_THREADS', None))


for bas in ('6-31g**', 'cc-pVTZ'):
    mol = pyscf.M(atom=[('C', r) for r in c60struct.make60(1.46,1.38)],
                  basis=bas,
                  max_memory=40000)

    cpu0 = time.clock(), time.time()
    mf = pyscf.scf.fast_newton(mol.RHF())
    cpu0 = logger.timer('SOSCF/%s' % bas, *cpu0)

    mf = mol.RHF().density_fit().run()
    cpu0 = logger.timer('density-fitting-HF/%s' % bas, *cpu0)
