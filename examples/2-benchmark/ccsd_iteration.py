#!/usr/bin/env python

import pyscf
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()

for n in (30, 50):
    mol = pyscf.gto.M(atom=['H 0 0 %f' % i for i in range(n)],
                      basis='ccpvqz')
    mf = mol.RHF().run()
    mycc = mf.CCSD()
    eris = mycc.ao2mo()
    _, t1, t2 = mycc.get_init_guess()
    cpu0 = get_cpu_timings()
    mycc.update_amps(t1, t2, eris)
    log.timer('H%d cc-pVQZ CCSD iteration' % n, *cpu0)
