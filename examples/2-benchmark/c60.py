#!/usr/bin/env python

import pyscf
from pyscf.tools import c60struct
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()

for bas in ('6-31g**', 'cc-pVTZ'):
    mol = pyscf.M(atom=[('C', r) for r in c60struct.make60(1.46,1.38)],
                  basis=bas,
                  max_memory=40000)

    cpu0 = get_cpu_timings()
    mf = pyscf.scf.fast_newton(mol.RHF())
    cpu0 = log.timer('SOSCF/%s' % bas, *cpu0)

    mf = mol.RHF().density_fit().run()
    cpu0 = log.timer('density-fitting-HF/%s' % bas, *cpu0)
