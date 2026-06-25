#!/usr/bin/env python
import pyscf
import sys
from pathlib import Path
from pyscf.tools.mo_mapping import mo_comps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()

verify_windows = '--pyscf-verify-windows' in sys.argv
# Keep the verification profile aligned with the Windows example-runner budget.
basis_sets = ('3-21g',) if verify_windows else ('3-21g', '6-31g*', 'cc-pVTZ', 'ANO-Roos-TZ')

for bas in basis_sets:
    mol = pyscf.M(atom = 'N 0 0 0; N 0 0 1.1',
                  basis = bas)
    cpu0 = get_cpu_timings()

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
