#!/usr/bin/env python

import pyscf
import sys
from pathlib import Path
from pyscf.tools import c60struct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()


def main():
    verify_windows = '--pyscf-verify-windows' in sys.argv
    # Use a smaller cage and basis only for the Windows verification profile.
    basis_sets = ('sto-3g',) if verify_windows else ('6-31g**', 'cc-pVTZ')
    cage = c60struct.make12(1.4) if verify_windows else c60struct.make60(1.46, 1.38)

    for bas in basis_sets:
        mol = pyscf.M(atom=[('C', r) for r in cage],
                      basis=bas,
                      max_memory=40000)

        cpu0 = get_cpu_timings()
        pyscf.scf.fast_newton(mol.RHF())
        cpu0 = log.timer('SOSCF/%s' % bas, *cpu0)

        mol.RHF().density_fit().run()
        log.timer('density-fitting-HF/%s' % bas, *cpu0)


if __name__ == '__main__':
    main()
