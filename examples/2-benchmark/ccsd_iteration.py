#!/usr/bin/env python

import pyscf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()


def main():
    verify_windows = '--pyscf-verify-windows' in sys.argv
    # Reduce both chain length and basis during Windows verification only.
    cases = ((6, 'sto-3g'),) if verify_windows else ((30, 'ccpvqz'), (50, 'ccpvqz'))

    for n, basis in cases:
        mol = pyscf.gto.M(atom=['H 0 0 %f' % i for i in range(n)],
                          basis=basis)
        mf = mol.RHF().run()
        mycc = mf.CCSD()
        eris = mycc.ao2mo()
        t1, t2 = mycc.get_init_guess()
        cpu0 = get_cpu_timings()
        mycc.update_amps(t1, t2, eris)
        log.timer('H%d %s CCSD iteration' % (n, basis), *cpu0)


if __name__ == '__main__':
    main()
