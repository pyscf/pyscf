import sys
import numpy as np
from pathlib import Path
from pyscf import fci

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()


def main():
    verify_windows = '--pyscf-verify-windows' in sys.argv
    # Keep the Windows verification profile below the example-runner timeout.
    orbital_counts = (10, 12) if verify_windows else (12, 14, 16, 18)

    for norb in orbital_counts:
        nelec = (norb // 2, norb // 2)
        npair = norb * (norb + 1) // 2

        h1 = np.random.random((norb, norb))
        h1 = h1 + h1.T
        h2 = np.random.random(npair * (npair + 1) // 2)

        na = fci.cistring.num_strings(norb, nelec[0])
        ci0 = np.random.random((na, na))
        ci0 *= 1 / np.linalg.norm(ci0)

        link = fci.cistring.gen_linkstr_index(range(norb), nelec[0], tril=True)
        cpu0 = get_cpu_timings()
        fci.direct_spin0.contract_2e(h2, ci0, norb, nelec, link)
        cpu0 = log.timer('FCI-spin0-solver (%do, %de)' % (norb, sum(nelec)), *cpu0)
        fci.direct_spin1.contract_2e(h2, ci0, norb, nelec, (link, link))
        log.timer('FCI-spin1-solver (%do, %de)' % (norb, sum(nelec)), *cpu0)


if __name__ == '__main__':
    main()
