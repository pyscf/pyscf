#!/usr/bin/env python
import pyscf
import sys
from pathlib import Path
from pyscf.tools.mo_mapping import mo_comps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()


def main():
    verify_windows = '--pyscf-verify-windows' in sys.argv
    # Keep the Windows verification profile small enough for the example runner.
    basis_sets = ('3-21g',) if verify_windows else ('3-21g', '6-31g**', 'cc-pVTZ', 'ANO-Roos-TZ')

    for bas in basis_sets:
        mol = pyscf.M(atom='''
c   1.217739890298750 -0.703062453466927  0.000000000000000
h   2.172991468538160 -1.254577209307266  0.000000000000000
c   1.217739890298750  0.703062453466927  0.000000000000000
h   2.172991468538160  1.254577209307266  0.000000000000000
c   0.000000000000000  1.406124906933854  0.000000000000000
h   0.000000000000000  2.509154418614532  0.000000000000000
c  -1.217739890298750  0.703062453466927  0.000000000000000
h  -2.172991468538160  1.254577209307266  0.000000000000000
c  -1.217739890298750 -0.703062453466927  0.000000000000000
h  -2.172991468538160 -1.254577209307266  0.000000000000000
c   0.000000000000000 -1.406124906933854  0.000000000000000
h   0.000000000000000 -2.509154418614532  0.000000000000000
''',
                      basis=bas)
        cpu0 = get_cpu_timings()

        mf = mol.RHF().run()
        cpu0 = log.timer('C6H6 %s RHF' % bas, *cpu0)

        mf.MP2().run()
        cpu0 = log.timer('C6H6 %s MP2' % bas, *cpu0)

        mymc = mf.CASSCF(6, 6)
        idx_2pz = mo_comps('2pz', mol, mf.mo_coeff).argsort()[-6:]
        mo = mymc.sort_mo(idx_2pz, base=0)
        mymc.kernel(mo)
        cpu0 = log.timer('C6H6 %s CASSCF' % bas, *cpu0)

        mf.CCSD().run()
        cpu0 = log.timer('C6H6 %s CCSD' % bas, *cpu0)

        mf = mol.RKS().run(xc='b3lyp')
        cpu0 = log.timer('C6H6 %s B3LYP' % bas, *cpu0)

        mf = mf.density_fit().run()
        log.timer('C6H6 %s density-fit RHF' % bas, *cpu0)


if __name__ == '__main__':
    main()

