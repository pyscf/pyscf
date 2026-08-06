#!/usr/bin/env python


'''
PM localization for PBC systems using both Gamma-point and k-point approaches.
'''

import sys
import numpy as np
from pyscf.pbc import gto, scf, lo, tools
from pyscf import lo as mol_lo
from pyscf.lib import logger

log = logger.Logger(sys.stdout, 6)


''' Perform a k-point SCF calculation
'''
cell = gto.Cell()
cell.atom = '''
Si    0.0000000000    0.0000000000    0.0000000000
Si    1.3575000000    1.3575000000    1.3575000000
'''
cell.a = '''
2.7150000000 2.7150000000 0.0000000000
0.0000000000 2.7150000000 2.7150000000
2.7150000000 0.0000000000 2.7150000000
'''
cell.basis = '''
#BASIS SET: (4s,4p,1d) -> [2s,2p,1d] Si
Si  S
    1.271038   -2.675576e-01
    0.307669    3.996909e-01
    0.141794    5.784306e-01
Si  S
    0.062460    1.000000e+00
Si  P
    1.610683   -2.629981e-02
    0.384570    3.047784e-01
    0.148473    5.453783e-01
Si  P
    0.055964    1.000000e+00
Si  D
    0.285590    1.000000e+00
''' # gth-cc-pvdz
cell.pseudo = 'gth-pbe'
cell.mesh = [23]*3
cell.verbose = 4
cell.build()

kmesh = [3,3,3]
kpts = cell.make_kpts(kmesh)
nkpts = len(kpts)

mf = scf.KRKS(cell, kpts=kpts).set(xc='pbe').run()


''' k-point-based PM localization (complex rotations)
'''
nocc = cell.nelectron // 2
mo = np.asarray([x[:,:nocc] for x in mf.mo_coeff])
mlo = lo.KPM(cell, mo, kpts)    # k-point PM with complex rotations
mlo.kernel()

# stability check
while True:
    mo, stable = mlo.stability(return_status=True)
    if stable:
        break
    mlo.kernel(mo)

# get wannier function
wann_coeff = mlo.get_wannier_function()


''' Supercell-based PM localization (complex rotations)
'''
mo = np.asarray([x[:,:nocc] for x in mf.mo_coeff])
mo_s = lo.base.wannierization(cell, kpts, mo)   # Initial WF in supercell
cell_s = tools.super_cell(cell, kmesh)
mlo_s = mol_lo.PMComplex(cell_s, mo_s)  # PM with complex rotations
mlo_s.kernel()

# stability check
while True:
    mo_s, stable = mlo_s.stability(return_status=True)
    if stable:
        break
    mlo_s.kernel(mo_s)

# PM objective comparison:
log.info('')
log.info('kpoint PM obj: %.6f', mlo.cost_function())
log.info('scell  PM obj: %.6f', mlo_s.cost_function() / nkpts)
log.info('')


# explicitly verify that Wannier function from k-point PM is indeed a solution for
# supercell PM:
mlo_s.kernel(wann_coeff)

log.info('')
log.info('kpoint PM obj: %.6f', mlo.cost_function())
log.info('scell  PM obj: %.6f', mlo_s.cost_function() / nkpts)
log.info('')
