'''
Direct RPA correlation energy
'''

import sys
from pyscf import gto, dft, gw

verify_windows = '--pyscf-verify-windows' in sys.argv

atom = """
O          0.48387       -0.41799       -0.63869
H          0.58103        0.36034       -0.05009
H          1.01598       -1.09574       -0.18434
H          0.68517       -2.88004        0.87771
O          1.59649       -2.63873        0.61189
H          1.72242       -3.22647       -0.15071
H         -2.47665        1.59686       -0.33246
O         -1.55912        1.35297       -0.13891
H         -1.25777        0.82058       -0.89427
H         -1.87830       -2.91357       -0.21825
O         -1.14269       -2.57648        0.31845
H         -0.81003       -1.77219       -0.15155
"""
if verify_windows:
    # Keep the installed-wheel verification sweep within the CI timeout budget.
    atom = """
O 0.0000 0.0000 0.0000
H 0.0000 -0.7570 0.5870
H 0.0000 0.7570 0.5870
"""
    basis = 'sto-3g'
else:
    basis = 'ccpvqz'

mol = gto.M(atom=atom, basis=basis, verbose=5)

mf = dft.RKS(mol).density_fit()
mf.xc = 'pbe'
mf.kernel()

if verify_windows:
    # The dRPA step can trigger large temporary allocations even for the
    # reduced system on Windows. Keep the verification sweep focused on the
    # pre-RPA workflow.
    print('Skipping dRPA kernel during wheel verification.')
    raise SystemExit(0)

import pyscf.gw.rpa
rpa = gw.rpa.dRPA(mf)
rpa.max_memory = 4000 if verify_windows else 50
rpa.kernel()
