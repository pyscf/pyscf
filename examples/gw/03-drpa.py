'''
Direct RPA correlation energy
'''

import numpy as np
from scipy import linalg
from pyscf import gto, dft, gw
from pyscf.gw import urpa

mol = gto.M(
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
""",
    basis = 'ccpvqz', verbose = 5,
    )

mf = dft.RKS(mol).density_fit()
mf.xc = 'pbe'
mf.kernel()

import pyscf.gw.rpa
rpa = gw.rpa.dRPA(mf)
rpa.max_memory = 50
rpa.kernel()

# On a small system, compare the frequency-integrated correlation energy
# against the plasmon-sum expression from the RPA excitation matrix.
small_mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto3g')
small_mf = dft.RKS(small_mol, xc='pbe').run()
small_rpa = gw.rpa.RPA(small_mf)
eris = small_rpa.ao2mo()
e_corr = small_rpa.kernel(eris=eris)
e_ov = -small_rpa.make_e_ov()
nov = e_ov.size
v_ov = np.asarray(eris.get_ov_blk(0, nov).T, order='C')
interaction = 2 * v_ov.T.dot(v_ov)
a = np.diag(e_ov) + interaction
b = interaction
rpa_matrix = np.block([[a, b], [-b.conj(), -a.conj()]])
positive = linalg.eigvals(rpa_matrix).real
positive = positive[positive > 0]
plasmon_sum = .5 * (positive.sum() - np.trace(a))
print('RPA frequency integration / plasmon sum:', e_corr, plasmon_sum)

# URPA also accepts both closed-shell and open-shell unrestricted references.
closed_mf = dft.UKS(small_mol, xc='pbe').run()
closed_rpa = urpa.URPA(closed_mf).run()
print('Closed-shell restricted / unrestricted RPA:',
      e_corr, closed_rpa.e_corr)

open_mol = gto.M(atom='F 0 0 0', basis='sto3g', spin=1)
open_mf = dft.UKS(open_mol, xc='pbe0').run()
open_rpa = urpa.URPA(open_mf).run()
print('Open-shell unrestricted RPA:', open_rpa.e_corr)
