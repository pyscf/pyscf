#!/usr/bin/env python

"""CMS-PDFT nonadiabatic couplings and state gradients for LiH.

Compare couplings under state interchange, with and without electron-translation
factors (ETFs). See 01-sacasscf_nac.py for NAC sign conventions and the
corresponding SA-CASSCF calculations. All comparisons here use one common
wavefunction, so its arbitrary state phases are held fixed.
"""

import numpy as np
from pyscf import gto, scf, mcpdft

mol = gto.M(atom='Li 0 0 0; H 0 0 1.5', basis='sto-3g', verbose=4)
mf = scf.RHF(mol).run()
# Use standard PySCF grids instead of the optional mrh/OpenMolcas grid preset
# in the original demonstration. Matching external energies requires matching grids.
mc = mcpdft.CASSCF(mf, 'ftLDA,VWN3', 2, 2, grids_level=3)
mc.fix_spin_(ss=0)
mc = mc.multi_state([0.5, 0.5], 'cms').run(conv_tol=1e-10)
print('CMS-PDFT state energies:', mc.e_states)
nacs = mc.nac_method()

# ETFs omit the CSF contribution. Ordinary NACs are antisymmetric under
# state interchange: d_01 = -d_10.
nac_01 = nacs.kernel(state=(0, 1), use_etfs=True)
nac_10 = nacs.kernel(state=(1, 0), use_etfs=True)
print('NAC (0, 1) with ETFs:\n', nac_01)
print('NAC (1, 0) with ETFs:\n', nac_10)
print('Antisymmetry error with ETFs:', np.linalg.norm(nac_01 + nac_10))

# Multiplying by the energy difference removes the small-gap denominator.
# Both the gap and the NAC change sign, making the product symmetric.
scaled_01 = nacs.kernel(state=(0, 1), use_etfs=True, mult_ediff=True)
scaled_10 = nacs.kernel(state=(1, 0), use_etfs=True, mult_ediff=True)
print('Energy-scaled NAC (0, 1) with ETFs:\n', scaled_01)
print('Energy-scaled NAC (1, 0) with ETFs:\n', scaled_10)
print('Symmetry error with ETFs:', np.linalg.norm(scaled_01 - scaled_10))

# Repeat with the CSF contribution included.
nac_01 = nacs.kernel(state=(0, 1), use_etfs=False)
nac_10 = nacs.kernel(state=(1, 0), use_etfs=False)
print('NAC (0, 1) including CSF contribution:\n', nac_01)
print('NAC (1, 0) including CSF contribution:\n', nac_10)
print('Antisymmetry error including CSF contribution:', np.linalg.norm(nac_01 + nac_10))

scaled_01 = nacs.kernel(state=(0, 1), use_etfs=False, mult_ediff=True)
scaled_10 = nacs.kernel(state=(1, 0), use_etfs=False, mult_ediff=True)
print('Energy-scaled NAC (0, 1) including CSF contribution:\n', scaled_01)
print('Energy-scaled NAC (1, 0) including CSF contribution:\n', scaled_10)
print('Symmetry error including CSF contribution:', np.linalg.norm(scaled_01 - scaled_10))

# State gradients accompany couplings in analyses near an intersection.
grad = mc.nuc_grad_method()
print('CMS-PDFT ground-state gradient:\n', grad.kernel(state=0))
print('CMS-PDFT excited-state gradient:\n', grad.kernel(state=1))
