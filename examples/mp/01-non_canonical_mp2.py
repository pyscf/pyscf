#!/usr/bin/env python

'''
MP2 with a non-canonical (non-Brillouin) reference, and ROMP2 for ROHF.

When the reference orbitals do not diagonalize the mean-field Fock matrix, the
occupied-virtual Fock block f_ov is nonzero and Brillouin's theorem does not
hold.  The first-order wavefunction then contains single excitations (T1),
which contribute to the MP2 correlation energy, and MP2 is solved iteratively.

Note that f_ov != 0 is the condition, not non-HF orbitals per se: a canonical
Kohn-Sham reference is non-HF but has f_ov = 0 (no T1), and localized HF
orbitals are non-canonical within the occupied/virtual blocks yet still have
f_ov = 0.  See issue #1687.

For an ROHF reference, mf.MP2() returns the restricted open-shell MP2 (ROMP2)
implementation, which semi-canonicalizes the ROHF orbitals and includes the T1
singles (Knowles et al., Chem. Phys. Lett. 186, 130 (1991)).  The same energy
is obtained with UMP2 on the ROHF wavefunction converted to UHF.
'''

import pyscf
from pyscf import mp


########################################
# A non-canonical HF reference
#
mol = pyscf.M(atom='''
O    0.   0.       0.
H    0.   -0.757   0.587
H    0.   0.757    0.587''',
basis='cc-pvdz', verbose=4)

# Kohn-Sham orbitals converted to an HF object.  They are not canonical for the
# HF Fock (f_ov != 0), so MP2 is solved iteratively with the T1 singles.
mf = mol.RKS().run()
mf = mf.to_hf()

pt = mf.MP2().run()
print('non-canonical MP2  E_corr = %.9f  E_singles = %.9f'
      % (pt.e_corr, pt.e_corr_singles))


########################################
# ROMP2 for an ROHF reference
#
mol = pyscf.M(
    atom = 'N 0 0 0; H 0 0 1.0; H 0.94 0 -0.33; H -0.94 0 -0.33',
    charge = 1, spin = 1, basis = 'cc-pvdz', verbose=4)

mf = mol.ROHF().run()

# ROMP2: semi-canonical ROHF reference, T1 included
pt = mf.MP2().run()
print('ROMP2              E_corr = %.9f  E_singles = %.9f'
      % (pt.e_corr, pt.e_corr_singles))

# UMP2 on the ROHF -> UHF conversion.  The non-canonical reference is detected
# automatically, so the T1 singles is included (same energy as ROMP2).
pt = mp.UMP2(mf.to_uhf()).run()
print('UMP2(to_uhf)       E_corr = %.9f' % pt.e_corr)

# Previous behavior (before #3471): UMP2 on the ROHF orbitals without the T1 singles.
pt = mp.UMP2(mf.to_uhf())
pt.exclude_t1 = True
pt.run()
print('UMP2 (exclude_t1)  E_corr = %.9f' % pt.e_corr)
