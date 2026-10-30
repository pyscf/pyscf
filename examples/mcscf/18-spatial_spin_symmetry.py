#!/usr/bin/env python

'''
Force the FCI solver of CASSCF solving particular spin state.
'''

from pyscf import scf
from pyscf import gto
from pyscf import mcscf
from pyscf.fci import direct_spin1_cyl_sym

mol = gto.M(
    atom = '''
      8  0  0  0
      8  0  0  1.1''',
    basis = 'ccpvdz',
    symmetry = True,
    spin = 2,
)

mf = scf.RHF(mol)
mf.kernel()

# The 3-Sigma_g^- ground state (A2g) is the lowest state of its D2h irrep, so
# the default (D2h) solver finds the correct component. A wfnsym label that is
# already unambiguous in the subgroup needs no special treatment.
mc = mcscf.CASSCF(mf, 8, 12)
mc.fcisolver.wfnsym = 'A2g'
mc.kernel()
print('Triplet Sigma_g^- %.15g  ref = -149.688656224059' % mc.e_tot)

# Specify CI wfn spatial symmetry by assigning fcisolver.wfnsym.
#
# For a linear molecule (Dinfh), an excited Sigma state cannot be labeled by
# the D2h subgroup alone: the sigma_v (+) and (-) components of Sigma share the
# same D2h irrep. For example, Sigma_u- (A2u) and a Delta_u component both
# descend to Au, and the Delta_u lies lower. Requesting wfnsym='A2u' from the
# default (D2h) solver may therefore converge to that component or fail with
# WfnSymmetryError. Use the cylindrical symmetry solver to target the Sigma
# state unambiguously. See also 18-sz_degeneracy.py for the Sz degeneracy of
# the A2g state handled by the same solver.
mc = mcscf.CASSCF(mf, 8, 12)
mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
mc.fcisolver.wfnsym = 'A2u'
mc.kernel()
print('Triplet Sigma_u^- %.15g  ref = -149.217106540676' % mc.e_tot)

# Specify CI wfn spatial symmetry and spin symmetry
# A fresh CASSCF object is needed for the quintet state. If the converged
# triplet CI (an eigenfunction of the spin-penalty shifted Hamiltonian) were
# reused as the FCI initial guess, the Davidson iteration would be unable to
# leave the triplet state.
mc = mcscf.CASSCF(mf, 8, 12)
mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
mc.fix_spin_(ss=6)  # Quintet, ss = S*(S+1) = 6
mc.fcisolver.wfnsym = 'A2u'
mc.kernel()
print('Quintet Sigma_u^- %.15g  ref = -148.920732172365' % mc.e_tot)


#
# In the following example, without fix_spin_ decoration, it's probably unable
# to converge to the correct spin state.
#
mol = gto.M(
    atom = 'Mn 0 0 0; Mn 0 0 2.5',
    basis = 'ccpvdz',
    symmetry = 1,
)
mf = scf.RHF(mol)
mf.set(level_shift=0.4).run()
mc = mcscf.CASCI(mf, 12, 14)
mc.fcisolver.max_cycle = 100
mo = mc.sort_mo_by_irrep({'A1g': 2, 'A1u': 2,
                          'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1,
                          'E2uy': 1, 'E2ux': 1, 'E2gy': 1, 'E2gx': 1},
                         {'A1g': 5, 'A1u': 5,
                          'E1uy': 2, 'E1ux': 2, 'E1gy': 2, 'E1gx': 2})
mc.kernel(mo)

mc.fix_spin_(shift=.5, ss=0)
mc.kernel(mo)

