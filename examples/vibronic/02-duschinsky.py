#!/usr/bin/env python

'''
The Duschinsky transformation between two electronic states.

Two harmonic states have different normal coordinates.  They are related by

    Q_f = J Q_i + K,        J = L_f^T L_i,   K = L_f^T M^(1/2) (x_i0 - x_f0)

where the rows of J index FINAL-state modes and the columns INITIAL-state modes.
Before J and K can be formed the two structures must be put into a common
(Eckart) frame; pyscf.vibronic does that automatically, rotating the final-state
mode matrix L_f together with the geometry.

This example uses two SCF states of water at different geometries to stand in
for a ground and an excited state.  In a real calculation the second state would
come from an excited-state optimisation and Hessian (TDDFT, CASSCF, EOM-CC, ...);
nothing in pyscf.vibronic depends on which method produced it.
'''

import numpy
from pyscf import gto, scf, vibronic
from pyscf.vibronic import units

basis = '631g'

# ---- state i: the reference geometry -------------------------------------
mol_i = gto.M(atom = '''O   0.   0.       0.1173
                        H   0.   0.7572  -0.4692
                        H   0.  -0.7572  -0.4692''', basis = basis)
mf_i = mol_i.RHF().run()
model_i = vibronic.HarmonicModel.from_mole(mol_i, mf_i.Hessian().kernel(),
                                           energy=mf_i.e_tot)

# ---- state f: a different geometry, deliberately translated and rotated ---
# The transformation must be invariant to this rigid-body motion.
mol_f = gto.M(atom = '''O   1.0   2.0    3.0
                        H   1.9   2.55   3.05
                        H   0.35  2.75   3.10''', basis = basis)
mf_f = mol_f.RHF().run()
model_f = vibronic.HarmonicModel.from_mole(mol_f, mf_f.Hessian().kernel(),
                                           energy=mf_f.e_tot,
                                           imaginary_policy='warn')

dusch = vibronic.duschinsky_transform(model_i, model_f)

numpy.set_printoptions(precision=4, suppress=True)
print('\nDuschinsky matrix J (rows = final modes, cols = initial modes):')
print(dusch.J)
print('\nJ^T J - I  (zero only if both states span the same vibrational space)')
print(dusch.J.T.dot(dusch.J) - numpy.eye(dusch.nvib_i))

print('\nK [bohr sqrt(m_e)]        %s' % numpy.round(dusch.K, 4))
print('K dimensionless            %s' % numpy.round(dusch.K_dimensionless, 4))
print('Huang-Rhys factors S       %s' % numpy.round(dusch.huang_rhys, 4))
print('reorganization energy      %.6f Eh = %.1f cm^-1'
      % (dusch.total_reorganization_energy,
         units.au2wavenumber(dusch.total_reorganization_energy)))

# All the diagnostics, including the Eckart residual before and after
# alignment and how much of the geometry change leaked out of the vibrational
# subspace (excluded_mode_norm, which must be small).
dusch.dump_diagnostics()

# Mode correlation.  Modes are matched by their J overlap with an optimal
# (Hungarian) assignment, never by frequency alone.  Degenerate modes are
# reported as blocks, because an individual correspondence inside a degenerate
# subspace carries no physical meaning.
dusch.match_modes().dump()

# Q_f = J Q_i + K is executable, not just a docstring claim:
rng = numpy.random.RandomState(0)
q_i = rng.randn(dusch.nvib_i)
print('\napply() reproduces J Q_i + K to %.2e'
      % abs(dusch.apply(q_i) - (dusch.J.dot(q_i) + dusch.K)).max())
