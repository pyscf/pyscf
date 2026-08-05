#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Harmonic Franck-Condon vibronic spectroscopy with Duschinsky rotation
=====================================================================

Given two electronic states, each characterised by an equilibrium geometry, a
Cartesian nuclear Hessian and an electronic energy, this subpackage builds the
two harmonic vibrational models, relates them through the Duschinsky
transformation

.. math::  Q_f = J Q_i + K,

and computes harmonic Franck-Condon factors, vibronic transition energies and
absorption or emission spectra.

The implementation is **method-independent**: it consumes geometries, Hessians
and energies, so the two states may come from HF, DFT, TDDFT, CASSCF, EOM-CC or
an external program.  Nothing here is specific to any one method.

Simple usage::

    >>> from pyscf import gto, scf, vibronic
    >>> # mol_i/hess_i for the initial state and mol_f/hess_f for the final
    >>> # state, each at its own optimised geometry
    >>> fc = vibronic.FranckCondon(mol_i, hess_i, mol_f, hess_f,
    ...                            initial_energy=e_i, final_energy=e_f)
    >>> fc.max_quanta = 6
    >>> res = fc.run()
    >>> sticks = res.stick_spectrum(kind='absorption')
    >>> spec = sticks.broaden(profile='gaussian', width=300, unit='cm-1')
    >>> spec.x, spec.y                                  # doctest: +SKIP

:meth:`~pyscf.vibronic.spectrum.StickSpectrum.broaden` returns a
:class:`~pyscf.vibronic.spectrum.BroadenedSpectrum` (with ``.x``/``.y``), while
the lower-level function :func:`~pyscf.vibronic.spectrum.broaden` returns a plain
``(grid, signal)`` tuple.

Scope
-----
Harmonic potential-energy surfaces within the Condon approximation, with the
full Duschinsky rotation (general mode mixing, frequency changes and equilibrium
displacement), transitions from the initial vibrational ground state, and
optionally from thermally populated initial states.

**Not** included, by design: VPT2, cubic/quartic force constants, general
anharmonic potentials, Herzberg-Teller (transition-dipole derivative) terms,
nonadiabatic couplings, solvent models, excited-state geometry optimisation and
plotting.  Every reported quantity is harmonic and should not be read as a
converged spectroscopic prediction.

Units
-----
Everything internal is in Hartree atomic units; see :mod:`pyscf.vibronic.units`
for the full table.  Frequencies are angular frequencies in Eh
(:math:`\\hbar = 1`), so ``units.au2wavenumber(freq)`` gives cm^-1.  Masses are
passed in **amu** at the API boundary, matching
:meth:`pyscf.gto.Mole.atom_mass_list`.

Energy conventions
------------------
``e_adiabatic``
    :math:`E^{\\rm elec}_f - E^{\\rm elec}_i`, bottom of well to bottom of well.
``e_00``
    :math:`E_{\\rm adiabatic} + {\\rm ZPE}_f - {\\rm ZPE}_i`, the
    zero-point-corrected origin.
vibronic transition energy
    :math:`E_{00} + \\sum_k \\omega_{f,k} v_{f,k} - \\sum_j \\omega_{i,j} v_{i,j}`.
vertical energies
    Reported as diagnostics only; individual lines are never placed with them.

For **absorption** pass the lower state as ``initial``; for **emission** pass the
*excited* state as ``initial``.  Requesting a ``kind`` that contradicts the sign
of :math:`\\Delta E` raises :class:`ValueError` rather than producing negative
photon energies.

Modules
-------
:mod:`~pyscf.vibronic.units`
    Unit conventions; the only place conversion factors are defined.
:mod:`~pyscf.vibronic.alignment`
    Centre of mass, inertia tensor, rotor classification, Kabsch/Eckart alignment.
:mod:`~pyscf.vibronic.normal_modes`
    :class:`~pyscf.vibronic.normal_modes.HarmonicModel`: translation/rotation
    projection, projected-Hessian diagonalisation, frequencies and modes.
:mod:`~pyscf.vibronic.duschinsky`
    The Duschinsky matrix ``J``, displacement ``K``, diagnostics, mode matching.
:mod:`~pyscf.vibronic.franck_condon`
    Overlap kernels (Doktorov recursion), state enumeration, array-level driver.
:mod:`~pyscf.vibronic.spectrum`
    Stick spectra, Gaussian/Lorentzian broadening.
:mod:`~pyscf.vibronic.analysis`
    Huang-Rhys/reorganisation reporting and sum-rule diagnostics.
:mod:`~pyscf.vibronic.workflow`
    :class:`~pyscf.vibronic.workflow.FranckCondon`, the high-level driver.
'''

from pyscf.vibronic import units
from pyscf.vibronic import alignment
from pyscf.vibronic import normal_modes
from pyscf.vibronic import duschinsky
from pyscf.vibronic import franck_condon
from pyscf.vibronic import spectrum
from pyscf.vibronic import analysis
from pyscf.vibronic import workflow

from pyscf.vibronic.normal_modes import HarmonicModel, harmonic_model
from pyscf.vibronic.duschinsky import (
    Duschinsky, duschinsky_transform, duschinsky_from_arrays, match_modes)
from pyscf.vibronic.franck_condon import (
    FranckCondonResult, franck_condon_factors, enumerate_states,
    overlap_1d, overlap_1d_table, overlap_00, multimode_overlaps, huang_rhys)
from pyscf.vibronic.spectrum import (
    StickSpectrum, BroadenedSpectrum, stick_spectrum, broaden,
    gaussian_profile, lorentzian_profile, trapezoid)
from pyscf.vibronic.workflow import FranckCondon

__all__ = [
    # submodules
    'units', 'alignment', 'normal_modes', 'duschinsky', 'franck_condon',
    'spectrum', 'analysis', 'workflow',
    # high-level API
    'FranckCondon',
    # normal modes
    'HarmonicModel', 'harmonic_model',
    # Duschinsky
    'Duschinsky', 'duschinsky_transform', 'duschinsky_from_arrays', 'match_modes',
    # Franck-Condon
    'FranckCondonResult', 'franck_condon_factors', 'enumerate_states',
    'overlap_1d', 'overlap_1d_table', 'overlap_00', 'multimode_overlaps',
    'huang_rhys',
    # spectra
    'StickSpectrum', 'BroadenedSpectrum', 'stick_spectrum', 'broaden',
    'gaussian_profile', 'lorentzian_profile', 'trapezoid',
]
