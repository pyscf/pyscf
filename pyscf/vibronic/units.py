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
Unit conventions for :mod:`pyscf.vibronic`.

This module is the *single* place where unit conversion factors appear.  No
other module in :mod:`pyscf.vibronic` may hard-code a conversion factor.

Internal convention
-------------------
Every quantity stored on a :class:`~pyscf.vibronic.normal_modes.HarmonicModel`,
:class:`~pyscf.vibronic.duschinsky.Duschinsky` or
:class:`~pyscf.vibronic.franck_condon.FranckCondonResult` object is in Hartree
atomic units (:math:`\\hbar = m_e = e = a_0 = 1`):

===========================  =========================  =========================
Quantity                     Symbol                     Atomic unit
===========================  =========================  =========================
Cartesian geometry           :math:`x`                  bohr
Atomic mass                  :math:`m`                  electron mass
Cartesian Hessian            :math:`H`                  Eh/bohr^2
Mass-weighted Hessian        :math:`\\tilde H`           Eh/(bohr^2 m_e)
Mass-weighted displacement   :math:`q = M^{1/2}(x-x_0)`  bohr sqrt(m_e)
Normal mode matrix           :math:`L`                  dimensionless
Normal coordinate            :math:`Q = L^T q`          bohr sqrt(m_e)
Force constant               :math:`\\lambda = \\omega^2`  Eh^2 (hbar = 1)
Angular frequency            :math:`\\omega`             Eh (hbar = 1)
Dimensionless coordinate     :math:`\\bar q = \\omega^{1/2} Q`  dimensionless
Energy                       :math:`E`                  Eh
===========================  =========================  =========================

Note in particular that masses are converted from the atomic mass unit (amu)
returned by :meth:`pyscf.gto.Mole.atom_mass_list` to *electron masses* with
:data:`AMU2AU`.  This differs from :mod:`pyscf.hessian.thermo`, which
mass-weights in amu and therefore reports a ``freq_au`` that is not a true
atomic-unit angular frequency.  Frequencies produced here are genuine atomic
units, so that :math:`\\hbar\\omega` is directly an energy in Hartree.

Anharmonicity is not treated anywhere in this package; ``omega`` is always the
harmonic angular frequency.
'''

import numpy
from pyscf.data import nist

__all__ = [
    'AMU2AU', 'BOHR', 'HARTREE2EV', 'HARTREE2CM', 'HARTREE2J', 'BOLTZMANN_AU',
    'NM_TIMES_HARTREE',
    'amu2au', 'au2amu',
    'au2wavenumber', 'wavenumber2au',
    'au2ev', 'ev2au',
    'au2nm', 'nm2au',
    'convert_energy_from_au', 'convert_energy_to_au',
    'ENERGY_UNITS',
]

#: amu -> electron mass
AMU2AU = nist.AMU2AU

#: bohr -> Angstrom
BOHR = nist.BOHR

#: Hartree -> eV
HARTREE2EV = nist.HARTREE2EV

#: Hartree -> cm^-1 (wavenumber)
HARTREE2CM = nist.HARTREE2WAVENUMBER

#: Hartree -> Joule
HARTREE2J = nist.HARTREE2J

#: Boltzmann constant in Eh/K
BOLTZMANN_AU = nist.BOLTZMANN / nist.HARTREE2J

#: Product of a wavelength in nm and the corresponding photon energy in Eh.
#: ``wavelength_nm = NM_TIMES_HARTREE / energy_Eh``  (and vice versa).
NM_TIMES_HARTREE = 1e7 / HARTREE2CM


def amu2au(mass):
    '''Convert a mass from unified atomic mass units to electron masses.'''
    return numpy.asarray(mass) * AMU2AU


def au2amu(mass):
    '''Convert a mass from electron masses to unified atomic mass units.'''
    return numpy.asarray(mass) / AMU2AU


def au2wavenumber(omega):
    '''Angular frequency in atomic units (Eh, hbar=1) -> wavenumber in cm^-1.'''
    return numpy.asarray(omega) * HARTREE2CM


def wavenumber2au(nu):
    '''Wavenumber in cm^-1 -> angular frequency in atomic units.'''
    return numpy.asarray(nu) / HARTREE2CM


def au2ev(energy):
    '''Energy in Hartree -> eV.'''
    return numpy.asarray(energy) * HARTREE2EV


def ev2au(energy):
    '''Energy in eV -> Hartree.'''
    return numpy.asarray(energy) / HARTREE2EV


def au2nm(energy):
    '''Photon energy in Hartree -> vacuum wavelength in nm.

    Zero energy maps to ``inf``; negative energies map to negative wavelengths,
    which is meaningless physically and signals a sign-convention error
    upstream.
    '''
    energy = numpy.asarray(energy, dtype=float)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        return NM_TIMES_HARTREE / energy


def nm2au(wavelength):
    '''Vacuum wavelength in nm -> photon energy in Hartree.'''
    wavelength = numpy.asarray(wavelength, dtype=float)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        return NM_TIMES_HARTREE / wavelength


#: Energy units accepted by the user-facing spectrum functions.  ``'nm'`` is
#: deliberately excluded from linear conversions because it is reciprocal in
#: energy; use :func:`au2nm` / :func:`nm2au` explicitly.
ENERGY_UNITS = {
    'au': 1.0,
    'hartree': 1.0,
    'eh': 1.0,
    'ev': HARTREE2EV,
    'cm-1': HARTREE2CM,
    'cm^-1': HARTREE2CM,
    'wavenumber': HARTREE2CM,
}


def _unit_factor(unit):
    key = unit.strip().lower()
    if key not in ENERGY_UNITS:
        raise ValueError(
            'Unsupported energy unit %r.  Supported: %s.  For wavelengths use '
            'au2nm/nm2au explicitly, since nm is reciprocal in energy.'
            % (unit, ', '.join(sorted(ENERGY_UNITS))))
    return ENERGY_UNITS[key]


def convert_energy_from_au(energy, unit):
    '''Convert an energy (or energy difference) from Hartree to ``unit``.'''
    return numpy.asarray(energy) * _unit_factor(unit)


def convert_energy_to_au(energy, unit):
    '''Convert an energy (or energy difference) given in ``unit`` to Hartree.'''
    return numpy.asarray(energy) / _unit_factor(unit)
