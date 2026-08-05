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
High-level Franck-Condon workflow driver.

This is the entry point most users want.  It ties together
:mod:`~pyscf.vibronic.normal_modes`, :mod:`~pyscf.vibronic.duschinsky`,
:mod:`~pyscf.vibronic.franck_condon` and :mod:`~pyscf.vibronic.spectrum` so that
no manipulation of mass-weighted Cartesian coordinates is required::

    from pyscf import gto, scf, vibronic

    fc = vibronic.FranckCondon(mol_i, hess_i, mol_f, hess_f,
                               initial_energy=e_i, final_energy=e_f)
    result = fc.run()
    sticks = result.stick_spectrum(kind='absorption')
    spec = sticks.broaden(profile='gaussian', width=300, unit='cm-1')
    x, y = spec.x, spec.y

(``StickSpectrum.broaden`` returns a ``BroadenedSpectrum`` object; the
lower-level :func:`pyscf.vibronic.spectrum.broaden` function returns a plain
``(grid, signal)`` tuple.)

The two states must each be at their **own** optimised geometry, with a Hessian
computed at that geometry; the harmonic model of a non-stationary structure is
not meaningful and imaginary frequencies raise by default.

The driver is method-independent.  ``mol_i``/``mol_f`` may be
:class:`pyscf.gto.Mole` objects or ready-made
:class:`~pyscf.vibronic.normal_modes.HarmonicModel` objects, so the underlying
energies and Hessians may come from HF, DFT, TDDFT, CASSCF, EOM-CC or an
external program.
'''

import sys
import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf.vibronic import units
from pyscf.vibronic import analysis
from pyscf.vibronic import franck_condon as fc_mod
from pyscf.vibronic.duschinsky import duschinsky_transform
from pyscf.vibronic.normal_modes import HarmonicModel

__all__ = ['FranckCondon']


class FranckCondon(lib.StreamObject):
    '''Harmonic Franck-Condon vibronic spectra with Duschinsky rotation.

    Args:
        initial_mol : :class:`pyscf.gto.Mole` or
            :class:`~pyscf.vibronic.normal_modes.HarmonicModel`
            The state the transition starts **from**.  For an absorption
            spectrum this is the ground state; for an emission spectrum it is
            the *excited* state (see :meth:`kernel` and the note on sign
            conventions in :mod:`pyscf.vibronic.spectrum`).
        initial_hessian : Cartesian Hessian of the initial state in Eh/bohr^2,
            either ``(natm,natm,3,3)`` as returned by ``mf.Hessian().kernel()``
            or flat ``(3N,3N)``.  Ignored (and may be ``None``) if
            ``initial_mol`` is already a ``HarmonicModel``.
        final_mol, final_hessian : the same for the state the transition ends at.

    Kwargs:
        initial_energy, final_energy : float
            Electronic (bottom-of-well) energies in Eh.  Their difference sets
            the adiabatic energy; if either is ``None`` the adiabatic energy is
            taken as ``0`` and only *relative* vibronic energies are meaningful.
        initial_mass, final_mass : (natm,) masses in **amu**, overriding
            ``mol.atom_mass_list()``.  Use this for isotopic substitution.  The
            two states must describe the same nuclei, so in practice pass the
            same array to both (see :meth:`with_isotopes`).
        transition_dipole : float, (3,) or None
            Electronic transition dipole at the Condon point, atomic units.
            Only its squared norm enters, as a constant scale factor.  ``None``
            means unit oscillator strength.
        isotope_avg : bool, passed to ``mol.atom_mass_list``.

    Attributes that control the calculation (set before :meth:`kernel`):
        max_quanta : int, default 4.  Maximum total quanta in the final state.
        max_modes_excited : int or None.  Maximum number of simultaneously
            excited final-state modes ("class" order).  ``None`` means no limit
            beyond ``max_quanta``.
        max_quanta_per_mode : int or None.
        active_threshold : float or None.  When set, classes of order >= 2 are
            restricted to modes with Huang-Rhys factor above this value (united
            with modes carrying a large ``J`` off-diagonal).
        max_states : int or None.  Hard cap on the number of enumerated states.
            Whatever is dropped is counted and reported, never silently ignored.
        intensity_threshold : float, default 0.  Lines weaker than this are not
            stored but are still counted in the sum rule.
        temperature : float, default 0 K.  ``> 0`` populates initial vibrational
            states from a Boltzmann distribution and uses the general
            ``<v_f|v_i>`` recursion.
        align : bool, default True.  Eckart-align the two structures.
        allow_imaginary : bool, default False.

    Results (set by :meth:`kernel`):
        model_i, model_f : the two :class:`HarmonicModel` objects.
        duschinsky : the :class:`~pyscf.vibronic.duschinsky.Duschinsky` object.
        result : the :class:`~pyscf.vibronic.franck_condon.FranckCondonResult`.

    Examples:

    >>> from pyscf import gto, scf, vibronic
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    >>> mf = scf.RHF(mol).run()
    >>> h = mf.Hessian().kernel()
    >>> fc = vibronic.FranckCondon(mol, h, mol, h,
    ...                            initial_energy=mf.e_tot, final_energy=mf.e_tot)
    >>> res = fc.run()
    >>> abs(res.fcf[0] - 1.0) < 1e-12          # identical states -> only 0-0
    True
    '''

    max_quanta = 4
    max_modes_excited = None
    max_quanta_per_mode = None
    active_threshold = None
    max_states = None
    intensity_threshold = 0.0
    temperature = 0.0
    max_quanta_init = None
    align = True
    allow_imaginary = False
    sum_rule_warn = 0.9

    def __init__(self, initial_mol, initial_hessian=None, final_mol=None,
                 final_hessian=None, initial_energy=None, final_energy=None,
                 initial_mass=None, final_mass=None, transition_dipole=None,
                 isotope_avg=True, verbose=None, stdout=None):
        if final_mol is None:
            raise ValueError('final_mol is required')
        self.stdout = sys.stdout if stdout is None else stdout
        ref = initial_mol if verbose is None else None
        self.verbose = getattr(ref, 'verbose', logger.NOTE) if verbose is None else verbose

        self._initial_mol = initial_mol
        self._initial_hessian = initial_hessian
        self._final_mol = final_mol
        self._final_hessian = final_hessian
        self.initial_energy = initial_energy
        self.final_energy = final_energy
        self.initial_mass = initial_mass
        self.final_mass = final_mass
        self.isotope_avg = isotope_avg
        self.transition_dipole = transition_dipole

        self.model_i = None
        self.model_f = None
        self.duschinsky = None
        self.result = None

        # Fail fast on argument-shape mistakes, rather than at kernel() time.
        self._check_state_args(initial_mol, initial_hessian, 'initial')
        self._check_state_args(final_mol, final_hessian, 'final')

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _check_state_args(mol, hessian, label):
        '''Validate the (mol, hessian) pairing for one electronic state.'''
        if isinstance(mol, HarmonicModel):
            if hessian is not None:
                raise ValueError(
                    '%s_mol is already a HarmonicModel, so %s_hessian must be None; '
                    'the Hessian is already part of the model.' % (label, label))
        elif hessian is None:
            raise ValueError(
                '%s_hessian is required when %s_mol is a Mole.  Pass the Cartesian '
                'Hessian in Eh/bohr^2, either (natm,natm,3,3) as returned by '
                'mf.Hessian().kernel() or flat (3N,3N).' % (label, label))

    def _build_model(self, mol, hessian, energy, mass, label):
        self._check_state_args(mol, hessian, label)
        if isinstance(mol, HarmonicModel):
            return mol
        return HarmonicModel.from_mole(
            mol, hessian, energy=energy, mass=mass, isotope_avg=self.isotope_avg,
            imaginary_policy='warn' if self.allow_imaginary else 'raise',
            verbose=self.verbose, stdout=self.stdout)

    @property
    def e_adiabatic(self):
        '''``E_elec_final - E_elec_initial`` in Eh, or ``0.0`` if unavailable.

        When it is ``0.0`` because an energy was not supplied, only *relative*
        vibronic energies are meaningful; the absolute position of the band is
        undefined.
        '''
        e_i = self.initial_energy
        e_f = self.final_energy
        if e_i is None and self.model_i is not None:
            e_i = self.model_i.energy
        if e_f is None and self.model_f is not None:
            e_f = self.model_f.energy
        if e_i is None or e_f is None:
            return 0.0
        return float(e_f) - float(e_i)

    def with_isotopes(self, mass):
        '''Return a copy of this driver with both states using ``mass`` (amu).

        Isotopic substitution changes the nuclear masses, not the electronic
        structure, so the same mass array must be used for both electronic
        states -- the Duschinsky transformation rejects a mass mismatch.  This
        helper makes that the only possible outcome.

        The geometries, Hessians and energies are reused unchanged, which is
        correct within the Born-Oppenheimer approximation.
        '''
        for label, mol in (('initial', self._initial_mol), ('final', self._final_mol)):
            if isinstance(mol, HarmonicModel):
                raise ValueError(
                    'with_isotopes() needs %s_mol to be a Mole, not a prebuilt '
                    'HarmonicModel (its masses are already baked in).  Rebuild the '
                    'driver from Mole objects, or construct the HarmonicModel '
                    'yourself with the desired mass.' % label)
        mass = numpy.asarray(mass, dtype=float).ravel()
        new = FranckCondon(
            self._initial_mol, self._initial_hessian, self._final_mol,
            self._final_hessian, initial_energy=self.initial_energy,
            final_energy=self.final_energy, initial_mass=mass, final_mass=mass,
            transition_dipole=self.transition_dipole,
            isotope_avg=self.isotope_avg, verbose=self.verbose, stdout=self.stdout)
        for key in ('max_quanta', 'max_modes_excited', 'max_quanta_per_mode',
                    'active_threshold', 'max_states', 'intensity_threshold',
                    'temperature', 'max_quanta_init', 'align', 'allow_imaginary'):
            setattr(new, key, getattr(self, key))
        return new

    # -- driver -------------------------------------------------------------

    def kernel(self):
        '''Run the whole workflow and return the
        :class:`~pyscf.vibronic.franck_condon.FranckCondonResult`.

        Steps: build both harmonic models, Eckart-align them and form the
        Duschinsky transformation, enumerate final vibrational states, evaluate
        the overlaps, and assemble the result.  The sum rule and the enumeration
        truncation are logged; a sum rule below :attr:`sum_rule_warn` produces a
        warning, because it means the spectrum is missing intensity.
        '''
        log = logger.new_logger(self, self.verbose)
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.model_i = self._build_model(self._initial_mol, self._initial_hessian,
                                         self.initial_energy, self.initial_mass, 'initial')
        self.model_f = self._build_model(self._final_mol, self._final_hessian,
                                         self.final_energy, self.final_mass, 'final')
        log.info('initial state: nvib = %d, rotor = %s, ZPE = %.8f Eh',
                 self.model_i.nvib, self.model_i.rotor_type, self.model_i.zpe)
        log.info('final   state: nvib = %d, rotor = %s, ZPE = %.8f Eh',
                 self.model_f.nvib, self.model_f.rotor_type, self.model_f.zpe)

        self.duschinsky = duschinsky_transform(
            self.model_i, self.model_f, align=self.align,
            allow_imaginary=self.allow_imaginary, verbose=self.verbose)
        cput0 = log.timer('Duschinsky transformation', *cput0)

        self.result = fc_mod.franck_condon_factors(
            self.model_i.freq, self.model_f.freq,
            self.duschinsky.J, self.duschinsky.K,
            e_adiabatic=self.e_adiabatic,
            max_quanta=self.max_quanta,
            max_modes_excited=self.max_modes_excited,
            max_quanta_per_mode=self.max_quanta_per_mode,
            active_threshold=self.active_threshold,
            max_states=self.max_states,
            intensity_threshold=self.intensity_threshold,
            temperature=self.temperature,
            max_quanta_init=self.max_quanta_init,
            duschinsky=self.duschinsky,
            verbose=self.verbose, stdout=self.stdout)
        log.timer('Franck-Condon overlaps', *cput0)

        # The closure rule converges to 1/|det J|, not to 1; see
        # FranckCondonResult.sum_rule_target.
        rep = analysis.sum_rule_report(self.result, warn_below=self.sum_rule_warn)
        if not rep['adequate']:
            log.warn('Franck-Condon sum rule captures only %.6f of its target %.8f '
                     '(= 1/|det J|); %.1f%% of the intensity lies outside the '
                     'enumerated final-state space.  Increase max_quanta or '
                     'max_modes_excited before interpreting the spectrum.',
                     rep['fraction_captured'], rep['target'],
                     100 * (1 - rep['fraction_captured']))
        else:
            log.info('Franck-Condon sum rule = %.8f of target %.8f (= 1/|det J|)',
                     rep['sum_rule'], rep['target'])
        return self.result

    def run(self, **kwargs):
        '''Set attributes from ``kwargs``, then :meth:`kernel`.'''
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError('%s has no attribute %r'
                                     % (self.__class__.__name__, key))
            setattr(self, key, val)
        return self.kernel()

    __call__ = run

    # -- reporting ----------------------------------------------------------

    def stick_spectrum(self, kind='absorption', **kwargs):
        '''Shorthand for ``self.result.stick_spectrum(...)``.

        Runs :meth:`kernel` first if it has not been run.  ``transition_dipole``
        defaults to the one supplied to the constructor.
        '''
        if self.result is None:
            self.kernel()
        kwargs.setdefault('transition_dipole', self.transition_dipole)
        return self.result.stick_spectrum(kind=kind, **kwargs)

    def analyze(self, verbose=None, nline=20):
        '''Print the Duschinsky and Franck-Condon summaries.'''
        if self.result is None:
            self.kernel()
        analysis.dump_duschinsky(self.duschinsky, verbose=verbose)
        analysis.dump_result(self.result, verbose=verbose, nline=nline)
        return self

    def e_00(self):
        '''Zero-point-corrected origin ``E_00`` in Eh.

        ``E_00 = E_adiabatic + ZPE_final - ZPE_initial``.  Distinct from the
        adiabatic energy (:attr:`e_adiabatic`, bottom of well to bottom of well)
        and from any vertical energy.
        '''
        if self.result is None:
            self.kernel()
        return self.result.e_00

    def summary(self, unit='eV'):
        '''Return a dict of the headline energies, converted to ``unit``.

        Keys ``e_adiabatic``, ``e_00``, ``zpe_initial``, ``zpe_final``,
        ``reorganization_energy``, ``sum_rule`` (dimensionless) and ``unit``.
        '''
        if self.result is None:
            self.kernel()
        conv = lambda x: float(units.convert_energy_from_au(x, unit))
        return {
            'e_adiabatic': conv(self.result.e_adiabatic),
            'e_00': conv(self.result.e_00),
            'zpe_initial': conv(self.result.zpe_i),
            'zpe_final': conv(self.result.zpe_f),
            'reorganization_energy': conv(self.duschinsky.total_reorganization_energy),
            'sum_rule': self.result.sum_rule,
            'unit': unit,
        }

    def __repr__(self):
        if self.result is None:
            return '<FranckCondon (not run)>'
        return ('<FranckCondon nvib=%d/%d nstate=%d E_00=%.4f eV sum_rule=%.6f>'
                % (self.model_i.nvib, self.model_f.nvib, self.result.nstate,
                   units.au2ev(self.result.e_00), self.result.sum_rule))
