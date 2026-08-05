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
Reporting and diagnostic analysis for :mod:`pyscf.vibronic`.

Nothing here performs a new electronic-structure or Franck-Condon calculation;
these are presentation and sanity-checking helpers built on top of
:class:`~pyscf.vibronic.duschinsky.Duschinsky` and
:class:`~pyscf.vibronic.franck_condon.FranckCondonResult`.

The quantities reported are the standard harmonic vibronic diagnostics:

Huang-Rhys factor
    :math:`S_k = \\tfrac12 \\omega_{f,k} K_k^2` (dimensionless), the mean number
    of quanta deposited in final-state mode :math:`k` by the displacement alone.
    For a single displaced mode with unchanged frequency the intensity
    distribution is Poissonian, :math:`|\\langle n|0\\rangle|^2 = e^{-S}S^n/n!`,
    so :math:`S` is directly the peak of the vibrational progression.
Mode reorganisation energy
    :math:`\\lambda_k = S_k \\hbar\\omega_{f,k}`, the energy released by relaxing
    along mode :math:`k` after a vertical transition.
Total reorganisation energy
    :math:`\\lambda = \\sum_k \\lambda_k`.  Together with the 0-0 energy this
    fixes the classical Stokes shift, :math:`\\approx 2\\lambda` in the
    equal-frequency (linear-coupling) limit.
Vertical energies
    :math:`E^{\\rm vert}_{\\rm abs} = E_{\\rm ad} + \\lambda_f` and
    :math:`E^{\\rm vert}_{\\rm em} = E_{\\rm ad} - \\lambda_i`, the gap between the
    two harmonic surfaces at the initial-state and at the final-state minimum.
    Exact within the harmonic model; see :func:`vertical_energies`.  Reported as
    diagnostics only -- individual vibronic lines are **never** placed with them.

.. warning::

    Everything in this module is harmonic and within the Condon approximation.
    No anharmonic or Herzberg-Teller correction is applied anywhere in
    :mod:`pyscf.vibronic`, so none of these numbers should be compared with
    experiment as though they were converged spectroscopic predictions.
'''

import numpy

from pyscf.lib import logger
from pyscf.vibronic import units

__all__ = [
    'huang_rhys_analysis', 'dump_duschinsky', 'dump_result',
    'mode_contributions', 'sum_rule_report', 'stokes_shift',
    'reorganization_energies', 'vertical_energies',
]


def huang_rhys_analysis(dusch):
    '''Per-mode Huang-Rhys factors and reorganisation energies.

    Args:
        dusch : :class:`~pyscf.vibronic.duschinsky.Duschinsky`.

    Returns:
        dict with keys

        ``mode``
            (nvib_f,) int, final-state mode indices sorted by **decreasing**
            Huang-Rhys factor (so the modes that dominate the vibronic
            progression come first).
        ``freq_wavenumber``
            (nvib_f,) final-state frequencies in cm^-1, same order.
        ``huang_rhys``
            (nvib_f,) dimensionless :math:`S_k`, same order.
        ``reorganization_energy``
            (nvib_f,) :math:`\\lambda_k = S_k \\omega_{f,k}` in Eh, same order.
        ``total_reorganization_energy``
            float, Eh.
        ``displacement``
            (nvib_f,) dimensionless displacement :math:`\\bar K_k`, same order.
    '''
    freq_f = numpy.asarray(dusch.freq_f, dtype=float)
    s = numpy.asarray(dusch.huang_rhys, dtype=float)
    lam = s * freq_f
    order = numpy.argsort(-s, kind='stable')
    kbar = numpy.asarray(dusch.K_dimensionless, dtype=float)
    return {
        'mode': order,
        'freq_wavenumber': units.au2wavenumber(freq_f[order]),
        'huang_rhys': s[order],
        'reorganization_energy': lam[order],
        'total_reorganization_energy': float(lam.sum()),
        'displacement': kbar[order],
    }


def mode_contributions(result, nmode_max=None):
    '''Intensity carried by each final-state mode, summed over all lines.

    For every final-state mode ``k`` this sums ``population * fcf`` over every
    enumerated transition in which mode ``k`` is excited, weighted by the number
    of quanta in it.  It answers "which modes actually build the band shape",
    which is not the same question as "which modes have a large Huang-Rhys
    factor" once Duschinsky mixing is significant.

    Args:
        result : :class:`~pyscf.vibronic.franck_condon.FranckCondonResult`.
        nmode_max : int or None, truncate the returned ordering.

    Returns:
        dict with ``mode``, ``freq_wavenumber``, ``intensity`` and
        ``quanta_weighted_intensity``, ordered by decreasing
        ``quanta_weighted_intensity``.
    '''
    states = numpy.asarray(result.states, dtype=float)
    weight = numpy.asarray(result.populations, dtype=float) * numpy.asarray(result.fcf, dtype=float)
    if states.size == 0:
        empty = numpy.zeros(0)
        return {'mode': numpy.zeros(0, dtype=int), 'freq_wavenumber': empty,
                'intensity': empty, 'quanta_weighted_intensity': empty}
    excited = (states > 0).astype(float)
    intensity = excited.T.dot(weight)
    quanta = states.T.dot(weight)
    order = numpy.argsort(-quanta, kind='stable')
    if nmode_max is not None:
        order = order[:nmode_max]
    return {
        'mode': order,
        'freq_wavenumber': units.au2wavenumber(numpy.asarray(result.freq_f)[order]),
        'intensity': intensity[order],
        'quanta_weighted_intensity': quanta[order],
    }


def reorganization_energies(dusch):
    '''The two reorganisation energies of the state pair, in Hartree.

    ``lambda_f`` is measured on the **final**-state surface at the initial-state
    minimum, and ``lambda_i`` on the **initial**-state surface at the
    final-state minimum:

    .. math::

        \\lambda_f = \\tfrac12 \\sum_k \\omega_{f,k}^2 K_k^2, \\qquad
        \\lambda_i = \\tfrac12 \\sum_j \\omega_{i,j}^2 \\,(J^{+}K)_j^2 .

    The first follows from :math:`Q_f = K` when :math:`Q_i = 0`; the second from
    :math:`Q_i = -J^{+}K` when :math:`Q_f = 0`.  Both are exact within the
    harmonic model -- no linear-coupling assumption is made, and they differ
    whenever the two states' frequencies differ.

    Note ``lambda_f`` equals ``sum_k S_k * omega_f,k``, i.e.
    :attr:`Duschinsky.total_reorganization_energy`.

    Returns:
        dict with ``lambda_f``, ``lambda_i``, ``lambda_f_modes``,
        ``lambda_i_modes``.
    '''
    freq_f = numpy.asarray(dusch.freq_f, dtype=float)
    freq_i = numpy.asarray(dusch.freq_i, dtype=float)
    k_vec = numpy.asarray(dusch.K, dtype=float)
    lam_f_modes = 0.5 * freq_f ** 2 * k_vec ** 2

    j = numpy.asarray(dusch.J, dtype=float)
    # Q_i at the final-state minimum: solve J Q_i = -K in the least-squares
    # sense, which is exact when J has full column rank.
    q_i = -numpy.linalg.pinv(j).dot(k_vec)
    lam_i_modes = 0.5 * freq_i ** 2 * q_i ** 2
    return {
        'lambda_f': float(numpy.nansum(lam_f_modes)),
        'lambda_i': float(numpy.nansum(lam_i_modes)),
        'lambda_f_modes': lam_f_modes,
        'lambda_i_modes': lam_i_modes,
    }


def vertical_energies(dusch, e_adiabatic):
    '''Vertical absorption and emission energies, in Hartree.

    These are **diagnostics only**.  Individual vibronic lines are never placed
    with them: line positions come from :math:`E_{00}` plus the vibrational
    quanta (see :mod:`pyscf.vibronic`).

    With the initial state the lower one and
    :math:`E_{\\rm ad} = E^{\\rm elec}_f - E^{\\rm elec}_i`,

    .. math::

        E^{\\rm vert}_{\\rm abs} = E_{\\rm ad} + \\lambda_f, \\qquad
        E^{\\rm vert}_{\\rm em}  = E_{\\rm ad} - \\lambda_i,

    the gap between the two harmonic surfaces evaluated at the initial-state and
    at the final-state minimum respectively (see
    :func:`reorganization_energies`).  Both are exact within the harmonic model.

    Distinguish these clearly from the adiabatic energy (bottom of well to
    bottom of well) and from :math:`E_{00}` (zero-point corrected origin); the
    four are different quantities and only the last two place spectral lines.

    Returns:
        dict with ``e_adiabatic``, ``vertical_absorption``,
        ``vertical_emission``, ``stokes_shift`` (their difference,
        :math:`\\lambda_f + \\lambda_i`), ``lambda_f`` and ``lambda_i``.
    '''
    e_ad = float(e_adiabatic)
    lam = reorganization_energies(dusch)
    e_abs = e_ad + lam['lambda_f']
    e_em = e_ad - lam['lambda_i']
    return {
        'e_adiabatic': e_ad,
        'vertical_absorption': e_abs,
        'vertical_emission': e_em,
        'stokes_shift': e_abs - e_em,
        'lambda_f': lam['lambda_f'],
        'lambda_i': lam['lambda_i'],
    }


def stokes_shift(dusch):
    '''Harmonic vertical Stokes shift :math:`\\lambda_f + \\lambda_i`, in Hartree.

    This is the separation of the two *vertical* energies (see
    :func:`vertical_energies`), which is exact within the harmonic model and
    correctly accounts for the frequency change between the states.  In the
    equal-frequency (linear-coupling) limit it reduces to the familiar
    :math:`2\\lambda`.

    .. warning::

        It is still a *vertical* quantity, not a prediction of the observed
        peak-to-peak separation: the maxima of the computed band shapes are
        shifted from the vertical energies by the vibronic envelope.  For the
        peak separation, broaden the computed absorption and emission spectra
        and take their maxima -- as ``examples/vibronic/04`` does.
    '''
    lam = reorganization_energies(dusch)
    return lam['lambda_f'] + lam['lambda_i']


def sum_rule_report(result, warn_below=0.9):
    '''Assess whether the enumerated final-state space is adequate.

    The closure rule over the *complete* set of final states is

    .. math::

        \\sum_{v_f} |\\langle v_f | 0_i \\rangle|^2 = \\frac{1}{|\\det J|},

    which equals 1 only when the Duschinsky matrix is orthogonal, i.e. when both
    electronic states span the same vibrational subspace.  Two different
    equilibrium geometries have slightly different rotational subspaces, so
    :math:`|\\det J|` typically deviates from 1 at the 1e-3 level.  Completeness
    is therefore judged against ``result.sum_rule_target``, not against 1;
    comparing against 1 would misreport that geometric shift as an enumeration
    error (and can even make the sum appear to exceed 1).

    A finite enumeration always falls **below** the target, and the shortfall is
    the fraction of intensity that has been missed.

    Returns:
        dict with ``sum_rule``, ``target`` (``1/|det J|``), ``missing``
        (``target - sum_rule``), ``fraction_captured``
        (``sum_rule / target``), ``adequate``
        (bool, ``fraction_captured >= warn_below``) and ``truncation``
        (a copy of the enumeration bookkeeping).
    '''
    s = float(result.sum_rule)
    target = float(getattr(result, 'sum_rule_target', 1.0))
    fraction = s / target if target > 0 else float('nan')
    return {
        'sum_rule': s,
        'target': target,
        'missing': target - s,
        'fraction_captured': fraction,
        'adequate': fraction >= warn_below,
        'truncation': dict(result.truncation),
    }


def dump_duschinsky(dusch, verbose=None, nmode_print=10):
    '''Print a Duschinsky/Huang-Rhys summary through :mod:`pyscf.lib.logger`.'''
    log = logger.new_logger(dusch, verbose)
    log.note('')
    log.note('==== Duschinsky analysis ====')
    log.note('vibrational modes: initial %d, final %d', dusch.nvib_i, dusch.nvib_f)

    diag = dict(getattr(dusch, 'diagnostics', {}) or {})
    for key, fmt in (('orthogonality_error', '%.3e'),
                     ('row_orthogonality_error', '%.3e'),
                     ('det_J', '%+.6f'),
                     ('excluded_mode_norm', '%.3e'),
                     ('displacement_reconstruction_error', '%.3e'),
                     ('eckart_residual_before', '%.3e'),
                     ('eckart_residual_after', '%.3e'),
                     ('max_offdiag_J', '%.4f')):
        if key in diag and diag[key] is not None:
            log.note(('  %-36s ' + fmt), key, diag[key])

    ana = huang_rhys_analysis(dusch)
    lam = ana['total_reorganization_energy']
    log.note('  %-36s %.6f Eh = %.1f cm^-1 = %.4f eV',
             'total reorganization energy', lam,
             units.au2wavenumber(lam), units.au2ev(lam))
    log.note('  %-36s %.6f Eh = %.1f cm^-1  (linear-coupling estimate only)',
             'classical Stokes shift ~ 2*lambda', 2 * lam, units.au2wavenumber(2 * lam))

    n = min(nmode_print, len(ana['mode']))
    if n:
        log.note('  strongest %d final-state mode(s) by Huang-Rhys factor:', n)
        log.note('    %5s %14s %10s %10s %14s', 'mode', 'freq/cm^-1', 'S', 'Kbar',
                 'lambda/cm^-1')
        for idx in range(n):
            log.note('    %5d %14.2f %10.5f %10.5f %14.2f',
                     ana['mode'][idx], ana['freq_wavenumber'][idx],
                     ana['huang_rhys'][idx], ana['displacement'][idx],
                     units.au2wavenumber(ana['reorganization_energy'][idx]))
    return dusch


def dump_result(result, verbose=None, nline=20, unit='cm-1'):
    '''Print a Franck-Condon result summary through :mod:`pyscf.lib.logger`.

    Lists the strongest ``nline`` transitions with their assignments, and always
    reports the sum rule and the enumeration truncation so a truncated
    calculation can never be mistaken for a converged one.
    '''
    log = logger.new_logger(result, verbose)
    log.note('')
    log.note('==== Franck-Condon result ====')
    log.note('  adiabatic energy E_ad     = %.8f Eh = %.4f eV',
             result.e_adiabatic, units.au2ev(result.e_adiabatic))
    log.note('  ZPE initial / final       = %.8f / %.8f Eh', result.zpe_i, result.zpe_f)
    log.note('  0-0 energy E_00           = %.8f Eh = %.4f eV = %.1f cm^-1 = %.1f nm',
             result.e_00, units.au2ev(result.e_00), units.au2wavenumber(result.e_00),
             units.au2nm(result.e_00))
    if result.duschinsky is not None:
        vert = vertical_energies(result.duschinsky, result.e_adiabatic)
        log.note('  vertical absorption       = %.8f Eh = %.4f eV   (diagnostic only)',
                 vert['vertical_absorption'], units.au2ev(vert['vertical_absorption']))
        log.note('  vertical emission         = %.8f Eh = %.4f eV   (diagnostic only)',
                 vert['vertical_emission'], units.au2ev(vert['vertical_emission']))
        log.note('  vertical Stokes shift     = %.8f Eh = %.1f cm^-1',
                 vert['stokes_shift'], units.au2wavenumber(vert['stokes_shift']))
    log.note('  temperature               = %.2f K', result.temperature)
    log.note('  states stored             = %d', result.nstate)

    rep = sum_rule_report(result)
    log.note('  sum rule (closure)        = %.8f  of target 1/|det J| = %.8f '
             '(%.6f captured, missing %.2e)',
             rep['sum_rule'], rep['target'], rep['fraction_captured'], rep['missing'])
    if not rep['adequate']:
        log.warn('Franck-Condon sum rule captures only %.4f of its target %.6f: the '
                 'enumerated final-state space is incomplete and the computed spectrum '
                 'is missing %.1f%% of the intensity.  Increase max_quanta / '
                 'max_modes_excited.',
                 rep['fraction_captured'], rep['target'],
                 100 * (1 - rep['fraction_captured']))
    for key in sorted(rep['truncation']):
        log.note('    truncation[%-22s] = %s', key, rep['truncation'][key])

    if result.nstate == 0:
        log.note('  (no transitions stored)')
        return result

    fac = units.convert_energy_from_au(1.0, unit)
    weight = numpy.asarray(result.populations) * numpy.asarray(result.fcf)
    order = numpy.argsort(-weight, kind='stable')[:nline]
    log.note('  strongest %d transition(s):', len(order))
    log.note('    %14s %12s %12s   %s', 'dE/' + unit, 'FCF', 'pop*FCF', 'assignment')
    for idx in order:
        log.note('    %14.2f %12.4e %12.4e   %s',
                 result.energies[idx] * fac, result.fcf[idx], weight[idx],
                 _assignment_label(result.states[idx], result.init_states[idx]))
    return result


def _assignment_label(state_f, state_i):
    '''Human-readable spectroscopic label, e.g. ``3^1 7^2 <- 0`` or ``0 <- 0``.'''
    def side(occ):
        occ = numpy.asarray(occ)
        parts = ['%d^%d' % (k, occ[k]) for k in numpy.nonzero(occ)[0]]
        return ' '.join(parts) if parts else '0'
    return '%s <- %s' % (side(state_f), side(state_i))
