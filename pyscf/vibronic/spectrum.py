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

r'''
Stick spectra and lineshape broadening for :mod:`pyscf.vibronic`.

This module is deliberately free of any plotting dependency; it produces
numbers only.

Sign convention (DESIGN.md section 3)
=====================================
A :class:`~pyscf.vibronic.franck_condon.FranckCondonResult` stores the
*signed* transition energy ``Delta E = E_final - E_initial``.  The caller
chooses which electronic state plays the role of "initial":

* **absorption** -- the initial state is the lower one, ``Delta E > 0``, the
  photon energy is ``+Delta E``;
* **emission** -- the caller must supply the *excited* state as the initial
  state, so ``Delta E < 0`` and the photon energy is ``-Delta E``.

Requesting a ``kind`` inconsistent with the sign of ``Delta E`` raises a
:class:`ValueError` rather than silently returning negative photon energies.

Intensity convention
====================
``stick_spectrum`` returns, by default, the FCF weighted by the energy
prefactor appropriate to the requested observable:

* absorption cross-section:  ``|mu|^2 * population * FCF * E_photon``
* emission rate:             ``|mu|^2 * population * FCF * E_photon**3``

Pass ``line_strength=True`` for the bare ``|mu|^2 * population * FCF`` with no
energy prefactor -- convenient when comparing against codes that report raw
Franck-Condon factors.  The energy-weighted form is the **default**.
'''

import numpy

from pyscf.vibronic import units

__all__ = [
    'StickSpectrum', 'BroadenedSpectrum',
    'stick_spectrum', 'broaden', 'trapezoid',
    'gaussian_profile', 'lorentzian_profile',
]

#: FWHM = GAUSS_FWHM_TO_SIGMA * sigma
GAUSS_FWHM_TO_SIGMA = 2.0 * numpy.sqrt(2.0 * numpy.log(2.0))

#: Default truncation radius for the Gaussian kernel, in units of the FWHM.
#: 5 FWHM is 11.8 sigma, where the Gaussian is ~1e-31 of its peak.
DEFAULT_GAUSSIAN_CUTOFF = 5.0

PROFILES = ('gaussian', 'lorentzian')


def gaussian_profile(x, x0, fwhm):
    r'''Area-normalised Gaussian with the given full width at half maximum.

    .. math::

        g(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-x_0)^2/(2\sigma^2)},
        \qquad \sigma = \mathrm{FWHM} / (2\sqrt{2\ln 2})

    ``int g(x) dx = 1`` exactly, and ``g(x0 +- FWHM/2) = g(x0)/2`` exactly.
    ``x``, ``x0`` and ``fwhm`` must share the same unit; the return value has
    the reciprocal unit.
    '''
    fwhm = float(fwhm)
    if fwhm <= 0.0:
        raise ValueError('fwhm must be positive, got %r' % fwhm)
    sigma = fwhm / GAUSS_FWHM_TO_SIGMA
    z = (numpy.asarray(x, dtype=float) - x0) / sigma
    return numpy.exp(-0.5 * z * z) / (sigma * numpy.sqrt(2.0 * numpy.pi))


def lorentzian_profile(x, x0, fwhm):
    r'''Area-normalised Lorentzian with the given full width at half maximum.

    .. math::

        L(x) = \frac{1}{\pi}\frac{\gamma}{(x-x_0)^2 + \gamma^2},
        \qquad \gamma = \mathrm{FWHM}/2

    ``int L(x) dx = 1`` over the whole real line.  Note the ``1/x^2`` tails:
    truncating at ``+-n`` FWHM keeps only ``(2/pi) arctan(2n)`` of the area, a
    loss of ``1 - (2/pi) arctan(2n) ~ 1/(pi n)`` -- 6.3% at 5 FWHM and still
    0.64% at 50 FWHM.  Cutoffs must therefore be far wider than for a Gaussian,
    which is why :func:`broaden` disables the Lorentzian cutoff by default.
    '''
    fwhm = float(fwhm)
    if fwhm <= 0.0:
        raise ValueError('fwhm must be positive, got %r' % fwhm)
    gamma = 0.5 * fwhm
    dx = numpy.asarray(x, dtype=float) - x0
    return gamma / (numpy.pi * (dx * dx + gamma * gamma))


_PROFILE_FUNCS = {'gaussian': gaussian_profile, 'lorentzian': lorentzian_profile}


def trapezoid(y, x):
    '''Trapezoidal integral of ``y`` over the (not necessarily uniform) grid ``x``.

    Written out explicitly rather than calling ``numpy.trapz``, which NumPy 2.0
    removed in favour of ``numpy.trapezoid``, while PySCF still supports
    ``numpy>=1.13``.  Using neither name keeps this working on every supported
    NumPy version.
    '''
    y = numpy.asarray(y, dtype=float)
    x = numpy.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError('y %s and x %s must have the same shape' % (y.shape, x.shape))
    if y.size < 2:
        return 0.0
    return float(0.5 * numpy.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1])))


class BroadenedSpectrum(object):
    '''Convoluted spectrum on a grid.

    Attributes:
        x : (npoints,) grid, in ``unit``.
        y : (npoints,) signal, per unit of ``x``.  ``trapezoid(y, x)`` equals the
            total stick intensity (up to the profile truncation).
        unit : str
        kind : 'absorption' | 'emission' | None
        profile : 'gaussian' | 'lorentzian'
        width : float, the FWHM used, in ``unit``.
    '''

    def __init__(self, x, y, unit='au', kind=None, profile=None, width=None):
        self.x = numpy.asarray(x, dtype=float)
        self.y = numpy.asarray(y, dtype=float)
        self.unit = unit
        self.kind = kind
        self.profile = profile
        self.width = width

    @property
    def area(self):
        '''Numerically integrated area, which should match the stick total.'''
        return trapezoid(self.y, self.x)

    def to_unit(self, unit):
        '''Return a new :class:`BroadenedSpectrum` with ``x`` in ``unit``.

        ``y`` is rescaled by the inverse factor so that the area is preserved.
        '''
        fac = (units.convert_energy_from_au(1.0, unit)
               / units.convert_energy_from_au(1.0, self.unit))
        return BroadenedSpectrum(self.x * fac, self.y / fac, unit=unit,
                                 kind=self.kind, profile=self.profile,
                                 width=None if self.width is None else self.width * fac)

    def __repr__(self):
        return ('<BroadenedSpectrum %s npoints=%d x=[%.6g, %.6g] %s area=%.6g>'
                % (self.profile, self.x.size,
                   self.x[0] if self.x.size else numpy.nan,
                   self.x[-1] if self.x.size else numpy.nan,
                   self.unit, self.area if self.x.size > 1 else 0.0))


class StickSpectrum(object):
    '''Discrete vibronic transitions.

    Attributes:
        energies : (nline,) photon energies, **positive**, in ``unit``.
        intensities : (nline,) intensities (see the module docstring).
        assignments : (nline, nmode) int16 final-state occupation vectors.
        init_assignments : (nline, nmode_i) int16 initial-state occupations.
        kind : 'absorption' | 'emission'
        temperature : float, Kelvin.
        unit : str, the unit of ``energies``.
        line_strength : bool, whether the energy prefactor was omitted.
    '''

    def __init__(self, energies, intensities, assignments=None, kind='absorption',
                 temperature=0.0, unit='au', init_assignments=None,
                 line_strength=False):
        self.energies = numpy.asarray(energies, dtype=float)
        self.intensities = numpy.asarray(intensities, dtype=float)
        if assignments is None:
            assignments = numpy.zeros((self.energies.size, 0), dtype=numpy.int16)
        self.assignments = numpy.asarray(assignments, dtype=numpy.int16)
        if init_assignments is None:
            init_assignments = numpy.zeros((self.energies.size, 0), dtype=numpy.int16)
        self.init_assignments = numpy.asarray(init_assignments, dtype=numpy.int16)
        self.kind = kind
        self.temperature = float(temperature)
        self.unit = unit
        self.line_strength = bool(line_strength)

    @property
    def nline(self):
        return self.energies.size

    @property
    def total_intensity(self):
        return float(numpy.sum(self.intensities))

    def to_unit(self, unit):
        '''Return a new :class:`StickSpectrum` with ``energies`` in ``unit``.

        Intensities are *not* rescaled: a stick carries an integrated
        intensity, which is unit independent.
        '''
        fac = (units.convert_energy_from_au(1.0, unit)
               / units.convert_energy_from_au(1.0, self.unit))
        return StickSpectrum(self.energies * fac, self.intensities,
                             assignments=self.assignments, kind=self.kind,
                             temperature=self.temperature, unit=unit,
                             init_assignments=self.init_assignments,
                             line_strength=self.line_strength)

    def broaden(self, profile='gaussian', width=300.0, unit='cm-1', grid=None,
                npoints=2000, padding=None, cutoff=None):
        '''Convolute the sticks with a lineshape; see :func:`broaden`.

        Returns:
            :class:`BroadenedSpectrum`
        '''
        au = self.to_unit('au')
        x, y = broaden(au.energies, au.intensities, profile=profile, width=width,
                       unit=unit, grid=grid, npoints=npoints, padding=padding,
                       cutoff=cutoff)
        return BroadenedSpectrum(x, y, unit=unit, kind=self.kind, profile=profile,
                                 width=width)

    def __repr__(self):
        return ('<StickSpectrum %s nline=%d total_intensity=%.6g unit=%s T=%.2fK>'
                % (self.kind, self.nline, self.total_intensity, self.unit,
                   self.temperature))


def _dipole_strength(transition_dipole):
    '''``|mu|^2`` from ``None``, a scalar, or a Cartesian vector.'''
    if transition_dipole is None:
        return 1.0
    mu = numpy.asarray(transition_dipole, dtype=float)
    if mu.ndim == 0:
        return float(mu) ** 2
    return float(numpy.dot(mu.ravel(), mu.ravel()))


def stick_spectrum(result, kind='absorption', temperature=None,
                   intensity_threshold=0.0, transition_dipole=None,
                   line_strength=False, merge_tol=None, unit='au'):
    '''Convert a Franck-Condon result into a :class:`StickSpectrum`.

    Args:
        result : :class:`~pyscf.vibronic.franck_condon.FranckCondonResult`
            Anything exposing ``energies``, ``fcf``, ``populations``,
            ``states`` and ``init_states`` works.
        kind : 'absorption' or 'emission'
            ``'absorption'`` requires ``Delta E > 0`` for every line;
            ``'emission'`` requires ``Delta E < 0`` (the caller must have
            supplied the *excited* state as the initial state).
        temperature : float or None
            Informational; defaults to ``result.temperature``.  Populations are
            taken from ``result`` and are *not* recomputed here.
        intensity_threshold : float
            Absolute cut on the returned intensities.  Default ``0.0`` keeps
            everything.
        transition_dipole : None, float or (3,) array
            ``|mu|^2`` multiplies every intensity.  ``None`` means 1.
        line_strength : bool
            ``True`` omits the energy prefactor and returns
            ``|mu|^2 * population * FCF``.  Default ``False`` (energy-weighted).
        merge_tol : float or None
            Merge lines whose photon energies agree to within this tolerance
            (in Hartree), summing their intensities.  The assignment kept is
            that of the most intense contributor.  ``None`` disables merging.
        unit : str
            Unit of the returned ``energies``.

    Returns:
        :class:`StickSpectrum`
    '''
    kind = str(kind).lower()
    if kind not in ('absorption', 'emission'):
        raise ValueError("kind must be 'absorption' or 'emission', got %r" % kind)

    de = numpy.asarray(result.energies, dtype=float)
    fcf = numpy.asarray(result.fcf, dtype=float)
    pops = numpy.asarray(getattr(result, 'populations', numpy.ones_like(de)), dtype=float)
    states = numpy.asarray(getattr(result, 'states', numpy.zeros((de.size, 0))), dtype=numpy.int16)
    init = numpy.asarray(getattr(result, 'init_states', numpy.zeros((de.size, 0))),
                         dtype=numpy.int16)
    if temperature is None:
        temperature = float(getattr(result, 'temperature', 0.0))

    if kind == 'absorption':
        e_photon = de.copy()
    else:
        e_photon = -de

    bad = e_photon < 0.0
    if numpy.any(bad):
        worst = float(e_photon[bad].min())
        raise ValueError(
            "kind=%r gives %d negative photon energies (most negative %.6e Eh).  "
            "pyscf.vibronic stores the SIGNED Delta E = E_final - E_initial.  For "
            "absorption the initial electronic state must be the lower one "
            "(Delta E > 0); for emission the caller must supply the EXCITED state "
            "as the initial state so that Delta E < 0.  It looks like the two "
            "states were passed in the wrong order." % (kind, int(bad.sum()), worst))

    mu2 = _dipole_strength(transition_dipole)
    inten = mu2 * pops * fcf
    if not line_strength:
        if kind == 'absorption':
            inten = inten * e_photon
        else:
            inten = inten * e_photon ** 3

    if merge_tol is not None:
        e_photon, inten, states, init = _merge_lines(e_photon, inten, states, init,
                                                     float(merge_tol))

    if intensity_threshold and intensity_threshold > 0.0:
        keep = inten >= float(intensity_threshold)
        e_photon = e_photon[keep]
        inten = inten[keep]
        states = states[keep]
        init = init[keep]

    order = numpy.argsort(e_photon, kind='stable')
    sticks = StickSpectrum(e_photon[order], inten[order], assignments=states[order],
                           kind=kind, temperature=temperature, unit='au',
                           init_assignments=init[order], line_strength=line_strength)
    if unit != 'au':
        sticks = sticks.to_unit(unit)
    return sticks


def _merge_lines(energies, intensities, states, init, tol):
    '''Merge lines closer than ``tol`` in energy, conserving total intensity.

    Deterministic: sticks are sorted by energy, then greedily grouped while the
    gap to the previous member is below ``tol``.  The merged energy is the
    intensity-weighted mean (or the plain mean when the group has zero total
    intensity), and the retained assignment is that of the most intense member
    (ties broken by the lowest original index).
    '''
    if tol <= 0.0 or energies.size == 0:
        return energies, intensities, states, init
    order = numpy.argsort(energies, kind='stable')
    e = energies[order]
    w = intensities[order]
    group = numpy.zeros(e.size, dtype=numpy.int64)
    g = 0
    for n in range(1, e.size):
        if e[n] - e[n - 1] > tol:
            g += 1
        group[n] = g
    ngroup = g + 1
    out_e = numpy.empty(ngroup)
    out_w = numpy.empty(ngroup)
    idx = numpy.empty(ngroup, dtype=numpy.int64)
    for gi in range(ngroup):
        sel = numpy.where(group == gi)[0]
        tot = float(w[sel].sum())
        out_w[gi] = tot
        out_e[gi] = float((e[sel] * w[sel]).sum() / tot) if tot != 0.0 else float(e[sel].mean())
        idx[gi] = order[sel[int(numpy.argmax(w[sel]))]]
    return out_e, out_w, states[idx], init[idx]


def broaden(energies, intensities, profile='gaussian', width=300.0, unit='cm-1',
            grid=None, npoints=2000, padding=None, cutoff=None):
    '''Convolute a stick spectrum with a normalised lineshape.

    Args:
        energies : (nline,) stick positions in **Hartree** (the internal unit).
        intensities : (nline,) stick intensities.
        profile : 'gaussian' or 'lorentzian'
        width : float
            **FWHM**, expressed in ``unit``.
        unit : str
            Unit of ``width``, ``grid``, ``padding``, ``cutoff`` and of the
            returned grid.  Accepted values are those of
            :data:`pyscf.vibronic.units.ENERGY_UNITS`.
        grid : (npoints,) array or None
            Explicit grid, in ``unit``.  Must be sorted ascending.  When
            ``None`` a uniform grid spanning the sticks plus ``padding`` on
            both sides is built with ``npoints`` points.
        npoints : int
        padding : float or None
            Extra span on each side, in ``unit``.  Defaults to ``5 * width``.
        cutoff : float or None
            Truncation radius in units of the FWHM.  Defaults to
            :data:`DEFAULT_GAUSSIAN_CUTOFF` for the Gaussian and to ``None``
            (no truncation) for the Lorentzian, whose ``1/x^2`` tails make any
            modest cutoff lose a significant fraction of the area.

    Returns:
        ``(grid, signal)``, both ``(npoints,)`` arrays.  ``grid`` is in
        ``unit`` and ``signal`` is intensity per unit of ``unit``, so that
        ``trapezoid(signal, grid)`` reproduces ``sum(intensities)``.

    A :class:`BroadenedSpectrum` wrapper is returned by
    :meth:`StickSpectrum.broaden`.
    '''
    profile = str(profile).lower()
    if profile not in _PROFILE_FUNCS:
        raise ValueError('profile must be one of %s, got %r' % (PROFILES, profile))
    width = float(width)
    if width <= 0.0:
        raise ValueError('width (FWHM) must be positive, got %r' % width)

    e = numpy.asarray(energies, dtype=float).ravel()
    w = numpy.asarray(intensities, dtype=float).ravel()
    if e.shape != w.shape:
        raise ValueError('energies and intensities must have the same shape')
    x0 = numpy.asarray(units.convert_energy_from_au(e, unit), dtype=float).ravel()

    if grid is None:
        if padding is None:
            padding = 5.0 * width
        if x0.size == 0:
            lo, hi = -padding, padding
        else:
            lo = x0.min() - padding
            hi = x0.max() + padding
        npoints = int(npoints)
        if npoints < 2:
            raise ValueError('npoints must be at least 2')
        if hi <= lo:
            hi = lo + 1.0
        grid = numpy.linspace(lo, hi, npoints)
    else:
        grid = numpy.asarray(grid, dtype=float).ravel()
        if grid.size < 2:
            raise ValueError('grid must have at least 2 points')
        if numpy.any(numpy.diff(grid) <= 0.0):
            raise ValueError('grid must be strictly increasing')

    if cutoff is None:
        cutoff = DEFAULT_GAUSSIAN_CUTOFF if profile == 'gaussian' else None

    func = _PROFILE_FUNCS[profile]
    signal = numpy.zeros_like(grid)
    if x0.size == 0:
        return grid, signal

    if cutoff is None:
        # Dense evaluation, chunked to bound the temporary array size.
        chunk = max(1, int(4e6 // max(grid.size, 1)))
        for start in range(0, x0.size, chunk):
            xs = x0[start:start + chunk]
            ws = w[start:start + chunk]
            signal += ws.dot(func(grid[None, :], xs[:, None], width))
    else:
        radius = float(cutoff) * width
        lo_idx = numpy.searchsorted(grid, x0 - radius, side='left')
        hi_idx = numpy.searchsorted(grid, x0 + radius, side='right')
        for n in range(x0.size):
            a, b = lo_idx[n], hi_idx[n]
            if b > a:
                signal[a:b] += w[n] * func(grid[a:b], x0[n], width)
    return grid, signal
