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
Harmonic normal-mode analysis in **true atomic units**, for
:mod:`pyscf.vibronic`.

This is deliberately a separate implementation from
:func:`pyscf.hessian.thermo.harmonic_analysis`, which mass-weights the Hessian
with masses in **amu** and therefore produces a ``freq_au`` that is not an
atomic-unit angular frequency.  Franck-Condon theory needs :math:`\\hbar\\omega`
as an energy in Hartree and needs the *mass-weighted* eigenvector matrix
:math:`L` (which ``thermo`` discards), so here

* masses are converted to **electron masses** (``units.AMU2AU``),
* :math:`\\tilde H_{ij} = H_{ij}/\\sqrt{m_i m_j}` in Eh/(bohr^2 m_e),
* :math:`\\lambda = \\omega^2` in Eh^2 (:math:`\\hbar = 1`),
* :math:`\\omega` in Eh, so ``units.au2wavenumber(omega)`` is cm^-1,
* ``modes`` is the (3N, nvib) matrix :math:`L` with orthonormal columns.

The wavenumbers produced here are numerically identical to ``thermo``'s (the
amu -> m_e rescaling cancels exactly against ``thermo``'s ``au2hz`` factor);
this is asserted in ``test_normal_modes.py`` and is the primary validation of
the whole mass-weighting/projection chain.

Sign (phase) convention
-----------------------
Eigenvectors are defined only up to a sign.  With :data:`FIX_MODE_PHASE` the
sign of each column of ``modes`` is fixed deterministically (see
:func:`fix_mode_phase`).  This is a **convention only**: every physical result
must be invariant to it.  Duschinsky matrices, in particular, change sign
row-/column-wise under a phase change while all observables do not.
'''

import sys
import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf.vibronic import units
from pyscf.vibronic import alignment

__all__ = [
    'HarmonicModel', 'harmonic_analysis', 'harmonic_model',
    'mass_weighted_hessian', 'translation_rotation_vectors',
    'projector_trans_rot', 'project_hessian', 'fix_mode_phase',
    'reshape_hessian',
    'FIX_MODE_PHASE', 'LINDEP_THRESHOLD', 'FREQ_ZERO_TOL',
    'HESSIAN_ASYMMETRY_TOL', 'IMAGINARY_POLICIES',
]

#: Fix the sign of each normal mode deterministically.  A convention only.
FIX_MODE_PHASE = True

#: Eigenvalue threshold used when building the vibrational subspace.
LINDEP_THRESHOLD = 1e-7

#: Retained frequencies below this value (a.u.) trigger a warning: they signal
#: an incomplete translation/rotation projection or a non-stationary geometry.
#: 10 cm^-1 expressed in Hartree.
FREQ_ZERO_TOL = float(units.wavenumber2au(10.0))

#: Hessian asymmetry (Eh/bohr^2) above which a warning is emitted.  Numerical
#: Hessians are mildly asymmetric; that is expected and tolerated.
HESSIAN_ASYMMETRY_TOL = 1e-5

#: Accepted values of ``imaginary_policy``.
IMAGINARY_POLICIES = ('raise', 'warn', 'ignore')


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def reshape_hessian(hessian, natm):
    '''Accept a Hessian in either PySCF ``(natm,natm,3,3)`` layout or flat
    ``(3N,3N)`` layout and return the flat ``(3N,3N)`` form.

    The PySCF convention is ``hess[a,b,x,y] = d2E/dR_{ax} dR_{by}``, which flattens
    as ``transpose(0,2,1,3).reshape(3N,3N)``.

    Raises:
        ValueError : on any other shape or on non-finite entries.
    '''
    h = numpy.asarray(hessian, dtype=float)
    n3 = 3 * natm
    if h.ndim == 4:
        if h.shape != (natm, natm, 3, 3):
            raise ValueError('4-index hessian must have shape (%d,%d,3,3), got %s'
                             % (natm, natm, h.shape))
        h = h.transpose(0, 2, 1, 3).reshape(n3, n3)
    elif h.ndim == 2:
        if h.shape != (n3, n3):
            raise ValueError('2-index hessian must have shape (%d,%d), got %s'
                             % (n3, n3, h.shape))
        h = h.copy()
    else:
        raise ValueError('hessian must be (natm,natm,3,3) or (3N,3N), got shape %s'
                         % (h.shape,))
    if not numpy.all(numpy.isfinite(h)):
        raise ValueError('hessian contains non-finite values (NaN or inf)')
    return h


def mass_weighted_hessian(mass_au, hessian_3n3n):
    '''Mass-weight a Cartesian Hessian.

    .. math:: \\tilde H_{ij} = H_{ij} / \\sqrt{m_i m_j}

    Args:
        mass_au : (natm,) masses in **electron masses**.
        hessian_3n3n : (3N,3N) Cartesian Hessian in Eh/bohr^2.

    Returns:
        (3N,3N) array in Eh/(bohr^2 m_e).
    '''
    mass_au = numpy.asarray(mass_au, dtype=float).reshape(-1)
    h = numpy.asarray(hessian_3n3n, dtype=float)
    n3 = 3 * mass_au.shape[0]
    if h.shape != (n3, n3):
        raise ValueError('hessian must have shape (%d,%d), got %s' % (n3, n3, h.shape))
    if numpy.any(mass_au <= 0):
        raise ValueError('all masses must be strictly positive')
    invsqrtm = numpy.repeat(mass_au**-.5, 3)
    return h * invsqrtm[:, None] * invsqrtm[None, :]


def translation_rotation_vectors(mass_au, coords):
    '''Mass-weighted translation and rotation vectors.

    The atomic-unit analogue of :func:`pyscf.hessian.thermo._get_TR`, returning
    an array instead of a tuple.

    Rows 0-2 are the three translations :math:`\\sqrt{m_a}\\,\\mathbf{e}_x`,
    :math:`\\sqrt{m_a}\\,\\mathbf{e}_y`, :math:`\\sqrt{m_a}\\,\\mathbf{e}_z`.
    Rows 3-5 are the three infinitesimal rotations about the *principal* axes,
    ordered so that the last one (row 5) is the rotation about the axis with the
    **smallest** moment of inertia.  For a linear molecule that row vanishes
    identically, which is why rows 3-5 must be truncated to rows 3-4 in the
    linear case (see :func:`projector_trans_rot`).

    The rows are neither normalised nor mutually orthogonal; orthonormalisation
    is the caller's job.

    Args:
        mass_au : (natm,) masses in electron masses.
        coords : (natm,3) coordinates in bohr.  Centred internally.

    Returns:
        (6, 3N) array.
    '''
    mass_au, coords = alignment._check_mass_coords(mass_au, coords)
    coords = coords - alignment.center_of_mass(mass_au, coords)
    massp = mass_au**.5
    natm = coords.shape[0]

    tr = numpy.zeros((6, natm, 3))
    for x in range(3):
        tr[x, :, x] = massp

    im = alignment.inertia_tensor(mass_au, coords)
    w, paxes = numpy.linalg.eigh(im)
    # order the principal axes so that the *last* one has the smallest moment
    paxes = paxes[:, ::-1]
    ex, ey, ez = paxes.T

    c = coords.dot(paxes)
    cx, cy, cz = c.T
    tr[3] = massp[:, None] * (cy[:, None] * ez - cz[:, None] * ey)
    tr[4] = massp[:, None] * (cz[:, None] * ex - cx[:, None] * ez)
    tr[5] = massp[:, None] * (cx[:, None] * ey - cy[:, None] * ex)
    return tr.reshape(6, natm * 3)


def _expected_nvib(rotor_type, natm):
    if rotor_type == 'ATOM':
        return 0, 3
    elif rotor_type == 'LINEAR':
        return 3 * natm - 5, 5
    elif rotor_type == 'REGULAR':
        return 3 * natm - 6, 6
    raise ValueError('unknown rotor_type %r, expected one of %s'
                     % (rotor_type, alignment.ROTOR_TYPES))


def projector_trans_rot(mass_au, coords, rotor_type=None, lindep_tol=LINDEP_THRESHOLD):
    '''Projector onto the vibrational subspace, and an orthonormal basis for it.

    The retained translation/rotation vectors are

    ========== ================== ==================
    rotor_type retained TR vectors ``nvib``
    ========== ================== ==================
    ATOM       3 (translations)   0
    LINEAR     5                  ``3N-5``
    REGULAR    6                  ``3N-6``
    ========== ================== ==================

    They are orthonormalised by QR; the projector is
    :math:`P = 1 - QQ^{\\mathsf{T}}`, and its eigenvectors with eigenvalue
    :math:`> \\texttt{lindep\\_tol}` (i.e. eigenvalue 1) form the returned basis.
    This is the construction used by :mod:`pyscf.hessian.thermo`, except that
    the basis itself is returned rather than being consumed internally.

    Args:
        mass_au : (natm,) masses in electron masses.
        coords : (natm,3) bohr.
        rotor_type : ``'ATOM'``/``'LINEAR'``/``'REGULAR'``.  Determined with
            :func:`pyscf.vibronic.alignment.classify_rotor` if ``None``.
        lindep_tol : eigenvalue threshold for the vibrational subspace.

    Returns:
        ``(P, basis)`` with ``P`` (3N,3N) the projector and ``basis``
        (3N, nvib) an orthonormal basis of the vibrational subspace.

    Raises:
        RuntimeError : if the dimension of the recovered subspace does not match
            the ``3N-6`` / ``3N-5`` / ``0`` expected from ``rotor_type``.  That
            indicates linearly dependent translation/rotation vectors, e.g. a
            near-linear structure straddling the classification tolerance.
    '''
    mass_au, coords = alignment._check_mass_coords(mass_au, coords)
    natm = coords.shape[0]
    n3 = 3 * natm
    if rotor_type is None:
        rotor_type = alignment.classify_rotor(mass_au, coords)
    nvib_expected, ntr = _expected_nvib(rotor_type, natm)
    if nvib_expected < 0:
        raise RuntimeError(
            'rotor_type %r with natm=%d implies nvib=%d < 0; the structure is '
            'inconsistent with its rotor classification' % (rotor_type, natm, nvib_expected))

    tr = translation_rotation_vectors(mass_au, coords)[:ntr]
    q, r = numpy.linalg.qr(tr.T)
    rdiag = abs(numpy.diag(r))
    if rdiag.size and rdiag.min() <= lindep_tol * max(rdiag.max(), 1e-300):
        raise RuntimeError(
            'the %d retained translation/rotation vectors for rotor_type=%r are '
            'linearly dependent (QR diagonal %s).  This usually means the structure '
            'is near-linear but was classified as REGULAR (or vice versa); check '
            'alignment.classify_rotor and its lin_tol.'
            % (ntr, rotor_type, numpy.array2string(rdiag, precision=3)))

    proj = numpy.eye(n3) - q.dot(q.T)
    w, v = numpy.linalg.eigh(proj)
    basis = v[:, w > lindep_tol]
    if basis.shape[1] != nvib_expected:
        raise RuntimeError(
            'vibrational subspace has dimension %d but %d (= 3*%d - %d) was expected '
            'for rotor_type=%r.  Projector eigenvalues: %s.  lindep_tol=%g.'
            % (basis.shape[1], nvib_expected, natm, ntr, rotor_type,
               numpy.array2string(numpy.sort(w)[::-1], precision=6, threshold=20),
               lindep_tol))
    return proj, basis


def project_hessian(hess_mw, basis):
    '''Project a mass-weighted Hessian onto the vibrational subspace.

    Returns:
        ``basis.T @ hess_mw @ basis``, shape (nvib, nvib).
    '''
    hess_mw = numpy.asarray(hess_mw, dtype=float)
    basis = numpy.asarray(basis, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != hess_mw.shape[0]:
        raise ValueError('basis shape %s is incompatible with hessian shape %s'
                         % (basis.shape, hess_mw.shape))
    return lib.einsum('pi,pq,qj->ij', basis, hess_mw, basis)


def fix_mode_phase(modes, rtol=1e-8):
    '''Fix the sign of each column of ``modes`` deterministically, in place.

    For every column the element of largest absolute value is located; among
    elements within a *relative* tolerance ``rtol`` of that maximum, the one with
    the **lowest index** is chosen, and the column is negated if necessary to
    make that element positive.  If the largest element is numerically zero the
    column is left untouched.

    The tie-breaking rule matters for symmetric molecules, where several
    components of a mode are equal by symmetry: without it the chosen pivot -
    and hence the sign - would depend on floating-point noise and could differ
    between runs, platforms or BLAS versions.

    This is a **convention only**; all physical results must be invariant to it.

    Returns:
        the same array, modified in place.
    '''
    modes = numpy.asarray(modes)
    if modes.ndim != 2:
        raise ValueError('modes must be 2-dimensional, got shape %s' % (modes.shape,))
    for k in range(modes.shape[1]):
        col = modes[:, k]
        amax = abs(col).max() if col.size else 0.
        if amax <= 0:
            continue
        # lowest index among the near-ties
        idx = int(numpy.argmax(abs(col) >= amax * (1 - rtol)))
        if col[idx] < 0:
            modes[:, k] = -col
    return modes


def harmonic_analysis(mass_au, coords, hessian, rotor_type=None,
                      lindep_tol=LINDEP_THRESHOLD, fix_phase=None):
    '''Low-level harmonic analysis, everything in atomic units.

    Args:
        mass_au : (natm,) masses in **electron masses**.
        coords : (natm,3) bohr.  Centred internally; not returned.
        hessian : (3N,3N) or (natm,natm,3,3), Eh/bohr^2.  Symmetrised
            internally.
        rotor_type : optional override, see :func:`projector_trans_rot`.
        lindep_tol : eigenvalue threshold for the vibrational subspace.
        fix_phase : bool or ``None`` (use :data:`FIX_MODE_PHASE`).

    Returns:
        dict with keys

        ``freq``
            (nvib,) signed angular frequency in a.u.:
            :math:`{\\rm sign}(\\lambda)\\sqrt{|\\lambda|}`, so an imaginary mode
            appears as a *negative* number.  Sorted ascending (stable sort), so
            imaginary modes come first.
        ``force_const``
            (nvib,) eigenvalues :math:`\\lambda = \\omega^2` in Eh^2, same order.
        ``modes``
            (3N, nvib) mass-weighted eigenvectors :math:`L`, columns orthonormal.
        ``imaginary``
            (nvib,) bool mask, ``True`` where :math:`\\lambda < 0`.
        ``rotor_type``, ``nvib``, ``basis``, ``hessian_asymmetry``.
    '''
    mass_au, coords = alignment._check_mass_coords(mass_au, coords)
    natm = coords.shape[0]
    h = reshape_hessian(hessian, natm)
    asymmetry = float(abs(h - h.T).max()) if h.size else 0.
    h = .5 * (h + h.T)

    coords = coords - alignment.center_of_mass(mass_au, coords)
    if rotor_type is None:
        rotor_type = alignment.classify_rotor(mass_au, coords)
    hmw = mass_weighted_hessian(mass_au, h)
    _, basis = projector_trans_rot(mass_au, coords, rotor_type, lindep_tol)
    nvib = basis.shape[1]

    if nvib == 0:
        force_const = numpy.zeros(0)
        modes = numpy.zeros((3 * natm, 0))
    else:
        hproj = project_hessian(hmw, basis)
        hproj = .5 * (hproj + hproj.T)
        force_const, vec = numpy.linalg.eigh(hproj)
        modes = basis.dot(vec)

    # signed frequency: -sqrt(|lam|) for lam < 0, so that a stable ascending
    # sort puts the imaginary modes first, most negative first.
    freq = numpy.sign(force_const) * numpy.sqrt(abs(force_const))
    order = numpy.argsort(freq, kind='stable')
    freq = freq[order]
    force_const = force_const[order]
    modes = modes[:, order]

    if fix_phase is None:
        fix_phase = FIX_MODE_PHASE
    if fix_phase:
        fix_mode_phase(modes)

    return {
        'freq': freq,
        'force_const': force_const,
        'modes': modes,
        'imaginary': force_const < 0,
        'rotor_type': rotor_type,
        'nvib': nvib,
        'basis': basis,
        'hessian_asymmetry': asymmetry,
    }


# ---------------------------------------------------------------------------
# HarmonicModel
# ---------------------------------------------------------------------------

class HarmonicModel(lib.StreamObject):
    '''Harmonic model of one electronic state: geometry, masses, Hessian and
    the resulting normal modes, all in atomic units.

    .. warning::

        **The** ``mass`` **argument is in amu by default**, because that is what
        :meth:`pyscf.gto.Mole.atom_mass_list` returns and what users have.  It is
        converted internally to electron masses and the attribute
        ``self.mass`` is in **electron masses**.  Pass ``mass_unit='au'`` if your
        masses are already in electron masses.  Everything else (``coords`` in
        bohr, ``hessian`` in Eh/bohr^2, ``energy`` in Eh) is in atomic units on
        both input and output.

    Args:
        atom_charges : (natm,) nuclear charges (int).
        coords : (natm,3) bohr.  Stored **centred at the centre of mass**.
        mass : (natm,) atomic masses; see ``mass_unit``.
        hessian : (natm,natm,3,3) or (3N,3N) Cartesian Hessian, Eh/bohr^2.

    Kwargs:
        energy : float or None.  Electronic energy at this geometry, Eh.
        mass_unit : ``'amu'`` (default) or ``'au'`` (electron masses).
        imaginary_policy : ``'raise'`` (default), ``'warn'`` or ``'ignore'``.
            See :attr:`imaginary_policy`.
        rotor_type : optional override of the automatic classification.
        fix_phase : bool or None; ``None`` uses :data:`FIX_MODE_PHASE`.
        lindep_tol : eigenvalue threshold for the vibrational subspace.
        mol : optional :class:`pyscf.gto.Mole` this model came from.
        verbose, stdout : usual PySCF logging controls.

    Attributes:
        mol : :class:`pyscf.gto.Mole` or None.
        atom_charges : (natm,) int.
        mass : (natm,) **electron masses**.
        coords : (natm,3) bohr, centred at the centre of mass.
        hessian : (3N,3N) Eh/bohr^2, symmetrised.
        energy : float Eh or None.
        freq : (nvib,) angular frequency in a.u. (Eh, hbar=1), sorted ascending.
            An imaginary mode is stored as the **negative** number
            :math:`-\\sqrt{|\\lambda|}`.
        force_const : (nvib,) :math:`\\lambda = \\omega^2` in Eh^2.
        modes : (3N, nvib) mass-weighted :math:`L`, columns orthonormal.
        rotor_type : ``'ATOM'`` | ``'LINEAR'`` | ``'REGULAR'``.
        nvib : int.
        imaginary : (nvib,) bool mask.
        zpe : float, :math:`\\frac12\\sum_k \\omega_k` over the **real** modes only.
        hessian_asymmetry : float, ``abs(H - H.T).max()`` before symmetrisation.

    Examples:

    >>> from pyscf import gto, scf, hessian
    >>> from pyscf.vibronic.normal_modes import HarmonicModel
    >>> mol = gto.M(atom='O 0 0 0.12; H 0 0.75 -0.48; H 0 -0.75 -0.48', basis='sto-3g')
    >>> mf = scf.RHF(mol).run()
    >>> h = mf.Hessian().kernel()
    >>> model = HarmonicModel.from_mole(mol, h, energy=mf.e_tot)
    >>> model.freq_wavenumber
    '''

    #: what to do about imaginary frequencies; see :data:`IMAGINARY_POLICIES`
    imaginary_policy = 'raise'

    def __init__(self, atom_charges, coords, mass, hessian, energy=None,
                 mass_unit='amu', imaginary_policy=None, rotor_type=None,
                 fix_phase=None, lindep_tol=LINDEP_THRESHOLD, mol=None,
                 verbose=None, stdout=None):
        self.mol = mol
        self.stdout = sys.stdout if stdout is None else stdout
        if verbose is None:
            verbose = getattr(mol, 'verbose', lib.logger.NOTE)
        self.verbose = verbose
        if imaginary_policy is not None:
            self.imaginary_policy = imaginary_policy
        self.rotor_type = rotor_type
        self.fix_phase = fix_phase
        self.lindep_tol = lindep_tol
        self.energy = None if energy is None else float(energy)

        self.atom_charges = numpy.asarray(atom_charges, dtype=int).reshape(-1)
        coords = numpy.asarray(coords, dtype=float)
        mass = numpy.asarray(mass, dtype=float).reshape(-1)
        self._mass_input_unit = mass_unit
        self.mass = self._convert_mass(mass, mass_unit)
        self.coords = coords
        self._hessian_input = hessian

        # results, filled by build()
        self.hessian = None
        self.freq = None
        self.force_const = None
        self.modes = None
        self.nvib = None
        self.imaginary = None
        self.zpe = None
        self.hessian_asymmetry = None
        self._basis = None

        self.build()

    @staticmethod
    def _convert_mass(mass, mass_unit):
        key = str(mass_unit).strip().lower()
        if key == 'amu':
            return numpy.asarray(units.amu2au(mass), dtype=float)
        elif key in ('au', 'a.u.', 'me', 'electron_mass'):
            return numpy.array(mass, dtype=float)
        raise ValueError("mass_unit must be 'amu' or 'au', got %r" % (mass_unit,))

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_mole(cls, mol, hessian, energy=None, mass=None, isotope_avg=True, **kw):
        '''Build a :class:`HarmonicModel` from a :class:`pyscf.gto.Mole`.

        Args:
            mol : :class:`pyscf.gto.Mole`.
            hessian : the Hessian in either the ``(natm,natm,3,3)`` layout
                returned by ``mf.Hessian().kernel()`` or the flat ``(3N,3N)``
                layout; the shape is detected automatically.  Eh/bohr^2.

        Kwargs:
            energy : float, electronic energy in Eh (e.g. ``mf.e_tot``).
            mass : (natm,) masses in **amu**, overriding
                ``mol.atom_mass_list(isotope_avg)``.  This is how isotopic
                substitution is requested, e.g. ``mass=[15.9949, 2.0141,
                2.0141]`` for D2O.  Use ``mass_unit='au'`` to give them in
                electron masses instead.
            isotope_avg : bool, passed to ``mol.atom_mass_list`` when ``mass``
                is ``None``.  ``True`` (default) gives the natural-abundance
                average mass, ``False`` the most abundant isotope.

        Any other keyword is forwarded to :class:`HarmonicModel`.
        '''
        if mass is None:
            mass = mol.atom_mass_list(isotope_avg=isotope_avg)
        kw.setdefault('verbose', mol.verbose)
        kw.setdefault('stdout', mol.stdout)
        return cls(mol.atom_charges(), mol.atom_coords(), mass, hessian,
                   energy=energy, mol=mol, **kw)

    # -- the analysis -------------------------------------------------------

    def build(self):
        '''Validate the input, centre the geometry, and run the harmonic
        analysis.  Called automatically by ``__init__``; call it again after
        changing an attribute.  Returns ``self``.
        '''
        log = logger.new_logger(self, self.verbose)

        natm = self.atom_charges.shape[0]
        if natm < 1:
            raise ValueError('at least one atom is required, got natm=%d' % natm)
        if self.coords.ndim != 2 or self.coords.shape != (natm, 3):
            raise ValueError('coords must have shape (%d,3), got %s'
                             % (natm, (self.coords.shape,)))
        if self.mass.shape[0] != natm:
            raise ValueError('mass has %d entries but there are %d atoms'
                             % (self.mass.shape[0], natm))
        if not numpy.all(numpy.isfinite(self.mass)):
            raise ValueError('mass contains non-finite values')
        if numpy.any(self.mass <= 0):
            raise ValueError('all masses must be strictly positive; got %s (electron masses)'
                             % (self.mass,))
        if not numpy.all(numpy.isfinite(self.coords)):
            raise ValueError('coords contains non-finite values')
        if self.imaginary_policy not in IMAGINARY_POLICIES:
            raise ValueError('imaginary_policy must be one of %s, got %r'
                             % (IMAGINARY_POLICIES, self.imaginary_policy))

        hess = reshape_hessian(self._hessian_input, natm)
        self.hessian_asymmetry = float(abs(hess - hess.T).max()) if hess.size else 0.
        if self.hessian_asymmetry > HESSIAN_ASYMMETRY_TOL:
            log.warn('Cartesian Hessian is asymmetric: max|H - H^T| = %.3e Eh/bohr^2 '
                     '(> %.1e).  It has been symmetrised as (H + H^T)/2, but such a '
                     'large asymmetry suggests a poorly converged numerical Hessian.',
                     self.hessian_asymmetry, HESSIAN_ASYMMETRY_TOL)
        self.hessian = .5 * (hess + hess.T)

        # Eckart translational condition
        self.coords = alignment.shift_to_center_of_mass(self.mass, self.coords)

        res = harmonic_analysis(self.mass, self.coords, self.hessian,
                                rotor_type=self.rotor_type,
                                lindep_tol=self.lindep_tol,
                                fix_phase=self.fix_phase)
        self.rotor_type = res['rotor_type']
        self.nvib = res['nvib']
        self.freq = res['freq']
        self.force_const = res['force_const']
        self.modes = res['modes']
        self.imaginary = res['imaginary']
        self._basis = res['basis']

        self._check_imaginary(log)
        self._check_near_zero(log)

        # ZPE excludes the imaginary modes.  A ZPE computed at a structure with
        # imaginary frequencies is not a physically meaningful quantity: the
        # structure is not a minimum and the harmonic partition function does
        # not exist along the unstable coordinate(s).
        self.zpe = .5 * float(self.freq[~self.imaginary].sum())
        return self

    kernel = build

    def _check_imaginary(self, log):
        nimag = int(self.imaginary.sum())
        if nimag == 0:
            return
        wn = units.au2wavenumber(abs(self.freq[self.imaginary]))
        listing = ', '.join('%.2fi' % x for x in numpy.atleast_1d(wn))
        if self.imaginary_policy == 'raise':
            raise RuntimeError(
                '%d imaginary frequency/frequencies found (cm^-1): %s.  The geometry '
                'is not a minimum of this electronic state, so a harmonic '
                'Franck-Condon treatment is not valid.  Re-optimise the geometry, or '
                "pass imaginary_policy='warn' to proceed anyway (the modes are kept "
                'and flagged in .imaginary; they are never dropped).' % (nimag, listing))
        elif self.imaginary_policy == 'warn':
            log.warn('%d imaginary frequency/frequencies (cm^-1): %s.  Kept and flagged '
                     'in .imaginary; excluded from the ZPE.  Results derived from this '
                     'model are not physically meaningful.', nimag, listing)

    def _check_near_zero(self, log):
        small = (~self.imaginary) & (self.freq < FREQ_ZERO_TOL)
        if not small.any():
            return
        wn = units.au2wavenumber(self.freq[small])
        log.warn('RuntimeWarning: %d retained vibrational frequency/frequencies below '
                 '%.2f cm^-1: %s cm^-1.  This indicates an incomplete '
                 'translation/rotation projection or a geometry that is not a '
                 'stationary point.',
                 int(small.sum()), units.au2wavenumber(FREQ_ZERO_TOL),
                 numpy.array2string(numpy.atleast_1d(wn), precision=3))

    # -- convenience --------------------------------------------------------

    @property
    def natm(self):
        '''Number of atoms.'''
        return self.atom_charges.shape[0]

    @property
    def n3(self):
        '''Number of Cartesian degrees of freedom, ``3*natm``.'''
        return 3 * self.natm

    @property
    def mass_amu(self):
        '''(natm,) atomic masses in amu (``self.mass`` is in electron masses).'''
        return units.au2amu(self.mass)

    @property
    def freq_wavenumber(self):
        '''(nvib,) frequencies in cm^-1.  Imaginary modes appear as negative
        numbers, matching the sign convention of :attr:`freq`.
        '''
        return units.au2wavenumber(self.freq)

    @property
    def cartesian_modes(self):
        '''(nvib, natm, 3) un-mass-weighted Cartesian displacements,
        :math:`m_a^{-1/2} L_{a x, k}`, with the same ``(mode, atom, xyz)``
        index order as ``thermo.harmonic_analysis()['norm_mode']``.

        Because the mass-weighting here uses electron masses while ``thermo``
        uses amu, this array equals ``thermo``'s ``norm_mode`` divided by
        ``sqrt(units.AMU2AU)`` (up to the arbitrary per-mode sign).  The
        directions are identical; only the normalisation differs, and
        :attr:`reduced_mass` -- which is the normalisation-independent physical
        quantity -- is reported in amu and agrees with ``thermo`` exactly.
        '''
        if self.nvib == 0:
            return numpy.zeros((0, self.natm, 3))
        modes = self.modes.reshape(self.natm, 3, self.nvib)
        return numpy.einsum('a,axk->kax', self.mass**-.5, modes)

    @property
    def reduced_mass(self):
        '''(nvib,) reduced masses in **amu**,
        :math:`\\mu_k = 1/\\sum_{a} |m_a^{-1/2} L_{ak}|^2`, the same definition
        (and the same numbers) as ``thermo.harmonic_analysis()['reduced_mass']``.
        '''
        if self.nvib == 0:
            return numpy.zeros(0)
        cm = self.cartesian_modes
        mu_au = 1. / numpy.einsum('kax,kax->k', cm, cm)
        return units.au2amu(mu_au)

    def dump_normal_modes(self, verbose=None):
        '''Print the normal modes with :mod:`pyscf.lib.logger`.'''
        log = logger.new_logger(self, verbose)
        log.note('Harmonic analysis: rotor_type = %s, natm = %d, nvib = %d',
                 self.rotor_type, self.natm, self.nvib)
        if self.energy is not None:
            log.note('Electronic energy = %.10f Eh', self.energy)
        log.note('ZPE = %.10f Eh = %.3f cm^-1  (%d imaginary mode(s) excluded)',
                 self.zpe, units.au2wavenumber(self.zpe), int(self.imaginary.sum()))
        if self.hessian_asymmetry is not None:
            log.debug('max|H - H^T| before symmetrisation = %.3e Eh/bohr^2',
                      self.hessian_asymmetry)
        if self.nvib == 0:
            log.note('No vibrational degrees of freedom.')
            return self

        symbols = [_symbol(z) for z in self.atom_charges]
        cart = self.cartesian_modes
        wn = self.freq_wavenumber
        mu = self.reduced_mass
        for k in range(self.nvib):
            tag = 'i' if self.imaginary[k] else ''
            log.note('mode %d   freq = %10.3f%s cm^-1   omega = %.8f Eh   '
                     'reduced mass = %8.4f amu', k, abs(wn[k]), tag or ' ',
                     abs(self.freq[k]), mu[k])
            log.note('         %-4s %12s %12s %12s', 'atom', 'x', 'y', 'z')
            for a in range(self.natm):
                log.note('         %-4s %12.6f %12.6f %12.6f',
                         symbols[a], cart[k, a, 0], cart[k, a, 1], cart[k, a, 2])
        return self


def _symbol(charge):
    from pyscf.data import elements
    try:
        return elements.ELEMENTS[int(charge)]
    except (IndexError, ValueError):
        return str(charge)


def harmonic_model(mol, hessian, energy=None, **kw):
    '''Convenience wrapper: ``HarmonicModel.from_mole(mol, hessian, energy, **kw)``.'''
    return HarmonicModel.from_mole(mol, hessian, energy=energy, **kw)
