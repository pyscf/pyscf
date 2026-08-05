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
The Duschinsky transformation [Duschinsky1937]_ between the normal coordinates
of two electronic states.

Normative relation
------------------
With :math:`Q_i` and :math:`Q_f` the mass-weighted normal coordinates of the
initial and final electronic state (units bohr :math:`\\sqrt{m_e}`),

.. math::

    Q_f = J\\, Q_i + K, \\qquad
    J = L_f^{\\mathsf{T}} L_i, \\qquad
    K = L_f^{\\mathsf{T}} M^{1/2} (x_{i0} - x_{f0})

* ``J`` has shape ``(nvib_f, nvib_i)``: **rows index FINAL-state modes, columns
  index INITIAL-state modes**.
* ``K`` has shape ``(nvib_f,)`` and units bohr :math:`\\sqrt{m_e}`.

Derivation (following REFERENCES.md sec. 2.3, verified there against
Sharp--Rosenstock 1964, Ruhoff 1994 and Santoro 2007).  Write the mass-weighted
displacement from each state's own equilibrium structure,

.. math::

    s_i = M^{1/2}(x - x_{i0}), \\quad s_f = M^{1/2}(x - x_{f0}), \\quad
    Q_s = L_s^{\\mathsf{T}} s_s

Then identically

.. math::

    Q_f = L_f^{\\mathsf{T}} M^{1/2}(x - x_{i0})
        + L_f^{\\mathsf{T}} M^{1/2}(x_{i0} - x_{f0})

and, for a configuration whose mass-weighted displacement lies in the initial
state's vibrational subspace, :math:`M^{1/2}(x - x_{i0}) = L_i Q_i`, which gives
:math:`J` and :math:`K` above.

Sign of ``K`` (this is the single easiest thing to get wrong)
------------------------------------------------------------
``K`` is built from the **INITIAL minus FINAL** geometry, projected with the
**FINAL**-state mode matrix.  The check that pins this down: put the molecule at
the *initial*-state equilibrium structure, i.e. :math:`Q_i = 0`.  The relation
then reads :math:`Q_f = K`, so

    **K is the position of the initial-state minimum expressed in final-state
    normal coordinates.**

Evaluating :math:`Q_f = L_f^{\\mathsf{T}} M^{1/2}(x_{i0} - x_{f0})` directly at
:math:`x = x_{i0}` reproduces exactly (2.3), so the two statements agree only
for the *initial minus final* ordering.  With the opposite ordering one obtains
:math:`Q_f = J Q_i - K`, which is also found in the literature; both conventions
appear, and what matters is internal consistency with the stated relation.
``test_duschinsky.py::KnownValues::test_pure_displacement_sign`` displaces the
initial-state geometry by a *known positive* amount along final-state mode ``k``
and asserts ``K[k] > 0`` with the analytically known magnitude, so a sign flip
cannot pass the test suite.

Alignment
---------
:math:`x_{i0}` and :math:`x_{f0}` must be expressed in a *common* frame,
otherwise ``K`` is contaminated by rigid-body translation and rotation and ``J``
by spurious mode mixing.  Here both structures are shifted to their own centre of
mass (the Eckart translational condition) and the **final**-state structure is
rotated onto the initial-state structure with the mass-weighted Kabsch rotation
(the linearised Eckart rotational condition), see
:func:`pyscf.vibronic.alignment.align_geometries`.

.. note::

   DESIGN.md sec. 3 fixes the direction as *final rotated onto initial*;
   REFERENCES.md sec. 9 item 7 lists the opposite direction (*initial onto
   final*) as the equally valid alternative.  ``J`` and ``K`` are invariant
   under a common rotation of both structures, so the two choices give identical
   results; this module implements the DESIGN.md direction, and
   ``test_duschinsky.py`` verifies the invariance explicitly.

**The final-state mode matrix is rotated by the same rotation.**  ``L_f`` lives
in mass-weighted Cartesian space, so each of its columns is reshaped to
``(natm, 3)``, right-multiplied by the rotation ``R``, and reshaped back.  Since
the block-diagonal rotation :math:`1_{natm} \\otimes R` is orthogonal, the
rotated ``L_f`` still has orthonormal columns; this is asserted at run time as a
self-check.  Forgetting to rotate the modes silently produces a wrong ``J``.

Restrictions (not "general Duschinsky support" beyond these)
------------------------------------------------------------
* Both states must have the same nuclei, in the same order, with the same
  masses, and the same rotor type -- all enforced with :class:`ValueError`.
* Only **rectilinear** mass-weighted Cartesian normal coordinates are treated.
  For large-amplitude geometry changes the rectilinear model itself breaks down;
  the ``excluded_mode_norm`` and ``orthogonality_error`` diagnostics detect this
  but the module does not fix it (that requires curvilinear coordinates, see
  Baiardi *et al.* 2016).
* Imaginary frequencies in either state are rejected by default
  (``allow_imaginary=False``): a Duschinsky transformation between structures
  that are not both minima has no harmonic Franck--Condon meaning.

References
----------
.. [Duschinsky1937] F. Duschinsky, *Zur Deutung der Elektronenspektren
   mehratomiger Molekuele. I.*, Acta Physicochim. URSS **7**, 551 (1937).
.. [Sharp1964] T. E. Sharp and H. M. Rosenstock, J. Chem. Phys. **41**, 3453
   (1964).
.. [Kuhn1955] H. W. Kuhn, *The Hungarian method for the assignment problem*,
   Naval Res. Logist. Q. **2**, 83 (1955).
'''

import sys
import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf.vibronic import units
from pyscf.vibronic import alignment

__all__ = [
    'Duschinsky', 'duschinsky_transform', 'duschinsky_from_arrays',
    'match_modes', 'ModeMatch', 'ModeMatching',
    'rotate_modes', 'degeneracy_blocks',
    'MASS_RTOL', 'EXCLUDED_NORM_TOL', 'ORTHOGONALITY_TOL',
    'ORTHONORMALITY_ASSERT_TOL', 'DEGENERACY_TOL_CM', 'DEGENERACY_TOL',
]

#: Relative tolerance on a per-atom mass difference between the two states.
#: An isotopic mismatch between two electronic states of the *same* molecule is
#: a user error, not something to be silently averaged over.
MASS_RTOL = 1e-8

#: ``excluded_mode_norm`` above this value triggers a warning: too much of the
#: mass-weighted geometry change is rigid-body translation/rotation rather than
#: vibration, so ``Q_f = J Q_i + K`` no longer holds.
EXCLUDED_NORM_TOL = 1e-3

#: ``orthogonality_error`` above this value triggers a warning (REFERENCES.md
#: sec. 2.5 recommends 1e-3).
ORTHOGONALITY_TOL = 1e-3

#: Tolerance of the run-time self-check that a rotated ``L_f`` still has
#: orthonormal columns.  A block-diagonal rotation is orthogonal, so this can
#: only fail through a coding error or a badly non-orthogonal input ``L_f``.
ORTHONORMALITY_ASSERT_TOL = 1e-9

#: Default degeneracy window for :func:`match_modes`, in cm^-1.
DEGENERACY_TOL_CM = 1.0

#: Default degeneracy window for :func:`match_modes`, in atomic units (Eh).
DEGENERACY_TOL = float(units.wavenumber2au(DEGENERACY_TOL_CM))


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def rotate_modes(modes, rot):
    '''Rotate a mass-weighted Cartesian mode matrix by a 3x3 rotation.

    ``modes`` is ``(3*natm, nvib)``; each column is a mass-weighted Cartesian
    displacement vector and therefore transforms exactly like the coordinates.
    With the row-vector convention of
    :func:`pyscf.vibronic.alignment.align_geometries` (``coords @ R``) the
    rotated modes are

    .. math:: L'_{a y, k} = \\sum_x L_{a x, k}\\, R_{x y}

    i.e. the block-diagonal orthogonal transformation
    :math:`1_{natm} \\otimes R` applied to the rows.

    Args:
        modes : (3*natm, nvib) array.
        rot : (3,3) rotation matrix.

    Returns:
        (3*natm, nvib) array.
    '''
    modes = numpy.asarray(modes, dtype=float)
    rot = numpy.asarray(rot, dtype=float)
    if modes.ndim != 2 or modes.shape[0] % 3 != 0:
        raise ValueError('modes must have shape (3*natm, nvib), got %s' % (modes.shape,))
    if rot.shape != (3, 3):
        raise ValueError('rot must have shape (3,3), got %s' % (rot.shape,))
    natm = modes.shape[0] // 3
    nvib = modes.shape[1]
    out = numpy.einsum('axk,xy->ayk', modes.reshape(natm, 3, nvib), rot)
    return out.reshape(3 * natm, nvib)


def _mass_weighted_displacement(mass_au, coords_i, coords_f):
    '''``d = M^{1/2} (x_i0 - x_f0)`` flattened to ``(3*natm,)``, bohr sqrt(m_e).

    Both geometries must already be in a common frame.
    '''
    sqrt_m = numpy.repeat(numpy.sqrt(mass_au), 3)
    return sqrt_m * (numpy.asarray(coords_i) - numpy.asarray(coords_f)).ravel()


def _check_modes(modes, name, n3=None):
    modes = numpy.asarray(modes, dtype=float)
    if modes.ndim != 2:
        raise ValueError('%s must be 2-dimensional (3*natm, nvib), got shape %s'
                         % (name, (modes.shape,)))
    if not numpy.all(numpy.isfinite(modes)):
        raise ValueError('%s contains non-finite values' % name)
    if n3 is not None and modes.shape[0] != n3:
        raise ValueError('%s has %d rows but 3*natm = %d'
                         % (name, modes.shape[0], n3))
    return modes


def _orthonormality_error(modes):
    if modes.shape[1] == 0:
        return 0.
    gram = modes.T.dot(modes)
    return float(abs(gram - numpy.eye(gram.shape[0])).max())


def _dimensionless(K, freq):
    '''``K_bar_k = sqrt(omega_k) K_k``.

    An imaginary mode is stored as a negative ``freq`` by
    :class:`~pyscf.vibronic.normal_modes.HarmonicModel`; there is no real
    dimensionless displacement in that case, so the entry is ``nan``.  This can
    only happen under ``allow_imaginary=True``, whose results are documented as
    not physically meaningful.
    '''
    K = numpy.asarray(K, dtype=float)
    freq = numpy.asarray(freq, dtype=float)
    w = numpy.where(freq > 0, freq, numpy.nan)
    with numpy.errstate(invalid='ignore'):
        return numpy.sqrt(w) * K


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def _build_diagnostics(J, K, displacement, modes_target, freq_target,
                       eckart_before, eckart_after, eckart_scale,
                       orthonormality_i, orthonormality_f):
    '''Assemble the diagnostics dictionary.

    ``modes_target`` is the mode matrix used to project ``displacement``, i.e.
    ``L_f`` for a forward transformation.  ``K`` is the displacement vector
    actually stored on the :class:`Duschinsky` object, which for the forward
    transformation equals ``modes_target.T @ displacement`` exactly but for
    :meth:`Duschinsky.inverse` is the pseudo-inverse expression (see there).
    '''
    J = numpy.asarray(J, dtype=float)
    K = numpy.asarray(K, dtype=float)
    nvib_f, nvib_i = J.shape

    gram_col = J.T.dot(J) if nvib_i else numpy.zeros((0, 0))
    gram_row = J.dot(J.T) if nvib_f else numpy.zeros((0, 0))
    orth = float(abs(gram_col - numpy.eye(nvib_i)).max()) if nvib_i else 0.
    orth_row = float(abs(gram_row - numpy.eye(nvib_f)).max()) if nvib_f else 0.

    det_J = float(numpy.linalg.det(J)) if (nvib_f == nvib_i and nvib_f > 0) else None

    if min(nvib_f, nvib_i) > 0:
        sv = numpy.linalg.svd(J, compute_uv=False)
    else:
        sv = numpy.zeros(0)

    # off-diagonal magnitude, over the rectangular "diagonal" mask
    offdiag_mask = ~numpy.eye(nvib_f, nvib_i, dtype=bool)
    max_offdiag = float(abs(J[offdiag_mask]).max()) if offdiag_mask.any() else 0.
    mode_mixing = (1. - (J**2).max(axis=1)) if nvib_i else numpy.ones(nvib_f)

    d = numpy.asarray(displacement, dtype=float)
    d2 = float(d.dot(d))
    d_norm = numpy.sqrt(d2)
    k_proj = modes_target.T.dot(d) if modes_target.shape[1] else numpy.zeros(0)
    captured = float(k_proj.dot(k_proj))
    # Below this the "displacement" is pure round-off in the coordinates
    # (eckart_scale = sum_a m_a |x_a|^2 is the natural squared scale of
    # M^{1/2} x), and the *relative* excluded norm is meaningless noise.
    d_floor = 1e2 * numpy.finfo(float).eps * max(numpy.sqrt(max(eckart_scale, 0.)), 1.)
    if d_norm > d_floor:
        excluded = max(d2 - captured, 0.) / d2
    else:
        excluded = 0.

    # ||L_f K - P_f d|| / ||d||, with P_f = L_f L_f^T.  An algebraic identity
    # (hence a pure numerical self-check, ~1e-16) for the forward transform;
    # for the pseudo-inverse of a non-orthogonal J it measures the deviation of
    # pinv(J) from J^T.
    recon = modes_target.dot(K) - modes_target.dot(k_proj)
    recon_err = float(numpy.linalg.norm(recon) / max(d_norm, d_floor))

    freq_target = numpy.asarray(freq_target, dtype=float)
    s = 0.5 * _dimensionless(K, freq_target)**2
    lam = s * freq_target
    total_lam = float(numpy.nansum(lam))

    return {
        'orthogonality_error': orth,
        'row_orthogonality_error': orth_row,
        'det_J': det_J,
        'subspace_overlap': sv,
        'subspace_overlap_min': float(sv.min()) if sv.size else 0.,
        'subspace_overlap_max': float(sv.max()) if sv.size else 0.,
        'excluded_mode_norm': float(excluded),
        'displacement_reconstruction_error': recon_err,
        'displacement_norm': float(d_norm),
        'eckart_residual_before': float(eckart_before),
        'eckart_residual_after': float(eckart_after),
        'eckart_residual_scale': float(eckart_scale),
        'max_offdiag_J': max_offdiag,
        'mode_mixing': mode_mixing,
        'max_mode_mixing': float(mode_mixing.max()) if mode_mixing.size else 0.,
        'total_reorganization_energy': total_lam,
        'total_reorganization_energy_cm': float(units.au2wavenumber(total_lam)),
        'modes_orthonormality_error_i': float(orthonormality_i),
        'modes_orthonormality_error_f': float(orthonormality_f),
    }


# ---------------------------------------------------------------------------
# Duschinsky container
# ---------------------------------------------------------------------------

class Duschinsky(lib.StreamObject):
    '''Result of a Duschinsky transformation, ``Q_f = J Q_i + K``.

    Attributes:
        J : (nvib_f, nvib_i) Duschinsky matrix ``L_f^T L_i``.  Rows index the
            **final**-state modes, columns the **initial**-state modes.
        K : (nvib_f,) displacement ``L_f^T M^{1/2}(x_i0 - x_f0)``, in
            bohr sqrt(m_e).
        K_dimensionless : (nvib_f,) ``sqrt(omega_f_k) * K_k``.
        freq_i, freq_f : (nvib_i,), (nvib_f,) angular frequencies in a.u. (Eh).
        huang_rhys : (nvib_f,) ``S_k = 0.5 * K_dimensionless[k]**2``.
        reorganization_energy : (nvib_f,) ``lambda_k = S_k * omega_f_k``, Eh.
        diagnostics : dict, see :meth:`dump_diagnostics`.
        model_i, model_f : the input :class:`~pyscf.vibronic.normal_modes.HarmonicModel`
            objects, or ``None`` when built from raw arrays.
        rotation : (3,3) the proper rotation applied to the final-state
            structure and to ``L_f``.  Identity if ``align=False``.
        mass : (natm,) masses in electron masses.
        coords_i : (natm,3) initial geometry, bohr, centred at its centre of
            mass.
        coords_f : (natm,3) final geometry, bohr, centred **and rotated** into
            the initial-state frame.
        modes_i : (3*natm, nvib_i) ``L_i``.
        modes_f : (3*natm, nvib_f) ``L_f`` **after** the alignment rotation.
        displacement : (3*natm,) ``d = M^{1/2}(x_i0 - x_f0)`` in the common
            frame.
        nvib_i, nvib_f : int.

    The signs of the columns of ``L_i``/``L_f`` are an arbitrary convention, so
    ``J`` and ``K`` are only defined up to ``J -> D_f J D_i``, ``K -> D_f K``
    with ``D`` diagonal sign matrices.  Every *physical* quantity here
    (``huang_rhys``, ``reorganization_energy``, ``abs(J)``, the singular values
    of ``J``) is invariant to that, which ``test_duschinsky.py`` asserts.
    '''

    def __init__(self, J, K, freq_i, freq_f, mass, coords_i, coords_f,
                 modes_i, modes_f, displacement, rotation, diagnostics,
                 model_i=None, model_f=None, verbose=None, stdout=None):
        self.stdout = sys.stdout if stdout is None else stdout
        self.verbose = logger.NOTE if verbose is None else verbose
        self.J = numpy.asarray(J, dtype=float)
        self.K = numpy.asarray(K, dtype=float)
        self.freq_i = numpy.asarray(freq_i, dtype=float)
        self.freq_f = numpy.asarray(freq_f, dtype=float)
        self.mass = numpy.asarray(mass, dtype=float)
        self.coords_i = numpy.asarray(coords_i, dtype=float)
        self.coords_f = numpy.asarray(coords_f, dtype=float)
        self.modes_i = numpy.asarray(modes_i, dtype=float)
        self.modes_f = numpy.asarray(modes_f, dtype=float)
        self.displacement = numpy.asarray(displacement, dtype=float)
        self.rotation = numpy.asarray(rotation, dtype=float)
        self.diagnostics = diagnostics
        self.model_i = model_i
        self.model_f = model_f

        if self.J.shape != (self.freq_f.size, self.freq_i.size):
            raise ValueError('J has shape %s but (nvib_f, nvib_i) = (%d, %d)'
                             % (self.J.shape, self.freq_f.size, self.freq_i.size))
        if self.K.shape != (self.freq_f.size,):
            raise ValueError('K has shape %s but nvib_f = %d'
                             % (self.K.shape, self.freq_f.size))

    # -- derived quantities -------------------------------------------------

    @property
    def nvib_i(self):
        '''Number of initial-state vibrational modes.'''
        return self.freq_i.size

    @property
    def nvib_f(self):
        '''Number of final-state vibrational modes.'''
        return self.freq_f.size

    @property
    def natm(self):
        '''Number of atoms.'''
        return self.mass.size

    @property
    def K_dimensionless(self):
        '''(nvib_f,) dimensionless displacement ``sqrt(omega_f_k) * K_k``.

        ``nan`` for an imaginary final-state mode (only reachable with
        ``allow_imaginary=True``).
        '''
        return _dimensionless(self.K, self.freq_f)

    @property
    def huang_rhys(self):
        '''(nvib_f,) Huang-Rhys factors ``S_k = 0.5 * K_dimensionless[k]**2``.

        Phase-invariant, hence a genuine physical observable.
        '''
        return 0.5 * self.K_dimensionless**2

    @property
    def reorganization_energy(self):
        '''(nvib_f,) per-mode reorganisation energy ``lambda_k = S_k*omega_f_k``
        in Eh.

        Equivalently ``0.5 * omega_f_k**2 * K_k**2``: the harmonic energy that
        the *final*-state surface stores in mode ``k`` at the initial-state
        equilibrium geometry.
        '''
        return self.huang_rhys * self.freq_f

    @property
    def total_reorganization_energy(self):
        '''Sum of :attr:`reorganization_energy`, Eh.  ``nan`` entries (imaginary
        modes) are skipped.
        '''
        return float(numpy.nansum(self.reorganization_energy))

    # -- the transformation itself ------------------------------------------

    def apply(self, Q_i):
        '''Apply the transformation: ``Q_f = J Q_i + K``.

        Args:
            Q_i : ``(nvib_i,)`` or ``(..., nvib_i)`` initial-state normal
                coordinates, bohr sqrt(m_e).

        Returns:
            array of the same leading shape with last dimension ``nvib_f``.
        '''
        Q_i = numpy.asarray(Q_i, dtype=float)
        if Q_i.shape[-1:] != (self.nvib_i,):
            raise ValueError('Q_i must have last dimension nvib_i = %d, got shape %s'
                             % (self.nvib_i, (Q_i.shape,)))
        return numpy.einsum('fi,...i->...f', self.J, Q_i) + self.K

    def inverse(self, verbose=None):
        '''The reverse transformation, ``Q_i = J_rev Q_f + K_rev``.

        Derivation.  Inverting the affine map ``Q_f = J Q_i + K`` gives

        .. math::

            Q_i = J^{+} Q_f - J^{+} K
            \\quad\\Longrightarrow\\quad
            J_{\\rm rev} = J^{+}, \\qquad K_{\\rm rev} = -J^{+} K

        with :math:`J^{+}` the Moore-Penrose pseudo-inverse
        (:func:`numpy.linalg.pinv`).  This is exact whenever ``J`` has full
        column rank, so ``inverse().apply(self.apply(Q_i)) == Q_i``.

        Relation to the *physical* reverse transformation.  Rebuilding the
        transformation with the two states swapped gives

        .. math::

            J_{\\rm rev}^{\\rm phys} = L_i^{\\mathsf{T}} L_f = J^{\\mathsf{T}},
            \\qquad
            K_{\\rm rev}^{\\rm phys}
              = L_i^{\\mathsf{T}} M^{1/2}(x_{f0} - x_{i0}) = -L_i^{\\mathsf{T}} d

        and if the two states span the *same* vibrational subspace
        (:math:`L_i L_i^{\\mathsf{T}} = L_f L_f^{\\mathsf{T}}`, so that ``J`` is
        orthogonal and :math:`P d = d`) then

        .. math::

            -L_i^{\\mathsf{T}} d = -L_i^{\\mathsf{T}} L_f L_f^{\\mathsf{T}} d
              = -J^{\\mathsf{T}} K = -J^{+} K

        i.e. the two agree exactly.  This is the relation quoted in
        REFERENCES.md sec. 2.3, ``K(i<-f) = -J^{-1} K(f<-i)``.  The deviation
        between the pseudo-inverse and the physical expression is asserted to be
        small and reported as ``diagnostics['inverse_physical_deviation_J']``
        and ``['inverse_physical_deviation_K']`` on the returned object; a large
        value means the two vibrational subspaces do not coincide and the
        reverse transformation is genuinely not the transpose.

        Returns:
            a new :class:`Duschinsky` with the roles of the two states swapped.
        '''
        log = logger.new_logger(self, verbose)
        J_rev = numpy.linalg.pinv(self.J)
        K_rev = -J_rev.dot(self.K)

        J_phys = self.J.T
        K_phys = -self.modes_i.T.dot(self.displacement)
        dev_J = float(abs(J_rev - J_phys).max()) if J_rev.size else 0.
        dev_K = float(abs(K_rev - K_phys).max()) if K_rev.size else 0.

        d_rev = -self.displacement
        eck = self.diagnostics
        diag = _build_diagnostics(
            J_rev, K_rev, d_rev, self.modes_i, self.freq_i,
            eck['eckart_residual_before'], eck['eckart_residual_after'],
            eck['eckart_residual_scale'],
            _orthonormality_error(self.modes_f), _orthonormality_error(self.modes_i))
        diag['inverse_physical_deviation_J'] = dev_J
        diag['inverse_physical_deviation_K'] = dev_K

        orth = self.diagnostics['orthogonality_error']
        if orth < ORTHOGONALITY_TOL:
            # For an orthogonal J the pseudo-inverse *is* the transpose and the
            # physical reverse displacement.  Guard the identity, scaled by the
            # non-orthogonality that is actually present.
            tol = max(1e-10, 100 * orth)
            if dev_J > tol:
                raise RuntimeError(
                    'inverse(): pinv(J) deviates from J^T by %.3e although J is '
                    'orthogonal to %.3e.  This is a bug.' % (dev_J, orth))
            kscale = max(float(abs(self.K).max()) if self.K.size else 0., 1.)
            if dev_K > tol * kscale * 10:
                raise RuntimeError(
                    'inverse(): -pinv(J) K deviates from the physical '
                    '-L_i^T M^{1/2}(x_i0-x_f0) by %.3e although J is orthogonal to '
                    '%.3e.  This is a bug.' % (dev_K, orth))
        else:
            log.warn('inverse(): J is not orthogonal (orthogonality_error = %.3e > %.1e), '
                     'so the algebraic inverse pinv(J) differs from the physical reverse '
                     'transformation L_i^T L_f by %.3e.  The returned object inverts the '
                     'affine map exactly; it is not L_i^T L_f.', orth, ORTHOGONALITY_TOL, dev_J)

        return Duschinsky(J_rev, K_rev, self.freq_f, self.freq_i, self.mass,
                          self.coords_f, self.coords_i, self.modes_f, self.modes_i,
                          d_rev, self.rotation.T, diag,
                          model_i=self.model_f, model_f=self.model_i,
                          verbose=self.verbose, stdout=self.stdout)

    # -- reporting ----------------------------------------------------------

    def dump_diagnostics(self, verbose=None):
        '''Print the diagnostics with :mod:`pyscf.lib.logger`.  Returns ``self``.'''
        log = logger.new_logger(self, verbose)
        d = self.diagnostics
        log.note('Duschinsky transformation: nvib_i = %d, nvib_f = %d',
                 self.nvib_i, self.nvib_f)
        log.note('  ||J^T J - 1||_max          = %.3e', d['orthogonality_error'])
        log.note('  ||J J^T - 1||_max          = %.3e', d['row_orthogonality_error'])
        if d['det_J'] is not None:
            log.note('  det J                      = %+.10f', d['det_J'])
        log.note('  singular values of J       = [%.10f, %.10f] (min, max)',
                 d['subspace_overlap_min'], d['subspace_overlap_max'])
        log.note('  max |J_offdiag|            = %.6f', d['max_offdiag_J'])
        log.note('  max mode mixing 1-max_j J^2= %.6f', d['max_mode_mixing'])
        log.note('  ||d||                      = %.6e bohr sqrt(m_e)', d['displacement_norm'])
        log.note('  excluded_mode_norm         = %.3e', d['excluded_mode_norm'])
        log.note('  displacement reconstruction= %.3e', d['displacement_reconstruction_error'])
        log.note('  Eckart residual before/after = %.3e / %.3e  (scale %.3e)',
                 d['eckart_residual_before'], d['eckart_residual_after'],
                 d['eckart_residual_scale'])
        log.note('  total reorganization energy = %.8f Eh = %.2f cm^-1',
                 d['total_reorganization_energy'], d['total_reorganization_energy_cm'])
        if self.nvib_f:
            log.note('  %-5s %14s %14s %12s %12s %12s', 'mode', 'omega_f/cm^-1',
                     'K/bohr.sqrt(me)', 'K_bar', 'S', 'lambda/cm^-1')
            wn = units.au2wavenumber(self.freq_f)
            kbar = self.K_dimensionless
            s = self.huang_rhys
            lam = units.au2wavenumber(self.reorganization_energy)
            for k in range(self.nvib_f):
                log.note('  %-5d %14.3f %14.6f %12.6f %12.6f %12.3f',
                         k, wn[k], self.K[k], kbar[k], s[k], lam[k])
        return self

    def __repr__(self):
        d = self.diagnostics
        return ('<Duschinsky nvib_i=%d nvib_f=%d ||J^TJ-1||=%.2e '
                'max|J_off|=%.3f max S=%.3f lambda_tot=%.1f cm^-1>'
                % (self.nvib_i, self.nvib_f, d['orthogonality_error'],
                   d['max_offdiag_J'],
                   float(self.huang_rhys.max()) if self.nvib_f else 0.,
                   d['total_reorganization_energy_cm']))

    def match_modes(self, degeneracy_tol=None):
        '''Convenience wrapper for :func:`match_modes` on this object.'''
        return match_modes(self, degeneracy_tol=degeneracy_tol,
                           verbose=self.verbose, stdout=self.stdout)


# ---------------------------------------------------------------------------
# the transformation
# ---------------------------------------------------------------------------

def duschinsky_from_arrays(L_i, L_f, mass, coords_i, coords_f, freq_i, freq_f,
                           mass_unit='au', align=True, mass_weighted_align=True,
                           allow_imaginary=False, model_i=None, model_f=None,
                           verbose=None, stdout=None):
    '''Low-level Duschinsky transformation from raw arrays.

    Use this when the normal modes come from another program, or in tests where
    ``L_i``/``L_f`` are constructed by hand.  Everything is in atomic units.

    Args:
        L_i : (3*natm, nvib_i) mass-weighted initial-state mode matrix, columns
            orthonormal.
        L_f : (3*natm, nvib_f) mass-weighted final-state mode matrix, columns
            orthonormal, expressed in the frame of ``coords_f``.
        mass : (natm,) atomic masses; see ``mass_unit``.
        coords_i : (natm,3) initial-state equilibrium geometry, bohr.  Any
            origin; centred internally.
        coords_f : (natm,3) final-state equilibrium geometry, bohr.
        freq_i : (nvib_i,) angular frequencies, a.u. (Eh).  A negative entry
            denotes an imaginary frequency, following the convention of
            :class:`~pyscf.vibronic.normal_modes.HarmonicModel`.
        freq_f : (nvib_f,) as ``freq_i``.

    Kwargs:
        mass_unit : ``'au'`` (default, **electron masses**) or ``'amu'``.  The
            default differs from
            :class:`~pyscf.vibronic.normal_modes.HarmonicModel` (which takes
            amu) because ``L_i``/``L_f`` must have been mass-weighted with the
            same masses that are passed here, and internally that is always
            electron masses.
        align : bool.  ``True`` (default) rotates ``coords_f`` and ``L_f`` onto
            ``coords_i`` with the Kabsch/Eckart rotation.  ``False`` skips only
            the *rotation*; both structures are still shifted to their centre of
            mass, since the Eckart translational condition is not optional.
        mass_weighted_align : bool, use the masses as Kabsch weights (the Eckart
            choice).  ``False`` gives an unweighted geometric fit, which is not
            the Eckart frame.
        allow_imaginary : bool.  ``False`` (default) raises :class:`ValueError`
            if either state has an imaginary frequency.
        model_i, model_f : optional objects to attach to the result.
        verbose, stdout : usual PySCF logging controls.

    Returns:
        :class:`Duschinsky`
    '''
    log = logger.new_logger(_LogHolder(stdout, verbose), verbose)

    mass = numpy.asarray(mass, dtype=float).reshape(-1)
    key = str(mass_unit).strip().lower()
    if key == 'amu':
        mass = numpy.asarray(units.amu2au(mass), dtype=float)
    elif key in ('au', 'a.u.', 'me', 'electron_mass'):
        mass = mass.copy()
    else:
        raise ValueError("mass_unit must be 'au' or 'amu', got %r" % (mass_unit,))

    mass, coords_i = alignment._check_mass_coords(mass, coords_i)
    _, coords_f = alignment._check_mass_coords(mass, coords_f)
    natm = mass.shape[0]
    n3 = 3 * natm

    L_i = _check_modes(L_i, 'L_i', n3)
    L_f = _check_modes(L_f, 'L_f', n3)
    freq_i = numpy.asarray(freq_i, dtype=float).reshape(-1)
    freq_f = numpy.asarray(freq_f, dtype=float).reshape(-1)
    if freq_i.size != L_i.shape[1]:
        raise ValueError('freq_i has %d entries but L_i has %d columns'
                         % (freq_i.size, L_i.shape[1]))
    if freq_f.size != L_f.shape[1]:
        raise ValueError('freq_f has %d entries but L_f has %d columns'
                         % (freq_f.size, L_f.shape[1]))
    if not (numpy.all(numpy.isfinite(freq_i)) and numpy.all(numpy.isfinite(freq_f))):
        raise ValueError('freq_i/freq_f contain non-finite values')

    _reject_imaginary(freq_i, freq_f, allow_imaginary, log)

    err_i = _orthonormality_error(L_i)
    if err_i > ORTHONORMALITY_ASSERT_TOL:
        raise ValueError('L_i does not have orthonormal columns: max|L_i^T L_i - 1| = %.3e'
                         % err_i)
    err_f_in = _orthonormality_error(L_f)
    if err_f_in > ORTHONORMALITY_ASSERT_TOL:
        raise ValueError('L_f does not have orthonormal columns: max|L_f^T L_f - 1| = %.3e'
                         % err_f_in)

    # --- alignment ---------------------------------------------------------
    x_i = alignment.shift_to_center_of_mass(mass, coords_i)
    x_f_raw = alignment.shift_to_center_of_mass(mass, coords_f)
    eckart_scale = float(numpy.einsum('a,ax,ax->', mass, x_i, x_i))
    eckart_before = alignment.eckart_residual(mass, x_i, x_f_raw)

    if align:
        x_f, rot = alignment.align_geometries(mass, x_i, x_f_raw,
                                              mass_weighted=mass_weighted_align)
        L_f_rot = rotate_modes(L_f, rot)
        err_f = _orthonormality_error(L_f_rot)
        # A block-diagonal rotation is orthogonal, so this cannot change the
        # Gram matrix -- assert it anyway as a self-check on the reshaping.
        if err_f > ORTHONORMALITY_ASSERT_TOL:
            raise RuntimeError(
                'the rotated L_f lost column orthonormality: max|L_f^T L_f - 1| = %.3e '
                '(was %.3e before rotation).  Since 1_natm (x) R is orthogonal this '
                'indicates a bug in rotate_modes().' % (err_f, err_f_in))
    else:
        x_f = x_f_raw
        rot = numpy.eye(3)
        L_f_rot = L_f.copy()
        err_f = err_f_in
        log.debug('align=False: the final-state structure is centred but NOT rotated '
                  'onto the initial-state structure.  Any relative orientation of the '
                  'two structures leaks into K and J; check excluded_mode_norm.')
    eckart_after = alignment.eckart_residual(mass, x_i, x_f)
    log.debug('Eckart rotational residual: %.3e -> %.3e (scale sum_a m_a |x_a|^2 = %.3e)',
              eckart_before, eckart_after, eckart_scale)
    log.debug('mass-weighted RMSD after alignment = %.6e bohr',
              alignment.rmsd(x_i, x_f, weights=mass))

    # --- J and K -----------------------------------------------------------
    d = _mass_weighted_displacement(mass, x_i, x_f)
    J = L_f_rot.T.dot(L_i)
    K = L_f_rot.T.dot(d)

    diag = _build_diagnostics(J, K, d, L_f_rot, freq_f,
                              eckart_before, eckart_after, eckart_scale,
                              err_i, err_f)
    diag['aligned'] = bool(align)
    diag['mass_weighted_align'] = bool(mass_weighted_align)

    dusch = Duschinsky(J, K, freq_i, freq_f, mass, x_i, x_f, L_i, L_f_rot, d,
                       rot, diag, model_i=model_i, model_f=model_f,
                       verbose=verbose, stdout=stdout)
    _warn_diagnostics(dusch, log)
    return dusch


def duschinsky_transform(model_i, model_f, align=True, mass_weighted_align=True,
                         allow_imaginary=False, mass_rtol=MASS_RTOL, verbose=None):
    '''Duschinsky transformation between two :class:`HarmonicModel` objects.

    Computes ``Q_f = J Q_i + K`` with ``J = L_f^T L_i`` and
    ``K = L_f^T M^{1/2}(x_i0 - x_f0)``; see the module docstring for the
    derivation and the sign convention.

    Args:
        model_i : initial-state
            :class:`~pyscf.vibronic.normal_modes.HarmonicModel`.
        model_f : final-state
            :class:`~pyscf.vibronic.normal_modes.HarmonicModel`.

    Kwargs:
        align : bool, rotate the final-state structure (and ``L_f``) onto the
            initial-state structure.  Default ``True``; switching it off is
            almost always wrong and is provided only for diagnostics.
        mass_weighted_align : bool, mass-weight the Kabsch rotation (the Eckart
            condition).  Default ``True``.
        allow_imaginary : bool, default ``False``: an imaginary frequency in
            either state raises :class:`ValueError`.
        mass_rtol : float, relative tolerance for the per-atom mass comparison.
        verbose : logging level.

    Returns:
        :class:`Duschinsky`

    Raises:
        ValueError : if the two models are not compatible -- different number of
            atoms, different nuclear charges, different masses, different number
            of vibrational modes, or a different rotor type -- or if either has
            an imaginary frequency and ``allow_imaginary`` is ``False``.
    '''
    _validate_models(model_i, model_f, mass_rtol)
    if verbose is None:
        verbose = getattr(model_i, 'verbose', None)
    return duschinsky_from_arrays(
        model_i.modes, model_f.modes, model_i.mass, model_i.coords, model_f.coords,
        model_i.freq, model_f.freq, mass_unit='au', align=align,
        mass_weighted_align=mass_weighted_align, allow_imaginary=allow_imaginary,
        model_i=model_i, model_f=model_f, verbose=verbose,
        stdout=getattr(model_i, 'stdout', None))


class _LogHolder(object):
    '''Minimal carrier of ``stdout``/``verbose`` for :func:`logger.new_logger`.'''

    def __init__(self, stdout, verbose):
        self.stdout = sys.stdout if stdout is None else stdout
        self.verbose = logger.NOTE if verbose is None else verbose


def _validate_models(model_i, model_f, mass_rtol=MASS_RTOL):
    '''Check that two harmonic models describe the same nuclei.

    Raises :class:`ValueError` with an explanation on any mismatch.
    '''
    for name, m in (('model_i', model_i), ('model_f', model_f)):
        for attr in ('atom_charges', 'mass', 'coords', 'modes', 'freq', 'nvib',
                     'rotor_type', 'imaginary'):
            if getattr(m, attr, None) is None:
                raise ValueError(
                    '%s does not look like a HarmonicModel: attribute %r is missing or '
                    'None.  Build it with HarmonicModel/harmonic_model first, or use '
                    'duschinsky_from_arrays() for raw arrays.' % (name, attr))

    if model_i.natm != model_f.natm:
        raise ValueError(
            'the two electronic states have different numbers of atoms (%d and %d).  '
            'A Duschinsky transformation relates two potential-energy surfaces of the '
            '*same* set of nuclei.' % (model_i.natm, model_f.natm))

    z_i = numpy.asarray(model_i.atom_charges, dtype=int)
    z_f = numpy.asarray(model_f.atom_charges, dtype=int)
    if not numpy.array_equal(z_i, z_f):
        bad = numpy.flatnonzero(z_i != z_f)
        raise ValueError(
            'the two electronic states have different nuclear charges at atom(s) %s: '
            '%s vs %s.  The nuclei -- and their order -- must be identical; a '
            'permuted atom ordering between the two states silently produces a wrong '
            'alignment and a wrong Duschinsky matrix.'
            % (bad.tolist(), z_i[bad].tolist(), z_f[bad].tolist()))

    m_i = numpy.asarray(model_i.mass, dtype=float)
    m_f = numpy.asarray(model_f.mass, dtype=float)
    scale = numpy.maximum(abs(m_i), abs(m_f))
    dev = abs(m_i - m_f) / numpy.where(scale > 0, scale, 1.)
    if numpy.any(dev > mass_rtol):
        bad = numpy.flatnonzero(dev > mass_rtol)
        raise ValueError(
            'the two electronic states use different atomic masses at atom(s) %s '
            '(relative deviation up to %.3e > mass_rtol %.1e): %s vs %s amu.  An '
            'isotopic mismatch between two electronic states of the same molecule is '
            'a user error: J = L_f^T L_i and K = L_f^T M^{1/2}(x_i0-x_f0) are only '
            'defined for a single mass matrix M.'
            % (bad.tolist(), dev.max(), mass_rtol,
               numpy.array2string(units.au2amu(m_i[bad]), precision=6),
               numpy.array2string(units.au2amu(m_f[bad]), precision=6)))

    if model_i.rotor_type != model_f.rotor_type:
        raise ValueError(
            'the two electronic states have different rotor types (%r and %r).  The '
            'number of rotational degrees of freedom projected out differs (5 for '
            'LINEAR, 6 for REGULAR, 3 for ATOM), so the two states retain vibrational '
            'subspaces of different dimension (nvib = %d vs %d) which do not even sit '
            'in the same space: J = L_f^T L_i would be rectangular and the vibrational '
            'subspaces would not coincide, so Q_f = J Q_i + K cannot hold.  A genuine '
            'linear-to-bent change requires a curvilinear (internal-coordinate) or '
            'fully anharmonic treatment, not a rectilinear Duschinsky rotation.'
            % (model_i.rotor_type, model_f.rotor_type, model_i.nvib, model_f.nvib))

    if model_i.nvib != model_f.nvib:
        raise ValueError(
            'the two electronic states have different numbers of vibrational modes '
            '(%d and %d) even though both are %r with %d atoms.  This should not '
            'happen; check the models.'
            % (model_i.nvib, model_f.nvib, model_i.rotor_type, model_i.natm))


def _reject_imaginary(freq_i, freq_f, allow_imaginary, log):
    '''Imaginary frequencies are stored as negative numbers by
    :class:`~pyscf.vibronic.normal_modes.HarmonicModel`.
    '''
    n_i = int((freq_i < 0).sum())
    n_f = int((freq_f < 0).sum())
    if n_i == 0 and n_f == 0:
        return
    parts = []
    if n_i:
        parts.append('initial state: %s cm^-1 (imaginary)'
                     % numpy.array2string(units.au2wavenumber(abs(freq_i[freq_i < 0])),
                                          precision=2))
    if n_f:
        parts.append('final state: %s cm^-1 (imaginary)'
                     % numpy.array2string(units.au2wavenumber(abs(freq_f[freq_f < 0])),
                                          precision=2))
    msg = ('imaginary frequencies found -- %s.  A Duschinsky transformation between '
           'structures that are not both minima is not meaningful: the harmonic '
           'Franck-Condon factors do not exist along an unstable coordinate, and the '
           'Eckart/Kabsch alignment assumes both structures are stationary points.  '
           'Re-optimise the geometries, or pass allow_imaginary=True to compute J and '
           'K anyway (the results are not physically meaningful).' % '; '.join(parts))
    if not allow_imaginary:
        raise ValueError(msg)
    log.warn('%s', msg)


def _warn_diagnostics(dusch, log):
    d = dusch.diagnostics
    if d['excluded_mode_norm'] > EXCLUDED_NORM_TOL:
        log.warn('excluded_mode_norm = %.3e > %.1e: %.2f%% of the mass-weighted '
                 'geometry change M^{1/2}(x_i0 - x_f0) does NOT lie in the final '
                 "state's vibrational subspace.  That part is residual "
                 'translation/rotation (or a breakdown of the rectilinear model), and '
                 'Q_f = J Q_i + K does not hold for it.  Check the alignment (align=%s) '
                 'and the atom ordering.',
                 d['excluded_mode_norm'], EXCLUDED_NORM_TOL,
                 100 * d['excluded_mode_norm'], d.get('aligned'))
    if d['orthogonality_error'] > ORTHOGONALITY_TOL:
        log.warn('||J^T J - 1||_max = %.3e > %.1e: the two states do not span the same '
                 'vibrational subspace, so J is not orthogonal.  The overlap singular '
                 'values run from %.6f to %.6f; a minimum well below 1 means leakage '
                 'into the translation/rotation subspace.',
                 d['orthogonality_error'], ORTHOGONALITY_TOL,
                 d['subspace_overlap_min'], d['subspace_overlap_max'])
    if d['det_J'] is not None and abs(abs(d['det_J']) - 1.) > ORTHOGONALITY_TOL:
        log.warn('|det J| = %.6f differs from 1 by more than %.1e.  See the '
                 'orthogonality diagnostics.', abs(d['det_J']), ORTHOGONALITY_TOL)


# ---------------------------------------------------------------------------
# mode matching / correlation
# ---------------------------------------------------------------------------

class ModeMatch(object):
    '''One record of a :class:`ModeMatching`.

    Attributes:
        mode_i : int, initial-state mode index.
        mode_f : int or ``None``.  The matched final-state mode, or ``None``
            when either the initial or the matched final mode belongs to a
            **degenerate block**, in which case an individual one-to-one
            correspondence carries no physical meaning (see
            :func:`match_modes`).
        mode_f_raw : int, the raw optimal-assignment partner.  Always an int.
            Inside a degenerate block it is an arbitrary artefact of the
            diagonalisation and must not be interpreted.
        overlap : float.  For a non-degenerate pair, ``abs(J[mode_f, mode_i])``.
            For a degenerate entry, the **block-to-block** overlap
            ``||J[block_f, block_i]||_F``, which is invariant under an
            orthogonal re-mixing of either degenerate subspace.
        freq_shift : float, ``omega_f - omega_i`` in a.u.  For a degenerate
            entry the block-averaged frequencies are used.
        degenerate : bool.
        block_i, block_f : int, block indices in
            :attr:`ModeMatching.blocks_i` / :attr:`ModeMatching.blocks_f`.
        block_singular_values : (min(len(block_f), len(block_i)),) array or
            ``None``.  Singular values of ``J[block_f, block_i]``; also
            invariant under re-mixing.  ``None`` for non-degenerate entries.
    '''

    def __init__(self, mode_i, mode_f, mode_f_raw, overlap, freq_shift,
                 degenerate, block_i, block_f, block_singular_values=None):
        self.mode_i = int(mode_i)
        self.mode_f = None if mode_f is None else int(mode_f)
        self.mode_f_raw = int(mode_f_raw)
        self.overlap = float(overlap)
        self.freq_shift = float(freq_shift)
        self.degenerate = bool(degenerate)
        self.block_i = int(block_i)
        self.block_f = int(block_f)
        self.block_singular_values = block_singular_values

    def __repr__(self):
        if self.degenerate:
            return ('<ModeMatch i=%d -> block_f=%d (degenerate, individual assignment '
                    'meaningless; raw partner %d) block overlap=%.6f>'
                    % (self.mode_i, self.block_f, self.mode_f_raw, self.overlap))
        return ('<ModeMatch i=%d -> f=%d overlap=%.6f dfreq=%.2f cm^-1>'
                % (self.mode_i, self.mode_f, self.overlap,
                   units.au2wavenumber(self.freq_shift)))


class ModeMatching(lib.StreamObject):
    '''Result of :func:`match_modes`.

    Attributes:
        matches : list of :class:`ModeMatch`, one per initial-state mode, in
            order of increasing ``mode_i``.
        blocks_i, blocks_f : list of lists of int.  The degeneracy blocks of the
            initial / final state.  A block of length 1 is non-degenerate.
        block_of_i, block_of_f : (nvib,) int arrays mapping a mode index to its
            block index.
        assignment : (nmatch, 2) int array of the raw ``(mode_f, mode_i)``
            optimal assignment.
        total_overlap : float, ``sum J[f,i]**2`` over the assignment -- the
            quantity the assignment maximises.
        degeneracy_tol : float, the window used, in a.u.
    '''

    def __init__(self, matches, blocks_i, blocks_f, assignment, total_overlap,
                 degeneracy_tol, freq_i=None, freq_f=None, verbose=None, stdout=None):
        self.stdout = sys.stdout if stdout is None else stdout
        self.verbose = logger.NOTE if verbose is None else verbose
        self.matches = matches
        self.blocks_i = blocks_i
        self.blocks_f = blocks_f
        self.assignment = numpy.asarray(assignment, dtype=int).reshape(-1, 2)
        self.total_overlap = float(total_overlap)
        self.degeneracy_tol = float(degeneracy_tol)
        self.freq_i = freq_i
        self.freq_f = freq_f
        self.block_of_i = _block_index_map(blocks_i)
        self.block_of_f = _block_index_map(blocks_f)

    def __len__(self):
        return len(self.matches)

    def __iter__(self):
        return iter(self.matches)

    def __getitem__(self, k):
        return self.matches[k]

    @property
    def n_degenerate(self):
        '''Number of records flagged degenerate.'''
        return sum(1 for m in self.matches if m.degenerate)

    def as_dict(self):
        '''``{mode_i: mode_f}`` for the unambiguous (non-degenerate) matches
        only.  Degenerate entries are deliberately absent.
        '''
        return dict((m.mode_i, m.mode_f) for m in self.matches if not m.degenerate)

    def dump(self, verbose=None):
        '''Print the matching.  Returns ``self``.'''
        log = logger.new_logger(self, verbose)
        log.note('Mode correlation (optimal assignment maximising sum J^2 = %.6f; '
                 'degeneracy window %.3f cm^-1)',
                 self.total_overlap, units.au2wavenumber(self.degeneracy_tol))
        log.note('  %-8s %-8s %10s %12s %s', 'mode_i', 'mode_f', 'overlap',
                 'dfreq/cm^-1', 'note')
        for m in self.matches:
            if m.degenerate:
                log.note('  %-8d %-8s %10.6f %12.2f degenerate block %d<-%d '
                         '(block overlap; individual assignment has no physical meaning)',
                         m.mode_i, '-', m.overlap,
                         units.au2wavenumber(m.freq_shift), m.block_f, m.block_i)
            else:
                log.note('  %-8d %-8d %10.6f %12.2f', m.mode_i, m.mode_f, m.overlap,
                         units.au2wavenumber(m.freq_shift))
        return self

    def __repr__(self):
        return ('<ModeMatching %d modes, %d degenerate, sum J^2 = %.6f>'
                % (len(self.matches), self.n_degenerate, self.total_overlap))


def _block_index_map(blocks):
    n = sum(len(b) for b in blocks)
    out = numpy.zeros(n, dtype=int)
    for ib, blk in enumerate(blocks):
        for k in blk:
            out[k] = ib
    return out


def degeneracy_blocks(freq, tol):
    '''Group modes with (nearly) equal frequencies into degeneracy blocks.

    ``freq`` is assumed sorted ascending (the
    :class:`~pyscf.vibronic.normal_modes.HarmonicModel` convention).  Two
    consecutive modes belong to the same block when their frequencies differ by
    at most ``tol``.

    .. note::

       The grouping is *transitive by construction*: a chain of modes each
       within ``tol`` of the next ends up in one block even if its two ends
       differ by more than ``tol``.  With a physically sensible ``tol`` (~1
       cm^-1) that is what is wanted for a genuinely degenerate manifold, but it
       does mean the block structure is not a pure equivalence relation on the
       frequencies.

    Args:
        freq : (nvib,) frequencies, a.u.
        tol : float, window in a.u.

    Returns:
        list of lists of int.
    '''
    freq = numpy.asarray(freq, dtype=float).reshape(-1)
    if freq.size == 0:
        return []
    if tol < 0:
        raise ValueError('degeneracy_tol must be non-negative, got %g' % tol)
    order = numpy.argsort(freq, kind='stable')
    blocks = [[int(order[0])]]
    for pos in range(1, freq.size):
        k = int(order[pos])
        prev = int(order[pos - 1])
        if abs(freq[k] - freq[prev]) <= tol:
            blocks[-1].append(k)
        else:
            blocks.append([k])
    return blocks


def match_modes(dusch_or_J, freq_i=None, freq_f=None, degeneracy_tol=None,
                verbose=None, stdout=None):
    '''Correlate the initial-state modes with the final-state modes.

    **Matched by ``J`` overlap, never by frequency.**  Two modes of the same
    frequency need not correspond at all -- in a molecule with several
    similar-frequency stretches, frequency ordering permutes essentially at
    random between two electronic states, and the physically meaningful
    correspondence is the one carried by the Duschinsky matrix.  The assignment
    therefore maximises

    .. math:: \\sum_{(k,j)\\ \\rm assigned} J_{kj}^2

    which is a linear assignment problem, solved **exactly** with the Hungarian
    (Kuhn--Munkres) algorithm via :func:`scipy.optimize.linear_sum_assignment`
    [Kuhn1955]_.  A greedy ``argmax`` is *not* used: greedy picks the largest
    element first and can be forced into a strictly worse total, e.g. for

    .. math:: J^2 = \\begin{pmatrix} 0.81 & 0.64 \\\\ 0.64 & 0.01\\end{pmatrix}

    greedy takes ``(0,0)`` and is then stuck with ``(1,1)``, total ``0.82``,
    while the optimum is the anti-diagonal, total ``1.28``.
    ``test_duschinsky.py::KnownValues::test_match_modes_beats_greedy`` asserts
    exactly this case.

    Degenerate modes are treated as **subspaces, not individuals**
    -----------------------------------------------------------------
    Inside a degenerate manifold the individual eigenvectors returned by a
    diagonaliser are an arbitrary orthogonal mixture: any
    :math:`L \\to L O` with :math:`O` orthogonal and block-local is an equally
    valid set of normal modes.  A one-to-one assignment inside such a block is
    therefore **not physically meaningful**, and this function does not pretend
    otherwise:

    * modes whose frequencies agree within ``degeneracy_tol`` are grouped into
      blocks (:func:`degeneracy_blocks`) for both states;
    * a record whose initial mode or whose matched final mode lies in a block of
      size > 1 has ``degenerate = True`` and ``mode_f = None``;
    * its reported ``overlap`` is the **block-to-block** quantity
      :math:`\\|J[\\text{block}_f, \\text{block}_i]\\|_F` and its
      ``block_singular_values`` are the singular values of the same sub-block.
      Both are invariant under any orthogonal re-mixing inside either block,
      because such a re-mixing acts as
      :math:`J[\\text{block}_f, :] \\to O^{\\mathsf{T}} J[\\text{block}_f, :]`.

    Args:
        dusch_or_J : a :class:`Duschinsky`, or a raw ``(nvib_f, nvib_i)`` array.
        freq_i, freq_f : required (a.u.) when a raw ``J`` is given *and*
            degeneracy detection is wanted.  If omitted with a raw ``J`` every
            mode is treated as non-degenerate and ``freq_shift`` is 0.
        degeneracy_tol : float, window in **atomic units** (Eh).  Default
            :data:`DEGENERACY_TOL` (= ``DEGENERACY_TOL_CM`` = 1 cm^-1).
        verbose, stdout : logging controls.

    Returns:
        :class:`ModeMatching`
    '''
    from scipy.optimize import linear_sum_assignment

    if isinstance(dusch_or_J, Duschinsky):
        J = dusch_or_J.J
        if freq_i is None:
            freq_i = dusch_or_J.freq_i
        if freq_f is None:
            freq_f = dusch_or_J.freq_f
        if stdout is None:
            stdout = dusch_or_J.stdout
        if verbose is None:
            verbose = dusch_or_J.verbose
    else:
        J = numpy.asarray(dusch_or_J, dtype=float)
    if J.ndim != 2:
        raise ValueError('J must be 2-dimensional (nvib_f, nvib_i), got shape %s'
                         % (J.shape,))
    nvib_f, nvib_i = J.shape

    if freq_i is None:
        freq_i = numpy.zeros(nvib_i)
        no_freq = True
    else:
        no_freq = False
        freq_i = numpy.asarray(freq_i, dtype=float).reshape(-1)
    if freq_f is None:
        freq_f = numpy.zeros(nvib_f)
        no_freq = True
    else:
        freq_f = numpy.asarray(freq_f, dtype=float).reshape(-1)
    if freq_i.size != nvib_i or freq_f.size != nvib_f:
        raise ValueError('freq_i/freq_f sizes (%d, %d) are inconsistent with J shape %s'
                         % (freq_i.size, freq_f.size, (J.shape,)))

    if degeneracy_tol is None:
        degeneracy_tol = DEGENERACY_TOL
    if no_freq:
        # no frequency information -> no degeneracy grouping is possible
        blocks_i = [[k] for k in range(nvib_i)]
        blocks_f = [[k] for k in range(nvib_f)]
    else:
        blocks_i = degeneracy_blocks(freq_i, degeneracy_tol)
        blocks_f = degeneracy_blocks(freq_f, degeneracy_tol)
    block_of_i = _block_index_map(blocks_i)
    block_of_f = _block_index_map(blocks_f)

    if min(nvib_f, nvib_i) == 0:
        return ModeMatching([], blocks_i, blocks_f, numpy.zeros((0, 2), dtype=int),
                            0., degeneracy_tol, freq_i, freq_f,
                            verbose=verbose, stdout=stdout)

    w = J**2
    rows, cols = linear_sum_assignment(-w)
    total = float(w[rows, cols].sum())
    partner = dict(zip(cols.tolist(), rows.tolist()))

    matches = []
    for i in range(nvib_i):
        if i not in partner:
            # nvib_f < nvib_i: this initial mode has no partner at all
            continue
        f = partner[i]
        bi = int(block_of_i[i])
        bf = int(block_of_f[f])
        blk_i = blocks_i[bi]
        blk_f = blocks_f[bf]
        degenerate = (len(blk_i) > 1) or (len(blk_f) > 1)
        if degenerate:
            sub = J[numpy.ix_(blk_f, blk_i)]
            overlap = float(numpy.linalg.norm(sub))
            sv = numpy.linalg.svd(sub, compute_uv=False)
            shift = float(freq_f[blk_f].mean() - freq_i[blk_i].mean())
            matches.append(ModeMatch(i, None, f, overlap, shift, True, bi, bf, sv))
        else:
            matches.append(ModeMatch(i, f, f, abs(J[f, i]),
                                     float(freq_f[f] - freq_i[i]), False, bi, bf, None))

    assignment = numpy.column_stack([rows, cols])
    return ModeMatching(matches, blocks_i, blocks_f, assignment, total,
                        degeneracy_tol, freq_i, freq_f,
                        verbose=verbose, stdout=stdout)
