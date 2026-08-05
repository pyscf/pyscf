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
Geometry alignment utilities for :mod:`pyscf.vibronic`.

Two harmonic potential-energy surfaces can only be compared -- and a Duschinsky
matrix can only be defined -- once the two equilibrium structures are expressed
in a *common* molecular frame.  The standard choice is the **Eckart frame**
[Eckart1935]_: the two structures are placed at their respective centres of mass
(the Eckart *translational* condition) and one is rotated onto the other so that

.. math::

    \\sum_a m_a \\, \\mathbf{x}^{\\rm ref}_a \\times \\mathbf{x}_a = 0

(the Eckart *rotational* condition).  For two nearby structures this rotation is
exactly the mass-weighted orthogonal Procrustes / Kabsch rotation
[Kabsch1976]_, [Coutsias2004]_, which this module solves by SVD.

Everything here is pure NumPy and unit-agnostic: masses and coordinates may be
given in any units, provided they are used consistently.  The rest of
:mod:`pyscf.vibronic` calls these routines with masses in electron masses and
coordinates in bohr.

References
----------
.. [Eckart1935] C. Eckart, *Some studies concerning rotating axes and polyatomic
   molecules*, Phys. Rev. **47**, 552 (1935).
.. [Kabsch1976] W. Kabsch, *A solution for the best rotation to relate two sets
   of vectors*, Acta Cryst. **A32**, 922 (1976); ibid. **A34**, 827 (1978).
.. [Coutsias2004] E. A. Coutsias, C. Seok and K. A. Dill, *Using quaternions to
   calculate RMSD*, J. Comput. Chem. **25**, 1849 (2004).
'''

import numpy

__all__ = [
    'center_of_mass', 'shift_to_center_of_mass', 'inertia_tensor',
    'principal_moments', 'rotation_constants', 'classify_rotor',
    'kabsch_rotation', 'align_geometries', 'eckart_frame', 'eckart_residual', 'rmsd',
    'LINEAR_TOL', 'CENTERING_TOL', 'ROTOR_TYPES',
]

#: Default *relative* tolerance used by :func:`classify_rotor` to decide that
#: the smallest principal moment of inertia is negligible.
LINEAR_TOL = 1e-6

#: Default tolerance (in units of the coordinates) used by
#: :func:`kabsch_rotation` to verify that its inputs are centred.
CENTERING_TOL = 1e-8

#: The three rotor classes recognised by :func:`classify_rotor`.
ROTOR_TYPES = ('ATOM', 'LINEAR', 'REGULAR')


def _check_mass_coords(mass, coords):
    mass = numpy.asarray(mass, dtype=float).reshape(-1)
    coords = numpy.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coords must have shape (natm,3), got %s' % (coords.shape,))
    if mass.shape[0] != coords.shape[0]:
        raise ValueError('mass has %d entries but coords has %d atoms'
                         % (mass.shape[0], coords.shape[0]))
    if mass.shape[0] == 0:
        raise ValueError('at least one atom is required')
    if not numpy.all(numpy.isfinite(mass)):
        raise ValueError('mass contains non-finite values')
    if not numpy.all(numpy.isfinite(coords)):
        raise ValueError('coords contains non-finite values')
    if numpy.any(mass <= 0):
        raise ValueError('all masses must be strictly positive')
    return mass, coords


def center_of_mass(mass, coords):
    '''Centre of mass of a structure.

    Args:
        mass : (natm,) array. Any mass unit; the result does not depend on it.
        coords : (natm,3) array of Cartesian coordinates.

    Returns:
        (3,) array, in the units of ``coords``.
    '''
    mass, coords = _check_mass_coords(mass, coords)
    return numpy.einsum('a,ax->x', mass, coords) / mass.sum()


def shift_to_center_of_mass(mass, coords):
    '''Translate a structure so that its centre of mass sits at the origin.

    Returns:
        (natm,3) array of centred coordinates.
    '''
    mass, coords = _check_mass_coords(mass, coords)
    return coords - center_of_mass(mass, coords)


def inertia_tensor(mass, coords):
    '''Inertia tensor about the centre of mass.

    .. math::

        I_{xy} = \\sum_a m_a \\left( \\delta_{xy} |\\mathbf{r}_a|^2
                 - r_{ax} r_{ay} \\right),
        \\qquad \\mathbf{r}_a = \\mathbf{x}_a - \\mathbf{X}_{\\rm com}

    The structure does *not* have to be centred on input; it is centred here.

    Returns:
        (3,3) symmetric array, units ``mass * coords**2``.
    '''
    mass, coords = _check_mass_coords(mass, coords)
    r = coords - center_of_mass(mass, coords)
    im = numpy.einsum('a,ax,ay->xy', mass, r, r)
    return numpy.eye(3) * im.trace() - im


def principal_moments(mass, coords):
    '''Principal moments of inertia, sorted ascending.

    Returns:
        (3,) array ``[I0, I1, I2]`` with ``I0 <= I1 <= I2``.
    '''
    return numpy.sort(numpy.linalg.eigvalsh(inertia_tensor(mass, coords)))


def rotation_constants(mass, coords):
    '''Rotational constants :math:`B_k = 1/(2 I_k)`, sorted **descending**
    (i.e. ``A >= B >= C``, the spectroscopic convention), computed from the
    principal moments about the centre of mass.

    With masses in electron masses and coordinates in bohr this is the
    rotational constant in Hartree (:math:`\\hbar = 1`), so
    ``units.au2wavenumber`` converts it to cm^-1.  No conversion factor is
    hard-coded here; see :mod:`pyscf.vibronic.units`.

    Vanishing moments (an atom, or the figure axis of a linear molecule) give
    ``inf``.  Use :func:`classify_rotor` rather than testing these values
    against a threshold.

    Returns:
        (3,) array, sorted descending.
    '''
    moments = principal_moments(mass, coords)
    with numpy.errstate(divide='ignore'):
        b = 1. / (2 * moments)
    return numpy.sort(b)[::-1].copy()


def classify_rotor(mass, coords, lin_tol=LINEAR_TOL):
    '''Classify a structure as ``'ATOM'``, ``'LINEAR'`` or ``'REGULAR'``.

    The number of vibrational degrees of freedom follows directly:
    0 for ``'ATOM'``, ``3N-5`` for ``'LINEAR'``, ``3N-6`` for ``'REGULAR'``.

    Criterion
    ---------
    * ``natm == 1``           -> ``'ATOM'``.
    * otherwise, let ``I0 <= I1 <= I2`` be the principal moments of inertia
      about the centre of mass.  The structure is ``'LINEAR'`` iff

      .. math::  I_0 \\le \\texttt{lin\\_tol} \\cdot I_2

      i.e. the smallest principal moment is *relatively* negligible.  This is a
      dimensionless, scale-invariant test: it is unchanged by a uniform scaling
      of the coordinates or of the masses, unlike an absolute threshold on a
      rotational constant.
    * everything else is ``'REGULAR'``.

    Resolvability of a near-linear structure
    ----------------------------------------
    For a symmetric triatomic A-B-A of bond length ``d`` bent away from
    linearity by a half-angle ``eps`` (radians), the small moment grows as
    ``I0 ~ eps**2`` while ``I2`` stays finite, so ``I0/I2 = c*eps**2``.  A
    structure is therefore resolved as ``'REGULAR'`` only once
    ``eps >~ sqrt(lin_tol/c)``.  For CO2 at ``d = 2.2`` bohr the measured
    coefficient is ``c = 0.2728``, so with the default ``lin_tol = 1e-6`` the
    boundary sits at ``eps = 1.9e-3 rad = 0.11 degrees``: a 0.057-degree bend is
    still reported ``'LINEAR'``, a 0.5-degree bend is comfortably
    ``'REGULAR'``.  Bends smaller than the boundary are treated as exactly
    linear, which is almost always what is wanted -- their two "bending"
    rotations are numerically indistinguishable from the corresponding rotations
    and projecting all six out would destroy a genuine vibration.  Raise
    ``lin_tol`` to be more aggressive about calling structures linear, lower it
    to resolve finer bends (at the price of an ill-conditioned rotational
    projection; see :func:`pyscf.vibronic.normal_modes.projector_trans_rot`).
    See ``test_alignment.py::KnownValues::test_near_linear_boundary``, which
    pins these numbers down.

    Args:
        mass : (natm,) array, any unit.
        coords : (natm,3) array.
        lin_tol : float, relative tolerance on ``I0/I2``.  Default
            :data:`LINEAR_TOL`.

    Returns:
        One of ``'ATOM'``, ``'LINEAR'``, ``'REGULAR'``.
    '''
    mass, coords = _check_mass_coords(mass, coords)
    if lin_tol < 0:
        raise ValueError('lin_tol must be non-negative')
    natm = coords.shape[0]
    if natm == 1:
        return 'ATOM'
    moments = principal_moments(mass, coords)
    if moments[2] <= 0:
        # every atom sits on the centre of mass -- degenerate, treat as an atom
        return 'ATOM'
    if moments[0] <= lin_tol * moments[2]:
        return 'LINEAR'
    return 'REGULAR'


def _weights(weights, natm):
    if weights is None:
        w = numpy.ones(natm)
    else:
        w = numpy.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != natm:
            raise ValueError('weights has %d entries but there are %d atoms'
                             % (w.shape[0], natm))
        if not numpy.all(numpy.isfinite(w)):
            raise ValueError('weights contains non-finite values')
        if numpy.any(w < 0):
            raise ValueError('weights must be non-negative')
        if w.sum() <= 0:
            raise ValueError('weights must not all be zero')
    return w


def kabsch_rotation(coords_ref, coords, weights=None, centering_tol=CENTERING_TOL):
    '''Weighted Kabsch rotation matrix.

    Returns the proper rotation ``R`` (3,3) that minimises

    .. math::

        \\sum_a w_a \\, \\bigl| (\\mathbf{x}_a^{\\mathsf{T}} R)
                       - \\mathbf{x}^{\\rm ref\\,\\mathsf{T}}_a \\bigr|^2 .

    **Multiplication side:** ``R`` acts on the *right* of a ``(natm,3)`` array
    of row vectors, so the rotated structure is ``coords @ R``.  Equivalently
    ``R.T`` acts on the left of a column vector: ``x_rot = R.T @ x``.

    **Both inputs must already be centred** at their weighted centroid --
    a rotation about the origin cannot repair a translation, and silently
    centring them here would hide caller bugs.  A :class:`ValueError` is raised
    if either weighted centroid exceeds ``centering_tol``.  Use
    :func:`align_geometries` if you want the centring done for you.

    Algorithm: SVD of the weighted covariance matrix
    ``C = coords.T @ diag(w) @ coords_ref``.  With ``C = U S V^T`` the optimum
    is ``R = U V^T``; if ``det(U V^T) < 0`` that would be a *reflection*, so the
    sign of the last singular vector is flipped, ``R = U diag(1,1,-1) V^T``,
    which is the closest **proper** rotation.  Improper rotations are therefore
    never returned: ``det(R) == +1`` always.

    Args:
        coords_ref : (natm,3) reference structure, centred.
        coords : (natm,3) structure to be rotated, centred.
        weights : (natm,) optional non-negative weights.  ``None`` means equal
            weights (plain Kabsch); pass the atomic masses for the
            mass-weighted (Eckart) rotation.
        centering_tol : float, tolerance on the weighted centroids.

    Returns:
        (3,3) rotation matrix with ``det(R) = +1``.
    '''
    coords_ref = numpy.asarray(coords_ref, dtype=float)
    coords = numpy.asarray(coords, dtype=float)
    if coords_ref.ndim != 2 or coords_ref.shape[1] != 3:
        raise ValueError('coords_ref must have shape (natm,3), got %s' % (coords_ref.shape,))
    if coords.shape != coords_ref.shape:
        raise ValueError('coords %s and coords_ref %s must have the same shape'
                         % (coords.shape, coords_ref.shape))
    if not (numpy.all(numpy.isfinite(coords)) and numpy.all(numpy.isfinite(coords_ref))):
        raise ValueError('coordinates contain non-finite values')
    natm = coords.shape[0]
    w = _weights(weights, natm)

    wsum = w.sum()
    for name, x in (('coords_ref', coords_ref), ('coords', coords)):
        centroid = w.dot(x) / wsum
        off = abs(centroid).max()
        if off > centering_tol:
            raise ValueError(
                '%s is not centred at its weighted centroid (max component %.3e > '
                'centering_tol %.3e).  kabsch_rotation() requires pre-centred input; '
                'use shift_to_center_of_mass() or align_geometries().'
                % (name, off, centering_tol))

    cov = numpy.einsum('ax,a,ay->xy', coords, w, coords_ref)
    u, _, vt = numpy.linalg.svd(cov)
    if numpy.linalg.det(u) * numpy.linalg.det(vt) < 0:
        u = u.copy()
        u[:, -1] *= -1
    return u.dot(vt)


def rmsd(coords_ref, coords, weights=None):
    '''Weighted RMSD between two structures *as given* (no alignment).

    .. math::

        {\\rm RMSD} = \\sqrt{\\frac{\\sum_a w_a |\\mathbf{x}_a
                      - \\mathbf{x}^{\\rm ref}_a|^2}{\\sum_a w_a}}
    '''
    coords_ref = numpy.asarray(coords_ref, dtype=float)
    coords = numpy.asarray(coords, dtype=float)
    if coords.shape != coords_ref.shape:
        raise ValueError('coords %s and coords_ref %s must have the same shape'
                         % (coords.shape, coords_ref.shape))
    w = _weights(weights, coords.shape[0])
    d2 = numpy.einsum('a,ax,ax->', w, coords - coords_ref, coords - coords_ref)
    return float(numpy.sqrt(d2 / w.sum()))


def align_geometries(mass, coords_ref, coords, mass_weighted=True):
    '''Put two structures into a common (Eckart) frame.

    Both structures are shifted to their own centre of mass -- this enforces the
    Eckart *translational* condition -- and ``coords`` is then rotated onto
    ``coords_ref`` with the (mass-weighted) Kabsch rotation, which is the
    solution of the Eckart *rotational* condition
    :math:`\\sum_a m_a \\mathbf{x}^{\\rm ref}_a \\times \\mathbf{x}_a = 0`
    [Eckart1935]_, [Kabsch1976]_, [Coutsias2004]_.

    .. warning::

        The equivalence between the Kabsch rotation and the Eckart frame holds
        for the *linearised* Eckart conditions, i.e. for two structures that
        differ by a small displacement.  For large-amplitude distortions the
        Kabsch rotation still minimises the mass-weighted RMSD (and still makes
        :func:`eckart_residual` vanish, since that is its stationarity
        condition), but the resulting frame is only one of possibly several
        stationary points of the Eckart problem and the separation of rotation
        from vibration is no longer accurate.  Check the residual displacement,
        and be aware that for very large rotations the global optimum may
        correspond to a different atom correspondence altogether -- atom
        ordering is *never* permuted here.

    Args:
        mass : (natm,) masses, any unit (used only if ``mass_weighted``).
        coords_ref : (natm,3) reference structure (any origin).
        coords : (natm,3) structure to align (any origin).
        mass_weighted : bool.  ``True`` (default) uses the masses as Kabsch
            weights, which is what the Eckart condition requires.  ``False``
            gives the plain equal-weight Kabsch fit; both structures are still
            centred at their **centre of mass**, not at their geometric
            centroid, so the Eckart translational condition still holds.

    Returns:
        ``(aligned_coords, R)`` where ``aligned_coords`` is
        ``shift_to_center_of_mass(mass, coords) @ R``, a (natm,3) array, and
        ``R`` is the (3,3) proper rotation.  The reference structure itself is
        *not* returned; obtain it with ``shift_to_center_of_mass(mass,
        coords_ref)``.
    '''
    mass, coords = _check_mass_coords(mass, coords)
    _, coords_ref = _check_mass_coords(mass, coords_ref)
    ref_c = coords_ref - center_of_mass(mass, coords_ref)
    cur_c = coords - center_of_mass(mass, coords)
    if mass_weighted:
        rot = kabsch_rotation(ref_c, cur_c, weights=mass)
    else:
        # Equal-weight covariance, but still about the centre of mass: the
        # equal-weight centroid of a COM-centred structure is generally nonzero,
        # so kabsch_rotation()'s centring check would (correctly) reject it.
        rot = _kabsch_uncentred(ref_c, cur_c, numpy.ones(mass.shape[0]))
    return cur_c.dot(rot), rot


def _kabsch_uncentred(coords_ref, coords, w):
    '''Kabsch rotation without the centring check.

    Used by :func:`align_geometries` for ``mass_weighted=False``, where the
    structures are deliberately centred at the *centre of mass* (to satisfy the
    Eckart translational condition) rather than at the equal-weight centroid,
    so the centring assertion of :func:`kabsch_rotation` would not apply.
    '''
    cov = numpy.einsum('ax,a,ay->xy', coords, w, coords_ref)
    u, _, vt = numpy.linalg.svd(cov)
    if numpy.linalg.det(u) * numpy.linalg.det(vt) < 0:
        u = u.copy()
        u[:, -1] *= -1
    return u.dot(vt)


def eckart_frame(mass, coords_ref, coords, mass_weighted=True):
    '''Alias of :func:`align_geometries`, named after the frame it produces.

    Kept because DESIGN.md lists ``alignment.eckart_frame`` in the public
    low-level API.  Returns ``(aligned_coords, R)``.
    '''
    return align_geometries(mass, coords_ref, coords, mass_weighted=mass_weighted)


def eckart_residual(mass, coords_ref, coords):
    '''Residual of the Eckart rotational condition, for diagnostics.

    .. math::

        r = \\left\\| \\sum_a m_a \\,
            \\mathbf{x}^{\\rm ref}_a \\times \\mathbf{x}_a \\right\\|

    Both structures are shifted to their own centre of mass first, so only the
    *rotational* part of the Eckart condition is probed.  The value is
    :math:`O(10^{-14})` relative to ``mass.sum() * |x|**2`` after
    :func:`align_geometries`, and O(1) for a misoriented structure.

    Units are ``mass * coords**2``, so compare it against a scale such as
    ``sum_a m_a |x_a|^2`` rather than against an absolute number.

    Returns:
        float
    '''
    mass, coords = _check_mass_coords(mass, coords)
    _, coords_ref = _check_mass_coords(mass, coords_ref)
    ref_c = coords_ref - center_of_mass(mass, coords_ref)
    cur_c = coords - center_of_mass(mass, coords)
    cross = numpy.cross(ref_c, cur_c)
    return float(numpy.linalg.norm(numpy.einsum('a,ax->x', mass, cross)))
