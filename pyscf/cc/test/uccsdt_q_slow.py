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

"""Small-system spin-orbital reference for the conventional CCSDT(Q).

Inputs are real antisymmetric amplitudes T2[ijab], T3[ijkabc], and
antisymmetrized physicists' integrals v[pqrs] = <pq||rs>.  All occupied
spin orbitals precede all virtual spin orbitals.  Outputs with eight
indices have order ijklabcd.  This deliberately forms full T4 tensors.

With Tn = sum(t * excitation)/(n!)**2, the right moment is
P4 ([V_N,T3] + [[V_N,T2],T2]/2)|0>.  ``terms`` separates its five
contraction classes.  Each antisymmetrizer is an unnormalized signed
sum over S4; the seed coefficients include the repeated permutations.
The left moments are P4 V_N T2|0> and P4 V_N T3|0>.  The latter equals
the sum of the two linear-T3 right-moment terms for real integrals.
"""

from itertools import permutations

import numpy as np


_PERMUTATIONS = tuple(
    (p, (-1)**sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)))
    for p in permutations(range(4)))


def _antisymmetrize(seed):
    """Apply the unnormalized occupied and virtual S4 antisymmetrizers."""
    occupied = np.zeros_like(seed)
    for perm, sign in _PERMUTATIONS:
        occupied += sign * seed.transpose(perm + (4, 5, 6, 7))
    result = np.zeros_like(seed)
    for perm, sign in _PERMUTATIONS:
        result += sign * occupied.transpose((0, 1, 2, 3) +
                                            tuple(i + 4 for i in perm))
    return result


def terms(v, t2, t3):
    """Return fully antisymmetric voov/oooo/vvvv/vvov/vooo and l2 terms.

    The three quadratic-T2 seeds have 144, 36, and 36 distinct signed
    occupied/virtual placements out of (4!)**2 = 576.  The two T3 seeds
    each have 24 placements.  The disconnected v*T2 left moment has
    36 placements.  Thus their coefficients before full
    antisymmetrization are -1/4, 1/16, 1/16, 1/24, -1/24, and 1/16.
    """
    nocc = t2.shape[0]
    o, a = slice(None, nocc), slice(nocc, None)
    einsum = np.einsum
    result = {}
    result['voov'] = _antisymmetrize(-0.25 * einsum(
        'amie,mkbc,jled->ijklabcd', v[a, o, o, a], t2, t2,
        optimize=True))
    result['oooo'] = _antisymmetrize(0.0625 * einsum(
        'mnij,mlad,nkbc->ijklabcd', v[o, o, o, o], t2, t2,
        optimize=True))
    result['vvvv'] = _antisymmetrize(0.0625 * einsum(
        'abef,jkfc,iled->ijklabcd', v[a, a, a, a], t2, t2,
        optimize=True))
    result['vvov'] = _antisymmetrize(einsum(
        'cdke,ijlabe->ijklabcd', v[a, a, o, a], t3,
        optimize=True) / 24.0)
    result['vooo'] = _antisymmetrize(-einsum(
        'cmkl,ijmabd->ijklabcd', v[a, o, o, o], t3,
        optimize=True) / 24.0)
    result['l2'] = _antisymmetrize(0.0625 * einsum(
        'abij,klcd->ijklabcd', v[a, a, o, o], t2,
        optimize=True))
    return result


def kernel(v, t2, t3, eps, return_details=False):
    """Return [Q] and complete (Q) corrections, optionally with all terms.

    ``eps`` uses the same occupied-first order as ``v``.  The denominator
    is the occupied energy sum minus the virtual energy sum.  As in the
    RCCSDT(Q) reference, denominator magnitudes <= 1e-15 contribute zero.
    With ``return_details=True``, return (e_bracket, e_paren, details),
    adding w4, l3, denominator, t4, and e_extra to the ``terms`` mapping.
    """
    details = terms(v, t2, t3)
    w4 = sum(details[key] for key in ('voov', 'oooo', 'vvvv', 'vvov', 'vooo'))
    l3 = details['vvov'] + details['vooo']
    nocc = t2.shape[0]
    eo, ev = np.asarray(eps[:nocc]), np.asarray(eps[nocc:])
    denominator = np.zeros(w4.shape)
    for axis in range(8):
        shape = [1] * 8
        shape[axis] = w4.shape[axis]
        denominator += (eo if axis < 4 else -ev).reshape(shape)
    t4 = np.zeros_like(w4)
    np.divide(w4, denominator, out=t4, where=np.abs(denominator) > 1e-15)
    e_bracket = np.einsum('ijklabcd,ijklabcd->', details['l2'], t4) / 576.0
    e_extra = np.einsum('ijklabcd,ijklabcd->', l3, t4) / 576.0
    e_paren = e_bracket + e_extra
    if return_details:
        details.update(w4=w4, l3=l3, denominator=denominator, t4=t4,
                       e_extra=e_extra)
        return e_bracket, e_paren, details
    return e_bracket, e_paren
