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

"""Independent determinant-space BCH oracle for CCSDTQ equation validation.

The routines in this module are deliberately independent of the production
CC residual contractions.  They evaluate exp(-T) H_N exp(T)|HF> exactly in a
small fixed-(Nalpha,Nbeta) Fock space and are intended for guarded oracle
checks and tiny-system diagnostics.  It is not a production solver backend.
"""

from itertools import combinations, permutations
from math import comb, factorial
import numpy as np
from scipy import sparse

# int.bit_count is available only on Python 3.10 and newer.
_bit_count = getattr(int, 'bit_count', lambda bits: bin(bits).count('1'))


def _parity(order):
    return (-1) ** sum(order[i] > order[j]
                       for i in range(len(order)) for j in range(i + 1, len(order)))


def _apply(bits, holes, particles):
    """Apply a†_p1 ... a†_pn a_hn ... a_h1 with bit-count signs."""
    sign = 1
    for i in holes:
        mask = 1 << i
        if not bits & mask:
            return None
        if _bit_count(bits & (mask - 1)) % 2:
            sign = -sign
        bits ^= mask
    for a in reversed(particles):
        mask = 1 << a
        if bits & mask:
            return None
        if _bit_count(bits & (mask - 1)) % 2:
            sign = -sign
        bits |= mask
    return bits, sign


def _operator_matrix(terms, determinants):
    index = {bits: i for i, bits in enumerate(determinants)}
    rows, cols, data = [], [], []
    for coefficient, holes, particles in terms:
        if coefficient == 0:
            continue
        for j, bits in enumerate(determinants):
            result = _apply(bits, holes, particles)
            if result is not None and result[0] in index:
                rows.append(index[result[0]])
                cols.append(j)
                data.append(coefficient * result[1])
    shape = (len(determinants),) * 2
    return sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()


def _canonical_tensor_element(tensor, holes, particles):
    """Return the antisymmetric canonical coefficient of one Tn tensor.

    The production code stores antisymmetric tensors.  Averaging all signed
    permutations here also gives the correct coefficient when a test passes
    an arbitrary (not yet antisymmetrized) tensor: the operator definition is
    ``sum(t * E) / (n!)**2``.
    """
    rank = len(holes)
    if rank <= 1:
        return tensor[holes + particles]
    value = 0.0
    for hp in permutations(range(rank)):
        hs = _parity(hp)
        h = tuple(holes[i] for i in hp)
        for pp in permutations(range(rank)):
            value += hs * _parity(pp) * tensor[h + tuple(particles[i] for i in pp)]
    return value / float(factorial(rank) ** 2)


def _normal_ordered_hamiltonian(f, v, determinants, nocc):
    """Build a normal-ordered one- plus two-body Hamiltonian matrix.

    ``f`` is a full spin-orbital Fock matrix and ``v[p,q,r,s]`` is the real
    antisymmetrized physicists' integral ``<pq||rs>``.  The returned sparse
    matrix has zero reference expectation for the determinant occupying
    orbitals ``range(nocc)``.  This helper deliberately uses the explicit
    normal-ordering contractions rather than a precomputed CC intermediate.
    """
    f = np.asarray(f)
    v = np.asarray(v)
    nmo = f.shape[0]
    if f.shape != (nmo, nmo) or v.shape != (nmo,) * 4:
        raise ValueError('f must be square and v must have shape (nmo,)*4')
    terms = []
    # One-body normal ordering: f_pq {a^p_q} = f_pq a^p_q
    # - Tr_occ(f) * I.  The scalar is added after constructing the matrix.
    for p in range(nmo):
        for q in range(nmo):
            if f[p, q] != 0:
                terms.append((f[p, q], (q,), (p,)))

    pairs = tuple(combinations(range(nmo), 2))
    # 1/4 * sum_{pqrs} v_pqrs a^p a^q a_s a_r.  For an antisymmetric v,
    # one representative of each ordered pair has unit coefficient.
    for p, q in pairs:
        for r, s in pairs:
            value = v[p, q, r, s]
            if value != 0:
                terms.append((value, (r, s), (p, q)))

    # Two-body normal ordering contributes -sum_i <p i||q i> to the
    # one-body part and +1/2 sum_ij <ij||ij> to the scalar.
    w = np.einsum('piqi->pq', v[:, :nocc, :, :nocc], optimize=True)
    for p in range(nmo):
        for q in range(nmo):
            if w[p, q] != 0:
                terms.append((-w[p, q], (q,), (p,)))
    e2 = 0.5 * sum(v[i, j, i, j]
                   for i in range(nocc) for j in range(nocc))
    matrix = _operator_matrix(terms, determinants)
    matrix.setdiag(matrix.diagonal() + e2 - np.trace(f[:nocc, :nocc]))
    return matrix


def _cluster_matrix(tamps, determinants, nocc, nmo):
    """Build the excitation matrix for full T1/T2/T3/T4 tensors."""
    if len(tamps) != 4:
        raise ValueError('tamps must be a 4-tuple of full T1, T2, T3, T4 tensors')
    nvir = nmo - nocc
    terms = []
    for rank, tensor in enumerate(tamps, 1):
        tensor = np.asarray(tensor)
        expected = (nocc,) * rank + (nvir,) * rank
        if tensor.shape != expected:
            raise ValueError('T%d has shape %s; expected %s' %
                             (rank, tensor.shape, expected))
        if rank > nocc or rank > nvir:
            continue
        for holes in combinations(range(nocc), rank):
            for particles in combinations(range(nvir), rank):
                value = _canonical_tensor_element(tensor, holes, particles)
                if value != 0:
                    terms.append((value, holes,
                                  tuple(nocc + a for a in particles)))
    return _operator_matrix(terms, determinants)


def _exp_apply(operator, vector, order, sign=1.0):
    """Apply the finite exp(sign * operator) series exactly."""
    result = np.asarray(vector, dtype=float).copy()
    term = result.copy()
    factorial = 1.0
    for power in range(1, order + 1):
        term = operator.dot(term)
        factorial *= power
        result += (sign ** power) * term / factorial
    return result


def _residual_tensor(vector, determinants, determinant_index, reference_bits,
                     nocc, nmo, rank):
    """Extract a fully antisymmetric rank-n residual tensor from coefficients."""
    nvir = nmo - nocc
    result = np.zeros((nocc,) * rank + (nvir,) * rank, dtype=float)
    if rank > nocc or rank > nvir:
        return result
    for holes in combinations(range(nocc), rank):
        for particles in combinations(range(nvir), rank):
            applied = _apply(reference_bits, holes,
                             tuple(nocc + a for a in particles))
            if applied is None:
                continue
            bits, sign = applied
            # A spin-nonconserving trial tensor connects outside the fixed
            # (Nalpha,Nbeta) Fock space and is intentionally projected out.
            if bits not in determinant_index:
                continue
            value = vector[determinant_index[bits]] / sign
            for hp in permutations(range(rank)):
                for pp in permutations(range(rank)):
                    idx = tuple(holes[i] for i in hp)
                    idx += tuple(particles[i] for i in pp)
                    result[idx] = _parity(hp) * _parity(pp) * value
    return result


def determinant_basis(orbspin, nocc):
    """Return the complete fixed-(Nalpha,Nbeta) determinant list.

    Occupied spin orbitals precede virtual ones.  The small-space guard is
    applied before constructing the list; this module must not be used as
    an accidentally exponential backend on production-sized molecules.
    """
    orbspin = np.asarray(orbspin)
    nmo = len(orbspin)
    if nocc < 0 or nocc > nmo or orbspin.shape != (nmo,):
        raise ValueError('invalid nocc/orbspin for BCH oracle')
    if not np.isin(orbspin, (0, 1)).all():
        raise ValueError('orbspin labels must be zero or one')
    alpha = np.flatnonzero(orbspin == 0).tolist()
    beta = np.flatnonzero(orbspin == 1).tolist()
    nelec = (sum(i < nocc for i in alpha), sum(i < nocc for i in beta))
    size = comb(len(alpha), nelec[0]) * comb(len(beta), nelec[1])
    if size > 2000:
        raise ValueError('BCH determinant oracle is limited to 2000 determinants')
    return [sum(1 << i for i in a + b)
            for a in combinations(alpha, nelec[0])
            for b in combinations(beta, nelec[1])]


def bch_residuals(f, v, tamps, orbspin, nocc, return_details=False):
    """Exact determinant-space BCH residuals through rank four.

    Parameters
    ----------
    f : (nmo,nmo) array
        Full spin-orbital Fock matrix.  It may be noncanonical and need not
        be diagonal.
    v : (nmo,nmo,nmo,nmo) array
        Real antisymmetrized physicists' integrals ``<pq||rs>``.
    tamps : tuple of four arrays
        Full T1, T2, T3 and T4 amplitudes with shapes
        ``(nocc,)*n + (nvir,)*n``.  Antisymmetric amplitudes are expected;
        signed permutation averaging makes arbitrary test tensors well-defined.
    orbspin : length-nmo integer array
        Spin label (0/1) of each spin orbital.  All occupied orbitals must
        precede virtual orbitals, matching the existing test oracle layout.
    nocc : int
        Number of occupied spin orbitals in the reference determinant.

    Returns
    -------
    energy, residuals : float, tuple
        ``energy`` is ``<HF| exp(-T) H_N exp(T) |HF>`` and residuals are the
        full antisymmetric tensors ``(R1,R2,R3,R4)``.
    """
    f = np.asarray(f)
    v = np.asarray(v)
    orbspin = np.asarray(orbspin)
    nmo = f.shape[0]
    if orbspin.shape != (nmo,):
        raise ValueError('invalid nocc/orbspin for BCH oracle')
    if any(np.iscomplexobj(x) for x in (f, v, *tamps)):
        raise NotImplementedError('BCH oracle supports real orbitals and amplitudes')
    determinants = determinant_basis(orbspin, nocc)
    size = len(determinants)
    determinant_index = {bits: i for i, bits in enumerate(determinants)}
    reference_bits = (1 << nocc) - 1
    ref = np.zeros(size)
    ref[determinant_index[reference_bits]] = 1.0
    h_normal = _normal_ordered_hamiltonian(f, v, determinants, nocc)
    cluster = _cluster_matrix(tamps, determinants, nocc, nmo)
    ket = _exp_apply(cluster, ref, nocc, sign=1.0)
    similarity = _exp_apply(cluster, h_normal.dot(ket), nocc, sign=-1.0)
    residuals = tuple(_residual_tensor(similarity, determinants, determinant_index,
                                       reference_bits, nocc, nmo, rank)
                      for rank in range(1, 5))
    energy = float(similarity[determinant_index[reference_bits]])
    if not return_details:
        return energy, residuals
    return energy, residuals, dict(determinants=determinants,
                                   determinant_index=determinant_index,
                                   reference_bits=reference_bits,
                                   h_normal=h_normal,
                                   cluster=cluster,
                                   exp_t=ket,
                                   similarity=similarity)


def spinorbital_inputs(mycc, eris=None, tamps=None):
    """Convert native full/compact amplitudes without production converters.

    This is a test-only adapter.  It forms full occupied-first spin-orbital
    arrays, so callers must limit it to very small orbital spaces.
    """
    if eris is None:
        eris = mycc.ao2mo()
    if tamps is None:
        tamps = mycc.tamps
    oa, ob = mycc.nocc
    ma, mb = mycc.nmo
    va, vb = ma - oa, mb - ob
    no, nv = oa + ob, va + vb
    alpha = np.r_[np.arange(oa), no + np.arange(va)]
    beta = np.r_[oa + np.arange(ob), no + va + np.arange(vb)]
    spin = np.empty(no + nv, dtype=int)
    spin[alpha], spin[beta] = 0, 1
    f = np.zeros((no + nv,) * 2)
    f[np.ix_(alpha, alpha)] = eris.fock[0]
    f[np.ix_(beta, beta)] = eris.fock[1]
    bare = np.zeros((no + nv,) * 4)
    bare[np.ix_(alpha, alpha, alpha, alpha)] = eris.pppp
    bare[np.ix_(beta, beta, beta, beta)] = eris.PPPP
    bare[np.ix_(alpha, beta, alpha, beta)] = eris.pPpP
    bare[np.ix_(beta, alpha, beta, alpha)] = eris.pPpP.transpose(1, 0, 3, 2)
    full = []
    for rank, blocks in enumerate(tamps, 1):
        tensor = np.zeros((no,) * rank + (nv,) * rank)
        perms = tuple((p, _parity(p)) for p in permutations(range(rank)))
        for holes in combinations(range(no), rank):
            ha = tuple(i for i in holes if i < oa)
            hb = tuple(i - oa for i in holes if i >= oa)
            for particles in combinations(range(nv), rank):
                pa = tuple(a for a in particles if a < va)
                pb = tuple(a - va for a in particles if a >= va)
                if len(ha) != len(pa):
                    continue
                block = blocks[rank - len(ha)]
                if len(ha) >= len(hb):
                    groups = (ha, pa, hb, pb)
                    dims = (oa, va, ob, vb)
                else:
                    groups = (hb, pb, ha, pa)
                    dims = (ob, vb, oa, va)
                active = [(g, n) for g, n in zip(groups, dims) if g]
                compact_shape = tuple(comb(n, len(g)) for g, n in active)
                if block.shape == compact_shape:
                    idx = tuple(list(combinations(range(n), len(g))).index(g)
                                for g, n in active)
                else:
                    idx = sum(groups, ())
                value = block[idx]
                for hp, hs in perms:
                    for pp, ps in perms:
                        idx = tuple(holes[i] for i in hp)
                        idx += tuple(particles[a] for a in pp)
                        tensor[idx] = hs * ps * value
        full.append(tensor)
    return dict(f=f, v=bare - bare.transpose(0, 1, 3, 2),
                tamps=tuple(full), orbspin=spin, nocc=no)


def kernel(mycc, eris=None, tamps=None, return_details=False):
    """Return exact BCH correlation energy and full residual tensors."""
    return bch_residuals(**spinorbital_inputs(mycc, eris, tamps),
                         return_details=return_details)
