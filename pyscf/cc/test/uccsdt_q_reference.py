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

"""Small-system UCCSDT(Q) oracle using fermion operators on determinants.

This test helper deliberately shares no contractions, permutation helpers, or
integral conversion code with the production correction.  Spin orbitals are
ordered occupied alpha, occupied beta, virtual alpha, virtual beta.  ``v`` is
the antisymmetrized physicists' integral <pq||rs>; amplitudes have occupied
indices first and virtual indices last.  Independent (strictly increasing)
hole and particle combinations have unit weight, since the two factorials in
T_n cancel the antisymmetric index permutations.
"""

from itertools import combinations, permutations
from math import comb

import numpy as np
from scipy import sparse

# int.bit_count is available only on Python 3.10 and newer.
_bit_count = getattr(int, 'bit_count', lambda bits: bin(bits).count('1'))


def _parity(order):
    return (-1) ** sum(order[i] > order[j]
                       for i in range(len(order)) for j in range(i + 1, len(order)))


def _amplitude(blocks, holes, particles, nocca, nvira):
    """Read one spin-allowed independent amplitude, full or compact."""
    rank = len(holes)
    nh = sum(i < nocca for i in holes)
    if nh != sum(a < nvira for a in particles):
        return 0.0
    ha = tuple(i for i in holes if i < nocca)
    hb = tuple(i - nocca for i in holes if i >= nocca)
    pa = tuple(a for a in particles if a < nvira)
    pb = tuple(a - nvira for a in particles if a >= nvira)
    if rank == 2:
        if nh == 2:
            return blocks[0][ha + pa]
        if nh == 1:
            return blocks[1][ha[0], pa[0], hb[0], pb[0]]
        return blocks[2][hb + pb]

    # In bba the alpha index is moved past two beta indices on each side;
    # both permutations are even.  No extra sign is introduced.
    block = blocks[3 - nh]
    h, p, other_h, other_p = (ha, pa, hb, pb) if nh >= 2 else (hb, pb, ha, pa)
    if block.ndim == 6:
        return block[h + p + other_h + other_p]
    if len(h) == 3:
        # Lexicographic combination indices match PySCF's compact storage.
        no = nocca if nh == 3 else blocks[1].shape[2]
        nv = nvira if nh == 3 else blocks[1].shape[3]
    else:
        # The opposite-spin dimensions of the other mixed block supply the
        # same-spin space sizes without depending on a CC object's maps.
        other = blocks[2] if nh == 2 else blocks[1]
        no, nv = other.shape[2:4]
    ih = list(combinations(range(no), len(h))).index(h)
    ip = list(combinations(range(nv), len(p))).index(p)
    return block[(ih, ip) + other_h + other_p]


def spinorbital_inputs(mycc, eris=None, tamps=None):
    """Return independent full spin-orbital inputs for tiny reference tests."""
    if eris is None:
        eris = mycc.ao2mo()
    if tamps is None:
        tamps = mycc.tamps
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    no, nv = nocca + noccb, nvira + nvirb
    alpha = np.r_[np.arange(nocca), no + np.arange(nvira)]
    beta = np.r_[nocca + np.arange(noccb), no + nvira + np.arange(nvirb)]
    orbspin = np.empty(no + nv, dtype=int)
    orbspin[alpha], orbspin[beta] = 0, 1
    bare = np.zeros((no + nv,) * 4)
    bare[np.ix_(alpha, alpha, alpha, alpha)] = eris.pppp
    bare[np.ix_(beta, beta, beta, beta)] = eris.PPPP
    bare[np.ix_(alpha, beta, alpha, beta)] = eris.pPpP
    bare[np.ix_(beta, alpha, beta, alpha)] = eris.pPpP.transpose(1, 0, 3, 2)
    v = bare - bare.transpose(0, 1, 3, 2)
    full = []
    for rank in (2, 3):
        tensor = np.zeros((no,) * rank + (nv,) * rank)
        perms = [(p, _parity(p)) for p in permutations(range(rank))]
        for holes in combinations(range(no), rank):
            for particles in combinations(range(nv), rank):
                value = _amplitude(tamps[rank - 1], holes, particles, nocca, nvira)
                if value == 0:
                    continue
                for hp, hs in perms:
                    for pp, ps in perms:
                        idx = tuple(holes[i] for i in hp) + tuple(particles[a] for a in pp)
                        tensor[idx] = hs * ps * value
        full.append(tensor)
    ea, eb = eris.mo_energy
    energies = np.r_[ea[:nocca], eb[:noccb], ea[nocca:], eb[noccb:]]
    return dict(t2=full[0], t3=full[1], v=v, mo_energy=energies,
                nocc=no, orbspin=orbspin)


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


def determinant_kernel(v, t2, t3, mo_energy, orbspin, nocc,
                       return_details=False):
    """Evaluate the defining operators in a fixed-Nalpha/Nbeta Fock space.

    W4 = P4 ([V_N,T3] + 1/2 [[V_N,T2],T2]) |0>.
    E[Q] = <T2 0|V_N|W4/D4>; Eextra = <T3 0|V_N|W4/D4>.
    The 2000-determinant limit protects against accidental benchmark-sized
    input to this intentionally exponential, test-only implementation.
    """
    nmo = len(mo_energy)
    spin_orbitals = [np.flatnonzero(np.asarray(orbspin) == s).tolist() for s in (0, 1)]
    nelec = [sum(i < nocc for i in block) for block in spin_orbitals]
    size = comb(len(spin_orbitals[0]), nelec[0]) * comb(len(spin_orbitals[1]), nelec[1])
    if size > 2000:
        raise ValueError('Determinant oracle is limited to 2000 determinants')
    determinants = [sum(1 << i for i in a + b)
                    for a in combinations(spin_orbitals[0], nelec[0])
                    for b in combinations(spin_orbitals[1], nelec[1])]
    reference_bits = (1 << nocc) - 1
    ref = np.zeros(size)
    ref[determinants.index(reference_bits)] = 1
    pairs = list(combinations(range(nmo), 2))
    terms = [(v[p, q, r, s], (r, s), (p, q))
             for p, q in pairs for r, s in pairs]
    w = np.einsum('piqi->pq', v[:, :nocc, :, :nocc])
    e2 = 0.5 * sum(v[i, j, i, j] for i in range(nocc) for j in range(nocc))
    terms += [(-w[p, q], (q,), (p,)) for p in range(nmo) for q in range(nmo)]
    vn = _operator_matrix(terms, determinants) + e2 * sparse.eye(size, format='csr')
    cluster = []
    for rank, tensor in ((2, t2), (3, t3)):
        terms = [(tensor[holes + particles], holes,
                  tuple(nocc + a for a in particles))
                 for holes in combinations(range(nocc), rank)
                 for particles in combinations(range(nmo - nocc), rank)]
        cluster.append(_operator_matrix(terms, determinants))
    c2, c3 = cluster
    t2ref, t3ref = c2 @ ref, c3 @ ref
    driven3 = vn @ t3ref - c3 @ (vn @ ref)
    driven22 = 0.5 * (vn @ (c2 @ t2ref) - 2 * c2 @ (vn @ t2ref)
                      + c2 @ (c2 @ (vn @ ref)))
    ranks = np.array([_bit_count(reference_bits & ~bits) for bits in determinants])
    driven3[ranks != 4] = 0
    driven22[ranks != 4] = 0
    denominator = np.zeros(size)
    q = np.zeros(size)
    for i, bits in enumerate(determinants):
        if ranks[i] == 4:
            holes = [p for p in range(nocc) if not bits & (1 << p)]
            particles = [p for p in range(nocc, nmo) if bits & (1 << p)]
            denominator[i] = sum(mo_energy[holes]) - sum(mo_energy[particles])
            if abs(denominator[i]) > 1e-15:
                q[i] = (driven3[i] + driven22[i]) / denominator[i]
    left2, left3 = vn.T @ t2ref, vn.T @ t3ref
    bracket, extra = float(left2 @ q), float(left3 @ q)
    if not return_details:
        return bracket, bracket + extra
    details = dict(determinants=determinants, ranks=ranks, w_t3=driven3,
                   w_t2t2=driven22, denominator=denominator, q=q,
                   left2=left2, left3=left3, e_extra=extra,
                   reference_vn=float(ref @ (vn @ ref)))
    return bracket, bracket + extra, details


def kernel(mycc, eris=None, tamps=None, return_details=False):
    """Return (E[Q], E(Q)); optionally append determinant-level details."""
    return determinant_kernel(**spinorbital_inputs(mycc, eris, tamps),
                              return_details=return_details)
