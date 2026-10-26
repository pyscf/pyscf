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

"""
UHF-CCSD(T) lambda and density intermediates over blocks of the virtual indices

uccsd_t_lambda.make_intermediates, uccsd_t_rdm._gamma1_intermediates and the
(T) part of uccsd_t_rdm._gamma2_intermediates form w, v, d3, rw and wvd as whole
ijkabc arrays for the four spin cases (aaa, bbb, baa, bba), several o**3 v**3
arrays at once. When those do not fit in max_memory, the same terms are
computed here over blocks of the three virtual indices, so a step holds
o**3 B**3 numbers (o**3 v B**2 for the virtual-virtual terms of gamma1, which
pair two blocks along one index). The terms are those of the modules above:
the spin cases' w and v (_cases) and the tables LAMBDA, GAMMA1 and GAMMA2 carry
their einsum strings, operands and factors; _acc cuts each operand on the
block's virtual letters. r6 and the j<->k swap of r4 act on the occupied axes
only and stay inside a block; p6 and the (j,b)<->(k,c) swap of r4 move the
virtual indices, so w and v are evaluated at the permuted blocks (_permuted).
"""

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pyscf import lib
from pyscf.lib import logger

P6 = (
    (0, 1, 2, 3, 4, 5),
    (1, 2, 0, 4, 5, 3),
    (2, 0, 1, 5, 3, 4),
    (0, 2, 1, 3, 5, 4),
    (2, 1, 0, 5, 4, 3),
    (1, 0, 2, 4, 3, 5),
)
SWAP_JK = (0, 2, 1, 3, 4, 5)
SWAP_JK_BC = (0, 2, 1, 3, 5, 4)
IDENTITY = (0, 1, 2, 3, 4, 5)
INTEGRALS = ("ovvv", "OVVV", "ovVV", "OVvv", "ovoo", "OVOO", "ovOO", "OVoo", "ovov", "OVOV", "ovOV")
UNPACKED = {"ovvv": "get_ovvv", "OVVV": "get_OVVV", "ovVV": "get_ovVV", "OVvv": "get_OVvv"}
SIX = ("rw", "rw*", "wvd", "wvd*")
# Arrays of o**3 v**3 the whole-array code holds at once (w, v, d3, rw, wvd and a
# temporary), and of o**3 B**3 a block step holds (w and v at up to six permuted
# blocks, their sums, rw, wvd, d3)
WHOLE_ARRAYS = 6
BLOCK_ARRAYS = 16

Blocks = tuple[slice, slice, slice]
Fn = Callable[[Blocks], np.ndarray]
Term = tuple[str, str, tuple[str, ...], float]


def block_size(mycc, t2ab):
    """None when the whole-array code fits in the free part of max_memory;
    otherwise the largest block size B whose step fits (at least 1)."""
    nocca, noccb, nvira, nvirb = t2ab.shape
    o, v = max(nocca, noccb), max(nvira, nvirb)
    free = max(0, mycc.max_memory - lib.current_memory()[0])
    if WHOLE_ARRAYS * o**3 * v**3 * 8e-6 <= free:
        return None
    for b in range(v, 0, -1):
        if (BLOCK_ARRAYS * b**3 + 2 * v * b**2) * o**3 * 8e-6 <= free:
            return b
    logger.warn(mycc, "UCCSD(T) intermediates: block size 1 exceeds max_memory %d MB", mycc.max_memory)
    return 1


# block algebra
def _ranges(n: int, size: int) -> list[slice]:
    return [slice(s, min(s + size, n)) for s in range(0, n, size)]


def _cut(letters: str, arr: np.ndarray, where: dict[str, slice]) -> np.ndarray:
    return arr[tuple(where.get(ch, slice(None)) for ch in letters)]


def _term(spec: str, x: np.ndarray, y: np.ndarray, blocks: Blocks) -> np.ndarray:
    """One of pyscf's w/v terms 'X,Y->ijkabc', restricted to the virtual blocks."""
    where = dict(zip(spec.split("->")[1][3:], blocks, strict=True))
    lx, ly = spec.split("->")[0].split(",")
    return np.einsum(spec, _cut(lx, x, where), _cut(ly, y, where), optimize=True)


def _acc(out: np.ndarray, spec: str, operands: tuple, where: dict[str, slice], scale: float) -> None:
    """out += scale · einsum(spec, …): the six-index operands are block arrays already; every other operand and the
    output are cut on `where`."""
    ins, lz = spec.split("->")
    cut = [
        x if len(letters) == 6 else _cut(letters, x, where) for letters, x in zip(ins.split(","), operands, strict=True)
    ]
    out[tuple(where.get(ch, slice(None)) for ch in lz)] += scale * np.einsum(spec, *cut, optimize=True)


def _permuted(f: Fn, blocks: Blocks, axes: tuple[int, ...]) -> np.ndarray:
    """(f's whole array).transpose(axes) on the virtual blocks `blocks`: f evaluated where the transpose reads from."""
    need: list[slice] = [slice(None)] * 3
    for n in range(3, 6):
        need[axes[n] - 3] = blocks[n - 3]
    return f(tuple(need)).transpose(axes)


def _cached(f: Fn) -> Fn:
    """f memoised on its blocks, for one block triple's permutations."""
    memo: dict = {}

    def g(b: Blocks) -> np.ndarray:
        key = tuple((s.start, s.stop) for s in b)
        if key not in memo:
            memo[key] = f(b)
        return memo[key]

    return g


def _r6(w: np.ndarray) -> np.ndarray:
    return (
        w
        + w.transpose(2, 0, 1, 3, 4, 5)
        + w.transpose(1, 2, 0, 3, 4, 5)
        - w.transpose(2, 1, 0, 3, 4, 5)
        - w.transpose(0, 2, 1, 3, 4, 5)
        - w.transpose(1, 0, 2, 3, 4, 5)
    )


def _p6(f: Fn, blocks: Blocks) -> np.ndarray:
    return sum(_permuted(f, blocks, ax) for ax in P6)


def _r4(f: Fn, blocks: Blocks) -> np.ndarray:
    """r4(f) on the block: u = f − f(j↔k), then u + u((j,b)↔(k,c))."""

    def u(b: Blocks) -> np.ndarray:
        x = f(b)
        return x - x.transpose(SWAP_JK)

    return u(blocks) + _permuted(u, blocks, SWAP_JK_BC)


def _d3(e: tuple[np.ndarray, np.ndarray, np.ndarray], blocks: Blocks) -> np.ndarray:
    """lib.direct_sum('ia+jb+kc->ijkabc', e0, e1, e2) on the block."""
    e0, e1, e2 = (x[:, b] for x, b in zip(e, blocks, strict=True))
    return e0[:, None, None, :, None, None] + e1[None, :, None, None, :, None] + e2[None, None, :, None, None, :]


# the inputs
@dataclass(frozen=True)
class Inputs:
    """What every (T) quantity reads: amplitudes, orbital-energy differences, Fock blocks, integrals (g) and their
    conjugates (c)."""

    t1a: np.ndarray
    t1b: np.ndarray
    t2aa: np.ndarray
    t2ab: np.ndarray
    t2bb: np.ndarray
    eia: np.ndarray
    eIA: np.ndarray
    fvo: np.ndarray
    fVO: np.ndarray
    g: dict
    c: dict

    def operand(self, name: str) -> np.ndarray:
        """A table operand: 'c.ovvv' the conjugated integral, 'g.ovov' the integral, 't2aa*' a conjugated amplitude,
        'fvo' as named."""
        kind, _, key = name.partition(".")
        if key:
            return {"g": self.g, "c": self.c}[kind][key]
        base = name.removesuffix("*")
        x = {
            "t1a": self.t1a,
            "t1b": self.t1b,
            "t2aa": self.t2aa,
            "t2ab": self.t2ab,
            "t2bb": self.t2bb,
            "fvo": self.fvo,
            "fVO": self.fVO,
        }[base]
        return x.conj() if name.endswith("*") else x


def inputs(t1, t2, eris) -> Inputs:
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    mo_ea, mo_eb = eris.mo_energy
    g = {k: np.asarray(getattr(eris, UNPACKED[k])() if k in UNPACKED else getattr(eris, k)) for k in INTEGRALS}
    return Inputs(
        t1a,
        t1b,
        t2aa,
        t2ab,
        t2bb,
        mo_ea[:nocca, None] - mo_ea[nocca:],
        mo_eb[:noccb, None] - mo_eb[noccb:],
        eris.focka[nocca:, :nocca],
        eris.fockb[noccb:, :noccb],
        g,
        {k: x.conj() for k, x in g.items()},
    )


@dataclass(frozen=True)
class Case:
    """One spin case: the virtual counts and orbital-energy differences of its three positions, pyscf's w and v, same
    spin or mixed."""

    nv: tuple[int, int, int]
    e: tuple[np.ndarray, np.ndarray, np.ndarray]
    w: Fn
    v: Fn
    mixed: bool


def _cases(s: Inputs) -> dict[str, Case]:
    c, t1a, t1b, t2aa, t2ab, t2bb, fvo, fVO = s.c, s.t1a, s.t1b, s.t2aa, s.t2ab, s.t2bb, s.fvo, s.fVO
    _, _, nva, nvb = t2ab.shape

    def w_aaa(b):
        return _term("ijae,kceb->ijkabc", t2aa, c["ovvv"], b) - _term("mkbc,iajm->ijkabc", t2aa, c["ovoo"], b)

    def v_aaa(b):
        return _term("jbkc,ia->ijkabc", c["ovov"], t1a, b) + _term("jkbc,ai->ijkabc", t2aa, fvo, b) * 0.5

    def w_bbb(b):
        return _term("ijae,kceb->ijkabc", t2bb, c["OVVV"], b) - _term("imab,kcjm->ijkabc", t2bb, c["OVOO"], b)

    def v_bbb(b):
        return _term("jbkc,ia->ijkabc", c["OVOV"], t1b, b) + _term("jkbc,ai->ijkabc", t2bb, fVO, b) * 0.5

    def w_baa(b):  # I, A beta; j, k, b, c alpha
        return (
            _term("jIeA,kceb->IjkAbc", t2ab, c["ovvv"], b) * 2
            + _term("jIbE,kcEA->IjkAbc", t2ab, c["ovVV"], b) * 2
            + _term("jkbe,IAec->IjkAbc", t2aa, c["OVvv"], b)
            - _term("mIbA,kcjm->IjkAbc", t2ab, c["ovoo"], b) * 2
            - _term("jMbA,kcIM->IjkAbc", t2ab, c["ovOO"], b) * 2
            - _term("jmbc,IAkm->IjkAbc", t2aa, c["OVoo"], b)
        )

    def v_baa(b):
        return (
            _term("jbkc,IA->IjkAbc", c["ovov"], t1b, b)
            + _term("kcIA,jb->IjkAbc", c["ovOV"], t1a, b) * 2
            + _term("jkbc,AI->IjkAbc", t2aa, fVO, b) * 0.5
            + _term("kIcA,bj->IjkAbc", t2ab, fvo, b) * 2
        )

    def w_bba(b):  # i, a alpha; j, k, b, c beta
        return (
            _term("ijae,kceb->ijkabc", t2ab, c["OVVV"], b) * 2
            + _term("ijeb,kcea->ijkabc", t2ab, c["OVvv"], b) * 2
            + _term("jkbe,iaec->ijkabc", t2bb, c["ovVV"], b)
            - _term("imab,kcjm->ijkabc", t2ab, c["OVOO"], b) * 2
            - _term("mjab,kcim->ijkabc", t2ab, c["OVoo"], b) * 2
            - _term("jmbc,iakm->ijkabc", t2bb, c["ovOO"], b)
        )

    def v_bba(b):
        return (
            _term("jbkc,ia->ijkabc", c["OVOV"], t1a, b)
            + _term("iakc,jb->ijkabc", c["ovOV"], t1b, b) * 2
            + _term("JKBC,ai->iJKaBC", t2bb, fvo, b) * 0.5
            + _term("iKaC,BJ->iJKaBC", t2ab, fVO, b) * 2
        )

    a, b_ = s.eia, s.eIA
    return {
        "aaa": Case((nva,) * 3, (a, a, a), w_aaa, v_aaa, False),
        "bbb": Case((nvb,) * 3, (b_, b_, b_), w_bbb, v_bbb, False),
        "baa": Case((nvb, nva, nva), (b_, a, a), w_baa, v_baa, True),
        "bba": Case((nva, nvb, nvb), (a, b_, b_), w_bba, v_bba, True),
    }


# rw and wvd per block
def _lambda_pair(case: Case, blocks: Blocks) -> tuple[np.ndarray, np.ndarray]:
    """(rw, wvd) of the lambda and of γ2: r6(p6(w))/d3 and r6(p6(2w + v))/d3, or r4(·)/d3 for a mixed case."""
    w, v = _cached(case.w), _cached(case.v)
    d3 = _d3(case.e, blocks)
    if case.mixed:
        return _r4(w, blocks) / d3, _r4(lambda b: w(b) * 2 + v(b), blocks) / d3
    return _r6(_p6(w, blocks)) / d3, _r6(_p6(lambda b: w(b) * 2 + v(b), blocks)) / d3


def _gamma1_pair(case: Case, blocks: Blocks) -> tuple[np.ndarray, np.ndarray]:
    """(rw, wvd) of γ1: p6(r6(w))/d3 and p6(w + v)/d3, or r4(w)/d3 and (w + v)/d3 for a mixed case."""
    w, v = _cached(case.w), _cached(case.v)
    d3 = _d3(case.e, blocks)
    if case.mixed:
        return _r4(w, blocks) / d3, (w(blocks) + v(blocks)) / d3
    return _p6(lambda b: _r6(w(b)), blocks) / d3, _p6(lambda b: w(b) + v(b), blocks) / d3


Pair = Callable[[Case, Blocks], tuple[np.ndarray, np.ndarray]]


def _resolve(name: str, rw: np.ndarray, wvd: np.ndarray, s: Inputs) -> np.ndarray:
    if name in SIX:
        x = rw if name.startswith("rw") else wvd
        return x.conj() if name.endswith("*") else x
    return s.operand(name)


def _six_letters(term: Term) -> list[str]:
    _, spec, names, _ = term
    return [letters for letters, n in zip(spec.split("->")[0].split(","), names, strict=True) if n in SIX]


# the sweeps
def _symmetry(case: Case, pair: Pair) -> tuple[tuple[int, ...], ...] | None:
    """The joint permutations of (ijk, abc) under which rw and wvd of `pair` are symmetric: p6's group for the same spin
    (r6 commutes with it), r4's (j,b)↔(k,c) for a mixed case; None for γ1's mixed cases, whose wvd = (w + v)/d3 is not
    symmetrised."""
    if not case.mixed:
        return P6
    return (IDENTITY, SWAP_JK_BC) if pair is _lambda_pair else None


def _canonical(case: Case, size: int) -> list[Blocks]:
    """One block triple per orbit of the symmetry: A ≤ B ≤ C for the same spin, B ≤ C for a mixed case."""
    if case.nv[1] != case.nv[2]:
        raise ValueError(f"the symmetry swaps the last two virtual positions, which differ in size here: nv {case.nv}")
    first, second = _ranges(case.nv[0], size), _ranges(case.nv[1], size)
    if not case.mixed:
        return [
            (first[i], first[j], first[k])
            for i in range(len(first))
            for j in range(i, len(first))
            for k in range(j, len(first))
        ]
    return [(a, second[j], second[k]) for a in first for j in range(len(second)) for k in range(j, len(second))]


def _orbit(group: tuple[tuple[int, ...], ...], blocks: Blocks) -> list[tuple[Blocks, tuple[int, ...]]]:
    """(T, axes) for each distinct ordering T the group maps `blocks` to: the array on T is the array on `blocks`
    transposed by axes."""
    seen: dict = {}
    for axes in group:
        t = (blocks[axes[3] - 3], blocks[axes[4] - 3], blocks[axes[5] - 3])
        seen.setdefault(tuple((x.start, x.stop) for x in t), (t, axes))
    return list(seen.values())


def _local_sweep(case: Case, pair: Pair, terms: list[Term], s: Inputs, out: dict, size: int) -> None:
    """Every term whose six-index operands share their virtual letters: rw and wvd once per block triple of `_canonical`
    when they are symmetric (`_symmetry`), every other ordering read as their transpose; one pass over all block triples
    otherwise."""
    if not terms:
        return
    group = _symmetry(case, pair)
    triples = _canonical(case, size) if group else itertools.product(*(_ranges(n, size) for n in case.nv))
    for blocks in triples:
        rw, wvd = pair(case, blocks)
        for t, axes in _orbit(group, blocks) if group else [(blocks, IDENTITY)]:
            rw_t, wvd_t = rw.transpose(axes), wvd.transpose(axes)
            for term in terms:
                target, spec, names, scale = term
                where = dict(zip(_six_letters(term)[0][3:], t, strict=True))
                _acc(out[target], spec, tuple(_resolve(n, rw_t, wvd_t, s) for n in names), where, scale)


def _paired_sweep(case: Case, pair: Pair, term: Term, s: Inputs, out: dict, size: int) -> None:
    """A term 'X,Y->ab' whose two six-index operands differ in one virtual position p: for each pair of blocks at the
    other two positions the blocks along p are held (o³·v·B² per operand) and every block pair along p is contracted."""
    target, spec, names, scale = term
    lx, ly = _six_letters(term)
    p = next(k for k in range(3, 6) if lx[k] != ly[k])
    others = [k for k in range(3, 6) if k != p]
    along = _ranges(case.nv[p - 3], size)
    for fixed in itertools.product(*(_ranges(case.nv[k - 3], size) for k in others)):
        held = []
        for blk in along:
            b: list[slice] = [slice(None)] * 3
            b[p - 3] = blk
            for k, sl in zip(others, fixed, strict=True):
                b[k - 3] = sl
            held.append(pair(case, (b[0], b[1], b[2])))
        shared = {lx[k]: sl for k, sl in zip(others, fixed, strict=True)}
        for (bx, (rwx, wvx)), (by, (rwy, wvy)) in itertools.product(zip(along, held, strict=True), repeat=2):
            x, y = _resolve(names[0], rwx, wvx, s), _resolve(names[1], rwy, wvy, s)
            _acc(out[target], spec, (x, y), {lx[p]: bx, ly[p]: by} | shared, scale)


def _run(table: dict[str, list[Term]], pair: Pair, s: Inputs, out: dict, size: int, log) -> None:
    if size < 1:
        raise ValueError(f"block {size}: at least one virtual index per block")
    which = next(k for k, v in TABLES.items() if v is table)
    for name, case in _cases(s).items():
        t0 = time.time()
        six = {term: _six_letters(term) for term in table[name]}
        paired = [t for t, ls in six.items() if len(ls) == 2 and ls[0][3:] != ls[1][3:]]
        _local_sweep(case, pair, [t for t in table[name] if t not in paired], s, out, size)
        for term in paired:
            _paired_sweep(case, pair, term, s, out, size)
        log.debug1("UCCSD(T) %s, %s block: %.0f s (block size %d)", which, name, time.time() - t0, size)


# pyscf's lines
# 'rw*' = rw.conj(), 'c.x' = the conjugated integral, 'g.x' = the integral, 't2aa*' = the conjugated amplitude.
LAMBDA: dict[str, list[Term]] = {
    "aaa": [
        ("l1a", "ijkabc,jbkc->ia", ("rw*", "g.ovov"), 0.25),
        ("pre_aa", "ijkabc,kceb->ijae", ("wvd", "c.ovvv"), 1.0),
        ("pre_aa", "ijkabc,iajm->mkbc", ("wvd", "c.ovoo"), -1.0),
        ("post_aa", "ijkabc,ai->jkbc", ("rw", "fvo"), 1.0),
    ],
    "bbb": [
        ("l1b", "ijkabc,jbkc->ia", ("rw*", "g.OVOV"), 0.25),
        ("pre_bb", "ijkabc,kceb->ijae", ("wvd", "c.OVVV"), 1.0),
        ("pre_bb", "ijkabc,iajm->mkbc", ("wvd", "c.OVOO"), -1.0),
        ("post_bb", "ijkabc,ai->jkbc", ("rw", "fVO"), 1.0),
    ],
    "baa": [
        ("l1a", "ijkabc,kcia->jb", ("rw*", "g.ovOV"), 0.5),
        ("l1b", "ijkabc,jbkc->ia", ("rw*", "g.ovov"), 0.25),
        ("pre_aa", "ijkabc,iaec->jkbe", ("wvd", "c.OVvv"), 1.0),
        ("pre_aa", "ijkabc,iakm->jmbc", ("wvd", "c.OVoo"), -1.0),
        ("post_aa", "ijkabc,ai->jkbc", ("rw", "fVO"), 1.0),
        ("ab", "ijkabc,kceb->jiea", ("wvd", "c.ovvv"), 1.0),
        ("ab", "ijkabc,kcea->jibe", ("wvd", "c.ovVV"), 1.0),
        ("ab", "ijkabc,kcjm->miba", ("wvd", "c.ovoo"), -1.0),
        ("ab", "ijkabc,kcim->jmba", ("wvd", "c.ovOO"), -1.0),
        ("ab", "ijkabc,bj->kica", ("rw", "fvo"), 1.0),
    ],
    "bba": [
        ("l1a", "ijkabc,jbkc->ia", ("rw*", "g.OVOV"), 0.25),
        ("l1b", "ijkabc,iakc->jb", ("rw*", "g.ovOV"), 0.5),
        ("pre_bb", "ijkabc,iaec->jkbe", ("wvd", "c.ovVV"), 1.0),
        ("pre_bb", "ijkabc,iakm->jmbc", ("wvd", "c.ovOO"), -1.0),
        ("post_bb", "ijkabc,ai->jkbc", ("rw", "fvo"), 1.0),
        ("ab", "ijkabc,kceb->ijae", ("wvd", "c.OVVV"), 1.0),
        ("ab", "ijkabc,kcea->ijeb", ("wvd", "c.OVvv"), 1.0),
        ("ab", "ijkabc,kcjm->imab", ("wvd", "c.OVOO"), -1.0),
        ("ab", "ijkabc,kcim->mjab", ("wvd", "c.OVoo"), -1.0),
        ("ab", "ijkabc,bj->ikac", ("rw", "fVO"), 1.0),
    ],
}
GAMMA1: dict[str, list[Term]] = {
    "aaa": [
        ("goo", "iklabc,jklabc->ij", ("wvd*", "rw"), 0.125),
        ("gvv", "ijkacd,ijkbcd->ab", ("wvd", "rw*"), 0.125),
        ("gvo", "jkbc,ijkabc->ai", ("t2aa*", "rw"), 0.125),
    ],
    "bbb": [
        ("gOO", "iklabc,jklabc->ij", ("wvd*", "rw"), 0.125),
        ("gVV", "ijkacd,ijkbcd->ab", ("wvd", "rw*"), 0.125),
        ("gVO", "jkbc,ijkabc->ai", ("t2bb*", "rw"), 0.125),
    ],
    "baa": [
        ("goo", "kilabc,kjlabc->ij", ("wvd*", "rw"), 0.25),
        ("goo", "kliabc,kljabc->ij", ("wvd*", "rw"), 0.25),
        ("gOO", "iklabc,jklabc->ij", ("wvd*", "rw"), 0.25),
        ("gvv", "ijkcad,ijkcbd->ab", ("wvd", "rw*"), 0.25),
        ("gvv", "ijkcda,ijkcdb->ab", ("wvd", "rw*"), 0.25),
        ("gVV", "ijkacd,ijkbcd->ab", ("wvd", "rw*"), 0.25),
        ("gvo", "kica,ijkabc->bj", ("t2ab*", "rw"), 0.5),
        ("gVO", "jkbc,ijkabc->ai", ("t2aa*", "rw"), 0.125),
    ],
    "bba": [
        ("goo", "iklabc,jklabc->ij", ("wvd*", "rw"), 0.25),
        ("gOO", "kilabc,kjlabc->ij", ("wvd*", "rw"), 0.25),
        ("gOO", "kliabc,kljabc->ij", ("wvd*", "rw"), 0.25),
        ("gvv", "ijkacd,ijkbcd->ab", ("wvd", "rw*"), 0.25),
        ("gVV", "ijkcad,ijkcbd->ab", ("wvd", "rw*"), 0.25),
        ("gVV", "ijkcda,ijkcdb->ab", ("wvd", "rw*"), 0.25),
        ("gVO", "ikac,ijkabc->bj", ("t2ab*", "rw"), 0.5),
        ("gvo", "jkbc,ijkabc->ai", ("t2bb*", "rw"), 0.125),
    ],
}
GAMMA2: dict[str, list[Term]] = {
    "aaa": [
        ("dovov", "ia,ijkabc->jbkc", ("t1a", "rw*"), 0.25),
        ("dooov", "mkbc,ijkabc->jmia", ("t2aa", "wvd*"), -0.125),
        ("dovvv", "kjcf,ijkabc->iafb", ("t2aa", "wvd*"), 0.125),
    ],
    "bbb": [
        ("dOVOV", "ia,ijkabc->jbkc", ("t1b", "rw*"), 0.25),
        ("dOOOV", "mkbc,ijkabc->jmia", ("t2bb", "wvd*"), -0.125),
        ("dOVVV", "kjcf,ijkabc->iafb", ("t2bb", "wvd*"), 0.125),
    ],
    "baa": [
        ("dovvv", "jiea,ijkabc->kceb", ("t2ab", "wvd*"), 0.25),
        ("dovVV", "jibe,ijkabc->kcea", ("t2ab", "wvd*"), 0.25),
        ("dOVvv", "jkbe,ijkabc->iaec", ("t2aa", "wvd*"), 0.125),
        ("dooov", "miba,ijkabc->jmkc", ("t2ab", "wvd*"), -0.25),
        ("dOOov", "jmba,ijkabc->imkc", ("t2ab", "wvd*"), -0.25),
        ("dooOV", "jmbc,ijkabc->kmia", ("t2aa", "wvd*"), -0.125),
        ("dovov", "ia,ijkabc->jbkc", ("t1b", "rw*"), 0.25),
        ("dovOV", "jb,ijkabc->kcia", ("t1a", "rw*"), 0.25),
    ],
    "bba": [
        ("dOVVV", "ijae,ijkabc->kceb", ("t2ab", "wvd*"), 0.25),
        ("dOVvv", "ijeb,ijkabc->kcea", ("t2ab", "wvd*"), 0.25),
        ("dovVV", "jkbe,ijkabc->iaec", ("t2bb", "wvd*"), 0.125),
        ("dOOOV", "imab,ijkabc->jmkc", ("t2ab", "wvd*"), -0.25),
        ("dooOV", "mjab,ijkabc->imkc", ("t2ab", "wvd*"), -0.25),
        ("dOOov", "jmbc,ijkabc->kmia", ("t2bb", "wvd*"), -0.125),
        ("dOVOV", "ia,ijkabc->jbkc", ("t1a", "rw*"), 0.25),
        ("dovOV", "jb,ijkabc->iakc", ("t1b", "rw*"), 0.25),
    ],
}


TABLES = {"lambda": LAMBDA, "gamma1": GAMMA1, "gamma2": GAMMA2}


# the entry points
def _eab(e1, e2):
    return lib.direct_sum("ia+jb->ijab", e1, e2)


def lambda_intermediates(mycc, t1, t2, eris, blksize):
    """(l1a_t, l1b_t, l2aa_t, l2ab_t, l2bb_t) of uccsd_t_lambda.make_intermediates."""
    log = logger.new_logger(mycc)
    s = inputs(t1, t2, eris)
    out = {
        "l1a": np.zeros_like(s.t1a),
        "l1b": np.zeros_like(s.t1b),
        "pre_aa": np.zeros_like(s.t2aa),
        "post_aa": np.zeros_like(s.t2aa),
        "pre_bb": np.zeros_like(s.t2bb),
        "post_bb": np.zeros_like(s.t2bb),
        "ab": np.zeros_like(s.t2ab),
    }
    _run(LAMBDA, _lambda_pair, s, out, blksize, log)
    l2aa = (out["pre_aa"] + out["pre_aa"].transpose(1, 0, 3, 2) + out["post_aa"]).conj() / _eab(s.eia, s.eia) * 0.5
    l2bb = (out["pre_bb"] + out["pre_bb"].transpose(1, 0, 3, 2) + out["post_bb"]).conj() / _eab(s.eIA, s.eIA) * 0.5
    l2ab = out["ab"].conj() / _eab(s.eia, s.eIA) * 0.5
    return out["l1a"] / s.eia, out["l1b"] / s.eIA, l2aa, l2ab, l2bb


def gamma1_intermediates(mycc, t1, t2, eris, blksize):
    """(goo, gOO, gvv, gVV, gvo, gVO), the (T) terms of uccsd_t_rdm._gamma1_intermediates."""
    log = logger.new_logger(mycc)
    s = inputs(t1, t2, eris)
    nocca, noccb, nvira, nvirb = s.t2ab.shape
    dt = s.t2ab.dtype
    out = {
        "goo": np.zeros((nocca, nocca), dt),
        "gOO": np.zeros((noccb, noccb), dt),
        "gvv": np.zeros((nvira, nvira), dt),
        "gVV": np.zeros((nvirb, nvirb), dt),
        "gvo": np.zeros((nvira, nocca), dt),
        "gVO": np.zeros((nvirb, noccb), dt),
    }
    _run(GAMMA1, _gamma1_pair, s, out, blksize, log)
    return out["goo"], out["gOO"], out["gvv"], out["gVV"], out["gvo"], out["gVO"]


def add_gamma2(mycc, t1, t2, eris, d2, blksize):
    """Add the (T) terms of uccsd_t_rdm._gamma2_intermediates to d2 in place."""
    log = logger.new_logger(mycc)
    s = inputs(t1, t2, eris)
    (dovov, dovOV, _, dOVOV), _, _, _, _, _, (dovvv, dovVV, dOVvv, dOVVV), (dooov, dooOV, dOOov, dOOOV) = d2
    out = {
        "dovov": dovov,
        "dovOV": dovOV,
        "dOVOV": dOVOV,
        "dovvv": dovvv,
        "dovVV": dovVV,
        "dOVvv": dOVvv,
        "dOVVV": dOVVV,
        "dooov": dooov,
        "dooOV": dooOV,
        "dOOov": dOOov,
        "dOOOV": dOOOV,
    }
    _run(GAMMA2, _lambda_pair, s, out, blksize, log)
