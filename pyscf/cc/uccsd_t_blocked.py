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
UHF-CCSD(T) lambda and density intermediates over blocks of the virtual indices

uccsd_t_lambda.make_intermediates, uccsd_t_rdm._gamma1_intermediates and the
(T) part of uccsd_t_rdm._gamma2_intermediates form w, v, d3, rw and wvd as whole
ijkabc arrays for the four spin cases (aaa, bbb, baa, bba), several o**3 v**3
arrays at once. When those do not fit in max_memory, the same terms are
computed here over blocks of the three virtual indices, so that a step holds
o**3 B**3 numbers (o**3 v B**2 for the virtual-virtual terms of gamma1, which
pair two blocks along one index). The whole-array code is left unchanged and
still runs whenever it fits; the blocked terms are tested against it.

w and v of each spin case are written out in _cases with the einsum strings of
the whole-array code. The contractions of rw and wvd are listed per spin case
in _lambda_terms, _gamma1_terms and _gamma2_terms, one line of the whole-array
code per entry: (target, einsum string, operands, factor), where the operands
'rw', 'wvd' and their conjugates 'rw*', 'wvd*' stand for the block arrays and
every other operand is the array itself. _acc slices each operand on the
virtual indices of the block.

r6 and the j<->k swap of r4 act on the occupied axes only and stay inside a
block; p6 and the (j,b)<->(k,c) swap of r4 move the virtual indices, so w and v
are evaluated at the permuted blocks (_permuted). Where rw and wvd are
symmetric under these permutations (the lambda and gamma2 terms, gamma1's
same-spin cases) they are formed once per unordered block triple and the other
orderings are their transposes (_symmetry, _orbit).
'''

import itertools
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.ccsd import BLKMIN

# p6 as joint transposes of (i,j,k,a,b,c)
_P6 = ((0, 1, 2, 3, 4, 5),
       (1, 2, 0, 4, 5, 3),
       (2, 0, 1, 5, 3, 4),
       (0, 2, 1, 3, 5, 4),
       (2, 1, 0, 5, 4, 3),
       (1, 0, 2, 4, 3, 5))
_SWAP_JK = (0, 2, 1, 3, 4, 5)
_SWAP_JK_BC = (0, 2, 1, 3, 5, 4)
_IDENTITY = (0, 1, 2, 3, 4, 5)
# Arrays held at once: of o**3 v**3 by the whole-array code; of o**3 B**3 by a
# block step and of o**3 v B**2 by gamma1's paired terms (measured peaks of
# this code, with a margin).
_WHOLE_ARRAYS = 6
_BLOCK_ARRAYS = 16
_PAIRED_ARRAYS = 10


def _step_mb(nocc, nvir, blksize):
    return (_BLOCK_ARRAYS * blksize**3 + _PAIRED_ARRAYS * nvir * blksize**2) * nocc**3 * 8e-6


def block_size(mycc, t2ab):
    '''Block size of the virtual indices for the UCCSD(T) lambda and density
    intermediates.

    Args:
        mycc : UCCSD object; its max_memory (MB) is the bound
        t2ab : the alpha-beta T2 amplitudes, shape (nocca, noccb, nvira, nvirb)

    Returns:
        None when the whole-array code of uccsd_t_lambda and uccsd_t_rdm fits
        in the free part of max_memory; otherwise the largest number of virtual
        orbitals per block whose step fits, but at least ccsd.BLKMIN.
    '''
    nocca, noccb, nvira, nvirb = t2ab.shape
    o, v = max(nocca, noccb), max(nvira, nvirb)
    # both paths hold the four o*v**3 integral blocks unpacked
    free = max(0, mycc.max_memory - lib.current_memory()[0] - 4 * o * v**3 * 8e-6)
    if _WHOLE_ARRAYS * o**3 * v**3 * 8e-6 <= free:
        return None
    blkmin = min(v, BLKMIN)
    blksize = v
    while blksize > blkmin and _step_mb(o, v, blksize) > free:
        blksize -= 1
    if _step_mb(o, v, blksize) > free:
        logger.warn(mycc, 'UCCSD(T) intermediates: blocks of %d virtual orbitals need %.0f MB, '
                    'more than the %.0f MB free of max_memory', blksize, _step_mb(o, v, blksize), free)
    logger.debug1(mycc, 'UCCSD(T) intermediates over blocks of %d of %d virtual orbitals', blksize, v)
    return blksize


def _ranges(n, size):
    return [slice(p0, p1) for p0, p1 in lib.prange(0, n, size)]


def _cut(letters, arr, where):
    return arr[tuple(where.get(ch, slice(None)) for ch in letters)]


def _term(spec, x, y, blocks):
    '''One of the w/v terms 'X,Y->ijkabc', restricted to the virtual blocks.'''
    where = dict(zip(spec.split('->')[1][3:], blocks))
    lx, ly = spec.split('->')[0].split(',')
    return numpy.einsum(spec, _cut(lx, x, where), _cut(ly, y, where), optimize=True)


def _acc(out, spec, operands, where, scale):
    '''out += scale * einsum(spec, *operands): the six-index operands are block
    arrays already; every other operand and the output are sliced on where.'''
    ins, lz = spec.split('->')
    cut = [x if len(letters) == 6 else _cut(letters, x, where) for letters, x in zip(ins.split(','), operands)]
    out[tuple(where.get(ch, slice(None)) for ch in lz)] += scale * numpy.einsum(spec, *cut, optimize=True)


def _permuted(f, blocks, axes):
    '''(f's whole array).transpose(axes) on the virtual blocks: f evaluated
    where the transpose reads from.'''
    need = [slice(None)] * 3
    for n in range(3, 6):
        need[axes[n] - 3] = blocks[n - 3]
    return f(tuple(need)).transpose(axes)


def _cached(f):
    '''f memoised on its blocks, for one block triple's permutations.'''
    memo = {}

    def g(b):
        key = tuple((s.start, s.stop) for s in b)
        if key not in memo:
            memo[key] = f(b)
        return memo[key]
    return g


def _r6(w):
    return (w + w.transpose(2, 0, 1, 3, 4, 5) + w.transpose(1, 2, 0, 3, 4, 5)
            - w.transpose(2, 1, 0, 3, 4, 5) - w.transpose(0, 2, 1, 3, 4, 5) - w.transpose(1, 0, 2, 3, 4, 5))


def _p6(f, blocks):
    return sum(_permuted(f, blocks, ax) for ax in _P6)


def _r4(f, blocks):
    '''r4(f) on the block: u = f - f(j<->k), then u + u((j,b)<->(k,c)).'''
    def u(b):
        x = f(b)
        return x - x.transpose(_SWAP_JK)
    return u(blocks) + _permuted(u, blocks, _SWAP_JK_BC)


def _d3(e, blocks):
    return lib.direct_sum('ia+jb+kc->ijkabc', e[0][:, blocks[0]], e[1][:, blocks[1]], e[2][:, blocks[2]])


class _Inputs:
    '''What every (T) quantity reads: amplitudes, orbital-energy differences,
    Fock blocks, and the conjugated integrals.'''
    def __init__(self, t1, t2, eris):
        self.t1a, self.t1b = t1
        self.t2aa, self.t2ab, self.t2bb = t2
        nocca, noccb = self.t2ab.shape[:2]
        mo_ea, mo_eb = eris.mo_energy
        self.eia = mo_ea[:nocca, None] - mo_ea[nocca:]
        self.eIA = mo_eb[:noccb, None] - mo_eb[noccb:]
        self.fvo = eris.focka[nocca:, :nocca]
        self.fVO = eris.fockb[noccb:, :noccb]
        self.ovov = numpy.asarray(eris.ovov)
        self.OVOV = numpy.asarray(eris.OVOV)
        self.ovOV = numpy.asarray(eris.ovOV)
        # the conjugated integrals that w, v and the contractions read
        self.c = {'ovvv': numpy.asarray(eris.get_ovvv()).conj(),
                  'OVVV': numpy.asarray(eris.get_OVVV()).conj(),
                  'ovVV': numpy.asarray(eris.get_ovVV()).conj(),
                  'OVvv': numpy.asarray(eris.get_OVvv()).conj(),
                  'ovoo': numpy.asarray(eris.ovoo).conj(),
                  'OVOO': numpy.asarray(eris.OVOO).conj(),
                  'ovOO': numpy.asarray(eris.ovOO).conj(),
                  'OVoo': numpy.asarray(eris.OVoo).conj(),
                  'ovov': self.ovov.conj(),
                  'OVOV': self.OVOV.conj(),
                  'ovOV': self.ovOV.conj()}


class _Case:
    '''One spin case: the virtual counts and orbital-energy differences of its
    three positions, its w and v as functions of the blocks, same spin or mixed.'''
    def __init__(self, nv, e, w, v, mixed):
        self.nv = nv
        self.e = e
        self.w = w
        self.v = v
        self.mixed = mixed


def _cases(s):
    c, t1a, t1b, t2aa, t2ab, t2bb, fvo, fVO = s.c, s.t1a, s.t1b, s.t2aa, s.t2ab, s.t2bb, s.fvo, s.fVO
    nva, nvb = t2ab.shape[2:]

    def w_aaa(b):
        return _term('ijae,kceb->ijkabc', t2aa, c['ovvv'], b) - _term('mkbc,iajm->ijkabc', t2aa, c['ovoo'], b)

    def v_aaa(b):
        return _term('jbkc,ia->ijkabc', c['ovov'], t1a, b) + _term('jkbc,ai->ijkabc', t2aa, fvo, b) * .5

    def w_bbb(b):
        return _term('ijae,kceb->ijkabc', t2bb, c['OVVV'], b) - _term('imab,kcjm->ijkabc', t2bb, c['OVOO'], b)

    def v_bbb(b):
        return _term('jbkc,ia->ijkabc', c['OVOV'], t1b, b) + _term('jkbc,ai->ijkabc', t2bb, fVO, b) * .5

    def w_baa(b):  # I, A beta; j, k, b, c alpha
        return (_term('jIeA,kceb->IjkAbc', t2ab, c['ovvv'], b) * 2
                + _term('jIbE,kcEA->IjkAbc', t2ab, c['ovVV'], b) * 2
                + _term('jkbe,IAec->IjkAbc', t2aa, c['OVvv'], b)
                - _term('mIbA,kcjm->IjkAbc', t2ab, c['ovoo'], b) * 2
                - _term('jMbA,kcIM->IjkAbc', t2ab, c['ovOO'], b) * 2
                - _term('jmbc,IAkm->IjkAbc', t2aa, c['OVoo'], b))

    def v_baa(b):
        return (_term('jbkc,IA->IjkAbc', c['ovov'], t1b, b)
                + _term('kcIA,jb->IjkAbc', c['ovOV'], t1a, b) * 2
                + _term('jkbc,AI->IjkAbc', t2aa, fVO, b) * .5
                + _term('kIcA,bj->IjkAbc', t2ab, fvo, b) * 2)

    def w_bba(b):  # i, a alpha; j, k, b, c beta
        return (_term('ijae,kceb->ijkabc', t2ab, c['OVVV'], b) * 2
                + _term('ijeb,kcea->ijkabc', t2ab, c['OVvv'], b) * 2
                + _term('jkbe,iaec->ijkabc', t2bb, c['ovVV'], b)
                - _term('imab,kcjm->ijkabc', t2ab, c['OVOO'], b) * 2
                - _term('mjab,kcim->ijkabc', t2ab, c['OVoo'], b) * 2
                - _term('jmbc,iakm->ijkabc', t2bb, c['ovOO'], b))

    def v_bba(b):
        return (_term('jbkc,ia->ijkabc', c['OVOV'], t1a, b)
                + _term('iakc,jb->ijkabc', c['ovOV'], t1b, b) * 2
                + _term('JKBC,ai->iJKaBC', t2bb, fvo, b) * .5
                + _term('iKaC,BJ->iJKaBC', t2ab, fVO, b) * 2)

    ea, eb = s.eia, s.eIA
    return {'aaa': _Case((nva,) * 3, (ea, ea, ea), w_aaa, v_aaa, False),
            'bbb': _Case((nvb,) * 3, (eb, eb, eb), w_bbb, v_bbb, False),
            'baa': _Case((nvb, nva, nva), (eb, ea, ea), w_baa, v_baa, True),
            'bba': _Case((nva, nvb, nvb), (ea, eb, eb), w_bba, v_bba, True)}


def _lambda_pair(case, blocks):
    '''(rw, wvd) of the lambda and of gamma2: r6(p6(w))/d3 and
    r6(p6(2w + v))/d3, or r4(.)/d3 for a mixed case.'''
    w, v = _cached(case.w), _cached(case.v)
    d3 = _d3(case.e, blocks)
    if case.mixed:
        return _r4(w, blocks) / d3, _r4(lambda b: w(b) * 2 + v(b), blocks) / d3
    return _r6(_p6(w, blocks)) / d3, _r6(_p6(lambda b: w(b) * 2 + v(b), blocks)) / d3


def _gamma1_pair(case, blocks):
    '''(rw, wvd) of gamma1: p6(r6(w))/d3 and p6(w + v)/d3, or r4(w)/d3 and
    (w + v)/d3 for a mixed case.'''
    w, v = _cached(case.w), _cached(case.v)
    d3 = _d3(case.e, blocks)
    if case.mixed:
        return _r4(w, blocks) / d3, (w(blocks) + v(blocks)) / d3
    return _p6(lambda b: _r6(w(b)), blocks) / d3, _p6(lambda b: w(b) + v(b), blocks) / d3


def _resolve(operand, rw, wvd):
    '''The block array that 'rw', 'rw*', 'wvd' or 'wvd*' stands for; any other
    operand is the array itself.'''
    if not isinstance(operand, str):
        return operand
    x = rw if operand.startswith('rw') else wvd
    return x.conj() if operand.endswith('*') else x


def _block_letters(term):
    '''The index letters of the term's six-index operands.'''
    _, spec, operands, _ = term
    return [letters for letters, x in zip(spec.split('->')[0].split(','), operands) if isinstance(x, str)]


def _symmetry(case, mixed_symmetric):
    '''The joint permutations of (ijk, abc) under which rw and wvd are
    symmetric: p6's group for the same spin (r6 commutes with it), r4's
    (j,b)<->(k,c) for a mixed case when mixed_symmetric; None for gamma1's
    mixed cases, whose wvd = (w + v)/d3 is not symmetrised.'''
    if not case.mixed:
        return _P6
    return (_IDENTITY, _SWAP_JK_BC) if mixed_symmetric else None


def _canonical(case, size):
    '''One block triple per orbit of the symmetry: A <= B <= C for the same
    spin, B <= C for a mixed case.'''
    first, second = _ranges(case.nv[0], size), _ranges(case.nv[1], size)
    if not case.mixed:
        n = len(first)
        return [(first[i], first[j], first[k]) for i in range(n) for j in range(i, n) for k in range(j, n)]
    n = len(second)
    return [(a, second[j], second[k]) for a in first for j in range(n) for k in range(j, n)]


def _orbit(group, blocks):
    '''(T, axes) for each distinct ordering T the group maps blocks to: the
    array on T is the array on blocks transposed by axes.'''
    seen = {}
    for axes in group:
        t = (blocks[axes[3] - 3], blocks[axes[4] - 3], blocks[axes[5] - 3])
        seen.setdefault(tuple((x.start, x.stop) for x in t), (t, axes))
    return list(seen.values())


def _local_sweep(case, pair, group, terms, out, size):
    '''Every term whose six-index operands share their virtual letters: rw
    and wvd once per block triple of _canonical when they are symmetric
    (group), every other ordering read as their transpose; one pass over all
    block triples otherwise.'''
    if group:
        triples = _canonical(case, size)
    else:
        triples = itertools.product(*(_ranges(n, size) for n in case.nv))
    for blocks in triples:
        rw, wvd = pair(case, blocks)
        if group:
            orderings = _orbit(group, blocks)
        else:
            orderings = [(blocks, _IDENTITY)]
        for t, axes in orderings:
            rw_t, wvd_t = rw.transpose(axes), wvd.transpose(axes)
            for term in terms:
                target, spec, operands, scale = term
                where = dict(zip(_block_letters(term)[0][3:], t))
                _acc(out[target], spec, [_resolve(x, rw_t, wvd_t) for x in operands], where, scale)


def _paired_sweep(case, pair, term, out, size):
    '''A term 'X,Y->ab' whose two six-index operands differ in one virtual
    position p: for each pair of blocks at the other two positions the blocks
    along p are held (o**3 v B**2 per operand) and every block pair along p is
    contracted.'''
    target, spec, operands, scale = term
    lx, ly = _block_letters(term)
    p = next(k for k in range(3, 6) if lx[k] != ly[k])
    others = [k for k in range(3, 6) if k != p]
    along = _ranges(case.nv[p - 3], size)
    for fixed in itertools.product(*(_ranges(case.nv[k - 3], size) for k in others)):
        held = []
        for blk in along:
            b = [slice(None)] * 3
            b[p - 3] = blk
            for k, sl in zip(others, fixed):
                b[k - 3] = sl
            held.append(pair(case, (b[0], b[1], b[2])))
        shared = {lx[k]: sl for k, sl in zip(others, fixed)}
        for (bx, (rwx, wvx)), (by, (rwy, wvy)) in itertools.product(zip(along, held), repeat=2):
            x, y = _resolve(operands[0], rwx, wvx), _resolve(operands[1], rwy, wvy)
            _acc(out[target], spec, (x, y), {lx[p]: bx, ly[p]: by, **shared}, scale)


def _run(mycc, s, terms, pair, mixed_symmetric, out, blksize):
    '''Add every term of every spin case to out, over blocks of blksize
    virtual orbitals.'''
    log = logger.new_logger(mycc)
    cpu1 = (logger.process_clock(), logger.perf_counter())
    for name, case in _cases(s).items():
        local, paired = [], []
        for term in terms[name]:
            letters = _block_letters(term)
            if len(letters) == 2 and letters[0][3:] != letters[1][3:]:
                paired.append(term)
            else:
                local.append(term)
        _local_sweep(case, pair, _symmetry(case, mixed_symmetric), local, out, blksize)
        for term in paired:
            _paired_sweep(case, pair, term, out, blksize)
        cpu1 = log.timer_debug1('UCCSD(T) intermediates, %s block (block size %d)' % (name, blksize), *cpu1)


# The contractions of rw and wvd in uccsd_t_lambda.make_intermediates,
# uccsd_t_rdm._gamma1_intermediates and uccsd_t_rdm._gamma2_intermediates,
# per spin case: (target, einsum string, operands, factor).
def _lambda_terms(s):
    c, fvo, fVO = s.c, s.fvo, s.fVO
    return {
        'aaa': [('l1a', 'ijkabc,jbkc->ia', ('rw*', s.ovov), .25),
                ('pre_aa', 'ijkabc,kceb->ijae', ('wvd', c['ovvv']), 1.),
                ('pre_aa', 'ijkabc,iajm->mkbc', ('wvd', c['ovoo']), -1.),
                ('post_aa', 'ijkabc,ai->jkbc', ('rw', fvo), 1.)],
        'bbb': [('l1b', 'ijkabc,jbkc->ia', ('rw*', s.OVOV), .25),
                ('pre_bb', 'ijkabc,kceb->ijae', ('wvd', c['OVVV']), 1.),
                ('pre_bb', 'ijkabc,iajm->mkbc', ('wvd', c['OVOO']), -1.),
                ('post_bb', 'ijkabc,ai->jkbc', ('rw', fVO), 1.)],
        'baa': [('l1a', 'ijkabc,kcia->jb', ('rw*', s.ovOV), .5),
                ('l1b', 'ijkabc,jbkc->ia', ('rw*', s.ovov), .25),
                ('pre_aa', 'ijkabc,iaec->jkbe', ('wvd', c['OVvv']), 1.),
                ('pre_aa', 'ijkabc,iakm->jmbc', ('wvd', c['OVoo']), -1.),
                ('post_aa', 'ijkabc,ai->jkbc', ('rw', fVO), 1.),
                ('ab', 'ijkabc,kceb->jiea', ('wvd', c['ovvv']), 1.),
                ('ab', 'ijkabc,kcea->jibe', ('wvd', c['ovVV']), 1.),
                ('ab', 'ijkabc,kcjm->miba', ('wvd', c['ovoo']), -1.),
                ('ab', 'ijkabc,kcim->jmba', ('wvd', c['ovOO']), -1.),
                ('ab', 'ijkabc,bj->kica', ('rw', fvo), 1.)],
        'bba': [('l1a', 'ijkabc,jbkc->ia', ('rw*', s.OVOV), .25),
                ('l1b', 'ijkabc,iakc->jb', ('rw*', s.ovOV), .5),
                ('pre_bb', 'ijkabc,iaec->jkbe', ('wvd', c['ovVV']), 1.),
                ('pre_bb', 'ijkabc,iakm->jmbc', ('wvd', c['ovOO']), -1.),
                ('post_bb', 'ijkabc,ai->jkbc', ('rw', fvo), 1.),
                ('ab', 'ijkabc,kceb->ijae', ('wvd', c['OVVV']), 1.),
                ('ab', 'ijkabc,kcea->ijeb', ('wvd', c['OVvv']), 1.),
                ('ab', 'ijkabc,kcjm->imab', ('wvd', c['OVOO']), -1.),
                ('ab', 'ijkabc,kcim->mjab', ('wvd', c['OVoo']), -1.),
                ('ab', 'ijkabc,bj->ikac', ('rw', fVO), 1.)],
    }


def _gamma1_terms(s):
    t2aa_conj, t2ab_conj, t2bb_conj = s.t2aa.conj(), s.t2ab.conj(), s.t2bb.conj()
    return {
        'aaa': [('goo', 'iklabc,jklabc->ij', ('wvd*', 'rw'), .125),
                ('gvv', 'ijkacd,ijkbcd->ab', ('wvd', 'rw*'), .125),
                ('gvo', 'jkbc,ijkabc->ai', (t2aa_conj, 'rw'), .125)],
        'bbb': [('gOO', 'iklabc,jklabc->ij', ('wvd*', 'rw'), .125),
                ('gVV', 'ijkacd,ijkbcd->ab', ('wvd', 'rw*'), .125),
                ('gVO', 'jkbc,ijkabc->ai', (t2bb_conj, 'rw'), .125)],
        'baa': [('goo', 'kilabc,kjlabc->ij', ('wvd*', 'rw'), .25),
                ('goo', 'kliabc,kljabc->ij', ('wvd*', 'rw'), .25),
                ('gOO', 'iklabc,jklabc->ij', ('wvd*', 'rw'), .25),
                ('gvv', 'ijkcad,ijkcbd->ab', ('wvd', 'rw*'), .25),
                ('gvv', 'ijkcda,ijkcdb->ab', ('wvd', 'rw*'), .25),
                ('gVV', 'ijkacd,ijkbcd->ab', ('wvd', 'rw*'), .25),
                ('gvo', 'kica,ijkabc->bj', (t2ab_conj, 'rw'), .5),
                ('gVO', 'jkbc,ijkabc->ai', (t2aa_conj, 'rw'), .125)],
        'bba': [('goo', 'iklabc,jklabc->ij', ('wvd*', 'rw'), .25),
                ('gOO', 'kilabc,kjlabc->ij', ('wvd*', 'rw'), .25),
                ('gOO', 'kliabc,kljabc->ij', ('wvd*', 'rw'), .25),
                ('gvv', 'ijkacd,ijkbcd->ab', ('wvd', 'rw*'), .25),
                ('gVV', 'ijkcad,ijkcbd->ab', ('wvd', 'rw*'), .25),
                ('gVV', 'ijkcda,ijkcdb->ab', ('wvd', 'rw*'), .25),
                ('gVO', 'ikac,ijkabc->bj', (t2ab_conj, 'rw'), .5),
                ('gvo', 'jkbc,ijkabc->ai', (t2bb_conj, 'rw'), .125)],
    }


def _gamma2_terms(s):
    t1a, t1b, t2aa, t2ab, t2bb = s.t1a, s.t1b, s.t2aa, s.t2ab, s.t2bb
    return {
        'aaa': [('dovov', 'ia,ijkabc->jbkc', (t1a, 'rw*'), .25),
                ('dooov', 'mkbc,ijkabc->jmia', (t2aa, 'wvd*'), -.125),
                ('dovvv', 'kjcf,ijkabc->iafb', (t2aa, 'wvd*'), .125)],
        'bbb': [('dOVOV', 'ia,ijkabc->jbkc', (t1b, 'rw*'), .25),
                ('dOOOV', 'mkbc,ijkabc->jmia', (t2bb, 'wvd*'), -.125),
                ('dOVVV', 'kjcf,ijkabc->iafb', (t2bb, 'wvd*'), .125)],
        'baa': [('dovvv', 'jiea,ijkabc->kceb', (t2ab, 'wvd*'), .25),
                ('dovVV', 'jibe,ijkabc->kcea', (t2ab, 'wvd*'), .25),
                ('dOVvv', 'jkbe,ijkabc->iaec', (t2aa, 'wvd*'), .125),
                ('dooov', 'miba,ijkabc->jmkc', (t2ab, 'wvd*'), -.25),
                ('dOOov', 'jmba,ijkabc->imkc', (t2ab, 'wvd*'), -.25),
                ('dooOV', 'jmbc,ijkabc->kmia', (t2aa, 'wvd*'), -.125),
                ('dovov', 'ia,ijkabc->jbkc', (t1b, 'rw*'), .25),
                ('dovOV', 'jb,ijkabc->kcia', (t1a, 'rw*'), .25)],
        'bba': [('dOVVV', 'ijae,ijkabc->kceb', (t2ab, 'wvd*'), .25),
                ('dOVvv', 'ijeb,ijkabc->kcea', (t2ab, 'wvd*'), .25),
                ('dovVV', 'jkbe,ijkabc->iaec', (t2bb, 'wvd*'), .125),
                ('dOOOV', 'imab,ijkabc->jmkc', (t2ab, 'wvd*'), -.25),
                ('dooOV', 'mjab,ijkabc->imkc', (t2ab, 'wvd*'), -.25),
                ('dOOov', 'jmbc,ijkabc->kmia', (t2bb, 'wvd*'), -.125),
                ('dOVOV', 'ia,ijkabc->jbkc', (t1a, 'rw*'), .25),
                ('dovOV', 'jb,ijkabc->iakc', (t1b, 'rw*'), .25)],
    }


def lambda_intermediates(mycc, t1, t2, eris, blksize):
    '''The (T) terms of uccsd_t_lambda.make_intermediates over blocks of the
    virtual indices.

    Args:
        mycc : UCCSD object
        t1, t2 : UCCSD amplitudes, (t1a, t1b) and (t2aa, t2ab, t2bb)
        eris : UCCSD integrals (uccsd._ChemistsERIs)
        blksize : number of virtual orbitals per block (see block_size)

    Returns:
        l1a_t, l1b_t, l2aa_t, l2ab_t, l2bb_t, with the shapes of t1 and t2
    '''
    s = _Inputs(t1, t2, eris)
    out = {'l1a': numpy.zeros_like(s.t1a), 'l1b': numpy.zeros_like(s.t1b),
           'pre_aa': numpy.zeros_like(s.t2aa), 'post_aa': numpy.zeros_like(s.t2aa),
           'pre_bb': numpy.zeros_like(s.t2bb), 'post_bb': numpy.zeros_like(s.t2bb),
           'ab': numpy.zeros_like(s.t2ab)}
    _run(mycc, s, _lambda_terms(s), _lambda_pair, True, out, blksize)
    eiajb = lib.direct_sum('ia+jb->ijab', s.eia, s.eia)
    eIAJB = lib.direct_sum('ia+jb->ijab', s.eIA, s.eIA)
    eiaJB = lib.direct_sum('ia+jb->ijab', s.eia, s.eIA)
    l2aa = (out['pre_aa'] + out['pre_aa'].transpose(1, 0, 3, 2) + out['post_aa']).conj() / eiajb * .5
    l2bb = (out['pre_bb'] + out['pre_bb'].transpose(1, 0, 3, 2) + out['post_bb']).conj() / eIAJB * .5
    l2ab = out['ab'].conj() / eiaJB * .5
    return out['l1a'] / s.eia, out['l1b'] / s.eIA, l2aa, l2ab, l2bb


def gamma1_intermediates(mycc, t1, t2, eris, blksize):
    '''The (T) terms of uccsd_t_rdm._gamma1_intermediates over blocks of the
    virtual indices.

    Args: as for lambda_intermediates.

    Returns:
        goo, gOO, gvv, gVV, gvo, gVO: (nocca, nocca), (noccb, noccb),
        (nvira, nvira), (nvirb, nvirb), (nvira, nocca), (nvirb, noccb)
    '''
    s = _Inputs(t1, t2, eris)
    nocca, noccb, nvira, nvirb = s.t2ab.shape
    dt = s.t2ab.dtype
    out = {'goo': numpy.zeros((nocca, nocca), dt), 'gOO': numpy.zeros((noccb, noccb), dt),
           'gvv': numpy.zeros((nvira, nvira), dt), 'gVV': numpy.zeros((nvirb, nvirb), dt),
           'gvo': numpy.zeros((nvira, nocca), dt), 'gVO': numpy.zeros((nvirb, noccb), dt)}
    _run(mycc, s, _gamma1_terms(s), _gamma1_pair, False, out, blksize)
    return out['goo'], out['gOO'], out['gvv'], out['gVV'], out['gvo'], out['gVO']


def add_gamma2(mycc, t1, t2, eris, d2, blksize):
    '''Add the (T) terms of uccsd_t_rdm._gamma2_intermediates to the UCCSD
    gamma2 intermediates d2 in place, over blocks of the virtual indices.

    Args: as for lambda_intermediates, and d2 from uccsd_rdm._gamma2_intermediates
    or uccsd_rdm._gamma2_outcore (only its ovov, ovvv and ooov blocks change).

    Returns:
        d2
    '''
    s = _Inputs(t1, t2, eris)
    dovov, dovOV, dOVov, dOVOV = d2[0]
    dovvv, dovVV, dOVvv, dOVVV = d2[6]
    dooov, dooOV, dOOov, dOOOV = d2[7]
    out = {'dovov': dovov, 'dovOV': dovOV, 'dOVOV': dOVOV,
           'dovvv': dovvv, 'dovVV': dovVV, 'dOVvv': dOVvv, 'dOVVV': dOVVV,
           'dooov': dooov, 'dooOV': dooOV, 'dOOov': dOOov, 'dOOOV': dOOOV}
    _run(mycc, s, _gamma2_terms(s), _lambda_pair, True, out, blksize)
    return d2
