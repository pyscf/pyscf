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

# Author: Yuhao Chen <chenyuhao@shu.edu.cn>

'''Conventional UHF-CCSDT[Q] and UHF-CCSDT(Q) for canonical real orbitals.

The second returned correction includes [Q].  Integrals are the original
physicists' ERIs, not the T1-dressed intermediates of the CCSDT iterations.
The five connected right-moment seeds below represent
P4 (V_N T3 + (V_N T2**2 / 2)_C)|0>.  Their occupied and virtual indices
are separately antisymmetrized before selecting spin blocks.  No full T4
or full spin-orbital T3 is constructed by this module.

UCCSDT.ccsdt_q(tamps=None, eris=None) returns ([Q], (Q)) without changing
the CCSDT amplitudes, integrals, e_corr, or e_tot.  Add only the second value
to the CCSDT energy for UCCSDT(Q).  Both compact and full native T3 layouts
and the existing frozen-orbital specifications are supported.  Inputs must
be real double precision and the UHF Fock matrices must be canonical.
Reuse the original AO-to-MO integrals through eris to avoid another
transformation.  The method has no dedicated density-fitted, out-of-core,
or GPU correction algorithm.  See examples/cc/67-uccsdt_q.py for usage.

Y. J. Bomble et al., J. Chem. Phys. 123, 054101 (2005).
DOI: 10.1063/1.1950567.
'''

import functools
import itertools
import math
import ctypes
from fractions import Fraction

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.rccsdt import _einsum


# Labels i-l/a-d are external occupied/virtual indices; m,n/e,f are summed.
# Coefficients precede the two *unnormalized* S4 antisymmetrizers.  The
# stabilizers have orders 4, 16, 16, 24, 24, and 16, respectively.
_SEEDS = (
    ('voov', Fraction(-1, 4), (('g', 'amie'), ('t2', 'mkbc'), ('t2', 'jled'))),
    ('oooo', Fraction(1, 16), (('g', 'mnij'), ('t2', 'mlad'), ('t2', 'nkbc'))),
    ('vvvv', Fraction(1, 16), (('g', 'abef'), ('t2', 'jkfc'), ('t2', 'iled'))),
    ('vvov', Fraction(1, 24), (('g', 'cdke'), ('t3', 'ijlabe'))),
    ('vooo', Fraction(-1, 24), (('g', 'cmkl'), ('t3', 'ijmabd'))),
    ('l2', Fraction(1, 16), (('g', 'abij'), ('t2', 'klcd'))),
)
_OCCUPIED = 'ijklmn'
_LABELS = 'ijklabcdmnef'
_CATEGORIES = tuple(row[0] for row in _SEEDS)


def _parity(order):
    return (-1)**sum(x > y for i, x in enumerate(order) for y in order[i+1:])


def _canonical_factors(factors):
    '''Collect equal contractions using intrinsic tensor antisymmetries.'''
    sign, result = 1, []
    for kind, labels in factors:
        rank = 2 if kind == 'g' else int(kind[-1])
        left, right = labels[:rank], labels[rank:]
        sign *= _parity(left) * _parity(right)
        left, right = ''.join(sorted(left)), ''.join(sorted(right))
        if kind == 'g' and right < left:  # real <pq||rs> = <rs||pq>
            left, right = right, left
        result.append((kind, left + right))
    return tuple(sorted(result)), sign


@functools.lru_cache(None)
def _expanded_terms():
    '''Expand the six seeds once, combining exact rational coefficients.

    Dummy-index renaming and exchange of identical T2 factors remove the
    repeated permutations.  Keeping the seed name makes every contraction
    traceable to its formula and allows independent term-by-term tests.
    '''
    result = {}
    permutations = tuple(itertools.permutations(range(4)))
    for name, coefficient, factors in _SEEDS:
        labels = ''.join(s for _, s in factors)
        od = sorted(set(labels) & set('mn'))
        vd = sorted(set(labels) & set('ef'))
        dummy_maps = [dict(zip(od + vd, op + vp))
                      for op in itertools.permutations(od)
                      for vp in itertools.permutations(vd)]
        for op in permutations:
            for vp in permutations:
                mapping = dict(zip('ijklabcd', ''.join('ijkl'[i] for i in op) +
                                   ''.join('abcd'[i] for i in vp)))
                candidates = []
                for dm in dummy_maps:
                    mapped = [(kind, ''.join(mapping.get(x, dm.get(x, x)) for x in s))
                              for kind, s in factors]
                    candidates.append(_canonical_factors(mapped))
                key, sign = min(candidates)
                key = name, key
                value = coefficient * _parity(op) * _parity(vp) * sign
                result[key] = result.get(key, 0) + value
    return tuple((name, float(c), factors) for (name, factors), c in result.items() if c)


@functools.lru_cache(None)
def _spin_terms(nalpha):
    '''Select the five spin sectors without constructing spin-orbital ERIs.'''
    external = (0,) * nalpha + (1,) * (4 - nalpha)
    spin = dict(zip('ijklabcd', external * 2))
    result = []
    for name, coefficient, factors in _expanded_terms():
        dummy = sorted(set(''.join(s for _, s in factors)) - set('ijklabcd'))
        for values in itertools.product((0, 1), repeat=len(dummy)):
            spins = dict(spin, **dict(zip(dummy, values)))
            allowed = True
            for kind, s in factors:
                a = tuple(spins[x] for x in s)
                if kind == 'g':
                    allowed &= (a[0] == a[2] and a[1] == a[3] or
                                a[0] == a[3] and a[1] == a[2])
                else:
                    r = int(kind[-1])
                    allowed &= sum(a[:r]) == sum(a[r:])
            if allowed:
                script = ','.join(s for _, s in factors) + '->abcd'
                result.append((name, coefficient, factors, script, spins))
    return tuple(result)


def _slices(indices):
    return tuple(slice(int(a[0]), int(a[-1]) + 1) if len(a) else slice(0, 0)
                 for a in indices)


def _packed_indices(n, indices):
    '''Lexicographic pair/triple index and antisymmetric sign for a tile.'''
    grid = np.broadcast_arrays(*np.ix_(*indices))
    sign = np.ones(grid[0].shape, dtype=np.int8)
    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            sign = sign * np.sign(grid[j] - grid[i]).astype(np.int8)
    ordered = np.sort(np.stack(grid), axis=0)
    a, b = ordered[:2]
    if len(indices) == 2:
        index = a * (2*n - a - 1) // 2 + b - a - 1
    else:
        c = ordered[2]
        index = (n*(n-1)*(n-2) - (n-a)*(n-a-1)*(n-a-2)) // 6
        index += (b-a-1) * (2*n-a-b-2) // 2 + c-b-1
    return np.where(sign, index, 0), sign


class _Amplitudes:
    '''Read only the requested spin/space tile from native UCCSDT storage.'''
    def __init__(self, tamps, nocc, nvir, compact):
        self.tamps, self.nocc, self.nvir, self.compact = tamps, nocc, nvir, compact

    def get(self, rank, spins, indices):
        na = spins[:rank].count(0)
        if na != spins[rank:].count(0):
            return None
        majority = 0 if na >= (rank + 1)//2 else 1
        op = sorted(range(rank), key=lambda i: spins[i] != majority)
        vp = sorted(range(rank), key=lambda i: spins[rank+i] != majority)
        sign = _parity(op) * _parity(vp)
        order = op + [rank+i for i in vp]
        block = self.tamps[rank-1][rank-na]
        if rank == 2 and na == 1:
            # The canonical mixed double is alpha,beta, in native iaJB order.
            order = [op[0], rank+vp[0], op[1], rank+vp[1]]
        elif rank == 3 and na in (1, 2):
            order = [op[0], op[1], rank+vp[0], rank+vp[1], op[2], rank+vp[2]]
        idx = [indices[i] for i in order]
        if rank == 2 or not self.compact:
            result = block[_slices(idx)]
        elif na in (0, 3):
            if min(self.nocc[majority], self.nvir[majority]) < 3:
                return None
            oi, os = _packed_indices(self.nocc[majority], idx[:3])
            vi, vs = _packed_indices(self.nvir[majority], idx[3:])
            result = block[oi[..., None, None, None], vi[None, None, None, ...]]
            result *= os[..., None, None, None] * vs[None, None, None, ...]
        else:
            if min(self.nocc[majority], self.nvir[majority]) < 2:
                return None
            oi, os = _packed_indices(self.nocc[majority], idx[:2])
            vi, vs = _packed_indices(self.nvir[majority], idx[2:4])
            result = block[oi[:, :, None, None, None, None],
                           vi[None, None, :, :, None, None],
                           idx[4][None, None, None, None, :, None],
                           idx[5][None, None, None, None, None, :]]
            result *= os[:, :, None, None, None, None] * vs[None, None, :, :, None, None]
        result = result.transpose(np.argsort(order))
        return result if sign == 1 else -result


class _IntegralBlocks:
    def __init__(self, eris, nocc):
        self.eris, self.nocc = eris, nocc

    def _bare(self, spins, indices):
        if spins[0] == spins[1]:
            block = self.eris.pppp if spins[0] == 0 else self.eris.PPPP
            return block[_slices(indices)]
        if spins[0] == 0:
            return self.eris.pPpP[_slices(indices)]
        order = (1, 0, 3, 2)
        return self.eris.pPpP[_slices([indices[i] for i in order])].transpose(order)

    def get(self, spins, indices, labels):
        indices = [a if x in _OCCUPIED else a + self.nocc[s]
                   for x, s, a in zip(labels, spins, indices)]
        direct = spins[0] == spins[2] and spins[1] == spins[3]
        exchange = spins[0] == spins[3] and spins[1] == spins[2]
        result = self._bare(spins, indices) if direct else None
        if exchange:
            order = (0, 1, 3, 2)
            exch = self._bare(tuple(spins[i] for i in order),
                              [indices[i] for i in order]).transpose(order)
            result = -exch if result is None else result - exch
        return result


def _block_size(mycc, nocc, nvir):
    '''Budget additional work arrays, including einsum backend temporaries.'''
    available = (mycc.max_memory - lib.current_memory()[0]) * 1e6
    n = max(*nocc, *nvir, 1)
    def workspace(b):
        return 8 * (12*b**4 + 12*n*n*b*b + 12*n*b**3 + 32*n*n)
    if available < workspace(1):
        raise MemoryError('Insufficient max_memory for UCCSDT(Q) work arrays; '
                          'the existing ERIs and amplitudes are already resident.')
    b = max(*nvir, 1)
    while workspace(b) > available:
        b = max(1, b // 2)
    return b


def _validate(mycc, eris, tamps):
    if tamps is None or len(tamps) != 3 or tuple(map(len, tamps)) != (2, 3, 4):
        raise ValueError('tamps must contain T1(a,b), T2(aa,ab,bb), and T3(aaa,aab,bba,bbb).')
    oa, ob = mycc.nocc
    eris_nocc = getattr(eris, 'nocc', (oa, ob))
    if tuple(eris_nocc) != (oa, ob):
        raise ValueError('ERI occupied counts %s do not match the CC active space %s.' %
                         (tuple(eris_nocc), (oa, ob)))
    va, vb = (n-o for n, o in zip(mycc.nmo, mycc.nocc))
    shapes = [[(oa, va), (ob, vb)], [(oa, oa, va, va), (oa, va, ob, vb), (ob, ob, vb, vb)]]
    if mycc.do_tri_max_t:
        shapes.append([(math.comb(oa, 3), math.comb(va, 3)),
                       (math.comb(oa, 2), math.comb(va, 2), ob, vb),
                       (math.comb(ob, 2), math.comb(vb, 2), oa, va),
                       (math.comb(ob, 3), math.comb(vb, 3))])
    else:
        shapes.append([(oa, oa, oa, va, va, va), (oa, oa, va, va, ob, vb),
                       (ob, ob, vb, vb, oa, va), (ob, ob, ob, vb, vb, vb)])
    for blocks, expected in zip(tamps, shapes):
        for a, shape in zip(blocks, expected):
            if a.shape != shape:
                raise ValueError('Amplitude shape %s does not match %s.' % (a.shape, shape))
            if a.dtype != np.float64:
                raise NotImplementedError('UCCSDT(Q) requires real float64 amplitudes.')
    matrices = (eris.pppp, eris.pPpP, eris.PPPP) + tuple(eris.fock) + tuple(eris.mo_energy)
    if any(a.dtype != np.float64 for a in matrices):
        raise NotImplementedError('UCCSDT(Q) requires real float64 integrals.')
    ma, mb = mycc.nmo
    for a, shape in zip((eris.pppp, eris.pPpP, eris.PPPP),
                        ((ma,)*4, (ma, mb, ma, mb), (mb,)*4)):
        if a.shape != shape:
            raise ValueError('ERI shape %s does not match the active space %s.' % (a.shape, shape))
    for nmo, fock, energy in zip(mycc.nmo, eris.fock, eris.mo_energy):
        if fock.shape != (nmo, nmo) or energy.shape != (nmo,):
            raise ValueError('Fock matrices and orbital energies must match the active orbital space.')
        if not np.isfinite(fock).all() or not np.isfinite(energy).all():
            raise ValueError('UCCSDT(Q) requires finite Fock matrices and orbital energies.')
        if np.max(np.abs(fock - np.diag(fock.diagonal())), initial=0.) > 1e-7:
            raise NotImplementedError('UCCSDT(Q) requires canonical UHF orbitals.')
    return (oa, ob), (va, vb)


def _moments(amplitudes, integrals, terms, ranges, shape, einsum):
    w4, l3, l2 = (np.zeros(shape) for _ in range(3))
    for name, coefficient, factors, script, spins in terms:
        operands = []
        for kind, labels in factors:
            ss = tuple(spins[x] for x in labels)
            idx = [ranges[x, s] for x, s in zip(labels, ss)]
            if any(len(a) == 0 for a in idx):
                break
            if kind == 'g':
                a = integrals.get(ss, idx, labels)
            else:
                a = amplitudes.get(int(kind[-1]), ss, idx)
            if a is None:
                break
            operands.append(a)
        else:
            value = einsum(script, *operands)
            value *= coefficient
            if name == 'l2':
                l2 += value
            else:
                w4 += value
                if name in ('vvov', 'vooo'):
                    l3 += value
    return w4, l3, l2


def _kernel_python(mycc, eris=None, tamps=None, verbose=logger.NOTE):
    '''Return ([Q], (Q)) corrections without changing the UCCSDT energy.

    ``tamps`` uses the CC object's native full or compact representation.
    ``eris`` must be the original (not T1-dressed) UHF AO2MO result.  The
    orbitals must be canonical and real; converged CCSDT amplitudes should
    be supplied.  Frozen occupied and virtual orbitals are already removed
    from these arrays by the existing UCCSDT integral transformation.
    '''
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mycc, verbose)
    if tamps is None:
        tamps = mycc.tamps
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    nocc, nvir = _validate(mycc, eris, tamps)
    if not mycc.converged:
        log.warn('UCCSDT amplitudes are not marked converged.')
    if sum(nocc) < 4 or sum(nvir) < 4:
        return 0., 0.
    sectors = [(na, _spin_terms(na)) for na in range(5)
               if na <= min(nocc[0], nvir[0]) and 4-na <= min(nocc[1], nvir[1])]
    if not sectors:
        return 0., 0.
    # Construct the reusable formula cache before measuring resident memory.
    block_size = _block_size(mycc, nocc, nvir)
    log.debug('UCCSDT(Q) virtual block size = %d', block_size)
    amplitudes = _Amplitudes(tamps, nocc, nvir, mycc.do_tri_max_t)
    integrals = _IntegralBlocks(eris, nocc)
    einsum = functools.partial(_einsum, mycc.einsum_backend)
    if mycc.einsum_backend == 'pytblis':
        try:
            import pytblis
        except ImportError:
            pass  # The shared wrapper falls back to NumPy.
        else:
            # Quadruples moments include three operands; contract accepts two.
            einsum = functools.partial(pytblis.einsum, optimize='optimal')
    eocc = [e[:o] for e, o in zip(eris.mo_energy, nocc)]
    evir = [e[o:] for e, o in zip(eris.mo_energy, nocc)]
    ranges = {(x, s): np.arange((nocc if x in _OCCUPIED else nvir)[s])
              for x in 'mnef' for s in (0, 1)}
    e_bracket = e_extra = 0.
    singular = 0
    for na, terms in sectors:
        nb = 4-na
        spins = (0,)*na + (1,)*nb
        # Occupied combinations are unique; virtual indices are summed in
        # full.  This leaves one, rather than two, same-spin factorials.
        weight = 1. / (math.factorial(na) * math.factorial(nb))
        ot = itertools.product(itertools.combinations(range(nocc[0]), na),
                               itertools.combinations(range(nocc[1]), nb))
        for ha, hb in ot:
            holes = ha + hb
            occupied_energy = sum(eocc[s][i] for s, i in zip(spins, holes))
            for x, s, i in zip('ijkl', spins, holes):
                ranges[x, s] = np.array([i])
            tiles = [range(0, nvir[s], block_size) for s in spins]
            for starts in itertools.product(*tiles):
                for x, s, start in zip('abcd', spins, starts):
                    ranges[x, s] = np.arange(start, min(start + block_size, nvir[s]))
                shape = tuple(len(ranges[x, s]) for x, s in zip('abcd', spins))
                w4, l3, l2 = _moments(amplitudes, integrals, terms, ranges, shape, einsum)
                denominator = np.full(shape, occupied_energy)
                for axis, (x, s) in enumerate(zip('abcd', spins)):
                    dims = [1]*4
                    dims[axis] = shape[axis]
                    denominator -= evir[s][ranges[x, s]].reshape(dims)
                mask = np.abs(denominator) > 1e-15
                singular += mask.size - np.count_nonzero(mask)
                np.divide(w4, denominator, out=w4, where=mask)
                w4[~mask] = 0.
                e_bracket += weight * np.vdot(l2, w4)
                e_extra += weight * np.vdot(l3, w4)
    if singular:
        log.warn('UCCSDT(Q): set %d components with |denominator| <= 1e-15 to zero.', singular)
    e_paren = e_bracket + e_extra
    if not math.isfinite(e_bracket) or not math.isfinite(e_paren):
        raise FloatingPointError('Nonfinite UCCSDT(Q) correction; check amplitudes and integrals.')
    log.info('[Q] correction = % .12e    (Q) correction = % .12e', e_bracket, e_paren)
    log.timer('UCCSDT(Q)', *time0)
    return float(e_bracket), float(e_paren)


def _encode_native_terms(sectors):
    '''Encode the verified spin contractions for the native interpreter.

    Each record has category, factor count, twelve label spins, then three
    (kind, six labels) factors. The C implementation uses the same six
    seeds, but packs and contracts them without Python callbacks.
    '''
    records, coefficients, offsets = [], [], [0]
    for sector in sectors:
        for name, coefficient, factors, _, spins in sector:
            row = np.full(35, -1, dtype=np.int32)
            row[:2] = _CATEGORIES.index(name), len(factors)
            row[2:14] = [spins.get(x, -1) for x in _LABELS]
            for f, (kind, labels) in enumerate(factors):
                start = 14 + 7*f
                row[start] = 0 if kind == 'g' else int(kind[-1])
                row[start+1:start+1+len(labels)] = [_LABELS.index(x) for x in labels]
            records.append(row)
            coefficients.append(coefficient)
        offsets.append(len(records))
    return (np.asarray(records, dtype=np.int32).reshape(-1, 35),
            np.asarray(coefficients, dtype=np.float64), np.asarray(offsets, dtype=np.int64))


@functools.lru_cache(None)
def _native_term_table():
    return _encode_native_terms([_spin_terms(na) for na in range(5)])


@functools.lru_cache(None)
def _native_library():
    native = lib.load_library('libccsdt')
    try:
        native.UCCSDTQ_workspace
        native.UCCSDTQ_kernel
    except AttributeError as err:
        raise RuntimeError('Rebuild libccsdt and restart Python to use the UCCSDT(Q) C kernels.') from err
    ptr = ctypes.c_void_p
    native.UCCSDTQ_workspace.argtypes = [ptr, ptr, ptr, ctypes.c_int]
    native.UCCSDTQ_workspace.restype = ctypes.c_uint64
    native.UCCSDTQ_kernel.argtypes = [
        ptr, ptr, ptr, ctypes.c_int, ptr, ptr, ptr, ptr, ptr, ptr,
        ctypes.c_int64, ctypes.c_int, ctypes.c_uint64, ptr, ptr, ptr, ptr]
    native.UCCSDTQ_kernel.restype = ctypes.c_int
    return native


class _NativeInputs:
    '''Keep native input arrays alive; preserve full/compact array strides.'''
    def __init__(self, mycc, eris, tamps, nocc, nvir):
        self.arrays = (eris.pppp, eris.pPpP, eris.PPPP, *tamps[1], *tamps[2])
        if any(not isinstance(a, np.ndarray) for a in (*self.arrays, *eris.mo_energy)):
            raise TypeError('The UCCSDT(Q) C kernel requires NumPy ndarray or memmap storage; '
                            'load HDF5 datasets explicitly before calling the native kernel.')
        if any(not a.flags.aligned for a in (*self.arrays, *eris.mo_energy)):
            raise ValueError('The UCCSDT(Q) C kernel requires aligned float64 array storage.')
        self.pointers = (ctypes.c_void_p * 10)(*[a.ctypes.data for a in self.arrays])
        self.strides = np.zeros((10, 6), dtype=np.int64)
        for row, a in zip(self.strides, self.arrays):
            row[:a.ndim] = np.asarray(a.strides) // a.itemsize
        self.sizes = np.asarray((*nocc, *nvir), dtype=np.int64)
        self.energies = tuple(np.ascontiguousarray(e) for e in eris.mo_energy)
        self.compact = int(mycc.do_tri_max_t)


def _task_count(nocc, nvir, block_size):
    def virtual_count(n, rank):
        if rank == 0:
            return 1
        # Coefficient of x**rank in prod_tiles(1+x+...+x**capacity).
        counts = [1] + [0]*rank
        for start in range(0, n, block_size):
            capacity = min(rank, block_size, n-start)
            counts = [sum(counts[k-j] for j in range(min(k, capacity)+1))
                      for k in range(rank+1)]
        return counts[rank]

    total = 0
    for na in range(5):
        nb = 4-na
        if na <= min(nocc[0], nvir[0]) and nb <= min(nocc[1], nvir[1]):
            total += (math.comb(nocc[0], na) * math.comb(nocc[1], nb) *
                      virtual_count(nvir[0], na) * virtual_count(nvir[1], nb))
    return total


def _native_block_size(mycc, nocc, nvir, records, offsets, nthreads):
    '''Fit all per-thread workspaces and expose enough tiles for OpenMP.'''
    native = _native_library()
    sizes = np.asarray((*nocc, *nvir), dtype=np.int64)
    max_virtual = max(*nvir, 1)
    # The largest tile that fits in memory is not the fastest tile: it
    # increases B**4 traffic and leaves too few tasks for the team.  The
    # validated cost/traffic sweep selects about one quarter of a larger
    # virtual space, with a floor of 12 (the useful L2-sized tile on the
    # measured CPU family).  Small spaces remain a single tile.
    block_size = max_virtual if max_virtual < 24 else max(12, (max_virtual + 3)//4)
    # Reserve task tuples/integers, NumPy task/reduction arrays, and generator
    # bookkeeping in addition to the per-worker native tensor buffers.
    available = (mycc.max_memory-lib.current_memory()[0])*1e6 - 16384 - 8192*nthreads
    while True:
        work = native.UCCSDTQ_workspace(sizes.ctypes.data, records.ctypes.data,
                                       offsets.ctypes.data, block_size)
        if not work:
            raise MemoryError('UCCSDT(Q) native workspace dimensions overflow.')
        tasks = _task_count(nocc, nvir, block_size)
        workers = min(tasks, nthreads)
        # Two waves of symmetry-distinct tasks also avoid excessively large
        # work arrays whose cache traffic can outweigh fewer contractions.
        if work*workers <= available and (tasks >= 2*nthreads or block_size == 1):
            return block_size
        if block_size == 1:
            raise MemoryError('Insufficient max_memory for UCCSDT(Q) thread workspaces; '
                              'reduce lib.num_threads() or increase max_memory.')
        block_size = max(1, block_size//2)


def _native_tasks(nocc, nvir, block_size):
    def virtual_tiles(n, rank):
        def visit(prefix, lower, repeated):
            if len(prefix) == rank:
                yield prefix
                return
            for start in range(lower, n, block_size):
                count = repeated+1 if prefix and prefix[-1] == start else 1
                if count > min(block_size, n-start):
                    continue
                if n-start-count < rank-len(prefix)-1:
                    break
                yield from visit(prefix+(start,), start, count)
        # Generate only feasible tile multisets. Filtering the full set of
        # combinations with replacement is prohibitively slow for tiny tiles.
        yield from visit((), 0, 0)

    for na in range(5):
        nb = 4-na
        if na > min(nocc[0], nvir[0]) or nb > min(nocc[1], nvir[1]):
            continue
        spins = (0,)*na + (1,)*nb
        for ha in itertools.combinations(range(nocc[0]), na):
            for hb in itertools.combinations(range(nocc[1]), nb):
                # Nested iterators avoid itertools.product materializing
                # every combination in a very finely tiled virtual space.
                for a in virtual_tiles(nvir[0], na):
                    for b in virtual_tiles(nvir[1], nb):
                        starts = a + b
                        shape = tuple(min(block_size, nvir[s]-start) for s, start in zip(spins, starts))
                        yield (na, *ha, *hb, *starts, *shape)


def _tile_multiplicity(task):
    '''Number of ordered tiles represented by a same-spin sorted tile.

    Each moment is antisymmetric, so its product with another moment and
    the denominator is symmetric. Only permutations within a spin sector
    are counted; indices inside repeated tiles remain fully summed in C.
    '''
    na = task[0]
    multiplicity = math.factorial(na) * math.factorial(4-na)
    for starts in (task[5:5+na], task[5+na:9]):
        for _, group in itertools.groupby(starts):
            multiplicity //= math.factorial(sum(1 for _ in group))
    return multiplicity


def _native_batch(inputs, records, coefficients, offsets, tasks, nthreads,
                  workspace_limit, moments=None):
    tasks = np.ascontiguousarray(tasks, dtype=np.int64)
    energies = np.empty((len(tasks), 2))
    singular = np.empty(len(tasks), dtype=np.int64)
    actual_threads = ctypes.c_int()
    status = _native_library().UCCSDTQ_kernel(
        inputs.pointers, inputs.strides.ctypes.data, inputs.sizes.ctypes.data,
        inputs.compact, inputs.energies[0].ctypes.data, inputs.energies[1].ctypes.data,
        records.ctypes.data, coefficients.ctypes.data, offsets.ctypes.data,
        tasks.ctypes.data, len(tasks), nthreads, workspace_limit,
        energies.ctypes.data, singular.ctypes.data, ctypes.byref(actual_threads),
        moments.ctypes.data if moments is not None else None)
    if status in (-1, -3):
        raise MemoryError('UCCSDT(Q) native workspace allocation/budget failed (status %d).' % status)
    if status:
        raise RuntimeError('UCCSDT(Q) native kernel failed (status %d).' % status)
    return energies, singular, actual_threads.value


def _native_task_array(nocc, nvir, block_size):
    """Stream task tuples directly into a compact native task array."""
    count = _task_count(nocc, nvir, block_size)
    values = np.fromiter((value for task in _native_tasks(nocc, nvir, block_size)
                          for value in task), dtype=np.int64,
                         count=13*count)
    if values.size != 13*count:
        raise RuntimeError('UCCSDT(Q) task generator produced an inconsistent count.')
    return values.reshape((count, 13))


def _native_task_budget(mycc, nocc, nvir, block_size, nthreads, workspace):
    """Return whether one-call task/output arrays fit the current memory budget.

    The native arena is already budgeted by _native_block_size.  This extra
    check accounts for the compact task descriptors and per-task energy and
    singular-count outputs needed to keep one OpenMP region alive.  If they do
    not fit, kernel() keeps the streaming batch path below.
    """
    count = _task_count(nocc, nvir, block_size)
    task_bytes = count * (13*8 + 2*8 + 8)
    available = ((mycc.max_memory - lib.current_memory()[0])*1e6
                 - 16384 - 8192*nthreads)
    workers = min(count, nthreads)
    return task_bytes <= max(0., available - workspace*workers)


def _native_moments(mycc, eris, tamps, nalpha, holes, starts, shape, terms=None):
    '''Small-block diagnostic used to check every native contraction class.'''
    nocc, nvir = _validate(mycc, eris, tamps)
    inputs = _NativeInputs(mycc, eris, tamps, nocc, nvir)
    sectors = [[] for _ in range(5)]
    sectors[nalpha] = _spin_terms(nalpha) if terms is None else terms
    records, coefficients, offsets = _encode_native_terms(sectors)
    moments = np.empty((3, *shape))
    tasks = [(nalpha, *holes, *starts, *shape)]
    workspace = _native_library().UCCSDTQ_workspace(
        inputs.sizes.ctypes.data, records.ctypes.data, offsets.ctypes.data, max(shape))
    _native_batch(inputs, records, coefficients, offsets, tasks, 1, workspace, moments)
    return tuple(moments)


def kernel(mycc, eris=None, tamps=None, verbose=logger.NOTE):
    '''Return ([Q], complete (Q)) using the C/OpenMP correction kernel.

    Uses the original real canonical UHF ERIs and native full/compact
    amplitudes. ``lib.num_threads()`` controls the maximum OpenMP team.
    Workspaces are private to each thread and counted against max_memory.
    For a fixed partition, task energies are added in a fixed order
    independent of thread scheduling. Different tile sizes can change
    floating-point summation roundoff.
    The CC energy and input arrays are not changed.
    '''
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mycc, verbose)
    if tamps is None:
        tamps = mycc.tamps
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    nocc, nvir = _validate(mycc, eris, tamps)
    if not mycc.converged:
        log.warn('UCCSDT amplitudes are not marked converged.')
    if not _task_count(nocc, nvir, max(*nvir, 1)):
        return 0., 0.
    inputs = _NativeInputs(mycc, eris, tamps, nocc, nvir)
    records, coefficients, offsets = _native_term_table()
    nthreads = max(1, lib.num_threads())
    block_size = _native_block_size(mycc, nocc, nvir, records, offsets, nthreads)
    workspace = _native_library().UCCSDTQ_workspace(
        inputs.sizes.ctypes.data, records.ctypes.data, offsets.ctypes.data, block_size)
    log.debug('UCCSDT(Q) C/OpenMP: block size %d, workspace %.3f MB/thread, requested threads %d',
              block_size, workspace/1e6, nthreads)
    e_bracket = e_extra = 0.
    singular = actual_threads = 0
    if _native_task_budget(mycc, nocc, nvir, block_size, nthreads, workspace):
        # Keep one native call for the whole correction.  UCCSDTQ_kernel
        # allocates its per-thread arena and enters OpenMP once per call;
        # splitting the task stream into small Python batches repeated both
        # operations and recomputed the workspace bound for every batch.  The
        # task array is compact (13 int64 values per task), and all reductions
        # below retain the generator's deterministic task order.
        tasks = _native_task_array(nocc, nvir, block_size)
        energies, counts, actual_threads = _native_batch(
            inputs, records, coefficients, offsets, tasks, nthreads, workspace)
        for task, (bracket, extra) in zip(tasks, energies):
            multiplicity = _tile_multiplicity(task)
            e_bracket += multiplicity * bracket
            e_extra += multiplicity * extra
        singular = int(counts.sum())
    else:
        # Keep the previous bounded stream when descriptors and per-task
        # outputs would consume the caller's max_memory allowance.
        tasks = iter(_native_tasks(nocc, nvir, block_size))
        while True:
            batch = list(itertools.islice(tasks, 8*nthreads))
            if not batch:
                break
            energies, counts, team = _native_batch(
                inputs, records, coefficients, offsets, batch, nthreads, workspace)
            for task, (bracket, extra) in zip(batch, energies):
                multiplicity = _tile_multiplicity(task)
                e_bracket += multiplicity * bracket
                e_extra += multiplicity * extra
            singular += int(counts.sum())
            actual_threads = max(actual_threads, team)
    if singular:
        log.warn('UCCSDT(Q): set %d components with |denominator| <= 1e-15 to zero.', singular)
    e_paren = e_bracket + e_extra
    if not math.isfinite(e_bracket) or not math.isfinite(e_paren):
        raise FloatingPointError('Nonfinite UCCSDT(Q) correction; check amplitudes and integrals.')
    log.info('UCCSDT(Q) C/OpenMP threads used = %d', actual_threads)
    log.info('[Q] correction = % .12e    (Q) correction = % .12e', e_bracket, e_paren)
    log.timer('UCCSDT(Q)', *time0)
    return float(e_bracket), float(e_paren)
