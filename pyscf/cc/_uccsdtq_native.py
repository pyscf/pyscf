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

"""Direct C/OpenMP spin-blocked contractions for UCCSDTQ.

This first native engine prioritizes bounded workspace and transparent
connection to the verified equations.  It reads compact T3/T4 directly and never
allocates a dense spin-orbital tensor.  It uses scalar contractions rather than
BLAS intermediates, so large-orbital calculations can be computationally costly.
"""
from collections import defaultdict
import ctypes
from functools import lru_cache
from itertools import permutations, product
from math import comb

import numpy as np
from pyscf import lib
from pyscf.cc import _uccsdtq_utils as storage
from pyscf.cc._uccsdtq_terms import TERMS


class _Factor(ctypes.Structure):
    _fields_ = [('array', ctypes.c_int32), ('ndim', ctypes.c_int32),
                ('indices', ctypes.c_int32 * 8)]


class _Term(ctypes.Structure):
    _fields_ = [('coefficient', ctypes.c_double), ('nfactor', ctypes.c_int32),
                ('ndummy', ctypes.c_int32), ('dummy', ctypes.c_int32 * 4),
                ('spin', ctypes.c_int32 * 4), ('factor', _Factor * 4)]


class _Array(ctypes.Structure):
    _fields_ = [('ndim', ctypes.c_int32), ('hamiltonian', ctypes.c_int32),
                ('compact', ctypes.c_int32), ('ngroup', ctypes.c_int32),
                ('shape', ctypes.c_int64 * 8), ('stride', ctypes.c_int64 * 8),
                ('offset', ctypes.c_int32 * 8),
                ('group_size', ctypes.c_int32 * 4),
                ('group_dim', ctypes.c_int32 * 4)]


def _parity(seq):
    return (-1) ** sum(x > y for i, x in enumerate(seq) for y in seq[i+1:])


def _factor(kind, indices, spins):
    """Map a spin-orbital factor into one native array with a fixed sign."""
    s = [spins[i] for i in indices]
    sign = 1
    if kind == 'f':
        if s[0] != s[1]:
            return None
        return sign, (s[0], tuple(indices))
    if kind == 'g':
        if s[0]+s[1] != s[2]+s[3]:
            return None
        if s[0] == s[1]:
            array = 2 if s[0] == 0 else 4
            groups = (indices[:2], indices[2:])
            key = []
            for group in groups:
                sign *= _parity(group)
                key.extend(sorted(group))
            return sign, (array, tuple(key))
        # Only within-pair interchange: T1 dressing forbids bra/ket swap.
        indices = list(indices)
        if s[0]:
            indices[0], indices[1] = indices[1], indices[0]
            sign = -sign
        if s[2]:
            indices[2], indices[3] = indices[3], indices[2]
            sign = -sign
        return sign, (3, tuple(indices))
    rank = int(kind[1:])
    na = rank - sum(s[:rank])
    if na != rank - sum(s[rank:]):
        return None
    nb = rank - na
    majority = 0 if na >= nb else 1
    key = []
    h = indices[:rank]
    v = indices[rank:]
    for group in (h, v):
        spin_order = [spins[i] != majority for i in group]
        sign *= _parity(spin_order)
    groups = [[i for i in group if spins[i] == spin]
              for spin in (majority, 1-majority) for group in (h, v)]
    for group in groups:
        sign *= _parity(group)
        key.extend(sorted(group))
    start = {2: 5, 3: 8, 4: 12}[rank]
    return sign, (start + nb, tuple(key))


def _renormalize_native(factors, mapping, rank_by_array):
    """Apply a same-spin dummy relabeling and restore internal antisymmetry."""
    sign, result = 1, []
    for array, indices in factors:
        labels = [mapping.get(i, i) for i in indices]
        if array in (0, 1, 3):
            groups = [labels]
        elif array in (2, 4):
            groups = [labels[:2], labels[2:]]
        else:
            rank, na = rank_by_array[array]
            a, b = max(na, rank-na), min(na, rank-na)
            groups = [labels[:a], labels[a:2*a], labels[2*a:2*a+b], labels[2*a+b:]]
        if array not in (0, 1, 3):
            for group in groups:
                sign *= _parity(group)
                group.sort()
        result.append((array, tuple(i for group in groups for i in group)))
    return sign, tuple(sorted(result))


@lru_cache(None)
def _descriptors(rank, nalpha):
    """Coalesce all external permutations and valid internal spin assignments."""
    # The same Hamiltonian/amplitude factors recur across many permutations.
    # Keep this cache local: its label/spin keys need not survive compilation.
    @lru_cache(maxsize=8192)
    def convert(kind, indices, spin_values):
        return _factor(kind, indices, dict(zip(indices, spin_values)))

    rank_by_array = {}
    for nrank, start in ((2, 5), (3, 8), (4, 12)):
        rank_by_array.update({start+b: (nrank, nrank-b) for b in range(nrank+1)})
    ext = list(range(rank)) + list(range(4, 4+rank))
    letters = 'ijkl'[:rank] + 'abcd'[:rank] + 'mnef'
    index = dict(zip(letters, ext + [8, 9, 10, 11]))
    external_spins = ([0]*nalpha + [1]*(rank-nalpha)) * 2
    base_spins = dict(zip(ext, external_spins))
    collected = defaultdict(int)
    for _, numerator, denominator, factors in TERMS[rank]:
        assert 1152 % denominator == 0
        weight = numerator * (1152 // denominator)
        dummy = sorted({index[c] for _, ix in factors for c in ix if c in 'mnef'})
        for internal_spins in product((0, 1), repeat=len(dummy)):
            spins = dict(base_spins)
            spins.update(zip(dummy, internal_spins))
            dummy_variants = [{}]
            for a, b in ((8, 9), (10, 11)):
                if a in dummy and b in dummy and spins[a] == spins[b]:
                    dummy_variants += [{**m, a: b, b: a} for m in dummy_variants]
            for hp in permutations(range(rank)):
                for vp in permutations(range(rank)):
                    rename = dict(zip(range(rank), hp))
                    rename.update(zip(range(4, 4+rank), (4+x for x in vp)))
                    native, sign = [], _parity(hp) * _parity(vp)
                    for kind, labels in factors:
                        mapped = [rename.get(index[c], index[c]) for c in labels]
                        converted = convert(kind, tuple(mapped), tuple(spins[i] for i in mapped))
                        if converted is None:
                            break
                        local_sign, key = converted
                        sign *= local_sign
                        native.append(key)
                    else:
                        best = None
                        best_sign = 1
                        for mapping in dummy_variants:
                            ds, candidate = _renormalize_native(native, mapping, rank_by_array)
                            if best is None or candidate < best:
                                best, best_sign = candidate, ds
                        collected[(tuple(zip(dummy, internal_spins)), best)] += weight*sign*best_sign
    records = []
    for (dummy_spins, factors), weight in sorted(collected.items()):
        if weight == 0:
            continue
        term = _Term()
        term.coefficient = weight / 1152.
        term.nfactor = len(factors)
        term.ndummy = len(dummy_spins)
        for j, (label, spin) in enumerate(dummy_spins):
            term.dummy[j], term.spin[j] = label, spin
        for j, (array, indices) in enumerate(factors):
            term.factor[j].array = array
            term.factor[j].ndim = len(indices)
            term.factor[j].indices[:len(indices)] = indices
        records.append(term)
    return (_Term * len(records))(*records)


def _metadata(mycc, arrays):
    oa, ob, va, vb = storage._dims(mycc)
    dims = ((oa, va), (ob, vb))
    metas = (_Array * 17)()
    for number, array in enumerate(arrays):
        meta = metas[number]
        meta.ndim = array.ndim
        meta.shape[:array.ndim] = array.shape
        meta.stride[:array.ndim] = tuple(s // 8 for s in array.strides)
        if number < 5:
            meta.hamiltonian = 1
            spins = [(0, 0), (1, 1), (0, 0, 0, 0), (0, 1, 0, 1), (1, 1, 1, 1)][number]
            for i, spin in enumerate(spins):
                meta.offset[i] = dims[spin][0]
        else:
            rank, start = (2, 5) if number < 8 else (3, 8) if number < 12 else (4, 12)
            block = number-start
            na, nb = rank-block, block
            majority = 0 if na >= nb else 1
            n, v = dims[majority]
            n2, v2 = dims[1-majority]
            sizes = (max(na, nb),)*2 + (min(na, nb),)*2
            ds = (n, v, n2, v2)
            groups = [(sz, d) for sz, d in zip(sizes, ds) if sz]
            meta.ngroup = len(groups)
            for j, (sz, d) in enumerate(groups):
                meta.group_size[j], meta.group_dim[j] = sz, d
            meta.compact = rank >= 3 and tuple(array.shape) == storage.block_shape(mycc, rank, block, True)
    return metas


@lru_cache(maxsize=1)
def _native_kernel():
    library = lib.load_library('libccsdt')
    try:
        version = library.UCCSDTQabi_version
        version.restype = ctypes.c_int
        if version() != 1:
            raise RuntimeError('UCCSDTQ native ABI mismatch; rebuild libccsdt')
        function = library.UCCSDTQresidual_block
    except AttributeError as err:
        raise RuntimeError('UCCSDTQ native ABI unavailable; rebuild libccsdt') from err
    function.restype = None
    return function


def _residual_shape(mycc, rank, block, t3):
    if rank == 3:
        return tuple(t3[block].shape)
    return storage.block_shape(mycc, rank, block, rank == 4 and mycc.do_tri_max_t)


def residuals(mycc, imds, t2, t3, t4, out=None):
    """Add raw T4 feedback to R2/R3 and the complete raw R4.

    T2 uses full native blocks. T3/T4 input can be full or compact; R3 output
    follows T3's layout and R4 follows ``mycc.do_tri_max_t``. Provided outputs are
    updated in place, and every input amplitude/integral remains unchanged.
    """
    oa, ob, va, vb = storage._dims(mycc)
    if tuple(map(len, (imds.t1_fock, imds.t1_eris, t2, t3, t4))) != (2, 3, 3, 4, 5):
        raise ValueError('invalid native UCCSDTQ spin-block counts')
    sources = list(imds.t1_fock) + list(imds.t1_eris) + list(t2) + list(t3) + list(t4)
    arrays = [np.asarray(a) for a in sources]
    nma, nmb = oa+va, ob+vb
    expected = [(nma, nma), (nmb, nmb), (nma,)*4, (nma,nmb,nma,nmb), (nmb,)*4]
    for rank in (2, 3, 4):
        expected.extend(storage.block_shape(mycc, rank, b, False) for b in range(rank+1))
    for number, (array, shape) in enumerate(zip(arrays, expected)):
        valid = tuple(array.shape) == shape
        if 8 <= number < 12:
            valid |= tuple(array.shape) == storage.block_shape(mycc, 3, number-8, True)
        elif number >= 12:
            valid |= tuple(array.shape) == storage.block_shape(mycc, 4, number-12, True)
        if not valid or array.dtype != np.float64:
            raise ValueError('invalid shape or dtype for UCCSDTQ native input %d' % number)
        if not array.flags.aligned:
            arrays[number] = np.require(array, requirements=['A'])
    metas = _metadata(mycc, arrays)
    pointers = (ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
    if out is None:
        out = tuple(tuple(np.zeros(_residual_shape(mycc, rank, block, t3)) for block in range(rank+1))
                    for rank in (2, 3, 4))
    if tuple(map(len, out)) != (3, 4, 5):
        raise ValueError('invalid native UCCSDTQ output spin-block counts')
    native = _native_kernel()
    for rank, blocks in zip((2, 3, 4), out):
        for block, result in enumerate(blocks):
            expected_shape = _residual_shape(mycc, rank, block, t3)
            if result.shape != expected_shape or result.dtype != np.float64 or not result.flags.writeable:
                raise ValueError('invalid native UCCSDTQ output shape, dtype, or writeability')
            if not result.size:
                continue
            terms = _descriptors(rank, rank-block)
            strides = np.asarray(result.strides, dtype=np.int64) // 8
            native(pointers, metas, terms, ctypes.c_int64(len(terms)),
                   ctypes.c_int32(oa), ctypes.c_int32(ob), ctypes.c_int32(va), ctypes.c_int32(vb),
                   ctypes.c_int32(rank), ctypes.c_int32(rank-block),
                   ctypes.c_int32(rank >= 3 and result.ndim < 2*rank),
                   result.ctypes.data_as(ctypes.c_void_p),
                   strides.ctypes.data_as(ctypes.c_void_p))
    return out
