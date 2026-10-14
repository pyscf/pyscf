# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Factorized, memory-bounded spin-block CCSDTQ contractions.

Connected star diagrams are contracted as binary tensor products.  Four-index
intermediates are reused; six-index intermediates are built in occupied batches
and consumed by every term using that batch before release.  Compact T4 is read
only into bounded matrix panels.  The direct implementation remains a separate
numerical oracle, including its exact rational spin/permutation descriptors.
'''

from collections import OrderedDict
import ctypes
from functools import lru_cache
from itertools import permutations, product
from math import comb, prod

import numpy as np
from pyscf import lib
from pyscf.cc import _uccsdtq_native as direct
from pyscf.cc import _uccsdtq_utils as storage


_PTR = ctypes.c_void_p
_INT = ctypes.c_int32
_I64 = ctypes.c_int64
_DOUBLE = ctypes.c_double


class _FinalPlan(ctypes.Structure):
    _fields_ = [(name, _INT) for name in ('rank', 'nalpha', 'oa', 'ob', 'va', 'vb',
                 'compact', 'nworkers', 'nrows', 'ncols', 'nshared', 'nmaps')]
    _fields_ += [('rows', _INT*8), ('cols', _INT*8), ('shared', _INT*4),
                 ('starts', _INT*12), ('counts', _INT*12), ('caps', _INT*12),
                 ('capacity', _I64)]


@lru_cache(maxsize=1)
def _fused_kernel():
    library = lib.load_library('libccsdt')
    try:
        if library.UCCSDTQfused_abi_version() != 1:
            raise RuntimeError('UCCSDTQ fused ABI mismatch; rebuild libccsdt')
        function = library.UCCSDTQfinal_contract
    except AttributeError as err:
        raise RuntimeError('UCCSDTQ fused ABI unavailable; rebuild libccsdt') from err
    function.argtypes = [_PTR]*17
    function.restype = ctypes.c_int
    return function


@lru_cache(maxsize=1)
def _kernels():
    library = lib.load_library('libccsdt')
    try:
        if library.UCCSDTQblocked_abi_version() != 1:
            raise RuntimeError('UCCSDTQ blocked ABI mismatch; rebuild libccsdt')
        gather = library.UCCSDTQgather
        gemm = library.UCCSDTQgemm
    except AttributeError as err:
        raise RuntimeError('UCCSDTQ blocked ABI unavailable; rebuild libccsdt') from err
    gather.argtypes = [_PTR, _PTR, _PTR, _INT, _PTR, _PTR, _PTR, _PTR, _PTR]
    gemm.argtypes = [_I64, _I64, _I64, _DOUBLE, _PTR, _PTR, _DOUBLE, _PTR]
    for function in (gather, gemm):
        function.restype = ctypes.c_int
    return gather, gemm


class _Node:
    def __init__(self, array=None, labels=(), left=None, right=None):
        self.array, self.left, self.right = array, left, right
        self.identity = None
        if array is not None:
            self.labels = tuple(labels)
            self.leaves = ((array, self.labels),)
        else:
            self.labels = tuple(i for i in left.labels if i not in right.labels)
            self.labels += tuple(i for i in right.labels if i not in left.labels)
            self.leaves = left.leaves + right.leaves


def _canonical(node, colors):
    '''Tensor identity and free-axis order, independent of spectator indices.'''
    # A node belongs to one compiled spin assignment; its symbolic identity
    # survives iterations, whereas its numerical value never does.
    if node.identity is not None:
        return node.identity
    free = set(node.labels)
    best = None
    for leaves in permutations(node.leaves):
        mapping, tags, factors = {}, [], []
        for array, labels in leaves:
            renamed = []
            for label in labels:
                if label not in mapping:
                    mapping[label] = len(mapping)
                    tags.append((colors[label], label in free))
                renamed.append(mapping[label])
            factors.append((array, tuple(renamed)))
        key = tuple(factors), tuple(tags)
        if best is None or key < best[0]:
            order = tuple(x for x in mapping if x in free)
            best = key, order
    node.identity = best
    return best


def _family(term):
    names = ['F' if f.array < 2 else 'V' if f.array < 5 else
             'T2' if f.array < 8 else 'T3' if f.array < 12 else 'T4'
             for f in term.factor[:term.nfactor]]
    return '*'.join(names)


def _antisymmetric_groups(array, labels):
    if array in (0, 1, 3):
        return ()
    if array in (2, 4):
        return (labels[:2], labels[2:])
    rank, start = (2, 5) if array < 8 else (3, 8) if array < 12 else (4, 12)
    nb = array-start
    major, minor = max(rank-nb, nb), min(rank-nb, nb)
    return tuple(group for group in (labels[:major], labels[major:2*major],
                 labels[2*major:2*major+minor], labels[2*major+minor:]) if len(group) > 1)


def _zero_spin_block(array, labels, colors, dimensions):
    # Repeated same-spin fermion indices vanish exactly. This is a dimension
    # check, not numerical screening of small amplitudes or integrals.
    for group in _antisymmetric_groups(array, labels):
        for space in (0, 1):
            slots = [i for i in group if colors[i][1] == space]
            if slots and len(slots) > dimensions[slots[0]]:
                return True
    return False


def _zero_batch(node, bounds, colors):
    for array, labels in node.leaves:
        for group in _antisymmetric_groups(array, labels):
            fixed = [(colors[i][1], bounds[i][0]) for i in group
                     if i in bounds and bounds[i][1] == 1]
            if len(set(fixed)) != len(fixed):
                return True
    return False


def _order(leaves, dimensions):
    best = None
    # Excitation vertices have no mutual contractions. Every merge extends
    # the connected H vertex, avoiding disconnected outer-product tensors.
    for tail in permutations(leaves[1:]):
        node, cost, valid = leaves[0], 0, True
        for index, right in enumerate(tail):
            cost += prod(dimensions[x] for x in set(node.labels) | set(right.labels))
            node = _Node(left=node, right=right)
            # A three-leaf diagram first forms W<=4, then a streamed W6.
            # This avoids nesting a six-index batch inside another W6 loop.
            maximum = 4 if index < len(tail)-2 else 6
            if index < len(tail)-1 and len(node.labels) > maximum:
                valid = False
                break
        if valid and (best is None or cost < best[0]):
            best = cost, node
    if best is None:
        raise RuntimeError('No connected CCSDTQ factorization with six-index intermediates')
    return best[1]


def _coalesce_virtual(jobs):
    '''Contract once for jobs differing only by external virtual placement.

    Occupied and summed indices retain their names.  No integral pair
    interchange or additional tensor symmetry is assumed.  Coefficients are
    accumulated with the direct descriptor table's exact denominator 1152.
    '''
    groups = OrderedDict()
    for job in jobs:
        rank, na, coefficient, axes, left, right, colors, dimensions = job
        slots = (iter(range(4, 4+na)), iter(range(4+na, 4+rank)))
        rename = {}
        for i in axes + right.labels:
            if 4 <= i < 4+rank and i not in rename:
                rename[i] = next(slots[colors[i][0]])
        key = (rank, na, tuple(rename.get(i, i) for i in axes), right.array,
               tuple(rename.get(i, i) for i in right.labels))
        if key not in groups:
            groups[key] = (job, rename, OrderedDict())
        representative, original_names, weights = groups[key]
        inverse = {v: k for k, v in rename.items()}
        mapping = tuple(inverse[original_names[i]] if i in original_names else i
                        for i in range(12))
        weights[mapping] = weights.get(mapping, 0) + round(coefficient*1152)
    result = []
    for job, _, weights in groups.values():
        nonzero = [(mapping, weight) for mapping, weight in weights.items() if weight]
        if nonzero:
            maps, values = zip(*nonzero)
            result.append((job, np.asarray(maps, dtype=np.int32),
                           np.asarray(values, dtype=np.float64)/1152))
    return result


@lru_cache(maxsize=4)
def _plans(dims, families):
    oa, ob, va, vb = dims
    sizes = ((oa, va), (ob, vb))
    groups = OrderedDict()
    for rank in (2, 3, 4):
        for na in range(rank, -1, -1):
            nb = rank-na
            if not comb(oa, na)*comb(ob, nb)*comb(va, na)*comb(vb, nb):
                continue
            for term in direct._descriptors(rank, na):
                if families is not None and _family(term) not in families:
                    continue
                spin = dict(zip(tuple(range(rank)) + tuple(range(4, 4+rank)),
                                ([0]*na + [1]*nb)*2))
                spin.update((int(term.dummy[i]), int(term.spin[i])) for i in range(term.ndummy))
                colors = {i: (s, int(4 <= i < 8 or i >= 10)) for i, s in spin.items()}
                dimensions = {i: sizes[s][v] for i, (s, v) in colors.items()}
                if any(n == 0 for n in dimensions.values()):
                    continue
                if any(_zero_spin_block(f.array, f.indices[:f.ndim], colors, dimensions)
                       for f in term.factor[:term.nfactor]):
                    continue
                leaves = [_Node(f.array, f.indices[:f.ndim]) for f in term.factor[:term.nfactor]]
                tree = _order(leaves, dimensions)
                key, axes = _canonical(tree.left, colors)
                if key not in groups:
                    groups[key] = [tree.left, axes, colors, dimensions, []]
                groups[key][-1].append((rank, na, term.coefficient, axes, tree.left,
                                        tree.right, colors, dimensions))
    for group in groups.values():
        group[-1] = _coalesce_virtual(group[-1])
    return tuple(group for group in groups.values() if group[-1])


def _tiles(labels, ranges, caps):
    starts = [range(ranges[i][0], ranges[i][0]+ranges[i][1], caps[i]) for i in labels]
    for start in product(*starts):
        yield {i: (s, min(caps[i], ranges[i][0]+ranges[i][1]-s))
               for i, s in zip(labels, start)}


@lru_cache(maxsize=4096)
def _matrix_axes(left, right):
    shared = tuple(i for i in left if i in right)
    rows = tuple(i for i in left if i not in right)
    cols = tuple(i for i in right if i not in left)
    return rows, cols, shared


def _tile_caps(rows, cols, shared, ranges, limit, occupied_points=False):
    labels = rows + cols + shared
    caps = {i: ranges[i][1] for i in labels}
    if occupied_points:
        for i in rows + cols:
            if i < 4:
                caps[i] = 1
    while True:
        m, n, k = (prod(caps[i] for i in axes) for axes in (rows, cols, shared))
        volumes = (m*k, k*n, m*n)
        if max(volumes) <= limit:
            return caps
        operands = (rows+shared, shared+cols, rows+cols)
        axes = operands[int(np.argmax(volumes))]
        candidates = [i for i in axes if caps[i] > 1]
        if not candidates:
            raise MemoryError('Insufficient CCSDTQ matrix workspace')
        # Prefer a large dimension. Contracted axes may also be tiled; their
        # partial products are summed before projecting the result.
        label = max(candidates, key=lambda i: caps[i])
        caps[label] = (caps[label]+1)//2


@lru_cache(maxsize=4096)
def _occupied_count(bounds, previous=-1):
    '''Count ordered holes without enumerating their Cartesian product.'''
    if not bounds:
        return 1
    start, count = bounds[0]
    if all(bound == bounds[0] for bound in bounds):
        return comb(max(0, start+count-max(start, previous+1)), len(bounds))
    return sum(_occupied_count(bounds[1:], value)
               for value in range(max(start, previous+1), start+count))


def _active_consumers(jobs, batch):
    '''Keep only consumers with a canonical occupied output in this W batch.'''
    active = []
    for item in jobs:
        (rank, na, _, axes, _, _, _, dimensions), _, _ = item
        occupied = [(0, dimensions[i]) for i in range(rank)]
        for label, bounds in zip(axes, batch):
            if label < rank:
                occupied[label] = bounds
        if _occupied_count(tuple(occupied[:na])) and _occupied_count(tuple(occupied[na:])):
            active.append(item)
    return active


class _Executor:
    def __init__(self, mycc, arrays, metas, workspace):
        self.mycc, self.arrays, self.metas = mycc, arrays, metas
        self.gather, self.gemm = _kernels()
        self.fused = _fused_kernel()
        self.threads = lib.num_threads()
        self.blasserial = lib.load_library('libccsdt').UCCSDTQblas_parallel() == 0
        self.buffers = (np.empty(0),)*3
        self.zero_origin = np.zeros(12, dtype=np.int32)
        # At most a parent result, nested child panels, two GEMM inputs and
        # a GEMM output coexist. Reserve separate space for reusable W<=4.
        self.limit = max(1, workspace//(12*8))
        self.cache_limit = workspace//4
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.fixed = np.full(12, -1, dtype=np.int32)
        self.stats = dict(workspace_bytes=workspace, panel_limit_bytes=self.limit*8,
                          cache_limit_bytes=self.cache_limit, max_cache_bytes=0,
                          gemm_calls=0, gemm_flops=0, six_index_batches=0,
                          small_intermediate_builds=0, small_cache_hits=0,
                          max_panel_elements=0, max_intermediate_elements=0,
                          fused_calls=0, mapped_scatters=0, max_workers=0,
                          max_fused_buffer_bytes=0, zero_batches_skipped=0,
                          unconsumed_batches_skipped=0)

    def _read(self, node, axes, ranges, colors, dimensions):
        if node.array is None:
            array, stored_axes = self._small(node, colors, dimensions)
            view = array[tuple(slice(*self._slice(ranges[i])) for i in stored_axes)]
            return np.ascontiguousarray(view.transpose(tuple(stored_axes.index(i) for i in axes)))
        shape = tuple(ranges[i][1] for i in axes)
        result = np.empty(shape)
        self.stats['max_panel_elements'] = max(self.stats['max_panel_elements'], result.size)
        labels = np.asarray(axes, dtype=np.int32)
        starts = np.asarray([ranges[i][0] for i in axes], dtype=np.int32)
        counts = np.asarray(shape, dtype=np.int32)
        factor = direct._Factor(node.array, len(node.labels))
        factor.indices[:len(node.labels)] = node.labels
        status = self.gather(self.arrays[node.array].ctypes.data, ctypes.byref(self.metas[node.array]),
                             ctypes.byref(factor), len(axes), labels.ctypes.data, starts.ctypes.data,
                             counts.ctypes.data, self.fixed.ctypes.data, result.ctypes.data)
        if status:
            raise ValueError('Invalid CCSDTQ gather tile')
        return result

    @staticmethod
    def _slice(bound):
        return bound[0], bound[0]+bound[1]

    def _small(self, node, colors, dimensions):
        key, axes = _canonical(node, colors)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats['small_cache_hits'] += 1
            return self.cache[key], axes
        size = 8*prod(dimensions[i] for i in axes)
        if len(axes) > 4 or size > self.cache_limit:
            raise MemoryError('CCSDTQ needs %.3f MB for a reusable four-index intermediate; '
                              'increase max_memory' % (size/1e6))
        while self.cache and self.cache_bytes+size > self.cache_limit:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= old.nbytes
        ranges = {i: (0, dimensions[i]) for i in axes}
        result = self._evaluate(node, axes, ranges, colors, dimensions)
        self.cache[key] = result
        self.cache_bytes += result.nbytes
        self.stats['small_intermediate_builds'] += 1
        self.stats['max_cache_bytes'] = max(self.stats['max_cache_bytes'], self.cache_bytes)
        return result, axes

    def _evaluate(self, node, axes, ranges, colors, dimensions):
        '''Build a bounded W batch; the only nested intermediates are W<=4.'''
        rows, cols, shared = _matrix_axes(node.left.labels, node.right.labels)
        all_ranges = dict(ranges)
        all_ranges.update((i, (0, dimensions[i])) for i in shared)
        caps = _tile_caps(rows, cols, shared, all_ranges, self.limit)
        output = np.zeros(tuple(ranges[i][1] for i in axes))
        self.stats['max_intermediate_elements'] = max(self.stats['max_intermediate_elements'], output.size)
        for external in _tiles(rows+cols, all_ranges, caps):
            shape = tuple(external[i][1] for i in rows+cols)
            result = np.zeros(shape)
            for internal in _tiles(shared, all_ranges, caps):
                bounds = {**external, **internal}
                a = self._read(node.left, rows+shared, bounds, colors, dimensions)
                b = self._read(node.right, shared+cols, bounds, colors, dimensions)
                self._gemm(a, b, result, rows, cols, shared, bounds)
            index = tuple(slice(external[i][0]-ranges[i][0],
                                external[i][0]-ranges[i][0]+external[i][1]) for i in axes)
            permutation = tuple((rows+cols).index(i) for i in axes)
            output[index] = result.transpose(permutation)
        return output

    def _gemm(self, a, b, c, rows, cols, shared, ranges):
        m, n, k = (prod(ranges[i][1] for i in labels) for labels in (rows, cols, shared))
        status = self.gemm(m, n, k, 1., a.ctypes.data, b.ctypes.data, 1., c.ctypes.data)
        if status:
            raise ValueError('CCSDTQ GEMM dimensions exceed the BLAS integer range')
        self.stats['gemm_calls'] += 1
        self.stats['gemm_flops'] += 2*m*n*k

    def _final(self, job, maps, coefficients, output, value=None, batch=None):
        rank, na, coefficient, axes, left, right, colors, dimensions = job
        rows, cols, shared = _matrix_axes(axes, right.labels)
        ranges = {i: (0, dimensions[i]) for i in rows+cols+shared}
        if batch is not None:
            ranges.update(zip(axes, batch))
        # Bound the count used for choosing threads without enumerating an
        # entire occupied space. Actual canonical tuples are visited in C.
        counta = _occupied_count(tuple(ranges[i] for i in range(na)))
        countb = _occupied_count(tuple(ranges[i] for i in range(na, rank)))
        if not counta*countb:
            return
        workers = min(self.threads, counta*countb, self.limit) if self.blasserial else 1
        volume = prod(ranges[i][1] for i in rows+cols+shared if i >= 4)
        if 2*volume*counta*countb < 100_000:
            workers = 1
        caps = _tile_caps(rows, cols, shared, ranges, max(1, self.limit//workers),
                          occupied_points=True)
        m, n, k = (prod(caps[i] for i in labels) for labels in (rows, cols, shared))
        capacity = max(m*k, k*n, m*n)
        required = workers*capacity
        if self.buffers[0].size < required:
            # Release old workspaces first; no transient double allocation.
            self.buffers = (np.empty(0),)*3
            self.buffers = tuple(np.empty(required) for _ in range(3))
            self.stats['max_fused_buffer_bytes'] = 3*required*8
        plan = _FinalPlan(rank, na, *storage._dims(self.mycc),
                          int(rank >= 3 and output.ndim < 2*rank), workers,
                          len(rows), len(cols), len(shared), len(maps))
        plan.rows[:len(rows)], plan.cols[:len(cols)], plan.shared[:len(shared)] = rows, cols, shared
        for label, (start, count) in ranges.items():
            plan.starts[label], plan.counts[label], plan.caps[label] = start, count, caps[label]
        plan.capacity = capacity
        if value is None:
            left_array, left_meta, left_labels = self.arrays[left.array], self.metas[left.array], left.labels
            origin = self.zero_origin
        else:
            left_array, left_labels = value, axes
            left_meta = direct._Array()
            left_meta.ndim = value.ndim
            left_meta.shape[:value.ndim] = value.shape
            left_meta.stride[:value.ndim] = tuple(s//8 for s in value.strides)
            origin = np.zeros(12, dtype=np.int32)
            for label, bounds in zip(axes, batch):
                origin[label] = bounds[0]
        lf = direct._Factor(0, len(left_labels))
        lf.indices[:len(left_labels)] = left_labels
        rf = direct._Factor(right.array, len(right.labels))
        rf.indices[:len(right.labels)] = right.labels
        strides = np.asarray(output.strides, dtype=np.int64)//8
        stats = np.zeros(6, dtype=np.int64)
        status = self.fused(left_array.ctypes.data, ctypes.byref(left_meta), ctypes.byref(lf), origin.ctypes.data,
                            self.arrays[right.array].ctypes.data, ctypes.byref(self.metas[right.array]),
                            ctypes.byref(rf), self.zero_origin.ctypes.data, ctypes.byref(plan),
                            maps.ctypes.data, coefficients.ctypes.data,
                            *(buffer.ctypes.data for buffer in self.buffers),
                            output.ctypes.data, strides.ctypes.data, stats.ctypes.data)
        if status:
            raise ValueError('Invalid or insufficient UCCSDTQ fused contraction workspace (%d)' % status)
        self.stats['gemm_calls'] += int(stats[0])
        self.stats['gemm_flops'] += int(stats[1])
        self.stats['max_panel_elements'] = max(self.stats['max_panel_elements'], int(stats[2]))
        self.stats['mapped_scatters'] += int(stats[4])
        self.stats['max_workers'] = max(self.stats['max_workers'], int(stats[5]))
        self.stats['fused_calls'] += 1

    def execute(self, groups, out):
        for node, axes, colors, dimensions, jobs in groups:
            if node.array is not None:
                for job, maps, coefficients in jobs:
                    self._final(job, maps, coefficients, out[job[0]-2][job[0]-job[1]])
                continue
            if len(axes) <= 4:
                value, _ = self._small(node, colors, dimensions)
                batch = tuple((0, dimensions[i]) for i in axes)
                for job, maps, coefficients in jobs:
                    self._final(job, maps, coefficients, out[job[0]-2][job[0]-job[1]], value, batch)
                continue
            # The outer loop belongs to W, not R4. Pick occupied axes used
            # as output indices by most consumers, minimizing sliced sums.
            # With serial BLAS, a three-point occupied tile gives the fused
            # kernel enough independent canonical tuples to keep a larger
            # OpenMP team busy.  It deliberately stays disabled for small
            # teams and threaded BLAS: in those cases the extra W6 points
            # would add work without providing useful parallelism.
            occupied = [j for j, i in enumerate(axes) if colors[i][1] == 0]
            occupied.sort(key=lambda j: -sum(job[3][j] < 4 for job, _, _ in jobs))
            caps = {i: dimensions[i] for i in axes}
            occupied_width = 3 if self.blasserial and self.threads >= 4 else 1
            for j in occupied[:2]:
                caps[axes[j]] = min(occupied_width, dimensions[axes[j]])
            while prod(caps.values()) > self.limit:
                largest = max(caps, key=caps.get)
                caps[largest] = (caps[largest]+1)//2
            ranges = {i: (0, dimensions[i]) for i in axes}
            for bounds in _tiles(axes, ranges, caps):
                if _zero_batch(node, bounds, colors):
                    self.stats['zero_batches_skipped'] += 1
                    continue
                batch = tuple(bounds[i] for i in axes)
                consumers = _active_consumers(jobs, batch)
                if not consumers:
                    self.stats['unconsumed_batches_skipped'] += 1
                    continue
                value = self._evaluate(node, axes, bounds, colors, dimensions)
                self.stats['six_index_batches'] += 1
                for job, maps, coefficients in consumers:
                    self._final(job, maps, coefficients, out[job[0]-2][job[0]-job[1]], value, batch)
        return out


def residuals(mycc, imds, t2, t3, t4, out=None, _families=None):
    '''Add factored T4 feedback to R2/R3 and the complete connected R4.

    ``max_memory`` bounds working panels/cache after existing resident arrays
    and returned residuals. No dense spin-orbital T4 is allocated. Six-index
    W batches obey the panel budget; a small W can fit in one occupied batch.
    Four-index intermediates must fit the reusable cache budget.
    ``_families`` is an internal equation-validation selector.
    '''
    dims = storage._dims(mycc)
    oa, ob, va, vb = dims
    if tuple(map(len, (imds.t1_fock, imds.t1_eris, t2, t3, t4))) != (2,3,3,4,5):
        raise ValueError('invalid UCCSDTQ spin-block counts')
    arrays = [np.asarray(a) for blocks in (imds.t1_fock, imds.t1_eris, t2, t3, t4) for a in blocks]
    nma, nmb = oa+va, ob+vb
    shapes = [(nma,nma),(nmb,nmb),(nma,)*4,(nma,nmb,nma,nmb),(nmb,)*4]
    shapes += [storage.block_shape(mycc, r, b, False) for r in (2,3,4) for b in range(r+1)]
    for number, (array, shape) in enumerate(zip(arrays, shapes)):
        valid = array.shape == shape
        if 8 <= number < 12:
            valid |= array.shape == storage.block_shape(mycc, 3, number-8, True)
        elif number >= 12:
            valid |= array.shape == storage.block_shape(mycc, 4, number-12, True)
        if not valid or array.dtype != np.float64:
            raise ValueError('invalid UCCSDTQ input shape or dtype at block %d' % number)
        if not array.flags.aligned:
            arrays[number] = np.require(array, requirements=['A'])
    max_memory = getattr(mycc, 'max_memory', 4000)
    available = int(max(0., max_memory-lib.current_memory()[0])*1e6)
    if available < 4096:
        raise MemoryError('Insufficient max_memory for the UCCSDTQ contraction workspace')
    groups = _plans(dims, None if _families is None else tuple(_families))
    # Cached symbolic plans are part of RSS, not hidden matrix workspace.
    available = int(max(0., max_memory-lib.current_memory()[0])*1e6)
    output_shapes = [[direct._residual_shape(mycc, r, b, t3)
                      for b in range(r+1)] for r in (2,3,4)]
    if out is None:
        available -= 8*sum(prod(s) for shapes in output_shapes for s in shapes)
    if available < 4096:
        raise MemoryError('Insufficient max_memory for the UCCSDTQ contraction workspace')
    # Leave a margin for Python descriptors and allocator/BLAS bookkeeping.
    workspace = available//2
    if out is None:
        out = tuple(tuple(np.zeros(s) for s in shapes) for shapes in output_shapes)
    if tuple(map(len, out)) != (3,4,5):
        raise ValueError('invalid UCCSDTQ output spin-block counts')
    for blocks, shapes in zip(out, output_shapes):
        for array, shape in zip(blocks, shapes):
            if array.shape != shape or array.dtype != np.float64 or not array.flags.writeable:
                raise ValueError('invalid UCCSDTQ output shape, dtype or writeability')
    executor = _Executor(mycc, arrays, direct._metadata(mycc, arrays), workspace)
    result = executor.execute(groups, out)
    mycc._uccsdtq_stats = executor.stats
    return result
