/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Bounded gather, BLAS contraction and scatter primitives for CCSDTQ.
 * The Python planner supplies tile bounds, factorization and cache lifetime.
 * These routines allocate no memory and never form full spin-orbital T4.
 * QFactor and QArray match the checked direct-kernel ABI version 1; the
 * blocked symbols have their own ABI version and leave that engine intact.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__unix__) || defined(__APPLE__))
#define UCCSDTQ_HAVE_DLADDR 1
#include <dlfcn.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stddef.h>
#include <limits.h>
#include "../vhf/fblas.h"

typedef struct {
    int32_t array, ndim, indices[8];
} QFactor;

typedef struct {
    int32_t ndim, hamiltonian, compact, ngroup;
    int64_t shape[8], stride[8];
    int32_t offset[8], group_size[4], group_dim[4];
} QArray;

int UCCSDTQblocked_abi_version(void) { return 1; }

int UCCSDTQomp_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#if defined(UCCSDTQ_HAVE_DLADDR) && !defined(__APPLE__)
extern int openblas_get_parallel(void) __attribute__((weak));
#endif

/* Query only the BLAS providing the actual dgemm symbol. Another OpenBLAS
 * elsewhere in the process must not enable outer threads around MKL, etc. */
int UCCSDTQblas_parallel(void)
{
#ifdef UCCSDTQ_HAVE_DLADDR
#if defined(__APPLE__)
    /* Accelerate and other BLAS providers need not export this optional API.
     * A weak import still requires link-time resolution when building a dylib. */
    int (*get_parallel)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "openblas_get_parallel");
#else
    int (*get_parallel)(void) = openblas_get_parallel;
#endif
    if (get_parallel) {
        Dl_info gemm_info, model_info;
        if (dladdr((void *)dgemm_, &gemm_info) &&
            dladdr((void *)get_parallel, &model_info) &&
            gemm_info.dli_fbase == model_info.dli_fbase)
            return get_parallel();
    }
#endif
    return -1;
}

static int parallel_tile(int64_t total)
{
#ifdef _OPENMP
    return total >= 8192 && !omp_in_parallel();
#else
    return 0;
#endif
}

static int64_t choose(int32_t n, int32_t k)
{
    if (k < 0 || n < k) return 0;
    int64_t value = 1;
    for (int32_t j = 1; j <= k; ++j) value = value * (n-j+1) / j;
    return value;
}

static int virtual_label(int32_t label)
{
    return (label >= 4 && label < 8) || label >= 10;
}

/* Sort one same-spin group and return its sign and packed index. */
static int group_index(const int32_t *values, const int32_t *labels,
                       int32_t size, int32_t dim, int64_t *index)
{
    if (size == 1) { *index = values[labels[0]]; return 1; }
    if (size == 2) {
        int32_t a = values[labels[0]], b = values[labels[1]], sign = 1;
        if (a == b) { *index = 0; return 0; }
        if (a > b) { int32_t temp = a; a = b; b = temp; sign = -1; }
        *index = (int64_t)a*(2*(int64_t)dim-a-1)/2 + b-a-1;
        return sign;
    }
    int32_t ix[4], sign = 1;
    for (int32_t j = 0; j < size; ++j) ix[j] = values[labels[j]];
    for (int32_t j = 1; j < size; ++j) {
        int32_t value = ix[j], k = j;
        while (k > 0 && ix[k-1] > value) {
            ix[k] = ix[k-1];
            --k;
            sign = -sign;
        }
        ix[k] = value;
    }
    *index = 0;
    for (int32_t j = 0; j < size; ++j) {
        if (j && ix[j] == ix[j-1]) return 0;
        int32_t start = j ? ix[j-1]+1 : 0;
        *index += choose(dim-start, size-j) - choose(dim-ix[j], size-j);
    }
    return sign;
}

/* Unit axes have already become fixed values in lower[].  The first varying
 * coordinate is the remaining quotient, requiring no division at all. */
static int varying_axes(int32_t naxis, const int32_t *labels,
                        const int32_t *counts, int32_t *vl, int32_t *vc)
{
    int32_t size = 0;
    for (int32_t axis = 0; axis < naxis; ++axis)
        if (counts[axis] > 1) {
            vl[size] = labels[axis];
            vc[size++] = counts[axis];
        }
    return size;
}

static void decode_tile(int64_t linear, int32_t nvar, const int32_t *labels,
                        const int32_t *counts, const int32_t *base,
                        int32_t *values)
{
    for (int32_t label = 0; label < 12; ++label) values[label] = base[label];
    for (int32_t axis = nvar-1; axis > 0; --axis) {
        values[labels[axis]] += linear % counts[axis];
        linear /= counts[axis];
    }
    if (nvar) values[labels[0]] += linear;
}

/* Return -1 for invalid labels/bounds or a size overflow, without touching
 * output. starts/counts follow the chosen dense tile axes, not label slots. */
static int tile_bounds(int32_t naxis, const int32_t *labels,
                       const int32_t *starts, const int32_t *counts,
                       const int32_t *fixed, int32_t *lower,
                       int32_t *upper, int64_t *total)
{
    if (naxis < 0 || naxis > 12) return -1;
    int32_t seen[12] = {0};
    *total = 1;
    for (int32_t label = 0; label < 12; ++label) lower[label] = upper[label] = fixed[label];
    for (int32_t axis = 0; axis < naxis; ++axis) {
        int32_t label = labels[axis], count = counts[axis], start = starts[axis];
        if (label < 0 || label >= 12 || seen[label] || count < 0 || start < 0) return -1;
        if (count && (*total > INT64_MAX / count || start > INT32_MAX-count+1)) return -1;
        seen[label] = 1;
        *total *= count;
        lower[label] = start;
        upper[label] = start + (count - 1);
    }
    return 0;
}

int UCCSDTQgather(const double *data, const QArray *meta,
                  const QFactor *factor, int32_t naxis,
                  const int32_t *labels, const int32_t *starts,
                  const int32_t *counts, const int32_t *fixed,
                  double *output)
{
    int32_t lower[12], upper[12];
    int64_t total;
    if (tile_bounds(naxis, labels, starts, counts, fixed, lower, upper, &total)) return -1;
    if (!total) return 0;
    if (factor->ndim < 0 || factor->ndim > 8 || meta->ndim < 0 || meta->ndim > 8) return -1;
    int32_t zero_tensor = 0;
    if (meta->compact) {
        if (meta->ngroup < 0 || meta->ngroup > 4 || meta->ndim != meta->ngroup) return -1;
        int32_t axis = 0;
        for (int32_t group = 0; group < meta->ngroup; ++group) {
            int32_t size = meta->group_size[group], dim = meta->group_dim[group];
            if (size < 1 || size > 4 || dim < 0 || meta->shape[group] != choose(dim, size)) return -1;
            if (dim < size) zero_tensor = 1;
            for (int32_t j = 0; j < size; ++j) {
                if (axis >= factor->ndim) return -1;
                int32_t label = factor->indices[axis++];
                if (label < 0 || label >= 12 || lower[label] < 0 || upper[label] >= dim) return -1;
            }
        }
        if (axis != factor->ndim) return -1;
    } else {
        if (meta->ndim != factor->ndim) return -1;
        for (int32_t axis = 0; axis < factor->ndim; ++axis) {
            int32_t label = factor->indices[axis];
            if (label < 0 || label >= 12) return -1;
            int32_t offset = meta->hamiltonian && virtual_label(label) ? meta->offset[axis] : 0;
            if (lower[label] < 0 || (int64_t)upper[label]+offset >= meta->shape[axis]) return -1;
        }
    }
    int32_t vl[12], vc[12];
    int32_t nvar = varying_axes(naxis, labels, counts, vl, vc);
    if (zero_tensor) {
#pragma omp parallel for schedule(static) if(parallel_tile(total))
        for (int64_t linear = 0; linear < total; ++linear) output[linear] = 0.;
    } else if (!meta->compact) {
        int64_t base_offset = 0, label_strides[12] = {0};
        for (int32_t axis = 0; axis < factor->ndim; ++axis) {
            int32_t label = factor->indices[axis];
            int32_t value = lower[label];
            if (meta->hamiltonian && virtual_label(label)) value += meta->offset[axis];
            base_offset += value * meta->stride[axis];
            label_strides[label] += meta->stride[axis];
        }
        /* Decode the outer coordinates once per short row segment. The
         * inner source stride is constant, including for transposed or
         * reversed views, so no integer division is needed per element. */
        const int32_t width = nvar ? vc[nvar-1] : 1;
        const int32_t segments = 1+(width-1)/64;
        const int64_t tasks = (total/width)*segments;
        const int64_t inner_stride = nvar ? label_strides[vl[nvar-1]] : 0;
#pragma omp parallel for schedule(static) if(parallel_tile(total))
        for (int64_t task = 0; task < tasks; ++task) {
            int64_t row = task/segments, remaining = row, offset = base_offset;
            const int32_t first = (task%segments)*64;
            const int32_t length = width-first < 64 ? width-first : 64;
            for (int32_t axis = nvar-2; axis > 0; --axis) {
                offset += (remaining % vc[axis]) * label_strides[vl[axis]];
                remaining /= vc[axis];
            }
            if (nvar > 1) offset += remaining * label_strides[vl[0]];
            offset += first*inner_stride;
            double *destination = output+row*width+first;
            for (int32_t i = 0; i < length; ++i)
                destination[i] = data[offset+i*inner_stride];
        }
    } else {
        int32_t variable_groups[4], group_axes[4], ngroup = 0;
        int32_t axis = 0, fixed_sign = 1;
        int64_t fixed_offset = 0;
        for (int32_t group = 0; group < meta->ngroup; ++group) {
            int32_t size = meta->group_size[group], is_fixed = 1;
            for (int32_t j = 0; j < size; ++j) {
                int32_t label = factor->indices[axis+j];
                if (lower[label] != upper[label]) is_fixed = 0;
            }
            if (is_fixed) {
                int64_t index;
                fixed_sign *= group_index(lower, factor->indices+axis,
                                           size, meta->group_dim[group], &index);
                fixed_offset += index * meta->stride[group];
            } else {
                variable_groups[ngroup] = group;
                group_axes[ngroup++] = axis;
            }
            axis += size;
        }
#pragma omp parallel for schedule(static) if(parallel_tile(total))
        for (int64_t linear = 0; linear < total; ++linear) {
            if (!fixed_sign) { output[linear] = 0.; continue; }
            int32_t values[12], sign = fixed_sign;
            int64_t offset = fixed_offset;
            decode_tile(linear, nvar, vl, vc, lower, values);
            for (int32_t j = 0; j < ngroup && sign; ++j) {
                int32_t group = variable_groups[j];
                int64_t index;
                sign *= group_index(values, factor->indices+group_axes[j],
                                     meta->group_size[group], meta->group_dim[group], &index);
                offset += index * meta->stride[group];
            }
            output[linear] = sign ? sign * data[offset] : 0.;
        }
    }
    return 0;
}

/* Row-major C[m,n] = alpha*A[m,k]*B[k,n] + beta*C[m,n].  The memory
 * transpose changes this into column-major C^T = B^T*A^T without copies.
 * BLAS parallelism is never nested inside an OpenMP gather/scatter region. */
int UCCSDTQgemm(int64_t m64, int64_t n64, int64_t k64, double alpha,
                const double *a, const double *b, double beta, double *c)
{
    if (m64 < 0 || n64 < 0 || k64 < 0 || m64 > INT_MAX || n64 > INT_MAX || k64 > INT_MAX) return -1;
    if (!m64 || !n64) return 0;
    int m = (int)m64, n = (int)n64, k = (int)k64;
    if (!k || alpha == 0.) {
        int64_t total = m64*n64;
#pragma omp parallel for schedule(static) if(parallel_tile(total))
        for (int64_t i = 0; i < total; ++i) c[i] = beta == 0. ? 0. : beta*c[i];
        return 0;
    }
    const char normal = 'N';
#ifdef _OPENMP
    if (m >= 128 && (double)m*n*k >= 5e5 && omp_get_max_threads() > 1 &&
        !omp_in_parallel() && UCCSDTQblas_parallel() == 0) {
        int threads = omp_get_max_threads();
        int64_t rows_per_block = (m64+threads-1)/threads;
        if (rows_per_block < 32) rows_per_block = 32;
        if (rows_per_block > 128) rows_per_block = 128;
        int64_t nblocks = (m64+rows_per_block-1)/rows_per_block;
#pragma omp parallel for schedule(static)
        for (int64_t block = 0; block < nblocks; ++block) {
            int64_t first = block*rows_per_block;
            int rows = (int)(m64-first < rows_per_block ? m64-first : rows_per_block);
            dgemm_(&normal, &normal, &n, &rows, &k, &alpha, b, &n,
                   a+first*k, &k, &beta, c+first*n, &n);
        }
        return 0;
    }
#endif
    dgemm_(&normal, &normal, &n, &m, &k, &alpha, b, &n, a, &k, &beta, c, &n);
    return 0;
}

static int next_permutation(int32_t *a, int32_t n)
{
    int32_t i = n-2;
    while (i >= 0 && a[i] >= a[i+1]) --i;
    if (i < 0) return 0;
    int32_t j = n-1, temp;
    while (a[j] <= a[i]) --j;
    temp = a[i]; a[i] = a[j]; a[j] = temp;
    for (int32_t lo = i+1, hi = n-1; lo < hi; ++lo, --hi) {
        temp = a[lo]; a[lo] = a[hi]; a[hi] = temp;
    }
    return 1;
}

static void scatter_full(double *output, const int32_t *coordinates,
                         const int32_t *sizes, const int64_t *strides,
                         int32_t group, int32_t axis, int64_t offset,
                         double value)
{
    if (group == 4) { output[offset] += value; return; }
    int32_t size = sizes[group], order[4] = {0, 1, 2, 3};
    do {
        int32_t sign = 1;
        for (int32_t i = 0; i < size; ++i)
            for (int32_t j = i+1; j < size; ++j)
                if (order[i] > order[j]) sign = -sign;
        int64_t next_offset = offset;
        for (int32_t i = 0; i < size; ++i)
            next_offset += coordinates[axis+order[i]] * strides[axis+i];
        scatter_full(output, coordinates, sizes, strides, group+1, axis+size,
                     next_offset, sign*value);
    } while (next_permutation(order, size));
}

int UCCSDTQscatter_tile(const double *tile, int32_t naxis,
                        const int32_t *labels, const int32_t *starts,
                        const int32_t *counts, const int32_t *fixed,
                        int32_t rank, int32_t nalpha,
                        int32_t oa, int32_t ob, int32_t va, int32_t vb,
                        int32_t compact, double alpha,
                        double *output, const int64_t *strides)
{
    int32_t lower[12], upper[12];
    int64_t total;
    if (tile_bounds(naxis, labels, starts, counts, fixed, lower, upper, &total)) return -1;
    if (!total) return 0;
    if (rank < 1 || rank > 4 || nalpha < 0 || nalpha > rank) return -1;
    int32_t nbeta = rank-nalpha;
    const int32_t occ[2] = {oa, ob}, vir[2] = {va, vb};
    for (int32_t axis = 0; axis < naxis; ++axis)
        if (!((labels[axis] >= 0 && labels[axis] < rank) ||
              (labels[axis] >= 4 && labels[axis] < 4+rank))) return -1;
    for (int32_t i = 0; i < rank; ++i) {
        int32_t spin = i >= nalpha;
        if (lower[i] < 0 || upper[i] >= occ[spin] || lower[4+i] < 0 || upper[4+i] >= vir[spin]) return -1;
    }
    int32_t majority = nalpha >= nbeta ? 0 : 1;
    int32_t r1 = majority ? nbeta : nalpha, r2 = rank-r1;
    const int32_t sizes[4] = {r1, r1, r2, r2};
    const int32_t dimensions[4] = {occ[majority], vir[majority], occ[1-majority], vir[1-majority]};
    int32_t vl[12], vc[12];
    int32_t nvar = varying_axes(naxis, labels, counts, vl, vc);
    int32_t coordinate_labels[8], fixed_groups[4], group_strides[4];
    int32_t ah = majority ? 2*r1 : 0, av = majority ? 2*r1+r2 : r1;
    int32_t bh = majority ? 0 : 2*r1, bv = majority ? r1 : 2*r1+r2;
    for (int32_t j = 0; j < nalpha; ++j) {
        coordinate_labels[ah+j] = j;
        coordinate_labels[av+j] = 4+j;
    }
    for (int32_t j = 0; j < nbeta; ++j) {
        coordinate_labels[bh+j] = nalpha+j;
        coordinate_labels[bv+j] = 4+nalpha+j;
    }
    int32_t axis = 0, output_axis = 0;
    int64_t fixed_offset = 0;
    for (int32_t group = 0; group < 4; ++group) {
        int32_t size = sizes[group], dim = dimensions[group];
        fixed_groups[group] = 1;
        group_strides[group] = output_axis;
        if (size) ++output_axis;
        for (int32_t j = 0; j < size; ++j) {
            int32_t label = coordinate_labels[axis+j];
            if (lower[label] != upper[label]) fixed_groups[group] = 0;
        }
        if (fixed_groups[group]) {
            int64_t combination = 0;
            for (int32_t j = 0; j < size; ++j) {
                int32_t value = lower[coordinate_labels[axis+j]];
                int32_t previous = j ? lower[coordinate_labels[axis+j-1]] : -1;
                if (value <= previous) return 0;
                if (compact) combination += choose(dim-previous-1, size-j) - choose(dim-value, size-j);
            }
            if (compact && size) fixed_offset += combination * strides[group_strides[group]];
        }
        axis += size;
    }
#pragma omp parallel for schedule(static) if(parallel_tile(total))
    for (int64_t linear = 0; linear < total; ++linear) {
        int32_t values[12], coordinates[8], canonical = 1;
        decode_tile(linear, nvar, vl, vc, lower, values);
        for (int32_t j = 0; j < 2*rank; ++j) coordinates[j] = values[coordinate_labels[j]];
        int32_t axis = 0;
        int64_t packed_offset = fixed_offset;
        for (int32_t group = 0; group < 4 && canonical; ++group) {
            int32_t size = sizes[group], dim = dimensions[group];
            if (!fixed_groups[group]) {
                int64_t combination = 0;
                for (int32_t j = 0; j < size; ++j) {
                    int32_t value = coordinates[axis+j];
                    if (j && value <= coordinates[axis+j-1]) { canonical = 0; break; }
                    int32_t start = j ? coordinates[axis+j-1]+1 : 0;
                    if (compact) combination += choose(dim-start, size-j) - choose(dim-value, size-j);
                }
                if (compact && size) packed_offset += combination * strides[group_strides[group]];
            }
            axis += size;
        }
        if (!canonical) continue;
        double value = alpha*tile[linear];
        if (compact) output[packed_offset] += value;
        else scatter_full(output, coordinates, sizes, strides, 0, 0, 0, value);
    }
    return 0;
}

/* Fused final contraction ABI 1. All label arrays use the same slots as the
 * primitive ABI. Python owns three nworkers*capacity workspaces; C performs
 * no allocation. Distinct canonical occupied tuples own disjoint output
 * images, allowing worker-private GEMM buffers and lock-free accumulation. */
typedef struct {
    int32_t rank, nalpha, oa, ob, va, vb, compact, nworkers;
    int32_t nrows, ncols, nshared, nmaps;
    int32_t rows[8], cols[8], shared[4];
    int32_t starts[12], counts[12], caps[12];
    int64_t capacity;
} QFinal;

int UCCSDTQfused_abi_version(void) { return 1; }

typedef struct {
    const double *left, *right;
    const QArray *left_meta, *right_meta;
    const QFactor *left_factor, *right_factor;
    const int32_t *left_origin, *right_origin;
    const QFinal *plan;
    const int32_t *maps;
    const double *coefficients;
    double *a, *b, *c, *output;
    const int64_t *output_strides;
    int64_t stats[5];
} QFinalContext;

static int64_t final_volume(int32_t size, const int32_t *labels,
                            const int32_t *counts)
{
    int64_t result = 1;
    for (int32_t i = 0; i < size; ++i) {
        int32_t count = counts[labels[i]];
        if (count < 0 || (count && result > INT64_MAX/count)) return -1;
        result *= count;
    }
    return result;
}

static int final_read(const double *data, const QArray *meta,
                       const QFactor *factor, const int32_t *origin,
                       int32_t naxis, const int32_t *labels,
                       const int32_t *starts, const int32_t *counts,
                       double *output)
{
    int32_t local_starts[8], local_counts[8], fixed[12];
    for (int32_t i = 0; i < 12; ++i) fixed[i] = -1;
    for (int32_t i = 0; i < naxis; ++i) {
        int32_t label = labels[i];
        local_starts[i] = starts[label]-origin[label];
        local_counts[i] = counts[label];
    }
    return UCCSDTQgather(data, meta, factor, naxis, labels,
                         local_starts, local_counts, fixed, output);
}

/* Pruning is tested against every virtual map; the representative map alone
 * can be noncanonical while another mapped contribution is required. */
static int final_possible(const QFinalContext *ctx, const int32_t *starts,
                           const int32_t *counts)
{
    const QFinal *plan = ctx->plan;
    for (int32_t map = 0; map < plan->nmaps; ++map) {
        if (ctx->coefficients[map] == 0.) continue;
        int32_t lo[4], hi[4], possible = 1;
        for (int32_t i = 0; i < plan->rank; ++i) {
            int32_t target = ctx->maps[12*map+4+i]-4;
            lo[target] = starts[4+i];
            hi[target] = starts[4+i]+counts[4+i]-1;
        }
        for (int32_t i = 1; i < plan->rank; ++i)
            if (i != plan->nalpha && lo[i-1] >= hi[i]) possible = 0;
        if (possible) return 1;
    }
    return 0;
}

static void final_unrank(int64_t index, int32_t dim, int32_t size, int32_t *values)
{
    if (size == 1) { values[0] = (int32_t)index; return; }
    int32_t previous = -1;
    for (int32_t j = 0; j < size; ++j) {
        int32_t value = previous+1;
        for (;;) {
            int64_t count = choose(dim-value-1, size-j-1);
            if (index < count) break;
            index -= count;
            ++value;
        }
        values[j] = value;
        previous = value;
    }
}

/* A full virtual rectangle contains every mapped value needed by each
 * canonical virtual tuple. Read all maps directly, combine once and write
 * the compact output (or its full antisymmetric images) once. No mapped
 * Cartesian scan or temporary transpose is needed. */
static void final_scatter_full_virtual(QFinalContext *ctx, const int32_t *occupied,
                                       const int32_t *output_labels,
                                       const int32_t *counts)
{
    const QFinal *plan = ctx->plan;
    const int32_t na = plan->nalpha, nb = plan->rank-na;
    const int32_t majority = na >= nb ? 0 : 1;
    const int32_t r1 = majority ? nb : na, r2 = plan->rank-r1;
    const int32_t sizes[4] = {r1, r1, r2, r2};
    const int32_t ah = majority ? 2*r1 : 0, av = majority ? 2*r1+r2 : r1;
    const int32_t bh = majority ? 0 : 2*r1, bv = majority ? r1 : 2*r1+r2;
    const int32_t alabels[4] = {0,1,2,3};
    int32_t blabels[4] = {0,0,0,0};
    for (int32_t i = 0; i < nb; ++i) blabels[i] = na+i;
    int64_t ha, hb;
    group_index(occupied, alabels, na, plan->oa, &ha);
    group_index(occupied, blabels, nb, plan->ob, &hb);
    int64_t tile_strides[12] = {0}, stride = 1;
    for (int32_t i = 2*plan->rank-1; i >= 0; --i) {
        int32_t label = output_labels[i];
        tile_strides[label] = stride;
        stride *= counts[label];
    }
    const int64_t nca = choose(plan->va, na), ncb = choose(plan->vb, nb);
#pragma omp parallel for schedule(static) if(parallel_tile(nca*ncb))
    for (int64_t ca = 0; ca < nca; ++ca) {
        int32_t alpha[4], beta[4], values[12] = {0}, coordinates[8];
        final_unrank(ca, plan->va, na, alpha);
        for (int32_t j = 0; j < na; ++j) {
            values[4+j] = coordinates[av+j] = alpha[j];
            coordinates[ah+j] = occupied[j];
        }
        for (int32_t j = 0; j < nb; ++j) coordinates[bh+j] = occupied[na+j];
        for (int64_t cb = 0; cb < ncb; ++cb) {
            final_unrank(cb, plan->vb, nb, beta);
            for (int32_t j = 0; j < nb; ++j)
                values[4+na+j] = coordinates[bv+j] = beta[j];
            double value = 0.;
            for (int32_t map = 0; map < plan->nmaps; ++map) {
                if (ctx->coefficients[map] == 0.) continue;
                int64_t source = 0;
                for (int32_t i = 4; i < 4+plan->rank; ++i)
                    source += values[ctx->maps[12*map+i]] * tile_strides[i];
                value += ctx->coefficients[map] * ctx->c[source];
            }
            if (plan->compact) {
                int64_t offset;
                if (!majority) {
                    offset = ha*ctx->output_strides[0] + ca*ctx->output_strides[1];
                    if (nb) offset += hb*ctx->output_strides[2] + cb*ctx->output_strides[3];
                } else {
                    offset = hb*ctx->output_strides[0] + cb*ctx->output_strides[1];
                    if (na) offset += ha*ctx->output_strides[2] + ca*ctx->output_strides[3];
                }
                ctx->output[offset] += value;
            } else scatter_full(ctx->output, coordinates, sizes, ctx->output_strides,
                                 0, 0, 0, value);
        }
    }
    for (int32_t map = 0; map < plan->nmaps; ++map)
        if (ctx->coefficients[map] != 0.) ++ctx->stats[4];
}

static int final_occupied(QFinalContext *ctx, const int32_t *occupied)
{
    const QFinal *plan = ctx->plan;
    int32_t output_labels[8], a_labels[8], b_labels[8];
    int32_t starts[12], counts[12], virtual_labels[4], virtual_tiles[4];
    int32_t internal_tiles[4], nvirtual = 0;
    int64_t nexternal = 1, ninternal = 1;
    for (int32_t i = 0; i < 12; ++i) {
        starts[i] = plan->starts[i];
        counts[i] = plan->counts[i];
    }
    for (int32_t i = 0; i < plan->rank; ++i) {
        starts[i] = occupied[i];
        counts[i] = 1;
    }
    for (int32_t i = 0; i < plan->nrows; ++i)
        output_labels[i] = a_labels[i] = plan->rows[i];
    for (int32_t i = 0; i < plan->ncols; ++i) {
        output_labels[plan->nrows+i] = plan->cols[i];
        b_labels[plan->nshared+i] = plan->cols[i];
    }
    for (int32_t i = 0; i < plan->nshared; ++i) {
        int32_t label = plan->shared[i];
        a_labels[plan->nrows+i] = b_labels[i] = label;
        internal_tiles[i] = 1+(plan->counts[label]-1)/plan->caps[label];
        ninternal *= internal_tiles[i];
    }
    for (int32_t i = 0; i < 2*plan->rank; ++i) {
        int32_t label = output_labels[i];
        if (label >= 4) {
            virtual_labels[nvirtual] = label;
            virtual_tiles[nvirtual] = 1+(plan->counts[label]-1)/plan->caps[label];
            nexternal *= virtual_tiles[nvirtual++];
        }
    }
    ++ctx->stats[3];
    for (int64_t external = 0; external < nexternal; ++external) {
        int64_t remaining = external;
        for (int32_t i = nvirtual-1; i >= 0; --i) {
            int32_t label = virtual_labels[i];
            int64_t offset = (remaining % virtual_tiles[i]) * plan->caps[label];
            remaining /= virtual_tiles[i];
            starts[label] = plan->starts[label]+offset;
            counts[label] = plan->counts[label]-offset < plan->caps[label] ?
                            plan->counts[label]-offset : plan->caps[label];
        }
        if (!final_possible(ctx, starts, counts)) continue;
        int64_t m = final_volume(plan->nrows, plan->rows, counts);
        int64_t n = final_volume(plan->ncols, plan->cols, counts);
        int64_t csize = m*n;
        if (csize > ctx->stats[2]) ctx->stats[2] = csize;
        for (int64_t i = 0; i < csize; ++i) ctx->c[i] = 0.;
        for (int64_t internal = 0; internal < ninternal; ++internal) {
            remaining = internal;
            for (int32_t i = plan->nshared-1; i >= 0; --i) {
                int32_t label = plan->shared[i];
                int64_t offset = (remaining % internal_tiles[i]) * plan->caps[label];
                remaining /= internal_tiles[i];
                starts[label] = plan->starts[label]+offset;
                counts[label] = plan->counts[label]-offset < plan->caps[label] ?
                                plan->counts[label]-offset : plan->caps[label];
            }
            int64_t k = final_volume(plan->nshared, plan->shared, counts);
            if (m*k > ctx->stats[2]) ctx->stats[2] = m*k;
            if (k*n > ctx->stats[2]) ctx->stats[2] = k*n;
            if (final_read(ctx->left, ctx->left_meta, ctx->left_factor,
                            ctx->left_origin, plan->nrows+plan->nshared,
                            a_labels, starts, counts, ctx->a) ||
                final_read(ctx->right, ctx->right_meta, ctx->right_factor,
                            ctx->right_origin, plan->nshared+plan->ncols,
                            b_labels, starts, counts, ctx->b)) return -1;
            if (UCCSDTQgemm(m, n, k, 1., ctx->a, ctx->b, 1., ctx->c)) return -1;
            ++ctx->stats[0];
            ctx->stats[1] += 2*m*n*k;
        }
        int full_virtual = 1;
        for (int32_t i = 0; i < plan->rank; ++i) {
            int32_t dim = i < plan->nalpha ? plan->va : plan->vb;
            if (starts[4+i] || counts[4+i] != dim) full_virtual = 0;
        }
        if (full_virtual) {
            final_scatter_full_virtual(ctx, occupied, output_labels, counts);
            continue;
        }
        int32_t mapped_labels[8], output_starts[8], output_counts[8], fixed[12];
        for (int32_t i = 0; i < 12; ++i) fixed[i] = -1;
        for (int32_t i = 0; i < 2*plan->rank; ++i) {
            output_starts[i] = starts[output_labels[i]];
            output_counts[i] = counts[output_labels[i]];
        }
        for (int32_t map = 0; map < plan->nmaps; ++map) {
            if (ctx->coefficients[map] == 0.) continue;
            for (int32_t i = 0; i < 2*plan->rank; ++i)
                mapped_labels[i] = ctx->maps[12*map+output_labels[i]];
            if (UCCSDTQscatter_tile(ctx->c, 2*plan->rank, mapped_labels,
                    output_starts, output_counts, fixed, plan->rank, plan->nalpha,
                    plan->oa, plan->ob, plan->va, plan->vb, plan->compact,
                    ctx->coefficients[map], ctx->output, ctx->output_strides)) return -1;
            ++ctx->stats[4];
        }
    }
    return 0;
}

int UCCSDTQfinal_contract(
    const double *left, const QArray *left_meta, const QFactor *left_factor,
    const int32_t *left_origin,
    const double *right, const QArray *right_meta, const QFactor *right_factor,
    const int32_t *right_origin,
    const QFinal *plan, const int32_t *maps, const double *coefficients,
    double *awork, double *bwork, double *cwork,
    double *output, const int64_t *output_strides, int64_t *statistics)
{
    for (int32_t i = 0; i < 6; ++i) statistics[i] = 0;
    if (plan->rank < 1 || plan->rank > 4 || plan->nalpha < 0 || plan->nalpha > plan->rank ||
        plan->nrows < 0 || plan->ncols < 0 || plan->nrows+plan->ncols != 2*plan->rank ||
        plan->nshared < 0 || plan->nshared > 4 || plan->nrows+plan->nshared > 8 ||
        plan->ncols+plan->nshared > 8 || plan->nmaps < 0 || plan->nworkers < 1 ||
        plan->capacity < 1) return -1;
    if (!plan->nmaps) return 0;
    int32_t seen[12] = {0}, acounts[12] = {0};
    int64_t occupied_tasks = 1;
    for (int32_t i = 0; i < plan->nrows+plan->ncols+plan->nshared; ++i) {
        int32_t label = i < plan->nrows ? plan->rows[i] :
                        i < plan->nrows+plan->ncols ? plan->cols[i-plan->nrows] :
                        plan->shared[i-plan->nrows-plan->ncols];
        if (label < 0 || label >= 12 || seen[label] || plan->counts[label] < 0 ||
            plan->starts[label] < 0 || plan->caps[label] < 1 ||
            plan->starts[label] > INT32_MAX-plan->counts[label]) return -1;
        if (i < plan->nrows+plan->ncols) {
            if (!(label < plan->rank || (label >= 4 && label < 4+plan->rank))) return -1;
        } else if (label < 8) return -1;
        if (!plan->counts[label]) return 0;
        seen[label] = 1;
        acounts[label] = label < 4 ? 1 :
                        plan->counts[label] < plan->caps[label] ? plan->counts[label] : plan->caps[label];
    }
    if (left_factor->ndim != plan->nrows+plan->nshared ||
        right_factor->ndim != plan->ncols+plan->nshared) return -1;
    int32_t left_seen[12] = {0}, right_seen[12] = {0};
    for (int32_t i = 0; i < left_factor->ndim; ++i) {
        int32_t label = left_factor->indices[i];
        if (label < 0 || label >= 12 || left_seen[label] || !seen[label] || left_origin[label] < 0) return -1;
        left_seen[label] = 1;
    }
    for (int32_t i = 0; i < right_factor->ndim; ++i) {
        int32_t label = right_factor->indices[i];
        if (label < 0 || label >= 12 || right_seen[label] || !seen[label] || right_origin[label] < 0) return -1;
        right_seen[label] = 1;
    }
    for (int32_t i = 0; i < plan->nrows; ++i)
        if (!left_seen[plan->rows[i]] || right_seen[plan->rows[i]]) return -1;
    for (int32_t i = 0; i < plan->ncols; ++i)
        if (!right_seen[plan->cols[i]] || left_seen[plan->cols[i]]) return -1;
    for (int32_t i = 0; i < plan->nshared; ++i)
        if (!left_seen[plan->shared[i]] || !right_seen[plan->shared[i]]) return -1;
    const int32_t occ[2] = {plan->oa, plan->ob}, vir[2] = {plan->va, plan->vb};
    for (int32_t i = 0; i < plan->rank; ++i) {
        int32_t spin = i >= plan->nalpha;
        if (!seen[i] || !seen[4+i] || plan->starts[i]+plan->counts[i] > occ[spin] ||
            plan->starts[4+i]+plan->counts[4+i] > vir[spin]) return -1;
        if (occupied_tasks > INT64_MAX/plan->counts[i]) return -1;
        occupied_tasks *= plan->counts[i];
    }
    int active_maps = 0;
    for (int32_t map = 0; map < plan->nmaps; ++map) {
        int32_t used[12] = {0};
        for (int32_t label = 0; label < 12; ++label) {
            int32_t target = maps[12*map+label];
            if (label >= 4 && label < 4+plan->rank) {
                if (target < 4 || target >= 4+plan->rank || used[target] ||
                    (label-4 < plan->nalpha) != (target-4 < plan->nalpha)) return -1;
            } else if (target != label) return -1;
            used[target] = 1;
        }
        if (coefficients[map] != 0.) ++active_maps;
    }
    if (!active_maps) return 0;
    int64_t m = final_volume(plan->nrows, plan->rows, acounts);
    int64_t n = final_volume(plan->ncols, plan->cols, acounts);
    int64_t k = final_volume(plan->nshared, plan->shared, acounts);
    if (m < 1 || n < 1 || k < 1 || m > INT_MAX || n > INT_MAX || k > INT_MAX ||
        m > plan->capacity/n || m > plan->capacity/k || k > plan->capacity/n) return -2;
    int workers = 1;
#ifdef _OPENMP
    if (!omp_in_parallel() && UCCSDTQblas_parallel() == 0) {
        workers = plan->nworkers;
        if (workers > omp_get_max_threads()) workers = omp_get_max_threads();
        if (workers > occupied_tasks) workers = (int)occupied_tasks;
    }
#endif
    int status = 0, actual_workers = 1;
    int64_t calls = 0, flops = 0, maxpanel = 0, cases = 0, scatters = 0;
#pragma omp parallel for schedule(dynamic,1) num_threads(workers) if(workers > 1) \
    reduction(min:status) reduction(max:maxpanel,actual_workers) reduction(+:calls,flops,cases,scatters)
    for (int64_t task = 0; task < occupied_tasks; ++task) {
        /* Distribute whole tuples, not one occupied index. Each canonical
         * tuple owns a disjoint output orbit, including full-storage images.
         * Invalid Cartesian tuples are skipped before packing or GEMM. */
        int64_t remaining = task;
        int32_t occupied[4], canonical = 1;
        for (int32_t i = plan->rank-1; i >= 0; --i) {
            occupied[i] = plan->starts[i] + remaining % plan->counts[i];
            remaining /= plan->counts[i];
        }
        for (int32_t i = 1; i < plan->rank; ++i)
            if (i != plan->nalpha && occupied[i] <= occupied[i-1]) canonical = 0;
        if (!canonical) continue;
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
        actual_workers = omp_get_num_threads();
#endif
        QFinalContext ctx = {left, right, left_meta, right_meta, left_factor, right_factor,
                             left_origin, right_origin, plan, maps, coefficients,
                             awork+thread*plan->capacity, bwork+thread*plan->capacity,
                             cwork+thread*plan->capacity, output, output_strides,
                             {0,0,0,0,0}};
        int result = final_occupied(&ctx, occupied);
        if (result < status) status = result;
        calls += ctx.stats[0];
        flops += ctx.stats[1];
        if (ctx.stats[2] > maxpanel) maxpanel = ctx.stats[2];
        cases += ctx.stats[3];
        scatters += ctx.stats[4];
    }
    statistics[0] = calls;
    statistics[1] = flops;
    statistics[2] = maxpanel;
    statistics[3] = cases;
    statistics[4] = scatters;
    statistics[5] = actual_workers;
    return status;
}
