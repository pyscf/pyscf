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

/* Direct connected CCSDTQ contractions.  Python supplies exact rational
 * spin-summed descriptors derived from independent Wick enumeration.
 * Each OpenMP iteration owns one independent amplitude and its full-storage
 * antisymmetric images.  Workspace is fixed-size thread-local stack storage;
 * packed T4 is read directly, without constructing dense T4 or spin orbitals.
 */
#include <stdint.h>
#include <stddef.h>

int UCCSDTQabi_version(void) { return 1; }

/* Axis energies cover either full indices or packed same-spin combinations.
 * Decode each row once, then divide its last axis without index division or
 * a denominator tensor.  Strides are in doubles and may be non-contiguous. */
void UCCSDTQdivide_r4_(double *residual, const double **energies,
                      int32_t ndim, const int64_t *shape, const int64_t *strides)
{
    int64_t rows = 1;
    for (int32_t axis = 0; axis < ndim-1; ++axis) rows *= shape[axis];
    const int64_t last = shape[ndim-1], step = strides[ndim-1];
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        int64_t index = row, offset = 0, coordinates[8];
        for (int32_t axis = ndim-2; axis >= 0; --axis) {
            coordinates[axis] = index % shape[axis];
            index /= shape[axis];
            offset += coordinates[axis] * strides[axis];
        }
        double prefix = 0.;
        for (int32_t axis = 0; axis < ndim-1; ++axis)
            prefix += energies[axis][coordinates[axis]];
        for (int64_t column = 0; column < last; ++column)
            residual[offset+column*step] /= prefix + energies[ndim-1][column];
    }
}

typedef struct {
    int32_t array, ndim, indices[8];
} QFactor;

typedef struct {
    double coefficient;
    int32_t nfactor, ndummy, dummy[4], spin[4];
    QFactor factor[4];
} QTerm;

typedef struct {
    int32_t ndim, hamiltonian, compact, ngroup;
    int64_t shape[8], stride[8];
    int32_t offset[8], group_size[4], group_dim[4];
} QArray;

static int64_t choose(int32_t n, int32_t k)
{
    if (k < 0 || n < k) return 0;
    int64_t result = 1;
    for (int32_t j = 1; j <= k; ++j) result = result * (n-j+1) / j;
    return result;
}

static void unrank_combination(int64_t index, int32_t n, int32_t k, int32_t *out)
{
    int32_t previous = -1;
    for (int32_t j = 0; j < k; ++j) {
        int32_t value = previous + 1;
        for (;;) {
            int64_t count = choose(n-value-1, k-j-1);
            if (index < count) break;
            index -= count;
            ++value;
        }
        out[j] = value;
        previous = value;
    }
}

static double tensor_value(const double *data, const QArray *meta,
                           const QFactor *factor, const int32_t *values)
{
    if (!meta->compact) {
        int64_t offset = 0;
        for (int32_t axis = 0; axis < factor->ndim; ++axis) {
            int32_t label = factor->indices[axis];
            int32_t value = values[label];
            if (meta->hamiltonian && ((label >= 4 && label < 8) || label >= 10))
                value += meta->offset[axis];
            offset += value * meta->stride[axis];
        }
        return data[offset];
    }
    int32_t sign = 1, axis = 0;
    int64_t offset = 0;
    for (int32_t group = 0; group < meta->ngroup; ++group) {
        int32_t size = meta->group_size[group], dim = meta->group_dim[group];
        int32_t ix[4];
        for (int32_t j = 0; j < size; ++j)
            ix[j] = values[factor->indices[axis++]];
        for (int32_t j = 1; j < size; ++j) {
            int32_t value = ix[j], k = j;
            while (k > 0 && ix[k-1] > value) {
                ix[k] = ix[k-1];
                --k;
                sign = -sign;
            }
            ix[k] = value;
        }
        int64_t rank = 0;
        for (int32_t j = 0; j < size; ++j) {
            if (j && ix[j] == ix[j-1]) return 0.;
            int32_t start = j ? ix[j-1]+1 : 0;
            rank += choose(dim-start, size-j) - choose(dim-ix[j], size-j);
        }
        offset += rank * meta->stride[group];
    }
    return sign * data[offset];
}

static double contract(const double **arrays, const QArray *metas,
                       const QTerm *term, int32_t *values,
                       const int32_t *occ, const int32_t *vir)
{
    int32_t dimensions[4] = {1, 1, 1, 1};
    for (int32_t j = 0; j < term->ndummy; ++j)
        dimensions[j] = term->dummy[j] < 10 ? occ[term->spin[j]] : vir[term->spin[j]];
    double result = 0.;
    for (int32_t a = 0; a < dimensions[0]; ++a) {
        if (term->ndummy > 0) values[term->dummy[0]] = a;
        for (int32_t b = 0; b < dimensions[1]; ++b) {
            if (term->ndummy > 1) values[term->dummy[1]] = b;
            for (int32_t c = 0; c < dimensions[2]; ++c) {
                if (term->ndummy > 2) values[term->dummy[2]] = c;
                for (int32_t d = 0; d < dimensions[3]; ++d) {
                    if (term->ndummy > 3) values[term->dummy[3]] = d;
                    double product = term->coefficient;
                    for (int32_t factor = 0; factor < term->nfactor; ++factor) {
                        int32_t array = term->factor[factor].array;
                        product *= tensor_value(arrays[array], metas+array,
                                                term->factor+factor, values);
                        if (product == 0.) break;
                    }
                    result += product;
                }
            }
        }
    }
    return result;
}

static int next_permutation(int32_t *a, int32_t n)
{
    int32_t i = n - 2;
    while (i >= 0 && a[i] >= a[i+1]) --i;
    if (i < 0) return 0;
    int32_t j = n - 1, temp;
    while (a[j] <= a[i]) --j;
    temp = a[i]; a[i] = a[j]; a[j] = temp;
    for (int32_t lo = i+1, hi = n-1; lo < hi; ++lo, --hi) {
        temp = a[lo]; a[lo] = a[hi]; a[hi] = temp;
    }
    return 1;
}

static void scatter(double *output, const int32_t *coordinates,
                    const int32_t *sizes, const int32_t *dimensions,
                    const int64_t *strides, int32_t group, int32_t axis,
                    int64_t offset, double value)
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
        scatter(output, coordinates, sizes, dimensions, strides, group+1, axis+size,
                next_offset, sign*value);
    } while (next_permutation(order, size));
}

void UCCSDTQresidual_block(const double **arrays, const QArray *metas,
                          const QTerm *terms, int64_t nterms,
                          int32_t oa, int32_t ob, int32_t va, int32_t vb,
                          int32_t rank, int32_t nalpha, int32_t compact,
                          double *output, const int64_t *output_strides)
{
    const int32_t occ[2] = {oa, ob}, vir[2] = {va, vb};
    int32_t nbeta = rank - nalpha;
    int32_t majority = nalpha >= nbeta ? 0 : 1;
    int32_t r1 = majority ? nbeta : nalpha, r2 = rank-r1;
    const int32_t sizes[4] = {r1, r1, r2, r2};
    const int32_t dimensions[4] = {occ[majority], vir[majority],
                                   occ[1-majority], vir[1-majority]};
    int64_t counts[4], total = 1;
    for (int32_t group = 0; group < 4; ++group) {
        counts[group] = choose(dimensions[group], sizes[group]);
        total *= counts[group];
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t linear = 0; linear < total; ++linear) {
        int32_t coordinates[8], values[12] = {0};
        int64_t current = linear, combination[4];
        for (int32_t group = 3; group >= 0; --group) {
            combination[group] = current % counts[group];
            current /= counts[group];
        }
        int32_t axis = 0;
        for (int32_t group = 0; group < 4; ++group) {
            unrank_combination(combination[group], dimensions[group], sizes[group], coordinates+axis);
            axis += sizes[group];
        }
        int32_t ah = majority ? 2*r1 : 0, av = majority ? 2*r1+r2 : r1;
        int32_t bh = majority ? 0 : 2*r1, bv = majority ? r1 : 2*r1+r2;
        for (int32_t j = 0; j < nalpha; ++j) {
            values[j] = coordinates[ah+j];
            values[4+j] = coordinates[av+j];
        }
        for (int32_t j = 0; j < nbeta; ++j) {
            values[nalpha+j] = coordinates[bh+j];
            values[4+nalpha+j] = coordinates[bv+j];
        }
        double result = 0.;
        for (int64_t term = 0; term < nterms; ++term)
            result += contract(arrays, metas, terms+term, values, occ, vir);
        if (compact) {
            int64_t offset = 0;
            int32_t output_axis = 0;
            for (int32_t group = 0; group < 4; ++group)
                if (sizes[group]) offset += combination[group] * output_strides[output_axis++];
            output[offset] += result;
        } else scatter(output, coordinates, sizes, dimensions, output_strides, 0, 0, 0, result);
    }
}
