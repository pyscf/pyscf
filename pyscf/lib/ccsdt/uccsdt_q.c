/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/* Spin-blocked UCCSDT[Q]/(Q) moments and energies.  Python supplies the exact
 * antisymmetrized seed descriptors; this file performs all tensor reading,
 * contraction, denominator application and energy reduction for a task.
 *
 * A task fixes four occupied indices and four virtual ranges.  In every
 * seed each free label occurs once and each dummy label twice.  Therefore
 * its contraction is one matrix product, or two products along a chain.
 * Three-factor seeds MUST contract an edge of this chain first: evaluating
 * two dummy sums independently for every output would add an extra power
 * of system size.  No full T3 unpacking or full T4 allocation is needed.
 *
 * OpenMP distributes independent tasks.  The small blocked matrix kernel
 * below vectorizes within a worker and never enters a nested BLAS thread
 * team.  Each task has a fixed arithmetic order regardless of its worker.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NLAB 12
#define TERM_SIZE 35
#define TASK_SIZE 13
#define NBUF 7

/* Label IDs: ijklabcdmnef.  Array order: ERI aa,ab,bb; T2 aa,ab,bb;
 * T3 aaa,aab,bba,bbb.  Strides are signed counts of double elements. */
typedef struct {
    const double *const *data;
    const int64_t *strides;
    const int64_t *sizes;
    int compact;
    int64_t dim[NLAB], base[NLAB];
} Context;

typedef struct {
    int kind, rank, block, sign;
    int labels[6], spins[6], order[6];
    int direct, exchange;
} Leaf;

typedef struct {
    const Leaf *leaf;
    const double *data;
    int nlabels, labels[NLAB];
} Tensor;

typedef struct {
    int nleft, nright, nout;
    int left[NLAB], right[NLAB], out[NLAB];
    size_t m, n, k;
} Product;

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (b && a > SIZE_MAX / b) {
        return -2;
    }
    *out = a * b;
    return 0;
}

static int has_label(const Tensor *t, int label)
{
    for (int i = 0; i < t->nlabels; i++) {
        if (t->labels[i] == label) {
            return 1;
        }
    }
    return 0;
}

static int product_plan(const Tensor *a, const Tensor *b,
                        const Context *ctx, Product *p)
{
    p->nleft = p->nright = p->nout = 0;
    p->m = p->n = p->k = 1;
    for (int i = 0; i < a->nlabels; i++) {
        int label = a->labels[i];
        if (!has_label(b, label)) {
            p->left[p->nleft++] = label;
            p->out[p->nout++] = label;
            if (mul_size(p->m, ctx->dim[label], &p->m)) {
                return -2;
            }
        }
    }
    for (int i = 0; i < a->nlabels; i++) {
        int label = a->labels[i];
        if (has_label(b, label)) {
            p->left[p->nleft++] = label;
            p->right[p->nright++] = label;
            if (mul_size(p->k, ctx->dim[label], &p->k)) {
                return -2;
            }
        }
    }
    for (int i = 0; i < b->nlabels; i++) {
        int label = b->labels[i];
        if (!has_label(a, label)) {
            p->right[p->nright++] = label;
            p->out[p->nout++] = label;
            if (mul_size(p->n, ctx->dim[label], &p->n)) {
                return -2;
            }
        }
    }
    return 0;
}

static Tensor product_tensor(const Product *p, const double *data)
{
    Tensor t;
    t.leaf = NULL;
    t.data = data;
    t.nlabels = p->nout;
    memcpy(t.labels, p->out, sizeof(int) * p->nout);
    return t;
}

static int permutation_sign(const int *order, int n)
{
    int sign = 1;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (order[i] > order[j]) {
                sign = -sign;
            }
        }
    }
    return sign;
}

static int prepare_term(const int32_t *term, Leaf *leaves, Tensor *tensors)
{
    int nf = term[1];
    if (nf < 2 || nf > 3 || term[0] < 0 || term[0] > 5) {
        return -2;
    }
    for (int f = 0; f < nf; f++) {
        const int32_t *src = term + 14 + 7*f;
        Leaf *leaf = leaves + f;
        Tensor *t = tensors + f;
        leaf->kind = src[0];
        leaf->rank = leaf->kind == 3 ? 3 : 2;
        if (leaf->kind != 0 && leaf->kind != 2 && leaf->kind != 3) {
            return -2;
        }
        t->leaf = leaf;
        t->data = NULL;
        t->nlabels = 0;
        int na = 0, nv = 0;
        for (int i = 0; i < 2*leaf->rank; i++) {
            int label = src[1+i];
            if (label < 0 || label >= NLAB || term[2+label] < 0 ||
                term[2+label] > 1) {
                return -2;
            }
            leaf->labels[i] = label;
            leaf->spins[i] = term[2+label];
            if (label >= 4) {
                t->labels[t->nlabels++] = label;
            }
            if (i < leaf->rank) {
                na += leaf->spins[i] == 0;
            } else {
                nv += leaf->spins[i] == 0;
            }
        }
        if (leaf->kind == 0) {
            leaf->direct = leaf->spins[0] == leaf->spins[2] &&
                           leaf->spins[1] == leaf->spins[3];
            leaf->exchange = leaf->spins[0] == leaf->spins[3] &&
                             leaf->spins[1] == leaf->spins[2];
            if (!leaf->direct && !leaf->exchange) {
                return -2;
            }
            continue;
        }
        if (na != nv) {
            return -2;
        }
        int rank = leaf->rank;
        int majority = na >= (rank+1)/2 ? 0 : 1;
        int op[3], vp[3], no = 0, nvp = 0;
        for (int spin = 0; spin < 2; spin++) {
            for (int i = 0; i < rank; i++) {
                if (leaf->spins[i] == (majority ^ spin)) {
                    op[no++] = i;
                }
                if (leaf->spins[rank+i] == (majority ^ spin)) {
                    vp[nvp++] = i;
                }
            }
        }
        leaf->sign = permutation_sign(op, rank) * permutation_sign(vp, rank);
        leaf->block = rank == 2 ? 3+rank-na : 6+rank-na;
        for (int i = 0; i < rank; i++) {
            leaf->order[i] = op[i];
            leaf->order[rank+i] = rank+vp[i];
        }
        if (rank == 2 && na == 1) {
            leaf->order[0] = op[0];
            leaf->order[1] = rank+vp[0];
            leaf->order[2] = op[1];
            leaf->order[3] = rank+vp[1];
        } else if (rank == 3 && na != 0 && na != 3) {
            leaf->order[0] = op[0];
            leaf->order[1] = op[1];
            leaf->order[2] = rank+vp[0];
            leaf->order[3] = rank+vp[1];
            leaf->order[4] = op[2];
            leaf->order[5] = rank+vp[2];
        }
    }
    return 0;
}

static double strided_read(const Context *ctx, int block,
                           const int64_t *indices, int ndim)
{
    const int64_t *strides = ctx->strides + 6*block;
    int64_t offset = 0;
    for (int i = 0; i < ndim; i++) {
        offset += indices[i] * strides[i];
    }
    return ctx->data[block][offset];
}

/* Return the lexicographic packed pair/triple and its antisymmetric sign.
 * The formulas are identical to Python _packed_indices. */
static int packed_index(int64_t n, const int64_t *src, int rank, int64_t *index)
{
    int64_t x[3] = {src[0], src[1], rank == 3 ? src[2] : 0};
    int sign = 1;
    for (int i = 0; i < rank; i++) {
        for (int j = i+1; j < rank; j++) {
            if (x[i] == x[j]) {
                *index = 0;
                return 0;
            }
            if (x[i] > x[j]) {
                int64_t tmp = x[i];
                x[i] = x[j];
                x[j] = tmp;
                sign = -sign;
            }
        }
    }
    int64_t a = x[0], b = x[1];
    if (rank == 2) {
        *index = a * (2*n-a-1)/2 + b-a-1;
    } else {
        *index = (n*(n-1)*(n-2) - (n-a)*(n-a-1)*(n-a-2))/6;
        *index += (b-a-1)*(2*n-a-b-2)/2 + x[2]-b-1;
    }
    return sign;
}

/* Full amplitudes, ERIs and intermediates are affine strided views.  Hoist
 * spin permutations and strides out of the element loop; only compact T3
 * needs the nonlinear pair/triple index reconstruction below. */
static void pack_affine(const Context *ctx, const Tensor *t,
                        const int *order, int nlabels, size_t count, double *out)
{
    const double *source[2];
    int64_t strides[2][NLAB] = {{0}};
    double sign[2] = {1., 1.};
    int nsources = 0;
    if (!t->leaf) {
        int64_t step = 1;
        for (int i = t->nlabels-1; i >= 0; i--) {
            int label = t->labels[i];
            strides[0][label] = step;
            step *= ctx->dim[label];
        }
        source[0] = t->data;
        nsources = 1;
    } else {
        const Leaf *leaf = t->leaf;
        for (int exchange = 0; exchange < (leaf->kind == 0 ? 2 : 1); exchange++) {
            if (leaf->kind == 0 && !(exchange ? leaf->exchange : leaf->direct)) {
                continue;
            }
            int block, native_order[6] = {0};
            if (leaf->kind == 0) {
                native_order[0] = 0;
                native_order[1] = 1;
                native_order[2] = exchange ? 3 : 2;
                native_order[3] = exchange ? 2 : 3;
                block = leaf->spins[0] == leaf->spins[1] ? 2*leaf->spins[0] : 1;
                if (block == 1 && leaf->spins[0] == 1) {
                    native_order[0] = 1;
                    native_order[1] = 0;
                    int tmp = native_order[2];
                    native_order[2] = native_order[3];
                    native_order[3] = tmp;
                }
                sign[nsources] = exchange ? -1. : 1.;
            } else {
                block = leaf->block;
                memcpy(native_order, leaf->order, sizeof(int)*2*leaf->rank);
                sign[nsources] = leaf->sign;
            }
            int64_t origin = 0;
            for (int i = 0; i < 2*leaf->rank; i++) {
                int axis = native_order[i], label = leaf->labels[axis];
                int64_t base = ctx->base[label];
                if (leaf->kind == 0 && ((label >= 4 && label <= 7) || label >= 10)) {
                    base += ctx->sizes[leaf->spins[axis]];
                }
                int64_t stride = ctx->strides[6*block+i];
                origin += base * stride;
                strides[nsources][label] = stride;
            }
            source[nsources++] = ctx->data[block]+origin;
        }
    }
    int last = nlabels ? order[nlabels-1] : 0;
    int64_t width = nlabels ? ctx->dim[last] : 1;
    for (size_t row = 0; row < count/(size_t)width; row++) {
        size_t rem = row;
        int64_t offset[2] = {0, 0};
        for (int j = nlabels-2; j >= 0; j--) {
            int label = order[j];
            int64_t local = rem % ctx->dim[label];
            rem /= ctx->dim[label];
            offset[0] += local * strides[0][label];
            offset[1] += local * strides[1][label];
        }
        const double *p = source[0]+offset[0];
        double *dest = out+row*width;
        int64_t ps = strides[0][last];
        if (nsources == 1) {
            if (ps == 1 && sign[0] == 1.) {
                memcpy(dest, p, width*sizeof(double));
            } else {
#pragma omp simd
                for (int64_t j = 0; j < width; j++) {
                    dest[j] = sign[0] * p[j*ps];
                }
            }
        } else {
            const double *q = source[1]+offset[1];
            int64_t qs = strides[1][last];
#pragma omp simd
            for (int64_t j = 0; j < width; j++) {
                dest[j] = sign[0]*p[j*ps] + sign[1]*q[j*qs];
            }
        }
    }
}

static void pack_tensor(const Context *ctx, const Tensor *t,
                        const int *order, int nlabels, size_t count, double *out)
{
    if (!count) {
        return;
    }
    if (!t->leaf || t->leaf->kind != 3 || !ctx->compact) {
        pack_affine(ctx, t, order, nlabels, count, out);
        return;
    }
    /* Decode fixed axes and their packed pair/triple once per row.  Only
     * the final packed coordinate can change in the innermost loop. */
    const Leaf *leaf = t->leaf;
    int same = leaf->block == 6 || leaf->block == 9;
    int rank = same ? 3 : 2;
    int majority = leaf->spins[leaf->order[0]];
    int last = order[nlabels-1], varied = 0;
    int64_t width = ctx->dim[last];
    for (int i = 0; i < 6; i++) {
        if (leaf->labels[leaf->order[i]] == last) {
            varied = i;
        }
    }
    int64_t local[NLAB] = {0};
    for (size_t row = 0; row < count/(size_t)width; row++) {
        size_t rem = row;
        for (int j = nlabels-2; j >= 0; j--) {
            int label = order[j];
            local[label] = rem % ctx->dim[label];
            rem /= ctx->dim[label];
        }
        int64_t native[6], packed[4];
        for (int j = 0; j < 6; j++) {
            int label = leaf->labels[leaf->order[j]];
            native[j] = ctx->base[label] + local[label];
        }
        int osign = packed_index(ctx->sizes[majority], native, rank, packed);
        int vsign = packed_index(ctx->sizes[2+majority], native+rank, rank, packed+1);
        if (!same) {
            packed[2] = native[4];
            packed[3] = native[5];
        }
        int64_t initial = native[varied];
        for (int64_t j = 0; j < width; j++) {
            native[varied] = initial+j;
            if (varied < rank) {
                osign = packed_index(ctx->sizes[majority], native, rank, packed);
            } else if (varied < 2*rank) {
                vsign = packed_index(ctx->sizes[2+majority], native+rank, rank, packed+1);
            } else {
                packed[varied-2] = native[varied];
            }
            int sign = leaf->sign * osign * vsign;
            out[row*width+j] = sign ? sign*strided_read(
                ctx, leaf->block, packed, same ? 2 : 4) : 0.;
        }
    }
}

/* Row-major C = A B.  Keep four output columns in registers until their
 * complete dot products are ready, instead of repeatedly loading/storing
 * C for every k.  Short occupied contractions use contiguous SIMD loops.
 * Each dot product retains the original increasing-k accumulation order. */
static void matrix_product(const double *a, const double *b, double *c,
                           size_t m, size_t n, size_t k)
{
    if (k == 1) {
        for (size_t i = 0; i < m; i++) {
            double v = a[i];
#pragma omp simd
            for (size_t j = 0; j < n; j++) {
                c[i*n+j] = v*b[j];
            }
        }
        return;
    }
    if (k == 2) {
        for (size_t i = 0; i < m; i++) {
            double v0 = a[i*2], v1 = a[i*2+1];
#pragma omp simd
            for (size_t j = 0; j < n; j++) {
                c[i*n+j] = v0*b[j] + v1*b[n+j];
            }
        }
        return;
    }
    if (k == 3) {
        for (size_t i = 0; i < m; i++) {
            double v0 = a[i*3], v1 = a[i*3+1], v2 = a[i*3+2];
#pragma omp simd
            for (size_t j = 0; j < n; j++) {
                c[i*n+j] = (v0*b[j] + v1*b[n+j]) + v2*b[2*n+j];
            }
        }
        return;
    }
    if (k == 4) {
        for (size_t i = 0; i < m; i++) {
            double v0 = a[i*4], v1 = a[i*4+1], v2 = a[i*4+2], v3 = a[i*4+3];
#pragma omp simd
            for (size_t j = 0; j < n; j++) {
                c[i*n+j] = ((v0*b[j] + v1*b[n+j]) + v2*b[2*n+j]) + v3*b[3*n+j];
            }
        }
        return;
    }
    for (size_t i = 0; i < m; i++) {
        size_t j = 0;
        for (; j+4 <= n; j += 4) {
            double c0 = 0., c1 = 0., c2 = 0., c3 = 0.;
            for (size_t x = 0; x < k; x++) {
                double v = a[i*k+x];
                c0 += v*b[x*n+j];
                c1 += v*b[x*n+j+1];
                c2 += v*b[x*n+j+2];
                c3 += v*b[x*n+j+3];
            }
            c[i*n+j] = c0;
            c[i*n+j+1] = c1;
            c[i*n+j+2] = c2;
            c[i*n+j+3] = c3;
        }
        for (; j < n; j++) {
            double value = 0.;
            for (size_t x = 0; x < k; x++) {
                value += a[i*k+x]*b[x*n+j];
            }
            c[i*n+j] = value;
        }
    }
}

static void contract_pair(const Context *ctx, const Tensor *left,
                          const Tensor *right, const Product *p,
                          double *a, double *b, double *out)
{
    pack_tensor(ctx, left, p->left, p->nleft, p->m*p->k, a);
    pack_tensor(ctx, right, p->right, p->nright, p->k*p->n, b);
    matrix_product(a, b, out, p->m, p->n, p->k);
}

static int share_dummy(const Tensor *a, const Tensor *b)
{
    for (int i = 0; i < a->nlabels; i++) {
        if (a->labels[i] >= 8 && has_label(b, a->labels[i])) {
            return 1;
        }
    }
    return 0;
}

static int update_cap(size_t *cap, size_t a, size_t b)
{
    size_t n;
    if (mul_size(a, b, &n)) {
        return -2;
    }
    if (*cap < n) {
        *cap = n;
    }
    return 0;
}

/* All chain edges enter the bound, even though execution chooses only one.
 * This covers changes of optimal contraction order in smaller edge tiles. */
static int term_workspace(const Context *ctx, const Tensor *t, int nf,
                           size_t *cap)
{
    Product first, second;
    for (int a = 0; a < nf-1; a++) {
        for (int b = a+1; b < nf; b++) {
            if (nf == 3 && !share_dummy(t+a, t+b)) {
                continue;
            }
            if (product_plan(t+a, t+b, ctx, &first) ||
                update_cap(cap+3, first.m, first.k) ||
                update_cap(cap+4, first.k, first.n) ||
                update_cap(cap+5, first.m, first.n)) {
                return -2;
            }
            if (nf == 3) {
                Tensor middle = product_tensor(&first, NULL);
                if (product_plan(&middle, t+3-a-b, ctx, &second) ||
                    update_cap(cap+3, second.m, second.k) ||
                    update_cap(cap+4, second.k, second.n) ||
                    update_cap(cap+6, second.m, second.n)) {
                    return -2;
                }
            }
        }
    }
    return 0;
}

static int workspace_caps(const int64_t *sizes, const int32_t *terms,
                          const int64_t *offsets, int block_size,
                          size_t *cap, size_t *bytes)
{
    if (block_size <= 0 || offsets[0] != 0) {
        return -2;
    }
    for (int i = 0; i < 4; i++) {
        if (sizes[i] < 0 || sizes[i] > INT_MAX) {
            return -2;
        }
    }
    memset(cap, 0, sizeof(size_t)*NBUF);
    Context ctx = {0};
    ctx.sizes = sizes;
    for (int na = 0; na <= 4; na++) {
        if (offsets[na] < 0 || offsets[na+1] < offsets[na] ||
            (uint64_t)offsets[na+1] > SIZE_MAX / TERM_SIZE / sizeof(int32_t)) {
            return -2;
        }
        if (na > sizes[0] || na > sizes[2] ||
            4-na > sizes[1] || 4-na > sizes[3]) {
            continue;
        }
        for (int64_t it = offsets[na]; it < offsets[na+1]; it++) {
            const int32_t *term = terms+it*TERM_SIZE;
            for (int label = 0; label < NLAB; label++) {
                int spin = term[2+label];
                ctx.dim[label] = 1;
                if (spin < 0 || label < 4) {
                    continue;
                }
                if (spin > 1) {
                    return -2;
                }
                int64_t dim = sizes[(label == 8 || label == 9 ? 0 : 2)+spin];
                ctx.dim[label] = label < 8 && dim > block_size ? block_size : dim;
            }
            size_t moment = 1;
            for (int i = 4; i < 8; i++) {
                if (mul_size(moment, ctx.dim[i], &moment)) {
                    return -2;
                }
            }
            for (int i = 0; i < 3; i++) {
                if (cap[i] < moment) {
                    cap[i] = moment;
                }
            }
            Leaf leaves[3];
            Tensor tensors[3];
            if (prepare_term(term, leaves, tensors) ||
                term_workspace(&ctx, tensors, term[1], cap)) {
                return -2;
            }
        }
    }
    size_t sum = 0;
    for (int i = 0; i < NBUF; i++) {
        if (cap[i] > SIZE_MAX - sum) {
            return -2;
        }
        sum += cap[i];
    }
    if (mul_size(sum, sizeof(double), bytes)) {
        return -2;
    }
    return 0;
}

/* Per-worker upper bound for the seven reused buffers, including both
 * admissible chain orders and all sectors in offsets.  Zero means invalid
 * dimensions/overflow (or no terms). */
uint64_t UCCSDTQ_workspace(const int64_t *sizes, const int32_t *terms,
                           const int64_t *offsets, int block_size)
{
    size_t cap[NBUF], bytes;
    if (workspace_caps(sizes, terms, offsets, block_size, cap, &bytes)) {
        return 0;
    }
    return (uint64_t)bytes;
}

static int evaluate_term(const Context *ctx, const int32_t *term,
                         double coefficient, double **buf)
{
    Leaf leaves[3];
    Tensor t[3];
    if (prepare_term(term, leaves, t)) {
        return -2;
    }
    Product first, second;
    Tensor result;
    int nf = term[1];
    if (nf == 2) {
        if (product_plan(t, t+1, ctx, &first)) {
            return -2;
        }
        contract_pair(ctx, t, t+1, &first, buf[3], buf[4], buf[5]);
        result = product_tensor(&first, buf[5]);
    } else {
        int besta = -1, bestb = -1;
        double bestcost = HUGE_VAL;
        for (int a = 0; a < 2; a++) {
            for (int b = a+1; b < 3; b++) {
                if (!share_dummy(t+a, t+b)) {
                    continue;
                }
                Product p, q;
                if (product_plan(t+a, t+b, ctx, &p)) {
                    return -2;
                }
                Tensor middle = product_tensor(&p, NULL);
                if (product_plan(&middle, t+3-a-b, ctx, &q)) {
                    return -2;
                }
                double cost = (double)p.m*p.n*p.k + (double)q.m*q.n*q.k;
                if (cost < bestcost) {
                    bestcost = cost;
                    besta = a;
                    bestb = b;
                    first = p;
                    second = q;
                }
            }
        }
        if (besta < 0) {
            return -2;
        }
        contract_pair(ctx, t+besta, t+bestb, &first, buf[3], buf[4], buf[5]);
        Tensor middle = product_tensor(&first, buf[5]);
        contract_pair(ctx, &middle, t+3-besta-bestb, &second,
                      buf[3], buf[4], buf[6]);
        result = product_tensor(&second, buf[6]);
    }
    /* The result has exactly the four external virtual axes.  Resolve its
     * transpose once; integer division in a per-element transpose dominates
     * small matrix contractions and would defeat native acceleration. */
    size_t stride[4] = {0}, step = 1, output = 0;
    for (int j = result.nlabels-1; j >= 0; j--) {
        int label = result.labels[j];
        stride[label-4] = step;
        step *= ctx->dim[label];
    }
    double *restrict target = buf[term[0] == 5 ? 2 : 0];
    double *restrict l3 = buf[1];
    for (int64_t a = 0; a < ctx->dim[4]; a++) {
        for (int64_t b = 0; b < ctx->dim[5]; b++) {
            for (int64_t c = 0; c < ctx->dim[6]; c++) {
                const double *src = result.data + a*stride[0]+b*stride[1]+c*stride[2];
                if (term[0] == 3 || term[0] == 4) {
#pragma omp simd
                    for (int64_t d = 0; d < ctx->dim[7]; d++) {
                        double value = coefficient * src[d*stride[3]];
                        l3[output+d] += value;
                    }
                } else {
#pragma omp simd
                    for (int64_t d = 0; d < ctx->dim[7]; d++) {
                        target[output+d] += coefficient * src[d*stride[3]];
                    }
                }
                output += ctx->dim[7];
            }
        }
    }
    return 0;
}

static int run_task(Context *ctx, const double *ea, const double *eb,
                    const int32_t *terms, const double *coeff,
                    const int64_t *offsets, const int64_t *task, double **buf,
                    double *energy, int64_t *singular, double *moments)
{
    int na = task[0];
    size_t count = 1;
    double occupied_energy = 0.;
    const double *orbital[2] = {ea, eb};
    for (int i = 0; i < 4; i++) {
        int spin = i >= na;
        ctx->base[i] = task[1+i];
        ctx->base[4+i] = task[5+i];
        ctx->dim[i] = 1;
        ctx->dim[4+i] = task[9+i];
        if (mul_size(count, task[9+i], &count)) {
            return -2;
        }
        occupied_energy += orbital[spin][task[1+i]];
    }
    for (int i = 0; i < 3; i++) {
        memset(buf[i], 0, count*sizeof(double));
    }
    for (int64_t it = offsets[na]; it < offsets[na+1]; it++) {
        const int32_t *term = terms+it*TERM_SIZE;
        int skip = 0;
        for (int i = 8; i < NLAB; i++) {
            int spin = term[2+i];
            ctx->base[i] = 0;
            ctx->dim[i] = spin < 0 ? 1 : ctx->sizes[(i < 10 ? 0 : 2)+spin];
            skip |= ctx->dim[i] == 0;
        }
        if (!skip) {
            int status = evaluate_term(ctx, term, coeff[it], buf);
            if (status) {
                return status;
            }
        }
    }
    /* vvov and vooo contribute to both W4 and L3.  Accumulate them once in
     * L3 and merge that vector into W4 after all seed terms have finished;
     * this removes one read-modify-write pass for every such term. */
#pragma omp simd
    for (size_t i = 0; i < count; i++) {
        buf[0][i] += buf[1][i];
    }
    if (moments) {
        for (int i = 0; i < 3; i++) {
            memcpy(moments+i*count, buf[i], count*sizeof(double));
        }
    }
    double bracket = 0., extra = 0.;
    int64_t zeros = 0;
    const double *virtual_energy[4];
    for (int axis = 0; axis < 4; axis++) {
        int spin = axis >= na;
        virtual_energy[axis] = orbital[spin] + ctx->sizes[spin] + ctx->base[4+axis];
    }
    /* Keep the original subtraction and accumulation order while reusing
     * partial denominators; no element-wise integer division is needed. */
    size_t output = 0;
    for (int64_t a = 0; a < ctx->dim[4]; a++) {
        double da = occupied_energy - virtual_energy[0][a];
        for (int64_t b = 0; b < ctx->dim[5]; b++) {
            double dab = da - virtual_energy[1][b];
            for (int64_t c = 0; c < ctx->dim[6]; c++) {
                double dabc = dab - virtual_energy[2][c];
                for (int64_t d = 0; d < ctx->dim[7]; d++, output++) {
                    double denominator = dabc - virtual_energy[3][d];
                    if (fabs(denominator) <= 1e-15) {
                        zeros++;
                    } else {
                        double q = buf[0][output] / denominator;
                        bracket += buf[2][output] * q;
                        extra += buf[1][output] * q;
                    }
                }
            }
        }
    }
    static const double weights[5] = {1./24., 1./6., 1./4., 1./6., 1./24.};
    energy[0] = weights[na] * bracket;
    energy[1] = weights[na] * extra;
    *singular = zeros;
    return 0;
}

/* Errors: -1 allocation, -2 invalid dimensions/overflow, -3 workspace limit.
 * moments is normally NULL; with exactly one task it returns raw W4,L3,L2.
 * energies and singular have one row per task, enabling deterministic
 * ordered accumulation in Python after the parallel region completes. */
int UCCSDTQ_kernel(const double *const *data, const int64_t *strides,
                   const int64_t *sizes, int compact,
                   const double *ea, const double *eb,
                   const int32_t *terms, const double *coeff,
                   const int64_t *offsets, const int64_t *tasks,
                   int64_t ntasks, int nthreads, uint64_t workspace_limit,
                   double *energies, int64_t *singular, int *actual_threads,
                   double *moments)
{
    if (ntasks < 0 || nthreads <= 0 || (moments && ntasks != 1)) {
        return -2;
    }
    *actual_threads = 0;
    if (!ntasks) {
        return 0;
    }
    if ((uint64_t)ntasks > SIZE_MAX / TASK_SIZE / sizeof(int64_t)) {
        return -2;
    }
    int block_size = 1;
    for (int64_t it = 0; it < ntasks; it++) {
        const int64_t *task = tasks+it*TASK_SIZE;
        int na = task[0];
        if (na < 0 || na > 4) {
            return -2;
        }
        for (int i = 0; i < 4; i++) {
            int spin = i >= na;
            if (task[1+i] < 0 || task[1+i] >= sizes[spin] ||
                task[5+i] < 0 || task[9+i] <= 0 ||
                task[9+i] > INT_MAX || task[5+i] > sizes[2+spin] - task[9+i]) {
                return -2;
            }
            if (block_size < task[9+i]) {
                block_size = task[9+i];
            }
        }
    }
    size_t cap[NBUF], bytes, total;
    if (workspace_caps(sizes, terms, offsets, block_size, cap, &bytes) || !bytes) {
        return -2;
    }
    if ((uint64_t)bytes > workspace_limit) {
        return -3;
    }
#ifndef _OPENMP
    nthreads = 1;
#endif
    if (ntasks < nthreads) {
        nthreads = (int)ntasks;
    }
    if (mul_size(bytes, nthreads, &total)) {
        return -2;
    }
    double *workspace = NULL;
    int status = 0;
#pragma omp parallel num_threads(nthreads)
    {
        int tid = 0, team = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
#pragma omp master
        {
            *actual_threads = team;
            /* OMP_THREAD_LIMIT/dynamic teams may supply fewer workers than
             * requested.  The checked total above bounds this allocation.
             * Allocate on the caller thread, which also frees the arena,
             * to avoid retaining arenas in different worker allocators. */
            workspace = malloc(bytes * (size_t)team);
            if (!workspace) {
                status = -1;
            }
        }
#pragma omp barrier
        double *buf[NBUF];
        if (workspace) {
            buf[0] = workspace + (bytes / sizeof(double))*tid;
            for (int i = 1; i < NBUF; i++) {
                buf[i] = buf[i-1] + cap[i-1];
            }
        }
        Context ctx = {0};
        ctx.data = data;
        ctx.strides = strides;
        ctx.sizes = sizes;
        ctx.compact = compact;
        int local_status = workspace ? 0 : -1;
#pragma omp for schedule(dynamic)
        for (int64_t it = 0; it < ntasks; it++) {
            if (!local_status) {
                local_status = run_task(&ctx, ea, eb, terms, coeff, offsets,
                                        tasks+it*TASK_SIZE, buf, energies+2*it,
                                        singular+it, moments);
            }
        }
        if (local_status) {
#pragma omp critical
            status = local_status;
        }
    }
    free(workspace);
    return status;
}
