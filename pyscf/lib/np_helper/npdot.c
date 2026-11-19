/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))

/*
 * numpy.dot may call unoptimized blas
 */
void NPdgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double *a, double *b, double *c,
             const double alpha, const double beta)
{
        const size_t Ldc = ldc;
        int i, j;
        if (m == 0 || n == 0) {
                return;
        } else if (k == 0) {
                for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                        c[i*Ldc+j] = 0;
                } }
                return;
        }
        a += offseta;
        b += offsetb;
        c += offsetc;

        if ((k/m) > 3 && (k/n) > 3) { // parallelize k

                if (beta == 0) {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*Ldc+j] = 0;
                                }
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*Ldc+j] *= beta;
                                }
                        }
                }

#pragma omp parallel private(i, j)
{
                double D0 = 0;
                double *cpriv = malloc(sizeof(double) * (m*n+2));
                size_t k0, k1, ij;
                NPomp_split(&k0, &k1, k);
                int dk = k1 - k0;
                if (dk > 0) {
                        size_t astride = k0;
                        size_t bstride = k0;
                        if (trans_a == 'N') {
                                astride *= lda;
                        }
                        if (trans_b != 'N') {
                                bstride *= ldb;
                        }
                        dgemm_(&trans_a, &trans_b, &m, &n, &dk,
                               &alpha, a+astride, &lda, b+bstride, &ldb,
                               &D0, cpriv, &m);
                }
#pragma omp critical
                if (dk > 0) {
                        for (ij = 0, i = 0; i < n; i++) {
                        for (j = 0; j < m; j++, ij++) {
                                c[i*Ldc+j] += cpriv[ij];
                        } }
                }

                free(cpriv);
}

        } else if (m > n*2) { // parallelize m

#pragma omp parallel
{
                size_t m0, m1;
                NPomp_split(&m0, &m1, m);
                int dm = m1 - m0;
                if (dm > 0) {
                        size_t astride = m0;
                        if (trans_a != 'N') {
                                astride *= lda;
                        }
                        dgemm_(&trans_a, &trans_b, &dm, &n, &k,
                               &alpha, a+astride, &lda, b, &ldb,
                               &beta, c+m0, &ldc);
                }
}

        } else { // parallelize n

#pragma omp parallel
{
                size_t n0, n1;
                NPomp_split(&n0, &n1, n);
                int dn = n1 - n0;
                if (dn > 0) {
                        size_t bstride = n0;
                        if (trans_b == 'N') {
                                bstride *= ldb;
                        }
                        dgemm_(&trans_a, &trans_b, &m, &dn, &k,
                               &alpha, a, &lda, b+bstride, &ldb,
                               &beta, c+Ldc*n0, &ldc);
                }
}
        }
}


/*
 * Apply L[P,c,a] and L[P,b,d] to r[i,c,d] and sum over P.
 * transpose_right selects L[P,d,b] for the right factor.
 * r_by_c is packed as (c,i,d). Each worker has a slice of work and partials
 * of size nocc*nvir*nvir. The result overwrites the first slice of partials
 * in (a,i,b) order.
 */
void NPcontract_df_vvvv(int nocc, int nvir, int naux, int nthreads,
                        int transpose_right, const double *r_by_c,
                        const double *lvv, double *work, double *partials)
{
        const int nov = nocc * nvir;
        const size_t vv = (size_t)nvir * nvir;
        const size_t size = (size_t)nocc * vv;
        const char normal = 'N';
        const char transpose = 'T';
        const char right_op = transpose_right ? normal : transpose;
        const double one = 1;
        const double zero = 0;

#pragma omp parallel num_threads(nthreads) default(none) \
        shared(naux, nvir, nov, vv, size, normal, transpose, right_op, one, zero, r_by_c, lvv, work, partials)
{
        const int tid = omp_get_thread_num();
        const int team_size = omp_get_num_threads();
        double *half = work + tid * size;
        double *sigma = partials + tid * size;
        size_t p0, p1;
        NPdset0(sigma, size);
        NPomp_split(&p0, &p1, naux);
        for (size_t p = p0; p < p1; p++) {
                const double *factor = lvv + p * vv;
                /* H[a,i,d] = sum_c L[c,a] r[c,i,d]. */
                dgemm_(&normal, &transpose, &nov, &nvir, &nvir, &one,
                       r_by_c, &nov, factor, &nvir, &zero, half, &nov);
                /* Reinterpret H as ((a,i),d) for the second GEMM. */
                dgemm_(&right_op, &normal, &nvir, &nov, &nvir, &one,
                       factor, &nvir, half, &nvir, &one, sigma, &nvir);
        }

#pragma omp barrier
        size_t start, end;
        NPomp_split(&start, &end, size);
        for (size_t i = start; i < end; i++) {
                double sum = partials[i];
                for (int t = 1; t < team_size; t++) {
                        sum += partials[t * size + i];
                }
                partials[i] = sum;
        }
}
}

void NPzgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double complex *a, double complex *b, double complex *c,
             const double complex *alpha, const double complex *beta)
{
        const size_t Ldc = ldc;
        int i, j;
        if (m == 0 || n == 0) {
                return;
        } else if (k == 0) {
                for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                        c[i*Ldc+j] = 0;
                } }
                return;
        }
        a += offseta;
        b += offsetb;
        c += offsetc;

        if ((k/m) > 3 && (k/n) > 3) { // parallelize k

                if (creal(*beta) == 0 && cimag(*beta) == 0) {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*Ldc+j] = 0;
                                }
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*Ldc+j] *= beta[0];
                                }
                        }
                }

#pragma omp parallel private(i, j)
{
                double complex Z0 = 0;
                double complex *cpriv = malloc(sizeof(double complex) * (m*n+2));
                size_t k0, k1, ij;
                NPomp_split(&k0, &k1, k);
                int dk = k1 - k0;
                if (dk > 0) {
                        size_t astride = k0;
                        size_t bstride = k0;
                        if (trans_a == 'N') {
                                astride *= lda;
                        }
                        if (trans_b != 'N') {
                                bstride *= ldb;
                        }
                        zgemm_(&trans_a, &trans_b, &m, &n, &dk,
                               alpha, a+astride, &lda, b+bstride, &ldb,
                               &Z0, cpriv, &m);
                }
#pragma omp critical
                if (dk > 0) {
                        for (ij = 0, i = 0; i < n; i++) {
                        for (j = 0; j < m; j++, ij++) {
                                c[i*Ldc+j] += cpriv[ij];
                        } }
                }
                free(cpriv);
}

        } else if (m > n*2) { // parallelize m

#pragma omp parallel
{
                size_t m0, m1;
                NPomp_split(&m0, &m1, m);
                int dm = m1 - m0;
                if (dm > 0) {
                        size_t astride = m0;
                        if (trans_a != 'N') {
                                astride *= lda;
                        }
                        zgemm_(&trans_a, &trans_b, &dm, &n, &k,
                               alpha, a+astride, &lda, b, &ldb,
                               beta, c+m0, &ldc);
                }
}

        } else { // parallelize n

#pragma omp parallel
{
                size_t n0, n1;
                NPomp_split(&n0, &n1, n);
                int dn = n1 - n0;
                if (dn > 0) {
                        size_t bstride = n0;
                        if (trans_b == 'N') {
                                bstride *= ldb;
                        }
                        zgemm_(&trans_a, &trans_b, &m, &dn, &k,
                               alpha, a, &lda, b+bstride, &ldb,
                               beta, c+Ldc*n0, &ldc);
                }
}
        }
}
