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
#include <string.h>
#include <complex.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"

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
        const size_t dimc = ldc;
        int i, j;
        if (m == 0 || n == 0) {
                return;
        } else if (k == 0) {
                for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                        c[i*dimc+j] = 0;
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
                                        c[i*dimc+j] = 0;
                                }
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*dimc+j] *= beta;
                                }
                        }
                }

#pragma omp parallel private(i, j)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((k+nthread-1) / nthread, 1);
                double D0 = 0;
                double *cpriv = malloc(sizeof(double) * (m*n+2));
                int di;
                size_t ij;
                size_t astride = nblk;
                size_t bstride = nblk;
                if (trans_a == 'N') {
                        astride *= lda;
                }
                if (trans_b != 'N') {
                        bstride *= ldb;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, k-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &m, &n, &di,
                                       &alpha, a+astride*i, &lda,
                                       b+bstride*i, &ldb,
                                       &D0, cpriv, &m);
                        }
                }
#pragma omp critical
                if (di > 0) {
                        for (ij = 0, i = 0; i < n; i++) {
                                for (j = 0; j < m; j++, ij++) {
                                        c[i*dimc+j] += cpriv[ij];
                                }
                        }
                }
                free(cpriv);
}

        } else if (m > n*2) { // parallelize m

#pragma omp parallel
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((m+nthread-1) / nthread, 1);
                nthread = (m+nblk-1) / nblk;
                int di;
                size_t bstride = nblk;
                if (trans_a != 'N') {
                        bstride *= lda;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, m-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &di, &n, &k,
                                       &alpha, a+bstride*i, &lda, b, &ldb,
                                       &beta, c+i*nblk, &ldc);
                        }
                }
}

        } else { // parallelize n

#pragma omp parallel
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((n+nthread-1) / nthread, 1);
                nthread = (n+nblk-1) / nblk;
                int di;
                size_t bstride = nblk;
                size_t cstride = dimc * nblk;
                if (trans_b == 'N') {
                        bstride *= ldb;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, n-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &m, &di, &k,
                                       &alpha, a, &lda, b+bstride*i, &ldb,
                                       &beta, c+cstride*i, &ldc);
                        }
                }
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
        const size_t dimc = ldc;
        int i, j;
        if (m == 0 || n == 0) {
                return;
        } else if (k == 0) {
                for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                        c[i*dimc+j] = 0;
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
                                        c[i*dimc+j] = 0;
                                }
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*dimc+j] *= beta[0];
                                }
                        }
                }

#pragma omp parallel private(i, j)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((k+nthread-1) / nthread, 1);
                double complex Z0 = 0;
                double complex *cpriv = malloc(sizeof(double complex) * (m*n+2));
                int di;
                size_t ij;
                size_t astride = nblk;
                size_t bstride = nblk;
                if (trans_a == 'N') {
                        astride *= lda;
                }
                if (trans_b != 'N') {
                        bstride *= ldb;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, k-i*nblk);
                        if (di > 0) {
                                zgemm_(&trans_a, &trans_b, &m, &n, &di,
                                       alpha, a+astride*i, &lda,
                                       b+bstride*i, &ldb,
                                       &Z0, cpriv, &m);
                        }
                }
#pragma omp critical
                if (di > 0) {
                        for (ij = 0, i = 0; i < n; i++) {
                                for (j = 0; j < m; j++, ij++) {
                                        c[i*dimc+j] += cpriv[ij];
                                }
                        }
                }
                free(cpriv);
}

        } else if (m > n*2) { // parallelize m

#pragma omp parallel
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((m+nthread-1) / nthread, 1);
                nthread = (m+nblk-1) / nblk;
                int di;
                size_t bstride = nblk;
                if (trans_a != 'N') {
                        bstride *= lda;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, m-i*nblk);
                        if (di > 0) {
                                zgemm_(&trans_a, &trans_b, &di, &n, &k,
                                       alpha, a+bstride*i, &lda, b, &ldb,
                                       beta, c+i*nblk, &ldc);
                        }
                }
}

        } else { // parallelize n

#pragma omp parallel
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((n+nthread-1) / nthread, 1);
                nthread = (n+nblk-1) / nblk;
                int di;
                size_t bstride = nblk;
                size_t cstride = dimc * nblk;
                if (trans_b == 'N') {
                        bstride *= ldb;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, n-i*nblk);
                        if (di > 0) {
                                zgemm_(&trans_a, &trans_b, &m, &di, &k,
                                       alpha, a, &lda, b+bstride*i, &ldb,
                                       beta, c+cstride*i, &ldc);
                        }
                }
}
        }
}
