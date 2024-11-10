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
 *
 * blas interface and blas-like functions
 */

#if defined __cplusplus
extern "C" {
#endif
#include <complex.h>

double dasum_(const int *n, const double *dx, const int *incx);
void dscal_(const int *n, const double *da, double *dx, const int *incx);
void daxpy_(const int *n, const double *da, const double *dx,
           const int *incx, double *dy, const int *incy);
double ddot_(const int *n, const double *dx, const int *incx,
             const double *dy, const int *incy);
void dcopy_(const int *n, const double *dx, const int *incx,
            const double *dy, const int *incy);
void dgemm_(const char*, const char*,
            const int*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*,
            const double*, double*, const int*);
void dgemv_(const char*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*,
            const double*, double*, const int*);
void dger_(const int *m, const int *n,
           const double *alpha, const double *x,
           const int *incx, const double *y, const int *incy,
           double *a, const int *lda);
void dsymm_(const char*, const char*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*,
            const double*, double*, const int*);
void dtrsm_(const char *side, const char *uplo, const char *transa, 
            const char *diag, const int *m, const int *n, const double *alpha,
            double *a, const int *lda, double *b, const int *ldb);

void dsyr_(const char *uplo, const int *n, const double *alpha,
           const double *x, const int *incx, double *a, const int *lda);
void dsyr2_(const char *uplo, const int *n, const double *alpha,
            const double *x, const int *incx, const double *y, const int *incy,
            double *a, const int *lda);
void dsyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const double *alpha, const double *a, const int *lda,
             const double *b, const int *ldb,
             const double *beta, double *c, const int *ldc);
void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda,
            const double *beta, double *c, const int *ldc);

void zgerc_(const int *m, const int *n,
            const double complex *alpha, const double complex *x, const int *incx,
            const double complex *y, const int *incy,
            double complex *a, const int *lda);
void zgemv_(const char*, const int*, const int*,
            const double complex*, const double complex*, const int*,
            const double complex*, const int*,
            const double complex*, double complex*, const int*);
void zgemm_(const char*, const char*,
            const int*, const int*, const int*,
            const double complex*, const double complex*, const int*,
            const double complex*, const int*,
            const double complex*, double complex*, const int*);


void CINTdset0(const int n, double *x);
void CINTdaxpy2v(const int n, const double a,
                 const double *x, const double *y, double *v);
void CINTdmat_transpose(double *a_t, const double *a,
                        const int m, const int n);
void CINTzmat_transpose(double complex *a_t, const double complex *a,
                        const int m, const int n);
void CINTzmat_dagger(double complex *a_c, const double complex *a,
                     const int m, const int n);

#if defined __cplusplus
} // end extern "C"
#endif
