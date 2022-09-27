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

#include <complex.h>

#define BLOCK_DIM    104

#define HERMITIAN    1
#define ANTIHERMI    2
#define SYMMETRIC    3

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

#define TRIU_LOOP(I, J) \
        for (j0 = 0; j0 < n; j0+=BLOCK_DIM) \
                for (I = 0, j1 = MIN(j0+BLOCK_DIM, n); I < j1; I++) \
                        for (J = MAX(I,j0); J < j1; J++)

void NPdsymm_triu(int n, double *mat, int hermi);
void NPzhermi_triu(int n, double complex *mat, int hermi);
void NPdunpack_tril(int n, double *tril, double *mat, int hermi);
void NPdunpack_row(int ndim, int row_id, double *tril, double *row);
void NPzunpack_tril(int n, double complex *tril, double complex *mat,
                    int hermi);
void NPdpack_tril(int n, double *tril, double *mat);
void NPzpack_tril(int n, double complex *tril, double complex *mat);

void NPdtranspose(int n, int m, double *a, double *at);
void NPztranspose(int n, int m, double complex *a, double complex *at);
void NPdtranspose_021(int *shape, double *a, double *at);
void NPztranspose_021(int *shape, double complex *a, double complex *at);

void NPdunpack_tril_2d(int count, int n, double *tril, double *mat, int hermi);
void NPzunpack_tril_2d(int count, int n,
                       double complex *tril, double complex *mat, int hermi);
void NPdpack_tril_2d(int count, int n, double *tril, double *mat);

void NPomp_split(size_t *start, size_t *end, size_t n);
void NPomp_dsum_reduce_inplace(double **vec, size_t count);
void NPomp_dprod_reduce_inplace(double **vec, size_t count);
void NPomp_zsum_reduce_inplace(double complex **vec, size_t count);
void NPomp_zprod_reduce_inplace(double complex **vec, size_t count);

void NPdset0(double *p, const size_t n);
void NPzset0(double complex *p, const size_t n);
void NPdcopy(double *out, const double *in, const size_t n);
void NPzcopy(double complex *out, const double complex *in, const size_t n);
