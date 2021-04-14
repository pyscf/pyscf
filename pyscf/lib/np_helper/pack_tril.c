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

#include "stdlib.h"
#include <complex.h>
#include "config.h"
#include "np_helper.h"

void NPdsymm_triu(int n, double *mat, int hermi)
{
        size_t i, j, j0, j1;

        if (hermi == HERMITIAN || hermi == SYMMETRIC) {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = mat[j*n+i];
                }
        } else {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = -mat[j*n+i];
                }
        }
}

void NPzhermi_triu(int n, double complex *mat, int hermi)
{
        size_t i, j, j0, j1;

        if (hermi == HERMITIAN) {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = conj(mat[j*n+i]);
                }
        } else if (hermi == SYMMETRIC) {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = mat[j*n+i];
                }
        } else {
                TRIU_LOOP(i, j) {
                        mat[i*n+j] = -conj(mat[j*n+i]);
                }
        }
}


void NPdunpack_tril(int n, double *tril, double *mat, int hermi)
{
        size_t i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        mat[i*n+j] = tril[ij];
                }
        }
        if (hermi) {
                NPdsymm_triu(n, mat, hermi);
        }
}

// unpack one row from the compact matrix-tril coefficients
void NPdunpack_row(int ndim, int row_id, double *tril, double *row)
{
        int i;
        size_t idx = ((size_t)row_id) * (row_id + 1) / 2;
        NPdcopy(row, tril+idx, row_id);
        for (i = row_id; i < ndim; i++) {
                idx += i;
                row[i] = tril[idx];
        }
}

void NPzunpack_tril(int n, double complex *tril, double complex *mat,
                    int hermi)
{
        size_t i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        mat[i*n+j] = tril[ij];
                }
        }
        if (hermi) {
                NPzhermi_triu(n, mat, hermi);
        }
}

void NPdpack_tril(int n, double *tril, double *mat)
{
        size_t i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        tril[ij] = mat[i*n+j];
                }
        }
}

void NPzpack_tril(int n, double complex *tril, double complex *mat)
{
        size_t i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        tril[ij] = mat[i*n+j];
                }
        }
}

/* out += in[idx[:,None],idy] */
void NPdtake_2d(double *out, double *in, int *idx, int *idy,
                int odim, int idim, int nx, int ny)
{
#pragma omp parallel default(none) \
        shared(out, in, idx,idy, odim, idim, nx, ny)
{
        size_t i, j;
        double *pin;
#pragma omp for schedule (static)
        for (i = 0; i < nx; i++) {
                pin = in + (size_t)idim * idx[i];
                for (j = 0; j < ny; j++) {
                        out[i*odim+j] = pin[idy[j]];
                }
        }
}
}

void NPztake_2d(double complex *out, double complex *in, int *idx, int *idy,
                int odim, int idim, int nx, int ny)
{
#pragma omp parallel default(none) \
        shared(out, in, idx,idy, odim, idim, nx, ny)
{
        size_t i, j;
        double complex *pin;
#pragma omp for schedule (static)
        for (i = 0; i < nx; i++) {
                pin = in + (size_t)idim * idx[i];
                for (j = 0; j < ny; j++) {
                        out[i*odim+j] = pin[idy[j]];
                }
        }
}
}

/* out[idx[:,None],idy] += in */
void NPdtakebak_2d(double *out, double *in, int *idx, int *idy,
                   int odim, int idim, int nx, int ny, int thread_safe)
{
        if (thread_safe) {
#pragma omp parallel default(none) \
        shared(out, in, idx,idy, odim, idim, nx, ny)
{
        size_t i, j;
        double *pout;
#pragma omp for schedule (static)
        for (i = 0; i < nx; i++) {
                pout = out + (size_t)odim * idx[i];
                for (j = 0; j < ny; j++) {
                        pout[idy[j]] += in[i*idim+j];
                }
        }
}
        } else {
                size_t i, j;
                double *pout;
                for (i = 0; i < nx; i++) {
                        pout = out + (size_t)odim * idx[i];
                        for (j = 0; j < ny; j++) {
                                pout[idy[j]] += in[i*idim+j];
                        }
                }
        }
}

void NPztakebak_2d(double complex *out, double complex *in, int *idx, int *idy,
                   int odim, int idim, int nx, int ny, int thread_safe)
{
        if (thread_safe) {
#pragma omp parallel default(none) \
        shared(out, in, idx,idy, odim, idim, nx, ny)
{
        size_t i, j;
        double complex *pout;
#pragma omp for schedule (static)
        for (i = 0; i < nx; i++) {
                pout = out + (size_t)odim * idx[i];
                for (j = 0; j < ny; j++) {
                        pout[idy[j]] += in[i*idim+j];
                }
        }
}
        } else {
                size_t i, j;
                double complex *pout;
                for (i = 0; i < nx; i++) {
                        pout = out + (size_t)odim * idx[i];
                        for (j = 0; j < ny; j++) {
                                pout[idy[j]] += in[i*idim+j];
                        }
                }
        }
}

void NPdunpack_tril_2d(int count, int n, double *tril, double *mat, int hermi)
{
#pragma omp parallel default(none) \
        shared(count, n, tril, mat, hermi)
{
        int ic;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                NPdunpack_tril(n, tril+n2*ic, mat+nn*ic, hermi);
        }
}
}

void NPzunpack_tril_2d(int count, int n,
                       double complex *tril, double complex *mat, int hermi)
{
#pragma omp parallel default(none) \
        shared(count, n, tril, mat, hermi)
{
        int ic;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                NPzunpack_tril(n, tril+n2*ic, mat+nn*ic, hermi);
        }
}
}

void NPdpack_tril_2d(int count, int n, double *tril, double *mat)
{
#pragma omp parallel default(none) \
        shared(count, n, tril, mat)
{
        int ic;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                NPdpack_tril(n, tril+n2*ic, mat+nn*ic);
        }
}
}

void NPzpack_tril_2d(int count, int n, double complex *tril, double complex *mat)
{
#pragma omp parallel default(none) \
        shared(count, n, tril, mat)
{
        int ic;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                NPzpack_tril(n, tril+n2*ic, mat+nn*ic);
        }
}
}

