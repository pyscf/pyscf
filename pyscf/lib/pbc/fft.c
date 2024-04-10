/* Copyright 2021- The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdio.h>
#include <complex.h>
#include <fft.h>
#include "config.h"

#define BLKSIZE 128
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

fftw_plan fft_create_r2c_plan(double* in, complex double* out, int rank, int* mesh)
{
    fftw_plan p;
    p = fftw_plan_dft_r2c(rank, mesh, in, out, FFTW_ESTIMATE);
    return p;
}

fftw_plan fft_create_c2r_plan(complex double* in, double* out, int rank, int* mesh)
{
    fftw_plan p;
    p = fftw_plan_dft_c2r(rank, mesh, in, out, FFTW_ESTIMATE);
    return p;
}

void fft_execute(fftw_plan p)
{
    fftw_execute(p);
}

void fft_destroy_plan(fftw_plan p)
{
    fftw_destroy_plan(p);
}

void _complex_fft(complex double* in, complex double* out, int* mesh, int rank, int sign)
{
    int i;
    int nx = mesh[0];
    int nyz = 1;
    for (i = 1; i < rank; i++) {
        nyz *= mesh[i];
    }
    int nmax = nyz / BLKSIZE * BLKSIZE;
    fftw_plan p_2d = fftw_plan_dft(rank-1, mesh+1, in, out, sign, FFTW_ESTIMATE);
    int nn[BLKSIZE] = {nx};
    fftw_plan p_3d_x = fftw_plan_many_dft(1, nn, BLKSIZE,
                                          out, NULL, nyz, 1,
                                          out, NULL, nyz, 1,
                                          sign, FFTW_ESTIMATE);

    #pragma omp parallel private(i)
    {
        int off;
        #pragma omp for schedule(dynamic)
        for (i = 0; i < nx; i++) {
            off = i * nyz;
            fftw_execute_dft(p_2d, in+off, out+off);
        }

        #pragma omp for schedule(dynamic)
        for (i = 0; i < nmax; i+=BLKSIZE) {
            fftw_execute_dft(p_3d_x, out+i, out+i);
        }
    }
    fftw_destroy_plan(p_2d);
    fftw_destroy_plan(p_3d_x);
    
    int nres = nyz - nmax;
    if (nres > 0) {
        fftw_plan p_3d_x = fftw_plan_many_dft(1, nn, nres,
                                          out+nmax, NULL, nyz, 1,
                                          out+nmax, NULL, nyz, 1,
                                          sign, FFTW_ESTIMATE);
        fftw_execute(p_3d_x);
        fftw_destroy_plan(p_3d_x);
    }
}

void fft(complex double* in, complex double* out, int* mesh, int rank)
{
    _complex_fft(in, out, mesh, rank, FFTW_FORWARD);
}

void ifft(complex double* in, complex double* out, int* mesh, int rank)
{
    _complex_fft(in, out, mesh, rank, FFTW_BACKWARD);
    size_t i, n = 1;
    for (i = 0; i < rank; i++) {
        n *= mesh[i];
    }
    double fac = 1. / (double)n;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] *= fac;
    }
}

void rfft(double* in, complex double* out, int* mesh, int rank)
{
    fftw_plan p = fftw_plan_dft_r2c(rank, mesh, in, out, FFTW_ESTIMATE); 
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void irfft(complex double* in, double* out, int* mesh, int rank)
{
    fftw_plan p = fftw_plan_dft_c2r(rank, mesh, in, out, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    size_t i, n = 1;
    for (i = 0; i < rank; i++) {
        n *= mesh[i];
    }
    double fac = 1. / (double)n;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] *= fac;
    }
}

void _copy_d2z(double complex *out, const double *in, const size_t n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = in[i] + 0*_Complex_I;
    }
}
}
