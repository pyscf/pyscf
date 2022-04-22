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
 */

#include <stdio.h>
#include <stdlib.h>
#include "np_helper/np_helper.h"
#include "config.h"
#include <math.h>

void NPdset0(double *p, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                p[i] = 0;
        }
}

void NPzset0(double complex *p, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                p[i] = 0;
        }
}

void NPdcopy(double *out, const double *in, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                out[i] = in[i];
        }
}

void NPzcopy(double complex *out, const double complex *in, const size_t n)
{
        size_t i;
        for (i = 0; i < n; i++) {
                out[i] = in[i];
        }
}

void NPdcopy_omp(double *out, const double *in, const size_t n)
{
#pragma omp parallel
{
        size_t i;
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
                out[i] = in[i];
        }
}
}

void NPzcopy_omp(double complex *out, const double complex *in, const size_t n)
{
#pragma omp parallel
{
        size_t i;
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
                out[i] = in[i];
        }
}
}

void NPcopy_d2z(double complex *out, const double *in, const size_t n)
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

void NPcopy_z2d(double *out, const double complex *in, const size_t n)
{
#pragma omp parallel
{
        size_t i;
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
                out[i] = creal(in[i]);
        }
}
}

void NPdreciprocal(double *out, const double *in, const size_t n)
{
#pragma omp parallel
{
        size_t i;
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
                out[i] = 1. / in[i];
        }
}
}

void NPzreciprocal(double complex *out, const double complex *in, const size_t n)
{
#pragma omp parallel
{
        size_t i;
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
                out[i] = 1. / in[i];
        }
}
}

void NPreciprocal_d2z(double complex *out, const double *in, const size_t n)
{
#pragma omp parallel
{
        size_t i;
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
                out[i] = 1. / in[i] + 0*_Complex_I;
        }
}
}

void NPdmultiplysum(double* out, double* a, double* b, int nrow, int ncol, int axis)
{
#pragma omp parallel
{
    size_t i, j;
    if (axis == 0){
        #pragma omp for schedule(static)
        for (j = 0; j < ncol; j++) {
            for (i = 0; i < nrow; i++) {
                out[j] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
    else{
        #pragma omp for schedule(static)
        for (i = 0; i < nrow; i++) {
            for (j = 0; j < ncol; j++){
                out[i] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
}
}

void NPzmultiplysum(double complex* out, double complex* a, double complex* b, 
                    int nrow, int ncol, int axis)
{
#pragma omp parallel
{
    size_t i, j;
    if (axis == 0){
        #pragma omp for schedule(static)
        for (j = 0; j < ncol; j++) {
            for (i = 0; i < nrow; i++) {
                out[j] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
    else{
        #pragma omp for schedule(static)
        for (i = 0; i < nrow; i++) {
            for (j = 0; j < ncol; j++){
                out[i] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
}
}

void NPcos(double* out, double* a, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = cos(a[i]);
    }
}
}

void NPsin(double* out, double* a, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = sin(a[i]);
    }
}
}

void NPdexp(double* out, double* a, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = exp(a[i]);
    }
}
}

void NPzexp(double complex* out, double complex* a, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = cexp(a[i]);
    }
}
}

void NPdsum(double* out, double* a, int lda, int n)
{
    size_t nrow = (size_t) n;
    size_t ncol = (size_t) lda;

    if (ncol > nrow) {
        size_t i;
        for (i = 0; i < nrow; i++) {
            out[i] = 0;
            double* aa = a + i*ncol;
#pragma omp parallel
{
            size_t j;
            double out_loc = 0;
            #pragma omp for schedule(static)
            for (j = 0; j < ncol; j++) {
                out_loc += aa[j];
            }
            #pragma omp critical
            out[i] += out_loc;
}
        }
    }
    else {

#pragma omp parallel
{
    size_t i, j;
    double *aa;
    if (ncol < 4) {
        #pragma omp for schedule(static)
        for (i=0; i<n; i++) {
            out[i] = 0;
            aa = a + i*ncol;
            for (j=0; j<ncol; j++) {
                out[i] += aa[j];
            }
        }
    }
    else {
        #pragma omp for schedule(static)
        for (i=0; i<nrow; i++) {
            out[i] = 0;
            aa = a + i*ncol;
            for (j=0; j<ncol-3; j+=4) {
                out[i] += aa[j];
                out[i] += aa[j+1];
                out[i] += aa[j+2];
                out[i] += aa[j+3];
            }
            for (j=j; j<ncol; j++) {
                out[i] += aa[j];
            }
        }
    }
}
    }
}

void NPzsum(double complex* out, double complex* a, int lda, int n)
{
#pragma omp parallel
{
    size_t i, j;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = 0;
        for (j = 0; j < lda; j++) {
            out[i] += a[i*lda+j];
        }
    }
}
}

void NPdadd(double* out, double* a, double* b, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}
}

void NPzadd(double complex* out, double complex* a, double complex* b, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}
}

void NPdsubtract(double* out, double* a, double* b, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}
}

void NPzsubtract(double complex* out, double complex* a, double complex* b, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}
}

void NPdmultiply(double* out, double* a, double* b, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}
}

void NPzmultiply(double complex* out, double complex* a, double complex* b, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}
}

void NPmultiply_dz(double complex* out, double* a, double complex* b, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] * creal(b[i]) + a[i] * cimag(b[i]) * _Complex_I;
    }
}
}

void NPmultiply_zd(double complex* out, double complex* b, double* a, int n)
{
#pragma omp parallel
{
    size_t i;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a[i] * creal(b[i]) + a[i] * cimag(b[i]) * _Complex_I;
    }
}
}

void NPdmultiply_scalar(double* out, double* matrix, double* number, size_t n)
{
#pragma omp parallel
{
    size_t i;
    double a = number[0];
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a * matrix[i];
    }
}
}

void NPzmultiply_scalar(double complex* out, double complex* matrix, double complex* number, size_t n)
{
#pragma omp parallel
{
    size_t i;
    double a = number[0];
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        out[i] = a * matrix[i];
    }
}
}

void NPdvdot(double* out, double* a, double* b, size_t n)
{
#pragma omp parallel
{
    size_t i;
    double sum_loc = 0;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        sum_loc += a[i] * b[i];
    }
    #pragma omp critical
    out[0] += sum_loc;
}
}

void NPzvdot(double complex* out, double complex* a, double complex* b, size_t n)
{
#pragma omp parallel
{
    size_t i;
    double complex sum_loc = 0;
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
        sum_loc += (creal(a[i]) - cimag(a[i])*_Complex_I) * b[i];
    }
    #pragma omp critical
    out[0] += sum_loc;
}
}

double* _drecursive_loop(double* out, double* array, int* out_shape,
                         int* shape, int idim, int ndim,
                         int axis, int istart, int iend)
{
    int i;
    size_t offset = 1;
    for (i = idim+1; i < ndim; i++) {
        offset *= out_shape[i];
    }

    double *pa = array, *pout=NULL;
    if (idim == axis) {
        for (i = istart; i < iend; i++) {
            pout = out + i * offset;
            pa = _drecursive_loop(pout, pa, out_shape, shape, idim+1, ndim, axis, istart, iend);
        }
    }
    else if (idim < ndim-1) {
        for (i = 0; i < shape[idim]; i++) {
            pout = out + i * offset;
            pa = _drecursive_loop(pout, pa, out_shape, shape, idim+1, ndim, axis, istart, iend);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < shape[idim]; i++) {
            out[i] = pa[i];
        }
        pa += shape[idim];
    }
    return pa;
}

void NPdconcatenate(double* out, double** arrays, int narrays,
                    int* out_shape, int* shapes, int ndim,
                    int axis)
{
    int iarray;
    int *shape;
    double *pa;

    int istart = 0;
    int iend = 0;
    for (iarray = 0; iarray < narrays; iarray++) {
        pa = arrays[iarray];
        shape = shapes + iarray*ndim;
        iend += shape[axis];
        _drecursive_loop(out, pa, out_shape, shape, 0, ndim, axis, istart, iend);
        istart = iend;
    }
}

double complex* _zrecursive_loop(double complex* out, double complex* array,
                                 int* out_shape, int* shape, int idim, int ndim,
                                 int axis, int istart, int iend)
{
    int i;
    size_t offset = 1;
    for (i = idim+1; i < ndim; i++) {
        offset *= out_shape[i];
    }

    double complex *pa = array, *pout=NULL;
    if (idim == axis) {
        for (i = istart; i < iend; i++) {
            pout = out + i * offset;
            pa = _zrecursive_loop(pout, pa, out_shape, shape, idim+1, ndim, axis, istart, iend);
        }
    }
    else if (idim < ndim-1) {
        for (i = 0; i < shape[idim]; i++) {
            pout = out + i * offset;
            pa = _zrecursive_loop(pout, pa, out_shape, shape, idim+1, ndim, axis, istart, iend);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < shape[idim]; i++) {
            out[i] = pa[i];
        }
        pa += shape[idim];
    }
    return pa;
}

void NPzconcatenate(double complex* out, double complex** arrays, int narrays,
                    int* out_shape, int* shapes, int ndim,
                    int axis)
{
    int iarray;
    int *shape;
    double complex *pa;

    int istart = 0;
    int iend = 0;
    for (iarray = 0; iarray < narrays; iarray++) {
        pa = arrays[iarray];
        shape = shapes + iarray*ndim;
        iend += shape[axis];
        _zrecursive_loop(out, pa, out_shape, shape, 0, ndim, axis, istart, iend);
        istart = iend;
    }
}
