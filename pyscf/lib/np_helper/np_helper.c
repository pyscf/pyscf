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

void NPdmultiplysum(double* out, double* a, double* b, int nrow, int ncol, int axis)
{
    if (axis == 0){
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < ncol; j++) {
            for (size_t i = 0; i < nrow; i++) {
                out[j] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
    else{
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrow; i++) {
            for (size_t j = 0; j < ncol; j++){
                out[i] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
}

void NPzmultiplysum(double complex* out, double complex* a, double complex* b, 
                    int nrow, int ncol, int axis)
{
    if (axis == 0){
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < ncol; j++) {
            for (size_t i = 0; i < nrow; i++) {
                out[j] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
    else{
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nrow; i++) {
            for (size_t j = 0; j < ncol; j++){
                out[i] += a[i*ncol+j] * b[i*ncol+j];
            }
        }
    }
}

void NPdexp(double* out, double* a, int n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<n; i++) {
        out[i] = exp(a[i]);
    }
}

void NPzexp(double complex* out, double complex* a, int n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<n; i++) {
        out[i] = cexp(a[i]);
    }
}
