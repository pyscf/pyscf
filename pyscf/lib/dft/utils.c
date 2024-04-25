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

#include <complex.h>
#include "config.h"
#include "vhf/fblas.h"
#if defined(HAVE_LIBXSMM)
#include "libxsmm.h"
#endif


void dgemm_wrapper(const char transa, const char transb,
                   const int m, const int n, const int k,
                   const double alpha, const double* a, const int lda,
                   const double* b, const int ldb,
                   const double beta, double* c, const int ldc)
{
#if defined(HAVE_LIBXSMM)
    if (transa == 'N') {
        //libxsmm_dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        int prefetch = LIBXSMM_PREFETCH_AUTO;
        int flags = transb != 'T' ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B;
        libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc,
                                                         &alpha, &beta, &flags, &prefetch);
        if (kernel) {
            kernel(a,b,c,a,b,c);
            return;
        }
    }
#endif
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void get_gga_vrho_gs(double complex *out, double complex *vrho_gs, double complex *vsigma1_gs,
                     double *Gv, double weight, int ngrid)
{
    int i;
    int ngrid2 = 2 * ngrid;
    double complex fac = -2. * _Complex_I;
#pragma omp parallel
{
    double complex v;
// ensure OpenMP 4.0
#if defined _OPENMP && _OPENMP >= 201307
    #pragma omp for simd schedule(static)
#else
    #pragma omp for schedule(static)
#endif
    for (i = 0; i < ngrid; i++) {
        v = ( Gv[i*3]   * vsigma1_gs[i]
             +Gv[i*3+1] * vsigma1_gs[i+ngrid]
             +Gv[i*3+2] * vsigma1_gs[i+ngrid2]) * fac + vrho_gs[i];
        out[i] = v * weight;
    }
}
}
