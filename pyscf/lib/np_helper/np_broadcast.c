/* Copyright 2025 The PySCF Developers. All Rights Reserved.

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

#include "np_helper/np_helper.h"
#include <complex.h>
#include <stdlib.h>

static const int BLOCKSIZE = 32;
static const int BLOCKSIZE_CPLX = 16;

/*
 * Performs the operation
 * C[:, i, j] += A[:, i, j] * B[i, j]
 * where A and C are 3D arrays and B is a 2D array.
 */
void NPomp_dmul_12(const size_t m, const size_t n, const size_t k,
                   const double *__restrict__ a, const size_t a_stride_0,
                   const size_t a_stride_1, double *__restrict__ b,
                   const size_t b_stride, double *__restrict__ c,
                   const size_t c_stride_0, const size_t c_stride_1) {
  const size_t kclean = k - k % BLOCKSIZE;
  const size_t k_rem = k - kclean;
#pragma omp parallel for schedule(static)
  for (size_t midx = 0; midx < m; midx++) {
    for (size_t tileidx = 0; tileidx < kclean; tileidx += BLOCKSIZE) {
      for (size_t nidx = 0; nidx < n; nidx++) {
#pragma omp simd
        for (size_t kidx = 0; kidx < BLOCKSIZE; kidx++) {
          c[midx * c_stride_0 + (tileidx + kidx) + c_stride_1 * nidx] +=
              a[midx * a_stride_0 + (tileidx + kidx) + a_stride_1 * nidx] *
              b[(tileidx + kidx) + b_stride * nidx];
        }
      }
    }
    for (size_t nidx = 0; nidx < n; nidx++) {
#pragma omp simd
      for (size_t kidx = 0; kidx < k_rem; kidx++) {
        c[midx * c_stride_0 + (kclean + kidx) + c_stride_1 * nidx] +=
            a[midx * a_stride_0 + (kclean + kidx) + a_stride_1 * nidx] *
            b[(kclean + kidx) + b_stride * nidx];
      }
    }
  }
}

/*
 * Performs the operation
 * C[:, i, j] += A[:, i, j] * B[i, j]
 * where A and C are 3D arrays and B is a 2D array.
 */
void NPomp_zmul_12(const size_t m, const size_t n, const size_t k,
                   const double complex *__restrict__ a, const size_t a_stride_0,
                   const size_t a_stride_1, double complex *__restrict__ b,
                   const size_t b_stride, double complex *__restrict__ c,
                   const size_t c_stride_0, const size_t c_stride_1) {
  const size_t kclean = k - k % BLOCKSIZE_CPLX;
  const size_t k_rem = k - kclean;
#pragma omp parallel for schedule(static)
  for (size_t midx = 0; midx < m; midx++) {
    for (size_t tileidx = 0; tileidx < kclean; tileidx += BLOCKSIZE_CPLX) {
      for (size_t nidx = 0; nidx < n; nidx++) {
#pragma omp simd
        for (size_t kidx = 0; kidx < BLOCKSIZE_CPLX; kidx++) {
          c[midx * c_stride_0 + (tileidx + kidx) + c_stride_1 * nidx] +=
              a[midx * a_stride_0 + (tileidx + kidx) + a_stride_1 * nidx] *
              b[(tileidx + kidx) + b_stride * nidx];
        }
      }
    }
    for (size_t nidx = 0; nidx < n; nidx++) {
#pragma omp simd
      for (size_t kidx = 0; kidx < k_rem; kidx++) {
        c[midx * c_stride_0 + (kclean + kidx) + c_stride_1 * nidx] +=
            a[midx * a_stride_0 + (kclean + kidx) + a_stride_1 * nidx] *
            b[(kclean + kidx) + b_stride * nidx];
      }
    }
  }
}
