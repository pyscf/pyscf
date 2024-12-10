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
 * Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
 */

#include <complex.h>
#include <math.h>
#include "np_helper.h"

const int TILESIZE = 32;
const int TILESIZE_CPLX = 16;

/*
 * Calculate the largest integer i such that
 *  i * (i - 1) / 2 <= ijouter.
 */
static inline int uncollapse_loop_index(const long long ijouter)
{
  return (int) floor((sqrt(0.25 + 2.0 * ijouter) + 0.5));
}

static inline void dtranspose_scale_tile_offdiag(double *A, const int ii,
                                                 const int jj,
                                                 const size_t lda_w,
                                                 const double alpha) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < ii + TILESIZE; i++) {
      const double tmp = A[i * lda_w + j];
      A[i * lda_w + j] = alpha * A[j * lda_w + i];
      A[j * lda_w + i] = alpha * tmp;
    }
  }
}

static inline void dtranspose_scale_tile_diag(double *A, const int ii,
                                              const int jj, const size_t lda_w,
                                              const double alpha) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double tmp = A[i * lda_w + j];
      A[i * lda_w + j] = alpha * A[j * lda_w + i];
      A[j * lda_w + i]= alpha * tmp;
    }
    A[j * lda_w + j] *= alpha;
  }
}

static inline void dtranspose_tile_diag(double *A, const int ii, const int jj,
                                        const size_t lda_w) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double tmp = A[i * lda_w + j];
      A[i * lda_w + j] = A[j * lda_w + i];
      A[j * lda_w + i] = tmp;
    }
  }
}

static inline void ztranspose_scale_tile_offdiag(double complex *A,
                                                 const int ii, const int jj,
                                                 const size_t lda_w,
                                                 const double complex alpha) {
  for (int j = jj; j < jj + TILESIZE_CPLX; j++) {
#pragma omp simd
    for(int i = ii; i < ii + TILESIZE_CPLX; i++) {
      const double complex tmp = A[i * lda_w + j];
      A[i * lda_w + j] = alpha * A[j * lda_w + i];
      A[j * lda_w + i] = alpha * tmp;
    }
  }
}

static inline void ztranspose_scale_tile_diag(double complex *A, const int ii,
                                              const int jj, const size_t lda_w,
                                              const double complex alpha) {
  for (int j = jj; j < jj + TILESIZE_CPLX; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double complex tmp = A[i * lda_w + j];
      A[i * lda_w + j] = alpha * A[j * lda_w + i];
      A[j * lda_w + i] = alpha * tmp;
    }
    A[j * lda_w + j] *= alpha;
  }
}

static inline void ztranspose_tile_diag(double complex *A, const int ii,
                                       const int jj, const size_t lda_w) {
  for (int j = jj; j < jj + TILESIZE_CPLX; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double complex tmp = A[i * lda_w + j];
      A[i * lda_w + j] = A[j * lda_w + i];
      A[j * lda_w + i] = tmp;
    }
  }
}

/*
 * In-place parallel matrix transpose, double version.
 * See https://colfaxresearch.com/multithreaded-transposition-of-square-matrices-with-common-code-for-intel-xeon-processors-and-intel-xeon-phi-coprocessors/
 */
void NPomp_d_itranspose_scale(const int n, const double alpha, double *A, int lda)
{
  const int nclean = n - n % TILESIZE;
  const int ntiles = nclean / TILESIZE;
  const size_t lda_w = (size_t) lda;

#pragma omp parallel
  {

/*
 *   ---------------------------
 *  |       ******************  |
 *  |       ******************  |
 *  |       ******************  |
 *  | ******      ************  |
 *  | ******      ************  |
 *  | ******      ************  |
 *  | ************      ******  |
 *  | ************      ******  |
 *  | ************      ******  |
 *  | ******************        |
 *  | ******************        |
 *  | ******************        |
 *  |                           |
 *   ----------------------------
 */

/*
 *    The following loop nest is equivalent to:
 *    for(int iouter = 1; iouter < ntiles; iouter++)
 *      for(int jouter = 0; jouter < iouter; jouter++)
 *
 *    See 10.1109/IPDPS.2017.34. 
 */
    int first_iteration = 1;
    int iouter, jouter;
#pragma omp for schedule(static) nowait
    for(long long ijouter = 0; ijouter < (ntiles*(ntiles-1))/2; ijouter++) {
      if(first_iteration) {
        iouter = uncollapse_loop_index(ijouter);
        jouter = ijouter - iouter * (iouter - 1) / 2;
        first_iteration = 0;
      } else {
        jouter++;
        if(jouter == iouter) {
          iouter++;
          jouter = 0;
        }
      }
      dtranspose_scale_tile_offdiag(A, iouter * TILESIZE, jouter * TILESIZE, lda_w, alpha);
    }


/*
 *   ---------------------------
 *  | ******                    |
 *  | ******                    |
 *  | ******                    |
 *  |       ******              |
 *  |       ******              |
 *  |       ******              |
 *  |             ******        |
 *  |             ******        |
 *  |             ******        |
 *  |                   ******  |
 *  |                   ******  |
 *  |                   ******  |
 *  |                           |
 *   ----------------------------
 */

    if(alpha != 1.0) {
#pragma omp for schedule(static) nowait
      for(int ii = 0; ii < nclean; ii+=TILESIZE) {
        dtranspose_scale_tile_diag(A, ii, ii, lda_w, alpha);
      }
    } else {
#pragma omp for schedule(static) nowait
      for(int ii = 0; ii < nclean; ii+=TILESIZE) {
        dtranspose_tile_diag(A, ii, ii, lda_w);
      }
    }


/*
 *   --------------------------
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  |                        ***|
 *  | ***********************   |
 *  | ***********************   |
 *   ---------------------------
 */

#pragma omp for schedule(static) nowait
    for(int j = 0; j < nclean; j++) {
      for(int i = nclean; i < n; i++) {
        const double tmp = A[i * lda_w + j];
        A[i * lda_w + j] = alpha * A[j * lda_w + i];
        A[j * lda_w + i] = alpha * tmp;
      }
    }
  } // end parallel region

/*
 *   --------------------------
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                           |
 *  |                        ***|
 *  |                        ***|
 *   ---------------------------
 */

  for(int j = nclean; j < n; j++) {
    for(int i = nclean; i < j; i++) {
      const double tmp = A[i * lda_w + j];
      A[i * lda_w + j] = alpha * A[j * lda_w + i];
      A[j * lda_w + i] = alpha * tmp;
    }
  }

  if(alpha != 1.0) {
    for(int i = nclean; i < n; i++) {
      A[i * lda_w + i] *= alpha;
    }
  }

}

/*
 * In-place parallel matrix transpose, double complex version.
 * See https://colfaxresearch.com/multithreaded-transposition-of-square-matrices-with-common-code-for-intel-xeon-processors-and-intel-xeon-phi-coprocessors/
 */
void NPomp_z_itranspose_scale(const int n, const double complex *alphaptr, double complex *A, int lda)
{
  const double complex alpha = *alphaptr;
  const int nclean = n - n % TILESIZE_CPLX;
  const int ntiles = nclean / TILESIZE_CPLX;
  const size_t lda_w = (size_t) lda;

#pragma omp parallel
  {

/*
 *    The following loop nest is equivalent to:
 *    for(int iouter = 1; iouter < ntiles; iouter++)
 *      for(int jouter = 0; jouter < iouter; jouter++)
 *
 *    See 10.1109/IPDPS.2017.34. 
 */
    int first_iteration = 1;
    int iouter, jouter;
#pragma omp for schedule(static) nowait
    for(long long ijouter = 0; ijouter < (ntiles*(ntiles-1))/2; ijouter++) {
      if(first_iteration) {
        iouter = uncollapse_loop_index(ijouter);
        jouter = ijouter - iouter * (iouter - 1) / 2;
        first_iteration = 0;
      } else {
        jouter++;
        if(jouter == iouter) {
          iouter++;
          jouter = 0;
        }
      }
      ztranspose_scale_tile_offdiag(A, iouter * TILESIZE_CPLX, jouter * TILESIZE_CPLX, lda_w, alpha);
    }

    if(alpha != 1.0) {
#pragma omp for schedule(static) nowait
      for(int ii = 0; ii < nclean; ii+=TILESIZE_CPLX) {
        ztranspose_scale_tile_diag(A, ii, ii, lda_w, alpha);
      }
    } else {
#pragma omp for schedule(static) nowait
      for(int ii = 0; ii < nclean; ii+=TILESIZE_CPLX) {
        ztranspose_tile_diag(A, ii, ii, lda_w);
      }
    }

#pragma omp for schedule(static) nowait
    for(int j = 0; j < nclean; j++) {
      for(int i = nclean; i < n; i++) {
        const double complex tmp = A[i * lda_w + j];
        A[i * lda_w + j] = alpha * A[j * lda_w + i];
        A[j * lda_w + i] = alpha * tmp;
      }
    }

  } // end parallel region

  for(int j = nclean; j < n; j++) {
    for(int i = nclean; i < j; i++) {
        const double complex tmp = A[i * lda_w + j];
        A[i * lda_w + j] = alpha * A[j * lda_w + i];
        A[j * lda_w + i] = alpha * tmp;
    }
  }

  if(alpha != 1.0) {
    for(int i = nclean; i < n; i++) {
      A[i * lda_w + i] *= alpha;
    }
  }
}



/*
 * Batched versions for 3D tensors
 */

void NPomp_dtensor_itranspose_scale021(const long long matstride, int nmat, int n, const double alpha,
                                      double *A, int lda)
{
  for (int imat = 0; imat < nmat; imat++) {
    NPomp_d_itranspose_scale(n, alpha, A + imat * matstride, lda);
  }
}

void NPomp_ztensor_itranspose_scale021(const long long matstride, int nmat, int n, const double complex *alpha,
                                      double complex *A, int lda)
{
    for (int imat = 0; imat < nmat; imat++) {
      NPomp_z_itranspose_scale(n, alpha, A + imat * matstride, lda);
    }
}