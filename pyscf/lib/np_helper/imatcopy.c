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
#include <omp.h>
#include "np_helper.h"

const int TILESIZE = 32;

#define ELEM_AT(A, i, j, lda) (A)[(i) * (lda) + (j)]

static inline void dtranspose_scale_tile_offdiag(double *A, const int ii, const int jj,
                                          int lda, const double alpha) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < ii + TILESIZE; i++) {
      const double tmp = ELEM_AT(A, i, j, lda);
      ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
      ELEM_AT(A, j, i, lda) = alpha * tmp;
    }
  }
}

static inline void dtranspose_scale_tile_diag(double *A, const int ii, const int jj,
                                       int lda, const double alpha) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double tmp = ELEM_AT(A, i, j, lda);
      ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
      ELEM_AT(A, j, i, lda) = alpha * tmp;
    }
    ELEM_AT(A, j, j, lda) *= alpha;
  }
}

static inline void dtranspose_tile_diag(double *A, const int ii, const int jj,
                                       int lda) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double tmp = ELEM_AT(A, i, j, lda);
      ELEM_AT(A, i, j, lda) = ELEM_AT(A, j, i, lda);
      ELEM_AT(A, j, i, lda) = tmp;
    }
  }
}

static inline void ztranspose_scale_tile_offdiag(double complex *A, const int ii,
                                          const int jj, int lda,
                                          const double complex alpha) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < ii + TILESIZE; i++) {
      const double complex tmp = ELEM_AT(A, i, j, lda);
      ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
      ELEM_AT(A, j, i, lda) = alpha * tmp;
    }
  }
}

static inline void ztranspose_scale_tile_diag(double complex *A, const int ii,
                                       const int jj, int lda,
                                       const double complex alpha) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double complex tmp = ELEM_AT(A, i, j, lda);
      ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
      ELEM_AT(A, j, i, lda) = alpha * tmp;
    }
    ELEM_AT(A, j, j, lda) *= alpha;
  }
}

static inline void ztranspose_tile_diag(double complex *A, const int ii,
                                       const int jj, int lda) {
  for (int j = jj; j < jj + TILESIZE; j++) {
#pragma omp simd
    for(int i = ii; i < j; i++) {
      const double complex tmp = ELEM_AT(A, i, j, lda);
      ELEM_AT(A, i, j, lda) = ELEM_AT(A, j, i, lda);
      ELEM_AT(A, j, i, lda) = tmp;
    }
  }
}

/*
 * In-place parallel matrix transpose
 * See https://colfaxresearch.com/multithreaded-transposition-of-square-matrices-with-common-code-for-intel-xeon-processors-and-intel-xeon-phi-coprocessors/
 */
void NPomp_d_itranspose_scale(const int n, const double alpha, double *A, int lda)
{
  const int nclean = n - n % TILESIZE;
  const int ntiles = nclean / TILESIZE;

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

#pragma omp for collapse(2) nowait
    for(int iouter = 1; iouter < ntiles; iouter++) {
      for(int jouter = 0; jouter < iouter; jouter++) {
        dtranspose_scale_tile_offdiag(A, iouter * TILESIZE, jouter * TILESIZE, lda, alpha);
      }
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
        dtranspose_scale_tile_diag(A, ii, ii, lda, alpha);
      }
    } else {
#pragma omp for schedule(static) nowait
      for(int ii = 0; ii < nclean; ii+=TILESIZE) {
        dtranspose_tile_diag(A, ii, ii, lda);
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
        const double tmp = ELEM_AT(A, i, j, lda);
        ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
        ELEM_AT(A, j, i, lda) = alpha * tmp;
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
      const double tmp = ELEM_AT(A, i, j, lda);
      ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
      ELEM_AT(A, j, i, lda) = alpha * tmp;
    }
  }

  if(alpha != 1.0) {
    for(int i = nclean; i < n; i++) {
      ELEM_AT(A, i, i, lda) *= alpha;
    }
  }

}


void NPomp_z_itranspose_scale(const int n, const double complex *alphaptr, double complex *A, int lda)
{
  const double complex alpha = *alphaptr;
  const int nclean = n - n % TILESIZE;
  const int ntiles = nclean / TILESIZE;

#pragma omp parallel
  {

#pragma omp for collapse(2) nowait
    for(int iouter = 1; iouter < ntiles; iouter++) {
      for(int jouter = 0; jouter < iouter; jouter++) {
        ztranspose_scale_tile_offdiag(A, iouter * TILESIZE, jouter * TILESIZE, lda, alpha);
      }
    }

    if(alpha != 1.0) {
#pragma omp for schedule(static) nowait
      for(int ii = 0; ii < nclean; ii+=TILESIZE) {
        ztranspose_scale_tile_diag(A, ii, ii, lda, alpha);
      }
    } else {
#pragma omp for schedule(static) nowait
      for(int ii = 0; ii < nclean; ii+=TILESIZE) {
        ztranspose_tile_diag(A, ii, ii, lda);
      }
    }

#pragma omp for schedule(static) nowait
    for(int j = 0; j < nclean; j++) {
      for(int i = nclean; i < n; i++) {
        const double complex tmp = ELEM_AT(A, i, j, lda);
        ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
        ELEM_AT(A, j, i, lda) = alpha * tmp;
      }
    }

  } // end parallel region

  for(int j = nclean; j < n; j++) {
    for(int i = nclean; i < j; i++) {
        const double complex tmp = ELEM_AT(A, i, j, lda);
        ELEM_AT(A, i, j, lda) = alpha * ELEM_AT(A, j, i, lda);
        ELEM_AT(A, j, i, lda) = alpha * tmp;
    }
  }

  if(alpha != 1.0) {
    for(int i = nclean; i < n; i++) {
      ELEM_AT(A, i, i, lda) *= alpha;
    }
  }
}



/*
 * Batched versions for 3D tensors
 */

void NPomp_dtensor_itranspose_scale021(const ssize_t matstride, int nmat, int n, const double alpha,
                                      double *A, int lda)
{
    if (omp_get_num_threads() >= nmat) {
#pragma omp parallel
{
    int nlevels = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
#pragma omp for schedule(static)
    for (int imat = 0; imat < nmat; imat++) {
      NPomp_d_itranspose_scale(n, alpha, A + imat * matstride, lda);
    }
    omp_set_max_active_levels(nlevels);
}
  } else {
    for (int imat = 0; imat < nmat; imat++) {
      NPomp_d_itranspose_scale(n, alpha, A + imat * matstride, lda);
    }
  }
}

void NPomp_ztensor_itranspose_scale021(const ssize_t matstride, int nmat, int n, const double complex *alpha,
                                      double complex *A, int lda)
{
    if (omp_get_num_threads() >= nmat) {
#pragma omp parallel
{
    int nlevels = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
#pragma omp for schedule(static)
    for (int imat = 0; imat < nmat; imat++) {
      NPomp_z_itranspose_scale(n, alpha, A + imat * matstride, lda);
    }
    omp_set_max_active_levels(nlevels);
}
  } else {
    for (int imat = 0; imat < nmat; imat++) {
      NPomp_z_itranspose_scale(n, alpha, A + imat * matstride, lda);
    }
  }
}