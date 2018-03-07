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

#include<stdio.h>
#include<stdlib.h>
#include <omp.h>
#include <cblas.h>

/*
!
!
!Compute Y += A*X for CSR matrix A and dense vectors X,Y
! From scipy/sparse/sparsetools/csr.h
!
!
! Input Arguments:
!   I  n_row         - number of rows in A
!   I  n_col         - number of columns in A
!   I  Ap[n_row+1]   - row pointer
!   I  Aj[nnz(A)]    - column indices
!   T  Ax[nnz(A)]    - nonzeros
!   T  Xx[n_col]     - input vector
!
! Output Arguments:
!  T  Yx[n_row]     - output vector
!
! Note:
!   Output array Yx must be preallocated
!
!   Complexity: Linear.  Specifically O(nnz(A) + n_row)
*/
void scsr_matvec(int nrow, int ncol, int nnz, int *Ap, int *Aj, 
    float *Ax, float *Xx, float *Yx)
{

  int i, jj;
  float sum = 0.0;

  # pragma omp parallel \
  shared (nrow, Yx, Ap, Ax, Xx, Aj) \
  private (i, jj, sum)
  {
    #pragma omp for
    for(i = 0; i < nrow; i++){
      sum = Yx[i];
      for(jj = Ap[i]; jj < Ap[i+1]; jj++){
        sum += Ax[jj] * Xx[Aj[jj]];
      }
      Yx[i] = sum;
    }
  }
}


void dcsr_matvec(int nrow, int ncol, int nnz, int *Ap, int *Aj, 
    double *Ax, double *Xx, double *Yx)
{

  int i, jj;
  double sum = 0.0;

  # pragma omp parallel \
  shared (nrow, Yx, Ap, Ax, Xx, Aj) \
  private (i, jj, sum)
  {
    #pragma omp for
    for(i = 0; i < nrow; i++){
      sum = Yx[i];
      for(jj = Ap[i]; jj < Ap[i+1]; jj++){
        sum += Ax[jj] * Xx[Aj[jj]];
      }
      Yx[i] = sum;
    }
  }
}


/*
 * Compute Y += A*X for CSC matrix A and dense vectors X,Y
 * From scipy/sparse/sparsetools/csc.h
 *
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - column pointer
 *   I  Ai[nnz(A)]    - row indices
 *   T  Ax[n_col]     - nonzeros
 *   T  Xx[n_col]     - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *
 * Note:
 *   Output array Yx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + n_col)
 *
 */
void scsc_matvec(int n_row, int n_col, int nnz,
            int *Ap, int *Ai, float *Ax, float *Xx, float *Yx)
{
    int col_start, col_end, j, ii, i;

    for( j = 0; j < n_col; j++){
        col_start = Ap[j];
        col_end   = Ap[j+1];

        for( ii = col_start; ii < col_end; ii++){
            i    = Ai[ii];
            Yx[i] += Ax[ii] * Xx[j];
        }
    }
}

void dcsc_matvec(int n_row, int n_col, int nnz,
            int *Ap, int *Ai, double *Ax, double *Xx, double *Yx)
{
    int col_start, col_end, j, ii, i;

    for( j = 0; j < n_col; j++){
        col_start = Ap[j];
        col_end   = Ap[j+1];

        for( ii = col_start; ii < col_end; ii++){
            i    = Ai[ii];
            Yx[i] += Ax[ii] * Xx[j];
        }
    }
}


/*
 * Compute Y += A*X for CSC matrix A and dense block vectors X,Y
 * From scipy/sparse/sparsetools/csc.h
 *
 *
 * Input Arguments:
 *   I  n_row            - number of rows in A
 *   I  n_col            - number of columns in A
 *   I  n_vecs           - number of column vectors in X and Y
 *   I  Ap[n_row+1]      - row pointer
 *   I  Aj[nnz(A)]       - column indices
 *   T  Ax[nnz(A)]       - nonzeros
 *   T  Xx[n_col,n_vecs] - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row,n_vecs] - output vector
 *
 * Note:
 *   Output array Yx must be preallocated
 *
*/
void scsc_matvecs(int n_row, int n_col, int n_vecs, 
      int *Ap, int *Ai, float *Ax, float *Xx, float *Yx)
{
  int i, j, ii;
  /*
  # pragma omp parallel \
  shared (n_row, n_col, n_vecs, Ap, Ai, Ax, Xx, Yx) \
  private (i, ii, j)
  {
    #pragma omp for
    */
    for( j = 0; j < n_col; j++){
      for( ii = Ap[j]; ii < Ap[j+1]; ii++){
        i = Ai[ii];
        //axpy(n_vecs, Ax[ii], Xx + (int)n_vecs * j, Yx + (int)n_vecs * i);
        cblas_saxpy (n_vecs, Ax[ii], &Xx[n_vecs*j], 1, &Yx[n_vecs*i], 1);
      }
    }
  //}
}

void dcsc_matvecs(int n_row, int n_col, int n_vecs, 
      int *Ap, int *Ai, double *Ax, double *Xx, double *Yx)
{
  int i, j, ii;
  /*
  # pragma omp parallel \
  shared (n_row, n_col, n_vecs, Ap, Ai, Ax, Xx, Yx) \
  private (i, ii, j)
  {
    #pragma omp for
    */
    for( j = 0; j < n_col; j++){
      for( ii = Ap[j]; ii < Ap[j+1]; ii++){
        i = Ai[ii];
        //axpy(n_vecs, Ax[ii], Xx + (int)n_vecs * j, Yx + (int)n_vecs * i);
        cblas_daxpy (n_vecs, Ax[ii], &Xx[n_vecs*j], 1, &Yx[n_vecs*i], 1);
      }
    }
  //}
}
