#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include<sys/param.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "m_iter_gpu_routines.h"

float *X4_d, *ksn2e_d, *ksn2f_d;
float *v_ext_d, *chi0_d;
float *vdp_d, *sab_d, *nb2v_d, *nm2v_re_d, *nm2v_im_d;
int norbs, nfermi, vstart, nprod, nvirt;
scsr_matrix cc_da_d, v_dab_d;
cusparseHandle_t handle_cuparse=0;
cublasHandle_t handle_cublas;


int sum_int_vec(int *mat, int N)
{
  int i;
  int sum_int = 0;

  for (i=0; i<N; i++)
  { 
    sum_int += mat[i];
  }

  return sum_int;
}

float sum_float_vec(float *mat, int N)
{
  int i;
  float sum_float = 0;

  for (i=0; i<N; i++)
  { 
    sum_float += abs(mat[i]);
  }

  return sum_float;
}

float sum_array_gpu(float *dev, int N1, int N2)
{
  float *temp;
  temp = (float*) malloc(sizeof(float)*N1*N2);
  checkCudaErrors(cudaMemcpy( temp, dev, sizeof(float) * N1*N2, cudaMemcpyDeviceToHost));
  
  float sum = sum_float_vec(temp, N1*N2);
  return sum;
}

/*
  initialize sparse matrix on the gpu
*/
extern "C" scsr_matrix init_sparse_matrix_csr_gpu_float(float *csrValA, int *csrRowPtrA, 
    int *csrColIndA, int m, int n, int nnz, int RowPtrSize)
{

  scsr_matrix csr;

  csr.m = m;
  csr.n = n;
  csr.nnz = nnz;
  csr.RowPtrSize = RowPtrSize;

  checkCudaErrors(cudaMalloc( (void **)&csr.data, sizeof(float) * nnz));
  checkCudaErrors(cudaMalloc( (void **)&csr.ColInd, sizeof(float) * nnz));
  checkCudaErrors(cudaMalloc( (void **)&csr.RowPtr, sizeof(int) * RowPtrSize));

  checkCudaErrors(cudaMemcpy( csr.data, csrValA, sizeof(float) * nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( csr.ColInd, csrColIndA, sizeof(int) * nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( csr.RowPtr, csrRowPtrA, sizeof(int) * RowPtrSize, cudaMemcpyHostToDevice));

  // Initialize and setup matrix descriptor
  checkCudaErrors(cusparseCreateMatDescr(&csr.descr)); 
  
  checkCudaErrors(cusparseSetMatType(csr.descr,CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatIndexBase(csr.descr, CUSPARSE_INDEX_BASE_ZERO));  

  return csr;
}

extern "C" void free_csr_matrix_gpu(scsr_matrix csr)
{
  checkCudaErrors(cudaFree(csr.data));
  checkCudaErrors(cudaFree(csr.ColInd));
  checkCudaErrors(cudaFree(csr.RowPtr));

  checkCudaErrors(cusparseDestroyMatDescr(csr.descr));
}

void normalize_energy_cpu(float *ksn2e, float *ksn2f, double omega_re, double omega_im, float *nm2v_re, float *nm2v_im,
      int nfermi, int norbs, int nvirt, int vstart)
{
  int i, j;
  float en=0.0, fn=0.0, em=0.0, fm=0.0, old_re, old_im;
  double d1p, d1pp, d2p, d2pp, alpha, beta;
  int m, index, count;

  FILE *f = fopen("C_version_norm.txt", "w");
  
  if (f == NULL)
  {
    printf("Error opening file!\n");
    exit(1);
  }

  count = 0;
  for (i =0; i < nfermi; i++)
  {
    en = ksn2e[i];
    fn = ksn2f[i];
    for (j=i+1; j < norbs; j++)
    {
      em = ksn2e[j];
      fm = ksn2f[j];

      m = j - vstart;
      index = i*nvirt + m;

      d1p = omega_re - (em-en); d1pp = omega_im;
      d2p = omega_re + (em-en); d2pp = omega_im;
      
      alpha = d1p/(d1p*d1p + d1pp*d1pp) - d2p/(d2p*d2p + d2pp*d2pp);
      beta = -d1pp/(d1p*d1p + d1pp*d1pp) + d2pp/(d2p*d2p + d2pp*d2pp);
      old_re = nm2v_re[index];
      old_im = nm2v_im[index];

      nm2v_re[index] = (fn - fm)*(old_re*alpha - old_im*beta);
      nm2v_im[index] = (fn - fm)*(old_re*beta + old_im*alpha);
      fprintf(f, "%d %d %d %d %d %f %f %f %f %f %f %f %f %f\n", count, i, j, m, index,
          en, em, fn, fm, old_re, old_im, nm2v_re[index], omega_re, omega_im);
      count += 1;
    }
  }

  fclose(f);
}


//ksn2e, ksn2f, nfermi, vstart, comega, nm2v_re, nm2v_im, ksn2e_dim
__global__ void normalize_energy_gpu(float *ksn2e, float *ksn2f, double omega_re, double omega_im, float *nm2v_re, float *nm2v_im,
      int nfermi, int norbs, int nvirt, int vstart)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; //nocc
  int j = blockIdx.y * blockDim.y + threadIdx.y; //nvirt
  float en=0.0, fn=0.0, em=0.0, fm=0.0, old_re, old_im;
  double d1p, d1pp, d2p, d2pp, alpha, beta;
  int m;

  if (i < nfermi)
  {
    en = ksn2e[i];
    fn = ksn2f[i];
    if (j>i && j < norbs )
    {
      em = ksn2e[j];
      fm = ksn2f[j];

      m = j - vstart;

      d1p = omega_re - (em-en); d1pp = omega_im;
      d2p = omega_re + (em-en); d2pp = omega_im;
      
      alpha = d1p/(d1p*d1p + d1pp*d1pp) - d2p/(d2p*d2p + d2pp*d2pp);
      beta = -d1pp/(d1p*d1p + d1pp*d1pp) + d2pp/(d2p*d2p + d2pp*d2pp);
      old_re = nm2v_re[i*nvirt + m];
      old_im = nm2v_im[i*nvirt + m];

      nm2v_re[i*nvirt + m] = (fn - fm)*(old_re*alpha - old_im*beta);
      nm2v_im[i*nvirt + m] = (fn - fm)*(old_re*beta + old_im*alpha);
      //printf("i = %d, j = %d, m = %d, alpha = %f, beta = %f, old_re = %f, old_im = %f, nm2v_re = %f, nm2v_im = %f\n", 
      //    i, j, m, alpha, beta, old_re, old_im, nm2v_re[index], nm2v_im[index]);

      //nm2v = nm2v * (fn-fm) * ( 1.0 / (comega - (em - en)) - 1.0 /(comega + (em - en)) );
    }
  }
}

extern "C" void init_tddft_iter_gpu(float *X4, int norbs_in, float *ksn2e,
                  float *ksn2f, int nfermi_in, int nprod_in, int vstart_in,
                  float *cc_da_vals, int *cc_da_rowPtr, int *cc_da_col_ind,
                  int *cc_da_shape, int cc_da_nnz, int cc_da_indptr_size,
                  float *v_dab_vals, int *v_dab_rowPtr, int *v_dab_col_ind,
                  int *v_dab_shape, int v_dab_nnz, int v_dab_indptr_size)
{

  norbs = norbs_in;
  nfermi = nfermi_in;
  nprod = nprod_in;
  vstart = vstart_in;
  nvirt = norbs - vstart;

  // init sparse matrices on GPU
  cc_da_d = init_sparse_matrix_csr_gpu_float(cc_da_vals, cc_da_rowPtr, 
                  cc_da_col_ind, cc_da_shape[0], cc_da_shape[1], cc_da_nnz, cc_da_indptr_size);

  v_dab_d = init_sparse_matrix_csr_gpu_float(v_dab_vals, v_dab_rowPtr, 
                  v_dab_col_ind, v_dab_shape[0], v_dab_shape[1], v_dab_nnz, v_dab_indptr_size);

  // For pascal GPU, the cudaMallocManaged() could probably allow to run larger systems.
  // Need to llok for more informations about this!!
  checkCudaErrors(cusparseCreate(&handle_cuparse));
  checkCudaErrors(cublasCreate(&handle_cublas));

  checkCudaErrors(cudaMalloc( (void **)&X4_d, sizeof(float) * norbs*norbs));
  checkCudaErrors(cudaMalloc( (void **)&ksn2e_d, sizeof(float) * norbs));
  checkCudaErrors(cudaMalloc( (void **)&ksn2f_d, sizeof(float) * norbs));
  
  checkCudaErrors(cudaMalloc( (void **)&v_ext_d, sizeof(float) * nprod));
  checkCudaErrors(cudaMalloc( (void **)&chi0_d, sizeof(float) * nprod));

  checkCudaErrors(cudaMalloc( (void **)&vdp_d, sizeof(float) * cc_da_d.m));
  checkCudaErrors(cudaMalloc( (void **)&sab_d, sizeof(float) * v_dab_d.n));
  checkCudaErrors(cudaMalloc( (void **)&nb2v_d, sizeof(float) * nfermi*norbs));
  checkCudaErrors(cudaMalloc( (void **)&nm2v_re_d, sizeof(float) * nfermi*nvirt));
  checkCudaErrors(cudaMalloc( (void **)&nm2v_im_d, sizeof(float) * nfermi*nvirt));

  checkCudaErrors(cudaMemcpy( X4_d, X4, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( ksn2e_d, ksn2e, sizeof(float) * norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( ksn2f_d, ksn2f, sizeof(float) * norbs, cudaMemcpyHostToDevice));

}

extern "C" void free_device()
{

  checkCudaErrors(cudaFree(X4_d));
  checkCudaErrors(cudaFree(ksn2e_d));
  checkCudaErrors(cudaFree(ksn2f_d));

  checkCudaErrors(cudaFree(v_ext_d));
  checkCudaErrors(cudaFree(chi0_d));
  checkCudaErrors(cudaFree(vdp_d));
  checkCudaErrors(cudaFree(sab_d));
  checkCudaErrors(cudaFree(nb2v_d));
  checkCudaErrors(cudaFree(nm2v_re_d));
  checkCudaErrors(cudaFree(nm2v_im_d));

  free_csr_matrix_gpu(cc_da_d);
  free_csr_matrix_gpu(v_dab_d);

  checkCudaErrors(cusparseDestroy(handle_cuparse));
  checkCudaErrors(cublasDestroy(handle_cublas));
}

extern "C" void apply_rf0_device(float *v_ext_real, float *v_ext_imag, double omega_re, 
      double omega_im, float *chi0_re, float *chi0_im, int *block_size, int *grid_size)
{
  float alpha = 1.0, beta = 0.0;

  dim3 dimBlock(block_size[0], block_size[1]);
  dim3 dimGrid(grid_size[0], grid_size[1]);
  
  // real part first
  checkCudaErrors(cudaMemcpy( v_ext_d, v_ext_real, sizeof(float) * nprod, cudaMemcpyHostToDevice));

  /*
     cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA, 
        int m, int n, int nnz, const float           *alpha, 
        const cusparseMatDescr_t descrA, 
        const float           *csrValA, 
        const int *csrRowPtrA, const int *csrColIndA,
        const float           *x, const float           *beta, 
        float           *y)
  */
  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cc_da_d.m, cc_da_d.n, cc_da_d.nnz, &alpha,
        cc_da_d.descr, cc_da_d.data, cc_da_d.RowPtr, cc_da_d.ColInd, 
        v_ext_d, &beta, vdp_d));
  
  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_TRANSPOSE,
        v_dab_d.m, v_dab_d.n, v_dab_d.nnz, &alpha,
        v_dab_d.descr, v_dab_d.data, v_dab_d.RowPtr, v_dab_d.ColInd, 
        vdp_d, &beta, sab_d));

  // Copy imaginary part to GPU to verlap copy and execution
  checkCudaErrors(cudaMemcpyAsync( v_ext_d, v_ext_imag, sizeof(float) * nprod, cudaMemcpyHostToDevice, 0));
  
  checkCudaErrors(cublasSgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, norbs, &alpha, sab_d, norbs,
        X4_d, norbs, &beta, nb2v_d, norbs));
  checkCudaErrors(cublasSgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha, &X4_d[vstart*norbs], norbs, nb2v_d,
       norbs, &beta, nm2v_re_d, nvirt));

   // imaginary part
  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cc_da_d.m, cc_da_d.n, cc_da_d.nnz, &alpha,
        cc_da_d.descr, cc_da_d.data, cc_da_d.RowPtr, cc_da_d.ColInd, 
        v_ext_d, &beta, vdp_d));

  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_TRANSPOSE,
        v_dab_d.m, v_dab_d.n, v_dab_d.nnz, &alpha,
        v_dab_d.descr, v_dab_d.data, v_dab_d.RowPtr, v_dab_d.ColInd, 
        vdp_d, &beta, sab_d));

  checkCudaErrors(cublasSgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, norbs, &alpha, sab_d, norbs,
        X4_d, norbs, &beta, nb2v_d, norbs));
  checkCudaErrors(cublasSgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha, &X4_d[vstart*norbs], norbs, nb2v_d,
        norbs, &beta, nm2v_im_d, nvirt));


  // Normalization!!
  normalize_energy_gpu<<<dimGrid, dimBlock>>>(ksn2e_d, ksn2f_d, omega_re, omega_im, 
      nm2v_re_d, nm2v_im_d, nfermi, norbs, nvirt, vstart);

  // Going back: real part first
  checkCudaErrors(cublasSgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha, &X4_d[vstart*norbs], norbs, nm2v_re_d,
        nvirt, &beta, nb2v_d, norbs));
  cublasSgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha, nb2v_d, norbs, X4_d,
        norbs, &beta, sab_d, norbs);

  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        v_dab_d.m, v_dab_d.n, v_dab_d.nnz, &alpha,
        v_dab_d.descr, v_dab_d.data, v_dab_d.RowPtr, v_dab_d.ColInd, 
        sab_d, &beta, vdp_d));

  // starting imaginary part
  /*
    Need to switch at this part because after the call to get chi0 then nm2v_im_d = 0.0 ???
    This behavior is very odd, and I don't get it
   */
  checkCudaErrors(cublasSgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha, 
        &X4_d[vstart*norbs], norbs, nm2v_im_d, nvirt, &beta, nb2v_d, norbs));

  checkCudaErrors(cublasSgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha, 
        nb2v_d, norbs, X4_d, norbs, &beta, sab_d, norbs));

  //finishing real part
  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_TRANSPOSE,
        cc_da_d.m, cc_da_d.n, cc_da_d.nnz, &alpha,
        cc_da_d.descr, cc_da_d.data, cc_da_d.RowPtr, cc_da_d.ColInd, 
        vdp_d, &beta, chi0_d));

  checkCudaErrors(cudaMemcpyAsync( chi0_re, chi0_d, sizeof(float) * nprod, cudaMemcpyDeviceToHost, 0));


  // finishing imaginary part
  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        v_dab_d.m, v_dab_d.n, v_dab_d.nnz, &alpha,
        v_dab_d.descr, v_dab_d.data, v_dab_d.RowPtr, v_dab_d.ColInd, 
        sab_d, &beta, vdp_d));

  checkCudaErrors(cusparseScsrmv(handle_cuparse, CUSPARSE_OPERATION_TRANSPOSE,
        cc_da_d.m, cc_da_d.n, cc_da_d.nnz, &alpha,
        cc_da_d.descr, cc_da_d.data, cc_da_d.RowPtr, cc_da_d.ColInd, 
        vdp_d, &beta, chi0_d));

  // back to host
  checkCudaErrors(cudaMemcpy( chi0_im, chi0_d, sizeof(float) * nprod, cudaMemcpyDeviceToHost));
}
