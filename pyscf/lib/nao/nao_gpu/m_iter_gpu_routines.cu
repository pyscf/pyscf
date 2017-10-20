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
float *v_ext_d;
int norbs, nfermi, vstart, nprod;
scsr_matrix cc_da_d, v_dab_d;
cusparseHandle_t handle_cuparse=0;



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
    sum_float += mat[i];
  }

  return sum_float;
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

  printf("v_dab_indptr_size = %d, norbs=%d\n", v_dab_indptr_size, norbs);

  checkCudaErrors(cusparseCreate(&handle_cuparse));

  checkCudaErrors(cudaMalloc( (void **)&X4_d, sizeof(float) * norbs*norbs));
  checkCudaErrors(cudaMalloc( (void **)&ksn2e_d, sizeof(float) * norbs));
  checkCudaErrors(cudaMalloc( (void **)&ksn2f_d, sizeof(float) * norbs));
  
  checkCudaErrors(cudaMalloc( (void **)&v_ext_d, sizeof(float) * nprod));

  checkCudaErrors(cudaMemcpy( X4_d, X4, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( ksn2e_d, ksn2e, sizeof(float) * norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( ksn2f_d, ksn2f, sizeof(float) * norbs, cudaMemcpyHostToDevice));

  // init sparse matrices on GPU
  cc_da_d = init_sparse_matrix_csr_gpu_float(cc_da_vals, cc_da_rowPtr, 
                  cc_da_col_ind, cc_da_shape[0], cc_da_shape[1], cc_da_nnz, cc_da_indptr_size);

  v_dab_d = init_sparse_matrix_csr_gpu_float(v_dab_vals, v_dab_rowPtr, 
                  v_dab_col_ind, v_dab_shape[0], v_dab_shape[1], v_dab_nnz, v_dab_indptr_size);
  printf("v_dab.shape = %d, %d\n", v_dab_d.m, v_dab_d.n);
}

extern "C" void free_device()
{

  checkCudaErrors(cudaFree(X4_d));
  checkCudaErrors(cudaFree(ksn2e_d));
  checkCudaErrors(cudaFree(ksn2f_d));

  checkCudaErrors(cudaFree(v_ext_d));

  free_csr_matrix_gpu(cc_da_d);
  free_csr_matrix_gpu(v_dab_d);

  checkCudaErrors(cusparseDestroy(handle_cuparse));
}


extern "C" void apply_rf0_device_test(float *v_ext_real, float *v_ext_imag, float *temp)
{
  float alpha = 1.0, beta = 0.0;
  float *vdp_d, *sab_d;

  printf("Hola!!\n");
  printf("v_dab.shape = %d, %d\n", v_dab_d.m, v_dab_d.n);

  // real part first
  checkCudaErrors(cudaMalloc( (void **)&vdp_d, sizeof(float) * cc_da_d.m));
  checkCudaErrors(cudaMalloc( (void **)&sab_d, sizeof(float) * v_dab_d.n));

}

extern "C" void apply_rf0_device(float *v_ext_real, float *v_ext_imag, float *temp)
{
  float alpha = 1.0, beta = 0.0;
  float *vdp_d, *sab_d;

  // real part first
  checkCudaErrors(cudaMalloc( (void **)&vdp_d, sizeof(float) * cc_da_d.m));
  checkCudaErrors(cudaMalloc( (void **)&sab_d, sizeof(float) * v_dab_d.n));


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

  checkCudaErrors(cudaMemcpy( temp, sab_d, sizeof(float) * v_dab_d.n, cudaMemcpyDeviceToHost));

  /*
  int *RowPtr, *ColInd;
  float *data;

  RowPtr = (int *) malloc(sizeof(int)*cc_da_d.RowPtrSize);
  ColInd = (int *) malloc(sizeof(int)*cc_da_d.nnz);
  data = (float *) malloc(sizeof(float)*cc_da_d.nnz);
  
  checkCudaErrors(cudaMemcpy( RowPtr, cc_da_d.RowPtr, sizeof(int) * cc_da_d.RowPtrSize, cudaMemcpyDeviceToHost));
  int sum_rowPtr = sum_int_vec(RowPtr, cc_da_d.RowPtrSize);
  
  checkCudaErrors(cudaMemcpy( ColInd, cc_da_d.ColInd, sizeof(int) * cc_da_d.nnz, cudaMemcpyDeviceToHost));
  int sum_colInd = sum_int_vec(ColInd, cc_da_d.nnz);
  
  checkCudaErrors(cudaMemcpy( data, cc_da_d.data, sizeof(float) * cc_da_d.nnz, cudaMemcpyDeviceToHost));
  float sum_data = sum_float_vec(data, cc_da_d.nnz);
*/

  /*
  printf("cc_da : gpu\n");
  printf("m = %d, n = %d, nnz = %d\n", cc_da_d.m, cc_da_d.n, cc_da_d.nnz);
  printf("sum_ind : %d, %d\n", sum_rowPtr, sum_colInd);
  printf("sum data: %f\n", sum_data);

  free(data);
  free(RowPtr);
  free(ColInd);
  */

  checkCudaErrors(cudaFree(vdp_d));
  checkCudaErrors(cudaFree(sab_d));
}
