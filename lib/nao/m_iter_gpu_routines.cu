#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <cblas.h>

float *X4_d, *ksn2e_d, *ksn2f_d, *nm2v_real_d, *nm2v_imag_d, *nb2v_d;
float *ab2v_d, *nb2v_tr;
int nfermi, vstart, nvirt, norbs, ksn2e_dim, ksn2f_dim;
cublasHandle_t handle_mat;

__global__ void print_array_gpu(float *arr, long dim1, long dim2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; //nocc
  int j = blockIdx.y * blockDim.y + threadIdx.y; //nvirt

  if (i < dim1)
  {
    if (j < dim2)
    {
      printf("arr[%d, %d] = %f\n", i, j, arr[j + dim2*i]);
    }
  }
}

__global__ void calc_XXVV_gpu(float *nm2v_re, float *nm2v_im, int nm2v_dim1, int nm2v_dim2,
    float *ksn2e, float *ksn2f, int nf, int vs, int kn_dim, double omega_re,
    double omega_im)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; //nocc
  int j = blockIdx.y * blockDim.y + threadIdx.y; //nvirt
  int m, index;
  float en, em, fn, fm, old_re, old_im;
  float alpha, beta, a, b;

  if (i < nf)
  {
    en = ksn2e[i];
    fn = ksn2f[i];
    if ( (j < kn_dim - i -1))
    {
      m = j + i + 1 - vs;
      if (m > 0)
      {
        em = ksn2e[i+1+j];
        fm = ksn2f[i+1+j];
        a = (omega_re - (em-en))*(omega_re - (em-en)) + omega_im*omega_im;
        b = (omega_re + (em-en))*(omega_re + (em-en)) + omega_im*omega_im;

        alpha =  (b*(omega_re - (em-en)) - a*(omega_re + (em-en)))/(a*b);
        beta = omega_im*(a-b)/(a*b);

        index = i*nm2v_dim2 + m;
        old_re = nm2v_re[index];
        old_im = nm2v_im[index];
        nm2v_re[index] = (fn - fm) * (old_re*alpha - old_im*beta);
        nm2v_im[index] = (fn - fm) * (old_re*beta + old_im*alpha);
      }
    }
  }
}

extern "C" void init_iter_gpu(float *X4, long norbs_in, float *ksn2e, long ksn2e_dim_in, float *ksn2f, long ksn2f_dim_in,
    long nfermi_in, long vstart_in)
{

  nfermi = nfermi_in;
  vstart = vstart_in;
  norbs = norbs_in;
  nvirt = norbs - vstart_in;
  ksn2e_dim = ksn2e_dim_in;
  ksn2f_dim = ksn2f_dim_in;

  // it is necessary to transpose the nb2v matrix because it is column major!
  nb2v_tr = (float *)malloc(sizeof(float)*nfermi*norbs);

  checkCudaErrors(cublasCreate(&handle_mat));

  checkCudaErrors(cudaMalloc( (void **)&X4_d, sizeof(float) * norbs*norbs));
  checkCudaErrors(cudaMalloc( (void **)&ksn2e_d, sizeof(float) * ksn2e_dim));
  checkCudaErrors(cudaMalloc( (void **)&ksn2f_d, sizeof(float) * ksn2f_dim));

  checkCudaErrors(cudaMalloc( (void **)&nm2v_real_d, sizeof(float) * nfermi*nvirt));
  checkCudaErrors(cudaMalloc( (void **)&nm2v_imag_d, sizeof(float) * nfermi*nvirt));

  checkCudaErrors(cudaMalloc( (void **)&nb2v_d, sizeof(float) * nfermi*norbs));

  checkCudaErrors(cudaMalloc( (void **)&ab2v_d, sizeof(float) * norbs*norbs));

  checkCudaErrors(cudaMemcpy( X4_d, X4, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( ksn2e_d, ksn2e, sizeof(float) * ksn2e_dim, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( ksn2f_d, ksn2f, sizeof(float) * ksn2f_dim, cudaMemcpyHostToDevice));
}

extern "C" void clean_gpu()
{
  free(nb2v_tr);
  checkCudaErrors(cudaFree(X4_d));
  checkCudaErrors(cudaFree(ksn2e_d));
  checkCudaErrors(cudaFree(ksn2f_d));

  checkCudaErrors(cudaFree(nm2v_real_d));
  checkCudaErrors(cudaFree(nm2v_imag_d));

  checkCudaErrors(cudaFree(nb2v_d));

  checkCudaErrors(cudaFree(ab2v_d));

  checkCudaErrors(cublasDestroy(handle_mat));
}

float calc_sum(float *mat, int dim1, int dim2)
{
  float sum;
  int i, j;

  sum = 0.0;
  for (i=0; i < dim1; i++)
  {
    for (j=0; j < dim2; j++)
    {
      sum += sqrt(mat[j+i*dim2]*mat[j+i*dim2]);
    }
  }
  return sum;
}

extern "C" void apply_rf0_gpu(float *nb2v_real, float *nb2v_imag, float *ab2v_real, float *ab2v_imag,
    double omega_re, double omega_im, int *blocks, int *grids)
{
  float alpha = 1.0, beta = 0.0;
  int i, j;
  dim3 dimBlock(blocks[0], blocks[1]);
  dim3 dimGrid(grids[0], grids[1]);

  /*
     Warnings!!!
        nb2v : col major
        x4: row major
        nb2v need to be in row major, transpose necessary??

        Maybe better to transpose X4, and use everything in column major??
  */

  // Real part
  // transpose nb2v !!!
  for (i = 0; i < nfermi; i++)
  {
    for (j = 0; j < norbs; j++)
    {
      nb2v_tr[ j + i*norbs] = nb2v_real[j*nfermi + i];
    }
  }
  checkCudaErrors(cudaMemcpy( nb2v_d, nb2v_tr, sizeof(float) * nfermi*norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cublasSgemm(handle_mat, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha,
      &X4_d[vstart*norbs], norbs, nb2v_d, norbs, &beta, nm2v_real_d, nvirt));

  // Imaginary part
  // transpose nb2v !!!
  for (i = 0; i < nfermi; i++)
  {
    for (j = 0; j < norbs; j++)
    {
      nb2v_tr[ j + i*norbs] = nb2v_imag[j*nfermi + i];
    }
  }
  checkCudaErrors(cudaMemcpy( nb2v_d, nb2v_tr, sizeof(float) * nfermi*norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cublasSgemm(handle_mat, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha,
      &X4_d[vstart*norbs], norbs, nb2v_d, norbs, &beta, nm2v_imag_d, nvirt));

  calc_XXVV_gpu<<< dimGrid, dimBlock >>>(nm2v_real_d, nm2v_imag_d, nfermi, nvirt,
    ksn2e_d, ksn2f_d, nfermi, vstart, ksn2e_dim, omega_re, omega_im);

  checkCudaErrors(cublasSgemm(handle_mat, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha,
      &X4_d[vstart*norbs], norbs, nm2v_real_d, nvirt, &beta, nb2v_d, norbs));
  checkCudaErrors(cublasSgemm(handle_mat, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha,
      nb2v_d, norbs, X4_d, norbs, &beta, ab2v_d, norbs));
  checkCudaErrors(cudaMemcpy( ab2v_real, ab2v_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost));

  checkCudaErrors(cublasSgemm(handle_mat, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha,
      &X4_d[vstart*norbs], norbs, nm2v_imag_d, nvirt, &beta, nb2v_d, norbs));
  checkCudaErrors(cublasSgemm(handle_mat, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha,
      nb2v_d, norbs, X4_d, norbs, &beta, ab2v_d, norbs));
  checkCudaErrors(cudaMemcpy( ab2v_imag, ab2v_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost));
}
