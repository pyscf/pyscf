#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include<sys/param.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_profiler_api.h>

#include "m_iter_gpu_routines.h"

float *X4_d, *ksn2e_d, *ksn2f_d;
float *sab_d;
float *nb2v_d, *nm2v_re_d, *nm2v_im_d;
int norbs, nfermi, vstart, nprod, nvirt;
cublasHandle_t handle_cublas_real, handle_cublas_imag;
cudaStream_t stream_mem, stream_real, stream_imag;


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
  free(temp);

  return sum;
}

void print_mat_gpu(float *dev, int N1, int N2)
{
  float *temp;
  int i, j;

  temp = (float*) malloc(sizeof(float)*N1*N2);
  checkCudaErrors(cudaMemcpy( temp, dev, sizeof(float) * N1*N2, cudaMemcpyDeviceToHost));
  
  printf("mat = \n");
  for (i = 0; i< N1; i++)
  {
    for (j = 0; j< N2; j++)
    {
      printf("  %f", temp[i*N2 + j]);
    }
    printf("\n");
  }

  free(temp);
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

  if (i < nfermi)
  {
    en = ksn2e[i];
    fn = ksn2f[i];
    if ( j < norbs - vstart )
    {
      em = ksn2e[j + vstart];
      fm = ksn2f[j + vstart];

      d1p = omega_re - (em-en); d1pp = omega_im;
      d2p = omega_re + (em-en); d2pp = omega_im;
      
      alpha = d1p/(d1p*d1p + d1pp*d1pp) - d2p/(d2p*d2p + d2pp*d2pp);
      beta = -d1pp/(d1p*d1p + d1pp*d1pp) + d2pp/(d2p*d2p + d2pp*d2pp);
      old_re = nm2v_re[i*nvirt + j];
      old_im = nm2v_im[i*nvirt + j];

      nm2v_re[i*nvirt + j] = (fn - fm)*(old_re*alpha - old_im*beta);
      nm2v_im[i*nvirt + j] = (fn - fm)*(old_re*beta + old_im*alpha);
      //printf("i = %d, j = %d, m = %d, alpha = %f, beta = %f, old_re = %f, old_im = %f, nm2v_re = %f, nm2v_im = %f\n", 
      //    i, j, m, alpha, beta, old_re, old_im, nm2v_re[index], nm2v_im[index]);

      //nm2v = nm2v * (fn-fm) * ( 1.0 / (comega - (em - en)) - 1.0 /(comega + (em - en)) );
    }
  }
}

__global__ void padding_nm2v( float *nm2v_re, float *nm2v_im, int nfermi, int norbs, int nvirt, int vstart)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; //nocc
  int j = blockIdx.y * blockDim.y + threadIdx.y; //nvirt

  if (i > vstart && i < nfermi)
  {
    if ( j < norbs - vstart )
    {
      nm2v_re[i*nvirt + j] = 0.0;
      nm2v_im[i*nvirt + j] = 0.0;
    }
  }
 
}

extern "C" void init_tddft_iter_gpu(float *X4, int norbs_in, float *ksn2e,
                  float *ksn2f,  int nfermi_in, int nprod_in, int vstart_in)
{

  cudaProfilerStart();
  norbs = norbs_in;
  nfermi = nfermi_in;
  nprod = nprod_in;
  vstart = vstart_in;
  nvirt = norbs - vstart;

  // For pascal GPU, the cudaMallocManaged() could probably allow to run larger systems.
  // Need to look for more informations about this!!
  checkCudaErrors(cublasCreate(&handle_cublas_real));
  checkCudaErrors(cublasCreate(&handle_cublas_imag));

  checkCudaErrors(cudaStreamCreate(&stream_mem));
  checkCudaErrors(cudaStreamCreate(&stream_real));
  checkCudaErrors(cudaStreamCreate(&stream_imag));

  checkCudaErrors(cublasSetStream(handle_cublas_real, stream_real));
  checkCudaErrors(cublasSetStream(handle_cublas_imag, stream_imag));

  checkCudaErrors(cudaMalloc( (void **)&ksn2e_d, sizeof(float) * norbs));
  checkCudaErrors(cudaMalloc( (void **)&ksn2f_d, sizeof(float) * norbs));
  checkCudaErrors(cudaMalloc( (void **)&X4_d, sizeof(float) * norbs*norbs));
  
  checkCudaErrors(cudaMalloc( (void **)&sab_d, sizeof(float) * norbs*norbs));

  checkCudaErrors(cudaMalloc( (void **)&nb2v_d, sizeof(float) * nfermi*norbs));
  checkCudaErrors(cudaMalloc( (void **)&nm2v_re_d, sizeof(float) * nfermi*nvirt));
  checkCudaErrors(cudaMalloc( (void **)&nm2v_im_d, sizeof(float) * nfermi*nvirt));

  checkCudaErrors(cudaMemcpy( X4_d, X4, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice));
  //checkCudaErrors(cublasSetMatrix(norbs, norbs, sizeof(float), X4, norbs, X4_d, norbs));
  checkCudaErrors(cudaMemcpy( ksn2e_d, ksn2e, sizeof(float) * norbs, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy( ksn2f_d, ksn2f, sizeof(float) * norbs, cudaMemcpyHostToDevice));

}

extern "C" void free_device()
{

  checkCudaErrors(cudaFree(X4_d));
  checkCudaErrors(cudaFree(ksn2e_d));
  checkCudaErrors(cudaFree(ksn2f_d));

  checkCudaErrors(cudaFree(sab_d));
  checkCudaErrors(cudaFree(nb2v_d));
  checkCudaErrors(cudaFree(nm2v_re_d));
  checkCudaErrors(cudaFree(nm2v_im_d));

  checkCudaErrors(cublasDestroy(handle_cublas_real));
  checkCudaErrors(cublasDestroy(handle_cublas_imag));
  checkCudaErrors(cudaStreamDestroy(stream_mem));
  checkCudaErrors(cudaStreamDestroy(stream_real));
  checkCudaErrors(cudaStreamDestroy(stream_imag));
  cudaProfilerStop();
}

extern "C" void apply_rf0_device(float *sab_real, float *sab_imag, 
    double omega_re, double omega_im, int *block_size, int *grid_size)
{
  float alpha = 1.0, beta = 0.0;

  dim3 dimBlock(block_size[0], block_size[1]);
  dim3 dimGrid(grid_size[0], grid_size[1]);
  
  // real part first
  checkCudaErrors(cudaMemcpy( sab_d, sab_real, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice));

  checkCudaErrors(cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, norbs, &alpha, sab_d, norbs,
        X4_d, norbs, &beta, nb2v_d, norbs));
  checkCudaErrors(cublasSgemm(handle_cublas_real, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha, &X4_d[vstart*norbs], norbs, nb2v_d,
       norbs, &beta, nm2v_re_d, nvirt));

  // imaginary part
  checkCudaErrors(cudaMemcpy( sab_d, sab_imag, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice));
  
  checkCudaErrors(cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, norbs, &alpha, sab_d, norbs,
        X4_d, norbs, &beta, nb2v_d, norbs));
  checkCudaErrors(cublasSgemm(handle_cublas_real, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha, &X4_d[vstart*norbs], norbs, nb2v_d,
        norbs, &beta, nm2v_im_d, nvirt));

  // Normalization!!
  normalize_energy_gpu<<<dimGrid, dimBlock>>>(ksn2e_d, ksn2f_d, omega_re, omega_im, 
      nm2v_re_d, nm2v_im_d, nfermi, norbs, nvirt, vstart);

  // Going back: real part first
  checkCudaErrors(cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha, &X4_d[vstart*norbs], norbs, nm2v_re_d,
        nvirt, &beta, nb2v_d, norbs));

  cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha, nb2v_d, norbs, X4_d,
        norbs, &beta, sab_d, norbs);
  checkCudaErrors(cudaMemcpy( sab_real, sab_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost));

  //imaginary part
  checkCudaErrors(cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha, 
        &X4_d[vstart*norbs], norbs, nm2v_im_d, nvirt, &beta, nb2v_d, norbs));

  checkCudaErrors(cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha, 
        nb2v_d, norbs, X4_d, norbs, &beta, sab_d, norbs));
  checkCudaErrors(cudaMemcpy( sab_imag, sab_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost));

}

extern "C" void calc_nb2v_from_sab(int reim)
{
  float alpha = 1.0, beta = 0.0;

  if (reim == 0)
  {
    cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, norbs, &alpha, sab_d, norbs,
          X4_d, norbs, &beta, nb2v_d, norbs);
  }
  else
  {
    cublasSgemm(handle_cublas_imag, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, norbs, &alpha, sab_d, norbs,
          X4_d, norbs, &beta, nb2v_d, norbs);
  }

}

extern "C" void get_nm2v_real()
{
  float alpha = 1.0, beta = 0.0;
  
  cublasSgemm(handle_cublas_real, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha, &X4_d[vstart*norbs], norbs, nb2v_d,
       norbs, &beta, nm2v_re_d, nvirt);

}

extern "C" void get_nm2v_imag()
{
  float alpha = 1.0, beta = 0.0;
  
  cublasSgemm(handle_cublas_imag, CUBLAS_OP_T, CUBLAS_OP_N, nvirt, nfermi, norbs, &alpha, &X4_d[vstart*norbs], norbs, nb2v_d,
       norbs, &beta, nm2v_im_d, nvirt);

}

extern "C" void calc_nb2v_from_nm2v_real()
{
  float alpha = 1.0, beta = 0.0;
  
  cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha, &X4_d[vstart*norbs], norbs, nm2v_re_d,
        nvirt, &beta, nb2v_d, norbs);
}

extern "C" void calc_nb2v_from_nm2v_imag()
{
  float alpha = 1.0, beta = 0.0;

  cublasSgemm(handle_cublas_imag, CUBLAS_OP_N, CUBLAS_OP_N, norbs, nfermi, nvirt, &alpha, &X4_d[vstart*norbs], norbs, nm2v_im_d,
        nvirt, &beta, nb2v_d, norbs);

}

extern "C" void get_sab(int reim)
{
  float alpha = 1.0, beta = 0.0;

  if (reim == 0)
  {
    cublasSgemm(handle_cublas_real, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha, nb2v_d, norbs, X4_d,
          norbs, &beta, sab_d, norbs);
  }
  else
  {
    cublasSgemm(handle_cublas_imag, CUBLAS_OP_N, CUBLAS_OP_T, norbs, norbs, nfermi, &alpha, nb2v_d, norbs, X4_d,
          norbs, &beta, sab_d, norbs);
  }
}

extern "C" void div_eigenenergy_gpu(double omega_re, double omega_im, int *block_size, int *grid_size)
{

  dim3 dimBlock(block_size[0], block_size[1]);
  dim3 dimGrid(grid_size[0], grid_size[1]);

  cudaDeviceSynchronize();
  padding_nm2v<<<dimGrid, dimBlock>>>( nm2v_re_d, nm2v_im_d, nfermi, norbs, nvirt, vstart);
  normalize_energy_gpu<<<dimGrid, dimBlock>>>(ksn2e_d, ksn2f_d, omega_re, omega_im, 
      nm2v_re_d, nm2v_im_d, nfermi, norbs, nvirt, vstart);
}

extern "C" void memcpy_sab_host2device(float *sab, int Async)
{
  if (Async == 0)
  {
    cudaMemcpyAsync( sab_d, sab, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice, 0);
  }
  else if (Async == 1)
  {
    cudaMemcpyAsync( sab_d, sab, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice, stream_real);
  }
  else if (Async == 2)
  {
    cudaMemcpyAsync( sab_d, sab, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice, stream_imag);
  }

  else
  {
    cudaMemcpy( sab_d, sab, sizeof(float) * norbs*norbs, cudaMemcpyHostToDevice);
  }
}

extern "C" void memcpy_sab_device2host(float *sab, int Async)
{
  if (Async == 0)
  {
    cudaMemcpyAsync( sab, sab_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost, 0);
  }
  else if (Async == 1)
  {
    cudaMemcpyAsync( sab, sab_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost, stream_real);
  }
  else if (Async == 2)
  {
    cudaMemcpyAsync( sab, sab_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost, stream_imag);
  }
  else if (Async == 3)
  {
    cudaMemcpyAsync( sab, sab_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost, stream_mem);
  }
  else
  {
    cudaMemcpy( sab, sab_d, sizeof(float) * norbs*norbs, cudaMemcpyDeviceToHost);
  }


}
