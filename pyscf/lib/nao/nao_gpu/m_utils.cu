/*
  simple wrapper to utility cuda routines  
 */

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>


extern "C" int CountDevices()
{
  int num_gpus = -1;
  checkCudaErrors(cudaGetDeviceCount(&num_gpus));
  return num_gpus;
}

extern "C" void SetDevice(int gpu_id)
{
  checkCudaErrors(cudaSetDevice(gpu_id));
}

extern "C" int GetDevice()
{
  int gpu_id = -1;
  checkCudaErrors(cudaGetDevice(&gpu_id));
  return gpu_id;
}
