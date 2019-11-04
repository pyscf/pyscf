/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef COMMON_NVRTC_HELPER_H_

#define COMMON_NVRTC_HELPER_H_ 1

#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define NVRTC_SAFE_CALL(Name, x)                                \
  do {                                                          \
    nvrtcResult result = x;                                     \
    if (result != NVRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " << Name << " failed with error " \
                << nvrtcGetErrorString(result);                 \
      exit(1);                                                  \
    }                                                           \
  } while (0)

void compileFileToPTX(char *filename, int argc, char **argv, char **ptxResult,
                      size_t *ptxResultSize, int requiresCGheaders) {
  std::ifstream inputFile(filename,
                          std::ios::in | std::ios::binary | std::ios::ate);

  if (!inputFile.is_open()) {
    std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    exit(1);
  }

  std::streampos pos = inputFile.tellg();
  size_t inputSize = (size_t)pos;
  char *memBlock = new char[inputSize + 1];

  inputFile.seekg(0, std::ios::beg);
  inputFile.read(memBlock, inputSize);
  inputFile.close();
  memBlock[inputSize] = '\x0';

  int numCompileOptions = 0;

  char *compileParams[1];

  if (requiresCGheaders) {
    std::string compileOptions;
    char HeaderNames[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#else
    snprintf(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#endif

    compileOptions = "--include-path=";

    std::string path = sdkFindFilePath(HeaderNames, argv[0]);
    if (!path.empty()) {
      std::size_t found = path.find(HeaderNames);
      path.erase(found);
    } else {
      printf(
          "\nCooperativeGroups headers not found, please install it in %s "
          "sample directory..\n Exiting..\n",
          argv[0]);
    }
    compileOptions += path.c_str();
    compileParams[0] = reinterpret_cast<char *>(
        malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[0], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[0], compileOptions.size(), "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }

  // compile
  nvrtcProgram prog;
  NVRTC_SAFE_CALL("nvrtcCreateProgram",
                  nvrtcCreateProgram(&prog, memBlock, filename, 0, NULL, NULL));

  nvrtcResult res = nvrtcCompileProgram(prog, numCompileOptions, compileParams);

  // dump log
  size_t logSize;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize",
                  nvrtcGetProgramLogSize(prog, &logSize));
  char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log));
  log[logSize] = '\x0';

  if (strlen(log) >= 2) {
    std::cerr << "\n compilation log ---\n";
    std::cerr << log;
    std::cerr << "\n end log ---\n";
  }

  free(log);

  NVRTC_SAFE_CALL("nvrtcCompileProgram", res);
  // fetch PTX
  size_t ptxSize;
  NVRTC_SAFE_CALL("nvrtcGetPTXSize", nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = reinterpret_cast<char *>(malloc(sizeof(char) * ptxSize));
  NVRTC_SAFE_CALL("nvrtcGetPTX", nvrtcGetPTX(prog, ptx));
  NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));
  *ptxResult = ptx;
  *ptxResultSize = ptxSize;

  if (requiresCGheaders) free(compileParams[0]);
}

CUmodule loadPTX(char *ptx, int argc, char **argv) {
  CUmodule module;
  CUcontext context;
  int major = 0, minor = 0;
  char deviceName[256];

  // Picks the best CUDA device available
  CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

  // get compute capabilities and the devicename
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));
  checkCudaErrors(cuCtxCreate(&context, 0, cuDevice));

  checkCudaErrors(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  free(ptx);

  return module;
}

#endif  // COMMON_NVRTC_HELPER_H_
