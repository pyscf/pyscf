cmake_minimum_required (VERSION 2.8)
project (pyscf)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2 -funroll-loops")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNDEBUG -ffast-math -fno-strict-overflow -fomit-frame-pointer")
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

enable_language(Fortran)
find_package(BLAS REQUIRED)
message("${BLAS_LIBRARIES}")
# if unable to find mkl library, just create BLAS_LIBRARIES here, e.g.
# set(BLAS_LIBRARIES "-L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx -lm")
# or
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")


# find libcint
FIND_PATH(CINT_INCLUDE_DIR cint.h)
FIND_LIBRARY(CINT_LIBRARIES cint)
IF (CINT_LIBRARIES AND CINT_INCLUDE_DIR)
  MESSAGE(STATUS "Found libcint: ${CINT_LIBRARIES}")
ELSE()
  MESSAGE(FATAL_ERROR "Could NOT find libcint")
ENDIF()

## find libxc
FIND_PATH(XC_INCLUDE_DIR xc.h)
FIND_LIBRARY(XC_LIBRARIES NAMES xc)
IF (XC_LIBRARIES AND XC_INCLUDE_DIR)
  MESSAGE(STATUS "Found libxc: ${XC_LIBRARIES}")
ELSE()
  MESSAGE(FATAL_ERROR "Could NOT find libxc")
ENDIF()

find_package(OpenMP)
#if(OPENMP_FOUND)
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#endif()

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()"
  OUTPUT_VARIABLE NUMPY_INCLUDE)
include_directories(${PROJECT_SOURCE_DIR} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h")

add_subdirectory(vhf)
add_subdirectory(ao2mo)
add_subdirectory(pycint)
add_subdirectory(dft)

