cmake_minimum_required (VERSION 2.8)
project (pyscf)

set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#enable_language(Fortran)
find_package(BLAS REQUIRED)
if(BLA_VENDOR MATCHES "Intel*")
  if (${DISABLE_AVX})
    find_library(BLAS_mkl_def_LIBRARY
      NAMES mkl_def
      PATHS ENV;LD_LIBRARY_PATH)
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES};${BLAS_mkl_def_LIBRARY})
  else()
    find_library(BLAS_mkl_avx_LIBRARY
      NAMES mkl_avx
      PATHS ENV;LD_LIBRARY_PATH)
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES};${BLAS_mkl_avx_LIBRARY})
  endif()
endif()
message("BLAS libraries: ${BLAS_LIBRARIES}")
# if unable to find mkl library, just create BLAS_LIBRARIES here, e.g.
# set(BLAS_LIBRARIES "-L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx -lm")
# or
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_def.so")

# set(BLAS_LIBRARIES "-Wl,-rpath=${MKLROOT}/lib/intel64/ ${BLAS_LIBRARIES}")


find_package(OpenMP)
if(OPENMP_FOUND)
  set(HAVE_OPENMP 1)
else ()
  set(OpenMP_C_FLAGS " ")
endif()

find_package(PythonInterp REQUIRED)
#find_package(PythonLibs REQUIRED)
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
#  OUTPUT_VARIABLE NUMPY_INCLUDE)
#include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

include(ExternalProject)
# remove the following comment for if ... eles, to bypass the compilation of libcint
#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/cint.h")
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG origin/master
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DWITH_RANGE_COULOMB=1 -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib -DBLA_VENDOR=${BLA_VENDOR}
  )
#endif()
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib ${PROJECT_SOURCE_DIR}/deps/lib64)

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h")
# to find config.h
include_directories("${PROJECT_BINARY_DIR}")

add_subdirectory(np_helper)
add_subdirectory(gto)
add_subdirectory(vhf)
add_subdirectory(ao2mo)
add_subdirectory(mcscf)
add_subdirectory(cc)
add_subdirectory(ri)
add_subdirectory(localizer)

add_subdirectory(pbc)

option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)

if(NOT DISABLE_DFT)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xc.h" AND ENABLE_LIBXC)
  ExternalProject_Add(libxc
    URL http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.2.2.tar.gz
    #URL http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.0.0.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
          --enable-shared --disable-fortran LIBS=-lm
  )
endif()
if(NOT DISABLE_DFT)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xcfun.h" AND ENABLE_XCFUN)
  ExternalProject_Add(libxcfun
    GIT_REPOSITORY https://github.com/dftlibs/xcfun.git
    GIT_TAG origin/stable-1.x
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
            -DXC_MAX_ORDER=3 -DXCFUN_ENABLE_TESTS=0
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
  )
endif()
endif()
add_subdirectory(dft)
endif()
