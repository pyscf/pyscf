cmake_minimum_required (VERSION 2.8)
project (pyscf)

set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
  if(UNIX AND NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()

set(CMAKE_MACOSX_RPATH OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#enable_language(Fortran)
find_package(BLAS)
check_function_exists(ffsll HAVE_FFS)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(HAVE_OPENMP 1)
else ()
  set(OpenMP_C_FLAGS " ")
endif()

#find_package(PythonInterp REQUIRED)
#find_package(PythonLibs REQUIRED)
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
#  OUTPUT_VARIABLE NUMPY_INCLUDE)
#include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

# Architecture specified cmake flags.  See also the templates in
# pyscf/lib/cmake_arch_config
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()
# if unable to find mkl library, just create BLAS_LIBRARIES here, e.g.
# set(BLAS_LIBRARIES "-L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx -lm")
# or
# set(BLAS_LIBRARIES "                  /path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_def.so")
# set(BLAS_LIBRARIES "-Wl,-rpath=${MKLROOT}/lib/intel64/ ${BLAS_LIBRARIES}")
#
# These settings can be written in the cmake.arch.inc file. This config file
# will automatically load all settings specified in cmake.arch.inc
#


include(ExternalProject)
option(BUILD_LIBCINT "Using libcint for analytical gaussian integral" ON)
option(WITH_F12 "Compling F12 integrals" ON)
if(BUILD_LIBCINT)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/cint.h")
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG v3.0.6
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DWITH_F12=${WITH_F12} -DWITH_RANGE_COULOMB=1 -DWITH_COULOMB_ERF=1
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib -DBLA_VENDOR=${BLA_VENDOR}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  )
endif()
endif()
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib ${PROJECT_SOURCE_DIR}/deps/lib64)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

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
add_subdirectory(icmpspt)
add_subdirectory(shciscf)
add_subdirectory(ri)
#add_subdirectory(localizer)
add_subdirectory(hci)
add_subdirectory(pbc)
add_subdirectory(extras/mbd)

option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)
option(BUILD_LIBXC "Download and build libxc library" ON)
option(BUILD_XCFUN "Download and build xcfun library" ON)

if(NOT DISABLE_DFT)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xc.h" AND
    ENABLE_LIBXC AND BUILD_LIBXC)
  ExternalProject_Add(libxc
    #GIT_REPOSITORY https://gitlab.com/libxc/libxc/
    #GIT_TAG master
    URL http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-3.0.0.tar.gz
    #URL http://www.sunqm.net/pyscf/files/src/libxc-3.0.0.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ./configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
          --enable-shared --disable-static --disable-fortran LIBS=-lm
          CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}

## For git checkout
#    CONFIGURE_COMMAND autoreconf -i || autoreconf && ./configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
#          --enable-shared --disable-static --disable-fortran LIBS=-lm
#          CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
  )
endif() # ENABLE_LIBXC

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xcfun.h" AND
    ENABLE_XCFUN AND BUILD_XCFUN)
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
endif() # ENABLE_XCFUN
add_subdirectory(dft)
endif() # DISABLE_DFT


if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
