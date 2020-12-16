# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 2.8)
project (pyscf)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} VERSION_GREATER 3.0.0)
  cmake_policy(SET CMP0042 NEW)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Architecture specified cmake flags.  See also the templates in
# pyscf/lib/cmake_arch_config
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

if (NOT BLAS_LIBRARIES)
#enable_language(Fortran)
find_package(BLAS)
check_function_exists(ffsll HAVE_FFS)
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


include(ExternalProject)
option(BUILD_LIBCINT "Using libcint for analytical gaussian integral" ON)
option(WITH_F12 "Compling F12 integrals" ON)
if(BUILD_LIBCINT)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/cint.h")
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG v4.0.7
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DWITH_F12=${WITH_F12} -DWITH_RANGE_COULOMB=1
            -DWITH_FORTRAN=0 -DWITH_CINT2_INTERFACE=0 -DMIN_EXPCUTOFF=20
            -DKEEP_GOING=1
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib -DBLA_VENDOR=${BLA_VENDOR}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DBLAS_LIBRARIES=${BLAS_LIBRARIES}
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

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  #TODO: test me set(CMAKE_MACOSX_RPATH On)
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

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
add_subdirectory(semiempirical)
add_subdirectory(agf2)

option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)
option(BUILD_LIBXC "Download and build libxc library" ON)
option(BUILD_XCFUN "Download and build xcfun library" ON)

if(NOT DISABLE_DFT)
#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xc.h" AND
#    ENABLE_LIBXC AND BUILD_LIBXC)
if(ENABLE_LIBXC AND BUILD_LIBXC)
  if(UNIX AND NOT APPLE)
    set(LIBM "LIBS=-lm")
  endif()
if(APPLE)
  ExternalProject_Add(libxc
    #GIT_REPOSITORY https://gitlab.com/libxc/libxc/
    #GIT_TAG master
    URL http://www.tddft.org/programs/libxc/down.php?file=4.3.4/libxc-4.3.4.tar.gz
    #URL https://gitlab.com/libxc/libxc/-/archive/4.3.4/libxc-4.3.4.tar.gz
    #URL http://www.sunqm.net/pyscf/files/src/libxc-4.3.4.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    BUILD_IN_SOURCE 1
    # When Xcode was installed on Mac, CMAKE_C_COMPILER may be set to the
    # Xcode C compiler, which leads to "stdio.h not found" error. See relevant
    # discussion in
    # https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
          --enable-vxc --enable-fxc --enable-kxc --enable-shared --disable-static --disable-fortran ${LIBM}

## For git checkout
#    CONFIGURE_COMMAND autoreconf -i || autoreconf && ./configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
#          --enable-shared --disable-static --disable-fortran LIBS=-lm
  )
else()
  ExternalProject_Add(libxc
    #GIT_REPOSITORY https://gitlab.com/libxc/libxc/
    #GIT_TAG master
    #URL http://www.tddft.org/programs/octopus/down.php?file=libxc/4.3.4/libxc-4.3.4.tar.gz
    #URL https://gitlab.com/libxc/libxc/-/archive/4.3.4/libxc-4.3.4.tar.gz
    URL http://www.sunqm.net/pyscf/files/src/libxc-4.3.4.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
          --enable-vxc --enable-fxc --enable-kxc --enable-shared --disable-static --disable-fortran ${LIBM}
          CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}

## For git checkout
#    CONFIGURE_COMMAND autoreconf -i || autoreconf && ./configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
#          --enable-shared --disable-static --disable-fortran LIBS=-lm
#          CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
  )
endif()
endif() # ENABLE_LIBXC

#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xcfun.h" AND
#    ENABLE_XCFUN AND BUILD_XCFUN)
if(ENABLE_XCFUN AND BUILD_XCFUN)
  ExternalProject_Add(libxcfun
    GIT_REPOSITORY https://github.com/sunqm/xcfun.git
    # copy of v2.1.0, downgrade to cmake 3.5
    GIT_TAG cmake-3.5
    # Range seperated parameters can be set in the python code. This patch to
    # RSH omega is not needed anymore.
    #PATCH_COMMAND patch -p1 < ${PROJECT_SOURCE_DIR}/libxcfun.patch
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
            -DXCFUN_MAX_ORDER=3 -DENABLE_TESTALL=0
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )
endif() # ENABLE_XCFUN
add_subdirectory(dft)
endif() # DISABLE_DFT

option(ENABLE_TBLIS "Using tblis tensor contraction library" OFF)

if(ENABLE_TBLIS)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/tblis/tblis.h")
  ExternalProject_Add(libtblis
    GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
    GIT_TAG master
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    #INSTALL_COMMAND "make install && cp <SOURCE_DIR>/src/python/tblis_einsum.py ${PROJECT_SOURCE_DIR}"
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
          CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
  )
endif()

add_library(tblis_einsum SHARED np_helper/as_einsum.cxx)
set_target_properties(tblis_einsum PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS "-std=c++11")
target_link_libraries(tblis_einsum tblis)
# add_dependencies(tblis_einsum libtblis)
endif() # ENABLE_TBLIS


if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
