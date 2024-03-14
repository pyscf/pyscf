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

cmake_minimum_required (VERSION 3.14)
project (pyscf)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW) # option() honors normal variables that already exist

include(FetchContent)

# Out-of-source build is required for scikit-build-core when using conda-build
if(DEFINED SKBUILD)
  set(PYSCF_LIBOUTDIR_ROOT ${PROJECT_BINARY_DIR})
else()
  set(PYSCF_LIBOUTDIR_ROOT ${PROJECT_SOURCE_DIR})
endif()

set(PYSCF_THIRDPARTY_LIBOUTDIR ${PYSCF_LIBOUTDIR_ROOT}/deps/lib)



if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(PYSCF_DEBINFO "Include debugging information in PySCF libraries (not libcint, libxc, libxcfun)" TRUE)

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
set(PYSCF_CFLAGS_APPENDS "-Wall")
set(CMAKE_C_STANDARD 99)

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      string(APPEND PYSCF_CFLAGS_APPENDS " -unroll-aggressive -ipo")
      string(APPEND PYSCF_CFLAGS_APPENDS " -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -mtune=native -ftree-vectorize")
      string(APPEND PYSCF_CFLAGS_APPENDS " -march=native -mtune=native -funroll-loops -ftree-vectorize")
      string(APPEND PYSCF_CFLAGS_APPENDS " -fno-math-errno")
    endif()
  endif()
else()
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-msse3" COMPILER_SUPPORTS_SSE3)
    if(COMPILER_SUPPORTS_SSE3)
      # Avoids error "‘SIMDD’ undeclared here (not in a function)" in the qcint two-electron integral interface
      string(APPEND PYSCF_CFLAGS_APPENDS " -msse3")
    endif()
  endif()
endif()

string(APPEND CMAKE_C_FLAGS " ${PYSCF_CFLAGS_APPENDS}")
string(APPEND CMAKE_CXX_FLAGS " ${PYSCF_CFLAGS_APPENDS}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Architecture specified cmake flags.  See also the templates in
# pyscf/lib/cmake_arch_config
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

if (NOT BLAS_LIBRARIES)
  #enable_language(Fortran)
  set(BLA_VENDOR Intel10_64lp_seq)
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    unset(BLA_VENDOR)
    find_package(BLAS)
  endif()
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

option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

# if(NOT DEFINED XCFUN_MAX_ORDER)
#   set(XCFUN_MAX_ORDER 3)
# endif()
set(XCFUN_MAX_ORDER 3 CACHE STRING "XCFun maximum derivative order")

set(PYSCF_LIBXC_MAX_ORDER 3 CACHE STRING "Libxc maximum derivative order")

#find_package(PythonInterp REQUIRED)
#find_package(PythonLibs REQUIRED)
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
#  OUTPUT_VARIABLE NUMPY_INCLUDE)
#include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_SOURCE_DIR}/config.h")
# to find config.h
include_directories("${PROJECT_BINARY_DIR}")

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

option(ENABLE_FFTW "Using fftw3" OFF)
option(BUILD_FFTW "Building fftw3" OFF)


# Overwrite CMAKE_C_CREATE_SHARED_LIBRARY in Modules/CMakeCInformation.cmake
# to remove the SONAME flag in the so file. The soname information causes
# dynamic link error when importing libcint library.
set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

include(ExternalProject)
option(BUILD_LIBCINT "Using libcint for analytical gaussian integral" ON)
option(WITH_F12 "Compling F12 integrals" ON)
option(USE_QCINT "Using the qcint library (optimized for x86-64) for gaussian integral evaluation" OFF)

if(BUILD_LIBCINT)
  set(LIBCINT_GIT https://github.com/chillenb/libcint.git) # libcint is a portable, cross-platform implementation
  set(LIBCINT_VERSION f595f20dab8e96372ddce180af669a7f791849aa)
  if (USE_QCINT)
    set(LIBCINT_GIT https://github.com/sunqm/qcint.git) # qcint is an optimized implementation for x86-64 architecture
    set(LIBCINT_VERSION v6.1.2)
    if(NOT BUILD_MARCH_NATIVE)
      message(WARNING "The BUILD_MARCH_NATIVE option is not specified! qcint may not compile unless you explicitly pass compiler flags that turn on vectorization!")    
    endif()
  endif()

  FetchContent_Declare(
    cint
    GIT_REPOSITORY ${LIBCINT_GIT}
    GIT_TAG ${LIBCINT_VERSION}
  )
  set(WITH_RANGE_COULOMB 1)
  set(WITH_FORTRAN 0)
  set(WITH_CINT2_INTERFACE 0)
  set(MIN_EXPCUTOFF 20)
  set(KEEP_GOING 1)

  # The following parameters do not need to be set manually for libcint
  # thanks to the use of FetchContent.

  #         -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  #         -DCMAKE_INSTALL_LIBDIR:PATH=lib
  #         -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
  #         -DBLAS_LIBRARIES:STRING=${BLAS_LIBRARIES}
  #         -DCMAKE_C_CREATE_SHARED_LIBRARY:STRING=${C_LINK_TEMPLATE}
  #         -DBUILD_MARCH_NATIVE:STRING=${BUILD_MARCH_NATIVE}


  FetchContent_GetProperties(cint)
  if(NOT cint_POPULATED)
    FetchContent_Populate(cint)
    # EXCLUDE_FROM_ALL means not to use the install commands from libcint's CMakeLists.txt.
    add_subdirectory(${cint_SOURCE_DIR} ${cint_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  # When doing an in-source build, libcint.so should be put in the source directory.
  # The following is not needed for out-of-source build with scikit-build-core.
  set_target_properties(cint
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PYSCF_THIRDPARTY_LIBOUTDIR}
  )

  message("libcint source dir: ${cint_SOURCE_DIR}")
  message("libcint binary dir: ${cint_BINARY_DIR}")

else() # BUILD_LIBCINT
  find_package(cint REQUIRED)
endif()

option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)
option(BUILD_LIBXC "Download and build libxc library" ON)
option(BUILD_XCFUN "Download and build xcfun library" ON)

option(ENABLE_LIBXSMM "Using libxsmm" OFF)
option(BUILD_LIBXSMM "Building libxsmm" OFF)
if(APPLE)
    set(ENABLE_LIBXSMM OFF)
endif()

if(NOT DISABLE_DFT)
add_subdirectory(dft)

if(ENABLE_LIBXC)
  if(BUILD_LIBXC)
    FetchContent_Declare(
      xc
      URL https://gitlab.com/libxc/libxc/-/archive/6.1.0/libxc-6.1.0.tar.gz
    )
    set(ENABLE_FORTRAN 0)

    # Set the maximum order of derivatives for libxc.
    set(DISABLE_VXC OFF)
    set(DISABLE_FXC OFF)
    set(DISABLE_KXC OFF)
    set(DISABLE_LXC OFF)
    if(PYSCF_LIBXC_MAX_ORDER LESS 1)
      set(DISABLE_VXC ON)
    endif()
    if(PYSCF_LIBXC_MAX_ORDER LESS 2)
      set(DISABLE_FXC ON)
    endif()
    if(PYSCF_LIBXC_MAX_ORDER LESS 3)
      set(DISABLE_KXC ON)
    endif()
    if(PYSCF_LIBXC_MAX_ORDER LESS 4)
      set(DISABLE_LXC ON)
    endif()

    set(ENABLE_XHOST ${BUILD_MARCH_NATIVE})
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    FetchContent_GetProperties(xc)
    if(NOT xc_POPULATED)
      FetchContent_Populate(xc)
      # EXCLUDE_FROM_ALL means not to use the install commands from xcfun.
      add_subdirectory(${xc_SOURCE_DIR} ${xc_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    target_include_directories(xc_itrf PUBLIC ${xc_BINARY_DIR}/src ${xc_BINARY_DIR} ${xc_SOURCE_DIR}/src)

    set_target_properties(xc
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${PYSCF_THIRDPARTY_LIBOUTDIR}
    )
  
  else()
    find_package(Libxc REQUIRED)
    if(NOT Libxc_FOUND)
      message(FATAL_ERROR "You set ENABLE_LIBXC, but not BUILD_LIBXC, and libxc can't be found")
    endif()
  endif() # BUILD_LIBXC
endif() # ENABLE_LIBXC

if(ENABLE_XCFUN)
  if(BUILD_XCFUN)

  FetchContent_Declare(
    xcfun
    GIT_REPOSITORY https://github.com/dftlibs/xcfun.git
    GIT_TAG a89b783
    PATCH_COMMAND git apply --reject ${PROJECT_SOURCE_DIR}/libxcfun.patch || true
  )
  set(ENABLE_TESTALL 0)

  FetchContent_GetProperties(xcfun)
  if(NOT xcfun_POPULATED)
    FetchContent_Populate(xcfun)
    # EXCLUDE_FROM_ALL means not to use the install commands from xcfun.
    add_subdirectory(${xcfun_SOURCE_DIR} ${xcfun_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  set_target_properties(xcfun
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PYSCF_THIRDPARTY_LIBOUTDIR}
  )

  else()
    find_package(XCFun REQUIRED)
    if(NOT XCFun_FOUND)
      message(FATAL_ERROR "You set ENABLE_XCFUN, but not BUILD_XCFUN, and xcfun can't be found")
    endif()
  endif() # BUILD_XCFUN
endif() # ENABLE_XCFUN

if(ENABLE_LIBXSMM AND BUILD_LIBXSMM)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/libxsmm.h")
    ExternalProject_Add(libxsmm
      GIT_REPOSITORY https://github.com/hfp/libxsmm.git
      GIT_TAG 1.17
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PYSCF_LIBOUTDIR_ROOT}/deps
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE True
      BUILD_COMMAND make -j PREFIX=<INSTALL_DIR> CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER} STATIC=0 MALLOC=0 INTRINSICS=2 install
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS ${PYSCF_LIBOUTDIR_ROOT}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}xsmm.${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    add_library(xsmm SHARED IMPORTED GLOBAL)
    set_target_properties(xsmm PROPERTIES IMPORTED_LOCATION ${PYSCF_THIRDPARTY_LIBOUTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}xsmm${CMAKE_SHARED_LIBRARY_SUFFIX})
    add_dependencies(dft libxsmm)
  endif()
endif()

endif() # DISABLE_DFT

add_subdirectory(pbc)

if(ENABLE_FFTW AND BUILD_FFTW)
#  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/fftw3.h")
    ExternalProject_Add(libfftw3
      URL https://www.fftw.org/fftw-3.3.10.tar.gz
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PYSCF_LIBOUTDIR_ROOT}/deps
      BUILD_IN_SOURCE True
      CONFIGURE_COMMAND <SOURCE_DIR>/configure 
      "--enable-static=no"
      "--enable-shared=yes"
      "--enable-threads=yes"
      "CC=${CMAKE_C_COMPILER}"
      "CFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}"
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS}"
      "STRIP=${CMAKE_STRIP}"
      "prefix=<INSTALL_DIR>"
      BUILD_COMMAND make -j install
    )
    add_library(fftw3 SHARED IMPORTED GLOBAL)
    add_library(fftw3_threads SHARED IMPORTED GLOBAL)
    set_target_properties(fftw3 PROPERTIES IMPORTED_LOCATION ${PYSCF_THIRDPARTY_LIBOUTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3${CMAKE_SHARED_LIBRARY_SUFFIX})
    set_target_properties(fftw3_threads PROPERTIES IMPORTED_LOCATION ${PYSCF_THIRDPARTY_LIBOUTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3_threads${CMAKE_SHARED_LIBRARY_SUFFIX})
    add_dependencies(fft libfftw3)
    add_dependencies(pbc libfftw3)
#  endif()
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()



# The following install commands are used when scikit-build-core
# is building a wheel.
if(DEFINED SKBUILD)
  if(BUILD_XCFUN)
    install(TARGETS xcfun
    LIBRARY DESTINATION lib/deps/lib
    NAMELINK_SKIP
    PUBLIC_HEADER DESTINATION deps/include)
  endif()

  if(BUILD_LIBXC)
    install(TARGETS xc
    LIBRARY DESTINATION lib/deps/lib
    NAMELINK_SKIP
    PUBLIC_HEADER DESTINATION lib/deps/include)
  endif()

  if(BUILD_LIBCINT)
    set_property(TARGET cint PROPERTY VERSION)
    install(TARGETS cint
    LIBRARY DESTINATION lib/deps/lib
    NAMELINK_SKIP
    PUBLIC_HEADER DESTINATION lib/deps/include
    )
  endif()
endif()

add_subdirectory(np_helper)
add_subdirectory(gto)
add_subdirectory(vhf)
add_subdirectory(ao2mo)
add_subdirectory(mcscf)
add_subdirectory(cc)
add_subdirectory(ri)
add_subdirectory(agf2)


if(DEFINED SKBUILD)
  install(TARGETS np_helper cgto cvhf ao2mo mcscf cc ri pbc agf2 fci dft
  LIBRARY DESTINATION lib)

  if(ENABLE_XCFUN)
    install(TARGETS xcfun_itrf LIBRARY DESTINATION lib)
  endif()

  if(ENABLE_LIBXC)
    install(TARGETS xc_itrf LIBRARY DESTINATION lib)
  endif()
endif()