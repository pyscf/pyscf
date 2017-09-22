cmake_minimum_required (VERSION 2.8)
project (pyscf)
enable_language(Fortran)

set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
  set(CMAKE_Fortran_FLAGS "-O2")
  if(UNIX AND NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()

set(CMAKE_MACOSX_RPATH OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#find_package(BLAS REQUIRED)
#check_function_exists(ffsll HAVE_FFS)
set(BLAS_LIBRARIES "-L/opt/intel/intelpython2/pkgs/mkl-2017.0.3-intel_5/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_def -liomp5 -lpthread -lm -ldl")
message("BLAS libraries: ${BLAS_LIBRARIES}")

set(LAPACK_LIBRARIES "-L/opt/intel/intelpython2/pkgs/mkl-2017.0.3-intel_5/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_def -liomp5 -lpthread -lm -ldl")
message("LAPACK libraries: ${LAPACK_LIBRARIES}")

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
option(BUILD_LIBCINT "Using libcint for analytical gaussian integral" ON)
if(BUILD_LIBCINT)
#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/cint.h")
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG cint3
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DWITH_F12=1 -DWITH_RANGE_COULOMB=1 -DWITH_COULOMB_ERF=1
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib -DBLA_VENDOR=${BLA_VENDOR}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  )
#endif()
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
add_subdirectory(localizer)
add_subdirectory(hci)

add_subdirectory(pbc)

add_subdirectory(extras/mbd)

add_subdirectory(nao)
add_subdirectory(sparsetools)

option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)
option(BUILD_LIBXC "Download and build libxc library" ON)
option(BUILD_XCFUN "Download and build xcfun library" ON)

if(NOT DISABLE_DFT)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xc.h" AND
    ENABLE_LIBXC AND BUILD_LIBXC)
  ExternalProject_Add(libxc
    URL http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.2.2.tar.gz
    #URL https://launchpad.net/libxc/2.2/2.2.0/+download/libxc-2.2.0.tar.gz
    #URL http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.0.0.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --libdir=<INSTALL_DIR>/lib
          --enable-shared --disable-fortran LIBS=-lm
          CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
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
