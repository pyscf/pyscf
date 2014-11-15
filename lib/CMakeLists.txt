cmake_minimum_required (VERSION 2.8)
project (pyscf)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2 -funroll-loops")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNDEBUG -ffast-math -fno-strict-overflow -fomit-frame-pointer")
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

enable_language(Fortran)
find_package(BLAS REQUIRED)
message("BLAS libraries: ${BLAS_LIBRARIES}")
# if unable to find mkl library, just create BLAS_LIBRARIES here, e.g.
# set(BLAS_LIBRARIES "-L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx -lm")
# or
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")

find_package(OpenMP)
#if(OPENMP_FOUND)
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#endif()

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()"
  OUTPUT_VARIABLE NUMPY_INCLUDE)
include_directories(${PROJECT_SOURCE_DIR} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

include(ExternalProject)
if (EXISTS "${PROJECT_BINARY_DIR}/deps/include/cint.h")
else()
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_BINARY_DIR}/deps
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DDISABLE_EXAMPLE=1
  )
endif()
if (EXISTS "${PROJECT_BINARY_DIR}/deps/include/xc.h")
else()
  ExternalProject_Add(libxc
    URL http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.2.0.tar.gz
    #URL http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.0.0.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_BINARY_DIR}/deps
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> 
          --enable-shared LIBS=-lm
  )
endif()
include_directories(${PROJECT_BINARY_DIR}/deps/include)
link_directories(${PROJECT_BINARY_DIR}/deps/lib)

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h")

add_subdirectory(np_helper)
add_subdirectory(vhf)
add_subdirectory(ao2mo)
add_subdirectory(mcscf)
add_subdirectory(dft)
add_subdirectory(cc)

