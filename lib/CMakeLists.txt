cmake_minimum_required (VERSION 2.8)
project (pyscf)

set(CINTDIR $ENV{HOME}/Software)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2 -funroll-loops")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNDEBUG -ffast-math -fno-strict-overflow -fomit-frame-pointer -Wall")
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(
  ${CINTDIR}/include)

enable_language(Fortran)
find_package(BLAS REQUIRED)

find_package(OpenMP)
#if(OPENMP_FOUND)
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#endif()

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()"
  OUTPUT_VARIABLE NUMPY_INCLUDE)
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h")

link_directories(
  ${CINTDIR}/lib)

add_subdirectory(vhf)
add_subdirectory(ao2mo)
add_subdirectory(pycint)

