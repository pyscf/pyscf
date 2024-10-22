# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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

add_library(pbc SHARED ft_ao.c fill_ints.c fill_ints_sr.c optimizer.c grid_ao.c
  nr_direct.c symmetry.c inner_dot.c cint2e.c cint3c2e.c nr_ecp.c transform_mo.c
  neighbor_list.c cell.c pp.c hf_grad.c fill_ints_screened.c)
add_dependencies(pbc cgto cvhf np_helper)

set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(pbc cgto cint cvhf np_helper ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

if(ENABLE_FFTW)
add_library(fft SHARED fft.c)
set_target_properties(fft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
target_link_libraries(fft fftw3_threads fftw3 ${BLAS_LIBRARIES})
endif()
