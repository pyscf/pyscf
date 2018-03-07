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

add_library(dft SHARED 
  CxLebedevGrid.c grid_basis.c nr_numint.c r_numint.c)

set_target_properties(dft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(dft cvhf cgto cint ${BLAS_LIBRARIES})


if(ENABLE_LIBXC)
add_library(xc_itrf SHARED libxc_itrf.c)
set_target_properties(xc_itrf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
target_link_libraries(xc_itrf xc)
endif()

if(ENABLE_XCFUN)
add_library(xcfun_itrf SHARED xcfun_itrf.c)
set_target_properties(xcfun_itrf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
target_link_libraries(xcfun_itrf xcfun)
endif()

