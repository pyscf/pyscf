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

set(BLA_VENDOR "INTEL")
message("BLAS VENDOR libraries: ${BLA_VENDOR}")
if ("${BLA_VENDOR}" STREQUAL "INTEL")
  add_library(sparse_blas SHARED
    m_sparse_blas.F90
    )

  set_target_properties(sparse_blas PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMPILE_FLAGS ${OpenMP_C_FLAGS}
    LINK_FLAGS ${OpenMP_C_FLAGS}
    )
  target_link_libraries(sparse_blas ${FFTW_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

add_library(sparsetools SHARED
  m_sparsetools.c m_blas_wrapper.F90
  )
set_target_properties(sparsetools PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS}
  )
target_link_libraries(sparsetools ${FFTW_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
