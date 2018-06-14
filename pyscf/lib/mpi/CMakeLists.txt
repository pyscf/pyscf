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


set(LINK_LIBS ${MPI_C_LINK_FLAGS})
message("LINK_LIBS: ${LINK_LIBS}")
message("MPI_C_FLAGS: ${MPI_C_FLAGS}")
add_library(mpi_wp SHARED
  m_blacs.c  m_mpi.c  m_utilities.c
  )
set_target_properties(mpi_wp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${MPI_C_FLAGS}
  LINK_FLAGS ${MPI_C_FLAGS}
  )
target_link_libraries(mpi_wp ${PYTHON_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
