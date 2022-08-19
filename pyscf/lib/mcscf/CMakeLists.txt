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

#include_directories(${CINT_INCLUDE_DIR})

add_library(fci SHARED
  fci_contract.c fci_contract_nosym.c
  fci_rdm.c fci_string.c fci_4pdm.c
  select_ci.c)
add_dependencies(fci np_helper)

set_target_properties(fci PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(fci np_helper ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

add_library(mcscf SHARED
  nevpt_contract.c)
add_dependencies(mcscf fci np_helper)

set_target_properties(mcscf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(mcscf fci cvhf ao2mo np_helper ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
