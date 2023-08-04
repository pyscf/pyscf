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

add_library(ao2mo SHARED
  restore_eri.c nr_ao2mo.c nr_incore.c r_ao2mo.c nrr_ao2mo.c)
add_dependencies(ao2mo cvhf)

set_target_properties(ao2mo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(ao2mo cvhf cint np_helper ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
