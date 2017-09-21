#include_directories(${CINT_INCLUDE_DIR})

add_library(fci SHARED
  fci_contract.c fci_contract_nosym.c
  fci_rdm.c fci_string.c fci_4pdm.c
  select_ci.c)

set_target_properties(fci PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(fci np_helper ${BLAS_LIBRARIES})

add_library(mcscf SHARED
  nevpt_contract.c)

set_target_properties(mcscf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(mcscf fci cvhf ao2mo ${BLAS_LIBRARIES})

