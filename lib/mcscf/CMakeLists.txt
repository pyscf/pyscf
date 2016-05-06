#include_directories(${CINT_INCLUDE_DIR})

add_library(fci SHARED
  fci_contract.c fci_contract_nosym.c
  fci_rdm.c fci_string.c fci_spin.c fci_4pdm.c)

set_target_properties(fci PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(fci ${BLAS_LIBRARIES})

add_library(mcscf SHARED
  nevpt_contract.c mc_eri.c)

set_target_properties(mcscf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(mcscf fci cvhf ao2mo ${BLAS_LIBRARIES})

