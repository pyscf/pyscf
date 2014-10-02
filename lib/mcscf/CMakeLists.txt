#include_directories(${CINT_INCLUDE_DIR})

add_library(mcscf SHARED
  fci_contract.c fci_rdm.c fci_string.c mc_eri.c)

set_target_properties(mcscf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(mcscf ao2mo ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

