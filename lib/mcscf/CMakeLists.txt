#include_directories(${CINT_INCLUDE_DIR})

add_library(_mcscf SHARED
  fci_contract.c fci_rdm.c _mcscf.c)

set_target_properties(_mcscf PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(_mcscf _ao2mo ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

