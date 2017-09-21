add_library(cc SHARED
  ccsd_pack.c ccsd_grad.c ccsd_t.c uccsd_t.c)

set_target_properties(cc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(cc ao2mo cvhf np_helper ${BLAS_LIBRARIES})

