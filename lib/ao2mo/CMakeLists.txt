add_library(ao2mo SHARED
  restore_eri.c
  nr_ao2mo_o3.c nr_ao2mo_o4.c nr_outcore.c nr_incore.c)

set_target_properties(ao2mo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(ao2mo cvhf cint ${BLAS_LIBRARIES})

