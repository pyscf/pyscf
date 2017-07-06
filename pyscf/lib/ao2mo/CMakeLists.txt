add_library(ao2mo SHARED
  restore_eri.c nr_ao2mo.c nr_incore.c r_ao2mo.c)

set_target_properties(ao2mo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(ao2mo cvhf cint np_helper ${BLAS_LIBRARIES})

