add_library(ri SHARED
  nr_2c2e.c nr_3c2e.c r_df_incore.c)

set_target_properties(ri PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(ri cint cvhf np_helper ${BLAS_LIBRARIES})

