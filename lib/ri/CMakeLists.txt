add_library(ri SHARED
  nr_2c2e.c nr_df_incore_o0.c)

set_target_properties(ri PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(ri cint np_helper ${BLAS_LIBRARIES})

