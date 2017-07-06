add_library(np_helper SHARED 
  transpose.c pack_tril.c npdot.c condense.c omp_reduce.c)

set_target_properties(np_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(np_helper ${BLAS_LIBRARIES})

