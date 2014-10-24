add_library(np_helper SHARED 
  transpose.c pack_tril.c npdot.c)

set_target_properties(np_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(np_helper ${OpenMP_C_FLAGS} ${BLAS_LIBRARIES})

