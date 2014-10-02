add_library(np_helper SHARED 
  contract.c pack_tril.c)

set_target_properties(np_helper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(np_helper ${BLAS_LIBRARIES})

