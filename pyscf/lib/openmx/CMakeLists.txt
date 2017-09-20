add_library(openmx SHARED
  openmx_unpack.c read_scfout.c
  )
  
set_target_properties(sparsetools PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS}
  )
  
target_link_libraries(openmx)
