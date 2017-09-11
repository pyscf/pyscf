add_library(pbc SHARED ft_ao.c fill_ints.c grid_ao.c)

set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(pbc cgto cint ${BLAS_LIBRARIES})

