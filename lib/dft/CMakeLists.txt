add_library(dft SHARED 
  CxLebedevGrid.c grid_basis.c libxc_itrf.c nr_numint.c deriv.c)

set_target_properties(dft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(dft cvhf cgto cint xc ${BLAS_LIBRARIES})

