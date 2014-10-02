include_directories(${CINT_INCLUDE_DIR} ${XC_INCLUDE_DIR})

add_library(dft SHARED 
  CxLebedevGrid.c grid_basis_o1.c libxc_itrf.c nr_numint_o1.c)

set_target_properties(dft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(dft cvhf ${CINT_LIBRARIES} ${XC_LIBRARIES}
  ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

