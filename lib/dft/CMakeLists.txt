include_directories(${CINT_INCLUDE_DIR} ${XC_INCLUDE_DIR})

add_library(_dft SHARED 
  CxLebedevGrid.c grid_basis_o1.c libxc_itrf.c nr_numint_o1.c _dft.c)

set_target_properties(_dft PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(_dft _vhf ${CINT_LIBRARIES} ${XC_LIBRARIES}
  ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

