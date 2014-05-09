include_directories(${CINT_INCLUDE_DIR})

add_library(_vhf SHARED 
  misc.c int2e_sph_o4.c nr_vhf_incore.c nr_vhf_direct.c
  optimizer.c _vhf.c)

set_target_properties(_vhf PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(_vhf ${CINT_LIBRARIES} ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

