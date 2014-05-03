
add_library(_vhf SHARED 
  misc.c nr_vhf_incore.c nr_vhf_direct.c optimizer.c _vhf.c)

set_target_properties(_vhf PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(_vhf cint ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

