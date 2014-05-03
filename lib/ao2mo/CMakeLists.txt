add_library(_ao2mo SHARED
  int2e_sph_o3.c int2e_sph_o4.c misc.c ci_misc_helper.c _ao2mo.c)

set_target_properties(_ao2mo PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(_ao2mo cint ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

