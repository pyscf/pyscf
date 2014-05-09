include_directories(${CINT_INCLUDE_DIR})

add_library(_ao2mo SHARED
  nr_ao2mo_O3.c int2e_sph_o3.c int2e_sph_o4.c misc.c ao2mo_helper.c _ao2mo.c)


set_target_properties(_ao2mo PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(_ao2mo _vhf ${CINT_LIBRARIES} ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

