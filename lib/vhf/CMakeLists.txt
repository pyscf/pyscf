add_library(cvhf SHARED 
  moleintor.c
  int2e_sph_o5.c nr_vhf_incore.c nr_incore_o0.c
  nr_vhf_direct.c nr_direct_o1.c optimizer.c
  time_rev.c r_direct_o1.c rkb_screen.c
  r_direct_dot.c rah_direct_dot.c rha_direct_dot.c)

set_target_properties(cvhf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(cvhf np_helper cint ${BLAS_LIBRARIES})

