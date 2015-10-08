add_library(cvhf SHARED 
  moleintor.c
  int2e_sph.c nr_incore.c nr_direct.c optimizer.c nr_direct_dot.c
  time_rev.c r_direct_o1.c rkb_screen.c
  r_direct_dot.c rah_direct_dot.c rha_direct_dot.c cint3c1e.c g3c1e.c)

set_target_properties(cvhf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(cvhf np_helper cint ${BLAS_LIBRARIES})

