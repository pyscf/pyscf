add_library(cgto SHARED 
  fill_int2c.c fill_nr_3c.c fill_r_3c.c fill_int2e.c ft_ao.c ft_ao_deriv.c
  grid_ao_drv.c fastexp.c deriv1.c deriv2.c nr_ecp.c
  autocode/auto_eval1.c)

set_target_properties(cgto PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(cgto cint np_helper ${BLAS_LIBRARIES})

