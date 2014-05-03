add_library(pycint SHARED
  cint_const_mod.F90 cint_interface.F90
  dkb_vhf_coul_O00.F90 dkb_vhf_coul_O01.F90 dkb_vhf_coul_O02.F90
  dkb_vhf.F90 hf_block_mod.F90 nr_vhf.F90 nr_vhf_grad_O00.F90
  nr_vhf_has_hs.F90 nr_vhf_hs_hs.F90 nr_vhf_igiao.F90
  nr_vhf_igiao_O00.F90 nr_vhf_O00.F90 rkb_giao_vhf.F90
  rkb_giao_vhf_O00.F90 rkb_vhf_coul.F90 rkb_vhf_coul_grad_O00.F90
  rkb_vhf_coul_O00.F90 rkb_vhf_gaunt.F90 rkb_vhf_gaunt_O00.F90
  rmb_vhf.F90 rmb_vhf_O00.F90 r_vhf_tas_ts.F90 r_vhf_ts_ts.F90
  dkb_vhf_coul.c misc.c nr_vhf_coul.c rkb_vhf.c vhf_drv.c swig_numpy.c
  pycint.c)

set_target_properties(pycint PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(pycint cint ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})

