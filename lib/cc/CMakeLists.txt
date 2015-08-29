add_library(cc SHARED
  ccsd_pack.c ccsd_mo2ao.c)

set_target_properties(cc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(cc np_helper ao2mo ${BLAS_LIBRARIES})

