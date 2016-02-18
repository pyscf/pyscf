add_library(dft SHARED 
  CxLebedevGrid.c grid_basis.c nr_numint.c deriv.c)

set_target_properties(dft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(dft cvhf cgto cint xc ${BLAS_LIBRARIES})


if(NOT DISABLE_LIBXC)
add_library(xc_itrf SHARED libxc_itrf.c)
set_target_properties(dft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
target_link_libraries(xc_itrf xc)
endif()

if(NOT DISABLE_XCFUN)
add_library(xcfun_itrf SHARED xcfun_itrf.c)
set_target_properties(xcfun_itrf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
target_link_libraries(xcfun_itrf xcfun)
endif()

