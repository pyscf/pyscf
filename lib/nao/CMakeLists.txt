#include_directories(${CINT_INCLUDE_DIR})

add_library(nao SHARED
  cputime.c m_color.F90 m_die.F90 m_log.F90 m_precision.F90 m_siesta_hsx.F90 m_siesta_wfsx.F90
  m_define_macro.F90 m_io.F90 m_null2char.F90 m_siesta.F90 m_warn.F90)

set_target_properties(nao PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_compile_options(nao PRIVATE -pedantic -Wall -ffree-line-length-0)
target_link_libraries(nao ${BLAS_LIBRARIES})
