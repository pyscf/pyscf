#include_directories(${CINT_INCLUDE_DIR})

add_library(hci SHARED
  hci.c)

set_target_properties(hci PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(hci np_helper cvhf ao2mo ${BLAS_LIBRARIES})

