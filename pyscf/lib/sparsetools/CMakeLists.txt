set(BLA_VENDOR "INTEL")
message("BLAS VENDOR libraries: ${BLA_VENDOR}")
if ("${BLA_VENDOR}" STREQUAL "INTEL")
  add_library(sparse_blas SHARED
    m_sparse_blas.F90
    )

  set_target_properties(sparse_blas PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMPILE_FLAGS ${OpenMP_C_FLAGS}
    LINK_FLAGS ${OpenMP_C_FLAGS}
    )
  target_link_libraries(sparse_blas ${FFTW_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

add_library(sparsetools SHARED
  m_sparsetools.c m_blas_wrapper.F90
  )
set_target_properties(sparsetools PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS}
  )
target_link_libraries(sparsetools ${FFTW_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
