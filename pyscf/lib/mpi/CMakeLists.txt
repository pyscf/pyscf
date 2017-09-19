
set(LINK_LIBS ${MPI_C_LINK_FLAGS})
message("LINK_LIBS: ${LINK_LIBS}")
message("MPI_C_FLAGS: ${MPI_C_FLAGS}")
add_library(libmpi_wp SHARED
  m_blacs.c  m_mpi.c  m_utilities.c
  )
set_target_properties(libmpi_wp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${MPI_C_FLAGS}
  LINK_FLAGS ${MPI_C_FLAGS}
  )
target_link_libraries(libmpi_wp ${PYTHON_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
