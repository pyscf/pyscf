
cuda_add_library(nao_gpu SHARED
    m_iter_gpu_routines.cu m_utils.cu
    )

set_target_properties(nao_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS}
)

target_link_libraries(nao_gpu ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${CUDA_LIBRARIES})
