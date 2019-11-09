# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(nao SHARED
  cputime.c m_color.F90 m_die.F90 m_log.F90 m_precision.F90 m_siesta_hsx.F90 m_siesta_wfsx.F90  m_define_macro.F90 m_io.F90 m_null2char.F90 m_siesta.F90 m_warn.F90 m_rsphar.F90 m_ao_eval.F90 m_interp_coeffs.F90 m_fact.F90 m_ao_hartree_lap.F90 m_thrj.F90 m_prod_talman.F90 m_numint.F90 m_thrj_nobuf.F90 m_precision.F90 m_sph_bes_trans.F90 m_pair_info.F90 m_interp.F90 m_bilocal_vertex.F90 m_system_vars.F90 m_arrays.F90 m_dft_hsx.F90 m_biloc_aux.F90 m_sp2norbs.F90 m_z2sym.F90 m_xyz.F90 m_siesta_dipo_types.F90 m_hs.F90 m_sc_dmatrix.F90 m_sets.F90 m_bulk_uc_vecs.F90 m_algebra.F90 m_sort.F90 m_matmul.F90 m_hsx.F90 m_dft_wf4.F90 m_dft_wf8.F90 m_harmonics.F90 m_interpolation.F90 m_log.F90  m_myreshape.F90 m_orb_rspace_type.F90 m_pair_info.F90 m_parallel.F90  m_precision.F90 m_prod_basis_param.F90 m_prod_basis_type.F90 m_prod_talman.F90 m_sph_bes_trans.F90 m_system_vars.F90 m_warn.F90 m_wigner_rotation.F90 m_upper.F90 m_uc_skeleton.F90 m_init_mu_sp2start_ao.F90 m_siesta_ion.F90 m_timing.F90 m_functs_l_mult_type.F90 m_functs_m_mult_type.F90 m_get_sizeof.F90 m_param_arr_type.F90 m_param.F90 m_input.F90 m_book_pb.F90 m_coeffs_type.F90 m_alloc_identity.F90 m_vertex_3cent.F90 m_prod_talman.F90 m_comp_vrtx_cc.F90 m_dp_aux.F90 m_hxc8.F90 m_block_crs8.F90 m_pb_coul_aux.F90 m_coul_param.F90 m_abramowitz.F90 m_wigner2.F90 m_prod_basis_gen.F90 m_block_split.F90 m_param_arr.F90 m_coul_comm.F90 m_pack_matrix.F90 m_csphar.F90 m_prod_basis_list.F90 m_tci_ac_dp.F90 m_tci_ac_ac.F90 m_pb_coul_12.F90 m_pb_coul_11.F90 m_tci_ac_ac_cpy.F90 m_pb_kernel_rows8.F90 m_pb_reexpr_comm.F90 m_pb_hkernel_pack8.F90 m_pb_coul_pack8.F90 m_expell_empty_pairs.F90 m_mmult_normalize.F90 m_sv_get.F90 m_init_vrtx_cc_apair.F90 m_sv_prod_log_get.F90 m_init_pair_info.F90 m_init_bpair_functs_vrtx.F90 m_init_book_dp_apair.F90 m_preinit_book_dp.F90 m_preinit_book_re.F90 m_hkernel_pb_bcrs8.F90 m_pb_ls_blocks_ovrlp_all.F90 m_pb_c2nf.F90 m_pb_coul_bcrs8.F90 m_constr_clist_sprc.F90 m_apair_put.F90 m_apair_size.F90 m_vrtx_cc_apair.F90 m_csr_bloch_mat.F90 m_dm_libnao.F90 m_dens_libnao.F90 m_spin_dens_fini8.F90 m_sparse_vec.F90 m_init_vrtx_cc_batch.F90 m_vrtx_cc_batch.F90 m_init_pair_info_array.F90 m_make_vrtx_cc.F90 m_pb_libnao.F90 m_para_libnao.F90 m_orb_rspace_aux_libnao.F90 m_biloc_aux_libnao.F90 m_dp_aux_libnao.F90 m_make_book_dp_longer.F90 m_all_legendre_poly.F90 m_spin_dens_aux.F90 m_aos_libnao.F90 m_sv_libnao_orbs.F90  m_sv_libnao_prds.F90 m_tddft_tem.F90 m_comp_spatial.F90 m_conv_dims.F90 m_conv_lims.F90 m_comp_vext_tem.F90
  )

set_target_properties(nao PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})
target_link_libraries(nao ${FFTW_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


find_package(CUDA)

if(CUDA_FOUND)
  INCLUDE(FindCUDA)

  set(CUDA_HELPER "${PROJECT_SOURCE_DIR}/nao/nao_gpu/cuda_samples/inc")
  set(CUDA_LIBRARIES "-L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcuda -lcudart -lcublas -lcudadevrt -lstdc++")
  set(CUDA_HOST_COMPILER "${HOST_COMPILER}")
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS "${CUDA_FLAGS}")

  message("CUDA FLAGS: ${CUDA_NVCC_FLAGS}")
  message("CUDA LIB: ${CUDA_LIBRARIES}")
  include_directories(${CUDA_HELPER})
  message("CUDA Helper: ${CUDA_HELPER}")

  add_subdirectory(nao_gpu)
endif()
