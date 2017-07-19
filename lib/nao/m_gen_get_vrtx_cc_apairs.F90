module m_gen_get_vrtx_cc_apairs

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use iso_c_binding, only: c_double, c_int64_t
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime

  contains

!
! The subroutine is generating the dominant product vertices and conversion coefficiens for a given atom pair
!
subroutine gen_get_vrtx_cc_apairs(npairs,p2srncc,ld,dout,nout) bind(c, name='gen_get_vrtx_cc_apairs')
  use m_sv_prod_log, only : a, dp_a, pb
  use m_bilocal_vertex, only : make_bilocal_vertex_rf
  use m_init_bpair_functs_vrtx, only : init_bpair_functs_vrtx
  use m_system_vars, only : get_natoms  
  use m_prod_basis_list, only : constr_clist_fini  
  use m_book_pb, only : book_pb_t
  use m_tci_ac_dp, only : tci_ac_dp
  use m_tci_ac_ac, only : tci_ac_ac
  use m_tci_ac_ac_cpy, only : tci_ac_ac_cpy
  use m_prod_basis_type, only : get_i2s
  use m_pb_reexpr_comm, only : init_counting_fini
  use m_apair_put, only : apair_put
  use m_pair_info, only : pair_info_t
  use m_init_pair_info_array, only : init_pair_info_array
  use m_make_vrtx_cc, only : make_vrtx_cc
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ld     ! leading dimension
  integer(c_int64_t), intent(in) :: npairs ! number of pairs
  real(c_double), intent(in) :: p2srncc(ld,npairs)
  integer(c_int64_t), intent(in) :: nout
  real(c_double), intent(inout) :: dout(nout)
  !
  ! Format of p2srncc(ld,npairs)
  !  sp1,sp2,rcen1,rcen2,ncc,cc1,cc2,cc3...
  !   1   2  3..5   6..8  9   10  11  12 ...
  !

  !! internal
  type(pair_info_t), allocatable :: bp2info(:)
  integer :: natoms, nbp_node, iv
   
  if( nout < 2 ) then; write(6,*) __FILE__, __LINE__; stop '!nout<2'; endif
  if(.not. associated(pb%sv)) then; write(6,*) __FILE__, __LINE__; stop '!a%sv'; endif
  if(.not. associated(a%sv)) then; write(6,*) __FILE__, __LINE__; stop '!a%sv'; endif
  natoms = get_natoms(pb%sv)
  dout = 0
  iv = 0
  
  call init_pair_info_array(p2srncc, a%sv, bp2info)
  nbp_node = size(bp2info)
  call make_vrtx_cc(a, nbp_node, bp2info, dp_a, pb, iv)

  !put data from pb into dout...
  
end subroutine ! gen_get_vrtx_cc_apairs



end module !m_gen_get_vrtx_cc_apairs
