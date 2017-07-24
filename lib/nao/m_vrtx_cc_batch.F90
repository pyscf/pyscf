module m_vrtx_cc_batch

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
subroutine vrtx_cc_batch(npairs,p2srncc,ld,p2ndp) bind(c, name='vrtx_cc_batch')
  use m_pb_libnao, only : pb
  use m_dp_aux_libnao, only : dp_a
  use m_biloc_aux_libnao, only : a
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
  use m_make_book_dp_longer, only : make_book_dp_longer
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ld     ! leading dimension
  integer(c_int64_t), intent(in) :: npairs ! number of pairs
  real(c_double), intent(in) :: p2srncc(ld,npairs)
  integer(c_int64_t), intent(inout) :: p2ndp(npairs) ! pair -> number of dominant products in this pair
  !
  ! Format of p2srncc(ld,npairs)
  !  sp1,sp2,rcen1,rcen2,ncc,cc1,cc2,cc3...
  !   1   2  3..5   6..8  9   10  11  12 ...
  !

  !! internal
  type(pair_info_t), allocatable :: bp2info(:)
  integer :: natoms, nbp_node, top, spp, n, iv
  integer(c_int64_t) :: p
  
  if( ld < 10 ) then; write(6,*) __FILE__, __LINE__; stop '!ld<10'; endif
  if( npairs < 1 ) then; write(6,*) __FILE__, __LINE__; stop '!npairs<1'; endif
  if(.not. associated(pb%sv)) then; write(6,*) __FILE__, __LINE__; stop '!a%sv'; endif
  if(.not. associated(a%sv)) then; write(6,*) __FILE__, __LINE__; stop '!a%sv'; endif
  natoms = get_natoms(pb%sv)
  iv = 0
  
  call init_pair_info_array(p2srncc, a%sv, bp2info)
  nbp_node = size(bp2info)

  call make_book_dp_longer(nbp_node, pb)
  n = size(pb%book_dp)
  if(n<1) _die('!n')
  _dealloc(pb%coeffs)
  allocate(pb%coeffs(n))

  call make_vrtx_cc(a, nbp_node, bp2info, dp_a, pb, iv)

  if(size(pb%book_dp)/=npairs+natoms) _die(' size(pb%book_dp)/=npairs+natoms ')

  p2ndp = 0
  do p=1,npairs
    top = pb%book_dp(p+natoms)%top
    if(top/=2) _die('top/=2')
    spp = pb%book_dp(p+natoms)%spp
    if (allocated(pb%sp_biloc2vertex(spp)%vertex)) p2ndp(p) = ubound(pb%sp_biloc2vertex(spp)%vertex,3)
  enddo
  
end subroutine !vrtx_cc_batch


!
! Gets the dominant product vertices and conversion coefficiens for a set of bilocal pairs
!
subroutine get_vrtx_cc_batch(ps_0b,pf_0b,dout,nout) bind(c, name='get_vrtx_cc_batch')
  use m_pb_libnao, only : pb
  use m_system_vars, only : get_natoms

  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ps_0b ! start pair (zero based)
  integer(c_int64_t), intent(in) :: pf_0b ! finish pair  (zero based)
  integer(c_int64_t), intent(in) :: nout  ! size of buffer 
  real(c_double), intent(inout) :: dout(nout) ! data buffer

  !! internal 
  integer(c_int64_t) :: f,s,n,p,top,spp,natoms
  
  natoms = get_natoms(pb%sv)

  f = 0
  do p=ps_0b+1,pf_0b
    top = pb%book_dp(p+natoms)%top
    if(top/=2) _die('top/=2')
    spp = pb%book_dp(p+natoms)%spp
    if (.not. allocated(pb%sp_biloc2vertex(spp)%vertex)) cycle
    s = f + 1; n = size(pb%sp_biloc2vertex(spp)%vertex); f = s + n - 1;
    if(f>nout) _die('f>nout')
    call dcopy(n, pb%sp_biloc2vertex(spp)%vertex,1, dout(s),1)

    s = f + 1; n = size(pb%coeffs(p)%coeffs_ac_dp); f = s + n - 1;
    if(f>nout) _die('f>nout')
    call dcopy(n, pb%coeffs(p)%coeffs_ac_dp,1, dout(s),1)
  enddo ! p
  
end subroutine !get_vrtx_cc_batch


end module !m_vrtx_cc_batch
