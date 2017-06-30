module m_vrtx_cc_apair

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
subroutine vrtx_cc_apair(sp12_c,rc12,dout,nout) bind(c, name='vrtx_cc_apair')
  use m_sv_prod_log, only : a, dp_a, bp2info, pb, ff2, evals, vertex_real2, oo2num, m2nf, vertex_cmplx2, rhotb, bessel_pp, f1f2_mom, r_scalar_pow_jp1, roverlap, S_comp, ylm
  use m_bilocal_vertex, only : make_bilocal_vertex_rf
  use m_init_pair_info, only : init_pair_info
  use m_init_bpair_functs_vrtx, only : init_bpair_functs_vrtx
  use m_system_vars, only : get_natoms  
  use m_prod_basis_list, only : constr_clist_fini  
  use m_book_pb, only : book_pb_t, print_info_book_elem, get_n3, get_nfunct3
  use m_constr_clist_sprc, only : constr_clist_sprc
  use m_tci_ac_dp, only : tci_ac_dp
  use m_tci_ac_ac, only : tci_ac_ac
  use m_tci_ac_ac_cpy, only : tci_ac_ac_cpy
  use m_prod_basis_type, only : get_i2s
  
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: sp12_c(2)
  real(c_double), intent(in) :: rc12(3,2)
  integer(c_int64_t), intent(in) :: nout
  real(c_double), intent(inout) :: dout(nout)
  !! internal  
  type(book_pb_t), allocatable :: ic2book(:)
  integer(c_int64_t) :: ic, nc, npre
  real(c_double) :: ttt(9), time(9), t1, t2, tt(9), tloc(9), rcut, center(3)
  logical :: lready, lcheck_cpy
  real(c_double), allocatable :: fmm_mem(:), vc_ac_ac(:,:), vc_ac_ac_ref(:,:)
  integer(blas_int), allocatable :: ipiv(:)
  integer(blas_int) :: info
  integer :: pair, natoms, ibp, sp12(2)
  integer, allocatable :: i2s(:)
   
  if(nout<1) _die('nout<1')
  if(.not. allocated(bp2info)) then; write(6,*) __FILE__, __LINE__; stop '!bp2info'; endif
  natoms = get_natoms(pb%sv)
  
  dout(1:nout) = 0
  sp12 = int(sp12_c)+1 ! I got zero-based indices...

  ibp  = 1
  pair = natoms + ibp
  lcheck_cpy = .false.
  tt = 0

  call init_pair_info(sp12, rc12, a%sv, bp2info(ibp))
  
  call make_bilocal_vertex_rf(a, bp2info(ibp), &
    ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, &
    vertex_cmplx2, rhotb, ttt)

  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call init_bpair_functs_vrtx(a, bp2info(ibp), &
    m2nf, evals, ff2, vertex_real2, lready, rcut, center, dp_a, &
    fmm_mem, pb%sp_biloc2vertex(ibp))

  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  pb%book_dp(pair)%top = -1
  pb%book_dp(pair)%spp = -999
  pb%coeffs(pair)%is_reexpr = -1  
  !write(6,*) __FILE__, __LINE__, lready, ibp
  if(fmm_mem(1)<1) return ! skip empty pair
  if(lready) return ! skip a known-in-advance empty pair
  pb%book_dp(pair)%ic = pair
  pb%book_dp(pair)%top = 2
  pb%book_dp(pair)%spp = ibp

  pb%book_dp(pair)%coord = center
  pb%book_dp(pair)%atoms = bp2info(ibp)%atoms
  pb%book_dp(pair)%cells(1:3,1:2) = bp2info(ibp)%cells(1:3,1:2)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  !call constr_clist_fini(pb%sv, pb%book_dp(pair), pb%pb_p, sp2rcut, ic2book)
  call constr_clist_sprc(pb%sv, sp12, rc12, pb%pb_p, dp_a%sp2rcut, ic2book)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call get_i2s(pb, ic2book, i2s)
  nc = size(ic2book)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  do ic=1,nc
    ic2book(ic)%si(3) = i2s(ic)
    ic2book(ic)%fi(3) = i2s(ic+1)-1
  enddo ! ic
  npre = i2s(size(i2s))-1
   
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  _dealloc(vc_ac_ac)
  _dealloc(ipiv)
  allocate(vc_ac_ac(npre,npre))
  allocate(ipiv(npre))
  _t1
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call tci_ac_ac_cpy(dp_a%hk, ic2book, ic2book, vc_ac_ac)
  _t2(tt(3))
        
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  if(lcheck_cpy) then
    _dealloc(vc_ac_ac_ref)
    allocate(vc_ac_ac_ref(npre,npre))    

  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call tci_ac_ac(dp_a%hk%ca, ic2book, ic2book, vc_ac_ac_ref, &
      bessel_pp, f1f2_mom, roverlap, ylm, S_comp, r_scalar_pow_jp1)

  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    if(sum(abs(vc_ac_ac_ref-vc_ac_ac))>1d-12) then
      write(6,*) sum(abs(vc_ac_ac_ref-vc_ac_ac))
      _die('!cpy?')
    endif
  endif

  !call apair_put(bp2info, ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, dout, nout)
 
end subroutine ! vrtx_cc_apair


end module !m_vrtx_cc_apair
