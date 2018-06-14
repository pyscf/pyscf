! Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

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
subroutine vrtx_cc_apair(sp12_0b,rc12,lscc_0b,ncc,dout,nout) bind(c, name='vrtx_cc_apair')
  use m_init_vrtx_cc_apair, only : ff2, evals, vertex_real2, oo2num, p2n, tmp, m2nf, vertex_cmplx2, rhotb, bessel_pp, f1f2_mom, r_scalar_pow_jp1, roverlap, S_comp, ylm
  use m_pb_libnao, only : pb
  use m_dp_aux_libnao, only : dp_a
  use m_biloc_aux_libnao, only : a
  
  use m_bilocal_vertex, only : make_bilocal_vertex_rf
  use m_init_pair_info, only : init_pair_info
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
  
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: sp12_0b(2)
  real(c_double), intent(in) :: rc12(3,2)
  integer(c_int64_t), intent(in) :: ncc
  integer(c_int64_t), intent(in) :: lscc_0b(ncc)
  integer(c_int64_t), intent(in) :: nout
  real(c_double), intent(inout) :: dout(nout)

  !! internal
  type(pair_info_t) :: bp2info
  type(book_pb_t), allocatable :: ic2book(:)
  integer(c_int64_t) :: ic, nc, npre
  real(c_double) :: ttt(9), t1, t2, tt(9), tt1(9)
  real(c_double) :: rcut, center(3)
  logical :: lready, lcheck_cpy
  real(c_double), allocatable :: fmm_mem(:), vc_ac_ac(:,:), vc_ac_ac_ref(:,:)
  integer(blas_int), allocatable :: ipiv(:)
  integer(blas_int) :: info
  integer :: pair, natoms, ibp, sp12(2), npdp
  integer, allocatable :: i2s(:)
   
  if( nout < 2 ) then; write(6,*) __FILE__, __LINE__; stop '!nout<2'; endif
  if(.not. allocated(rhotb)) then; write(6,*) __FILE__, __LINE__; stop '!rhotb'; endif
  natoms = get_natoms(pb%sv)

  dout = 0
  sp12 = int(sp12_0b)+1 ! I got zero-based indices...
  
  !write(6,*) ' lscc_0b ', lscc_0b
  !write(6,*) ' ncc ', ncc

  ibp  = 1
  pair = natoms + ibp
  lcheck_cpy = .false.
  tt = 0
  ttt = 0
  tt1 = 0

  _t1
  call init_pair_info(sp12, rc12, ncc, lscc_0b(:)+1, a%sv, bp2info)
  _t2(tt(1))
  
  call make_bilocal_vertex_rf(a, bp2info, ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, &
    vertex_cmplx2, rhotb, ttt)!, tt1)
  _t2(tt(2))  

  call init_bpair_functs_vrtx(a, bp2info, m2nf, evals, ff2, &
    vertex_real2, lready, rcut, center, dp_a, fmm_mem, pb%sp_biloc2vertex(ibp))
  _t2(tt(3))
  
  !write(6,*) __FILE__, __LINE__, ibp, sum(pb%sp_biloc2vertex(ibp)%vertex)

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
  pb%book_dp(pair)%atoms = bp2info%atoms
  pb%book_dp(pair)%cells(1:3,1:2) = bp2info%cells(1:3,1:2)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  !call constr_clist_fini(pb%sv, pb%book_dp(pair), pb%pb_p, sp2rcut, ic2book)
  !call constr_clist_sprc(pb%sv, sp12, rc12, pb%pb_p, dp_a%sp2rcut, ic2book)
  
  call conv_lscc_0b(pb%sv, lscc_0b, ncc, ic2book)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call get_i2s(pb, ic2book, i2s)
  nc = size(ic2book)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  do ic=1,nc
    ic2book(ic)%si(3) = i2s(ic)
    ic2book(ic)%fi(3) = i2s(ic+1)-1
  enddo ! ic
  
  !write(6,*) __FILE__, __LINE__
  !write(6,*) i2s
  
  npre = i2s(size(i2s))-1
   
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  _dealloc(vc_ac_ac)
  _dealloc(ipiv)
  allocate(vc_ac_ac(npre,npre))
  allocate(ipiv(npre))
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call tci_ac_ac_cpy(dp_a%hk, ic2book, ic2book, vc_ac_ac)
  _t2(tt(4))
        
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

  info = 0
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call DGETRF(npre, npre, vc_ac_ac, npre, ipiv, info )
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  if(info/=0) then
    write(0,*) __FILE__, __LINE__, info, npre
    call die('DGETRF: info/=0')
  endif
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  _t2(tt(5))
  npdp = size(pb%sp_biloc2vertex(ibp)%vertex,3)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  p2n(pair) = npdp
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call init_counting_fini(ic2book, pb%coeffs(pair))
  _dealloc(pb%coeffs(pair)%coeffs_ac_dp)
  allocate(pb%coeffs(pair)%coeffs_ac_dp(npre, npdp))
  pb%coeffs(pair)%is_reexpr = 1

  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call tci_ac_dp(dp_a%hk%ca, ic2book, fmm_mem, pb%coeffs(pair)%coeffs_ac_dp, &
    bessel_pp, f1f2_mom, roverlap, ylm, S_comp, r_scalar_pow_jp1, tmp)
  _t2(tt(6))  
  !! Reexpressing coefficients are computed and stored
  
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call DGETRS("N", npre, npdp, vc_ac_ac,npre, ipiv, pb%coeffs(pair)%coeffs_ac_dp, npre, info)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  if(info/=0) then
    write(0,*) __FILE__, __LINE__, info, npre, npdp, ibp
    call die('DGETRS: info/=0')
  endif
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  _t2(tt(7))
  !! END of Reexpressing coefficients are computed and stored
 
  !write(6,'(a,i5,3x,7f9.2,3x,8f9.2)')  __FILE__, __LINE__, tt(1:7), ttt(1:8)
  
  call apair_put(pb, pair, dout, nout)
 
end subroutine ! vrtx_cc_apair


!
! Convert list of participating centers (atoms)
!
subroutine conv_lscc_0b(sv, lscc_0b, ncc, a)
  use m_system_vars, only : system_vars_t, get_atom2coord_ptr, get_atom2sp_ptr
  use m_book_pb, only : book_pb_t
  use iso_c_binding, only : c_double, c_int64_t
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer(c_int64_t), intent(in) :: lscc_0b(:)
  integer(c_int64_t), intent(in) :: ncc
  type(book_pb_t), intent(inout), allocatable :: a(:)
  !! internal
  integer(c_int64_t) :: ic, ia
  real(8), pointer :: atom2coord(:,:)
  integer, pointer :: atom2sp(:)
  atom2coord => get_atom2coord_ptr(sv)
  atom2sp => get_atom2sp_ptr(sv)
  
  _dealloc(a)
  allocate(a(ncc))
  
  do ic=1,ncc
    ia = lscc_0b(ic)+1 ! zero-based we are accepting ....
    a(ic)%atoms      = int(ia)
    a(ic)%cells      = 0
    a(ic)%coord      = atom2coord(:,ia)
    a(ic)%spp        = atom2sp(ia)
    a(ic)%top        = 1
    a(ic)%ic         = int(ia) ! This must be a local pair index, but atom should be ok with current conventions
    a(ic)%si         = -999
    a(ic)%fi         = -999
    a(ic)%si(3)      = -998 !
    a(ic)%fi(3)      = -999 !
  enddo ! atom
  
end subroutine ! conv_lscc_0b


end module !m_vrtx_cc_apair
