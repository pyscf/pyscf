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

module m_comp_vrtx_cc

  use m_precision, only : blas_int
#include "m_define_macro.F90" 
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use m_prod_basis_param, only : prod_basis_param_t
  use m_orb_rspace_type, only : orb_rspace_aux_t
  use m_parallel, only : para_t
  use m_pair_info, only : pair_info_t
  use m_log, only : log_memory_note
  
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime

!
! This is generation of vertex coefficients and conversion coefficients
! at the same time.
!  

  contains

!
! The program which creates bilocal_dominant
!
subroutine comp_vrtx_cc(sv, pb_p, bp2info, dp_a, pb, para, iv_in)
  use m_system_vars, only : system_vars_t
  use m_log, only : log_memory_note, log_timing_note
  use m_fact, only : init_fact
  use m_parallel, only : para_t
  use m_prod_basis_param, only : prod_basis_param_t
  use m_orb_rspace_type, only : init_orb_rspace_aux
  use m_pair_info, only : pair_info_t, distr_atom_pairs_info
  use m_biloc_aux, only : biloc_aux_t, init_biloc_aux
  use m_dp_aux, only : dp_aux_t
  use m_prod_basis_type, only : prod_basis_t
  
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  type(prod_basis_param_t), intent(in) :: pb_p
  type(pair_info_t), allocatable, intent(in) :: bp2info(:)
  type(dp_aux_t), intent(in) :: dp_a
  type(prod_basis_t), intent(inout) :: pb
  type(para_t), intent(in) :: para
  integer, intent(in) :: iv_in
  !! internal
  type(biloc_aux_t) :: a
  type(orb_rspace_aux_t) :: orb_a
  integer :: ul, nbp_node, iv
  real(8) :: t1,t2,t=0

  !! executable statements
  iv = iv_in - 1
_mem_note
  call init_fact()  !! Initializations for product reduction in Talman's way
  call init_orb_rspace_aux(sv, orb_a, ul)
  call init_biloc_aux(sv, pb_p, para, orb_a, a)
_mem_note
  
_t1
  nbp_node = size(bp2info)
  call make_bilocal_vertex(a, nbp_node, bp2info, dp_a, pb, iv)
_t2(t)
  call log_timing_note('timing-comp_vrtx_cc-', t, iv)
    
_mem_note

end subroutine ! bilocal_vertex_rd

!
!   MAKE_dominant_vertex  a la Talman 
!
subroutine make_bilocal_vertex(a, nbp, bp2info, dp_a, pb, iv_in)
  use m_system_vars, only : system_vars_t, get_nr, get_nmult_max, get_sp2rcut
  use m_log, only : log_memory_note, log_timing_note
  use m_system_vars, only : get_nr, get_jmx, get_norbs_max, get_natoms
  use m_prod_basis_param, only : get_jcutoff, prod_basis_param_t
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t
  use m_dp_aux, only : dp_aux_t
  use m_prod_basis_type, only : prod_basis_t, get_i2s, get_top_dp, get_npairs
  use m_bilocal_vertex, only : make_bilocal_vertex_rf
  use m_prod_basis_list, only : constr_clist_fini
  use m_book_pb, only : book_pb_t, print_info_book_elem, get_n3, get_nfunct3
  use m_tci_ac_dp, only : tci_ac_dp
  use m_tci_ac_ac, only : tci_ac_ac
  use m_tci_ac_ac_cpy, only : tci_ac_ac_cpy
  use m_precision, only : blas_int
  use m_pb_reexpr_comm, only : init_counting_fini
  use m_coeffs_type, only : init_unit_coeffs, alloc_coeffs
  use m_expell_empty_pairs, only : expell_empty_pairs
  use m_init_bpair_functs_vrtx, only : init_bpair_functs_vrtx
  !use m_functs_m_mult_type, only : functs_m_mult_t
    
  implicit none
  type(biloc_aux_t), intent(in) :: a
  integer, intent(in) :: nbp
  type(pair_info_t), allocatable, intent(in) :: bp2info(:)
  type(dp_aux_t), intent(in) :: dp_a
  type(prod_basis_t), intent(inout) :: pb
  integer, intent(in) :: iv_in

  !Internal
  type(system_vars_t), pointer :: sv => null()
  type(book_pb_t), allocatable :: ic2book(:)
  real(8) :: ttt(9), time(9), t1, t2, tt(9), tloc(9), tt1(9)
  integer :: nr,jcutoff,nf_max,jmx,norbs_max,ibp,nterm_max, iv, nbp1, nf
  integer :: pair, natoms, npre, ic, nc, npdp, i
  real(8), allocatable :: ff2(:,:,:,:,:), evals(:,:,:), vertex_real2(:,:,:,:,:)
  real(8), allocatable :: rhotb(:,:), sp2rcut(:)
  integer, allocatable :: oo2num(:,:), m2nf(:), i2s(:), p2n(:)
  complex(8), allocatable :: vertex_cmplx2(:,:,:,:,:)
  logical :: lready, lcheck_cpy
  real(8) :: rcut, center(3), ac_rcut
  complex(8), allocatable :: ylm(:)
  real(8), allocatable :: bessel_pp(:,:), f1f2_mom(:), vc_ac_ac(:,:), vc_ac_ac_ref(:,:)
  real(8), allocatable :: roverlap(:,:), S_comp(:),r_scalar_pow_jp1(:), tmp(:)
  real(8), allocatable :: fmm_mem(:)
  integer(blas_int), allocatable :: ipiv(:)
  integer(blas_int) :: info
!  type(functs_m_mult_t) :: frea
  
  iv = iv_in - 1
  
  natoms = get_natoms(a%sv)
  nr = get_nr(a%sv)
  jcutoff = get_jcutoff(a%pb_p)
  jmx = get_jmx(a%sv)
  norbs_max = get_norbs_max(a%sv)
  nf_max = a%nf_max
  nterm_max = a%nterm_max
  ac_rcut   = a%pb_p%ac_rcut
  sv => a%sv
  call get_sp2rcut(sv, sp2rcut)
  
  if (.not. allocated(bp2info)) _die('bp2info not allocated')

  nbp1 = size(bp2info)
  if(nbp1<1) _die('!nbp<1')
  if(nbp1<nbp) _die('!nbp1<nbp')

  time = 0
  tloc = 0
  _dealloc(pb%sp_biloc2functs)
  _dealloc(pb%sp_biloc2vertex)
!  allocate(pb%sp_biloc2functs(nbp)) ! it should remain unallocated
  allocate(pb%sp_biloc2vertex(nbp))

  allocate(p2n(nbp+natoms))
  p2n = -999
  
  do pair=1,natoms
    p2n(pair) = pb%book_dp(pair)%fi(3)-pb%book_dp(pair)%si(3)+1
  enddo

  write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__, ' Hola!!!!!'
  
  lcheck_cpy = .false.
  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP SHARED(nbp,bp2info,nr,jmx,jcutoff,nterm_max,ac_rcut,sp2rcut) &
  !$OMP SHARED(dp_a, pb,a,nf_max,norbs_max,iv,sv,natoms,time,p2n,tloc) & 
  !$OMP SHARED(lcheck_cpy) &
  !$OMP PRIVATE(ff2,vertex_cmplx2,rhotb,evals,ibp,vertex_real2,ttt) &
  !$OMP PRIVATE(center, rcut, lready, oo2num, m2nf, ic2book, pair,ic,nc) &
  !$OMP PRIVATE(info, ipiv, vc_ac_ac,npdp,npre,i2s,fmm_mem,tmp,t1,t2,tt,tt1) &
  !$OMP PRIVATE(r_scalar_pow_jp1,S_comp,ylm,roverlap,f1f2_mom,bessel_pp) &
  !$OMP PRIVATE(vc_ac_ac_ref)
  allocate(ff2(nr,0:jcutoff,nf_max,-jmx*2:jmx*2,2))
  allocate(vertex_cmplx2(-jmx*2:jmx*2,nf_max,norbs_max,norbs_max,2))
  allocate(vertex_real2(-jmx*2:jmx*2,nf_max,norbs_max,norbs_max,2))
  allocate(evals(nf_max,nf_max+1,-jmx*2:jmx*2))
  allocate(rhotb(nr,nterm_max))

  !! Comput of Coulomb matrix elements
  allocate(bessel_pp(a%nr, 0:2*a%jcutoff));
  allocate(f1f2_mom(a%nr));
  allocate(roverlap(-a%jcutoff:a%jcutoff,-a%jcutoff:a%jcutoff));
  allocate(ylm(0:(2*a%jcutoff+1)**2));
  allocate(S_comp(0:2*a%jcutoff));
  allocate(r_scalar_pow_jp1(0:2*a%jcutoff));
  allocate(tmp(-a%jcutoff:a%jcutoff))
  !! END of Comput of Coulomb matrix elements
  ttt = 0
  tt = 0
  tt1 = 0
  !$OMP DO SCHEDULE(DYNAMIC,1)
  do ibp=1, nbp
    pair = ibp + natoms
    _t1
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call make_bilocal_vertex_rf(a, bp2info(ibp), &
      ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, vertex_cmplx2, rhotb, ttt)!, tt1)
    _t2(tt(1))  
    _t1
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call init_bpair_functs_vrtx(a, bp2info(ibp), &
      m2nf, evals, ff2, vertex_real2, lready, rcut, center, dp_a, &
      fmm_mem, pb%sp_biloc2vertex(ibp))
    _t2(tt(2))  
      
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    pb%book_dp(pair)%top = -1
    pb%book_dp(pair)%spp = -999
    pb%coeffs(pair)%is_reexpr = -1  
!    write(6,*) __FILE__, __LINE__, lready, ibp, frea%nfunct
    if(fmm_mem(1)<1) cycle ! skip empty pair
    if(lready) cycle ! skip a known-in-advance empty pair
    pb%book_dp(pair)%ic = pair
    pb%book_dp(pair)%top = 2
    pb%book_dp(pair)%spp = ibp

    pb%book_dp(pair)%coord = center
    pb%book_dp(pair)%atoms = bp2info(ibp)%atoms
    pb%book_dp(pair)%cells(1:3,1:2) = bp2info(ibp)%cells(1:3,1:2)
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call constr_clist_fini(sv, pb%book_dp(pair), pb%pb_p, sp2rcut, ic2book)
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
    
    _t1    
    info = 0
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call DGETRF(npre, npre, vc_ac_ac, npre, ipiv, info )
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    if(info/=0) then
      write(0,*) __FILE__, __LINE__, info, npre, ibp, nbp
      call die('DGETRF: info/=0')
    endif
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    _t2(tt(4))
    npdp = size(pb%sp_biloc2vertex(ibp)%vertex,3)
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    p2n(pair) = npdp
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call init_counting_fini(ic2book, pb%coeffs(pair))
    allocate(pb%coeffs(pair)%coeffs_ac_dp(npre, npdp))
    pb%coeffs(pair)%is_reexpr = 1
    _t1
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call tci_ac_dp(dp_a%hk%ca, ic2book, fmm_mem, pb%coeffs(pair)%coeffs_ac_dp, &
      bessel_pp, f1f2_mom, roverlap, ylm, S_comp, r_scalar_pow_jp1, tmp)
    _t2(tt(5))  
    !! Reexpressing coefficients are computed and stored
    _t1
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call DGETRS("N", npre, npdp, vc_ac_ac,npre, ipiv, pb%coeffs(pair)%coeffs_ac_dp, npre, info)
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    if(info/=0) then
      write(0,*) __FILE__, __LINE__, info, npre, npdp, ibp
      call die('DGETRS: info/=0')
    endif
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    _t2(tt(6))
    !! END of Reexpressing coefficients are computed and stored
    
!    write(6,*) __FILE__, __LINE__, nc, npre, ic2book(:)%ic
!    write(6,'(a,i5,3x,2i8,3x,6f9.2)')  __FILE__, __LINE__, ibp, nbp, tt(1:6)
    
  enddo
  !$OMP END DO
  !$OMP CRITICAL
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  time = time + ttt
  tloc = tloc + tt
  !$OMP END CRITICAL
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  _dealloc(ff2)
  _dealloc(vertex_cmplx2)
  _dealloc(vertex_real2)
  _dealloc(rhotb)
  _dealloc(evals)
  _dealloc(i2s)
  _dealloc(r_scalar_pow_jp1)
  _dealloc(S_comp)
  _dealloc(ylm)
  _dealloc(roverlap)
  _dealloc(f1f2_mom)
  _dealloc(bessel_pp)
  _dealloc(sp2rcut)
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  !$OMP END PARALLEL

  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  call expell_empty_pairs(pb%book_dp, p2n, pb%coeffs)
  pb%irr_trans_inv_sym = 1
  pb%nfunct_irr = get_nfunct3(pb%book_dp)
  
  _dealloc(pb%book_re)
  allocate(pb%book_re(natoms))
  pb%book_re(1:natoms) = pb%book_dp(1:natoms)
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  do i = 1, natoms; pb%book_re(i)%atoms = 0; enddo
  
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
  do i=1,natoms
    call alloc_coeffs(1, pb%coeffs(i))
    nf = get_n3(pb%book_dp(i))
    pb%coeffs(i)%ind2book_dp_re(1) = i
    pb%coeffs(i)%ind2book_re(1) = i
    pb%coeffs(i)%ind2sfp_loc(1:2,1) = [1,nf]
    call init_unit_coeffs(nf, pb%coeffs(i))
  enddo
  !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
#ifdef TIMING
  write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__, ' time: ', time
  write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__, ' tloc: ', tloc
#endif

end subroutine !make_bilocal_vertex



end module !m_comp_vrtx_cc
