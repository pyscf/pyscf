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
  call init_biloc_aux(sv, pb_p, bp2info, para, orb_a, a)
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
  real(8) :: ttt(9), time(9), t1, t2, tt(9), tloc(9)
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
  !$OMP PRIVATE(info, ipiv, vc_ac_ac,npdp,npre,i2s,fmm_mem,tmp,t1,t2,tt) &
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
  !$OMP DO SCHEDULE(DYNAMIC,1)
  do ibp=1, nbp
    pair = ibp + natoms
    _t1
    !write(6,'(a,i7,a6,9g10.2)') __FILE__, __LINE__
    call make_bilocal_vertex_rf(a, bp2info(ibp), &
      ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, &
      vertex_cmplx2, rhotb, ttt)
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

!
!
!
subroutine init_bpair_functs_vrtx(a, pair_info, m2nf, evals, ff2, &
  vertex_real2, lready, rcut, center, dp_a, fmm_mem, vmm)
  use m_prod_basis_param, only : prod_basis_param_t
  use m_prod_basis_param, only : get_eigmin_bilocal, get_jcutoff
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t
  use m_dp_aux, only : dp_aux_t
  use m_prod_basis_type, only : functs_m_mult_t, vertex_3cent_t
  use m_prod_basis_type, only : prod_basis_t
  use m_pair_info, only : pair_info_t, get_rf_ls2so
  use m_functs_m_mult_type, only : gather, dealloc
  use m_mmult_normalize, only : mmult_normalize
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: pair_info
  integer, allocatable, intent(in) :: m2nf(:)
  real(8), intent(in), allocatable :: evals(:,:,:)
  real(8), intent(in), allocatable :: ff2(:,:,:,:,:)
  real(8), intent(in), allocatable :: vertex_real2(:,:,:,:,:)
  logical, intent(in) :: lready 
  real(8), intent(in) :: rcut
  real(8), intent(in) :: center(:)
  type(dp_aux_t), intent(in) :: dp_a
  real(8), intent(inout), allocatable :: fmm_mem(:)
  !type(functs_m_mult_t), intent(inout) :: fmm
  type(vertex_3cent_t), intent(inout) :: vmm
!  type(prod_basis_t), intent(inout) :: pb
 
  !! internal
  real(8) :: bilocal_eigmin, pi
  integer :: npmax, m, n, np, k, nev, jcutoff, j12_mx, no(2)
  integer :: atoms(2), nn3(3), nppp, prod, j
  type(functs_m_mult_t) :: fmm
  
  pi = 4D0*atan(1d0)

  _dealloc(fmm%ir_j_prd2v)
  _dealloc(fmm%prd2m)
  _dealloc(vmm%vertex)

  !! Initial work: allocation/preinitialisation of vertex coefficients
  no(1:2)  = dp_a%sp2norbs(pair_info%species)
  if(lready) then
    npmax = -1
    fmm%nfunct = npmax   
    call gather(fmm, fmm_mem)
    call dealloc(fmm)
    return
  endif

  nev = size(evals,2)
  jcutoff = get_jcutoff(a%pb_p)
  atoms(1:2) = pair_info%atoms(1:2)

  _dealloc(fmm%crc)
  allocate(fmm%crc(7,1))
  fmm%crc = -999

  !! Determine the number of bilocal product functions !!!!
  j12_mx = ubound(m2nf,1)
  bilocal_eigmin = get_eigmin_bilocal(a%pb_p)  
  npmax = 0;
  nppp  = 0
  do m=-j12_mx, j12_mx
    n = m2nf(m);
    if(n<1) cycle
    np=0
    do k=1,n
      if(evals(k,nev,m)<=bilocal_eigmin) cycle
      np=np+1
    enddo ! k
    npmax = max(np, npmax)
    nppp  = nppp + np
  enddo ! m
  fmm%nfunct = nppp
  if (npmax < 1) then
    call gather(fmm, fmm_mem)
    call dealloc(fmm)
    return
  endif  
 
  nn3 = [a%nr,a%pb_p%jcutoff,nppp]
  allocate(fmm%ir_j_prd2v(nn3(1),0:nn3(2),nn3(3)))
  fmm%ir_j_prd2v = 0
  
  allocate(fmm%prd2m(nppp))
  fmm%prd2m = -999

  nn3 = [no(1:2),nppp]
  allocate(vmm%vertex(nn3(1),nn3(2),nn3(3)))
  vmm%vertex = 0
  !! END of Initial work: allocation/preinitialsation of vertex coefficients 
  
  fmm%crc(1:7,1) = [center(1:3), rcut, 1D0*1, 1D0*nppp, 1D0*nppp]

  fmm%coords  = pair_info%coords ! two centers because functionals are given in a rotated frame
  fmm%cells   = pair_info%cells  ! 
  fmm%species = pair_info%species! atom species
  fmm%atoms   = pair_info%atoms  ! atoms
  fmm%rcuts   = dp_a%sp2rcut(pair_info%species)! Spatial cutoff of the orbital functions
  fmm%nr = size(fmm%ir_j_prd2v,1)
  fmm%jmax = ubound(fmm%ir_j_prd2v,2)

  vmm%centers = -999
  
  prod = 0
  do m=-j12_mx,j12_mx
    n = m2nf(m)
    if(n<1) cycle
    np=0
    do k=1,n
      if(evals(k,nev,m)<=bilocal_eigmin) cycle
      np=np+1
      prod = prod + 1
      !! Products initialization (bilocal)
      fmm%prd2m(prod) = m
      do j=abs(m),a%pb_p%jcutoff
        fmm%ir_j_prd2v(1:a%nr,j,prod) = sqrt(4*pi/(2*j+1))*ff2(1:a%nr,j,k,m,2)
      enddo

      vmm%vertex(:,:,prod) = vertex_real2(m,k,1:no(1),1:no(2),2)
    enddo ! mprod
  enddo ! m

  if(a%pb_p%normalize_dp>0) call mmult_normalize(a%rr, fmm, vmm)

  call gather(fmm, fmm_mem)
  call dealloc(fmm)
  
end subroutine ! init_bpair_functs_vrtx


end module !m_comp_vrtx_cc
