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

module m_pb_coul_aux
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_log, only : log_size_note, log_memory_note
  use m_arrays, only : d_array3_t
  use m_functs_l_mult_type, only: functs_l_mult_t
  use m_functs_m_mult_type, only: functs_m_mult_t
  use m_prod_basis_type, only : prod_basis_t
  use m_sph_bes_trans, only : Talman_plan_t

  implicit none
  private d_array3_t, functs_l_mult_t, functs_m_mult_t, die, prod_basis_t

  type pb_coul_aux_t
    type(Talman_plan_t) :: tp
    type(prod_basis_t), pointer :: pb => null()
    type(d_array3_t), allocatable :: j1_j2_to_gaunt1(:,:)
    complex(8), allocatable :: tr_c2r_diag1(:)
    complex(8), allocatable :: tr_c2r_diag2(:)
    complex(8), allocatable :: conjg_c2r_diag1(:)
    complex(8), allocatable :: conjg_c2r_diag2(:)
    real(8), allocatable    :: GGG(:,:,:)
    real(8), allocatable    :: Gamma_Gaunt(:,:,:,:) ! only effective in speeding up the multipole calc...
    real(8) :: uc_vecs(3,3) = 0
    integer :: nr      = -999
    real(8) :: dkappa  = -999
    integer :: jcutoff = -999
    integer :: ncenters = -999
    logical :: logical_overlap = .false.
    real(8), allocatable    :: pair2wigner_matrices(:,:,:,:)
    
    type(functs_l_mult_t), pointer :: sp_local2functs_mom(:) =>null()
    type(functs_m_mult_t), pointer :: sp_biloc2functs_mom(:) =>null()

    type(functs_l_mult_t), allocatable :: sp_local2moms(:)
    type(functs_m_mult_t), allocatable :: sp_biloc2moms(:)

    real(8), allocatable :: spp2rcut_type_of_mu1(:)
    integer, allocatable :: spp2nfun_type_of_mu1(:)

    real(8), allocatable :: spp2rcut_type_of_mu2(:)
    integer, allocatable :: spp2nfun_type_of_mu2(:)

    real(8), allocatable :: p2srrsfn(:,:) ! pair -> spp,Rvec,rcut,si,fi,ni
    real(8), allocatable :: pp(:)
    integer, allocatable :: mu_sp2jsfn(:,:,:) ! mult,specie -> j,si,fi,ni
    integer, allocatable :: sp2nmu(:) ! specie -> nmultipletts

    integer :: nfmx = -999
    integer :: jcl  = -999  ! maximal angular momentum of local functions (functions in l-multipletts)
    integer :: book_type = -999
    integer :: use_mult = 1
    real(8) :: scr_const = -999 ! screening constant
    real(8), allocatable :: scr_pp(:)
    integer :: bcrs_mv_block_size = -999
    integer :: nblocks = -999
    integer, allocatable :: block2start(:)
  end type ! pb_coul_aux_t

contains

!
!
!
subroutine dealloc(ca)
  use m_sph_bes_trans, only : sbt_destroy
  implicit none
  type(pb_coul_aux_t), intent(inout) :: ca

!    complex(8), allocatable :: tr_c2r_diag1(:)
!    complex(8), allocatable :: tr_c2r_diag2(:)
!    complex(8), allocatable :: conjg_c2r_diag1(:)
!    complex(8), allocatable :: conjg_c2r_diag2(:)
!    real(8), allocatable    :: GGG(:,:,:)

  ca%pb=>null()
    
  _dealloc(ca%j1_j2_to_gaunt1)
  _dealloc(ca%tr_c2r_diag1)
  _dealloc(ca%tr_c2r_diag2)
  _dealloc(ca%conjg_c2r_diag1)
  _dealloc(ca%conjg_c2r_diag2)
  _dealloc(ca%GGG)
  _dealloc(ca%Gamma_Gaunt)
  _dealloc(ca%pair2wigner_matrices)
  !_dealloc(ca%sp_local2functs_mom)
  !_dealloc(ca%sp_biloc2functs_mom)
  ca%sp_local2functs_mom=>null()
  ca%sp_biloc2functs_mom=>null()
  _dealloc(ca%sp_local2moms)
  _dealloc(ca%sp_biloc2moms)

  _dealloc(ca%spp2rcut_type_of_mu1)
  _dealloc(ca%spp2nfun_type_of_mu1)
  _dealloc(ca%spp2rcut_type_of_mu2)
  _dealloc(ca%spp2nfun_type_of_mu2)
  _dealloc(ca%pp)
  _dealloc(ca%scr_pp)
  _dealloc(ca%p2srrsfn)
  _dealloc(ca%mu_sp2jsfn)
  _dealloc(ca%block2start)
  _dealloc(ca%sp2nmu)

  call sbt_destroy(ca%tp)

  ca%uc_vecs = 0
  ca%nr = -999
  ca%dkappa = -999
  ca%jcutoff = -999
  ca%ncenters = -999
  ca%logical_overlap = .false.
  ca%nfmx = -999
  ca%jcl = -999
  ca%book_type = -999
  ca%use_mult = -999
  ca%scr_const = -999
  ca%bcrs_mv_block_size = -999
  ca%nblocks = -999
  
end subroutine ! dealloc

!
! Initialization of the aux variables
!
subroutine init_aux_pb_cp(pb, cp, aux, iv_in)
  use m_prod_basis_type, only : prod_basis_t
  use m_coul_param, only : coul_param_t, get_kernel_type, get_use_mult, get_scr_const
  implicit  none
  ! external
  type(prod_basis_t), intent(in) :: pb
  type(coul_param_t), intent(in) :: cp
  type(pb_coul_aux_t), intent(inout) :: aux
  integer, intent(in), optional :: iv_in
  !! internal
  character(100) :: ctype
  integer :: use_mult, iv
  real(8) :: scr_const
  logical :: logical_overlap
  
  if(present(iv_in)) then; iv = iv_in; else; iv = 0; endif
  
  ctype = get_kernel_type(cp)
  use_mult = get_use_mult(cp)
  scr_const = get_scr_const(cp)


  select case (ctype)
  case('HARTREE')
    logical_overlap = .false.
  case('OVERLAP')
    logical_overlap = .true.
  case default
    write(6,'(a,a)') 'ctype', ctype
    _die('unknown ctype')
  endselect

  !
  ! The prod basis functions are used in ths routine
  ! init_aux_pb_cp call from 
  !
  call init_aux_pb(pb, logical_overlap, scr_const, use_mult, aux, iv)
      
end subroutine ! init_aux_pb_cp  

!
! Initialization of the aux variables
!
subroutine init_aux_pb(pb, logical_overlap, scr_const, use_mult, aux, iv)
#define _sname 'init_aux_pb'
  use m_interpolation, only : get_dr_jt
  use m_abramowitz, only : gamma
  use m_harmonics, only : init_c2r_hc_c2r
  use m_wigner2, only : Wigner
  use m_wigner_rotation, only : simplified_wigner
  use m_book_pb, only : book_pb_t
  use m_prod_basis_gen, only : get_nbook, get_book
  use m_prod_basis_type, only : prod_basis_t, get_jcutoff, get_nr, init_functs_mom_space
  use m_prod_basis_type, only : get_nspp, get_rcut, get_nfunct, get_uc_vecs
  use m_prod_basis_type, only : get_coord_center, get_nfunct_per_book, get_rcut_per_center
  use m_prod_basis_type, only : get_spp_sp_fp, get_book_type
  use m_functs_l_mult_type, only : get_jcutoff_lmult, get_j_si_fi, get_nmult_max, get_nmult
  use m_functs_l_mult_type, only : init_moms_lmult
  use m_functs_m_mult_type, only : init_moms_mmult
  use m_prod_basis_param, only : get_jcutoff_param=>get_jcutoff
  use m_block_split, only : block_split
  use m_param_arr, only : get_i
  use m_coul_comm, only : comp_gaunt_coeff_kernel
  use m_sph_bes_trans, only : sbt_plan
  
  implicit  none
  !
  type(prod_basis_t), intent(in), target :: pb
  logical, intent(in) :: logical_overlap
  real(8), intent(in) :: scr_const
  integer, intent(in) :: use_mult
  type(pb_coul_aux_t), intent(inout) :: aux
  integer, intent(in) :: iv

  !! internal 
  type(book_pb_t) :: book
  integer :: jcl, j1, j2, j, m, itype, type_of_center, jc, nmumx, sj,fj,jjjmx
  complex(8), allocatable :: tr_c2r(:,:)
  complex(8), allocatable :: conjg_c2r(:,:)
  complex(8), allocatable :: c2r(:,:), hc_c2r(:,:) ! (-jcutoff:jcutoff,-jcutoff:jcutoff)
  real(8), allocatable :: real_wigner(:,:)
  integer :: mu, nmu, ic1, spp, nspp
  real(8) :: dR(3), Rvec(3), rcut, si, fi, ni, sp
  !! Dimensions
  real(8) :: pi

  call dealloc(aux)
  
  aux%pb => pb 
  aux%jcutoff = max(get_jcutoff(pb), get_jcutoff_param(pb%pb_p))
  aux%dkappa=get_dr_jt(pb%pp)
  aux%nr = get_nr(pb)
  aux%ncenters = get_nbook(pb)
  pi = 4.0D0*atan(1.0D0)
  !! END of Dimensions

  aux%bcrs_mv_block_size = 15
  call block_split(aux%ncenters, aux%bcrs_mv_block_size, aux%nblocks, aux%block2start)

  aux%uc_vecs = get_uc_vecs(pb)

  allocate(aux%pp(aux%nr))
  aux%pp = pb%pp

  aux%logical_overlap = logical_overlap
  if(logical_overlap) then; itype=2; else; itype=1; endif
  if(itype/=1) write(6,*) _sname//': warn: not coulomb will be comput.', itype
  if(scr_const<0) _die('scr_const<0')
  aux%scr_const = scr_const

  call init_c2r_hc_c2r(aux%jcutoff, c2r, hc_c2r);

  !! Complex to real matrices
_mem_note  
  allocate(tr_c2r(-aux%jcutoff:aux%jcutoff,-aux%jcutoff:aux%jcutoff))
  allocate(conjg_c2r(-aux%jcutoff:aux%jcutoff,-aux%jcutoff:aux%jcutoff))
_mem_note  
  tr_c2r = transpose(c2r);
  conjg_c2r = conjg(c2r);

_mem_note
  allocate(aux%tr_c2r_diag1(-aux%jcutoff:aux%jcutoff))
  allocate(aux%tr_c2r_diag2(-aux%jcutoff:aux%jcutoff))
  allocate(aux%conjg_c2r_diag1(-aux%jcutoff:aux%jcutoff))
  allocate(aux%conjg_c2r_diag2(-aux%jcutoff:aux%jcutoff))
_mem_note
  do m=-aux%jcutoff, aux%jcutoff
    aux%tr_c2r_diag1(m) = tr_c2r(m,m)
    aux%tr_c2r_diag2(-m) = tr_c2r(-m,m)
    aux%conjg_c2r_diag1(m) = conjg_c2r(m,m)
    aux%conjg_c2r_diag2(-m) = conjg_c2r(m,-m)
  end do

  aux%tr_c2r_diag1(0) = aux%tr_c2r_diag1(0)/2
  aux%tr_c2r_diag2(0) = aux%tr_c2r_diag2(0)/2

  aux%conjg_c2r_diag1(0) = aux%conjg_c2r_diag1(0)/2
  aux%conjg_c2r_diag2(0) = aux%conjg_c2r_diag2(0)/2

  if(iv>1)write(6,'(a)') _sname//': init: tr_c2r conjg_c2r';

  !! A custom Gaunt coefficient
  jc = aux%jcutoff
  call comp_gaunt_coeff_kernel(jc, jc, itype, aux%dkappa, aux%j1_j2_to_gaunt1)
  allocate(aux%GGG(0:2*jc, 0:jc, 0:jc))
  do j=0,2*jc
  do j1=0,jc
  do j2=0,jc
    aux%GGG(j,j2,j1) = 2.0D0/pi*(pi**1.5D0/8.0D0) * &
      Gamma(j+0.5D0) / Gamma(j1+1.5D0) / Gamma(j2+1.5D0) / aux%dkappa;
  enddo
  enddo
  enddo
  
  jcl = get_jcutoff_lmult(pb%sp_local2functs)
  aux%jcl = jcl
  allocate(aux%Gamma_Gaunt(-jcl:jcl,-jcl:jcl,0:jcl,0:jcl))
  do j1=0,jcl
  do j2=0,jcl
    aux%Gamma_Gaunt(-j1:j1,-j2:j2,j1,j2) = aux%GGG(j1+j2,j2,j1)*aux%j1_j2_to_gaunt1(j1,j2)%array(:,:,j1+j2)
  enddo
  enddo
  !! END of A custom Gaunt coefficient

  !! Initialize the pair2wigner_matrices 
_mem_note  

  aux%book_type = get_book_type(pb)

  _dealloc(aux%pair2wigner_matrices)
  if(aux%book_type==1) then  
    allocate(real_wigner(-jc:jc, -jc:jc))
    allocate(aux%pair2wigner_matrices(-jc:jc,-jc:jc,0:jc,aux%ncenters))
    aux%pair2wigner_matrices = 0
  endif  
_mem_note
  
  allocate(aux%p2srrsfn(8,aux%ncenters))
  do ic1=1,aux%ncenters
    book = get_book(pb, ic1)
    type_of_center = book%top
    spp = book%spp

    if(type_of_center==1) then
      dR = (/0D0, 0D0, 1D0/)
    else if (type_of_center==2) then
      dR = pb%sp_biloc2functs(spp)%coords(:,2)-pb%sp_biloc2functs(spp)%coords(:,1)
      if(all(dR==0)) then
        !! well, this could happen with separate_core 1 option...
        dR = (/0D0, 0D0, 1D0/)
      endif  
    else
      _die('wrong type of center?? ')
    endif

    if(aux%book_type==1) then
      do j=0,jc;
        call simplified_wigner( dR, j, real_wigner(-j:j,-j:j), aux%pair2wigner_matrices(-j:j,-j:j,j,ic1))
      enddo
    endif

    sp = book%spp
    Rvec = get_coord_center(aux%pb, book, 1)
    rcut = get_rcut_per_center(aux%pb, book, 1)
    si = book%si(3)
    fi = book%fi(3)
    ni = get_nfunct_per_book(aux%pb, book)

    aux%p2sRrsfn(1:8,ic1) = [sp,Rvec,rcut,si,fi,ni]
  enddo ! ic1
_mem_note  
  !! END of Initialize the pair2wigner_matrices 

!
!The prod basis functions are used here (call from init_aux_pb_cp)
!
  aux%sp_local2functs_mom => null()
  aux%sp_biloc2functs_mom => null()
  if(allocated(pb%sp_local2functs_mom))aux%sp_local2functs_mom=>pb%sp_local2functs_mom
  if(allocated(pb%sp_biloc2functs_mom))aux%sp_biloc2functs_mom=>pb%sp_biloc2functs_mom

  if(.not. associated(aux%sp_local2functs_mom)) _die('!%sp_local2functs_mom')  

  if(.not. logical_overlap) then
    _dealloc(aux%sp_local2moms)
    _dealloc(aux%sp_biloc2moms)
    select case(aux%book_type)
    case(1)
      if(allocated(pb%sp_local2functs)) &
        call init_moms_lmult(pb%sp_local2functs, pb%rr, aux%sp_local2moms)


      if(allocated(pb%sp_biloc2functs)) &
        call init_moms_mmult(pb%sp_biloc2functs, pb%rr, aux%sp_biloc2moms)
    case(2)

      if(allocated(pb%sp_local2functs)) &
        call init_moms_lmult(pb%sp_local2functs, pb%rr, aux%sp_local2moms)
    case default
      _die('!%book_type')
    end select
  endif    

_mem_note

  !! Auxiliary for local functions
  nspp = get_nspp(pb, 1)
  _dealloc(aux%spp2rcut_type_of_mu1)
  allocate(aux%spp2rcut_type_of_mu1(nspp))
  allocate(aux%spp2nfun_type_of_mu1(nspp))
  nmumx = get_nmult_max(pb%sp_local2functs)
  allocate(aux%mu_sp2jsfn(3,nmumx,nspp))
  aux%mu_sp2jsfn = 0
  allocate(aux%sp2nmu(nspp))
  aux%sp2nmu = 0
_mem_note
  do spp=1, nspp; 
    aux%spp2rcut_type_of_mu1(spp) = get_rcut(pb, 1, spp, 1)
    aux%spp2nfun_type_of_mu1(spp) = get_nfunct(pb, 1, spp)
    nmu = get_nmult(pb%sp_local2functs(spp))
    aux%sp2nmu(spp) = nmu
    do mu=1, nmu
      call get_j_si_fi(pb%sp_local2functs(spp), mu, j, sj, fj)
      aux%mu_sp2jsfn(1:3,mu,spp) = [j,sj,fj]
    enddo ! mu
  enddo ! spp
  !! END of Auxiliary for local functions

  aux%nfmx = maxval(aux%spp2nfun_type_of_mu1)

  if(aux%book_type==1) then

    nspp = get_nspp(pb, 2)
    _dealloc(aux%spp2rcut_type_of_mu2)
_mem_note  
    allocate(aux%spp2rcut_type_of_mu2(nspp))
    allocate(aux%spp2nfun_type_of_mu2(nspp))
    do spp=1, nspp; 
      aux%spp2rcut_type_of_mu2(spp) = get_rcut(pb, 2, spp, 1)
      aux%spp2nfun_type_of_mu2(spp) = get_nfunct(pb, 2, spp)
    enddo
_mem_note
    aux%nfmx = max(aux%nfmx, maxval(aux%spp2nfun_type_of_mu2))
  endif
  
  aux%use_mult = use_mult

_mem_note
  allocate(aux%scr_pp(aux%nr))
  aux%scr_pp = (aux%pp**2)/( aux%pp**2 + aux%scr_const**2 )
_mem_note
  
  jjjmx = max(aux%jcutoff, aux%jcl)
!  write(6,*) __FILE__, __LINE__, aux%jcutoff, aux%jcl
  
  call sbt_plan(aux%tp, aux%nr, jjjmx, aux%pb%rr, aux%pb%pp, .true.)
#undef _sname
end subroutine !init_aux_pb

!
!
!
subroutine alloc_init_ls_blocks(pb, ls_blocks, exclude_bi_bi)
  use m_prod_basis_type, only : prod_basis_t
  use m_prod_basis_gen, only : get_nbook, get_book
  use m_book_pb, only : book_pb_t
  implicit none 
  !! external
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), allocatable, intent(inout) :: ls_blocks(:,:)
  logical, intent(in) :: exclude_bi_bi
  !! internal
  integer :: ip1, ip2, natom_pairs
  integer(8) :: iblock, step, n
  type(book_pb_t) :: book1, book2

  _dealloc(ls_blocks)
  natom_pairs = get_nbook(pb)

  do step=1,2
    iblock = 0
    do ip2=1, natom_pairs
      do ip1=1,ip2
        book1 = get_book(pb, ip1)
        book2 = get_book(pb, ip2)
        if(exclude_bi_bi) then
          if(book1%top==2 .and. book2%top==2) cycle
        endif
        iblock = iblock + 1
        if(step==2) then
          ls_blocks(1,iblock) = book1
          ls_blocks(2,iblock) = book2
        endif
      enddo ! ip1
    enddo ! ip2
    if(step==1) then
      n = iblock
      allocate( ls_blocks(2,n) )
    endif
  enddo ! step  
  
end subroutine !alloc_init_ls_blocks

!
!
!
subroutine comp_kernel_block_22_ptr(aux, spp1, spp2, &
  is_overlap, wigner_matrices1, wigner_matrices2, &
  array, f1_mom, f1f2_mom, S, real_overlap, tmp, &
  bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
  
  use m_functs_m_mult_type, only : get_nfunct_mmult, get_m, get_ff_mmult
  use m_coul_comm, only : comp_overlap_v6_arg
  
  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1(:), spp2(:)
  logical, intent(in) :: is_overlap
  real(8), intent(in) :: wigner_matrices1(:,:,:)
  real(8), intent(in) :: wigner_matrices2(:,:,:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1_mom(:), f1f2_mom(:), S(:), real_overlap(:,:)
  real(8), intent(inout), allocatable :: tmp(:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  !! internal
  integer :: i1,i2,J1,J2,M1,M2,j,mm1
  real(8) :: sigma_element, f1j1, f1j1f2j2, rot_real_overlap
  integer :: nwigner1(3), nwigner2(3)
  !write(6,*) 'comp_kernel_block_22: enter...'
  if(size(spp1)/=3) _die('size spp1/=3')
  if(size(spp2)/=3) _die('size spp2/=3')
  
  nwigner1 = shape(wigner_matrices1)/2 + 1
  nwigner2 = shape(wigner_matrices2)/2 + 1

  do i2=spp2(2), spp2(3)
    M2=get_m(aux%sp_biloc2functs_mom(spp2(1)), i2)
    do i1=spp1(2), spp1(3)
      M1=get_m(aux%sp_biloc2functs_mom(spp1(1)), i1)

      sigma_element = 0;
      do j1=abs(M1), aux%jcutoff
        f1j1 = 0
        if(is_overlap) then
          call get_ff_mmult(aux%sp_biloc2functs_mom(spp1(1)), j1, i1, f1_mom)
        else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
          f1j1 = aux%sp_biloc2moms(spp1(1))%ir_j_prd2v(1,j1,i1);
        end if

        do j2=abs(M2), aux%jcutoff
          S = 0
          if(is_overlap) then !! overlapping or not overlapping orbitals
            f1f2_mom = aux%sp_biloc2functs_mom(spp2(1))%ir_j_prd2v(:,j2,i2)*f1_mom

            do j=abs(j1-j2),j1+j2,2; 
              S(j)=sum(f1f2_mom*bessel_pp(:,j));
              S(j)=S(j)+f1f2_mom(1)*bessel_pp(1,j)/aux%dkappa
            enddo;
                   
          else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
            f1j1f2j2 = f1j1 * aux%sp_biloc2moms(spp2(1))%ir_j_prd2v(1,j2,i2)
            do j=abs(j1-j2),j1+j2,2;
              if(j/=(j1+j2)) then
                S(j)=0
                cycle
              endif
              S(j)= aux%GGG(j,j2,j1)*f1j1f2j2*r_scalar_pow_jp1(j);
            enddo
          else
            S = 0;
          endif !! endif overlapping or not overlapping orbitals

          call comp_overlap_v6_arg(aux%jcutoff, j1,j2, aux%j1_j2_to_gaunt1, &
            aux%tr_c2r_diag1, aux%tr_c2r_diag2, aux%conjg_c2r_diag1, aux%conjg_c2r_diag2, & 
            S, ylm_thrpriv, real_overlap)

          !! Rotate the spherical harmonics in real_overlap
          do mm1=-j1,j1
            tmp(mm1)=sum(real_overlap(mm1,-j2:j2)*wigner_matrices2(M2+nwigner2(1),&
              -j2+nwigner2(2):j2+nwigner2(2),j2+1))
          enddo
          rot_real_overlap = sum(tmp(-j1:j1)*wigner_matrices1(M1+nwigner1(1),&
            -j1+nwigner1(2):j1+nwigner1(2),j1+1))
          !! END of Rotate the spherical harmonics in real_overlap 

          !! rotated_real_overlap is done
          sigma_element = sigma_element + rot_real_overlap;

        enddo ! l2
      enddo ! l1
        
      array(i1,i2) = sigma_element
    enddo ! i1
  enddo ! i2
  
end subroutine ! comp_kernel_block_22


!
!
!
subroutine comp_kernel_block_22(aux, spp1, spp2, &
  is_overlap, wigner_matrices1, wigner_matrices2, &
  array, f1_mom, f1f2_mom, S, real_overlap, tmp, &
  bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
  
  use m_functs_m_mult_type, only : get_nfunct_mmult, get_m, get_ff_mmult
  use m_coul_comm, only : comp_overlap_v6_arg
  
  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1(:), spp2(:)
  logical, intent(in) :: is_overlap
  real(8), intent(in), allocatable :: wigner_matrices1(:,:,:)
  real(8), intent(in), allocatable :: wigner_matrices2(:,:,:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1_mom(:), f1f2_mom(:), S(:), real_overlap(:,:)
  real(8), intent(inout), allocatable :: tmp(:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  !! internal
  integer :: i1,i2,J1,J2,M1,M2,j,mm1
  real(8) :: sigma_element, f1j1, f1j1f2j2, rot_real_overlap
  
  !write(6,*) 'comp_kernel_block_22: enter...'
  if(size(spp1)/=3) _die('size spp1/=3')
  if(size(spp2)/=3) _die('size spp2/=3')

  do i2=spp2(2), spp2(3)
    M2=get_m(aux%sp_biloc2functs_mom(spp2(1)), i2)
    do i1=spp1(2), spp1(3)
      M1=get_m(aux%sp_biloc2functs_mom(spp1(1)), i1)

      sigma_element = 0;
      do j1=abs(M1), aux%jcutoff
        f1j1 = 0
        if(is_overlap) then
          call get_ff_mmult(aux%sp_biloc2functs_mom(spp1(1)), j1, i1, f1_mom)
        else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
          f1j1 = aux%sp_biloc2moms(spp1(1))%ir_j_prd2v(1,j1,i1);
        end if

        do j2=abs(M2), aux%jcutoff
          S = 0
          if(is_overlap) then !! overlapping or not overlapping orbitals
            f1f2_mom = aux%sp_biloc2functs_mom(spp2(1))%ir_j_prd2v(:,j2,i2)*f1_mom

            do j=abs(j1-j2),j1+j2,2; 
              S(j)=sum(f1f2_mom*bessel_pp(:,j));
              S(j)=S(j)+f1f2_mom(1)*bessel_pp(1,j)/aux%dkappa
            enddo;
                   
          else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
            f1j1f2j2 = f1j1 * aux%sp_biloc2moms(spp2(1))%ir_j_prd2v(1,j2,i2)
            do j=abs(j1-j2),j1+j2,2;
              if(j/=(j1+j2)) then
                S(j)=0
                cycle
              endif
              S(j)= aux%GGG(j,j2,j1)*f1j1f2j2*r_scalar_pow_jp1(j);
            enddo
          else
            S = 0;
          endif !! endif overlapping or not overlapping orbitals

          call comp_overlap_v6_arg(aux%jcutoff, j1,j2, aux%j1_j2_to_gaunt1, &
            aux%tr_c2r_diag1, aux%tr_c2r_diag2, aux%conjg_c2r_diag1, aux%conjg_c2r_diag2, & 
            S, ylm_thrpriv, real_overlap)

          !! Rotate the spherical harmonics in real_overlap
          do mm1=-j1,j1
            tmp(mm1)=sum(real_overlap(mm1,-j2:j2)*wigner_matrices2(M2,-j2:j2,j2))
          enddo
          rot_real_overlap = sum(tmp(-j1:j1)*wigner_matrices1(M1,-j1:j1,j1))
          !! END of Rotate the spherical harmonics in real_overlap 

          !! rotated_real_overlap is done
          sigma_element = sigma_element + rot_real_overlap;

        enddo ! l2
      enddo ! l1
        
      array(i1,i2) = sigma_element
    enddo ! i1
  enddo ! i2
  
end subroutine ! comp_kernel_block_22

!
!
!
subroutine comp_kernel_block_21_ptr(aux, spp1, spp2, &
  is_overlap, wigner_matrices1, array, f1f2_mom, S, real_overlap, tmp, &
  bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
 
  use m_functs_m_mult_type, only : get_nfunct_mmult,get_m 
  use m_functs_l_mult_type, only : get_nmult, get_j_si_fi
  use m_coul_comm, only : comp_overlap_v6_arg
  
  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1(:), spp2
  logical, intent(in) :: is_overlap
  real(8), intent(in) :: wigner_matrices1(:,:,:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1f2_mom(:), S(:), real_overlap(:,:), tmp(:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  !! internal
  integer :: i1,J1,J2,M1,M2,j,nmu2,mu2,si2,fi2
  real(8) :: f1j1f2j2
  integer :: nwigner1(3)
  
  nwigner1 = shape(wigner_matrices1)/2 + 1
  nmu2  = get_nmult(aux%sp_local2functs_mom(spp2))
  if(size(spp1)/=3) _die('size spp1 /=3')

  do mu2=1,nmu2;
    call get_j_si_fi(aux%sp_local2functs_mom(spp2), mu2, j2, si2,fi2)

    do i1=spp1(2), spp1(3)
      m1 = get_m(aux%sp_biloc2functs_mom(spp1(1)), i1)

      tmp = 0
      do j1=abs(m1), aux%jcutoff;
        S = 0
        if(is_overlap) then !! overlapping or not overlapping orbitals
          f1f2_mom = aux%sp_local2functs_mom(spp2)%ir_mu2v(:,mu2) * &
                     aux%sp_biloc2functs_mom(spp1(1))%ir_j_prd2v(:,j1,i1)
                 
          do j=abs(j1-j2),j1+j2,2;
            S(j)=sum(f1f2_mom*bessel_pp(:,j));
            S(j)=S(j)+f1f2_mom(1)*bessel_pp(1,j)/aux%dkappa
          enddo;
                   
        else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
          f1j1f2j2 = aux%sp_local2moms(spp2)%ir_mu2v(1,mu2) * &
                     aux%sp_biloc2moms(spp1(1))%ir_j_prd2v(1,j1,i1)

          do j=abs(j1-j2),j1+j2,2;
            if(j/=(j1+j2)) then
              S(j)=0
              cycle
            endif
            S(j)= aux%GGG(j,j2,j1)*f1j1f2j2*r_scalar_pow_jp1(j);
          enddo
        else
          S = 0;
        endif !! endif overlapping or not overlapping orbitals

        call comp_overlap_v6_arg(aux%jcutoff, j1,j2, aux%j1_j2_to_gaunt1, &
          aux%tr_c2r_diag1, aux%tr_c2r_diag2, aux%conjg_c2r_diag1, aux%conjg_c2r_diag2, & 
          S, ylm_thrpriv, real_overlap)

        !! Rotate the spherical harmonics in real_overlap
        do m2=-j2,j2
          tmp(m2) = tmp(m2) + &
            sum(wigner_matrices1(m1+nwigner1(1),-j1+nwigner1(2):j1+nwigner1(2),&
              j1+1)*real_overlap(-j1:j1,m2));
        enddo ! m2  
        !! END of Rotate the spherical harmonics in real_overlap 

      enddo ! j1

      array(i1,si2:fi2) = tmp(-j2:j2)
    enddo ! i1
  enddo ! mu2
     
end subroutine ! comp_kernel_block_21


!
!
!
subroutine comp_kernel_block_21(aux, spp1, spp2, &
  is_overlap, wigner_matrices1, array, f1f2_mom, S, real_overlap, tmp, &
  bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
 
  use m_functs_m_mult_type, only : get_nfunct_mmult,get_m 
  use m_functs_l_mult_type, only : get_nmult, get_j_si_fi
  use m_coul_comm, only : comp_overlap_v6_arg
  
  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1(:), spp2
  logical, intent(in) :: is_overlap
  real(8), intent(in), allocatable :: wigner_matrices1(:,:,:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1f2_mom(:), S(:), real_overlap(:,:), tmp(:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  !! internal
  integer :: i1,J1,J2,M1,M2,j,nmu2,mu2,si2,fi2
  real(8) :: f1j1f2j2
  
  nmu2  = get_nmult(aux%sp_local2functs_mom(spp2))
  if(size(spp1)/=3) _die('size spp1 /=3')

  do mu2=1,nmu2;
    call get_j_si_fi(aux%sp_local2functs_mom(spp2), mu2, j2, si2,fi2)

    do i1=spp1(2), spp1(3)
      m1 = get_m(aux%sp_biloc2functs_mom(spp1(1)), i1)

      tmp = 0
      do j1=abs(m1), aux%jcutoff;
        S = 0
        if(is_overlap) then !! overlapping or not overlapping orbitals
          f1f2_mom = aux%sp_local2functs_mom(spp2)%ir_mu2v(:,mu2) * &
                     aux%sp_biloc2functs_mom(spp1(1))%ir_j_prd2v(:,j1,i1)
                 
          do j=abs(j1-j2),j1+j2,2;
            S(j)=sum(f1f2_mom*bessel_pp(:,j));
            S(j)=S(j)+f1f2_mom(1)*bessel_pp(1,j)/aux%dkappa
          enddo;
                   
        else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
          f1j1f2j2 = aux%sp_local2moms(spp2)%ir_mu2v(1,mu2) * &
                     aux%sp_biloc2moms(spp1(1))%ir_j_prd2v(1,j1,i1)

          do j=abs(j1-j2),j1+j2,2;
            if(j/=(j1+j2)) then
              S(j)=0
              cycle
            endif
            S(j)= aux%GGG(j,j2,j1)*f1j1f2j2*r_scalar_pow_jp1(j);
          enddo
        else
          S = 0;
        endif !! endif overlapping or not overlapping orbitals

        call comp_overlap_v6_arg(aux%jcutoff, j1,j2, aux%j1_j2_to_gaunt1, &
          aux%tr_c2r_diag1, aux%tr_c2r_diag2, aux%conjg_c2r_diag1, aux%conjg_c2r_diag2, & 
          S, ylm_thrpriv, real_overlap)

        !! Rotate the spherical harmonics in real_overlap
        do m2=-j2,j2
          tmp(m2) = tmp(m2) + &
            sum(wigner_matrices1(m1,-j1:j1,j1)*real_overlap(-j1:j1,m2));
        enddo ! m2  
        !! END of Rotate the spherical harmonics in real_overlap 

      enddo ! j1

      array(i1,si2:fi2) = tmp(-j2:j2)
    enddo ! i1
  enddo ! mu2
     
end subroutine ! comp_kernel_block_21

!
!
!
real(8) function get_rcut(aux, type_of_mu, spp)
  implicit none
  ! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: type_of_mu, spp
  ! internal

  get_rcut = -999
  
  if(type_of_mu==1) then
    if(.not. allocated(aux%spp2rcut_type_of_mu1)) &
      _die('not allocated spp2rcut_type_of_mu1')
    if(spp<0 .or. spp>size(aux%spp2rcut_type_of_mu1)) &
     _die('spp not ok')

    get_rcut = aux%spp2rcut_type_of_mu1(spp)
  else if (type_of_mu==2) then

    if(.not. allocated(aux%spp2rcut_type_of_mu2)) &
      _die('not allocated spp2rcut_type_of_mu2')
    if(spp<0 .or. spp>size(aux%spp2rcut_type_of_mu2)) &
     _die('spp not ok')

    get_rcut = aux%spp2rcut_type_of_mu2(spp)
  else
    _die('unknown type_of_mu')
  endif

end function ! get_rcut

!
!
!
integer function get_nfunct(aux, type_of_mu, spp)
  implicit none
  ! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: type_of_mu, spp
  ! internal

  get_nfunct = 0
  if(type_of_mu==1) then
    if(.not. allocated(aux%spp2nfun_type_of_mu1)) &
      _die('not allocated spp2nfun_type_of_mu1')
    if(spp<0 .or. spp>size(aux%spp2nfun_type_of_mu1)) &
     _die('spp not ok')

    get_nfunct = aux%spp2nfun_type_of_mu1(spp)

  else if (type_of_mu==2) then

    if(.not. allocated(aux%spp2nfun_type_of_mu2)) &
      _die('not allocated spp2nfun_type_of_mu2')
    if(spp<0 .or. spp>size(aux%spp2nfun_type_of_mu2)) &
     _die('spp not ok')

    get_nfunct = aux%spp2nfun_type_of_mu2(spp)
  else
    _die('unknown type_of_mu')
  endif

end function ! get_nfunct

!
! This is obsolete and difficult to understand/ must be deleted soon
!
subroutine conv_hkernel_pack_re2dp(pb, vC_pack_re, vC_pack_dp)
  use m_precision, only : blas_int
  use m_pack_matrix, only : put_block_pack_mat
  use m_prod_basis_type, only : prod_basis_t, get_nfunct_domiprod, get_npairs
  use m_prod_basis_type, only : get_nfunct_max_re_pp, get_nfunct_max_pp
  !! external
  implicit none
  type(prod_basis_t), intent(in) :: pb
  real(8), intent(in), allocatable :: vC_pack_re(:)
  real(8), intent(inout), allocatable :: vC_pack_dp(:)
  !! internal
  integer :: nprod, npairs, pair1, pair2, si1(3), fi1(3), si2(3), fi2(3)
  !integer :: ni1(3), ni2(3)
  integer(blas_int) :: n1(2), n2(2), nfmx, nprod_mx
  real(8), allocatable :: vblock(:,:), vblock_dp(:,:), vblock_ac_dp(:,:)
  nprod = get_nfunct_domiprod(pb)
  npairs = get_npairs(pb)
  
  if(.not. allocated(pb%coeffs)) _die('.not. allocated(pb%coeffs)')
  if(.not. allocated(pb%book_re)) _die('.not. allocated(pb%book_re)')

  !! Allocate result
  _dealloc(vC_pack_dp)
  allocate(vC_pack_dp(nprod*(nprod+1)/2))
  !! END of Allocate result
 
  nfmx = get_nfunct_max_re_pp(pb)
  nprod_mx = get_nfunct_max_pp(pb)
  
  allocate(vblock(nfmx,nfmx))
  allocate(vblock_dp(nprod_mx, nprod_mx)) ! auxiliary
  allocate(vblock_ac_dp(nfmx, nprod_mx)) ! auxiliary
  
  do pair2=1,npairs
    si2 = pb%book_dp(pair2)%si; fi2 = pb%book_dp(pair2)%fi; !ni2 = fi2 - si2 + 1
    do pair1=1,pair2
      si1 = pb%book_dp(pair1)%si; fi1 = pb%book_dp(pair1)%fi; !ni1 = fi1 - si1 + 1
      
      call get_block_from_reexpr_kernel(pb, vC_pack_re, pair1,pair2, vblock,vblock_dp)
      
      n1 = ubound(pb%coeffs(pair1)%coeffs_ac_dp)
      n2 = ubound(pb%coeffs(pair2)%coeffs_ac_dp)
      call DGEMM('N', 'N', n1(1), n2(2), n2(1), 1D0, &
        vblock, nfmx, pb%coeffs(pair2)%coeffs_ac_dp, n2(1), 0D0, vblock_ac_dp, nfmx)
      call DGEMM('T', 'N', n1(2), n2(2), n1(1), 1D0, &
        pb%coeffs(pair1)%coeffs_ac_dp, n1(1), vblock_ac_dp, nfmx, 0D0, vblock_dp, nprod_mx)

      call put_block_pack_mat(vblock_dp, si1(3), fi1(3), si2(3), fi2(3), vC_pack_dp)
 
    enddo ! pair1
  enddo ! pair2  

end subroutine ! conv_hkernel_pack_re2dp

!
!
!
subroutine get_block_from_reexpr_kernel(pb, vC_pack_re, pair1, pair2, vblock, vblock_dp)
  use m_prod_basis_type, only : prod_basis_t
  use m_pack_matrix, only : get_block
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair1, pair2
  real(8), intent(in) :: vC_pack_re(:)
  real(8), intent(inout) :: vblock(:,:)
  real(8), intent(inout) :: vblock_dp(:,:)
  !! internal
  integer :: nind1, nind2, ind1, ind2, pair2_co, pair1_co, ldvdp
  integer :: s1, f1, n1, s2, f2, n2, si1, fi1, si2, fi2

  if(.not. allocated(pb%coeffs)) &
    _die('!%coeffs)')
  if(.not. allocated(pb%coeffs(pair1)%ind2book_re)) &
    _die('!%coeffs(pair1)%ind2book_re)')
  if(.not. allocated(pb%coeffs(pair2)%ind2book_re)) &
    _die('!%coeffs(pair2)%ind2book_re)')
    
  nind1 = size(pb%coeffs(pair1)%ind2book_re)
  nind2 = size(pb%coeffs(pair2)%ind2book_re)
  ldvdp = size(vblock_dp,1)
  
  f2 = 0
  do ind2=1,nind2
    pair2_co = pb%coeffs(pair2)%ind2book_re(ind2); 
    if(pair2_co<0) _die('pair2_co<0')
    si2 = pb%book_re(pair2_co)%si(3); fi2 = pb%book_re(pair2_co)%fi(3)
    s2 = f2 + 1; n2 = fi2 - si2 + 1; f2 = s2 + n2 - 1;
    f1 = 0
    do ind1=1,nind1
      pair1_co = pb%coeffs(pair1)%ind2book_re(ind1); 
      if(pair1_co<0) _die('pair1_co<0')
      si1 = pb%book_re(pair1_co)%si(3); fi1 = pb%book_re(pair1_co)%fi(3)
      s1 = f1 + 1; n1 = fi1 - si1 + 1; f1 = s1 + n1 - 1;

      call get_block(vC_pack_re, si1, fi1, si2, fi2, vblock_dp, ldvdp)
      vblock(s1:f1,s2:f2) =  vblock_dp(1:n1, 1:n2)
    enddo
  enddo
  
end subroutine ! init_hkernel_reexpr_block

!!
!!
!!
!subroutine init_mult_moments(pb, sp_local2moms, sp_biloc2moms)
!  use m_functs_m_mult_type, only : init_moms_mmult
!  use m_functs_l_mult_type, only : init_moms_lmult
!  use m_prod_basis_type, only : prod_basis_t  
!  implicit none
!  !! external
!  type(prod_basis_t), intent(in) :: pb
!  type(functs_l_mult_t), intent(inout), allocatable :: sp_local2moms(:)
!  type(functs_m_mult_t), intent(inout), allocatable :: sp_biloc2moms(:)
!  !! internal
  
!  _dealloc(sp_local2moms)
!  if(allocated(pb%sp_local2functs)) &
!    call init_moms_lmult(pb%sp_local2functs, pb%rr, sp_local2moms)

!  _dealloc(sp_biloc2moms)
!  if(allocated(pb%sp_biloc2functs)) &
!    call init_moms_mmult(pb%sp_biloc2functs, pb%rr, sp_biloc2moms)

!end subroutine !  init_mult_moments

!
!
!
integer function get_jcutoff(aux)
  implicit none
  type(pb_coul_aux_t), intent(in) :: aux
  if(aux%jcutoff<0) _die('jcutoff<0')
  get_jcutoff = aux%jcutoff
end function ! get_jcutoff  


!!
!!
!!
subroutine distr_blocks(ls_blocks, node2size, node2displ, node2fb, node2lb, para)
  use m_book_pb, only : book_pb_t
  use m_parallel, only : para_t
  use m_die, only : die
  use m_warn, only : warn
   
  implicit none
  !! external
  type(book_pb_t), intent(in) :: ls_blocks(:,:)
  integer(8), intent(inout), allocatable :: node2size(:), node2displ(:)
  integer, intent(inout), allocatable :: node2fb(:), node2lb(:)
  type(para_t), intent(in) :: para

  !! internal
  integer :: block, np1, np2, nblocks, size_of_block, si13, si23, nodes
  integer(8) :: total_size
!  real(8) :: factor1, factor2!, cost_of_block

  !! Output
  nodes = para%nodes
  if(nodes>1) then
    write(0,*) 'The distribution has to be rewritten for many nodes//and general ls_cpairs'
    _warn('distribute_blocks: para%nodes>1')
  endif  
  
  _dealloc(node2size)
  _dealloc(node2displ)
  _dealloc(node2fb)
  _dealloc(node2lb)
  
  allocate(node2size(0:nodes-1))
  allocate(node2displ(0:nodes-1))
  allocate(node2fb(0:nodes-1))
  allocate(node2lb(0:nodes-1))

  total_size = 0
  nblocks = size(ls_blocks,2)
  do block=1,nblocks
    si13 = ls_blocks(1,block)%si(3); si23 = ls_blocks(2,block)%si(3)
    if(si13<1 .or. si23<1) _die('si<1')

    np1 = ls_blocks(1,block)%fi(3) - si13 + 1
    np2 = ls_blocks(2,block)%fi(3) - si23 + 1
    if(np1<1 .or. np2<1) _die('np<1')
    
!    factor1=1.0D0
!    if(ls_blocks(1,block)%top==2) factor1=jcutoff+2

!    factor2=1.0D0
!    if(ls_blocks(2,block)%top==2) factor2=jcutoff+2
    
    if(si13==si23) size_of_block = np1*(np1+1)/2
    if(si13/=si23) size_of_block = np1*np2
    total_size = total_size + size_of_block
!    cost_of_block = size_of_block*factor1*factor2 + 70.0D0; !! cost of the block

  enddo ! block

  node2size(0:nodes-1) = total_size
  node2displ(0:nodes-1) = 0
  node2fb(0:nodes-1) = 1
  node2lb(0:nodes-1) = nblocks
  
end subroutine !distr_blocks

!
!
!
subroutine comp_aux(a, rcut1, Rvec1, rcut2, Rvec2, ylm, is_overlap, r_scalar_pow_jp1, bessel_pp)
!  use m_pb_coul_aux, only : pb_coul_aux_t
  use m_abramowitz, only : spherical_bessel
  use m_csphar, only : csphar
  implicit  none
  !! external
  type(pb_coul_aux_t), intent(in) :: a
  real(8), intent(in) :: rcut1, Rvec1(:)
  real(8), intent(in) :: rcut2, Rvec2(:)
  complex(8), intent(inout), allocatable :: ylm(:)
  logical, intent(inout) :: is_overlap
  real(8), intent(inout), allocatable :: r_scalar_pow_jp1(:)
  real(8), intent(inout), allocatable :: bessel_pp(:,:)
  
  !! internal
  real(8) :: r_vec(3), r_scalar
  integer :: L, ir, j
  
  r_vec = Rvec2 - Rvec1

  call csphar(r_vec, ylm, 2*a%jcutoff);

  r_scalar = sqrt(sum(r_vec*r_vec));
  if(a%use_mult>0) then
    is_overlap = (r_scalar<(rcut1+rcut2))
  else
    is_overlap = .true.
  endif  

    
  if(a%logical_overlap) then   ! We compute overlap
    if(is_overlap) then ! only if the orbitals overlap we need this bessel function
      do L=0,2*a%jcutoff
        do ir=1,a%nr;
          bessel_pp(ir,L)=a%pp(ir)**3 * spherical_bessel(L,r_scalar*a%pp(ir))
        enddo !ir
      enddo ! L
    endif

  else  ! We compute Hartree kernel

    if(is_overlap) then ! only if the orbitals overlap we need this bessel function
      do L=0,2*a%jcutoff
        do ir=1,a%nr;
          bessel_pp(ir,L)=spherical_bessel(L,r_scalar*a%pp(ir))*a%pp(ir)
        enddo !ir
      enddo ! L

      if(a%scr_const>0) then
        do l=0,2*a%jcutoff; bessel_pp(:,l)=bessel_pp(:,l)*a%scr_pp(:); enddo ! L
      endif
               
    else  !! non overlaping functions
      do j=0,2*a%jcutoff; r_scalar_pow_jp1(j) = 1.0D0/(r_scalar**(j+1)); enddo
    endif !! is_overlap

  endif  !! logical_sigma_overlap

end subroutine ! comp_aux

!
!
!
subroutine alloc_runtime_coul(node2fb,node2lb,block2runtime_coul,para)
  use m_parallel, only : para_t
  implicit none
  !! external
  integer, allocatable, intent(in) :: node2fb(:), node2lb(:)
  real(8), allocatable, intent(inout)    :: block2runtime_coul(:)
  type(para_t), intent(in) :: para
  !! internal

#ifdef TIMING
  _dealloc(block2runtime_coul)
  allocate(block2runtime_coul(node2fb(para%rank):node2lb(para%rank)))
  block2runtime_coul = -999
#endif

end subroutine ! alloc_runtime_coul  


end module !m_coul_aux
