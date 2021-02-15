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

module m_prod_basis_type

! The purpose of the module is to store and deal with a real space information of a product basis
#include "m_define_macro.F90"
  use m_die, only : die
  use m_book_pb, only : book_pb_t
  use m_functs_m_mult_type, only : functs_m_mult_t
  use m_functs_l_mult_type, only : functs_l_mult_t
  use m_coeffs_type, only : d_coeffs_ac_dp_t
  use m_system_vars, only : system_vars_t
  use m_prod_basis_param, only : prod_basis_param_t, dealloc_pb_p=>dealloc
  use m_vertex_3cent, only : vertex_3cent_t
  
  implicit none
  private die, d_coeffs_ac_dp_t

  !! Vertex part, local pairs
  type vertex_1cent_t
    real(8), allocatable :: vertex(:,:,:)
    integer :: sp = -999  ! atom specie
    integer :: spp =-999 ! product specie (probable the same pointer as atom specie ``sp'' ???)
  end type ! vertex_1cent_t
  !! END of Vertex part, local pairs

  !! Top level type
  type prod_basis_t
    type(system_vars_t), pointer :: sv
    type(prod_basis_param_t) :: pb_p 
    
    !! Data generally used by the other fields in this structure 
    real(8) :: uc_vecs(3,3) = 0          ! description of unit cell orts (xyz, 123)
    real(8), allocatable :: rr(:), pp(:) ! logarithmic grids in real and momentum space
    real(8), allocatable :: atom2coord(:,:) ! cartesian coordinates of atoms (in unit cell)
    !! END of Data generally used by the other fields in this structure
    
    !! Data connected to standard dominant products, other data may refer to this data
    type(vertex_1cent_t), allocatable  :: sp_local2vertex(:)
    type(functs_l_mult_t), allocatable :: sp_local2functs(:)
    type(functs_l_mult_t), allocatable :: sp_local2functs_mom(:)

    type(vertex_3cent_t), allocatable  :: sp_biloc2vertex(:)
    type(functs_m_mult_t), allocatable :: sp_biloc2functs(:)
    type(functs_m_mult_t), allocatable :: sp_biloc2functs_mom(:)

    type(book_pb_t), allocatable :: book_dp(:) ! center 2 coordinates, species indices, global counting in dominant products
    integer :: nfunct_irr        = -1 ! size of matrices in the product basis 
       ! (this number should be count excluding translation and inversion symmetries)
    integer :: irr_trans_inv_sym = -1 ! symmetries that are present in the basis (copy of similar field from overlap)
    !! END of Data connected to standard dominant products, other data may refer to this data

    !! Data connected to reexpression of certain dominant products in terms of the others
    type(d_coeffs_ac_dp_t), allocatable :: coeffs(:)     ! Conversion coefficients between 
          ! dominant products and a limited subset of them
    type(book_pb_t), allocatable :: book_re(:) ! A global counting in reexpressing basis
    !! END of Data connected to reexpression of certain dominant products in terms of the others
   
    !! Data connected to folding into unit cell
    type(book_pb_t), allocatable :: book_uc(:)
    !! END of Data connected to folding into unit cell

    !! This tells which book structure is used in loops over product basis:
    !! if book_type==1 then %book_dp(:) will be used i.e. dominant products
    !! if book_type==2 then %book_re(:) will be used i.e. mixed basis in which given bilocal pairs get reexpressed
    integer :: book_type = -1
    
    character(100) :: BlochPhaseConv = "" ! TEXTBOOK or SIMPLE    
  end type ! prod_basis_t
  !! END of Top level type

#define _dealloc_alloc_thesame(master, slave) \
  if(allocated(slave)) deallocate(slave); \
  if(.not. allocated(master)) call die('_dealloc_alloc_thesame', __FILE__, __LINE__); \
  allocate(slave(size(master)));

  contains


subroutine dealloc(v)
  implicit none
  type(prod_basis_t), intent(inout) :: v
  
  v%sv => null()
  call dealloc_pb_p(v%pb_p)
  
  _dealloc(v%rr)
  _dealloc(v%pp)
  _dealloc(v%atom2coord)
  _dealloc(v%sp_local2vertex)
  _dealloc(v%sp_local2functs)
  _dealloc(v%sp_local2functs_mom)
  _dealloc(v%sp_biloc2vertex)
  _dealloc(v%sp_biloc2functs)
  _dealloc(v%sp_biloc2functs_mom)
  _dealloc(v%book_dp)
  _dealloc(v%coeffs)
  _dealloc(v%book_re)
  _dealloc(v%book_uc)

  v%uc_vecs = 0
  v%nfunct_irr = 0
  v%irr_trans_inv_sym = 0
  v%book_type = 0
  v%BlochPhaseConv = ""

end subroutine ! dealloc

!
!
!
subroutine get_spent_ram(pb, ram_bytes) 
  use m_get_sizeof, only : get_sizeof
  use m_functs_m_mult_type, only : get_spent_ram_bsp=>get_spent_ram
  use m_functs_l_mult_type, only : get_spent_ram_lsp=>get_spent_ram
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  real(8), intent(inout) :: ram_bytes(4)
  !! internal
  integer :: n, i
  ram_bytes = 0

  _size_alloc(pb%sp_biloc2functs, n)
  do i=1,n; ram_bytes(1) = ram_bytes(1) + get_spent_ram_bsp(pb%sp_biloc2functs(i)); enddo ! i
  
  _size_alloc(pb%sp_biloc2vertex,  n)
  do i=1,n
    ram_bytes(2) = ram_bytes(2) + get_spent_ram_vertex(pb%sp_biloc2vertex(i)%vertex)
  enddo ! i

  _size_alloc(pb%sp_local2functs, n)
  do i=1,n; ram_bytes(3) = ram_bytes(3) + get_spent_ram_lsp(pb%sp_local2functs(i)); enddo ! i
  
  _size_alloc(pb%sp_local2vertex, n)
  do i=1,n; 
    ram_bytes(4) = ram_bytes(4) + get_spent_ram_vertex(pb%sp_local2vertex(i)%vertex); 
  enddo ! i
  
end subroutine! get_spent_ram


!
!
!
function get_spent_ram_vertex(vertex) result (ram_bytes)
  use m_get_sizeof
  implicit none
  real(8), allocatable, intent(in) :: vertex(:, :, :)
  real(8) :: ram_bytes

  integer(8) :: nr

  nr = 0
  ram_bytes = 0D0
  if (.not. allocated(vertex)) return!_die('vertex not allocated')
  _add_size_alloc(nr, vertex)

  ram_bytes = nr*get_sizeof(vertex(1,1,1))

end function !get_spent_ram_vertex


!
!
!
logical function is_init(pb)
  implicit none
  type(prod_basis_t), intent(in) :: pb
  logical :: linit(9)

  linit = .false.
  linit(1) = associated(pb%sv)
  linit(2) = sum(abs(pb%uc_vecs))/=0
  linit(3) = allocated(pb%pp)
  linit(4) = allocated(pb%rr)
  linit(5) = allocated(pb%atom2coord)
  linit(6) = allocated(pb%sp_local2vertex)
  linit(7) = allocated(pb%sp_local2functs)
  linit(8) = allocated(pb%sp_local2functs_mom)
  linit(9) = allocated(pb%book_dp)
  is_init = all(linit)

end function ! is_init

!
! Compute the coordinate of a center for a given bookkeeping record and 
! the index of the center within this bookkeeping record.
! The bookkeeping records enumerate atom pairs.
!
function get_coord_center(pb, bk, icwp) result(Rvec)
  implicit none
  !! external
  type(book_pb_t), intent(in) :: bk
  integer, intent(in) :: icwp       ! index of the center within bookkeping record
  type(prod_basis_t), intent(in) :: pb
  real(8) :: Rvec(3)

  select case (bk%top)
  case(1) 
    Rvec = matmul(pb%uc_vecs(1:3,1:3), bk%cells(:,3)) + bk%coord
  case(2)
    Rvec = pb%sp_biloc2functs(bk%spp)%crc(1:3,icwp)
  case default
    _die('!%top')
  end select
    
end function ! get_coord_center

!
!
!
real(8) function get_rcut_per_center(pb, bk, icwp)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(in) :: bk
  integer, intent(in) :: icwp       ! index of the center within bookkeeping record

  get_rcut_per_center = -999
  select case (bk%top)
  case(1)
    get_rcut_per_center = pb%sp_local2functs(bk%spp)%rcut
  case(2)
    get_rcut_per_center = pb%sp_biloc2functs(bk%spp)%crc(4,icwp)
  case default
    _die('!%top')
  end select
  if(get_rcut_per_center<0) _die('rcut_per_center<0')

end function ! get_rcut_per_center

!
!
!
function get_spp_sp_fp(pb, bk, icwp) result(sppa)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(in) :: bk
  integer, intent(in) :: icwp  ! index of the center within bookkeping record
  integer :: sppa(3)
  !! internal

  select case (bk%top)
  case(1)
    sppa = [bk%spp, 1, bk%fi(3)-bk%si(3)+1]
  case(2)
    sppa = [bk%spp, int(pb%sp_biloc2functs(bk%spp)%crc(5:6,icwp))]
  case default
    _die('!%top')
  end select
 
  if(sppa(2)<1) then
    write(6,*) bk%top, bk%spp
    if(bk%top==2) then
      write(6,*) pb%sp_biloc2functs(bk%spp)%crc
    endif
    _die('sppa(2)<1')
  endif

end function ! get_spp_sp_fp

!
! Finds out the number of centers within a bookkeeping record 
!
integer function get_nc_book(pb, bk)
  implicit none
  !! external
  type(book_pb_t), intent(in) :: bk
  type(prod_basis_t), intent(in) :: pb
  
  get_nc_book = -999
  select case (bk%top)
  case(1)
    get_nc_book = 1
  case(2)
    get_nc_book = size(pb%sp_biloc2functs(bk%spp)%crc,2)
    if(get_nc_book<1) _die('?nc_book<1')
  case default
    _die('!%top')
  end select

end function ! get_nc_book

!
! Pointer to the original system variables data
! for which the product basis is generated.
!
function get_sv_ptr(pb) result(sv_ptr)
  use m_system_vars, only : system_vars_t, get_norbs
  implicit none
  type(prod_basis_t), intent(in), target :: pb
  type(system_vars_t), pointer :: sv_ptr
  !! internal
  integer :: no
  no = get_norbs(pb%sv)
  if(no<1) _die('no<1')
  sv_ptr => pb%sv
  
end function ! get_sv_ptr

!
!
!
character(100) function get_BlochPhaseConv(pb)
  use m_upper, only : upper
  implicit none
  type(prod_basis_t), intent(in) :: pb
  if(len_trim(pb%BlochPhaseConv)<1) _die('len_trim(BlochPhaseConv)<1')
  get_BlochPhaseConv = upper(pb%BlochPhaseConv)
end function !  get_BlochPhaseConv 

!
!
!
subroutine set_BlochPhaseConv(BlochPhaseConv, pb)
  use m_upper, only : upper
  implicit none
  character(*), intent(in) :: BlochPhaseConv
  type(prod_basis_t), intent(inout) :: pb
  if(len_trim(BlochPhaseConv)<1) _die('len_trim(BlochPhaseConv)<1')
  pb%BlochPhaseConv = upper(BlochPhaseConv)
end subroutine !  set_BlochPhaseConv

!
!
!
character(100) function get_prod_basis_type(pb)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  if(allocated(pb%coeffs) .and. allocated(pb%book_re)) then
    get_prod_basis_type = "MIXED"
  else
    get_prod_basis_type = "DOMIPROD"
  endif
end function ! get_prod_basis_type      

!
!
!
function get_uc_volume(pb) result(v_uc)
  use m_algebra, only : cross_product
  implicit none
  type(prod_basis_t), intent(in) :: pb
  real(8) :: v_uc
  !! internal
  real(8) :: uc_vecs(3,3)
  
  uc_vecs = get_uc_vecs(pb)
  v_uc = abs( dot_product( uc_vecs(:,1),cross_product(uc_vecs(:,2),uc_vecs(:,3))))
  
end function ! get_uc_volume

!
!
!
subroutine aug_local_dp(functs_aug, sp_local2functs, sp_local2vertex)
  use m_functs_l_mult_type, only : get_jcut_lmult, get_nmult
  use m_functs_l_mult_type, only : get_j_si_fi, get_nr_lmult, get_ff_lmult
  implicit none
  !! external
  type(functs_l_mult_t), intent(in), allocatable :: functs_aug(:)
  type(functs_l_mult_t), intent(inout), allocatable :: sp_local2functs(:)
  type(vertex_1cent_t), intent(inout), allocatable :: sp_local2vertex(:)
  !! internal
  integer :: nspp,spp,jcut_ori,jcut_aug,j,si,fi,nn(3)
  integer :: nmult_ori, nmult_aug, mu,nr1,nr,nmult_tot,mu_aug,s,f,n
  type(functs_l_mult_t) :: functs
  type(vertex_1cent_t)  :: vertex
  
  if(.not. allocated(functs_aug)) return
  if(.not. allocated(sp_local2functs)) _die('na sp_local2functs')
  if(.not. allocated(sp_local2vertex)) _die('na sp_local2vertex')
  nspp = size(sp_local2functs)
  if(nspp/=size(functs_aug)) _die('nspp/=size(functs_aug)')
  if(nspp/=size(sp_local2vertex)) _die('nspp/=size(sp_local2vertex')
  nr1 = get_nr_lmult(sp_local2functs(1))
  nr = get_nr_lmult(functs_aug(1))
  if(nr/=nr1) _die('nr/=nr1')
  
  do spp=1,nspp
    jcut_ori = get_jcut_lmult(sp_local2functs(spp))
    jcut_aug = get_jcut_lmult(functs_aug(spp))
    if(jcut_aug<=jcut_ori) cycle ! nothing to add ?
    nmult_ori = get_nmult(sp_local2functs(spp))
    nmult_aug = get_nmult(functs_aug(spp))

    functs = sp_local2functs(spp)
    vertex = sp_local2vertex(spp)

    _dealloc(sp_local2functs(spp)%mu2rcut)
    _dealloc(sp_local2functs(spp)%mu2j)
    _dealloc(sp_local2functs(spp)%mu2si)
    _dealloc(sp_local2functs(spp)%ir_mu2v)
    _dealloc(sp_local2vertex(spp)%vertex)
     
    nmult_tot = count(functs_aug(spp)%mu2j>jcut_ori) + nmult_ori
    allocate(sp_local2functs(spp)%mu2j(nmult_tot))
    allocate(sp_local2functs(spp)%mu2rcut(nmult_tot))
    allocate(sp_local2functs(spp)%mu2si(nmult_tot))
    allocate(sp_local2functs(spp)%ir_mu2v(nr,nmult_tot))
    sp_local2functs(spp)%mu2j(1:nmult_ori) = functs%mu2j(1:nmult_ori)
    sp_local2functs(spp)%ir_mu2v(:,1:nmult_ori) = functs%ir_mu2v(:,1:nmult_ori)
    sp_local2functs(spp)%mu2rcut = functs%rcut
    sp_local2functs(spp)%rcut = functs%rcut
    mu = nmult_ori
    do mu_aug=1,nmult_aug
      call get_j_si_fi(functs_aug(spp), mu_aug, j, si, fi)
      if(j<=jcut_ori) cycle
      mu = mu + 1
      sp_local2functs(spp)%mu2j(mu)      = j
      call get_ff_lmult(functs_aug(spp), mu_aug, sp_local2functs(spp)%ir_mu2v(:,mu))
    enddo ! mu_aug 
  
!    write(6,'(a35,100i5)') 'sp_local2functs(spp)%mu2j', sp_local2functs(spp)%mu2j
!    write(6,'(a35,e20.10)') 'sum', sum(sp_local2functs(spp)%ir_mu2v)
    f = 0
    do mu=1,nmult_tot
      j = sp_local2functs(spp)%mu2j(mu)
      s = f + 1; n = 2*j+1; f = s + n - 1;
      sp_local2functs(spp)%mu2si(mu)=s
    enddo ! 
    sp_local2functs(spp)%nfunct = f

    nn = ubound(vertex%vertex)
    !write(6,'(a35,3i5,2x,1i15)') 'nn,f', nn, f 
    allocate(sp_local2vertex(spp)%vertex(nn(1),nn(2),f))
    sp_local2vertex(spp)%vertex = 0
    sp_local2vertex(spp)%vertex(:,:,1:nn(3)) = vertex%vertex(:,:,1:nn(3))

  enddo ! spp
  
  
end subroutine ! aug_local_dp

!
!
!
function get_jcutoff_lmult(pb) result(jcut)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: jcut
  !! internal
  integer :: sp, nspp
  jcut = 0
  nspp=0
  if(allocated(pb%sp_local2functs) ) nspp = size(pb%sp_local2functs)
  jcut = 0
  do sp=1,nspp
    jcut = max(jcut, maxval(pb%sp_local2functs(sp)%mu2j))
  enddo ! sp
    
end function !  get_jcut_lmult 

!
! Which unit cell basis vectors are used by prod_basis_t. Consistency checks ?
!
function get_uc_vecs(pb) result(uc_vecs)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  real(8) :: uc_vecs(3,3)  
  integer :: i
  do i=1,3; if(sum(abs(pb%uc_vecs(:,i)))<1d-14) _die('?uc_vecs'); enddo
  uc_vecs = pb%uc_vecs
end function ! get_uc_vecs

!
!
!
function get_coord_uc(book) result(coord)
  implicit none
  type(book_pb_t), intent(in) :: book
  real(8) :: coord(3)
  coord = book%coord
end function ! get_coord_uc
  

!
! Initializes a continuous global counting for product functions
!
subroutine init_global_counting3(pb, book)
  implicit none
  ! external
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(inout), allocatable :: book(:)
  ! internal
  integer :: ibook, f3, s3, nbook, n3

  !! Init si(3), fi(3) fields
  if(.not. allocated(book)) &
    _die('.not. allocated(book)')

  nbook = size(book)
  f3 = 0
  do ibook=1, nbook
    n3 = get_nfunct(pb, book(ibook)%top, book(ibook)%spp)
    s3 = f3 + 1; f3 = s3 + n3 - 1
    book(ibook)%si(3) = s3
    book(ibook)%fi(3) = f3
  enddo !ibook
  !! END of Init si(3), fi(3) fields

end subroutine ! init_global_counting3

!
!
!
function get_nspp(pb, type_of_centers) result(nf)
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: type_of_centers
  integer :: nf
  !! internal
  nf = 0
  if(type_of_centers==1) then
    nf = 0
    if(allocated(pb%sp_local2functs)) nf = size(pb%sp_local2functs)
  else if (type_of_centers==2) then
    nf = 0
    if(allocated(pb%sp_biloc2functs)) nf = size(pb%sp_biloc2functs)
  else
    write(0,*) 'type_of_centers', type_of_centers
    _die('unknown type_of_centers')
  endif

end function !   get_nspp  
 

!
!
!
function get_rcut(pb, type_of_center, spp, ic) result(res)
  use m_functs_m_mult_type, only : get_ncenters
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: type_of_center, spp, ic
  real(8) :: res
  !! internal
  integer :: nlc
  
  res = -999
  if(type_of_center==1) then
    if(.not. allocated(pb%sp_local2functs)) &
      _die('.not. allocated(pb%sp_local2functs)')
    if(spp<1 .or. spp>size(pb%sp_local2functs)) &
      _die('spp<1 .or. spp>size(pb%sp_local2functs)')
    res = pb%sp_local2functs(spp)%rcut
  else if (type_of_center==2) then
    if(.not. allocated(pb%sp_biloc2functs)) &
      _die('.not. allocated(pb%sp_biloc2functs)')
    if(spp<1 .or. spp>size(pb%sp_biloc2functs)) &
      _die('spp<1 .or. spp>size(pb%sp_biloc2functs)')
    nlc = get_ncenters(pb%sp_biloc2functs(spp))
    if(ic<1 .or. ic>nlc) _die('!ic') 
    res = pb%sp_biloc2functs(spp)%crc(4,ic)
  else
    _die('wrong type_of_center ??')
  endif    
  
end function !  get_rcut 

!
!
!
function get_rcut_book(pb, book) result(res)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(in) :: book
  real(8) :: res
  !! internal

  res = get_rcut(pb, book%top, book%spp, 1) 

end function !  get_rcut 

!
!
!
subroutine init_functs_mom_space(pb, sp_local2functs_mom, sp_biloc2functs_mom)
  use m_sph_bes_trans, only : Talman_plan_t, sbt_plan
  use m_functs_m_mult_type, only : init_functs_mmult_mom_space
  use m_functs_l_mult_type, only : init_functs_lmult_mom_space
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  type(functs_l_mult_t), intent(inout), allocatable :: sp_local2functs_mom(:)
  type(functs_m_mult_t), intent(inout), allocatable :: sp_biloc2functs_mom(:)
  !! internal
  integer :: nr,jcutoff
  type(Talman_plan_t) :: Talman_plan
  jcutoff = get_jcutoff(pb)
  nr = get_nr(pb)
 
  call sbt_plan(Talman_plan, nr, jcutoff, pb%rr, pb%pp, .true.)
  call init_functs_lmult_mom_space(Talman_plan, pb%sp_local2functs, sp_local2functs_mom)
  call init_functs_mmult_mom_space(Talman_plan, pb%sp_biloc2functs, sp_biloc2functs_mom)

end subroutine ! init_functs_mmult_mom_space

!
!
!
subroutine init_internal_functs_mom_space(pb)
  use m_sph_bes_trans, only : Talman_plan_t, sbt_plan
  use m_functs_m_mult_type, only : init_functs_mmult_mom_space
  use m_functs_l_mult_type, only : init_functs_lmult_mom_space
  implicit none
  !! external
  type(prod_basis_t), intent(inout) :: pb
  !! internal
  integer :: nr,jcutoff
  type(Talman_plan_t) :: Talman_plan
  jcutoff = get_jcutoff(pb)
  nr = get_nr(pb)
  !! END of Dimensions
 
  call sbt_plan(Talman_plan, nr, jcutoff, pb%rr, pb%pp, .true.)
  call init_functs_lmult_mom_space(Talman_plan, pb%sp_local2functs, pb%sp_local2functs_mom)
  call init_functs_mmult_mom_space(Talman_plan, pb%sp_biloc2functs, pb%sp_biloc2functs_mom)

end subroutine ! init_internal_functs_mmult_mom_space

!
!
!
function  get_nr(pb) result(nf)
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: nr_pp, nr_rr
  
  if(.not. allocated(pb%rr)) _die('.not. allocated(pb%rr)');
  if(.not. allocated(pb%pp)) _die('.not. allocated(pb%pp)');
  
  nr_pp = size(pb%pp)
  nr_rr = size(pb%rr)
  if(nr_pp/=nr_rr) _die('nr_pp/=nr_rr')
  
  !! Some more consistency checks can be added here...
  nf = nr_pp
end function !get_nr

!
!
!
function get_jcutoff(pb) result(nf)
  use m_functs_m_mult_type, only: get_jcutoff_mmult
  use m_functs_l_mult_type, only: get_jcutoff_lmult
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: nspp

  nf = -1;
  nspp = 0; if(allocated(pb%sp_local2functs)) nspp = size(pb%sp_local2functs)
  if(nspp>0) nf = max(nf, get_jcutoff_lmult(pb%sp_local2functs))
    
  nspp = 0; if(allocated(pb%sp_biloc2functs)) nspp = size(pb%sp_biloc2functs)
  if(nspp>0) nf = max(nf, get_jcutoff_mmult(pb%sp_biloc2functs))

end function ! get_jcutoff

!
! Gets maximal number of orbitals in the product basis
!
function get_norbs_max(pb) result(nf)
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: spp, nspp, n1, n2

  nf = -1;
  nspp = 0; if(allocated(pb%sp_local2vertex)) nspp = size(pb%sp_local2vertex)
  do spp=1,nspp
    if(.not. allocated(pb%sp_local2vertex(spp)%vertex)) &
      _die('.not. allocated(pb%sp_local2vertex(spp)%vertex)')
    n1 = size(pb%sp_local2vertex(spp)%vertex,1)
    n2 = size(pb%sp_local2vertex(spp)%vertex,2)
    if (n1/=n2) _die('n1/=n2')
    nf = max(nf, n1, n2)
  enddo ! spp
    
  nspp = 0; if(allocated(pb%sp_biloc2vertex)) nspp = size(pb%sp_biloc2vertex)
  do spp=1,nspp
    if(.not. allocated(pb%sp_biloc2vertex(spp)%vertex)) cycle
      !_die('.not. allocated(pb%sp_biloc2vertex(spp)%vertex)')
    n1 = size(pb%sp_biloc2vertex(spp)%vertex,1)
    n2 = size(pb%sp_biloc2vertex(spp)%vertex,2)
    nf = max(nf, n1, n2)
  enddo ! spp

end function !   get_norbs_max
  
!
! Determines the maximal number of functions for atom pairs in the basis
! Right now it will determine maximal number of functions in the product species...
!
function get_nfunct_max_pp_fu(pb) result(nf)
  use m_functs_m_mult_type, only: get_nfunct_mmult
  use m_functs_l_mult_type, only: get_nfunct_lmult
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: spp, nspp
  
  nf = -1
  
  nspp = 0; if(allocated(pb%sp_local2functs)) nspp = size(pb%sp_local2functs)
  do spp=1,nspp
    nf = max(nf, get_nfunct_lmult(pb%sp_local2functs(spp)))
  enddo ! spp
    
  nspp = 0; if(allocated(pb%sp_biloc2functs)) nspp = size(pb%sp_biloc2functs)
  do spp=1,nspp
    nf = max(nf, get_nfunct_mmult(pb%sp_biloc2functs(spp)))
    !nf = max(nf, pb%sp_biloc2functs(spp)%nfunct)
  enddo ! spp

end function !   get_nfunct_max_pp_fu

!
! Determines the maximal number of functions for atom pairs in the basis
!
function get_nfunct_max_pp(pb) result(nf)
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: pair, npairs, top, n
  
  nf = -1
  npairs = get_npairs(pb)
  do pair=1,npairs
    top = get_top_dp(pb, pair)
    if(top==-1) cycle
    n = pb%book_dp(pair)%fi(3)-pb%book_dp(pair)%si(3) + 1
    if(n<1) _die('n<1')
    nf = max(nf, n) 
  enddo ! pair  

end function !   get_nfunct_max_pp


!
! Returns a vertex part of dominant products
!
subroutine get_vertex_pair_dp(pb, pair, op, vertex, n)
  !! external
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair, op
  real(8), intent(inout) :: vertex(:,:)
  integer, intent(inout) :: n(3)
  !! internal
  integer :: spp, ns,o,mu,n2(2)
  
  if( .not. allocated(pb%book_dp) ) &
    _die('.not. allocated(pb%book_dp)' )
  if( pair<0 .or. pair>size(pb%book_dp)) &
    _die('pair<0 .or. pair>size(pb%book_dp)')
  
  spp = pb%book_dp(pair)%spp
  
  n = 0
  if(pb%book_dp(pair)%top==1) then

    n = ubound(pb%sp_local2vertex(spp)%vertex)
    if(any(n<1)) _die('any(n<1)')
    n2 = (/n(1)*n(2),n(3)/)
    if(any(ubound(vertex)<n2)) _die('buffer too small')

    if(op==0) then
      
      do mu=1,n(3)
        do o=1,n(2)
          vertex((o-1)*n(1)+1:o*n(1), mu) = &
            pb%sp_local2vertex(spp)%vertex(1:n(1), o, mu)
        enddo ! b 
      enddo ! mu
          
    else if (op==1) then
      
      do mu=1,n(3)
        do o=1,n(1)
          vertex((o-1)*n(2)+1:o*n(2), mu) = &
            pb%sp_local2vertex(spp)%vertex(o, 1:n(2), mu)
        enddo ! b 
      enddo ! mu

      ns = n(2); n(2) = n(1); n(1) = ns
    else
      _die('unknown op')
    endif
          
  else if (pb%book_dp(pair)%top==2) then

    n = ubound(pb%sp_biloc2vertex(spp)%vertex)
    if(any(n<1)) _die('any(n<1)')
    n2 = (/n(1)*n(2),n(3)/)
    if(any(ubound(vertex)<n2)) _die('buffer too small')
    
    if(op==0) then
        
      do mu=1,n(3)
        do o=1,n(2)
          vertex((o-1)*n(1)+1:o*n(1), mu) = &
            pb%sp_biloc2vertex(spp)%vertex(1:n(1), o, mu)
        enddo ! b 
      enddo ! mu

    else if(op==1) then

      do mu=1,n(3)
        do o=1,n(1)
          vertex((o-1)*n(2)+1:o*n(2), mu) = &
            pb%sp_biloc2vertex(spp)%vertex(o, 1:n(2), mu)
        enddo ! b 
      enddo ! mu

      ns = n(2); n(2) = n(1); n(1) = ns
    else
      _die('unknown op')
    endif
    
  else
    _die(' wrong pb%book(pair)%top ??')
  endif    
  
end subroutine !   get_vertex_pair_dp

!
! Reallocates if necessary and returns a vertex part corressponding to a given pair of atoms 
! This subroutine must be working only with dominant product basis
!
subroutine get_vertex_of_pair_alloc_dp(pb, pair, vertex_pair, n)
  !! external
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  real(8), allocatable, intent(inout) :: vertex_pair(:,:,:)
  integer, intent(inout) :: n(3)
  !! internal
  integer :: spp
  
  if( .not. allocated(pb%book_dp) ) _die('.not. allocated(pb%book_dp)' )
  if( pair<0 .or. pair>size(pb%book_dp)) _die('pair<0 .or. pair>size(pb%book_dp)')
  
  spp = pb%book_dp(pair)%spp
  
  n = 0
  if(pb%book_dp(pair)%top==1) then
    n = ubound(pb%sp_local2vertex(spp)%vertex)
  else if (pb%book_dp(pair)%top==2) then
    n = ubound(pb%sp_biloc2vertex(spp)%vertex)
  else
    _die(' wrong pb%book(pair)%top ??')
  endif    
  
  if(any(n<1)) _die('any(n<1)')
  
  if(.not. allocated(vertex_pair) ) then
    allocate(vertex_pair(n(1),n(2),n(3)))
  else
    if(any(n/=ubound(vertex_pair))) then
      deallocate(vertex_pair)
      allocate(vertex_pair(n(1),n(2),n(3)))
    endif  
  endif
  
  if(pb%book_dp(pair)%top==1) then
    vertex_pair = pb%sp_local2vertex(spp)%vertex
  else if (pb%book_dp(pair)%top==2) then
    vertex_pair = pb%sp_biloc2vertex(spp)%vertex
  else
    _die(' wrong pb%book(pair)%top ??')
  endif    
  
end subroutine !   get_vertex_of_pair_alloc_dp

!
!
!
subroutine get_vertex_dp_ptr_subr(pb, pair, vertex, n)
  implicit none
  !! external
  type(prod_basis_t), intent(in), target :: pb
  integer, intent(in) :: pair
  real(8), intent(out), pointer :: vertex(:,:,:)
  integer, intent(inout) :: n(3)
  !! internal
  type(book_pb_t) :: book
  
  book = get_book_dp(pb, pair)
  n = book%fi - book%si + 1
  if(book%top==1) then
    vertex => pb%sp_local2vertex(book%spp)%vertex
  else if (book%top==2) then
    vertex => pb%sp_biloc2vertex(book%spp)%vertex
  else 
    nullify(vertex)
    _die('unknown type_of_center')
  endif  
  
end subroutine ! get_vertex_dp_ptr_subr 

!
!
!
function get_vertex_dp_ptr(pb, pair) result(vertex)
  implicit none
  !! external
  type(prod_basis_t), intent(in), target :: pb
  integer, intent(in) :: pair
  real(8), pointer :: vertex(:,:,:)

  !! internal
  type(book_pb_t) :: book
  
  book = get_book_dp(pb, pair)

  if(book%top==1) then
    vertex => pb%sp_local2vertex(book%spp)%vertex
  else if (book%top==2) then
    vertex => pb%sp_biloc2vertex(book%spp)%vertex
  else 
    nullify(vertex)
    _die('unknown type_of_center')
  endif  
  
end function ! get_vertex_dp_ptr 

!
!
!
subroutine get_vertex_book_ptr_pp(pb, pair, vertex, book)
  implicit none
  !! external
  type(prod_basis_t), intent(in), target :: pb
  integer, intent(in) :: pair
  real(8), intent(inout), pointer :: vertex(:,:,:)
  type(book_pb_t), intent(inout), pointer :: book 
  !! internal

  book => get_book_dp_ptr(pb, pair)

  if(book%top==1) then
    vertex => pb%sp_local2vertex(book%spp)%vertex
  else if (book%top==2) then
    vertex => pb%sp_biloc2vertex(book%spp)%vertex
  else
    nullify(vertex)
    _die('unknown type_of_center')
  endif

end subroutine ! get_vertex_book_ptr_pp


!
!
!
function get_nfunct_reexpr(pb) result(nf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: ncoeffs, ic, nf1, nc_rspace, n
 
  nf = 0
  ncoeffs = 0
  if(allocated(pb%coeffs)) ncoeffs = size(pb%coeffs)
  if(ncoeffs<1) return

  ! Count number of functions used in reexpressed basis
  nf1 = 0
  nc_rspace = get_ncenters_re(pb)
  
  do ic=1,nc_rspace
    n = get_nfunct_per_book(pb, pb%book_re(ic))
    nf1 = nf1 + n
  enddo ! ic
  ! END of Count number of functions used in reexpressed basis
  nf = nf1

!
! WARNING: write(6,*) is not reentrant for a reason...
! NAMELY:  write(6,*) f(a) may stop if f(a) also uses write(6,*)...
!
end function !  get_nfunct_reexpr 

!
! Finds maximal number of reexpressing functions per pair
!
function get_nfunct_max_re_pp(pb) result(nf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: npairs, pair, ind, nf1, n, nind, ib
 
  nf = 0
  npairs = 0
  if(allocated(pb%coeffs)) npairs = size(pb%coeffs)
  if(npairs<1) return

  ! Count maximal number of reexpressing functions used for a pair
  nf1 = 0
  do pair=1,npairs
    if(.not. allocated(pb%coeffs(pair)%ind2book_re)) cycle

    nind = size(pb%coeffs(pair)%ind2book_re)
    n = 0
    do ind=1,nind
      ib = pb%coeffs(pair)%ind2book_re(ind)
      n = n + get_nfunct_per_book(pb, pb%book_re(ib))
    enddo
    nf1 = max(n, nf1)
  enddo ! ic
  ! END of Count maximal number of reexpressing functions used for a pair
  
  nf = nf1

end function ! get_nfunct_max_re_pp

!
!
!
function get_ncoeffs(pb) result(nf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  
  nf = 0
  if(allocated(pb%coeffs)) nf = size(pb%coeffs)

end function ! get_ncoeffs 

!
! Gets number of atom pairs irreducible by translation or inversion symmetries
! This number must be suitable for computing global kernels, or for counting
! of number of functions in the product basis, or for checking against 
! dipoles or overlaps.
!
function get_npairs(pb) result(nf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  
  !! internal
  nf = 0
  if(.not. allocated(pb%book_dp)) _die('.not. allocated(pb%book_dp)')
  nf = size(pb%book_dp)
  
end function ! get_npairs

!
! Gets number of centers in the mixed product basis.
!
function get_ncenters_re(pb) result(nf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: ncoeffs, nf1
  nf = 0
  ncoeffs = 0
  if(allocated(pb%coeffs)) ncoeffs = size(pb%coeffs)
  if(ncoeffs<1) return

  nf1 = 0
  if(allocated(pb%book_re)) nf1 = size(pb%book_re)
  if(nf1<1) _die('!nf1<1')
  nf = nf1
  
end function ! get_ncenters_re

!
! Get's information about a pair
!
function get_book_dp(pb, pair) result(book)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  type(book_pb_t) :: book
  !! internal
  if(.not. allocated(pb%book_dp)) _die('!book_dp')
  if(pair<1 .or. pair>size(pb%book_dp)) _die('!pair')
  book = pb%book_dp(pair)
end function ! get_book_dp

!
! Get's information about a pair
!
function get_book_dp_ptr(pb, pair) result(book)
  implicit none
  !! external
  type(prod_basis_t), intent(in), target :: pb
  integer, intent(in) :: pair
  type(book_pb_t), pointer :: book
  !! internal
  if(.not. allocated(pb%book_dp)) _die('!book_dp')
  if(pair<1 .or. pair>size(pb%book_dp)) _die('!pair')
  book => pb%book_dp(pair)
end function ! get_book_dp_ptr


!
! A formal function now. But maybe useful at later stages when it does some cross-checks
!
function get_isym(pb) result(isym)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: isym

  isym = pb%irr_trans_inv_sym
  
  if(isym<0 .or. isym>1) _die('isym<0 .or. isym>1')

end function ! get_isym   

!!
!!
!!
function get_nfunct_per_book(pb, book) result(nf)
  use m_functs_m_mult_type, only : get_nfunct_mmult
  use m_functs_l_mult_type, only : get_nfunct_lmult
  implicit none
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(in) :: book
  integer :: nf
  !! internal
  integer :: nf3
  nf = -1
  if(book%top==1) then
    nf = get_nfunct_lmult(pb%sp_local2functs(book%spp))
    !nf = pb%sp_local2functs(book%spp)%nfunct
  else if(book%top==2) then
    nf = get_nfunct_mmult(pb%sp_biloc2functs(book%spp))
    !nf = pb%sp_biloc2functs(book%spp)%nfunct
  else 
    _die('wrong book ? ')
  endif
  
  nf3 = book%fi(3)-book%si(3)+1
  if(nf/=nf3) then
    write(6,*) nf, nf3, book%spp, book%top, book%ic
    _die('nf/=nf3')
  endif  

end function ! get_nfunct_per_book

!
!
!
function get_nfunct(pb, type_of_center, spp) result(res)
  use m_functs_m_mult_type, only : get_nfunct_mmult
  use m_functs_l_mult_type, only : get_nfunct_lmult
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: type_of_center, spp
  integer :: res
  !! internal
  
  res = -999
  if(type_of_center==1) then
    if(.not. allocated(pb%sp_local2functs)) &
      _die('.not. allocated(pb%sp_local2functs)')
    if(spp<1 .or. spp>size(pb%sp_local2functs)) &
      _die('spp<1 .or. spp>size(pb%sp_local2functs)')
    res = get_nfunct_lmult(pb%sp_local2functs(spp))
    
  else if (type_of_center==2) then
    if(.not. allocated(pb%sp_biloc2functs)) &
      _die('.not. allocated(pb%sp_biloc2functs)')
    if(spp<1 .or. spp>size(pb%sp_biloc2functs))&
       _die('spp<1 .or. spp>size(pb%sp_biloc2functs)')
    res = get_nfunct_mmult(pb%sp_biloc2functs(spp))
    !res = pb%sp_biloc2functs(spp)%nfunct
    
  else
    _die('wrong type_of_center ??')
  endif    
  
end function !  get_nfunct


!!
!! index ==> start index correspondence
!!
subroutine get_i2s(pb, i2b, i2s)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(in) :: i2b(:)
  integer, intent(inout), allocatable :: i2s(:)
  !! internal
  integer :: ni, i, s, f, n
  
  ni = size(i2b)
  if(ni<1) _die('ni<1')
  _dealloc(i2s)
  allocate(i2s(ni+1))
  f = 0
  do i=1,ni
    s = f + 1 
    n = get_nfunct(pb, i2b(i)%top, i2b(i)%spp)
    f = s+n-1
    i2s(i) = s
  enddo
  i2s(ni+1) = f+1

end subroutine ! 


!
! Number of functions in dominant products
!
function get_nfunct_domiprod(pb) result(nf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf, nbook, ibook, nf1
  
  if(.not. allocated(pb%book_dp)) _die('allocated(pb%book_dp)')
  
  nf = pb%nfunct_irr
  ! Count number of dominant products
  nf1 = 0
  nbook = size(pb%book_dp)
  do ibook=1,nbook
    if(pb%book_dp(ibook)%top<0) cycle
    nf1 = nf1 + (pb%book_dp(ibook)%fi(3)-pb%book_dp(ibook)%si(3)+1)
  enddo
  ! END of Count number of dominant products
  if(nf1/=nf) then
    write(6,*) nf1, nf, nbook
    _die('nf1/=nf')
  endif  
  
end function !  get_nfunct_domiprod 

!
!
!
function get_pair_type(pb, pair) result(pt)
  use m_book_pb, only : get_pair_type_book
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  integer :: pt
  !! internal
  
  if(.not. allocated(pb%book_dp)) &
    _die('.not. allocated(pb%book_dp)')
  if(pair<1 .or. pair>size(pb%book_dp)) &
    _die('pair<1 .or. pair>size(pb%book_dp)')
  
  pt = get_pair_type_book(pb%book_dp(pair))

end function ! get_pair_type  

!
! Provides the type of pair in domiprod bookkeeping
!
function get_top_dp(pb, pair) result(top)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  integer :: top
  !! internal
  
  if(.not. allocated(pb%book_dp)) &
    _die('.not. allocated(pb%book_dp)')
  if(pair<1 .or. pair>size(pb%book_dp)) &
    _die('pair<1 .or. pair>size(pb%book_dp)')
  
  top = pb%book_dp(pair)%top

end function ! get_top_dp


!
!
!
function get_coeffs_pp_ptr(pb, pair) result(coeffs)
  implicit none
  !! external
  type(prod_basis_t), intent(in), target :: pb
  integer, intent(in) :: pair
  real(8), pointer :: coeffs(:,:)
  !! internal
  integer :: npairs 
  npairs = get_npairs(pb)
  if(pair<1 .or. pair>npairs) _die('!pair')
  if(.not. allocated(pb%coeffs)) _die('!%coeffs')
  coeffs => pb%coeffs(pair)%coeffs_ac_dp
  
end function ! get_coeffs_pp_ptr  

!
! Get start and finish product functions for a dominant product basis
! local counting is first two entries, global counting second two 
! 
function get_sfp_domiprod(pb, pair) result(sfsf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  integer :: sfsf(1:4)
  !! internal
  integer :: npairs
  
  npairs = get_npairs(pb)
  if(pair<1 .or. pair>npairs) _die('!pair')

  sfsf(3) = pb%book_dp(pair)%si(3)
  sfsf(4) = pb%book_dp(pair)%fi(3)

  sfsf(1) = 1
  sfsf(2) = sfsf(4) - sfsf(3) + 1

end function ! get_sfp_domiprod 

!
! Get start and finish product functions for a mixed basis
! local counting is first two entries, global counting second two 
!  
function get_sfp_mixed(pb, pair, ic) result(sfsf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair, ic
  integer :: sfsf(1:4)
  !! internal
  integer :: npairs, ib
  
  npairs = get_npairs(pb)
  if(pair<1 .or. pair>npairs) _die('!pair')
  if(.not. allocated(pb%coeffs)) _die('!%coeffs')
  if(.not. allocated(pb%coeffs(pair)%ind2book_re)) &
_die('!%coeffs(pair)%ind2book_re')
  if(.not. allocated(pb%book_re)) _die('%book_re')
  
  if(ic<1 .or. ic>size(pb%coeffs(pair)%ind2book_re)) _die('!ic')
  ib = pb%coeffs(pair)%ind2book_re(ic)
  if(ib<1 .or. ib>size(pb%book_re)) _die('!ib')
  
  sfsf(3) = pb%book_re(ib)%si(3)
  sfsf(4) = pb%book_re(ib)%fi(3)
  sfsf(1) = pb%coeffs(pair)%ind2sfp_loc(1,ic)
  sfsf(2) = pb%coeffs(pair)%ind2sfp_loc(2,ic)

end function ! get_sfp_mixed

!
! Determines if a pair is bilocal or not  
!
logical function get_is_bilocal(pb, pair)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  !! internal
  integer :: npairs
  
  npairs = get_npairs(pb)
  get_is_bilocal = .false.
  if(pair<1 .or. pair>npairs) _die('!pair')
  if(pb%book_dp(pair)%top==2) then
    get_is_bilocal = .true.
  else if(pb%book_dp(pair)%top==1) then 
    get_is_bilocal = .false.
  else
    _die('!%top')
  endif  
end function ! get_is_bilocal

!
! Function delivers type of currently active bookkeping 
!
function get_book_type(pb) result(itype)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: itype
  !! internal
  itype = pb%book_type
  if(itype==1) then
    !! Dominant product's bookkeeping of product basis centers/functions/species
    if(.not. allocated(pb%book_dp)) _die('.not. allocated(pb%book_dp)')
    
  else if (itype==2) then
    !! Reexpressing basis (MIXED) bookkeping of product basis centers/functions/species
    if(.not. allocated(pb%book_dp)) _die('.not. allocated(pb%book_dp)')
    if(.not. allocated(pb%book_re)) _die('.not. allocated(pb%book_re)')
    if(.not. allocated(pb%coeffs)) _die('.not. allocated(pb%coeffs)')

  else
    write(0,'(a43,i5)') 'pb%book_type', pb%book_type
    write(0,'(a43,l5)') 'allocated(pb%book_dp)', allocated(pb%book_dp)
    write(0,'(a43,l5)') 'allocated(pb%coeffs)', allocated(pb%coeffs)
    write(0,'(a43,l5)') 'allocated(pb%book_re)', allocated(pb%book_re)
    write(0,'(a43,l5)') 'allocated(pb%book_uc)', allocated(pb%book_uc)    
    _die('unknown situation')  
  endif  
  
end function !  get_book_type 

!
! Subroutine determines which type of bookkeping will be used further
!
subroutine set_book_type(itype, pb)
  implicit none
  !! external
  integer, intent(in) :: itype
  type(prod_basis_t), intent(inout) :: pb
  !! internal
  integer :: itype_chk

  pb%book_type = itype
  itype_chk = get_book_type(pb)

end subroutine ! set_book_type

!
! the correspondence pair2nprod for dominant products
! 
subroutine get_pair2nprod_dp(pb, pair2nprod)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(inout), allocatable :: pair2nprod(:)
  !! internal
  integer :: npairs, pair
  
  npairs = get_npairs(pb)
  _dealloc(pair2nprod)
  allocate(pair2nprod(npairs))
  pair2nprod = -999
  do pair=1,npairs
    pair2nprod(pair) = pb%book_dp(pair)%fi(3) - pb%book_dp(pair)%si(3) + 1
  enddo
  if(any(pair2nprod<1)) _die('!pair2nprod<1')

end subroutine ! get_pair2nprod_dp


!
! the correspondence product center -> number of product functions
! for mixed (Reexpressing) basis
! 
subroutine get_pc2npf_re(pb, pc2npf)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(inout), allocatable :: pc2npf(:)
  !! internal
  integer :: npc, pc
  
  npc = get_ncenters_re(pb)
  if(npc<1) _die('npc<1')
  _dealloc(pc2npf)
  allocate(pc2npf(npc))
  pc2npf = -999
  do pc=1,npc
    pc2npf(pc) = pb%book_re(pc)%fi(3) - pb%book_re(pc)%si(3) + 1
  enddo ! pc
  if(any(pc2npf<1)) _die('!pc2npf<1')

end subroutine ! get_pc2npf_re


!
! This delivers the number of non-zero matrix elements in the 
! conversion matrix between dominant products and atom-centered products
!
integer(8) function get_nnonzero_cm(pb)
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  !! internal
  integer :: np, p

  if(.not. allocated(pb%coeffs)) then
    get_nnonzero_cm = get_nfunct_domiprod(pb)
    return
  endif  
  
  get_nnonzero_cm = 0
  np = size(pb%coeffs)
  do p=1,np
    get_nnonzero_cm = get_nnonzero_cm + size(pb%coeffs(p)%coeffs_ac_dp)
  enddo ! p
  
end function ! get_nnonzero_cm

end module !m_prod_basis_type
