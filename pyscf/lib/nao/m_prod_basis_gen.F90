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

module m_prod_basis_gen

! The purpose of the module is to store and deal with a real space information of a product basis
#include "m_define_macro.F90"
  use m_die, only : die
  
  implicit none
  private die

  contains

!
!
!
subroutine get_rcuts2(pb, ic, rcut2_1, rcut2_2)
  use m_prod_basis_type, only : book_pb_t, prod_basis_t
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ic
  real(8), intent(inout) :: rcut2_1, rcut2_2
  !! internal
  type(book_pb_t) :: book
  
  book = get_book(pb, ic)
  
  if(book%top==1) then
    rcut2_1 = pb%sp_local2functs(book%spp)%rcut**2
    rcut2_2 = rcut2_1
    
  else if (book%top==2) then
    rcut2_1 = pb%sp_biloc2functs(book%spp)%rcuts(1)**2
    rcut2_2 = pb%sp_biloc2functs(book%spp)%rcuts(2)**2
  else
    _die('unknown type of center')
  endif
    
end subroutine ! get_rcuts2

!
!
!
subroutine get_coords(pb, ic, coord1, coord2)
  use m_prod_basis_type, only : book_pb_t, prod_basis_t
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ic
  real(8), intent(inout) :: coord1(3), coord2(3)
  !! internal
  type(book_pb_t) :: book
  
  book = get_book(pb, ic)
  
  if(book%top==1) then
    coord1 = book%coord
    coord2 = coord1
  else if (book%top==2) then
    coord1 = pb%sp_biloc2functs(book%spp)%coords(:,1)
    coord2 = pb%sp_biloc2functs(book%spp)%coords(:,2)
  else
    _die('unknown type of center')
  endif
    
end subroutine ! get_coords 


!
! This subroutine returns number of centers which serve to express products of orbitals within a pair of atoms
!
function get_ncenters_for_a_pair(pb, pair) result(nc)
  use m_prod_basis_type, only : prod_basis_t, get_book_type
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  integer :: nc
  !! internal
  integer :: book_type
  
  book_type = get_book_type(pb)
  nc = -1
  if(book_type==1) then
    nc = 1
  else if (book_type==2) then
    nc = 0
    if(.not. allocated(pb%coeffs)) &
      _die('.not. allocated(pb%coeffs)')
    if(allocated(pb%coeffs(pair)%ind2book_re)) &
      nc = size(pb%coeffs(pair)%ind2book_re)
  else
    _die('unknown booking type')  
  endif  

end function ! get_ncenters_for_a_pair

!
! This subroutine returns number of centers which serve to express products of orbitals within a pair of atoms
! per pair for a custom bookeeping type
!
function get_ncenters_pp_bt(pb, pair, ibook_type) result(nc)
  use m_prod_basis_type, only : prod_basis_t
  use m_coeffs_type, only : get_nind
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair, ibook_type
  integer :: nc
  !! internal
 
  nc = -1
  select case(ibook_type)
  case(1)
    nc = 1
    
  case(2)
    if(.not. allocated(pb%coeffs)) _die('!%coeffs')
    nc = get_nind(pb%coeffs(pair))
  
  case default 
    write(6,*) 'ibook_type', ibook_type
    _die('unknown book_type')  
  end select  

end function ! get_ncenters_pp_bt


!
! Returns a vertex for a given pair -- general.
!
subroutine get_vertex_pair(pb, pair, op, vertex, n, vertex_dp_aux)
  use m_precision, only : blas_int
  use m_prod_basis_type, only : prod_basis_t, get_vertex_pair_dp, get_book_type
  !! external
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair, op
  real(8), intent(inout) :: vertex(:,:), vertex_dp_aux(:,:)
  integer, intent(inout) :: n(3)
  !! internal
  integer :: bt, ndp(3)
  integer(blas_int) :: lda, ldd, n12
  
  
  bt = get_book_type(pb)
  
  if(bt==1) then
    
    call get_vertex_pair_dp(pb, pair, op, vertex, n)
    
  else if (bt==2) then
    if(.not. allocated(pb%coeffs)) _die('mix !generat.')
    call get_vertex_pair_dp(pb, pair, op, vertex_dp_aux, ndp)
    n = ndp
    n(3) = size(pb%coeffs(pair)%coeffs_ac_dp,1)
    n12 = n(1)*n(2)
    ldd = size(vertex_dp_aux,1)
    lda = size(vertex,1)
    call DGEMM('N', 'T', n12,n(3),ndp(3), 1d0, vertex_dp_aux, ldd, &
      pb%coeffs(pair)%coeffs_ac_dp,n(3), 0d0, vertex, lda)
  else
    _die('unknown book_type')
  endif  

end subroutine ! get_vertex_pair

!
! Returns a vertex for a given pair and given bookkeeping type (i.e. domiprod or mixed)
!
subroutine get_vertex_pair_bt(pb, pair, op, ibook_type, vertex, n, vertex_dp_aux)
  use m_precision, only : blas_int
  use m_prod_basis_type, only : prod_basis_t, get_vertex_pair_dp
  !! external
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair, op, ibook_type
  real(8), intent(inout) :: vertex(:,:), vertex_dp_aux(:,:)
  integer, intent(inout) :: n(3)
  !! internal
  integer :: ndp(3)
  integer(blas_int) :: lda, ldd, n12
  
  select case (ibook_type)
  
  case(1)
    
    call get_vertex_pair_dp(pb, pair, op, vertex, n)
  
  case(2)   
  
    call get_vertex_pair_dp(pb, pair, op, vertex_dp_aux, ndp)
    n = ndp
    n(3) = size(pb%coeffs(pair)%coeffs_ac_dp,1)
    n12 = n(1)*n(2)
    ldd = size(vertex_dp_aux,1)
    lda = size(vertex,1)
    call DGEMM('N', 'T', n12,n(3),ndp(3), 1d0, vertex_dp_aux, ldd, &
      pb%coeffs(pair)%coeffs_ac_dp,n(3), 0d0, vertex, lda)
  case default
    write(6,*) 'book_type', ibook_type
    _die('unknown book_type')
  end select

end subroutine ! get_vertex_pair_bt

!
! Reallocates if necessary and returns a vertex part corressponding to a given pair of atoms 
! This subroutine must be working also for atom-centered or mixed product basis
!
subroutine get_vertex_of_pair_alloc(pb, pair, vertex_pair, n)
  use m_prod_basis_type, only : get_vertex_of_pair_alloc_dp, prod_basis_t
  use m_prod_basis_type, only : get_book_type
  !! external
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  real(8), allocatable, intent(inout) :: vertex_pair(:,:,:)
  integer, intent(inout) :: n(3)
  !! internal
  integer :: bt, ndp(3), n12
  real(8), allocatable :: vertex_dp(:,:,:)
  
  call get_vertex_of_pair_alloc_dp(pb, pair, vertex_dp, ndp)
  bt = get_book_type(pb)
  if(bt==1) then
    n = ndp
  else if (bt==2) then
    n = ndp
    n(3) = size(pb%coeffs(pair)%coeffs_ac_dp,1)
    if(size(pb%coeffs(pair)%coeffs_ac_dp,2)/=ndp(3)) &
      _die('! consistent')
  else
    _die('unknown book_type')
  endif  
  
  if(.not. allocated(vertex_pair) ) then
    allocate(vertex_pair(n(1),n(2),n(3)))
  else
    if(any(n/=ubound(vertex_pair))) then
      deallocate(vertex_pair)
      allocate(vertex_pair(n(1),n(2),n(3)))
    endif
  endif

  if(bt==1) then
    vertex_pair = vertex_dp
  else if (bt==2) then
    n12 = n(1)*n(2)
    call DGEMM('N', 'T', n12, n(3), ndp(3), 1d0, vertex_dp, n12, &
      pb%coeffs(pair)%coeffs_ac_dp, n(3), 0d0, vertex_pair, n12)
  else
    _die('unknown book_type')
  endif  


end subroutine ! get_vertex_of_pair_alloc


!
! Finds maximal number of functions per pair
!
function get_nfunct_max_per_pair(pb) result(nf)
  use m_prod_basis_type, only : prod_basis_t, get_nfunct_max_re_pp
  use m_prod_basis_type, only : get_nfunct_max_pp, get_book_type

  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: bt
   
  bt = get_book_type(pb) 
  nf = -1
  if(bt==1) then
    nf = get_nfunct_max_pp(pb)
  else if (bt==2) then
    nf = get_nfunct_max_re_pp(pb)
  else 
    _die('unknown bt')
  endif  
  
end function ! get_nfunct_max_per_pair

!
! Finds maximal number of functions per pair
!
function get_nfunct_max_pp_bt(pb, ibook_type) result(nf)
  use m_prod_basis_type, only : prod_basis_t
  use m_prod_basis_type, only : get_nfunct_max_pp
  use m_prod_basis_type, only : get_nfunct_max_re_pp
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ibook_type
  integer :: nf
  !! internal

  nf = -1
  select case(ibook_type)
  case(1)
    nf = get_nfunct_max_pp(pb)
  case(2)
    nf = get_nfunct_max_re_pp(pb)
  case default 
    write(6,*) 'book_type ', ibook_type
    _die('unknown book_type')
  end select 
  
  if(nf<1) _die('!get_nfunct_max_pp_bt')
  
end function ! get_nfunct_max_pp_bt

!
! Gets number of parts of the product decomposition as fixed by current bookkeeping
! bookkeping record refers to a part of product decomposition with 
! atom(pair) resolution.
!
function get_nbook(pb) result(nf)
  use m_prod_basis_type, only : prod_basis_t, get_npairs, get_ncenters_re
  use m_prod_basis_type, only : get_book_type
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf
  !! internal
  integer :: ibook_type
  nf = 0
  ibook_type = get_book_type(pb)
  nf = get_nbook_bt(pb, ibook_type)
  
end function ! get_nbook

!
! Gets number of parts of the product decomposition as fixed by custom bookkeeping
!
integer function get_nbook_bt(pb, ibook_type)
  use m_prod_basis_type, only : prod_basis_t, get_npairs, get_ncenters_re
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ibook_type

  get_nbook_bt = 0
  
  select case(ibook_type)
  case(1)
    get_nbook_bt = get_npairs(pb)
  case(2) 
    get_nbook_bt = get_ncenters_re(pb)
  case default 
    write(6,*) 'book_type ', ibook_type
    _die('!book_type')
  end select
  
end function ! get_nbook_bt


!
! Get's information about a center via pointing to a pair 
! and to a center within this pair (relevant for atom-centered decomposition)
!
function get_book_pair_ind(pb, pair, ind) result(book)
  use m_prod_basis_type, only : prod_basis_t, book_pb_t, get_book_type
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair, ind
  type(book_pb_t) :: book
  !! internal
  integer :: ibook_type
  
  ibook_type = get_book_type(pb)
  
  if(ibook_type==1) then;
    if(ind/=1) _die('ind/=1')
    book = get_book(pb, pair)
  else if (ibook_type==2) then;
    if(.not. allocated(pb%coeffs)) &
      _die('.not. allocated(pb%coeffs)')
    if(pair>size(pb%coeffs)) _die('pair>size(pb%coeffs)')
    if(ind<1)_die('ind<1')
    if(ind>size(pb%coeffs(pair)%ind2book_re)) &
      _die('ind>size')
    book = get_book(pb, pb%coeffs(pair)%ind2book_re(ind))
  else
    write(6,*) 'ibook_type', ibook_type
    _die('unknown situation')
  endif
  
end function ! get_book_pair_ind

!
! Get's information about a center via pointing to a pair and 
! to a center within this pair for a given bookkeping type
!
function get_book_bt_pair_ind(pb, ibook_type, pair, ind) result(book)
  use m_prod_basis_type, only : prod_basis_t, book_pb_t, get_book_dp
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ibook_type, pair, ind 
  type(book_pb_t) :: book
  !! internal

  select case(ibook_type)
  case(1) 
    if(ind/=1) _die('ind/=1')
    book = get_book_dp(pb, pair)

  case(2) 
  
    if(.not. allocated(pb%coeffs)) &
      _die('.not. allocated(pb%coeffs)')
    if(pair>size(pb%coeffs)) _die('pair>size(pb%coeffs)')
    if(ind<1)_die('ind<1')
    if(ind>size(pb%coeffs(pair)%ind2book_re)) &
      _die('ind>size')
    book = get_book(pb, pb%coeffs(pair)%ind2book_re(ind))
  case default 
    write(6,*) 'book_type', ibook_type
    _die('!book_type')
  end select
  
end function ! get_book_pair_ind_bt

!
!
!
function get_book(pb, ic) result(book)
  use m_prod_basis_type, only : prod_basis_t, book_pb_t, get_book_type
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ic
  type(book_pb_t) :: book
  !! internal
  integer :: ibook_type
  ibook_type = get_book_type(pb)
  
  if(ibook_type==1) then;
    if(.not. allocated(pb%book_dp)) _die('???')
    if(ic>size(pb%book_dp) .or. ic<1) then
      write(0,*) 'ic', ic
      write(0,*) 'size(pb%book_dp)', size(pb%book_dp)
       _die('boundary is violated')
    endif
    book = pb%book_dp(ic);

  else if (ibook_type==2) then;
    if(.not. allocated(pb%book_re)) _die('%book_re')
    if(.not. allocated(pb%book_dp)) _die('%book_dp')

    if(ic>size(pb%book_re) .or. ic<1) then
      write(0,*) 'ic', ic
      write(0,*) 'size(pb%book_re)', size(pb%book_re)
       _die('boundary is violated')
    endif
    book = pb%book_re(ic)

  else
    write(6,*) 'ibook_type', ibook_type
    _die('unknown situation')
  endif
  
end function ! get_book

!
! Get's the bookkeping information for a give bookkeping type (i.e. for dominant products or mixed products)
!
function get_book_bt(pb, ic, ibook_type) result(book)
  use m_prod_basis_type, only : prod_basis_t, book_pb_t
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ic, ibook_type
  type(book_pb_t) :: book
  !! internal
  
  select case(ibook_type) 
  
  case(1)
    if(.not. allocated(pb%book_dp)) _die('???')
    if(ic>size(pb%book_dp) .or. ic<1) then
      write(0,*) 'ic', ic
      write(0,*) 'size(pb%book_dp)', size(pb%book_dp)
       _die('boundary is violated')
    endif
    book = pb%book_dp(ic);

  case(2)
  
    if(.not. allocated(pb%book_re)) _die('???')
    if(.not. allocated(pb%book_dp)) _die('???')

    if(ic>size(pb%book_re) .or. ic<1) then
      write(0,*) 'ic', ic
      write(0,*) 'size(pb%book_re)', size(pb%book_re)
       _die('boundary is violated')
    endif
    book = pb%book_re(ic)

  case default 
    write(6,*) 'ibook_type', ibook_type
    _die('!book_type')
  end select ! bt
  
end function ! get_book_bt

!
! Get's information about a center via pointing to a pair and to a center within this pair
!
integer function get_ibook_bt_pair_ind(pb, ibook_type, pair, ind)
  use m_prod_basis_type, only : prod_basis_t, get_npairs
  use m_coeffs_type, only : get_nind
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ibook_type, pair, ind
  !! internal
  integer :: npairs, nind
  
  npairs = get_npairs(pb)  
  if(pair<1 .or. pair>npairs) _die('!pair')  
  
  select case (ibook_type)
  case(1)
    if(ind/=1) then
      write(6,*) 'ind ', ind
      _die('ind/=1')
    endif  
    get_ibook_bt_pair_ind = pair
    
  case(2)
    
    nind = get_nind(pb%coeffs(pair))
    if(ind<1 .or. ind>nind) _die('!ind')  
    
    get_ibook_bt_pair_ind = pb%coeffs(pair)%ind2book_re(ind)
    
  case default 
    write(6,*) 'book_type ', ibook_type
    _die('!book_type')
  end select ! ibook_type
  
end function ! get_ibook_bt_pair_ind

!!
!! Number of functions for a given bookkeeping index (i.e. for a given part of product decomposition)
!! nfunct p[er] b[ook]
!!
function get_nfunct_pbook(pb, ic) result(nf)
  use m_functs_m_mult_type, only: get_nfunct_mmult
  use m_functs_l_mult_type, only: get_nfunct_lmult
  use m_prod_basis_type, only : prod_basis_t, book_pb_t
  implicit none
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ic
  integer :: nf
  !! internal
  type(book_pb_t) :: book
  
  book = get_book(pb, ic)
  
  nf = -1
  if(book%top==1) then
  
    nf = get_nfunct_lmult(pb%sp_local2functs(book%spp))
    
  else if(book%top==2) then
  
    nf = get_nfunct_mmult(pb%sp_biloc2functs(book%spp))
    
  else 
    !! call print_info_book_elem(6, book)
    _die('wrong book(ic)%top ? ')
  endif    

end function ! get_nfunct_pbook 

!
! Number of functions in product basis 
!
function get_nfunct_prod_basis(pb) result(nf)
  use m_prod_basis_type, only : prod_basis_t
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer :: nf, nc, ic, nf1
  
  nc = get_nbook(pb)  
  nf1 = 0
  do ic=1,nc; 
    nf1 = nf1 + get_nfunct_pbook(pb, ic);
  enddo; ! Count number of functions in product basis
  nf = nf1
  
  if(nf1/=nf) _die('nf1/=nf')
  
end function !  get_nfunct_prod_basis 

!
!
!
function get_ncenters_pbook(pb, ib) result(nf)
  use m_functs_m_mult_type, only : get_ncenters
  use m_prod_basis_type, only : prod_basis_t, book_pb_t 
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ib
  integer :: nf
  !! internal
  type(book_pb_t) :: book
     
  book = get_book(pb, ib)
  nf = -1
  if(book%top==1) then
  
    nf = 1
    
  else if(book%top==2) then
  
    nf = get_ncenters(pb%sp_biloc2functs(book%spp))
    
  else 
    !! call print_info_book_elem(6, book)
    _die('wrong book(ic)%top ? ')
  endif    
  
end function ! get_ncenters_pbook

!
!
!
function get_center(pb, ibook, ic_wb) result(center)
  use m_book_pb, only : book_pb_t
  use m_prod_basis_type, only : prod_basis_t
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: ibook, ic_wb
  real(8) :: center(1:3)
  type(book_pb_t) :: book
 
  book = get_book(pb, ibook)
  
  if(book%top==1) then
  
    center = book%coord
    
  else if(book%top==2) then

    if(ic_wb<1) _die('ic_wb<1')  
    if(ic_wb>1) _die('not impl.')
    
    center = book%coord
    
  else 
    center = -999
    !! call print_info_book_elem(6, book)
    _die('wrong book(ic)%top ? ')
  endif    
 
end function !get_center


end module !m_prod_basis_gen
