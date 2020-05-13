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

module m_book_pb
! The purpose of the module is to store and deal with a real space information of a product basis
#include "m_define_macro.F90"
  use m_die, only : die
  
  implicit none
  integer, parameter :: METRIC_SIZE=8
  integer, parameter :: SAME_METRIC_SIZE=19

  !! Center to coordinate, shifts and specie pointer
  type book_pb_t
    integer :: ic=-999  ! number of the center in a global counting (A GLOBAL COUNTING is not necessarily THE GLOBAL COUNTING you would need...)
    integer :: top=-999 ! type of pair: 1 -- local, 2 -- bilocal pairs
    integer :: spp=-999 ! pointer to a product specie in a list of product species
    real(8) :: coord(3)=-999   ! real space coordinate of the center without translations
    integer :: cells(3,3)=-999 ! cells(1..3) number of unit cells along 1..3 direction on which the centers are shifted relative to 'first' unit cell
    integer :: atoms(2)=-999   ! atoms (in the unit cell) for which the pair is created
    integer :: si(3)=-999 !  start indices of the part in the global counting of real space degrees of freedom for atoms and prod basis
    integer :: fi(3)=-999 ! finish indices of the part in the global counting of real space degrees of freedom for atoms and prod basis
  end type ! book_pb_t

  type book_array1_t 
    type(book_pb_t), allocatable :: a(:)
  end type !book_array1_t

  contains

!
!
!
function get_nfunct3(b) result(n)
  implicit none
  !! external
  type(book_pb_t), allocatable, intent(in) :: b(:)
  integer :: n
  !! internal
  integer :: npairs, p

  _size_alloc(b,npairs)
  n = 0
  do p=1,npairs
    select case(b(p)%top)
    case(-1); cycle
    case(1:2); continue
    case default; write(6,*) b(p)%top, p; _die('!type of pair?')
    end select
    if(b(p)%fi(3)<1) _die('b(p)%fi(3)<1')
    if(b(p)%si(3)<1) _die('b(p)%si(3)<1')
    n = n + (b(p)%fi(3)-b(p)%si(3)+1)
  enddo ! i 

end function ! get_nfunct3

!
! The essential information which allows to distinguish one product center from the other.
! written in a simple double precision array.
!
function  get_same_metric(b) result(metric)
  implicit none
  !! external
  type(book_pb_t), intent(in) :: b ! book entry
  real(8) :: metric(SAME_METRIC_SIZE)

  metric = (/ b%top*1D0, b%coord, b%cells(1:3,1:3)*1D0, &
    b%atoms(1:2)*1D0, b%si(1:2)*1D0, b%fi(1:2)*1D0/)

end function ! get_same_metric

!
!
!
integer function get_n3(b)
  type(book_pb_t), intent(in) :: b
  if(b%si(3)<0 .or. b%fi(3)<0) _die('! si fi?')
  get_n3 = b%fi(3) - b%si(3) + 1
end function ! get_n3  
  
!
! Compatibility (geometrical properties) cross check of pair2clist_adj and pair2clist_mix
! One list (pair2clist_adj) contains rather auxiliary bookkeeping of 
! dominant product's centers together with centres that will be used to reexpress
! dominant products. Second list contains only bookkeeping of reexpressing centres.
! This second bookkeeping will be used later and it is easier to use than
! pointing to adjoint bookeeping and finding correspondence in mixed bookkeeping.
! 
subroutine compat_check_book_adj_mix(pair2clist_adj,book_adj, pair2clist_mix,book_mix,iv)
  use m_log, only : log_size_note
  implicit none
  !! external
  type(book_array1_t), allocatable, intent(in) :: pair2clist_adj(:), pair2clist_mix(:)
  type(book_pb_t), allocatable, intent(in) :: book_adj(:), book_mix(:)
  integer, intent(in) :: iv
  !! internal
  integer :: npairs1, npairs2, pair, ind, nind1, nind2, sofm, ic1, ic2
  real(8), allocatable :: metric1(:), metric2(:), metric3(:), metric4(:)
  
  if(.not. allocated(pair2clist_adj)) &
    _die('.not. allocated(pair2clist_adj)')
  if(.not. allocated(pair2clist_mix)) &
    _die('.not. allocated(pair2clist_mix)')
  if(.not. allocated(book_adj)) &
    _die('.not. allocated(book_adj)')
  if(.not. allocated(book_mix)) &
    _die('.not. allocated(book_mix)')
    
  npairs1 = size(pair2clist_adj)
  npairs2 = size(pair2clist_mix)
  if(npairs1/=npairs2) _die('npairs1/=npairs2')
  
  sofm = get_geom_metric_size()
  allocate(metric1(sofm))
  allocate(metric2(sofm))
  allocate(metric3(sofm))
  allocate(metric4(sofm))
  
  do pair=1,npairs1
    if(.not. allocated(pair2clist_adj(pair)%a)) &
      _die('.not. allocated(pair2clist_adj(pair)%a)')
    if(.not. allocated(pair2clist_mix(pair)%a)) &
      _die('.not. allocated(pair2clist_mix(pair)%a)')
    nind1 =  size(pair2clist_adj(pair)%a)
    nind2 =  size(pair2clist_mix(pair)%a)
    if(nind1/=nind2) _die('nind1/=nind2')
    do ind=1,nind1
      metric1 = get_geom_metric(pair2clist_adj(pair)%a(ind))
      metric2 = get_geom_metric(pair2clist_mix(pair)%a(ind))
      if(any(metric1/=metric2)) then
        _die('any(metric1/=metric2)')
      endif
      ic1 = pair2clist_adj(pair)%a(ind)%ic
      ic2 = pair2clist_mix(pair)%a(ind)%ic
      if(ic1<0 .or. ic1>size(book_adj)) &
        _die('ic1<0 .or. ic1>size(book_adj)')
      if(ic2<0 .or. ic2>size(book_mix)) &
        _die('ic2<0 .or. ic2>size(book_mix)')
      metric3 = get_geom_metric(book_adj(ic1))
      metric4 = get_geom_metric(book_mix(ic2))
      if(any(metric2/=metric3)) then
        _die('any(metric2/=metric3)')
      endif
      if(any(metric3/=metric4)) then
        _die('any(metric3/=metric4)')
      endif

    enddo ! ind 
  enddo ! pair
  
  call log_size_note('compat_check_book_adj_mix... ', 'passed.', iv)
  !! Test passed!
end subroutine ! compat_check_book_adj_mix


!
!
!
function get_R2mR1(book1, book2, uc_vecs) result(res)
  implicit none
  ! external
  type(book_pb_t), intent(in) :: book1, book2
  real(8), intent(in) :: uc_vecs(3,3)
  real(8) :: res(3)
  ! internal
  real(8) :: svecs(3,2), cells(3,2)

  cells(1:3,1) = book1%cells(:,3)
  cells(1:3,2) = book2%cells(:,3)

  svecs = matmul(uc_vecs(1:3,1:3), cells(1:3,1:2))

  res = book2%coord + svecs(:,2) - book1%coord - svecs(:,1)
end function ! get_R2mR1

!
!
!
integer function get_geom_metric_size()
  get_geom_metric_size =  METRIC_SIZE
end function ! get_geom_metric_size


!
! Gets a list of essential metrics from an array of book_pb_t elements
!
subroutine get_list_essent_metrics(book, ls)
  implicit none
  !! external
  type(book_pb_t), intent(in), allocatable :: book(:)
  real(8), intent(inout), allocatable :: ls(:,:)
  !! internal
  integer :: i, n

  if(.not. allocated(book)) &
    _die('.not. allocated(book)')

  n = size(book)
  _dealloc(ls)
  allocate(ls(METRIC_SIZE,n))
  do i=1,n
    ls(:,i) = get_geom_metric(book(i))
  enddo

end subroutine ! get_list_essent_metrics


!
! The essential information which allows to distinguish one product center from the other.
! written in a simple double precision array.
!
function  get_geom_metric(be) result(metric)
  implicit none
  !! external
  type(book_pb_t), intent(in) :: be ! book entry
  real(8) :: metric(METRIC_SIZE)

  metric = (/ be%top*1D0, be%spp*1D0, be%coord, be%cells(:,3)*1D0 /)

end function ! get_geom_metric



!
!
!
function get_pair_type_book(book) result(itype)
  implicit none
  !! external
  type(book_pb_t), intent(in) :: book
  integer :: itype
  !! intent
  
  itype = -1
  if(  &
    book%atoms(1) == book%atoms(2) .and. all(book%cells(:,1) == book%cells(:,2)) ) then ! This must be local atom pair
    itype = 1;
  else if ( &
    (book%atoms(1) /=  book%atoms(2) .and. all(book%cells(:,1) == book%cells(:,2)) ) .or. &
    any(book%cells(:,1)/=book%cells(:,2)) ) then ! This must be a bilocal atom pair
    itype = 2;
  else
    _die('unknown pair type.')
  endif     

end function ! get_pair_type  

!!
!! 
!!
subroutine sort_book(book_unsort, book)
  implicit none
  !! external
  type(book_pb_t), intent(in), allocatable :: book_unsort(:)  
  type(book_pb_t), intent(inout), allocatable :: book(:)
  !! internal
  integer :: nbook, i, pair
  
  if(.not. allocated(book_unsort)) _die('.not. allocated(book_unsort)')
  nbook = size(book_unsort)

  !! Sort the pairs that local pairs appear first
  _dealloc(book)
  allocate(book(nbook))
  i = 0
  do pair=1,nbook
    if(book_unsort(pair)%top/=1) cycle
    i = i + 1
    book(i) = book_unsort(pair)
  enddo

  do pair=1,nbook
    if(book_unsort(pair)%top/=2) cycle
    i = i + 1
    book(i) = book_unsort(pair)
  enddo
  !! END of Sort the pairs that local pairs appear first
    
end subroutine ! sort_book_dp

!
!
!
subroutine print_info_book(ifile, book)
  implicit none
  !! external
  integer, intent(in) :: ifile
  type(book_pb_t), intent(in), allocatable :: book(:)
  !! internal
  integer(8) :: i, n
  
  if(.not. allocated(book)) then
    write(ifile,'(a,i5,a35)') __FILE__, __LINE__, '.not. allocated(book)==>return '
    return
  endif
  
  n = size(book)
  write(ifile, '(a8,2x,3a5,2x,3a10,2x,a10,2x,a9,1x,a9,1x,a9,2x,a21,2x,a21)') &
    'i', 'ic ', 'tofc', 'spp', 'coord(1)', 'coord(2)', 'coord(3)', &
    'atoms(1:2)', 'cells(:,1)', 'cells(:,2)', 'cells(:,3)', 'si(1:3)', 'fi(1:3)'

  do i=1,n
    if(book(i)%top==-1) then
      write(ifile, '(i8,2x,a)') i, 'top==-1 ==> skip'
      cycle
    else
      write(ifile, '(i8,2x,3i5,2x,3f10.6,2x,2i5,2x,3i3,1x,3i3,1x,3i3,2x,3i7,2x,3i7)') &
        i, book(i)%ic, book(i)%top, book(i)%spp, book(i)%coord, &
        book(i)%atoms, book(i)%cells, book(i)%si, book(i)%fi
    endif    
  enddo ! i

end subroutine ! print_info_book

!
!
!
subroutine print_info_book_elem(ifile, book)
  implicit none
  !! external
  integer, intent(in) :: ifile
  type(book_pb_t), intent(in) :: book
  !! internal
  integer :: i
  
  i = 1
  write(ifile, '(a8,2x,3a5,2x,3a10,2x,a10,2x,a12,1x,a12,1x,a12,2x,a21,2x,a21)') &
    'i', 'ic', 'tyofpai', 'spp', 'coord(1)', 'coord(2)', 'coord(3)', &
    'atoms(1:2)', 'cells(:,1)', 'cells(:,2)', 'cells(:,3)', 'si(1:3)', 'fi(1:3)'

  write(ifile, '(i8,2x,3i5,2x,3f10.6,2x,2i5,2x,3i4,1x,3i4,1x,3i4,2x,3i7,2x,3i7)') &
    i, book%ic, book%top, book%spp, book%coord, book%atoms, book%cells, &
    book%si, book%fi

end subroutine ! print_info_book_elem

!!
!!
!!
subroutine print_info_book_array(ifile, ba)
  implicit none
  !! external
  integer, intent(in) :: ifile
  type(book_array1_t), allocatable, intent(in) :: ba(:)
  !! internal
  integer :: nba, i
  nba = 0
  if(allocated(ba)) nba = size(ba)
    
  write(ifile,*) 'print_info_book_array: size(ba)', nba
  do i=1,nba;
    write(ifile,'(a,i5,a)') 'pair', i, '; ba(pair) follows from next line'
    call print_info_book(ifile, ba(i)%a)
  enddo ! i

end subroutine ! print_info


!!
!!
!!
subroutine report_book_array(fname, ba, iv, coords)
  use m_log, only : log_size_note
  use m_io, only : get_free_handle
  implicit none
  !! external
  character(*), intent(in) :: fname
  type(book_array1_t), allocatable, intent(in) :: ba(:)
  integer, intent(in) :: iv
  real(8), optional :: coords(:,:)
  !! internal
  integer :: ifile, ios
  
  ifile = get_free_handle()
  open(ifile, file=fname, action='write', iostat=ios)
  if(ios/=0) _die('ios/=0')
  if (present(coords)) then
    write(ifile,'(a)') 'coords(:,:) is present. It follows (as is):'
    write(ifile,'(3f10.6)') coords
  endif
  call print_info_book_array(ifile, ba)
  close(ifile)
  call log_size_note('written: ', trim(fname), iv)
end subroutine ! report_book_array


!!
!!
!!
subroutine report_book_pb(fname, book, iv, coords)
  use m_log, only : log_size_note
  use m_io, only : get_free_handle  
  implicit none
  !! external
  character(*), intent(in) :: fname
  type(book_pb_t), allocatable, intent(in) :: book(:)
  integer, intent(in) :: iv
  real(8), optional :: coords(:,:)
  !! internal
  integer :: ifile, ios
  
  ifile = get_free_handle()
  open(ifile, file=fname, action='write', iostat=ios)
  if(ios/=0) _die('ios/=0')
  if (present(coords)) then
    write(ifile,'(a)') 'coords(:,:) is present. It follows (as is):'
    write(ifile,'(3f10.6)') coords
  endif
  call print_info_book(ifile, book)
  close(ifile)
  call log_size_note('written: ', trim(fname), iv)
end subroutine ! report_book_pb

!
!
!
subroutine report_ls_blocks(suffix, ls_blocks, iv)
  implicit none
  !! external
  character(*), intent(in) :: suffix
  type(book_pb_t), intent(in), allocatable :: ls_blocks(:,:)
  integer, intent(in) :: iv
  !! internal
  type(book_pb_t), allocatable :: ls_blocks_aux(:)
  
  !! Print information on pairs of pairs we want to get Hartree kernel
  if(.not. allocated(ls_blocks)) _die('.not. allocated(ls_blocks)')
  allocate(ls_blocks_aux(size(ls_blocks,2)))
  ls_blocks_aux = ls_blocks(1,:);
  call report_book_pb("report_ls_blocks1_"//trim(suffix)//".txt", &
    ls_blocks_aux, iv)
  ls_blocks_aux = ls_blocks(2,:);
  call report_book_pb("report_ls_blocks2_"//trim(suffix)//".txt", &
    ls_blocks_aux, iv)
  !! END of Print information on pairs of pairs we want to get Hartree kernel

end subroutine ! report_ls_blocks

!
!
!
subroutine set_cell3(cell, book)
  implicit none
  !! external
  integer, intent(in) :: cell(3)
  type(book_pb_t), intent(inout) :: book
  !! internal
  
  book%cells(1:3,3) = cell(1:3)
  
end subroutine ! set_cell3  


end module !m_book_pb
