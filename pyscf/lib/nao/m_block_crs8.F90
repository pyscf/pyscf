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

module m_block_crs8

#include "m_define_macro.F90"
  implicit none

!
! Block-wise Compressed Sparse Row format
!
  type block_crs8_t
    integer              :: m = -999 ! number of rows
    integer              :: n = -999 ! number of columns
    real(8), allocatable :: d(:)      ! data 
    integer, allocatable :: ic2sfn(:,:) ! a main defining correspondence center2number_of_elements
    integer, allocatable :: blk2cc(:,:) ! a main defining correspondence of non-zero blocks (a row-wise, row-continue, column-continue list)
    integer, allocatable :: blk2col(:)  ! a correspondence block --> column index in terms of blocks (similar to col_ind)
    integer, allocatable :: row2ptr(:)  ! a correspondence row --> start for that row in blk2col(:)  (similar to row_ptr)
    integer(8), allocatable :: blk2sfn(:,:) ! block to start/finish/size of that block in the data d(:)
  end type !block_crs8_t  
  
  contains


!
!
!
logical function is_init(bcrs, cfile, iline)
  implicit none
  type(block_crs8_t), intent(in) :: bcrs
  character(*), intent(in) :: cfile
  integer, intent(in) :: iline

  is_init = .false.

  if(.not. allocated(bcrs%d)) then
    write(6,*) cfile, iline
    stop '!%d'
  endif

  is_init = .true.
  
end function ! is_init

!
!
!
subroutine init(ic2size, blk2cc, csym, m)
  implicit none
  integer, intent(in) :: ic2size(:)
  integer, intent(in) :: blk2cc(:,:)
  character(*), intent(in) :: csym
  type(block_crs8_t), intent(inout) :: m
  !! internal
  integer :: nc, ne, ic2, cc(2), nn(2), ib, ibp, cc_prev(2)
  integer(8) :: s, f, n, ntot

  nc = size(ic2size)
  if(nc .lt. 1) then
    write(6,*) __FILE__, __LINE__
    stop 'nc .lt. 1'
  endif

  if(any(ic2size .lt. 1)) then
    write(6,*) __FILE__, __LINE__
    stop 'ic2s .lt. 1?'
  endif

  nn = ubound(blk2cc) 
  if(nn(1) .lt. 2) then
    write(6,*) __FILE__, __LINE__
    stop 'nn(1) .lt. 2'
  endif
  
  if(nn(2) .lt. 1) then
    write(6,*) __FILE__, __LINE__
    stop 'nn(2) .lt. 1'
  endif

  ne  = sum(ic2size)
  m%m = ne
  m%n = ne  !! for rectangular matrices, we need two  ic2size...
  _dealloc(m%ic2sfn)
  allocate(m%ic2sfn(3,nc))
  f = 0
  do ic2=1,nc
    s = f + 1
    n = ic2size(ic2)
    f = s + n - 1
    m%ic2sfn(1:3, ic2) = [int(s),int(f),int(n)] 
  enddo

  _dealloc(m%blk2cc)
  allocate(m%blk2cc(2,nn(2)))
  m%blk2cc = blk2cc
  
  _dealloc(m%blk2col)
  allocate(m%blk2col(nn(2))) ! a correspondence block --> column index in terms of blocks (similar to col_ind)

  _dealloc(m%row2ptr)
  allocate(m%row2ptr(nc+1)) ! a correspondence row --> start for that row in blk2col(:)  (similar to row_ptr)
  
  _dealloc(m%blk2sfn)
  allocate(m%blk2sfn(3,nn(2))) ! block to start/finish/size of that block in the data d(:)

  ntot = 0
  cc_prev = 0
  m%row2ptr = 0 ! for a later check for empty block-rows
  f = 0
  do ib=1,nn(2)
    cc = blk2cc(1:2,ib)

    select case(csym)
    case('U', 'u') 
      if(cc(1)>cc(2)) then
        write(6,*) __FILE__, __LINE__; stop '!cc(1)>cc(2)'; 
      endif
    case('L', 'l')
      if(cc(1)<cc(2)) then
        write(6,*) __FILE__, __LINE__; stop '!cc(1)<cc(2)'; 
      endif
    case('N', 'n')
      continue
    case default
      write(6,*) __FILE__, __LINE__; stop '!csym'
    end select  

    !! Ensuring row-major order, column increasing, row increasing order ...    
    if(any(cc<1) .or. any(cc>nc)) then;
      write(6,*)"any(cc<1)", any(cc<1)
      write(6,*)"any(cc>nc)", any(cc>nc)
      write(6,*)"nc", nc
      do ibp=1,nn(2)
        write(6,*)blk2cc(1:2,ibp)
      enddo
      write(6,*) __FILE__, __LINE__; stop '!cc'; 
    endif

    if(cc_prev(1)>cc(1)) then; 
      write(6,*)'cc_prev(1)>=cc(1)'
      write(6,*) cc_prev(1), cc(1)
      write(6,*) __FILE__, __LINE__;
      stop '!cc';
    endif
    if(cc(1)>cc_prev(1)) then
      m%row2ptr(cc(1)) = ib
      cc_prev(2) = 0
    endif
    if(cc_prev(2)>=cc(2)) then;
      do ibp=1,nn(2)
        print*, blk2cc(1:2,ibp)
      enddo

      write(6,*)'cc_prev(2)>=cc(2)'
      write(6,*) cc_prev(2), cc(2)
      write(6,*) __FILE__, __LINE__;
      stop '!cc';
    endif
    !! END of Ensuring row-major order, column increasing, row increasing order ...    

    m%blk2col(ib) = cc(2)

    cc_prev = cc
    s = f + 1
    n = m%ic2sfn(3,cc(1))*m%ic2sfn(3,cc(2))
    f = s + n - 1
    m%blk2sfn(1:3, ib) = [s,f,n]
  enddo
  m%row2ptr(nc+1) = nn(2)+1

  if(any(m%row2ptr<1)) then
    write(6,*) 'm%row2ptr'
    write(6,'(9999i5)') m%row2ptr
    write(6,*) __FILE__, __LINE__;
    stop '!row2ptr'
  endif
  
  _dealloc(m%d)
  ntot = f
  allocate(m%d(ntot))
  m%d = 0

end subroutine ! init


!
!
!
subroutine gen_random_square_mat(nc, bcrs)
  implicit none
  !! external
  integer, intent(in) :: nc ! number of centers
  type(block_crs8_t), intent(inout) :: bcrs
  !! internal
  integer :: step, ib, ic1, ic2
  integer :: m 
  ! number of elements in the vector which can be represented by the list of segments 

  integer, allocatable :: ic2size(:)
  real(8), allocatable :: dc2size(:) 
  ! The array is telling the size of each block.
  ! this could be a correspondence atom2norbitals
  ! or center2number_of_products, for example.

  real(8), allocatable :: cc2nz(:,:) ! auxiliary to generate blk2cc()
  integer, allocatable :: blk2cc(:,:) ! 1-2, block
  ! The array is a list of non-zero blocks
  
  call dealloc(bcrs)
  if(nc<1) then
    write(6,*)'warn: nc<0 ==> return', __FILE__, __LINE__
    return
  endif
  
  allocate(ic2size(nc))
  allocate(dc2size(nc))
  
  call random_number(dc2size)
  dc2size = 30*dc2size+1
  ic2size = int(dc2size)
  m = sum(ic2size)

  allocate(cc2nz(nc,nc))
  call random_number(cc2nz)
  cc2nz = cc2nz - 0.666e0
  
  do step=1,2
    ib = 0
    do ic1=1,nc
      do ic2=1,nc
        if(cc2nz(ic1,ic2)<=0 .and. (ic2 .ne. ic1)) cycle
        ib = ib + 1
        if(step<2) cycle
        blk2cc(1:2,ib) = [ic1,ic2]
      enddo ! ic1
    enddo ! ic2
    if(step==1) then
      allocate(blk2cc(2,ib))
    endif
  enddo ! step

  !write(6,*) 'ic2size'
  !write(6,*) ic2size

  !write(6,*) 'blk2cc'
  !do ib=1,size(blk2cc,2)
  !  write(6,'(i4,2x,2i7)') ib, blk2cc(1:2,ib)
  !enddo ! ib  

  call init(ic2size, blk2cc, 'N', bcrs)

  !write(6,*) '%row2ptr'
  !write(6,'(9999i5)') bcrs%row2ptr

  !write(6,*) '%blk2col'
  !write(6,'(9999i5)') bcrs%blk2col

  call random_number(bcrs%d)
  
  _dealloc(ic2size)
  _dealloc(dc2size)
  _dealloc(cc2nz)
  _dealloc(blk2cc)

end subroutine ! 

!
!
!
subroutine dealloc(m)
  implicit none
  type(block_crs8_t), intent(inout) :: m
  
  m%m = -999
  m%n = -999
  _dealloc(m%d)
  _dealloc(m%ic2sfn) 
  _dealloc(m%blk2cc)
  _dealloc(m%blk2col)
  _dealloc(m%row2ptr)
  _dealloc(m%blk2sfn)
  
end subroutine ! dealloc

!
!
!
logical function is_ok(m, bcrs, cfile, iline)
  implicit none
  integer, intent(in) :: m
  type(block_crs8_t), intent(in) :: bcrs
  character(*), intent(in) :: cfile
  integer, intent(in) :: iline

  is_ok = .false.

  if(bcrs%m<1) then
    write(6,*) cfile, __LINE__
    stop '!m<1'
  endif

  if(m/=bcrs%m) then
    write(6,*) cfile, __LINE__
    stop '!m/=%m'
  endif

  if(.not. allocated(bcrs%d)) then
    write(6,*) cfile, iline
    stop '!%d'
  endif

  if(.not. allocated(bcrs%ic2sfn)) then
    write(6,*) cfile, iline
    stop '!%ic2sfn'
  endif

  if(sum(bcrs%ic2sfn(3,:))/=m) then 
     write(6,*) cfile, iline
     stop '!%ic2sfn/=m'
  endif

  is_ok = .true.

end function ! is_ok

!
!
!
function  get_nblock_cols(bcrs) result(n)
  implicit none
  !! external
  type(block_crs8_t), intent(in) :: bcrs
  integer :: n
  !! 
  n = -1
  if(.not. allocated(bcrs%ic2sfn)) then
    write(6,*) __FILE__, __LINE__
    stop '!%ic2sfn'
  endif  
  n = size(bcrs%ic2sfn,2)

end function ! get_nblock_cols

!
!
!
function get_nn(bcrs) result(nn)
  implicit none
  type(block_crs8_t), intent(in) :: bcrs
  integer :: nn(2)
  
  nn = [bcrs%m, bcrs%n]
  if(any(nn<1)) then
    write(6,*) __FILE__, __LINE__
    stop '(!nn<1)'
  endif  
  
end function ! get_nn

!
!
!
function get_nnz(bcrs) result(n)
  implicit none
  type(block_crs8_t), intent(in) :: bcrs
  integer :: n
  !! 
  integer :: nn(2)
  
  nn = get_nn(bcrs)
  if(.not. is_ok(nn(1), bcrs, __FILE__, __LINE__)) then
    write(0,*) __FILE__, __LINE__, '!ok'
  endif
  
  n = size(bcrs%d)
  
end function ! get_nnz

!
!
!
function  get_nblocks_stored(m) result(n)
  implicit none
  !! external
  type(block_crs8_t), intent(in) :: m
  integer :: n
  !! internal
  integer :: nn(3)
  
  nn(1) = size(m%blk2cc,2)
  nn(2) = size(m%blk2col,1)
  nn(3) = size(m%blk2sfn,2)
  if(any(nn<1)) then
    write(0,*) __FILE__, __LINE__
    stop '!any(nn<1)'
  endif  
  n = nn(1)
  if(any(nn/=n)) then
    write(0,*) __FILE__, __LINE__
    stop '!any(nn/=n)' 
  endif
end function ! get_nblocks_stored

!
! Gets the block number for a given centers 
!
function get_block_given_cc(m, c1, c2) result(blk)
  implicit none
  ! external
  type(block_crs8_t), intent(in) :: m
  integer, intent(in) :: c1, c2
  integer(8) :: blk
  ! internal
  integer(8) :: bs, bf, b
  
  bs = m%row2ptr(c1)
  bf = m%row2ptr(c1+1)-1
  
  blk = -1
  do b=bs,bf
    if(c2==m%blk2col(b)) then
      blk = b
      return
    endif  
  enddo !

end function ! get_block_given_cc

end module !m_block_crs4
