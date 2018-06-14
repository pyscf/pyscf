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

module m_sparse_vec

  implicit none

  interface get_nnz
    module procedure get_nnz_real4
    module procedure get_nnz_real8
  end interface !get_nnz


  contains

!
!
!
integer function get_nnz_real4(inz2vo)
  implicit none
  ! external
  real(4), intent(in), allocatable :: inz2vo(:,:)
  ! internal
  integer :: nn(2)
  if(.not. allocated(inz2vo)) then
    write(6,*) __FILE__, __LINE__
    stop '!inz2vo... OMP ?'
  endif  
    
  nn = lbound(inz2vo)
  if(any(nn/=[0,1])) then
    write(6,*) __FILE__, __LINE__
    stop 'any(nn/=[0,1])'
  endif  
  if(ubound(inz2vo,2)/=2) then
    write(6,*) __FILE__, __LINE__
    stop 'ubound(inz2vo,2)/=2'
  endif
    
  nn = int(inz2vo(0,1:2))
  if(nn(1)/=nn(2)) then
    write(6,*) __FILE__, __LINE__
    stop 'nn(1)/=nn(2)'
  endif  
  get_nnz_real4 = nn(1)
  if(get_nnz_real4<0) then
    write(6,*) __FILE__, __LINE__
    stop 'get_nnz<0 ? can this be true?'
  endif  
  
end function ! get_nnz  



!
!
!
integer function get_nnz_real8(inz2vo)
  implicit none
  ! external
  real(8), intent(in), allocatable :: inz2vo(:,:)
  ! internal
  integer :: nn(2)
  if(.not. allocated(inz2vo)) then
    write(6,*) __FILE__, __LINE__
    stop '!inz2vo... OMP ?'
  endif  
    
  nn = lbound(inz2vo)
  if(any(nn/=[0,1])) then
    write(6,*) __FILE__, __LINE__
    stop 'any(nn/=[0,1])'
  endif  
  if(ubound(inz2vo,2)/=2) then
    write(6,*) __FILE__, __LINE__
    stop 'ubound(inz2vo,2)/=2'
  endif
    
  nn = int(inz2vo(0,1:2))
  if(nn(1)/=nn(2)) then
    write(6,*) __FILE__, __LINE__
    stop 'nn(1)/=nn(2)'
  endif  
  get_nnz_real8 = nn(1)
  if(get_nnz_real8<0) then
    write(6,*) __FILE__, __LINE__
    stop 'get_nnz<0 ? can this be true?'
  endif  
  
end function ! get_nnz  


end module !m_sparse_vec
