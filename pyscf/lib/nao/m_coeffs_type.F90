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

module m_coeffs_type
!
! Computes the polarizability with eigenstates of KS equation 
!
#include "m_define_macro.F90"

  use m_die, only : die
  
  implicit none
  private die
   
  type s_coeffs_ac_dp_t
    real(4), allocatable :: coeffs_ac_dp(:,:) ! these are coefficients to reexpress dominant products in terms of other
    integer, allocatable :: ind2book_dp_re(:) ! pointers to bookkeeping records
    integer, allocatable :: ind2book_re(:)    ! pointers to bookkeeping records
    integer, allocatable :: ind2sfp_loc(:,:)  ! local counting of atom-centered products
    integer :: is_reexpr = -999
  end type !s_coeffs_ac_dp_t

  type d_coeffs_ac_dp_t
    real(8), allocatable :: coeffs_ac_dp(:,:) ! these are coefficients to reexpress dominant products in terms of other
    integer, allocatable :: ind2book_dp_re(:) ! pointers to bookkeeping records
    integer, allocatable :: ind2book_re(:)    ! pointers to bookkeeping records
    integer, allocatable :: ind2sfp_loc(:,:)  ! local counting of atom-centered products
    integer :: is_reexpr = -999
  end type !d_coeffs_ac_dp_t

  contains 

!
!
!
subroutine dealloc(c)
  implicit none
  !! external
  type(d_coeffs_ac_dp_t), intent(inout) :: c
  _dealloc(c%coeffs_ac_dp)
  _dealloc(c%ind2book_dp_re)
  _dealloc(c%ind2book_re)
  _dealloc(c%ind2sfp_loc)
  c%is_reexpr = -999
end subroutine ! dealloc  
  

!
!
!
function get_diff(c1, c2) result(d)
  implicit none
  !! external
  type(d_coeffs_ac_dp_t), intent(in) :: c1
  type(d_coeffs_ac_dp_t), intent(in) :: c2
  real(8) :: d
  !! internal
  integer :: n1, n2

  d = 0

  _size_alloc(c1%coeffs_ac_dp,n1)
  _size_alloc(c2%coeffs_ac_dp,n2)
  if(n1/=n2) then
    d = 999
    _die('!n1/=n2')
    return  
  endif
  if(n1>0) d = d + sum(abs(c1%coeffs_ac_dp-c2%coeffs_ac_dp))/n1
  if(d>1d-14) _die('!d>1d-14')

  _size_alloc(c1%ind2book_dp_re,n1)
  _size_alloc(c2%ind2book_dp_re,n2)
  if(n1/=n2) then
    d = 999
    _die('!n1/=n2')
    return  
  endif
  if(n1>0) d = d + sum(abs(c1%ind2book_dp_re-c2%ind2book_dp_re))/n1
  if(d>1d-14) then
    write(6,*) c1%ind2book_dp_re
    write(6,*) c2%ind2book_dp_re
    _die('!d>1d-14')
  endif  

  _size_alloc(c1%ind2book_re,n1)
  _size_alloc(c2%ind2book_re,n2)
  if(n1/=n2) then
    d = 999
    _die('!n1/=n2')
    return  
  endif
  if(n1>0) d = d + sum(abs(c1%ind2book_re-c2%ind2book_re))/n1
  if(d>1d-14) _die('!d>1d-14')

  _size_alloc(c1%ind2sfp_loc,n1)
  _size_alloc(c2%ind2sfp_loc,n2)
  if(n1/=n2) then
    d = 999
    _die('!n1/=n2')
    return  
  endif
  if(n1>0) d = d + sum(abs(c1%ind2sfp_loc-c2%ind2sfp_loc))/n1
  if(d>1d-14) _die('!d>1d-14')
  
  d = d + abs(c1%is_reexpr - c2%is_reexpr)
  if(d>1d-14) _die('!d>1d-14')

end function ! get_diff


!
!
!
subroutine alloc_coeffs(nind, c)
  implicit none
  type(d_coeffs_ac_dp_t), intent(inout) :: c
  integer, intent(in) :: nind
  
  if(nind<1) _die('nind<1')
  
  _dealloc(c%ind2book_dp_re)
  _dealloc(c%ind2book_re)
  _dealloc(c%ind2sfp_loc)
  
  allocate(c%ind2book_dp_re(nind)) !! Allocation of ind2book_adj pointers
  allocate(c%ind2book_re(nind)) !! Allocation of ind2book_mix pointers
  allocate(c%ind2sfp_loc(2,nind)) !! local counting 

end subroutine ! alloc_coeffs  


!
!
!
subroutine init_coeffs4(coeffs8, coeffs4)
  implicit none
  !! external
  type(d_coeffs_ac_dp_t), intent(in), allocatable :: coeffs8(:)
  type(s_coeffs_ac_dp_t), intent(inout), allocatable :: coeffs4(:)
  !! internal
  integer :: i,n,m(2),k(1)
  _dealloc(coeffs4)
  if(.not. allocated(coeffs8)) return
  n = size(coeffs8)
  allocate(coeffs4(n))
  
  do i=1,n
    if(.not. allocated(coeffs8(i)%coeffs_ac_dp)) _die('na')
    m = ubound(coeffs8(i)%coeffs_ac_dp)
    allocate(coeffs4(i)%coeffs_ac_dp(m(1),m(2)))
    coeffs4(i)%coeffs_ac_dp = real(coeffs8(i)%coeffs_ac_dp,4)

    if(.not. allocated(coeffs8(i)%ind2book_dp_re)) _die('na')
    k = ubound(coeffs8(i)%ind2book_dp_re)
    allocate(coeffs4(i)%ind2book_dp_re(k(1)))
    coeffs4(i)%ind2book_dp_re = coeffs8(i)%ind2book_dp_re

    if(.not. allocated(coeffs8(i)%ind2book_re)) _die('na')
    k = ubound(coeffs8(i)%ind2book_re)
    allocate(coeffs4(i)%ind2book_re(k(1)))
    coeffs4(i)%ind2book_re = coeffs8(i)%ind2book_re
    
    coeffs4(i)%is_reexpr = coeffs8(i)%is_reexpr

    if(.not. allocated(coeffs8(i)%ind2sfp_loc)) _die('ind2sfp_loc')
    m = ubound(coeffs8(i)%ind2sfp_loc)
    allocate(coeffs4(i)%ind2sfp_loc(m(1),m(2)))
    coeffs4(i)%ind2sfp_loc = coeffs8(i)%ind2sfp_loc

  enddo ! i

end subroutine !  init_coeffs4 

!
!
!
integer function get_nind(c)
  implicit none
  !! external
  type(d_coeffs_ac_dp_t), intent(in) :: c
  !! internal
  integer :: nn(3)
  
  if(.not. allocated(c%ind2book_dp_re)) _die('!ind2book_dp_re')
  if(.not. allocated(c%ind2book_re)) _die('!ind2book_re')
  if(.not. allocated(c%ind2sfp_loc)) _die('!ind2sfp_loc')
  
  nn(1) = size(c%ind2book_dp_re)
  nn(2) = size(c%ind2book_re)
  nn(3) = size(c%ind2sfp_loc,2)
  
  if(any(nn/=nn(1))) _die('!nn')
  get_nind = nn(1)
  
end function ! get_nind  

!
!
!
function get_diff_coeffs(c1, c2) result(d)
  implicit none
  !! external
  type(d_coeffs_ac_dp_t), intent(in), allocatable :: c1(:)
  type(d_coeffs_ac_dp_t), intent(in), allocatable :: c2(:)
  real(8) :: d
  !! internal
  integer :: n1, n2, p
  
  d = 0
  _size_alloc(c1, n1)
  _size_alloc(c2, n2)
  if(n1/=n2) then
    d = 999
    _die('!n1/=n2')
    return
  endif
  if(n1<1) return
  
  do p=1,n1
    d = d + get_diff(c1(p), c2(p))
    if(d>1d-14) then
      write(6,*) p, d
      _die('!d>1d-14')
      return
    endif
  enddo ! p
  
  
end function ! get_diff_coeffs


!
!
!
subroutine init_unit_coeffs(nfp, coeffs)
  use m_alloc_identity, only : alloc_identity
  !! external
  integer, intent(in) :: nfp
  type(d_coeffs_ac_dp_t), intent(inout) :: coeffs

  call alloc_identity(nfp, coeffs%coeffs_ac_dp) ! unit matrix to be formally correct
  coeffs%is_reexpr = 0
end subroutine ! init_unit_coeffs



end module !m_coeffs_type
