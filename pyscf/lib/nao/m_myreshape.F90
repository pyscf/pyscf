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

module m_myreshape

#include "m_define_macro.F90" 
  
  implicit none

  contains

!
!
!
subroutine myreshape_41(a4, a)
  implicit none
  real(8), intent(in) :: a4(:,:,:,:)
  real(8), intent(out) :: a(*)

  integer :: i,i1, i2, i3, i4

  i = 0
  do i4=lbound(a4,4), ubound(a4,4)
    do i3=lbound(a4,3), ubound(a4,3)
      do i2=lbound(a4,2), ubound(a4,2)
        do i1=lbound(a4,1), ubound(a4,1)
          i = i + 1
          a(i) = a4(i1,i2,i3,i4)
        end do
      end do
    end do
  end do

end subroutine ! myreshape_41

!
!
!
subroutine myreshape_14(a, a4)
  implicit none
  real(8), intent(in) :: a(*)
  real(8), intent(out) :: a4(:,:,:,:)

  integer :: i,i1, i2, i3, i4

  i = 0
  do i4=lbound(a4,4), ubound(a4,4)
    do i3=lbound(a4,3), ubound(a4,3)
      do i2=lbound(a4,2), ubound(a4,2)
        do i1=lbound(a4,1), ubound(a4,1)
          i = i + 1
          a4(i1,i2,i3,i4) = a(i)
        end do
      end do
    end do
  end do

end subroutine ! myreshape_14


end module !m_myreshape
