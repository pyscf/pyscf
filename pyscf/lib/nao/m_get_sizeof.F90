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

module m_get_sizeof

  implicit none

  interface get_sizeof
!    module procedure i1
    module procedure i2
    module procedure i4
    module procedure i8
    module procedure r4
    module procedure r8
  end interface ! get_sizeof

  contains

!
! Finds size of integer in bytes
!
!integer function i1(v); integer(1) :: v; i1 = 1; end function 

integer function i2(v); integer(2) :: v; i2 = 2; end function

integer function i4(v); integer(4) :: v; i4 = 4; end function

integer function i8(v); integer(8) :: v; i8 = 8; end function

integer function r4(v); real(4) :: v; r4 = 4; end function

integer function r8(v); real(8) :: v; r8 = 8; end function


end module !m_get_sizeof 

