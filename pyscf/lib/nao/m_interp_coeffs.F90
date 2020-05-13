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

module m_interp_coeffs
  use iso_c_binding, only: c_double, c_int64_t

  implicit none

#include "m_define_macro.F90"

  contains

!
! Compute interpolation coefficients
!
subroutine interp_coeffs(r, nr, gammin_jt, dg_jt, k, coeffs)
  implicit none 
  !! external
  real(c_double), intent(in) :: r, gammin_jt, dg_jt
  integer(c_int64_t), intent(in) :: nr
  integer(c_int64_t), intent(out) :: k
  real(c_double), intent(out) :: coeffs(:)
  !! internal
  real(c_double) :: dy, lr
  
  if (r<=0) then
    coeffs = 0
    coeffs(1) = 1
    k = 1
    return
  endif  

  lr = log(r)
  k  = int((lr-gammin_jt)/dg_jt+1)
  k  = min(max(k,3_c_int64_t), nr-3_c_int64_t)
  dy = (lr-gammin_jt-(k-1_c_int64_t)*dg_jt)/dg_jt
  
  coeffs(1) =     -dy*(dy**2-1.0D0)*(dy-2.0D0)*(dy-3.0D0)/120.0D0
  coeffs(2) = +5.0D0*dy*(dy-1.0D0)*(dy**2-4.0D0)*(dy-3.0D0)/120.0D0
  coeffs(3) = -10.0D0*(dy**2-1.0D0)*(dy**2-4.0D0)*(dy-3.0D0)/120.0D0
  coeffs(4) = +10.0D0*dy*(dy+1.0D0)*(dy**2-4.0D0)*(dy-3.0D0)/120.0D0
  coeffs(5) = -5.0D0*dy*(dy**2-1.0D0)*(dy+2.0D0)*(dy-3.0D0)/120.0D0
  coeffs(6) =      dy*(dy**2-1.0D0)*(dy**2-4.0D0)/120.0D0

  k = k - 2
  return
   
end subroutine !interp_coeffs

end module !m_interp_coeffs
