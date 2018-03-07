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

module m_bulk_uc_vecs

#include "m_define_macro.F90"
  use m_log, only : die

  implicit none
  private die
  
  contains

!
!
!
function get_uc_volume(uc_vecs) result(V_uc)
  use m_algebra, only : cross_product

  implicit none

  ! external
  real(8), intent(in) :: uc_vecs(3,3)
  real(8) :: V_uc

  if(sum(abs(uc_vecs))==0) _die('sum(abs(uc_vecs))==0')
  
  V_uc = abs(dot_product( uc_vecs(:,1),cross_product(uc_vecs(:,2),uc_vecs(:,3))));

end function !get_uc_volume  

!
!
!
subroutine uc_vecs2uc_vecs_mom(uc_vectors, uc_vectors_mom)
  use m_algebra, only : matinv_d
  implicit none
  ! external
  real(8), intent(in)  :: uc_vectors(3,3)      ! xyz, 123
  real(8), intent(out) :: uc_vectors_mom(3,3)  ! xyz, 123
  ! internal
  real(8) :: pi 
  pi = 4D0*atan(1D0)
  
  uc_vectors_mom = transpose(uc_vectors)
  call matinv_d(uc_vectors_mom)
  uc_vectors_mom = 2*pi*uc_vectors_mom
  
end subroutine ! uc_vectors_2_uc_vectors_mom

!
!
!
function get_uc_vecs_mom(uc_vectors) result(uc_vectors_mom)
  implicit none
  ! external
  real(8), intent(in)  :: uc_vectors(3,3)      ! xyz, 123
  real(8) :: uc_vectors_mom(3,3)  ! xyz, 123
  
  call uc_vecs2uc_vecs_mom(uc_vectors, uc_vectors_mom)
  
end function ! get_uc_vecs_mom

!
! kvec = matmul(uc_vecs, cvec)
! cvec = matmul(coeff_vecs, kvec)
!
! The subroutine returns coeff_vecs for a given uc_vecs, i.e. inverts uc_vecs.
!
function get_coeff_vecs(uc_vecs) result(coeff_vecs)

  use m_algebra, only : matinv_d
  implicit none
  !! external
  real(8), intent(in) :: uc_vecs(3,3)
  real(8) :: coeff_vecs(3,3)

  coeff_vecs = uc_vecs
  call matinv_d(coeff_vecs)
  
end function ! get_coeff_vecs


!
! Shift a vector inside of the first BZ as Monkhorst and Pack discretize it
!
function get_kvec_bz(kvec, uc_vecs_mom, coeff_vecs_mom) result(kvec_bz)
  implicit none
  !! external
  real(8), intent(in) :: kvec(3)
  real(8), intent(in) :: uc_vecs_mom(3,3), coeff_vecs_mom(3,3)
  real(8) :: kvec_bz(3)
  !! internal
  real(8) :: coeffs(3), kvec2(3), error

  kvec_bz = 0
  coeffs = matmul(coeff_vecs_mom, kvec)

  !! Some cross check of provided uc_vecs_mom, coeff_vecs_mom
  kvec2= matmul(uc_vecs_mom, coeffs)
  error = sum(abs(kvec2-kvec))
  if(error>1d-12) _die('error>1d-12')
  !! END of Some cross check of provided uc_vecs_mom, coeff_vecs_mom
  
  coeffs = coeffs - nint(coeffs)
  kvec_bz = matmul(uc_vecs_mom, coeffs)
  
end function ! get_kvec_bz 


end module !m_bulk_uc_vecs
