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

module m_hs

#include "m_define_macro.F90"
  use m_sc_dmatrix, only : sc_dmatrix_t
  implicit none
  !
  ! The type hs_t is supposed to be a more general/convenient replacement for dft_hsx_t.
  ! First of all, the type hs_t is nested therefore we will be able to 
  ! improve the type in details without disturbing its upper level construction.
  ! The type consists of two fields spin2hamilt(:) and overlap.
  ! Both fields have a type ``sc_dmatrix_t'' see m_sc_* for explanations.
  !
  type hs_t
    logical :: is_gamma = .false.
    ! if is_gamma==False, then the orbitals must be Bloch's orbitals (Hamiltonian is Hermitian, complex eigen vectors, etc)
    ! if is_gamma==True, then the orbitals must be real-space localized orbitals (Hamiltonian is real, real eigen vectors, etc)

    type(sc_dmatrix_t), allocatable :: spin2hamilt(:) ! Hamiltonian in a super-cell-separated form
    type(sc_dmatrix_t) :: overlap                ! overlap in a super-cell-separated form
    type(sc_dmatrix_t), allocatable :: nabla(:)  ! xyz components of nabla operator 
    real(8) :: tot_electr_chrg = -999     ! total electronic charge   
  end type ! hs_t

contains

!
!
!
function get_nspin(hs) result(n)
  use m_log, only : die
  implicit none
  type(hs_t), intent(in) :: hs
  integer :: n
  if(.not. allocated(hs%spin2hamilt)) _die('.not. allocated(spin2hamilt)')
  n = size(hs%spin2hamilt)
  if(n<1) _die('n<1')
  
end function ! get_nspin


!
!
!
subroutine dealloc(hs)
  use m_sc_dmatrix, only : dealloc_sc=>dealloc
  implicit none
  type(hs_t), intent(inout) :: hs
  
  _dealloc(hs%spin2hamilt)
  _dealloc(hs%nabla)
!  call dealloc_sc(hs%overlap)

end subroutine ! dealloc  



end module !m_hs
