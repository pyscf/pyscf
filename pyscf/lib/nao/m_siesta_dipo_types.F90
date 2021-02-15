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

module m_siesta_dipo_types
#include "m_define_macro.F90"
  use m_die, only : die
  use m_precision, only : siesta_int
  implicit none
  private die

!!! DSP modified for dipole
  type siesta_dipo_t
    integer(siesta_int)              :: norbitals = -999     !! Number of orbitals in (unit cell)
    integer(siesta_int)              :: norbitals_sc = -999  !! Number of orbitals in super cell
    integer(siesta_int)              :: nnonzero = -999      !! Number of nonzero matrix elements in H and S
    integer(siesta_int), allocatable :: sc_orb2uc_orb(:)     !! Index "super cell orbital --> unit cell orbital" indxuo in SIESTA
    integer(siesta_int), allocatable :: row2nnzero(:)        !! (norbitals)
    integer(siesta_int), allocatable :: sparse_ind2column(:) !! (nnonzero)
    real(4), allocatable    :: dipo_sparse(:,:)        !! (nnonzero,nspin)
    real(4), allocatable    :: dipo_corr_sparse(:,:)          !! (nnonzero)
    real(4), allocatable    :: overlap_sparse(:)          !! (nnonzero)
  end type siesta_dipo_t

!! DSP Quick change to compute dipole matrix from Siesta input
!! Next type contains information about the dipole matrix elements in the orbital basis
!! and the correction due to the non-local pseudos
  type dft_dipo_t
      real(8), allocatable ::  dipo(:,:,:)   ! (xyz, orbital, supercell_orbital)
      real(8), allocatable :: dipo_corr(:,:,:) !  (xyz, orbital, supercell_orbital)
      real(8), allocatable :: overlap(:,:) !  (orbital, supercell_orbital)
  end type ! dft_dipo_t

contains 

subroutine dealloc(d)
  implicit none
  type(dft_dipo_t), intent(inout) :: d
  _dealloc(d%dipo)
  _dealloc(d%dipo_corr)
  _dealloc(d%overlap)
  
end subroutine ! dealloc


end module m_siesta_dipo_types

  
