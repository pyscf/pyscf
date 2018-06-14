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

module m_csr_bloch_mat

#include "m_define_macro.F90"
  use m_warn, only : warn
  use iso_c_binding, only: c_float, c_double, c_int64_t, c_float_complex
  
  private
  
  contains

!
! M(a,b=b(B)) = sum_B  M_sc(a,B)*exp(i*k*rvec_B)
!
subroutine c_csr_bloch_mat(r2s, nrow, i2col, i2dat, i2xyz, nnz, orb_sc2orb_uc,norbs_sc, kvec, cmat)  bind(c, name='c_csr_bloch_mat')
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: nrow  ! number of rows
  integer(c_int64_t), intent(in) :: nnz   ! number of non-zero matrix elements
  integer(c_int64_t), intent(in) :: norbs_sc    ! number of orbitals in super cell (maximal value in i2col)  
  integer(c_int64_t), intent(in) :: r2s(nrow+1) ! row -> start index in data array and i2col array 
  integer(c_int64_t), intent(in) :: i2col(nnz)  ! index -> column 
  real(c_float), intent(in)      :: i2dat(nnz)  ! index -> matrix element
  real(c_float), intent(in) :: i2xyz(3,nnz)  ! index -> difference vector
  integer(c_int64_t), intent(in) :: orb_sc2orb_uc(norbs_sc) ! orbital in super cell -> orbital in unit cell
  real(c_float), intent(in)      :: kvec(3) ! k vector
  complex(c_float), intent(inout) :: cmat(nrow,nrow) ! output matrix
  !! internal
  integer(c_int64_t) :: row, ind, col
  real(c_float), allocatable :: saux(:)
  complex(c_float), allocatable :: caux(:)
  
!  write(6,*) nrow
!  write(6,*) nnz
!  write(6,*) norbs_sc
!  write(6,*) sum(r2s)
!  write(6,*) sum(i2col)
!  write(6,*) sum(i2dat)
!  write(6,*) sum(i2xyz)
!  write(6,*) sum(orb_sc2orb_uc)
!  write(6,*) kvec
!  !write(6,*) cmat
!  write(6,*)
  
  allocate(caux(nnz))
  allocate(saux(nnz))

  call sgemv('t', 3,nnz, 1.0_c_float, i2xyz,3, kvec,1, 0.0_c_float, saux,1)
  caux = exp(cmplx(0.0_c_float, 1.0_c_float, c_float) * saux)*i2dat
  
  cmat = 0
  do row=1,nrow
    do ind=r2s(row), r2s(row+1)-1
      col = orb_sc2orb_uc( i2col(ind+1)+1 )+1
      cmat(row,col)=cmat(row,col)+caux(ind+1)
    enddo ! ind
  enddo ! row 

  _dealloc(caux)
  _dealloc(saux)

end subroutine ! c_csr_bloch_mat


end module !m_csr_bloch_mat
