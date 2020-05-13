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

module m_ao_hartree_lap
  use iso_c_binding, only: c_double, c_int64_t

  implicit none

#include "m_define_macro.F90"

  contains

!
! Compute values of Hartree potential for given radial orbitals
! using Laplace transform...
!
subroutine ao_hartree_lap(rr, nmu, &
  ir_mu2ao, & ! (nr,nmu)
  nr, &
  mu2j, & 
  ir_mu2vh & ! (nr,nmu)
  ) bind(c, name='ao_hartree_lap')

  implicit none 
  !! external
  integer(c_int64_t), intent(in)  :: nmu
  integer(c_int64_t), intent(in)  :: nr
  real(c_double), intent(in)      :: rr(nr)
  real(c_double), intent(in)  :: ir_mu2ao(nr,nmu)
  integer(c_int64_t), intent(in)  :: mu2j(nmu)
  real(c_double), intent(out) :: ir_mu2vh(nr,nmu) ! norbs is unknown.
  !! internal
  real(c_double), allocatable :: rr3(:), ff(:)
  real(c_double) :: rrg, rrl, dr, pi
  integer(c_int64_t) :: mu,am,ir,irp
!  write(6,*) nmu
!  write(6,*) nr
!  write(6,*) ir_mu2ao(1:3,1)
!  write(6,*) mu2j
!  write(6,*) mu2rcut
!  write(6,*) ir_mu2vh(1:3,1)
  pi = 4d0*atan(1d0)
  dr = log(rr(2)/rr(1))

  allocate(ff(nr))
  allocate(rr3(nr))
  rr3 = rr**3 * (4*pi*dr)
  
  ir_mu2vh = 0
  do mu=1,nmu
    am = mu2j(mu)
    ff = ir_mu2ao(:,mu)*rr3/(2*am+1d0)
    do ir=1,nr
      do irp=1,nr
        rrg = max(rr(irp), rr(ir))**(am+1)
        rrl = min(rr(irp), rr(ir))**am
        ir_mu2vh(ir,mu) = ir_mu2vh(ir,mu) + rrl/rrg * ff(irp)
      enddo
    enddo
  enddo ! mu
  
  deallocate(ff)
  deallocate(rr3)
  
end subroutine !ao_hartree_lap


end module !m_ao_hartree_lap
