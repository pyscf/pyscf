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

module m_ao_eval
  use iso_c_binding, only: c_double, c_int64_t

  implicit none

#include "m_define_macro.F90"

  contains

!
! Compute values of atomic orbitals for a given specie
!
subroutine ao_eval(nmu, &
  ir_mu2v_rl, & ! (nr,nmu)
  nr, &
  rhomin_jt, &
  dr_jt, &
  mu2j, & 
  mu2s, &
  mu2rcut, &
  rcen, & ! rcen(3)
  ncoords, &
  coords, & ! (3,ncoords)
  norbs, & 
  res, & ! (ldres,norbs)
  ldres ) bind(c, name='ao_eval')

  use m_rsphar, only : rsphar, dealloc_rsphar, init_rsphar
  use m_interp_coeffs, only : interp_coeffs
  
  implicit none 
  !! external
  integer(c_int64_t), intent(in)  :: nmu
  integer(c_int64_t), intent(in)  :: nr
  real(c_double), intent(in)  :: ir_mu2v_rl(nr,nmu)
  real(c_double), intent(in)  :: rhomin_jt
  real(c_double), intent(in)  :: dr_jt
  integer(c_int64_t), intent(in)  :: mu2j(nmu)
  integer(c_int64_t), intent(in)  :: mu2s(nmu+1)
  real(c_double), intent(in)  :: mu2rcut(nmu)
  real(c_double), intent(in)  :: rcen(3)
  integer(c_int64_t), intent(in)  :: ncoords
  real(c_double), intent(in)  :: coords(3,ncoords)
  integer(c_int64_t), intent(in)  :: norbs 
  integer(c_int64_t), intent(in)  :: ldres        ! must be >=ncoords
  real(c_double), intent(inout) :: res(ldres,norbs) ! norbs is unknown.
  !! internal
  real(c_double), allocatable :: rsh(:)
  real(c_double) :: coeffs(6), r, fval, coord(3), rcutmx
  integer(c_int64_t) :: jmx_sp, icrd, mu, j, s,f,k

  rcutmx = maxval(mu2rcut)
  jmx_sp = maxval(mu2j)
  call init_rsphar(jmx_sp)
  res = 0D0

  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP PRIVATE (icrd, rsh, coord, r, coeffs, mu) &
  !$OMP PRIVATE (j, s, f, k, fval) &
  !$OMP SHARED (coords, rcen, jmx_sp, ncoords, res, rcutmx) &
  !$OMP SHARED (nr, rhomin_jt, dr_jt, nmu, ir_mu2v_rl, mu2j, mu2s)

  allocate(rsh(0:(jmx_sp+1)**2-1))

  !$OMP DO
  do icrd = 1,ncoords
    coord = coords(:,icrd)-rcen
    call rsphar(coord, jmx_sp, rsh)
    r = sqrt(sum(coord**2))
    if(r>rcutmx) cycle
    call interp_coeffs(r, nr, rhomin_jt, dr_jt, k, coeffs)
    do mu=1,nmu
      ! if(r>mu2rcut(mu)) cycle
      j=mu2j(mu)
      s=mu2s(mu)+1
      f=mu2s(mu+1)
      fval = sum(ir_mu2v_rl(k:k+5,mu)*coeffs)
      if (j>0) fval = fval * (r**j)
      res(icrd,s:f) = fval * rsh(j*(j+1)-j:j*(j+1)+j)
    enddo ! mu
  enddo ! icrd
  !$OMP END DO
  
  _dealloc(rsh)
  !$OMP END PARALLEL
end subroutine !ao_eval

end module !m_ao_eval
