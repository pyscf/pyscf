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

module m_aos_libnao
  use iso_c_binding, only: c_double, c_int64_t
  use m_system_vars, only: system_vars_t
  use m_spin_dens_aux, only : spin_dens_aux_t, comp_coeff
  implicit none
#include "m_define_macro.F90"

  type(system_vars_t), pointer :: sv => null()
  type(spin_dens_aux_t) :: sda
  real(c_double), allocatable :: rsh(:)
  !$OMP THREADPRIVATE(rsh)

  contains

!
! Compute values of atomic orbitals for the whole molecule and a given set of coordinates
!
subroutine aos_libnao(ncoords, coords, norbs, oc2val, ldo ) bind(c, name='aos_libnao')
  use m_rsphar, only : rsphar
  use m_die, only : die
  implicit none
  !! external
  integer(c_int64_t), intent(in)  :: ncoords
  real(c_double), intent(in)  :: coords(3,ncoords)
  integer(c_int64_t), intent(in)  :: norbs
  integer(c_int64_t), intent(in)  :: ldo
  real(c_double), intent(inout)  :: oc2val(ldo,ncoords)

  ! Interne Variable:
  real(8) :: br0(3), rho, br(3)
  real(8) :: fr_val
  integer(c_int64_t) :: jmx_sp, icoord
  integer  :: atm, spa, mu, j, jjp1,k,so,start_ao
  real(8)  :: coeff(-2:3)

  !! values of localized orbitals
  if (norbs/=sda%norbs) then
    write(6,*) norbs, sda%norbs
    _die('norbs/=sda%norbs')
  endif

  !write(6,*) ncoords
  !write(6,*) norbs
  !write(6,*) ldo
  !write(6,*) __FILE__, __LINE__, sda%mu_sp2j
  !write(6,*) sda%sp2nmult
  
  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP SHARED(ncoords, sda, oc2val, coords) &
  !$OMP PRIVATE(fr_val, atm, spa, jmx_sp, rho, br0, k, coeff, so, mu, start_ao, j, jjp1, icoord, br)
  !$OMP DO
  do icoord=1,ncoords
    br = coords(1:3,icoord)
    do atm=1,sda%natoms;
      spa  = sda%atom2sp(atm);
      jmx_sp = maxval(sda%mu_sp2j(:,spa))
      br0  = br - sda%coord(:,atm);  !!print *, 'br01',br,br01,br1;
      rho = sqrt(sum(br0**2));
      if (rho>sda%atom2rcut(atm)) cycle
      
      call comp_coeff(sda, coeff, k, rho)
      call rsphar(br0, jmx_sp, rsh(0:));
      so = sda%atom2start_orb(atm)
    
      do mu=1,sda%sp2nmult(spa); 
        if(rho>sda%mu_sp2rcut(mu,spa)) cycle;
        start_ao = sda%mu_sp2start_ao(mu, spa)
        fr_val = sum(coeff*sda%psi_log(k-2:k+3,mu,spa));
        j = sda%mu_sp2j(mu,spa);
        jjp1 = j*(j+1);
        oc2val(start_ao+so-1:start_ao+2*j+so-1, icoord)= fr_val*rsh(jjp1-j:jjp1+j)
      enddo ! mu
    enddo ! atom
  enddo ! icoord
  !$OMP ENDDO
  !$OMP ENDPARALLEL
 
end subroutine ! aos_libnao

!
!
!
subroutine init_aos_libnao(norbs, info) bind(c, name='init_aos_libnao') 
  use m_system_vars, only : get_norbs, get_nspin, get_jmx
  use m_sv_libnao_orbs, only : sv_libnao=>sv_orbs
  use m_spin_dens_aux, only : init_spin_dens_withoutdm
  use m_rsphar, only : rsphar, init_rsphar
  use m_die, only : die
  implicit none
  ! external
  integer(c_int64_t), intent(in) :: norbs
  integer(c_int64_t), intent(inout) :: info
  ! internal
  integer(c_int64_t) :: n, nspin, jmx
  sv => null()
  
  n = get_norbs(sv_libnao)
  !write(6,*) 'n ', n
  nspin = sv_libnao%nspin
  if ( nspin < 1 .or. nspin > 2 ) then
    write(6,*) nspin
    info = 1
    _die('nspin < 1 .or. nspin > 2')
  endif

  if ( n /= norbs ) then
    write(6,*) n, norbs
    info = 2
    _die('n /= norbs')
  endif

  sv => sv_libnao
  call init_spin_dens_withoutdm(sv, sda)
  jmx = get_jmx(sv)
  call init_rsphar(jmx)
  !$OMP PARALLEL DEFAULT(NONE) SHARED(jmx)
  !$OMP CRITICAL
  _dealloc(rsh)
  allocate(rsh(0:(jmx+1)**2-1))
  !$OMP END CRITICAL
  !$OMP END PARALLEL
  info = 0
 
end subroutine !init_dens_libnao

end module !m_aos_libnao
