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

module m_dens_libnao

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use m_spin_dens_aux, only : spin_dens_aux_t
  use iso_c_binding, only: c_double, c_double_complex, c_int64_t
 
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  type(system_vars_t), pointer :: sv => null()
  real(c_double), pointer :: cbask2dm(:,:,:,:,:) => null()

  type(spin_dens_aux_t) :: sda
  
  contains

!
! 
!
subroutine dens_libnao(c2xyz, nc, sc2dens, nspin) bind(c, name='dens_libnao')
  use m_spin_dens_fini8, only : spin_dens
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: nc, nspin
  real(c_double), intent(in) :: c2xyz(3,nc)
  real(c_double), intent(inout) :: sc2dens(nspin,nc)
  !! internal
  integer(c_int64_t) :: ic
  
  if ( nc<1 ) _die(' nc<1 ')
  if ( nspin<1 .or. nspin>2 ) _die(' nspin<1 .or. nspin>2 ')
  if ( sda%nspin /= nspin ) _die('sda%nspin /= nspin')

  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP PRIVATE (ic) &
  !$OMP SHARED(nc, nspin, sda, c2xyz, sc2dens)
  !$OMP DO
  do ic=1,nc;
    call spin_dens(sda, c2xyz(1:3,ic), sc2dens(1:nspin,ic))
  enddo 
  !$OMP END DO
  !$OMP END PARALLEL

end subroutine ! dens_libnao


!
!
!
subroutine init_dens_libnao(info) bind(c, name='init_dens_libnao') 
  use m_system_vars, only : get_norbs, get_nspin
  use m_sv_libnao_orbs, only : sv_libnao=>sv_orbs
  use m_dm_libnao, only : cbask2dm_libnao=>cbask2dm
  use m_spin_dens_aux, only : init_spin_dens_aux
  implicit none
  ! external
  integer(c_int64_t), intent(inout) :: info
  ! internal
  integer(c_int64_t) :: n, nspin, nn(5)

  cbask2dm => null()
  sv => null()
  
  n = get_norbs(sv_libnao)
  nspin = sv_libnao%nspin
  nn = ubound(cbask2dm_libnao)
  if ( any(n /= nn(2:3)) ) _die('any(n /= nn(2:3))')
  if ( nspin /= nn(4) ) then
    write(6,*) nspin, nn
    _die('nspin /= nn(4)')
  endif
  cbask2dm => cbask2dm_libnao
  sv => sv_libnao
  
  call init_spin_dens_aux(sv, cbask2dm(1,:,:,:,1), sda)
  info = 0
end subroutine !init_dens_libnao

end module !m_dm_libnao
