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

module m_sv_libnao_orbs

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use iso_c_binding, only: c_double, c_double_complex, c_int64_t, c_int
 
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  type(system_vars_t), target :: sv_orbs
  
  contains

!
! 
!
subroutine init_sv_libnao_orbs(dinp,ninp, size_x) bind(c, name='init_sv_libnao_orbs')

  use m_fact, only : init_fact
  use m_sv_get, only : sv_get
  use m_system_vars, only : get_nr, init_size_dft_wf_X
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  integer(c_int), intent(in) :: size_x(1:5)
  real(c_double), intent(in) :: dinp(ninp)
  !! internal

  !! executable statements
  call init_fact()  !! Initializations for product reduction/spherical harmonics/wigner3j in Talman's way
  call sv_get(dinp,ninp, sv_orbs)
  call init_size_dft_wf_X(size_x, sv_orbs)

end subroutine ! init_sv_libnao_orbs

end module !m_sv_libnao_orbs
