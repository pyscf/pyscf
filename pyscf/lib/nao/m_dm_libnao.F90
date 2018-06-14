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

module m_dm_libnao

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use iso_c_binding, only: c_double, c_double_complex, c_int64_t
 
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  real(c_double), allocatable, target :: cbask2dm(:,:,:,:,:)
  
  contains

!
! 
!
subroutine init_dm_libnao(cbask2dm_in, nreim, no, ns, nkp, alloc_stat) bind(c, name='init_dm_libnao')
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: nreim,no,ns,nkp
  real(c_double), intent(in) :: cbask2dm_in(nreim,no,no,ns,nkp)
  integer(c_int64_t), intent(inout) :: alloc_stat
  !! internal
  integer(c_int64_t) :: nsize

  if ( nreim<1 .or. nreim>2 ) _die(' nreim<1 .or. nreim>2 ')
  if ( no<1 ) _die(' no<1 ')
  if ( ns<1 .or. ns>2 ) _die(' ns<1 .or. ns>2 ')
  if ( nkp<1 ) _die(' nkp<1 ')
  
  _dealloc(cbask2dm)
  allocate(cbask2dm(nreim,no,no,ns,nkp), stat=alloc_stat)
  nsize = nreim*no*no*ns*nkp
  if(alloc_stat==0) call dcopy(nsize, cbask2dm_in,1, cbask2dm,1)

end subroutine ! init_dm_libnao


end module !m_dm_libnao
