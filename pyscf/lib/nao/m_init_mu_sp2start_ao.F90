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

module m_init_mu_sp2start_ao


#include "m_define_macro.F90"
  use m_die, only : die
  use m_warn, only : warn
  
  implicit none
    
  contains

subroutine init_mu_sp2start_ao(sp2nmult, mu_sp2j, mu_sp2start_ao)
  implicit none
  ! external
  integer, intent(in) :: sp2nmult(:), mu_sp2j(:,:)
  integer, intent(inout), allocatable :: mu_sp2start_ao(:,:)
  !internal
  integer :: nmult_mx, nsp, s,f,j,sp,mu
  
  nmult_mx = size(mu_sp2j,1)
  nsp = size(mu_sp2j,2)
  if(nsp<1) _die('!nsp')
  if(nmult_mx<1) _die('!nmult_mx')
  if(nsp/=size(sp2nmult)) _die('!sp2nmult')
  
  _dealloc(mu_sp2start_ao)
  allocate(mu_sp2start_ao(nmult_mx, nsp))
  mu_sp2start_ao =-999
  
  !! Updating %mu_sp2start_ao
  do sp=1,nsp
    f = 0
    do mu=1,sp2nmult(sp)
      j = mu_sp2j(mu,sp)
      s = f + 1
      f = s + 2*j
      mu_sp2start_ao(mu,sp) = s
    enddo ! mu
  enddo ! sp
  !! END of Updating %mu_sp2start_ao

end subroutine ! init_mu_sp2start_ao

end module !m_init_mu_sp2start_ao


