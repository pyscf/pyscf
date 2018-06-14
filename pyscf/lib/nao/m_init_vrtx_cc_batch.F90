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

module m_init_vrtx_cc_batch

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use iso_c_binding, only: c_double, c_double_complex, c_int64_t
 
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  contains

!
! 
!
subroutine init_vrtx_cc_batch(dinp,ninp) bind(c, name='init_vrtx_cc_batch')

  use m_fact, only : init_fact
  use m_sv_libnao_prds, only : sv=>sv_prds
  use m_pb_libnao, only : pb
  use m_para_libnao, only : para
  use m_biloc_aux_libnao, only : a
  use m_dp_aux_libnao, only : dp_a
  use m_orb_rspace_aux_libnao, only : orb_a
    
  use m_sv_prod_log_get, only : sv_prod_log_get
  use m_biloc_aux, only : init_biloc_aux
  use m_orb_rspace_type, only : init_orb_rspace_aux
  use m_parallel, only : init_parallel  
  use m_dp_aux, only: preinit_dp_aux  
  use m_init_book_dp_apair, only : init_book_dp_apair
  use m_hkernel_pb_bcrs8, only : hkernel_pb_bcrs
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  real(c_double), intent(in) :: dinp(ninp)
  !! internal
  integer :: ul
  
  !! executable statements
  call init_parallel(para, 0)
  call init_fact()  !! Initializations for product reduction/spherical harmonics/wigner3j in Talman's way
  call sv_prod_log_get(dinp,ninp, sv, pb)
  call preinit_dp_aux(sv, dp_a)
  call init_book_dp_apair(pb)   
  call hkernel_pb_bcrs(pb, dp_a%hk)
  call init_orb_rspace_aux(sv, orb_a, ul)
  call init_biloc_aux(sv, pb%pb_p, para, orb_a, a)

end subroutine ! init_vrtx_cc_batch


end module !m_init_vrtx_cc_batch
