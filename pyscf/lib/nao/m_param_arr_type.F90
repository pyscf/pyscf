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

module m_param_arr_type

#include "m_define_macro.F90"
  use m_param, only : param_t

  
  implicit none

  type param_arr_t
    type(param_t), allocatable :: a(:)
    integer :: n = -999 
  end type ! param_arr_t
    
end module !m_param_arr_type
