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

module m_coul_param

#include "m_define_macro.F90"
  use m_die, only : die
  implicit none
  private die

  type coul_param_t
    integer :: use_mult=-999
    character(100) :: kernel_type=""
    real(8) :: scr_const=-999
  end type ! coul_param_t

contains

!
!
!
character(100) function get_kernel_type(cp)
  use m_upper, only : upper
  implicit none
  type(coul_param_t), intent(in) :: cp
  if(len_trim(cp%kernel_type)<1) _die('len_trim(kernel_type)<1')
  get_kernel_type = upper(cp%kernel_type)
end function ! get_kernel_type 

!
!
!
integer function get_use_mult(cp)
  implicit none
  type(coul_param_t), intent(in) :: cp
  if(cp%use_mult<0) _die('cp%use_mult<0')
  get_use_mult = cp%use_mult
end function ! get_use_mult

!
!
!
real(8) function get_scr_const(cp)
  implicit none
  type(coul_param_t), intent(in) :: cp
  if(cp%scr_const<0) _die('cp%scr_const<0')
  get_scr_const = cp%scr_const
end function ! get_scr_const

!
!
!
subroutine init_coul_param_expl(kernel_type, scr_const, use_mult, cp)
  use m_upper, only : upper
  implicit none
  !! external
  real(8), intent(in) :: scr_const
  character(*), intent(in) :: kernel_type
  integer, intent(in) :: use_mult
  type(coul_param_t), intent(inout) :: cp
  !! internal
  
  cp% scr_const = scr_const
  cp% kernel_type = kernel_type
  if(upper(kernel_type)=="HARTREE" .and. scr_const/=0) then
    cp% use_mult = 0
  else
    cp% use_mult = use_mult
  endif    
  
end subroutine ! init_coul_param_expl


end module !m_coul_param
