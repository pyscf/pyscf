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

module m_color
  implicit none

#include "m_define_macro.F90"

  type color_t
    character(8) :: HEADER  = char(27)//'[95m'
    character(8) :: OKBLUE  = char(27)//'[94m'
    character(8) :: WARNING = char(27)//'[93m'
    character(8) :: OKGREEN = char(27)//'[92m'
    character(8) :: FAIL    = char(27)//'[91m'

    character(8) :: C100  = char(27)//'[100m'
    character(8) :: C99   = char(27)//'[99m'
    character(8) :: C98   = char(27)//'[98m'
    character(8) :: C97   = char(27)//'[97m'
    character(8) :: CYAN    = char(27)//'[96m'
    character(8) :: MAGENTA = char(27)//'[95m'
    character(8) :: BLUE    = char(27)//'[94m'
    character(8) :: YELLOW  = char(27)//'[93m'
    character(8) :: GREEN   = char(27)//'[92m'
    character(8) :: RED     = char(27)//'[91m'
    character(8) :: GRAY    = char(27)//'[90m'

    character(4) :: ENDC    = char(27)//'[0m'
  end type !color_t
  
  type(color_t), save :: bc
  
  contains

!
!
!
subroutine color_on(bc)
!#ifdef NAGFOR
!  use f90_unix_env, only : getenv
!#endif   
  implicit none
  ! external
  type(color_t), intent(inout) :: bc
  ! internal
  character(500) :: cvar
  integer :: istat
  intrinsic Get_Environment_Variable
  
  !call getenv('TERM', cvar)
  call GET_ENVIRONMENT_VARIABLE('TERM', cvar, status=istat)
  if(istat/=0) cvar = ''

  if(trim(cvar)=='cygwin' .or. trim(cvar)=='mingw') then 
    ! These definitions deliver colors in windows-based terminals
    bc%HEADER  = char(27)//'[01;35m' 
    bc%OKBLUE  = char(27)//'[01;34m'
    bc%WARNING = char(27)//'[01;33m'
    bc%OKGREEN = char(27)//'[01;32m'
    bc%FAIL    = char(27)//'[01;31m'

    bc%MAGENTA = char(27)//'[01;35m'
    bc%BLUE    = char(27)//'[01;34m'
    bc%YELLOW  = char(27)//'[01;33m'
    bc%GREEN   = char(27)//'[01;32m'
    bc%RED     = char(27)//'[01;31m'
    bc%GRAY    = char(27)//'[01;30m'
    
  else 
    ! Otherwise, assume that we have a linux-based teminal with 'normal' escape
    bc%HEADER  = char(27)//'[95m'
    bc%OKBLUE  = char(27)//'[94m'
    bc%WARNING = char(27)//'[93m'
    bc%OKGREEN = char(27)//'[92m'
    bc%FAIL    = char(27)//'[91m'

    bc%MAGENTA = char(27)//'[95m'
    bc%BLUE    = char(27)//'[94m'
    bc%YELLOW  = char(27)//'[93m'
    bc%GREEN   = char(27)//'[92m'
    bc%RED     = char(27)//'[91m'
    bc%GRAY    = char(27)//'[90m'
  endif

  bc%ENDC    = char(27)//'[0m'
  
end subroutine !color_on

!
!
!
subroutine color_off(bc)
  implicit none
  type(color_t), intent(inout) :: bc

  bc%HEADER  = ' '
  bc%OKBLUE  = ' '
  bc%WARNING = ' '
  bc%OKGREEN = ' '
  bc%FAIL    = ' '

  bc%MAGENTA = ' '
  bc%BLUE    = ' '
  bc%YELLOW  = ' '
  bc%GREEN   = ' '
  bc%RED     = ' '
  bc%GRAY    = ' ' 

  bc%ENDC    = ' '
  
end subroutine !color_off
  

end module !m_color

