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

module m_warn
  implicit none
  
  contains

!
! A warning subroutine
!
subroutine warn(message, line_num)
  use m_color, only : bc
  !! external
  character(*), intent(in), optional :: message
  integer, intent(in), optional :: line_num

  if(present(message)) then
    if(present(line_num)) then
      write(6,'(a,a,a,i7,a)') bc%WARNING, trim(message), ' at line ',line_num,bc%ENDC    
      write(0,'(a,a,a,i7,a)') bc%WARNING, trim(message), ' at line ',line_num,bc%ENDC
    else
      write(6,'(a,a,a)') bc%WARNING, trim(message), bc%ENDC    
      write(0,'(a,a,a)') bc%WARNING, trim(message), bc%ENDC
    endif  
  endif

end subroutine ! warn

end module !m_warn

