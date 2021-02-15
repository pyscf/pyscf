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

module m_param_arr

#include "m_define_macro.F90"
  use m_die, only : die
  use m_warn, only : warn
  use m_param, only : param_t
  use m_upper, only : upper
  use m_input, only : MAXLEN
  use m_param_arr_type, only : param_arr_t

  
  implicit none
  private die
  private warn
  private param_arr_t

  interface init_param
    module procedure init_integer
    module procedure init_character
    module procedure init_logical
    module procedure init_real8
  end interface

  interface get_p
    module procedure get_p_i
    module procedure get_p_c
    module procedure get_p_d
  end interface !get_p 

  contains

!
!
!
subroutine report(p, fname, cfile, iline)
  use m_io, only : get_free_handle
  use m_param, only : write_info_header, write_info_1line, write_info
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: fname
  character(*), intent(in) :: cfile
  integer, intent(in) :: iline
  !! internal
  integer :: ifile, ios, j
  
  ifile = get_free_handle()
  
  open(ifile, file=fname, action="write", form="formatted", iostat=ios)
  if(ios .ne. 0) _die('!ios')
  write(ifile,'(a,i6)') "Report from "//trim(cfile)//', ', iline
  write(ifile,*)
  write(ifile,'(a)') "List of parameters"
  call write_info_header(ifile)
  do j=1,get_nop(p); call write_info_1line(p%a(j), ifile); enddo
  write(ifile,*)
  write(ifile,'(a)') "All information available about parameters"
  do j=1,get_nop(p)
    write(ifile,*)
    call write_info(p%a(j), ifile)
  enddo
  close(ifile)

end subroutine ! report  

!
!
!
subroutine get_p_i(p, pname, cv)
  use m_param, only : write_info
  implicit none
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: pname
  integer, intent(inout) :: cv
  !! internal
  integer :: l

  cv = -999
  l = get_pnumber(p, pname)
  if(l .lt. 1) then
    write(6,*) 'Parameter ', trim(pname)
    _die('!not found')
  endif
  
  if(p%a(l)%ctype/='INTEGER') then
     call write_info(p%a(l), 6)
    _die('!type/=INTEGER')
  endif

  if(.not. allocated(p%a(l)%icv)) _die('!icv')
  cv = p%a(l)%icv(1)

end subroutine ! get_p_i

!
!
!
subroutine get_p_d(p, pname, cv)
  use m_param, only : write_info
  implicit none
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: pname
  real(8), intent(inout) :: cv
  !! internal
  integer :: l

  cv = -999
  l = get_pnumber(p, pname)
  if(l .lt. 1) then
    write(6,*) 'Parameter ', trim(pname)
    _die('!not found')
  endif
  
  if(p%a(l)%ctype/='REAL8') then
     call write_info(p%a(l), 6)
    _die('!type/=REAL8')
  endif

  if(.not. allocated(p%a(l)%dcv)) _die('!dcv')
  cv = p%a(l)%dcv(1)

end subroutine ! get_p_d

!
!
!
subroutine get_p_c(p, pname, cv, l_as_is)
  use m_param, only : write_info
  use m_upper, only : upper
  implicit none
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: pname
  character(*), intent(inout) :: cv
  logical, optional, intent(in) :: l_as_is
  !! internal
  integer :: l

  cv = ''
  l = get_pnumber(p, pname)
  if(l .lt. 1) then
    write(6,*) 'Parameter ', trim(pname)
    _die('!not found')
  endif
  
  if(p%a(l)%ctype/='CHARACTER') then
     call write_info(p%a(l), 6)
    _die('!type/=CHARACTER')
  endif

  if(.not. allocated(p%a(l)%ccv)) _die('!ccv')
  cv = upper(p%a(l)%ccv(1))
  if(present(l_as_is)) then
    if(l_as_is) cv = p%a(l)%ccv(1)
  endif

end subroutine ! get_p_c

!
!
!
function get_i(p, pname) result(cv)
  implicit none
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: pname
  integer :: cv
  call get_p_i(p, pname, cv)
end function ! get_i  

!
!
!
function get_c(p, pname) result(cv)
  implicit none
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: pname
  character(MAXLEN) :: cv
  
  call get_p_c(p, pname, cv)
end function ! get_c 

!
!
!
function get_s(p, pname) result(cv)
  implicit none
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: pname
  character(MAXLEN) :: cv
  
  call get_p_c(p, pname, cv, .true.)
end function ! get_s 


!
!
!
function get_d(p, pname) result(cv)
  implicit none
  !! external
  type(param_arr_t), intent(in) :: p
  character(*), intent(in) :: pname
  real(8) :: cv
  call get_p_d(p, pname, cv)
end function ! get_d 

!
!
!
subroutine init_character(pname, inp, dv, p, cdescr, iv, fname, fline)
  use m_input, only : input_t
  use m_param, only : init_param_p=>init_param
  implicit none
  ! external
  character(*), intent(in)   :: pname
  type(input_t), intent(in)  :: inp
  character(*), intent(in)   :: dv
  type(param_arr_t), intent(inout) :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l

  call reset_enlarge(p)
  l = get_pnumber(p, pname)
  if(l .lt. 1) then
    p%n = p%n + 1
    if(p%n .lt. 1) _die('!p%n') 
    if(p%n .gt. size(p%a)) _die('!p%n')
    call init_param_p(pname, inp, dv, p%a(p%n), cdescr, iv, fname, fline)
  else
    if(l .gt. size(p%a)) _die('!l')
    call init_param_p(pname, inp, dv, p%a(l), cdescr, iv, fname, fline)
  endif
   
end subroutine ! init_character

!
!
!
subroutine init_integer(pname, inp, dv, p, cdescr, iv, fname, fline)
  use m_input, only : input_t
  use m_param, only : init_param_p=>init_param
  implicit none
  ! external
  character(*), intent(in)   :: pname
  type(input_t), intent(in)  :: inp
  integer, intent(in)   :: dv
  type(param_arr_t), intent(inout) :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l

  call reset_enlarge(p)
  l = get_pnumber(p, pname)
  if(l .lt. 1) then
    p%n = p%n + 1
    if(p%n .lt. 1) _die('!p%n') 
    if(p%n .gt. size(p%a)) _die('!p%n')
    call init_param_p(pname, inp, dv, p%a(p%n), cdescr, iv, fname, fline)
  else
    if(l .gt. size(p%a)) _die('!l')
    call init_param_p(pname, inp, dv, p%a(l), cdescr, iv, fname, fline)
  endif
   
end subroutine ! init_integer

!
!
!
subroutine init_logical(pname, inp, dv, p, cdescr, iv, fname, fline)
  use m_input, only : input_t
  use m_param, only : init_param_p=>init_param
  implicit none
  ! external
  character(*), intent(in)   :: pname
  type(input_t), intent(in)  :: inp
  logical, intent(in)   :: dv
  type(param_arr_t), intent(inout) :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l

  call reset_enlarge(p)
  l = get_pnumber(p, pname)
  if(l .lt. 1) then
    p%n = p%n + 1
    if(p%n .lt. 1) _die('!p%n') 
    if(p%n .gt. size(p%a)) _die('!p%n')
    call init_param_p(pname, inp, dv, p%a(p%n), cdescr, iv, fname, fline)
  else
    if(l .gt. size(p%a)) _die('!l')
    call init_param_p(pname, inp, dv, p%a(l), cdescr, iv, fname, fline)
  endif
   
end subroutine ! init_logical

!
!
!
subroutine init_real8(pname, inp, dv, p, cdescr, iv, fname, fline)
  use m_input, only : input_t
  use m_param, only : init_param_p=>init_param
  implicit none
  ! external
  character(*), intent(in)   :: pname
  type(input_t), intent(in)  :: inp
  real(8), intent(in)   :: dv
  type(param_arr_t), intent(inout) :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l

  call reset_enlarge(p)
  l = get_pnumber(p, pname)
  if(l .lt. 1) then
    p%n = p%n + 1
    if(p%n .lt. 1) _die('!p%n') 
    if(p%n .gt. size(p%a)) _die('!p%n')
    call init_param_p(pname, inp, dv, p%a(p%n), cdescr, iv, fname, fline)
  else
    if(l .gt. size(p%a)) _die('!l')
    call init_param_p(pname, inp, dv, p%a(l), cdescr, iv, fname, fline)
  endif
   
end subroutine ! init_real8


!
!
!
subroutine reset_enlarge(p)
  implicit none
  type(param_arr_t), intent(inout) :: p

  if(.not. allocated(p%a)) then
    allocate(p%a(2048))
    p%n = 0
  else
    if(p%n .lt. size(p%a)) return
    _die('!realloc not impl')
  end if

end subroutine !reset_enlarge 

!
! Finds the parameter by the name of the parameter
!
integer function get_pnumber(p, pname)
  use m_upper, only : upper
  implicit none
  character(*), intent(in) :: pname
  type(param_arr_t), intent(in) :: p
  integer :: i
  character(MAXLEN) :: pup 
  
  get_pnumber = -1
  if(.not. allocated(p%a)) _die('!%a')
  pup = upper(pname)
  do i=1,p%n
    if(p%a(i)%pname/=pup) cycle
    get_pnumber = i
    return
  enddo !i

end function ! get_pnumber

!
! Get number of parameters in the structure
!
integer function get_nop(p)
  implicit none
  type(param_arr_t), intent(in) :: p

  get_nop = p%n

end function ! get_nop


end module !m_param_arr
