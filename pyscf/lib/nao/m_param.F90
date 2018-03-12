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

module m_param

#include "m_define_macro.F90"
  use m_die, only : die
  use m_warn, only : warn
  use m_input, only : MAXLEN
  use m_upper, only : upper
  
  implicit none
  private die
  private warn

  interface init_param
    module procedure init_real8
    module procedure init_character
    module procedure init_integer
    module procedure init_logical
  end interface

  type param_t
    real(8), allocatable :: dcv(:) ! current value
    real(8), allocatable :: ddv(:) ! default value
    integer, allocatable :: icv(:) 
    integer, allocatable :: idv(:)
    character(MAXLEN), allocatable :: ccv(:) ! current value
    character(MAXLEN), allocatable :: cdv(:) ! default value
    logical, allocatable :: lcv(:) ! current value
    logical, allocatable :: ldv(:) ! default value
    character(20)     :: ctype = ''

    character(MAXLEN) :: pname = ''            ! Name of the parameter
    character(MAXLEN), allocatable :: descr(:) ! Description of the parameter
    character(MAXLEN) :: fname = '' ! source file name where the parameter was initialized
    integer           :: fline = -999 ! line in the source file where parameter was initialized
    integer           :: iline = -999 ! line of the input file where the parameter was found 
    integer           :: ios   = -999 ! input/output status while the parameter was read              
  end type ! param_t
 
    
  contains

!
!
!
subroutine init_real8(pname, inp, def_val, p, cdescr, iv, fname, fline)
  use m_input, only : input_t, get_line_num
  use m_log, only : log_size_note
  implicit none
  ! external
  character(*), intent(in)  :: pname
  type(input_t), intent(in) :: inp
  real(8), intent(in)   :: def_val
  type(param_t), intent(out)  :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l, ios, lt, iv_i
  real(8) :: cv

  call reset(p)
  p%pname = upper(pname)
  p%ctype = 'REAL8'
  cv = def_val
  call init_p_d(cv, p%dcv)
  call init_p_d(cv, p%ddv)
  l = get_line_num(pname, inp)
  p%iline = l
  ios = -1
  if(l>0) then
    lt = len_trim(pname)
    read(inp%lines(l)(lt+1:),*, iostat=ios) cv
  endif
  if(ios==0) call init_p_d(cv, p%dcv)
  p%ios = ios

  iv_i = 0
  if(present(iv)) iv_i = iv
  call log_size_note(pname, cv, iv_i)

  _dealloc(p%descr)
  if(present(cdescr))then
    call init_descr(cdescr, p%descr)
  endif
  p%fname = ''
  if(present(fname)) p%fname = fname
  p%fline = -999
  if(present(fline)) p%fline = fline
  
end subroutine ! init_real8


!
!
!
subroutine init_character(pname, inp, def_val, p, cdescr, iv, fname, fline)
  use m_input, only : input_t, get_line_num
  use m_log, only : log_size_note
  implicit none
  ! external
  character(*), intent(in)   :: pname
  type(input_t), intent(in)  :: inp
  character(*), intent(in)   :: def_val
  type(param_t), intent(out) :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l, ios, lt, iv_i
  character(MAXLEN) :: cv

  call reset(p)
  p%pname = upper(pname)
  p%ctype = 'CHARACTER'
  cv = def_val
  call init_p_c(cv, p%ccv)
  call init_p_c(cv, p%cdv)
  l = get_line_num(pname, inp)
  p%iline = l
  ios = -1
  if(l>0) then
    lt = len_trim(pname)
    read(inp%lines(l)(lt+1:),*, iostat=ios) cv
  endif
  if(ios==0) call init_p_c(cv, p%ccv)
  p%ios = ios

  iv_i = 0
  if(present(iv)) iv_i = iv
  call log_size_note(pname, cv, iv_i)

  _dealloc(p%descr)
  if(present(cdescr))call init_descr(cdescr, p%descr)
  p%fname = ''
  if(present(fname)) p%fname = fname
  p%fline = -999
  if(present(fline)) p%fline = fline
  
end subroutine ! init_character

!
!
!
subroutine init_logical(pname, inp, def_val, p, cdescr, iv, fname, fline)
  use m_input, only : input_t, get_line_num
  use m_log, only : log_size_note
  implicit none
  ! external
  character(*), intent(in)   :: pname
  type(input_t), intent(in)  :: inp
  logical, intent(in)   :: def_val
  type(param_t), intent(out) :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l, ios, lt, iv_i
  logical :: cv

  call reset(p)
  p%pname = upper(pname)
  p%ctype = 'LOGICAL'
  cv = def_val
  call init_p_l(cv, p%lcv)
  call init_p_l(cv, p%ldv)
  l = get_line_num(pname, inp)
  p%iline = l
  ios = -1
  if(l>0) then
    lt = len_trim(pname)
    read(inp%lines(l)(lt+1:),*, iostat=ios) cv
  endif
  iv_i = 0
  if(present(iv)) iv_i = iv
  call log_size_note(pname, cv, iv_i)
  if(ios==0) call init_p_l(cv, p%lcv)

  p%ios = ios
  _dealloc(p%descr)
  if(present(cdescr))call init_descr(cdescr, p%descr)
  p%fname = ''
  if(present(fname)) p%fname = fname
  p%fline = -999
  if(present(fline)) p%fline = fline
  
end subroutine ! init_logical

!
!
!
subroutine init_integer(pname, inp, def_val, p, cdescr, iv, fname, fline)
  use m_input, only : input_t, get_line_num
  use m_log, only : log_size_note
  implicit none
  ! external
  character(*), intent(in)   :: pname
  type(input_t), intent(in)  :: inp
  integer, intent(in)   :: def_val
  type(param_t), intent(out) :: p
  character(*), intent(in), optional :: cdescr
  integer, intent(in), optional :: iv
  character(*), intent(in), optional :: fname
  integer, intent(in), optional :: fline
  ! internal
  integer :: l, ios, lt, iv_i
  integer :: cv

  call reset(p)
  p%pname = upper(pname)
  p%ctype = 'INTEGER'
  cv = def_val
  call init_p_i(cv, p%icv)
  call init_p_i(cv, p%idv)
  l = get_line_num(pname, inp)
  p%iline = l
  ios = -1
  if(l>0) then
    lt = len_trim(pname)
    read(inp%lines(l)(lt+1:),*, iostat=ios) cv
  endif
  iv_i = 0
  if(present(iv)) iv_i = iv
  call log_size_note(pname, cv, iv_i)
  if(ios==0) call init_p_i(cv, p%icv)

  p%ios = ios
  _dealloc(p%descr)
  if(present(cdescr))call init_descr(cdescr, p%descr)
  p%fname = ''
  if(present(fname)) p%fname = fname
  p%fline = -999
  if(present(fline)) p%fline = fline
  
end subroutine ! init_integer

!
!
!
subroutine write_info(p, ifile)
  use m_color, only : bc
  implicit none
  !! external
  type(param_t), intent(in) :: p
  integer, intent(in) :: ifile
  !! internal
  !integer :: n, i

  write(ifile, '(a,a,a,a)')      'Parameter name: ', bc%red, trim(p%pname), bc%endc

  select case(p%ctype)
  case('REAL8')
    write(ifile, '(a,a,g25.15,a)') ' Current value: ', bc%blue, p%dcv(1), bc%endc
    write(ifile, '(a,a,g25.15,a)') ' Default value: ', bc%gray, p%ddv(1), bc%endc
  case('CHARACTER')
    write(ifile, '(a,a,a,a)') ' Current value: ', bc%blue, trim(p%ccv(1)), bc%endc
    write(ifile, '(a,a,a,a)') ' Default value: ', bc%gray, trim(p%cdv(1)), bc%endc
  case('INTEGER')
    write(ifile, '(a,a,i15,a)') ' Current value: ', bc%blue, p%icv(1), bc%endc
    write(ifile, '(a,a,i15,a)') ' Default value: ', bc%gray, p%idv(1), bc%endc
  case('LOGICAL')
    write(ifile, '(a,a,l15,a)') ' Current value: ', bc%blue, p%lcv(1), bc%endc
    write(ifile, '(a,a,l15,a)') ' Default value: ', bc%gray, p%ldv(1), bc%endc
  case default
    write(6,*) trim(p%ctype)
    stop 'unknown ctype'
  end select
 
  call write_pinfo(p, ifile)
 
end subroutine ! write_info

!
!
!
subroutine write_info_1line(p, ifile)
  use m_color, only : bc
  implicit none
  !! external
  type(param_t), intent(in) :: p
  integer, intent(in) :: ifile
  !! internal
  !integer :: n, i

  select case(p%ctype)
  case('REAL8')
    write(ifile, '(a,a20,a,g15.7,a,g15.7,3a)') bc%red,trim(p%pname),&
      bc%blue,p%dcv(1), bc%gray,p%ddv(1),bc%green,trim(p%descr(1)),bc%endc
  case('CHARACTER')
    write(ifile, '(a,a20,a,a15,a,a15,3a)') bc%red,trim(p%pname),&
      bc%blue,trim(p%ccv(1)), bc%gray,trim(p%cdv(1)),bc%green,trim(p%descr(1)),bc%endc
  case('INTEGER')
    write(ifile, '(a,a20,a,i15,a,i15,3a)') bc%red,trim(p%pname),&
      bc%blue,p%icv(1), bc%gray,p%idv(1),bc%green,trim(p%descr(1)),bc%endc
  case('LOGICAL')
    write(ifile, '(a,a20,a,l15,a,l15,3a)') bc%red,trim(p%pname),&
      bc%blue,p%lcv(1), bc%gray,p%ldv(1),bc%green,trim(p%descr(1)),bc%endc
  case default
    write(6,*) trim(p%ctype)
    stop 'unknown ctype'
  end select
 
end subroutine ! write_info_1line

!
!
!
subroutine write_info_header(ifile)
  use m_color, only : bc
  implicit none
  !! external
  integer, intent(in) :: ifile
  !! internal
  !integer :: n, i

  write(ifile, '(a,a20,a,a15,a,a15,3a)') bc%red,'Parameter name', &
    bc%blue,'Current value', bc%gray,'Default value',bc%green,'Description',bc%endc
  write(ifile, "(80('-'))")

end subroutine ! write_info_header


!
!
!
subroutine write_pinfo(p, ifile)
  use m_color, only : bc
  implicit none
  !! external
  type(param_t), intent(in) :: p
  integer, intent(in) :: ifile
  !! internal
  integer :: n, i

  write(ifile, '(a,a,a,a)')      'Parameter type: ', bc%gray, p%ctype, bc%endc
  if(allocated(p%descr)) then
    n = size(p%descr)
    if(n==1) then
      write(ifile, '(a,a,a,a)')'Description: ', bc%green, trim(p%descr(1)), bc%endc
    else
      write(ifile, '(a)') '  Description: '
      do i=1,n
        write(ifile, '(a,a,a)')bc%green, trim(p%descr(i)), bc%endc
      enddo
    endif

  else
    write(ifile, '(a)') '  No description for this parameter.'
  endif
  write(ifile, '(a,i9)') '    Input line: ', p%iline
  write(ifile, '(a,i9)') '  Input status: ', p%ios
  write(ifile, '(a,a)')  '  Source fname: ', trim(p%fname)
  write(ifile, '(a,i9)') '  Source fline: ', p%fline    
  
end subroutine ! write_pinfo


!
!
!
subroutine init_descr(a,b)
  implicit none
  !! external
  character(*), intent(in) :: a
  character(MAXLEN), allocatable, intent(inout) :: b(:)
  !! internal

  _dealloc(b)
  if(len_trim(a)<1) return
  allocate(b(1))
  b(1) = a

end subroutine ! init_descr_descr


!
!
!
subroutine reset(p)
  implicit none
  type(param_t), intent(inout) :: p


  _dealloc(p%dcv)
  _dealloc(p%ddv)
  _dealloc(p%icv)
  _dealloc(p%idv)
  _dealloc(p%ccv)
  _dealloc(p%cdv)
  _dealloc(p%lcv)
  _dealloc(p%ldv)
  p%ctype = ''
  p%pname = ''
  _dealloc(p%descr)
  p%fname = ''
  p%fline = -999
  p%ios   = -999

end subroutine !reset

!
!
!
subroutine init_p_d(v, vv)
  implicit none
  real(8), intent(in) :: v
  real(8), intent(inout), allocatable :: vv(:)
  
  if(allocated(vv)) then
    if(size(vv)/=1) deallocate(vv)
  endif
  if(.not. allocated(vv)) then
    allocate(vv(1))
  endif

  vv(1) = v

end subroutine !init_p_d

!
!
!
subroutine init_p_c(v, vv)
  implicit none
  character(*), intent(in) :: v
  character(MAXLEN), intent(inout), allocatable :: vv(:)
  
  if(allocated(vv)) then
    if(size(vv)/=1) deallocate(vv)
  endif
  if(.not. allocated(vv)) then
    allocate(vv(1))
  endif

  vv(1) = v

end subroutine !init_p_c

!
!
!
subroutine init_p_l(v, vv)
  implicit none
  logical, intent(in) :: v
  logical, intent(inout), allocatable :: vv(:)
  
  if(allocated(vv)) then
    if(size(vv)/=1) deallocate(vv)
  endif
  if(.not. allocated(vv)) then
    allocate(vv(1))
  endif

  vv(1) = v

end subroutine !init_p_l

!
!
!
subroutine init_p_i(v, vv)
  implicit none
  integer, intent(in) :: v
  integer, intent(inout), allocatable :: vv(:)
  
  if(allocated(vv)) then
    if(size(vv)/=1) deallocate(vv)
  endif
  if(.not. allocated(vv)) then
    allocate(vv(1))
  endif

  vv(1) = v

end subroutine !init_p_i


end module m_param
