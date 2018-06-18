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

module m_input

!> @file m_input 
!! This is to define the structure input_t that would hold the input parameters
!! once they are read from a file and then can be interpreted with some 
!! default values later practically from anywhere in the program.


#include "m_define_macro.F90"
  use m_die, only : die
  use m_warn, only : warn
  
  implicit none
  private die
  private warn
  
  interface init_parameter

    module procedure init_param_char
    module procedure init_param_char_nodef
    module procedure init_param_complex
    
    module procedure init_param_real_nodef
    module procedure init_param_real_flag

    module procedure init_param_real_array
    
    module procedure init_param_logical

    module procedure init_param_int
    module procedure init_param_int_array
    
  end interface

  integer, parameter :: MAXLEN = 256

  !> The structure to hold the input file as a collection of strings.
  type input_t
    integer :: verbosity =-999
    character(20) :: io_units = '';
    character(20) :: eps_units = '';
    character(MAXLEN), allocatable :: lines(:)       ! The input file is read into this array
  end type ! input_t
  !! END of INPUT TYPE
    
  contains

!> \fn Some input parameters are very common and they are interpreted just after 
!! all strings from input file are stored.
character(20) function get_io_units(inp)
  implicit none
  type(input_t), intent(in) :: inp
  if(len_trim(inp%io_units)<1) _die('len_trim(io_units)<1')
  get_io_units = inp%io_units
end function ! get_io_units

!
!
!
character(20) function get_eps_units(inp)
  implicit none
  type(input_t), intent(in) :: inp
  if(len_trim(inp%eps_units)<1) _die('len_trim(eps_units)<1')
  get_eps_units = inp%eps_units
end function ! get_eps_units

!
!
!
integer function get_verbosity(inp)
  implicit none
  type(input_t), intent(in) :: inp
  if(inp%verbosity==-999) _die('iv<1')
  get_verbosity = inp%verbosity
end function ! get_verbosity


!
! Reads all the input parameters from a file at once and puts it to a structure input_t
!
subroutine  input_calc_params(fname, inp, iv_in, ilog);
  use m_log, only : log_size_note, die
  use m_io, only : get_free_handle
  implicit none
  character(*), intent(in) :: fname
  type(input_t), intent(inout) :: inp
  integer, intent(in) :: iv_in, ilog
  !! internal
  integer :: iv, ido, ios, i, ifile
  character(1000) :: fname_rep

  !! Input
  call input_params_lines(fname, inp, iv_in, ilog)
  call init_parameter('verbosity', inp, 1, inp%verbosity, iv_in)
  iv = inp%verbosity
  call init_parameter('io_units', inp, 'eV', inp%io_units, iv)
  call init_parameter('eps_units', inp, 'eV', inp%eps_units, iv)
  
  call init_parameter('report_input', inp, 1, ido, iv)
  if(ido>0) then
    ifile=get_free_handle()
    fname_rep = "report_input.txt"
    open(ifile, file=fname_rep, action="write", iostat=ios)
    if(ios/=0) _die('ioerror')
    do i=1,size(inp%lines); write(ifile,'(i5,2x,a)') i, trim(inp%lines(i)); enddo
    close(ifile)
    call log_size_note('written', fname_rep, iv)  
  endif 
end subroutine !input_calc_params

!
! Reads all the input parameters from a file at once and puts it to a structure input_t
!
subroutine  input_params_lines(fname, inp, iv_in, ilog);
  use m_log, only : die
  use m_io, only : get_free_handle
  implicit none
  
  character(*), intent(in) :: fname
  type(input_t), intent(inout) :: inp
  integer, intent(in) :: iv_in, ilog

  !! Input
  ! correlate with "read(f, '(1000A1)', iostat=ios) (line_char(j),j=1,M); "
  integer, parameter :: M=MAXLEN 
  integer :: ios, f,j, iline, lt_line, step, iv
  character(1), dimension(M) :: line_char
  character(len=M)           :: line

  iv = iv_in - 1
  
  f = get_free_handle();
  line = '';
  open(f, file=fname, action='read', status='old', iostat=ios);
  if(ios/=0) call die('input_calc_params: error: file '//trim(fname));
  do step=1,2
    iline = 0
    ios = 0
    do while(ios==0)
      !! Read line
      line_char = ''; line='';
      read(f, '(1000A1)', iostat=ios) (line_char(j),j=1,M);
      do j=1,M;
        !if(int(line_char(j))<32) exit
        if( iachar(line_char(j))==13 ) exit ! minimal treatment against CRLF line endings
        line(j:j)=line_char(j);
      enddo;
      lt_line = len_trim(line)
      !! END of Read line
!      if(lt_line<1) cycle
      if(iv>1 .and. step==2)write(ilog,'(1x,a)') trim(line)
      iline = iline + 1
      if(step==2) inp%lines(iline) = trim(adjustl(line))
    enddo ! while(ios==0)
    if(step==1) then; if(allocated(inp%lines)) deallocate(inp%lines); allocate(inp%lines(iline)); endif
    rewind(f)
  enddo ! step
  if(iv>0)write(ilog,'(a,i6)') 'input_calc_params: iline', iline
  if(.not. allocated(inp%lines)) then; 
    allocate(inp%lines(1)); inp%lines(1) = 'empty?'; endif
  
  if(iv>1) then
    do iline=1,size(inp%lines);  write(ilog,'(i5,a)') iline, trim(inp%lines(iline)); enddo
  endif  
  close(f)
  
end subroutine !input_params_lines

!
!
!
subroutine init_param_char(param_name, inp, def_value, actual_value, iv)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in)  :: param_name
  type(input_t), intent(in) :: inp
  character(*), intent(in)  :: def_value
  character(*), intent(out) :: actual_value
  integer, intent(in)       :: iv
  integer :: l
  actual_value = def_value
  l = get_line_num(param_name, inp)
  if(l>0) then
    call get_param_char(param_name, inp%lines(l), actual_value)
  endif
  call log_size_note(param_name, actual_value, iv, 'input_parameter');
  
end subroutine ! init_param_char

!
!
!
character(MAXLEN) function get_cline(inp, iline)
  use m_upper, only : upper
  implicit none
  !! external
  type(input_t), intent(in) :: inp
  integer, intent(in) :: iline
  !! internal
  integer :: n
  n = get_nlines(inp)
  if(iline<1 .or. iline>n) then
    write(6,'(a,a,i8)') __FILE__, ' at ', __LINE__
    write(6,'(a,2i8)') 'iline, n ', iline, n
    stop 'iline<1 .or. iline>n'
  endif
  get_cline = upper(inp%lines(iline))
end function ! get_cline  

!
!
!
integer function get_nlines(inp)
  implicit none
  type(input_t), intent(in) :: inp
  if(.not. allocated(inp%lines)) then
    write(6,'(a,a,i8)') __FILE__, ' at ', __LINE__
    stop 'not allocated(lines)'
  endif
  get_nlines = size(inp%lines)
end function ! get_nlines

!
!
!
subroutine init_param_char_nodef(param_name, inp, actual_value, iv)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in)  :: param_name
  type(input_t), intent(in) :: inp
  character(*), intent(out) :: actual_value
  integer, intent(in)       :: iv
  integer :: l
  l = get_line_num(param_name, inp)
  actual_value = ''
  if(l>0) call get_param_char(param_name, inp%lines(l), actual_value)
  if(len_trim(actual_value)<1) then
    write(6,*) 'param_name: ', param_name
    write(0,*) 'param_name: ', param_name
    _die('no default value and no parameter')
  endif
  call log_size_note(param_name, actual_value, iv, 'input_parameter');
  
end subroutine ! init_param_char_nodef

!
!
!
subroutine get_param_char(param_name, line, res)
  implicit none
  character(*), intent(in) :: param_name, line
  character(*), intent(inout) :: res
  integer :: lt, ios, j
  integer, parameter :: M=1000
  character :: line_char(M)
  character(M) :: line_tmp
  
  lt = len_trim(param_name)
  !! Read line
  line_char = ''; res='';
  line_tmp = adjustl(line(lt+1:))
  read(line_tmp, '(1000A1)', iostat=ios) (line_char(j),j=1,M);
  do j=1,M;
    if( iachar(line_char(j))==32 ) exit ! stop of space character appears
    if( iachar(line_char(j))==13 ) exit ! minimal treatment against CRLF line endings
    res(j:j)=line_char(j);
  enddo;
  !! END of Read line

end subroutine ! get_param_char

!
!
!
subroutine init_param_complex(param_name, inp, def_value, actual_value, iv)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in)  :: param_name
  type(input_t), intent(in) :: inp
  complex(8), intent(in)    :: def_value
  complex(8), intent(out)   :: actual_value
  integer, intent(in)       :: iv
  integer :: l
  actual_value = def_value
  l = get_line_num(param_name, inp)
  if(l>0) actual_value = get_param_complex(param_name, inp%lines(l))
  call log_size_note(param_name, actual_value, iv);
  
end subroutine ! init_param_complex

!
!
!
function get_param_complex(param_name, line) result(res)
  use m_log, only : die
  implicit none
  character(*), intent(in) :: param_name, line
  complex(8) :: res
  integer :: lt, ios
  
  lt = len_trim(param_name)
  read(line(lt+1:),*, iostat=ios) res
  if(ios/=0) _die('ios/=0') 
  
end function ! get_param_complex

!
!
!
subroutine init_param_int(param_name, inp, def_value, actual_value, iv, do_not_report)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in) :: param_name
  type(input_t), intent(in) :: inp
  integer, intent(in)   :: def_value
  integer, intent(out)  :: actual_value
  integer, intent(in)   :: iv
  integer, intent(in), optional :: do_not_report
  integer :: l
  actual_value = def_value  
  l = get_line_num(param_name, inp)
  if(l>0) actual_value = get_param_int(param_name, inp%lines(l))
  if(actual_value<0) actual_value = def_value  
  if(present(do_not_report)) then
    if(do_not_report>0) return
  endif
  call log_size_note(param_name, actual_value, iv, 0, 'input_parameter');
  
end subroutine ! init_param_int

!
!
!
function get_param_int(param_name, line) result(res)
  implicit none
  !! external
  character(*), intent(in) :: param_name, line
  integer :: res
  !! internal
  integer :: lt, ios
  
  lt = len_trim(param_name)
  read(line(lt+1:),*,iostat=ios) res
  if(ios/=0) then
    write(6,*) 'get_param_int: ios', ios
    call warn('get_param_int: ios/=0 '//__FILE__, __LINE__)
    res = -1
  endif  
  
end function ! get_param_int

!
!
!
subroutine init_param_int_array(param_name, inp, def_value, actual_value, iv, do_not_report)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in) :: param_name
  type(input_t), intent(in) :: inp
  integer, intent(in)   :: def_value(:)
  integer, intent(inout), allocatable :: actual_value(:)
  integer, intent(in)   :: iv
  integer, intent(in), optional :: do_not_report
  integer :: l

  _dealloc(actual_value)
  allocate(actual_value(size(def_value)))
  actual_value = def_value
  l = get_line_num(param_name, inp)
  if(l>0) call get_param_int_array(param_name, inp%lines(l), def_value, actual_value)
  if(present(do_not_report)) then; if(do_not_report>0) return; endif
  call log_size_note(param_name, actual_value, iv, 0, 'input_parameter')

end subroutine ! init_param_int_array

!!
!!
!!
!subroutine get_param_int_array(param_name, line, res)
!  implicit none
!  !! external
!  character(*), intent(in) :: param_name, line
!  integer, allocatable, intent(inout) :: res(:)
!  !! internal
!  integer :: lt, ios, i, f, n,d,a,j,imax
!  integer :: aux(100000), nn(20)
!  
!  lt = len_trim(param_name)
!  d = 0
!  n = 0
!  f =-999
!  nn = 0
!  a = -999
!  imax = len_trim(line)-lt
!  aux = -999
!  do i=1,imax
!    read(line(lt+i:lt+i),*,iostat=ios)f
!    if(ios==0 .and. d==0) n=n+1
!    if(ios==0) then; 
!      d=d+1; 
!      if(d>20) _die('')
!      nn(d) = f; endif
!    if((ios<0 .and. n>0 .and. d>0) .or. (i==imax .and. d>0)) then
!      a = 0
!      do j=1,d; a = a + nn(j)*10**(d-j); enddo
!      if(n>100000) _die('')
!      aux(n) = a
!      nn = 0
!    endif
!!    write(6,'(5i5,3x,100i5)') i, ios, d, f, n, nn(1:2), a
!    if(ios<0) then; d=0; nn=0; endif
!  enddo ! imax
!     
!  _dealloc(res)
!  if(n>0) then   
!    allocate(res(n))
!    res(1:n) = aux(1:n)
!  endif
!
!end subroutine ! get_param_int_array

!
!
!
subroutine get_param_int_array(param_name, line, def_value, res)
  implicit none
  !! external
  character(*), intent(in) :: param_name, line
  integer, intent(in)   :: def_value(:)
  integer, intent(inout) :: res(:)
  !! internal
  character(2000) :: line_wo_pname
  integer :: ios
  if(size(def_value)>size(res)) _die('size(def_value)>size(res)?')
  line_wo_pname = line(len_trim(param_name)+1:)
  read(line_wo_pname,*,iostat=ios)res
  if(ios/=0) then
    write(0,'(a,a)') 'param_name: ', trim(param_name)
    _warn('error in the reading of the parameter!')
    _warn('default will be used!')
    res = def_value
  endif  
  
end subroutine get_param_int_array 

!
!
!
subroutine init_param_real_array(param_name, inp, def_value, actual_value, iv, do_not_report)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in) :: param_name
  type(input_t), intent(in) :: inp
  real(8), intent(in)   :: def_value(:)
  real(8), intent(inout), allocatable :: actual_value(:)
  integer, intent(in)   :: iv
  integer, intent(in), optional :: do_not_report
  integer :: l

  _dealloc(actual_value)
  allocate(actual_value(size(def_value)))
  actual_value = def_value
  l = get_line_num(param_name, inp)
  if(l>0) call get_param_real_array(param_name, inp%lines(l), def_value, actual_value)
  if(present(do_not_report)) then; if(do_not_report>0) return; endif
  call log_size_note(param_name, actual_value, iv, 'input_parameter')

end subroutine ! init_param_real_array

!
!
!
subroutine get_param_real_array(param_name, line, def_value, res)
  implicit none
  !! external
  character(*), intent(in) :: param_name, line
  real(8), intent(in)   :: def_value(:)
  real(8), intent(inout) :: res(:)
  !! internal
  character(2000) :: line_wo_pname
  integer :: ios
  if(size(def_value)>size(res)) _die('size(def_value)>size(res)?')
  line_wo_pname = line(len_trim(param_name)+1:)
  read(line_wo_pname,*,iostat=ios)res
  if(ios/=0) then
    write(0,'(a,a)') 'param_name: ', trim(param_name)
    _warn('error in the reading of the parameter!')
    _warn('default will be used!')
    res = def_value
  endif  
  
end subroutine ! get_param_real_array

!
!
!
subroutine init_param_real_nodef(param_name, inp, actual_value, iv, do_not_report)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in) :: param_name
  type(input_t), intent(in) :: inp
  real(8), intent(out)  :: actual_value
  integer, intent(in)   :: iv
  integer, intent(in), optional :: do_not_report
  integer :: l
  l = get_line_num(param_name, inp)
  if(l>0) then
    actual_value = get_param_real(param_name, inp%lines(l))
  else
    write(6,*) 'param_name: ', param_name
    write(0,*) 'param_name: ', param_name
    _die('no default value and no parameter')
  endif    
  if(present(do_not_report)) then; if(do_not_report>0) return; endif
  call log_size_note(param_name, actual_value, iv, 'input_parameter');
  
end subroutine ! init_param_real_nodef

!
!
!
function get_param_real(param_name, line) result(res)
  implicit none
  character(*), intent(in) :: param_name, line
  real(8) :: res
  integer :: lt, ios
  
  lt = len_trim(param_name)
  read(line(lt+1:),*, iostat=ios) res
  if(ios/=0) then
    call warn('get_param_real: ios/=0 '//__FILE__, __LINE__)
    res = -1
  endif  
  
end function ! get_param_real


!
!
!
subroutine init_param_real_flag(param_name, inp, def_value, actual_value, iv, do_not_report)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in) :: param_name
  type(input_t), intent(in) :: inp
  real(8), intent(in)   :: def_value
  real(8), intent(out)  :: actual_value
  integer, intent(in)   :: iv
  integer, intent(in), optional :: do_not_report
  integer :: l, ios, lt
  actual_value = def_value  
  l = get_line_num(param_name, inp)
  ios = -1
  if(l>0) then
    lt = len_trim(param_name)
    read(inp%lines(l)(lt+1:),*, iostat=ios)actual_value 
  endif  
  if(ios/=0) actual_value = def_value

  if(present(do_not_report)) then; if(do_not_report>0) return; endif
  call log_size_note(param_name, actual_value, iv, 'input_parameter');
  
end subroutine ! init_param_real

!
!
!
subroutine init_param_logical(param_name, inp, def_value, actual_value, iv)
  use m_log, only : log_size_note
  implicit none
  character(*), intent(in) :: param_name
  type(input_t), intent(in) :: inp
  logical, intent(in)   :: def_value
  logical, intent(out)  :: actual_value
  integer, intent(in)   :: iv
  integer :: l
  actual_value = def_value
  l = get_line_num(param_name, inp)
  if(l>0) actual_value = get_param_logical(param_name, inp%lines(l))
  call log_size_note(param_name, actual_value, iv);
  
end subroutine ! init_param_logical

!
!
!
function get_param_logical(param_name, line) result(res)
  use m_log, only : die
  implicit none
  character(*), intent(in) :: param_name, line
  logical :: res
  integer :: lt, ios
  
  lt = len_trim(param_name)
  read(line(lt+1:),*, iostat=ios) res
  if(ios/=0) call die('get_param_logical: ios/=0 '//__FILE__, __LINE__)
  
end function ! get_param_logical

!
!
!
function get_line_num(param_name, inp) result(iline_found)
  use m_upper, only : upper
  implicit none
  type(input_t), intent(in) :: inp
  character(*), intent(in) :: param_name
  integer :: iline_found
  !! internal
  character(MAXLEN) :: upper_param_name, upper_line
  integer :: iline, nsize, lt
  if(.not. allocated(inp%lines)) then
    call warn("Did you forget to call input_calc_params('tddft_lr.inp', inp, iv, ilog) ? ")
    _die('.not. allocated(inp%lines)')
  endif
  nsize = size(inp%lines)
  upper_param_name = upper(param_name)
  lt = len_trim(param_name)
  iline_found = -1
  do iline=1,nsize
    if(lt+1>len(inp%lines(iline))) cycle
    upper_line = upper(inp%lines(iline))
    if(trim(upper_param_name)//' A' /= upper_line(1:lt+1)//'A') cycle
    if(iline_found==-1) then
      iline_found = iline
    else
      write(6,*) __FILE__, __LINE__
      write(6,'(a)') 'parameter '//trim(param_name)//' appears at least twice'
      write(6,'(a,1x,2i6)') 'check lines: ', iline_found, iline
      write(6,'(a,1x,i6,1x,a)') 'line: ', iline_found, trim(inp%lines(iline_found))
      write(6,'(a,1x,i6,1x,a)') 'line: ', iline, trim(inp%lines(iline))
      iline_found = iline
      _die('appears twice');
    endif  
  enddo
end function ! get_line_num

end module !m_input


