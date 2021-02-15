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

module m_log

  !!
  !! Because of this use statements the subroutines die(...) and warn(...)
  !! could be "used" via the module m_log. 
  !!
  use m_die, only : die
!  use m_warn, only : warn
  !use m_log_type, only : log_t
  
  implicit none

#include "m_define_macro.F90"
 
  interface log_timing_note
    module procedure log_timing_note2;
  end interface ! log_timing_note

  interface log_size_note
    module procedure log_size_note_character;
    module procedure log_size_note_double;
    module procedure log_size_note_double_array;
    module procedure log_size_note_complex8;
    module procedure log_size_note_i4;
    module procedure log_size_note_i8;
    module procedure log_size_note_i4_array;
    module procedure log_size_note_i8_array;
    module procedure log_size_note_logical
  end interface log_size_note ! log_size_note

  integer :: ilog = 6, ilog_save = -999;
  character(1000) :: log_filename='LOG-DOMIPROD';

  integer :: itimes = -999;
  character(1000) :: times_filename='TIMES-DOMIPROD';

  integer :: isizes = -999;
  character(1000) :: sizes_filename='SIZES-DOMIPROD';

  character(1000) :: memory_filename='MEMORY-DOMIPROD'

#ifdef USE_MEMORY_NOTE
  integer, private :: log_memory_note_first = 1
#endif
  
  contains

!
! Initialize file names and file units (numbers) for log- and times- files.
!
subroutine init_logs(stdout_to_file, fname_suffix, mynode, iv, ilog, cdate, ctime)
  use m_io, only : get_free_handle
  use m_color, only : bc, color_on  
  implicit none
  character(len=*), intent(in) :: fname_suffix
  integer, intent(in) :: iv, stdout_to_file, mynode
  integer, intent(inout) :: ilog
  character(*), intent(in), optional :: cdate, ctime
  ! internal
  integer :: ios

  call color_on(bc)

  if(present(cdate) .and. .not. present(ctime)) then
    write(6,*) 'init_logs: ', trim(fname_suffix), ' compiled by ', cdate
  else if (.not. present(cdate) .and. present(ctime)) then
    write(6,*) 'init_logs: ', trim(fname_suffix), ' compiled at', ctime
  else if (present(cdate) .and. present(ctime)) then
    write(6,*) 'init_logs: ', trim(fname_suffix), ' compiled by ', cdate, ' at ', ctime
  endif

  !! Init iv and output files
  if(stdout_to_file>0) then
    write(log_filename, '(a,i0.4,a)') 'LOG.DOMIPROD-', mynode, trim(fname_suffix);
    ilog = get_free_handle()
    open(ilog, file=log_filename, action='write', iostat=ios);
    if(ios==0) then;
      if(iv>0) write(6,*)'init_logs: ', trim(log_filename), ' is open...';
    else
      write(0,*) 'init_logs: ', trim(log_filename), ' cannot be open';
    endif
  endif

  if (mynode>0) then
    ilog = 8
    open(ilog, file=trim(log_filename), action='write')
    write(6,*) 'init_parallel: see also ', trim(log_filename)
  endif

  write(times_filename, '(a,i0.4,a)') 'TIMES.DOMIPROD-', mynode, trim(fname_suffix);
  itimes=get_free_handle();
  open(itimes, file=times_filename, action='write', iostat=ios);
  if(ios==0) then;
    if(iv>0) write(6,*)'init_logs: ', trim(times_filename), ' is open...';
  else
    write(0,*) 'init_logs: ', trim(times_filename), ' cannot be open';
  endif

  write(sizes_filename, '(a,i0.4,a)') 'SIZES.DOMIPROD-', mynode, trim(fname_suffix);
  isizes=get_free_handle();
  open(isizes, file=sizes_filename, action='write', iostat=ios);
  if(ios==0) then;
    if(iv>0) write(6,*)'init_logs: ', trim(sizes_filename), ' is open...';
  else
    write(0,*) 'init_logs: ', trim(sizes_filename), ' cannot be open.';
  endif

  write(memory_filename, '(a,i0.4,a)') 'MEMORY.DOMIPROD-', mynode, trim(fname_suffix);
  if(iv>0) write(6,*)'init_logs: ', trim(memory_filename), ' will be open...';

end subroutine !init_logs

!!
!!
!!
subroutine log_memory_note(mesg, iv_in, line, iter)
  implicit none 
  character(*), intent(in), optional :: mesg
  integer, intent(in), optional :: iv_in
  integer, intent(in), optional :: line, iter

#ifdef USE_MEMORY_NOTE
  !! internal
  integer :: iv
  integer(4) :: pid
  character(1000) :: fname, cmd, message
  character(60)  :: string
  iv = 1
  if(present(iv_in)) iv = iv_in
 
  message = 'unknown message'
  if(present(mesg)) message = mesg 

  fname = trim(memory_filename)
  if(present(line) .and. present(iter)) then
    write(string,'(a,1x,i8,a,i8,a)') adjustr(trim(message)), line, ' iter ', iter
  else if(present(line) .and. (.not. present(iter))) then
    write(string,'(a,1x,i8)') adjustr(trim(message)), line
  else 
    write(string,'(a)') adjustr(trim(message))
  endif 
    
  if(log_memory_note_first==1) then
    cmd = 'echo "">'//trim(fname)
    write(6,*) trim(cmd)
    call system(cmd)
  endif

  call getpid(pid)
  write(cmd,'(a,i8,a)')&
    'printf "$(date +%s) $(ps h -p ',pid,' -o rsz,pmem,vsz)'&
    //' '//trim(string)//' '//'"\\n>>'//trim(fname)
  if(iv>0)write(ilog,'(a,a,a,a)') ' memory_note:', trim(cmd),' ',trim(string)
  call system(cmd)

  log_memory_note_first = 0
#endif 

end subroutine ! log_memory_note

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_timing_note2(filename, total_time, iv1, mynode1)
  use m_color, only : bc
#ifdef TIMING
  use m_io, only : get_free_handle
#endif    
  implicit none
  character(*), intent(in)      :: filename
  real(8), intent(in)           :: total_time
  integer, optional, intent(in) :: iv1
  integer, optional, intent(in) :: mynode1
  
#ifdef TIMING
  integer :: handle, mynode, iv
  character(200) :: filename_real;
  character( 8)  :: date
  character(10)  :: time
  character(10)  :: zone
  integer        :: values(8) 
  iv=0; 
  if(present(iv1)) iv=iv1;

  call date_and_time(date, time, zone, values)

  if(itimes>0) then ! if the file is open, then write a note to this file...
            write(itimes, '(a50,2x,g16.8,1x,a9,1x,a9)') &
                                  trim(filename), total_time, date, time;
    if(iv>0)write(ilog,   '(a,a50,2x,g16.8,1x,a,a9,1x,a9,a)')&
      bc%GREEN//' timing_note: ', trim(filename), total_time, date, time,bc%ENDC;

  else ! ... otherwise, open a short note file
    mynode=0; if(present(mynode1)) mynode=mynode1

    write(filename_real, '(a,i0.4,a)') trim(filename), mynode, '.txt';
    handle = get_free_handle();
    open(handle, file=trim(filename_real), action='write');
    write(handle, *) total_time, date, time;
    close(handle);
    if(iv>0)write(ilog,*) 'timing_note: ', trim(filename_real);
  endif
#endif

end subroutine ! log_timing_note2

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_character(name, note, iv1, ctag)
  implicit none
  character(*), intent(in)      :: name
  character(*), intent(in)           :: note
  integer, optional, intent(in) :: iv1
  character(*), optional, intent(in) :: ctag


  integer :: iv
  iv=0;  if(present(iv1)) iv=iv1;

  if(isizes>0) then
    if(present(ctag)) then
      write(isizes,   '(a50,2x,a,3x,a)') trim(name), trim(note), trim(ctag);
    else
      write(isizes,   '(a50,2x,a)') trim(name), trim(note);
    endif  
    if(iv>0)write(ilog, '(a,a50,2x,a)')' size_note: ', trim(name), trim(note);
  else
    _die('init log file first')
  endif

end subroutine ! size_note

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_double(vname, size_MB, iv1, ctag)
  character(*), intent(in)      :: vname
  real(8), intent(in)           :: size_MB
  integer, optional, intent(in) :: iv1
  character(*), optional, intent(in) :: ctag

  integer :: iv
  iv=0;  if(present(iv1)) iv=iv1;
  if(isizes>0) then
    if(present(ctag)) then
      write(isizes,'(a50,2x,g18.9,3x,a)') trim(vname), size_MB, trim(ctag);
    else
      write(isizes,'(a50,2x,g18.9)') trim(vname), size_MB;
    endif
      
    if(iv>0)write(ilog, '(a,a50,2x,g18.9)')' size_note: ', trim(vname), size_MB;
  else
    _die('init log file first')
  endif

end subroutine ! size_note_double

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_double_array(vname, vals, iv1, ctag)
  character(*), intent(in)      :: vname
  real(8), intent(in), allocatable :: vals(:)
  integer, optional, intent(in) :: iv1
  character(*), optional, intent(in) :: ctag

  integer :: iv, lb, ub, n
  character(200) :: fstring,fstring_out;
  
  lb=0; ub=-1;
  if(allocated(vals)) then; 
    lb = lbound(vals,1)
    ub = ubound(vals,1)
  endif
  iv=0;  if(present(iv1)) iv=iv1;
  
  n = ub - lb + 1
  if(n>0) then
    write(fstring,'(a,i10.0,a)')'(a50,2x,',size(vals),'g18.9,3x,a)'
    write(fstring_out,'(a,i10.0,a)')'(a,a50,2x,',size(vals),'g18.9,3x,a)'
  else 
    write(fstring,'(a,i10.0,a)')'(a50,2x,3x,a)'
    write(fstring_out,'(a,i10.0,a)')'(a,a50,2x,3x,a)'
  endif
  
  !write(6,'(a30,a200)') 'fstring',fstring
  !write(6,'(a30,a200)') 'fstring_out',fstring_out
    
  if(isizes>0) then
    if(present(ctag)) then
      if(n>0) then
        write(isizes,fstring) trim(vname), vals, trim(ctag);
      else
        write(isizes,fstring) trim(vname), trim(ctag);
      endif  
    else
      if(n>0) then
        write(isizes,fstring) trim(vname), vals;
      else
        write(isizes,fstring) trim(vname);
      endif    
    endif
    if(n>0) then
      if(iv>0)write(ilog,fstring_out)' size_note: ', trim(vname), vals;
    else
      if(iv>0)write(ilog,fstring_out)' size_note: ', trim(vname);
    endif    
  else
    _die('init log file first')
  endif

end subroutine ! size_note_double_array


!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_complex8(vname, param, iv1, mynode1)
  use m_io, only : get_free_handle
  implicit none
  character(*), intent(in)      :: vname
  complex(8), intent(in)        :: param
  integer, optional, intent(in) :: iv1
  integer(2), optional, intent(in) :: mynode1

  integer :: handle, mynode, iv
  character(200) :: filename_real;

  iv=0;  if(present(iv1)) iv=iv1;

  if(isizes>0) then
          write(isizes,   '(a50,2x,g18.9,g18.9)') trim(vname), param;
    if(iv>0)write(ilog, '(a,a50,2x,g18.9,g18.9)')' size_note: ', trim(vname), param;
  else
    mynode=0; if(present(mynode1)) mynode=mynode1

    write(filename_real, '(a,i0.4,a)') trim(vname), mynode, '.txt';
    handle = get_free_handle();
    open(handle, file=trim(filename_real), action='write');
    write(handle, *) param
    close(handle);
    if(iv>0)write(ilog,*) 'size_note: ', trim(filename_real);
  endif

end subroutine ! size_note_complex8

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_i4(name, dim1, iv1, mynode1, ctag)
  use m_io, only : get_free_handle
  use m_color, only : bc
  implicit none
  character(*), intent(in)      :: name
  integer(4), intent(in)           :: dim1
  integer, optional, intent(in) :: iv1
  integer, optional, intent(in) :: mynode1
  character(*), intent(in), optional :: ctag

  !! internal
  integer :: handle, mynode, iv
  character(200) :: filename_real;

  iv=0;     if(present(iv1)) iv=iv1;

  if(isizes>0) then
    if(present(ctag)) then
      write(isizes,'(a50,2x,i10,3x,a)') trim(name), dim1, trim(ctag);
    else 
      write(isizes,'(a50,2x,i10)') trim(name), dim1
    endif  
    if(iv>0)write(ilog, '(a,a50,2x,i10,a)')&
      bc%blue//' size_note: ', trim(name), dim1,bc%endc;
  else
    mynode=0; if(present(mynode1)) mynode=mynode1

    write(filename_real, '(a,i0.4,a)') trim(name), mynode, '.txt';
    handle = get_free_handle();
    open(handle, file=trim(filename_real), action='write');
    write(handle, *) dim1
    close(handle);
    if(iv>0)write(ilog,*) 'size_note: ', trim(filename_real);
  endif

end subroutine ! size_note_i4

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_i8(name, dim1, iv1, mynode1, ctag)
  use m_io, only : get_free_handle
  use m_color, only : bc  
  implicit none
  character(*), intent(in)      :: name
  integer(8), intent(in)        :: dim1
  integer, optional, intent(in) :: iv1
  integer, optional, intent(in) :: mynode1
  character(*), intent(in), optional :: ctag
  
  integer :: handle, mynode, iv
  character(200) :: filename_real;

  iv=0;     if(present(iv1)) iv=iv1;

  if(isizes>0) then
    if(present(ctag)) then
      write(isizes,   '(a50,2x,i10,3x,a)') trim(name), dim1, trim(ctag);
    else
      write(isizes,   '(a50,2x,i10)') trim(name), dim1;
    endif  
    if(iv>0)write(ilog, '(a,a50,2x,i10,a)') bc%blue//' size_note: ', trim(name), dim1, bc%endc;
  else
    mynode=0; if(present(mynode1)) mynode=mynode1

    write(filename_real, '(a,i0.4,a)') trim(name), mynode, '.txt';
    handle = get_free_handle();
    open(handle, file=trim(filename_real), action='write');
    write(handle, *) dim1
    close(handle);
    if(iv>0)write(ilog,*) 'size_note: ', trim(filename_real);
  endif

end subroutine ! size_note

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_i4_array(name, dim1, iv1, mynode1, ctag)
  use m_io, only : get_free_handle
  implicit none
  character(*), intent(in)      :: name
  integer(4), intent(in), allocatable :: dim1(:)
  integer, optional, intent(in) :: iv1
  integer, optional, intent(in) :: mynode1
  character(*), intent(in), optional :: ctag
  
  integer :: handle, mynode, iv
  character(200) :: filename_real;

  if(.not. allocated(dim1)) return

  iv=0;     if(present(iv1)) iv=iv1;

  if(isizes>0) then
    if(present(ctag)) then
      write(isizes,   '(a50,2x,a,3x,10000i10)') trim(name), trim(ctag), dim1;
    else
      write(isizes,   '(a50,2x,10000i10)') trim(name), dim1;
    endif  
    if(iv>0)write(ilog, '(a,2x,a50,2x,10000i10)')' size_note: ', trim(name), dim1;
  else
    mynode=0; if(present(mynode1)) mynode=mynode1

    write(filename_real, '(a,i0.4,a)') trim(name), mynode, '.txt';
    handle = get_free_handle();
    open(handle, file=trim(filename_real), action='write');
    write(handle, *) dim1
    close(handle);
    if(iv>0)write(ilog,*) 'size_note: ', trim(filename_real);
  endif

end subroutine ! log_size_note_i4_array

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_i8_array(name, dim1, iv1, mynode1, ctag)
  use m_io, only : get_free_handle
  implicit none
  character(*), intent(in)      :: name
  integer(8), intent(in), allocatable :: dim1(:)
  integer, optional, intent(in) :: iv1
  integer, optional, intent(in) :: mynode1
  character(*), intent(in), optional :: ctag
  
  integer :: handle, mynode, iv
  character(200) :: filename_real;

  if(.not. allocated(dim1)) return
  iv=0;     if(present(iv1)) iv=iv1;

  if(isizes>0) then
    if(present(ctag)) then
      write(isizes,   '(a50,2x,a,3x,10000i10)') trim(name), trim(ctag), dim1;
    else
      write(isizes,   '(a50,2x,10000i10)') trim(name), dim1;
    endif  
    if(iv>0)write(ilog, '(a,2x,a50,2x,10000i10)')' size_note: ', trim(name), dim1;
  else
    mynode=0; if(present(mynode1)) mynode=mynode1

    write(filename_real, '(a,i0.4,a)') trim(name), mynode, '.txt';
    handle = get_free_handle();
    open(handle, file=trim(filename_real), action='write');
    write(handle, *) dim1
    close(handle);
    if(iv>0)write(ilog,*) 'size_note: ', trim(filename_real);
  endif

end subroutine ! log_size_note_i8_array

!!
!! Writes a short "note" to a file with a given name filename
!!
subroutine log_size_note_logical(name, dim1, iv1, mynode1)
  use m_io, only : get_free_handle
  implicit none
  character(*), intent(in)      :: name
  logical, intent(in)        :: dim1
  integer, optional, intent(in) :: iv1
  integer(2), optional, intent(in) :: mynode1
  
  integer :: handle, mynode, iv
  character(200) :: filename_real;

  iv=0;     if(present(iv1)) iv=iv1;

  if(isizes>0) then
          write(isizes,   '(a50,2x,l2)') trim(name), dim1;
    if(iv>0)write(ilog, '(a,a50,2x,l2)')' size_note: ', trim(name), dim1;
  else
    mynode=0; if(present(mynode1)) mynode=mynode1

    write(filename_real, '(a,i0.4,a)') trim(name), mynode, '.txt';
    handle = get_free_handle();
    open(handle, file=trim(filename_real), action='write');
    write(handle, *) dim1
    close(handle);
    if(iv>0)write(ilog,*) 'size_note: ', trim(filename_real);
  endif

end subroutine ! size_note

!
! Finds the available RAM
!
subroutine log_available_vram(ram_gb)
  implicit none
  real(8), intent(out) :: ram_gb

  ! internal
  real(8), allocatable :: tmp(:)
  integer :: ngb, istat

  ngb = 1;
  istat = 0
  do while (.true.)
    allocate(tmp(1024**3/8 * ngb), stat=istat)
    if(istat/=0) exit
    ngb = ngb + 1;
    deallocate(tmp)
  end do

  ram_gb = real(ngb,8)

end subroutine 


end module !m_log
