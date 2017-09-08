module m_io

#include "m_define_macro.F90"  
  implicit none

  contains  

!
! Reads a full line from a formatted file. This is necessary if the line contains spaces
!
subroutine read_line(ifile, line)
  implicit none
  !! external
  integer, intent(in) :: ifile
  character(*), intent(inout) :: line
  !! internal
  integer :: ios, j
  integer, parameter :: M=1000
  character(1) :: line_char(M)
  
  line_char = ''; line='';
  read(ifile, '(1000A1)', iostat=ios) (line_char(j),j=1,M);
  do j=1,M;
    !if(int(line_char(j))<32) exit
    if( iachar(line_char(j))==13 ) exit ! minimal treatment against CRLF line endings
    line(j:j)=line_char(j);
  enddo;
  !! END of Read line

end subroutine ! read_line

!!
!! 
!!
integer function get_free_handle()
  implicit none
  ! internal
  integer :: ihandle
  logical :: logical_opened

  get_free_handle = -1;
  do ihandle=200,65000;
    inquire(ihandle, opened=logical_opened); 
    if(.not. logical_opened) then; get_free_handle=ihandle; return; endif;
  end do

end function !get_free_handle;


end module !m_io
