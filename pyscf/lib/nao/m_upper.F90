module m_upper

  implicit none

  contains

!
! http://www.star.le.ac.uk/~cgp/fortran.html
!
function upper(string) result(r)
  implicit none
  !! external
  character(len=*), intent(in) :: string
  character(len=len(string)) :: r 
  !! internal
  integer :: j

  do j = 1,len(string)
    if(string(j:j) >= "a" .and. string(j:j) <= "z") then
      r(j:j) = achar(iachar(string(j:j)) - 32)
    else
      r(j:j) = string(j:j)
    end if
  end do
  
end function !upper

end module !m_upper


