module m_alloc_identity

#include "m_define_macro.F90"

  contains

!
!
!
subroutine alloc_identity(n, mat)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout), allocatable :: mat(:,:)
  !
  integer :: i

  if(n<1) then
    write(6,*) n, __FILE__, __LINE__
    stop 'n<1'
  endif
  
  _dealloc(mat)
  allocate(mat(n,n))
  mat = 0
  
  do i=1,n; mat(i,i) = 1; enddo 
  
end subroutine ! alloc_identity


end module !m_alloc_identity
