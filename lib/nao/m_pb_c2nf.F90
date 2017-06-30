module m_pb_c2nf

#include "m_define_macro.F90"
  implicit none

  contains

!
! Number of functions per center correspondence
!
subroutine pb_c2nf(pb, c2nf)
  use m_prod_basis_type, only : prod_basis_t
  use m_prod_basis_gen, only : get_nfunct_pbook, get_nbook
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(inout), allocatable :: c2nf(:)
  integer :: nc, ic
  
  nc = get_nbook(pb)
  _dealloc(c2nf)
  allocate(c2nf(nc))
  do ic=1,nc; c2nf(ic) = get_nfunct_pbook(pb, ic); enddo
  
end subroutine ! pb_c2nf 


end module ! m_pb_c2nf
