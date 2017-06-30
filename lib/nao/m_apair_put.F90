module m_apair_put

  use iso_c_binding, only: c_double, c_int64_t
#include "m_define_macro.F90"

  implicit none

  contains

!
!
!
subroutine apair_put(pb, pair, d,n)
  use m_system_vars, only : system_vars_t
  use m_prod_basis_type, only : prod_basis_t, get_vertex_dp_ptr, get_book_dp_ptr
  use m_book_pb, only : book_pb_t
  
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: d(n)
  !! internal
  type(book_pb_t), pointer :: bk
  integer(c_int64_t) :: b, p, s, f, nn(3)
  real(c_double), pointer :: vrtx(:,:,:)
  
  if( n<2 ) then; write(6,*)__FILE__, __LINE__, n; stop '!n<2'; endif
  bk => get_book_dp_ptr(pb, pair)
  if( bk%ic<1 ) then; write(6,*)__FILE__, __LINE__, bk%ic; stop '!%ic<1'; endif

  vrtx => get_vertex_dp_ptr(pb, pair)
  nn = ubound(vrtx)-lbound(vrtx)+1
  if(any(nn<1)) then; write(6,*)__FILE__, __LINE__, bk%ic; stop '!%nn<1'; endif
  s = 1
  d(s) = nn(3); s=s+1
  d(s) = nn(2); s=s+1
  d(s) = nn(1); s=s+1
  d(s) = bk%ic; s=s+1
  d(s) = bk%top; s=s+1
  d(s) = bk%spp; s=s+1

  do p=1,nn(3)
    do b=1,nn(2)
      f = s + nn(1) - 1; 
      !write(6,*)__FILE__, __LINE__, p,b,s,f, size(vrtx(:,b,p));
      if ( f>n ) then; write(6,*)__FILE__, __LINE__, f,n,nn; stop '!%f,n,nn'; endif
      d(s:f) = vrtx(1:nn(1),b,p); 
      s = f + 1;
    enddo
  enddo
  
  write(6,*) __FILE__, __LINE__
  write(6,*) pb%coeffs(pair)%ind2book_re
  write(6,*) pb%coeffs(pair)%ind2sfp_loc
  
  
end subroutine ! apair_put

end module ! m_apair_put
