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
  use m_prod_basis_type, only : prod_basis_t, get_vertex_dp_ptr, get_book_dp_ptr, get_coeffs_pp_ptr
  use m_book_pb, only : book_pb_t
  
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(in) :: pair
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(inout) :: d(n)
  !! internal
  type(book_pb_t), pointer :: bk
  integer(c_int64_t) :: i, b, p, s, f, nn(3),nac
  real(c_double), pointer :: vrtx(:,:,:), cc(:,:)
  
  if( n<2 ) then; write(6,*)__FILE__, __LINE__, n; stop '!n<2'; endif
  bk => get_book_dp_ptr(pb, pair)
  if( bk%ic<1 ) then; write(6,*)__FILE__, __LINE__, bk%ic; stop '!%ic<1'; endif

  vrtx => get_vertex_dp_ptr(pb, pair)
  if(size(vrtx)<1) then; write(6,*)__FILE__, __LINE__, size(vrtx); stop '!%vrtx<1'; endif
  i = 1
  nn = ubound(vrtx)
  d(i) = nn(3); i=i+1
  d(i) = nn(2); i=i+1
  d(i) = nn(1); i=i+1
  d(i) = bk%ic; i=i+1
  d(i) = bk%top; i=i+1
  d(i) = bk%spp; i=i+1

  if (allocated(pb%coeffs)) then
    cc   => get_coeffs_pp_ptr(pb, pair)
    d(i) = 1; i=i+1
    d(i) = ubound(cc,1); i=i+1
    d(i) = ubound(cc,2); i=i+1
    if(ubound(cc,2)/=nn(3)) then; write(6,*)__FILE__, __LINE__, ubound(cc,2), nn(3); stop '!ubound(cc,2)/=nn(3)'; endif
    d(i) = size(pb%coeffs(pair)%ind2book_re); i=i+1 ! number of participating centers
  else
    d(i) = 0; i=i+1
    d(i) = 0; i=i+1
    d(i) = 0; i=i+1
    d(i) = 0; i=i+1 ! number of participating centers
  endif

  s = i
  do p=1,nn(3)
    do b=1,nn(2)
      f = s + nn(1) - 1; 
      !write(6,*)__FILE__, __LINE__, p,b,s,f, size(vrtx(:,b,p));
      if ( f>n ) then; write(6,*)__FILE__, __LINE__, f,n,nn; stop '!f>n,nn'; endif
      d(s:f) = vrtx(1:nn(1),b,p); 
      s = f + 1;
    enddo ! b
  enddo ! p

  if (allocated(pb%coeffs)) then
    cc => get_coeffs_pp_ptr(pb, pair)
    nac = size(cc,1)
    do p=1,nn(3)
      f=s+nac-1;
      if ( f>n ) then; write(6,*)__FILE__, __LINE__, f,n; stop '!f>n'; endif
      d(s:f) = pb%coeffs(pair)%coeffs_ac_dp(:,p);
      s = f + 1;
    enddo
  endif

  !write(6,*) __FILE__, __LINE__
  !write(6,*) pb%coeffs(pair)%ind2book_re
  !write(6,*) pb%coeffs(pair)%ind2sfp_loc

end subroutine ! apair_put

end module ! m_apair_put
