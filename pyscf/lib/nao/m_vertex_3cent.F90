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

module m_vertex_3cent

! The purpose of the module is to store and deal with a real space information of a product basis
#include "m_define_macro.F90"
  use m_die, only : die
  
  implicit none
  private die

  !! Vertex for an atom pair (or a vertex part for a tri-center situation)
  type vertex_3cent_t
    real(8), allocatable :: vertex(:,:,:)
    integer :: centers(3) = -999 ! pointer to three centers for which we have vertex coefficients
    ! centers(1) is index of first center (atom) in the vertex
    ! centers(2) is index of second center (atom) in the vertex
    ! centers(3) is index of third center (a product center) in the vertex
  end type ! vertex_3cent_t
  !! END of Vertex for an atom pair (or a vertex part for a tri-center situation)

contains

!
!
!
function get_diff(p2f1, p2f2) result(d)
  implicit none
  !! external
  type(vertex_3cent_t), intent(in), allocatable :: p2f1(:), p2f2(:)
  real(8) :: d
  !! internal
  integer :: n1, n2, p

  d = 0  
  _size_alloc(p2f1,n1)
  _size_alloc(p2f2,n2)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!d>1d-14')
    return
  else if (n1>0) then
    d = 0
    do p=1,n1
      d = d + get_diff_sp(p2f1(p), p2f2(p))
    enddo 
  endif

end function !  get_diff 


!
!
!
function get_diff_sp(f1, f2) result(d)
  implicit none
  !! external
  type(vertex_3cent_t), intent(in) :: f1, f2
  real(8) :: d
  !! internal
  real(8) :: sa
  integer :: n1, n2
  
  d = 0
  _size_alloc(f1%vertex,n1)
  _size_alloc(f2%vertex,n2)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%vertex-f2%vertex))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif


  n1 = size(f1%centers)
  n2 = size(f2%centers)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%centers-f2%centers))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif

end function !  get_diff_sp


end module !m_vertex_3cent
