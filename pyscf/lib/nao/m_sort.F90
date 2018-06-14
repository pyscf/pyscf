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

module m_sort

  implicit none
  
  interface hpsort
    module procedure hpsort4
    module procedure hpsort8
    module procedure hpsort_dp4
    module procedure hpsort_dp8
  end interface ! hpsort 

  contains
  
!
!
!
subroutine simple_sort(dim, original_a, sorted_a,perm)

  implicit none
  ! extern
  integer::dim
  real(8), dimension(dim)::original_a, sorted_a
  integer, dimension(dim)::perm
  ! intern
  logical::swapped
  real(8)::dummy
  integer::i,aux

  sorted_a=original_a

  do i=1,dim; perm(i)=i; enddo ! i
  swapped=.true.
  do while(swapped)   
    swapped=.false.
    do i=1,dim-1
      if (  sorted_a(i) > sorted_a(i+1) ) then
        dummy    =   sorted_a(i+1);   
        sorted_a(i+1)=         sorted_a(i);      
        sorted_a(i) = dummy
        aux      =   perm(i+1);
        perm(i+1) = perm(i); 
        perm(i) = aux
        swapped=.true.
      endif 
    enddo ! i
  enddo ! while

end subroutine simple_sort

!
!
!
subroutine hpsort_dp4(n,ra,rp)
!! http://pagesperso-orange.fr/jean-pierre.moreau/Fortran/sort3_f90.txt
  implicit none

  integer(4), intent(in)    :: n
  real(8), intent(inout) :: ra(n)
  integer(4), intent(out)   :: rp(n)

  !! internal
  integer :: l, ir, i, j, rrap
  real(8) :: rra
  !! executable
  l=n/2+1
  ir=n
  !the index l will be decremented from its initial value during the
  !"hiring" (heap creation) phase. once it reaches 1, the index ir 
  !will be decremented from its initial value down to 1 during the
  !"retirement-and-promotion" (heap selection) phase.
  do i=1,n; rp(i)=i; enddo;
  if(n<2) return
  
10 continue
  if(l > 1)then
    l=l-1
    rra=ra(l); rrap = rp(l);
  else
    rra=ra(ir); rrap = rp(ir);
    ra(ir)=ra(1); rp(ir) = rp(1);
    ir=ir-1
    if(ir.eq.1)then
      ra(1)=rra; rp(1) = rrap;
      return
    end if
  end if
  i=l
  j=l+l
20 if(j.le.ir)then
  if(j < ir)then
    if(ra(j) < ra(j+1))  j=j+1
  end if
  if(rra < ra(j))then
    ra(i)=ra(j); rp(i) = rp(j);
    i=j; j=j+j
  else
    j=ir+1
  end if
  goto 20
  end if
  ra(i)=rra; rp(i) = rrap;
  goto 10

end subroutine ! hpsort_dp4

!
!
!
subroutine hpsort_dp8(n,ra,rp)
!! http://pagesperso-orange.fr/jean-pierre.moreau/Fortran/sort3_f90.txt
  implicit none

  integer(8), intent(in)    :: n
  real(8), intent(inout) :: ra(n)
  integer(8), intent(out)   :: rp(n)

  !! internal
  integer(8) :: l, ir, i, j, rrap
  real(8) :: rra
  !! executable
  l=n/2+1
  ir=n
  !the index l will be decremented from its initial value during the
  !"hiring" (heap creation) phase. once it reaches 1, the index ir 
  !will be decremented from its initial value down to 1 during the
  !"retirement-and-promotion" (heap selection) phase.
  do i=1,n; rp(i)=i; enddo;
  if(n<2) return
  
10 continue
  if(l > 1)then
    l=l-1
    rra=ra(l); rrap = rp(l);
  else
    rra=ra(ir); rrap = rp(ir);
    ra(ir)=ra(1); rp(ir) = rp(1);
    ir=ir-1
    if(ir.eq.1)then
      ra(1)=rra; rp(1) = rrap;
      return
    end if
  end if
  i=l
  j=l+l
20 if(j.le.ir)then
  if(j < ir)then
    if(ra(j) < ra(j+1))  j=j+1
  end if
  if(rra < ra(j))then
    ra(i)=ra(j); rp(i) = rp(j);
    i=j; j=j+j
  else
    j=ir+1
  end if
  goto 20
  end if
  ra(i)=rra; rp(i) = rrap;
  goto 10

end subroutine ! hpsort_dp8


!
!
!
subroutine hpsort4(n,ra,rp)
!! http://pagesperso-orange.fr/jean-pierre.moreau/Fortran/sort3_f90.txt
  implicit none

  integer(4), intent(in)    :: n
  integer(4), intent(inout) :: ra(n)
  integer(4), intent(out)   :: rp(n)

  !! internal
  integer(4) :: l, ir, i, j
  integer(4) :: rra, rrap
  !! executable
  l=n/2+1
  ir=n
  !the index l will be decremented from its initial value during the
  !"hiring" (heap creation) phase. once it reaches 1, the index ir 
  !will be decremented from its initial value down to 1 during the
  !"retirement-and-promotion" (heap selection) phase.
  do i=1,n; rp(i)=i; enddo;
  if(n<2) return

10 continue
  if(l > 1)then
    l=l-1
    rra=ra(l); rrap = rp(l);
  else
    rra=ra(ir); rrap = rp(ir);
    ra(ir)=ra(1); rp(ir) = rp(1);
    ir=ir-1
    if(ir.eq.1)then
      ra(1)=rra; rp(1) = rrap;
      return
    end if
  end if
  i=l
  j=l+l
20 if(j.le.ir)then
  if(j < ir)then
    if(ra(j) < ra(j+1))  j=j+1
  end if
  if(rra < ra(j))then
    ra(i)=ra(j); rp(i) = rp(j);
    i=j; j=j+j
  else
    j=ir+1
  end if
  goto 20
  end if
  ra(i)=rra; rp(i) = rrap;
  goto 10

end subroutine !hpsort4

!
!
!
subroutine hpsort8(n,ra,rp)
!! http://pagesperso-orange.fr/jean-pierre.moreau/Fortran/sort3_f90.txt
  implicit none

  integer(8), intent(in)    :: n
  integer(8), intent(inout) :: ra(n)
  integer(8), intent(out)   :: rp(n)

  !! internal
  integer(8) :: l, ir, i, j
  integer(8) :: rra, rrap
  !! executable
  l=n/2+1
  ir=n
  !the index l will be decremented from its initial value during the
  !"hiring" (heap creation) phase. once it reaches 1, the index ir 
  !will be decremented from its initial value down to 1 during the
  !"retirement-and-promotion" (heap selection) phase.
  do i=1,n; rp(i)=i; enddo;
  if(n<2) return

10 continue
  if(l > 1)then
    l=l-1
    rra=ra(l); rrap = rp(l);
  else
    rra=ra(ir); rrap = rp(ir);
    ra(ir)=ra(1); rp(ir) = rp(1);
    ir=ir-1
    if(ir.eq.1)then
      ra(1)=rra; rp(1) = rrap;
      return
    end if
  end if
  i=l
  j=l+l
20 if(j.le.ir)then
  if(j < ir)then
    if(ra(j) < ra(j+1))  j=j+1
  end if
  if(rra < ra(j))then
    ra(i)=ra(j); rp(i) = rp(j);
    i=j; j=j+j
  else
    j=ir+1
  end if
  goto 20
  end if
  ra(i)=rra; rp(i) = rrap;
  goto 10

end subroutine !hpsort8

!
!
!
subroutine qsort(a, n, t)

!     non-recursive stack version of quicksort from n.wirth's pascal
!     book, 'algorithms + data structures = programs'.

!     single precision, also changes the order of the associated array t.

implicit none

integer, intent(in)    :: n
double precision, intent(inout)    :: a(n)
integer, intent(inout) :: t(n)

!     local variables

integer                :: i, j, k, l, r, s, stackl(15), stackr(15), ww
double precision                   :: w, x

s = 1
stackl(1) = 1
stackr(1) = n

!     keep taking the top request from the stack until s = 0.

10 continue
l = stackl(s)
r = stackr(s)
s = s - 1

!     keep splitting a(l), ... , a(r) until l >= r.

20 continue
i = l
j = r
k = (l+r) / 2
x = a(k)

!     repeat until i > j.

do
  do
    if (a(i).lt.x) then                ! search from lower end
      i = i + 1
      cycle
    else
      exit
    end if
  end do

  do
    if (x.lt.a(j)) then                ! search from upper end
      j = j - 1
      cycle
    else
      exit
    end if
  end do

  if (i.le.j) then                     ! swap positions i & j
    w = a(i)
    ww = t(i)
    a(i) = a(j)
    t(i) = t(j)
    a(j) = w
    t(j) = ww
    i = i + 1
    j = j - 1
    if (i.gt.j) exit
  else
    exit
  end if
end do

if (j-l.ge.r-i) then
  if (l.lt.j) then
    s = s + 1
    stackl(s) = l
    stackr(s) = j
  end if
  l = i
else
  if (i.lt.r) then
    s = s + 1
    stackl(s) = i
    stackr(s) = r
  end if
  r = j
end if

if (l.lt.r) go to 20
if (s.ne.0) go to 10

return
end subroutine ! qsort


subroutine detailed_sort(a,sorted_a, n, direct_permutation, inverse_permutation)
implicit none

integer, intent(in)    :: n
double precision, intent(inout)    :: a(n)
double precision, intent(out)    :: sorted_a(n)

integer, intent(out) :: direct_permutation(n),inverse_permutation(n)

! internal variables
double precision::b(n),copy_of_a(n)
integer::i,t(n)

copy_of_a=a

do i=1,n; t(i)=i; enddo;  call qsort(copy_of_a, n, t)

sorted_a=copy_of_a

direct_permutation=t; b=dble(t)

do i=1,n; t(i)=i; enddo; call qsort(b,n,t)

inverse_permutation=t

end subroutine detailed_sort


end module !m_sort

