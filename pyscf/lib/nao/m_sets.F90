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

module m_sets

#define _dealloc_u(a,u) if(allocated(a))then;if(any(ubound(a)/=u))deallocate(a);endif;

contains

!
! Build a set out of a list
!
subroutine dlist2set(list, nset, set_o, l2s, s2n)
  implicit none
  !! external
  real(8), allocatable, intent(in) :: list(:,:)
  integer, intent(out) :: nset
  real(8), intent(inout), allocatable, optional :: set_o(:,:) ! set_o[utput]
  integer, intent(inout), allocatable, optional :: l2s(:) ! list to set correspondence
  integer, intent(inout), allocatable, optional :: s2n(:) ! number of occurencies in list
  !! internal
  integer :: step, ilist, i, ifound
  real(8), allocatable :: set_in(:,:)
  !! Dimensions
  integer :: nlist, n, m
  if(.not. allocated(list)) return
  nlist = size(list,2)
  n = size(list,1)
  !! END of Dimensions
  
  !! Build a SET
  allocate(set_in(n,nlist))
  do step=1,2
    nset = 0
    do ilist=1, nlist
      ifound = -1
      do i=1,nset
        if(all(set_in(:,i)==list(:,ilist))) then
          ifound = i
          exit
        endif
      enddo
      if(ifound>0) then; 
        if(step==2 .and. present(l2s)) l2s(ilist)=ifound;
        if(step==2 .and. present(s2n)) s2n(ifound)=s2n(ifound)+1;
        cycle;
      endif
      nset = nset + 1
      set_in(:,nset) = list(:,ilist)
      if(step==2 .and. present(l2s)) l2s(ilist)=nset;
      if(step==2 .and. present(s2n)) s2n(nset)=1;
    enddo ! atom_sc
    if(step==1) then
      if(nset/=nlist) then; deallocate(set_in); allocate(set_in(n,nset)); endif
      set_in=-999
      if(present(l2s))then;
_dealloc_u(l2s,nlist)
        if(.not. allocated(l2s)) allocate(l2s(nlist));
      endif
      if(present(s2n))then;
_dealloc_u(s2n,nset)      
        if(.not. allocated(s2n)) allocate(s2n(nset));
      endif
    endif
  enddo ! step
  
  if(present(set_o)) then
    m = nset
_dealloc_u(set_o,(/n,m/))
    if(.not. allocated(set_o)) allocate(set_o(n,nset))
    set_o = set_in
  endif  
  !! END of Build a SET
  if(allocated(set_in)) deallocate(set_in)
  
end subroutine ! dlist2set


!
! Build a set out of a list
!
subroutine ilist2set(list, nset, set_o, l2s, s2n)
  implicit none
  !! external
  integer, allocatable, intent(in) :: list(:,:)
  integer, intent(out) :: nset
  integer, intent(inout), allocatable, optional :: set_o(:,:) ! set_o[utput]
  integer, intent(inout), allocatable, optional :: l2s(:) ! list to set correspondence
  integer, intent(inout), allocatable, optional :: s2n(:) ! number of occurencies in list
  !! internal
  integer :: step, ilist, i, ifound
  integer, allocatable :: set_in(:,:)
  !! Dimensions
  integer :: nlist, n, m
  if(.not. allocated(list)) return
  nlist = size(list,2)
  n = size(list,1)
  !! END of Dimensions
  
  !! Build a SET
  allocate(set_in(n,nlist))
  do step=1,2
    nset = 0
    do ilist=1, nlist
      ifound = -1
      do i=1,nset
        if(all(set_in(:,i)==list(:,ilist))) then
          ifound = i
          exit
        endif
      enddo
      if(ifound>0) then; 
        if(step==2 .and. present(l2s)) l2s(ilist)=ifound;
        if(step==2 .and. present(s2n)) s2n(ifound)=s2n(ifound)+1;
        cycle;
      endif
      nset = nset + 1
      set_in(:,nset) = list(:,ilist)
      if(step==2 .and. present(l2s)) l2s(ilist)=nset;
      if(step==2 .and. present(s2n)) s2n(nset)=1;
    enddo ! atom_sc
    if(step==1) then
      if(nset/=nlist) then; deallocate(set_in); allocate(set_in(n,nset)); endif
      set_in=-999
      if(present(l2s))then;
_dealloc_u(l2s,nlist)
        if(.not. allocated(l2s)) allocate(l2s(nlist));
      endif
      if(present(s2n))then;
_dealloc_u(s2n,nset)      
        if(.not. allocated(s2n)) allocate(s2n(nset));
      endif
    endif
  enddo ! step
  
  if(present(set_o)) then
    m = nset
_dealloc_u(set_o,(/n,m/))
    if(.not. allocated(set_o)) allocate(set_o(n,nset))
    set_o = set_in
  endif  
  !! END of Build a SET
  if(allocated(set_in)) deallocate(set_in)
  
end subroutine ! ilist2set

!
! Build a set out of a list
!
subroutine ilist2set1(nlist, list, nset, set_o, l2s, s2n)
  implicit none
  !! external
  integer, intent(in) :: nlist
  integer, intent(in) :: list(:)
  integer, intent(out) :: nset
  integer, intent(inout), allocatable, optional :: set_o(:) ! set_o[utput]
  integer, intent(inout), allocatable, optional :: l2s(:) ! list to set correspondence
  integer, intent(inout), allocatable, optional :: s2n(:) ! number of occurencies in list
  !! internal
  integer :: step, ilist, i, ifound
  integer, allocatable :: set_in(:)
  !! Dimensions
  integer :: m
  if(nlist<1) then
    write(6,*) __FILE__, __LINE__
    write(6,*) nlist
    stop ' nlist<1'
  endif

  if(nlist>size(list)) then
    write(6,*) __FILE__, __LINE__
    write(6,*) nlist, size(list)
    stop ' nlist>size(list)'
  endif
  !! END of Dimensions
  
  !! Build a SET
  allocate(set_in(nlist))
  do step=1,2
    nset = 0
    do ilist=1, nlist
      ifound = -1
      do i=1,nset
        if(set_in(i)==list(ilist)) then
          ifound = i
          exit
        endif
      enddo
      if(ifound>0) then; 
        if(step==2 .and. present(l2s)) l2s(ilist)=ifound;
        if(step==2 .and. present(s2n)) s2n(ifound)=s2n(ifound)+1;
        cycle;
      endif
      nset = nset + 1
      set_in(nset) = list(ilist)
      if(step==2 .and. present(l2s)) l2s(ilist)=nset;
      if(step==2 .and. present(s2n)) s2n(nset)=1;
    enddo ! atom_sc
    if(step==1) then
      if(nset/=nlist) then; deallocate(set_in); allocate(set_in(nset)); endif
      set_in=-999
      if(present(l2s))then;
_dealloc_u(l2s,nlist)
        if(.not. allocated(l2s)) allocate(l2s(nlist));
      endif
      if(present(s2n))then;
_dealloc_u(s2n,nset)      
        if(.not. allocated(s2n)) allocate(s2n(nset));
      endif
    endif
  enddo ! step
  
  if(present(set_o)) then
    m = nset
_dealloc_u(set_o,(/m/))
    if(.not. allocated(set_o)) allocate(set_o(nset))
    set_o = set_in
  endif  
  !! END of Build a SET
  if(allocated(set_in)) deallocate(set_in)
  
end subroutine ! ilist2set1


end module !m_sets
