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

module m_sp2norbs

#include "m_define_macro.F90"
    
  implicit none

contains

!
!
!
subroutine get_uc_orb2atom(atom2sp, mu_sp2j, sp2nmult, uc_orb2atom)
  implicit none
  !! external
  integer, intent(in) :: atom2sp(:), mu_sp2j(:,:), sp2nmult(:)
  integer, intent(inout) :: uc_orb2atom(:)
  !! internal
  integer :: nsp1,nsp2,nsp3,nmu_mx1,nmu_mx2,mu,nmu,step,m,j,orb,atom,sp
  integer :: natoms, norbs, nmu_mx, nsp
  !_dealloc(uc_orb2atom)
  nsp1 = maxval(atom2sp)
  nsp2 = size(sp2nmult)
  nsp3 = size(mu_sp2j,2)
  if(nsp1/=nsp2) stop 'get_uc_orb2atom: 1'
  if(nsp2/=nsp3) stop 'get_uc_orb2atom: 2 '
  nsp = nsp1
  if(nsp<1) stop 'get_uc_orb2atom: 3'
  nmu_mx1 = size(mu_sp2j,1)
  nmu_mx2 = maxval(sp2nmult)
  if(nmu_mx1/=nmu_mx2) stop 'get_uc_orb2atom: 4'
  nmu_mx = nmu_mx1
  if(nmu_mx<1) stop 'get_uc_orb2atom: 5'
  natoms = size(atom2sp)
  if(natoms<1) stop 'get_uc_orb2atom: 6'
  
  do step=1,2
    norbs = 0
    orb = 0 
    do atom=1,natoms
      sp = atom2sp(atom)
      nmu = sp2nmult(sp)
      do mu=1,nmu
        j = mu_sp2j(mu,sp)
        if(step==1) norbs = norbs + 2*j+1
        if(step==2) then
          do m=-j,j
            orb = orb + 1
            uc_orb2atom(orb) = atom
          enddo ! m
        endif ! step==2  
      enddo ! mu
    enddo ! atom
    if(step==1) then
      if(size(uc_orb2atom)<norbs) stop 'get_uc_orb2atom: 7 size(uc_orb2atom)<norbs'
    endif  
  enddo ! step
  
end subroutine ! get_uc_orb2atom  


!
!
!
function get_nspecies(atom2sp, mu_sp2j, sp2nmult) result(nsp)
  implicit none
  !! external
  integer, intent(in) :: atom2sp(:), sp2nmult(:), mu_sp2j(:,:)
  integer :: nsp
  !! internal
  integer :: n(3)
  
  n(1) = maxval(atom2sp)
  n(2) = size(sp2nmult)
  n(3) = size(mu_sp2j,2)
  nsp = n(1)
  if(any(n/=nsp)) stop 'm_sp2norbs: get_nspecies: 1'
  if(nsp<1) stop 'm_sp2norbs: get_nspecies: 2'

end function ! get_nspecies  

!
!
!
function get_norbs(atom2sp, mu_sp2j, sp2nmult) result(norbs)
  implicit none
  !! external
  integer, intent(in) :: atom2sp(:), sp2nmult(:), mu_sp2j(:,:)
  integer :: norbs
  !! internal
  integer :: nsp, natoms, atom
  integer, allocatable :: sp2norbs(:)
  nsp = get_nspecies(atom2sp, mu_sp2j, sp2nmult)
  natoms = size(atom2sp)
  if(natoms<1) stop 'm_sp2norbs: get_norbs: 1'
  allocate(sp2norbs(nsp))
  call get_sp2norbs_arg(nsp, mu_sp2j, sp2nmult, sp2norbs)
  if(any(sp2norbs<1)) stop 'm_sp2norbs: get_norbs: 2'
  norbs = 0
  do atom=1,natoms
    norbs = norbs + sp2norbs(atom2sp(atom))
  enddo !   
  _dealloc(sp2norbs)
  
end function ! get_nspecies  
  
!
!
!
subroutine get_sp2norbs_arg(nsp, mu_sp2j, sp2nmult, sp2norbs)
  implicit none
  !! external
  integer, intent(in) :: nsp, mu_sp2j(:,:), sp2nmult(:)
  integer, intent(inout) ::  sp2norbs(:)
  !! internal
  integer :: isp
    
  do isp=1,nsp;sp2norbs(isp)=sum(2*mu_sp2j(1:sp2nmult(isp),isp)+1);enddo
  
end subroutine ! get_sp2norbs_arg

!
!
!
subroutine get_orb2ao(atom2sp, mu_sp2j, sp2nmult, orb2ao)
  implicit none
  !! external
  integer, intent(in) :: atom2sp(:), mu_sp2j(:,:), sp2nmult(:)
  integer, intent(inout) :: orb2ao(:)
  !! internal
  integer :: sp, atom, nmu, ao, orb, natoms, mu, m, j

  natoms = size(atom2sp)
  if(natoms<1) stop 'm_sp2norbs: get_orb2ao: 1'
  
  orb = 0
  do atom=1,natoms
    sp = atom2sp(atom)
    nmu = sp2nmult(sp)
    ao = 0
    do mu=1,nmu
      j = mu_sp2j(mu, sp)
      do m=-j,j
        ao = ao + 1
        orb = orb + 1
        if(orb>size(orb2ao)) stop 'get_orb2ao: orb>size(orb2ao)'
        orb2ao(orb) = ao
      enddo ! m
    enddo  ! mu
  enddo ! atom
  
end subroutine ! get_orb2ao
 

end module !m_sp2norbs
