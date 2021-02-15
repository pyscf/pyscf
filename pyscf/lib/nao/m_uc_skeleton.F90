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

module m_uc_skeleton

#include "m_define_macro.F90"
  use m_die, only : die
  implicit none
  private die
  
  type uc_skeleton_t
    character(100) :: systemlabel = ""

    character(20), allocatable :: sp2label(:)
    integer, allocatable :: sp2nmult(:)
    integer, allocatable :: sp2norbs(:)
    integer, allocatable :: sp2element(:)

    integer, allocatable :: mu_sp2j(:,:)
    integer, allocatable :: mu_sp2n(:,:)
    real(8), allocatable :: mu_sp2rcut(:,:)
    integer, allocatable :: mu_sp2start_ao(:,:)
    
    integer, allocatable :: atom2sp(:)
    real(8), allocatable :: atom2coord(:,:)

    real(8) :: uc_vecs(3,3) = -999
    integer, allocatable :: atom2sfo(:,:) ! added 07.01.2015
    
  end type ! uc_skeleton_t

contains


subroutine dealloc(v)
  implicit none
  type(uc_skeleton_t), intent(inout) :: v
  _dealloc(v%sp2label)
  _dealloc(v%sp2nmult)
  _dealloc(v%sp2norbs)
  _dealloc(v%sp2element)
  _dealloc(v%mu_sp2j)
  _dealloc(v%mu_sp2n)
  _dealloc(v%mu_sp2rcut)
  _dealloc(v%mu_sp2start_ao)
  _dealloc(v%atom2sp)
  _dealloc(v%atom2coord)
  _dealloc(v%atom2sfo)

  v%uc_vecs = -999
  v%systemlabel = ""
  
end subroutine ! dealloc

!
! The initialization procedure is in the module m_wfsx__uc_skeleton
! and procedure init_basis_with_species(...) in the module m_scf_comm

subroutine print_uc_skeleton(ifile, uc)
  implicit none
  !! external
  integer, intent(in) :: ifile
  type(uc_skeleton_t), intent(in) :: uc
  !! internal
  integer :: nsp, sp, mu, nmu
  call check_uc_skeleton(uc)
  
  nsp = get_nspecies(uc)

  write(ifile,'(a20,2x,a)') '%systemlabel', trim(uc%systemlabel)
  write(ifile,'(a20,2x,i4)')  'get_nspecies(uc)', nsp
  write(ifile,'(a4,2x,a4,2x,a4,2x,a)') 'sp', 'nmu', 'elem', 'sp2label(sp)'
  write(ifile,'(i4,2x,i4,2x,i4,2x,a)') (sp, uc%sp2nmult(sp), uc%sp2element(sp), &
    trim(uc%sp2label(sp)), sp=1,nsp)

  write(ifile,'(a4,2x,a4,4x,a4,2x,a4)') 'mu', 'sp', 'j', 'rcut'
  do sp=1,nsp
    nmu = uc%sp2nmult(sp)
    do mu=1,nmu
      write(ifile,'(i4,2x,i4,4x,i4,2x,g20.9)') mu, sp, uc%mu_sp2j(mu,sp), uc%mu_sp2rcut(mu,sp)
    enddo  ! mu
  enddo ! sp  
  
end subroutine !  print_uc_skeleton 


!
! This is a procedure to get just one specie out of 
! a global pool into the type uc_skeleton_t
!
subroutine init_uc_skeleton_specie(a, sp, b)
  use m_init_mu_sp2start_ao, only : init_mu_sp2start_ao
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: a
  integer, intent(in) :: sp
  type(uc_skeleton_t), intent(inout) :: b
  !! internal
  integer :: nsp, nn(2), no, mu, j, nmu
  character(200) :: clabel
  integer, pointer :: mu_sp2j(:,:)
  real(8), pointer :: mu_sp2rcut(:,:)
  
  nsp = get_nspecies(a)
  if(sp<1 .or. sp>nsp) _die('wrong sp ?')
  
  write(clabel, '(a,a,i0.2)') trim(a%systemlabel),'-', sp
  b%systemlabel(1:100) = clabel(1:100)

_dealloc_u(b%atom2sp,1)
  if(.not. allocated(b%atom2sp)) allocate(b%atom2sp(1))
  b%atom2sp(1) = 1

  nn = [3,1]
_dealloc_u(b%atom2coord,nn)
  if(.not. allocated(b%atom2coord)) allocate(b%atom2coord(3,1))
  b%atom2coord = 0

_dealloc_u(b%sp2label,1)
  if(.not. allocated(b%sp2label)) allocate(b%sp2label(1))
  b%sp2label = a%sp2label(sp)

_dealloc_u(b%sp2nmult,1)
  if(.not. allocated(b%sp2nmult)) allocate(b%sp2nmult(1))
  b%sp2nmult = get_nmult(a, sp)

_dealloc_u(b%sp2element,1)
  if(.not. allocated(b%sp2element)) allocate(b%sp2element(1))
  b%sp2element = a%sp2element(sp)

  nn = [get_nmult(a, sp), 1]
_dealloc_u(b%mu_sp2j,nn)
  if(.not. allocated(b%mu_sp2j)) allocate(b%mu_sp2j(nn(1),1))
  mu_sp2j => get_mu_sp2j_ptr(a)
  b%mu_sp2j(1:nn(1),1) = mu_sp2j(1:nn(1), sp)

  nn = [get_nmult(a, sp), 1]
_dealloc_u(b%mu_sp2rcut,nn)
  if(.not. allocated(b%mu_sp2rcut)) allocate(b%mu_sp2rcut(nn(1),1))
  mu_sp2rcut => get_mu_sp2rcut_ptr(a)
  b%mu_sp2rcut(1:nn(1),1) = mu_sp2rcut(1:nn(1), sp)

  b%uc_vecs = a%uc_vecs
  
_dealloc_u(b%atom2sfo, (/2,1/))
  if(.not. allocated(b%atom2sfo)) allocate(b%atom2sfo(2,1))
  b%atom2sfo(1,1) = 1
  no = 0
  nmu = get_nmult(a, sp)
  do mu=1, nmu
    j = get_j(a, mu, sp)
    no = no + 2*j+1
  enddo ! 
  b%atom2sfo(2,1) = b%atom2sfo(1,1) + no - 1

  call get_sp2norbs(b, b%sp2norbs)
  call init_mu_sp2start_ao(b%sp2nmult, b%mu_sp2j, b%mu_sp2start_ao)

end subroutine ! init_uc_skeleton_specie  

!
!
!
function get_atom2sfo_ptr(uc) result(ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: uc
  integer, pointer :: ptr(:,:)
  if(.not. allocated(uc%atom2sfo)) _die('! %atom2sfo')
  ptr => uc%atom2sfo
end function ! get_atom2sfo_ptr

!
!
!
function get_sp2element_ptr(uc) result(ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: uc
  integer, pointer :: ptr(:)

  if(.not. allocated(uc%sp2element)) _die('.not. allocated(uc%sp2element)')

  ptr => uc%sp2element

end function !get_sp2element


!
!
!
function get_sp2norbs_ptr(uc) result(ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: uc
  integer, pointer :: ptr(:)
  !internal
  integer :: nsp
  nsp = get_nspecies(uc)
  if(.not. allocated(uc%sp2norbs)) _die('! %sp2norbs')
  if(nsp /= size(uc%sp2norbs)) then
    write(6,'(2i8)') nsp, size(uc%sp2norbs)
    write(6,'(999i6)') uc%sp2norbs
    _die('hm != ?...')
  endif  
  if(any(uc%sp2norbs<1)) _die('any(uc%sp2norbs<1)')

  ptr => uc%sp2norbs
end function ! get_atom2sfo_ptr

!
!
!
function get_mu_sp2start_ao_ptr(uc) result(ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: uc
  integer, pointer :: ptr(:,:)
  !internal
  integer :: nsp, nmu_mx
  
  if(.not. allocated(uc%mu_sp2start_ao)) _die('! %mu_sp2start_ao')
  nsp = get_nspecies(uc)
  if(nsp /= size(uc%mu_sp2start_ao,2)) _die('hm != ?...')
  nmu_mx = get_nmult_max(uc)
  if(nmu_mx /= size(uc%mu_sp2start_ao,1)) _die('hm != ?...')
  ptr => uc%mu_sp2start_ao
  
end function ! get_atom2sfo_ptr

!
!
!
function get_mu_sp2j_ptr(uc) result(ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: uc
  integer, pointer :: ptr(:,:)
  if(.not. allocated(uc%mu_sp2j)) _die('.not. allocated(mu_sp2j)')
  ptr => uc%mu_sp2j
end function ! get_mu_sp2j_ptr

!
!
!
function get_mu_sp2rcut_ptr(uc) result(ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: uc
  real(8), pointer :: ptr(:,:)
  if(.not. allocated(uc%mu_sp2rcut)) _die('.not. allocated(mu_sp2rcut)')
  ptr => uc%mu_sp2rcut
end function ! get_mu_sp2rcut_ptr
  
!
!
!
function get_atom2sp_ptr(uc) result(atom2specie_ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: uc
  integer, pointer :: atom2specie_ptr(:)
  !! 
  if(.not. allocated(uc%atom2sp)) _die('.not. allocated(atom2sp)')
  atom2specie_ptr => uc%atom2sp
end function ! get_atom2sp_ptr 

!
!
!
function get_sp2nmult_ptr(ucs) result(ptr)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in), target :: ucs
  integer, pointer :: ptr(:)
  !! internal
  if(.not. allocated(ucs%sp2nmult)) _die('.not. allocated(sp2nmult)')
  if(sum(abs(ucs%sp2nmult))<1) _die('sum(abs(sp2nmult))<1')
  ptr => ucs%sp2nmult
end function ! get_sp2nmult

!
!
!
function get_coord(ucs, atom) result(coord)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: atom
  real(8) :: coord(3)
  !! internal
  integer :: natoms
  natoms = -1
  if(.not. allocated(ucs%atom2coord)) _die('.not. allocated(atom2coord)')
  natoms = size(ucs%atom2coord,2)
  if(natoms<1) _die('natoms<1')
  
  if(atom<1 .or. atom>natoms) _die('atom<1 .or. atom>natoms')
  coord(1:3) = ucs%atom2coord(:,atom)
  
end function ! get_coords

!
!
!
function get_atom2coord_ptr(ucs) result(atom2coord_ptr)
  implicit none
  type(uc_skeleton_t), intent(in), target :: ucs
  real(8), pointer :: atom2coord_ptr(:,:)
  !! internal
  integer :: natoms
  natoms = get_natoms(ucs)
  if(natoms<1) _die('natoms<1')
  atom2coord_ptr=>ucs%atom2coord
  
end function ! get_atom2coord_ptr

!
!
!
function get_rcuts(ucs, atoms) result(rcuts)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: atoms(:)
  real(8) :: rcuts(size(atoms))
  !! internal
  integer :: natoms, i, n
  n = size(atoms)
  natoms = get_natoms(ucs)
  if(n<1) _die('n<1')
  if(any(atoms<1) .or. any(atoms>natoms)) &
    _die('any(atoms<1) .or. any(atoms>natoms)')
  
  do i=1,n
    rcuts(i) = maxval(ucs%mu_sp2rcut(:,ucs%atom2sp(atoms(i))))
  enddo 
  
end function ! get_rcuts   

!
!
!
function get_coords(ucs, atoms, cells, n) result(coords)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: n
  integer, intent(in) :: atoms(n)
  real(8), intent(in) :: cells(3,n)
  real(8) :: coords(3,n)
  !! internal
  integer :: natoms
  real(8) :: uc_vecs(3,3)
  natoms = get_natoms(ucs)
  uc_vecs = get_uc_vecs(ucs)
  
  if(n<1) _die('n<1')
  if(any(atoms<1) .or. any(atoms>natoms)) &
    _die('any(atoms<1) .or. any(atoms>natoms)')
  
  coords(1:3,1:n) = ucs%atom2coord(:,atoms(1:n)) + &
    matmul(uc_vecs(1:3,1:3), cells(1:3,1:n))
  
end function ! get_coords


!
!
!
function get_jmx_sp(ucs, sp) result(rc)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: sp
  integer :: rc
  !! internal
  rc = maxval(ucs%mu_sp2j(:,sp)) 
end function ! get_rcut_max


!
!
!
function get_rcut_max(ucs) result(rc)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  real(8) :: rc
  !! internal
  call check_uc_skeleton(ucs)
  rc = maxval(ucs%mu_sp2rcut)
  
end function ! get_rcut_max

!
!
!
subroutine get_sp2jmx(uc, sp2jmx)
  implicit none
  type(uc_skeleton_t), intent(in) :: uc
  integer, intent(inout), allocatable :: sp2jmx(:)
  !! internal
  integer :: nsp, sp
  integer, pointer :: mu_sp2j(:,:)
  nsp = get_nspecies(uc)
  mu_sp2j => get_mu_sp2j_ptr(uc)
  
_dealloc_u(sp2jmx, nsp)
  if(.not.allocated(sp2jmx)) allocate(sp2jmx(nsp))

  do sp=1,nsp; 
    sp2jmx(sp) = maxval(mu_sp2j(:,sp));
  enddo ! sp

end subroutine ! get_sp2jmx  

!
! function returns number of orbitals in the calculation
!
function get_norbs(ucs) result(n)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer :: n
  !! internal
  integer, allocatable :: sp2norbs(:)
  integer :: natoms, atom
  call get_sp2norbs(ucs, sp2norbs)
  natoms = get_natoms(ucs)
  n = 0
  do atom=1,natoms
    n = n + sp2norbs(get_sp(ucs, atom))
  enddo ! atom
    
end function !get_natoms


!
! function returns number of atoms in the calculation
!
function get_natoms(ucs) result(n)
  implicit none
  type(uc_skeleton_t), intent(in) :: ucs
  integer :: n
  call check_uc_skeleton(ucs)
  n = size(ucs%atom2sp)
end function !get_natoms


!
! function returns number of species (of atoms) in the calculation
!
function get_nspecies(ucs) result(n)
  implicit none
  type(uc_skeleton_t), intent(in) :: ucs
  integer :: n
  call check_uc_skeleton(ucs)
  n = size(ucs%sp2nmult)
end function !get_nspecies


!
! function returns maximal number of angular momenta in the calculation
!
function get_jmx(ucs) result(n)
  implicit none
  type(uc_skeleton_t), intent(in) :: ucs
  integer :: n
  call check_uc_skeleton(ucs)
  !do i = lbound(ucs%mu_sp2j, 2), ubound(ucs%mu_sp2j, 2)
  !  print*, 'ucs%mu_sp2j(:, ', i, ') = ', ucs%mu_sp2j(:, 1)
  !enddo
  !_die('look')
  n = maxval(ucs%mu_sp2j)
end function !get_jmx


!
!
!
function get_zelem(ucs, specie) result(zelem)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: specie
  integer :: zelem
  call check_uc_skeleton(ucs)
  
  if(specie<1 .or. specie>size(ucs%atom2sp)) _die('!specie')
  zelem = ucs%sp2element(specie)
  
end function ! get_zelem  


!
!
!
function get_nmult_max(ucs) result(n)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer :: n
  
  call check_uc_skeleton(ucs)
  n = size(ucs%mu_sp2j,1)
end function ! get_nmult_max  

!!
!!
!!
function get_uc_vecs(ucs) result(uc_vecs)
  implicit none 
  ! external
  type(uc_skeleton_t), intent(in) :: ucs
  real(8) :: uc_vecs(3,3) ! xyz, 123
  ! internal
  integer :: i
  
  do i=1,3; 
    if(all(ucs%uc_vecs(1:3,i)==0)) _die('all(ucs%uc_vecs(1:3,i)==0)')
    uc_vecs(1:3,i) = ucs%uc_vecs(1:3,i)
  enddo

end function !get_uc_vecs

!
!
!
subroutine get_atom2start_orb(ucs, atom2start_orb)
  implicit none
  ! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(inout), allocatable :: atom2start_orb(:) 
  ! internal
  integer :: atom, natoms
  integer, allocatable :: sp2norbs(:)

  call get_sp2norbs(ucs, sp2norbs)
  natoms = get_natoms(ucs)
  _dealloc(atom2start_orb)
  allocate(atom2start_orb(natoms))
  atom2start_orb(1) = 1
  do atom=2,natoms;
    atom2start_orb(atom) = atom2start_orb(atom-1)+sp2norbs(ucs%atom2sp(atom-1));
  enddo

end subroutine ! init_atom2start_orb

!
!
!
function get_nmult(ucs, sp) result(n)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: sp
  integer :: n
  !! internal
  n = ucs%sp2nmult(sp)
end function ! get_nmu

!
!
!
function get_j(ucs, mu, sp) result(j)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: mu, sp
  integer :: j
  !! internal
  j = ucs%mu_sp2j(mu,sp)
end function ! get_nmu

!
!
!
function get_sp(ucs, atom) result(sp)
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(in) :: atom
  integer :: sp
  !! internal
  if(.not. allocated(ucs%atom2sp)) _die('!ucs%atom2sp')
  if(atom<1 .or. atom>size(ucs%atom2sp)) _die('!atom')
  sp = ucs%atom2sp(atom)
end function ! get_nmu
 
!
!
!
subroutine check_uc_skeleton(ucs)!, fname, line)
  use m_log, only : die
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
!  character(*), intent(in), optional :: fname
!  integer, intent(in), optional :: line
  !! internal
  integer :: n, a(6), i, ub2(2), lb2(2), na
  
  !! Check natoms
  a(1)=0; if(allocated(ucs%atom2sp)) a(1) = size(ucs%atom2sp,1)
  a(2)=0; if(allocated(ucs%atom2coord)) a(2) = size(ucs%atom2coord,2)
  n = a(1)
  if(any(a(1:2)/=n) .or. n<1) then
    write(6,'(a,6i6)') 'a(1:2) ', a(1:2)
    write(6,'(a,i6)') 'n ', n     
    write(6,'(a)') 'check_uc_skeleton: did you initialize uc or sv ?'
    _die('uc skeleton is not initialized')
  endif
  
  !! Check nspecies
  i=1; 
  a(i)=0; if(allocated(ucs%atom2sp))    a(i) = maxval(ucs%atom2sp);    i=i+1
  a(i)=0; if(allocated(ucs%mu_sp2j))    a(i) = size(ucs%mu_sp2j,2);    i=i+1
  a(i)=0; if(allocated(ucs%sp2nmult))   a(i) = size(ucs%sp2nmult,1);   i=i+1
  a(i)=0; if(allocated(ucs%mu_sp2rcut)) a(i) = size(ucs%mu_sp2rcut,2); i=i+1
  a(i)=0; if(allocated(ucs%sp2label))   a(i) = size(ucs%sp2label,1);   i=i+1
  a(i)=0; if(allocated(ucs%mu_sp2start_ao)) a(i) = size(ucs%mu_sp2start_ao,2); i=i+1
  n = a(1)
  if(any(a(1:6)/=n) .or. n<1) then
    write(6,'(a,60i6)') 'a ', a(1:6)
    write(6,'(a,i6)') 'n ', n 
    _die('any(a/=n) .or. n<1')
  endif  

  !! Check nmult
  i=1; 
  a(i)=0; if(allocated(ucs%mu_sp2j))    a(i) = size(ucs%mu_sp2j,1);    i=i+1
  a(i)=0; if(allocated(ucs%sp2nmult))   a(i) = maxval(ucs%sp2nmult,1); i=i+1
  a(i)=0; if(allocated(ucs%mu_sp2rcut)) a(i) = size(ucs%mu_sp2rcut,1); i=i+1
  a(i)=0; if(allocated(ucs%mu_sp2start_ao)) a(i) = size(ucs%mu_sp2start_ao,1); i=i+1
  n = a(1)
  if(any(a(1:4)/=n) .or. n<1) then
    write(6,'(a,2x,i6,4i6,2x,i6)') __FILE__, __LINE__, a(1:4), n
    _die('any(a/=n) .or. n<1')
  endif
  
  !! Check atom2sfo
  if(.not. allocated(ucs%atom2sfo)) _die('!atom2sfo')
  lb2 = lbound(ucs%atom2sfo)
  ub2 = ubound(ucs%atom2sfo)
  na = size(ucs%atom2sp)
  if(any(lb2/=1) .and. any(ub2/=[2,na])) _die('!atom2sfo')

end subroutine ! check_uc_skeleton  

!
!
!
subroutine get_ao_sp2m(ucs, ao_sp2m)
  use m_log, only : die
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, allocatable, intent(inout) :: ao_sp2m(:,:)
  !! internal
  integer, allocatable :: sp2norbs(:)
  integer :: sp, nsp, nomx, nmu, ao, j, m, mu
  call get_sp2norbs(ucs, sp2norbs)
  nomx = maxval(sp2norbs)
  if(nomx<1) _die('nomx<1')
  nsp  = size(sp2norbs)
  if(nsp<1) _die('nsp<1')
    
  _dealloc(ao_sp2m)
  allocate(ao_sp2m(nomx,nsp))
  
  do sp=1,nsp
    nmu = get_nmult(ucs, sp)
    ao = 0
    do mu=1,nmu
      j = get_j(ucs, mu, sp)
      do m=-j,j
        ao = ao + 1
        ao_sp2m(ao,sp) = m
      enddo ! m
    enddo ! mu
  enddo ! sp
  
end subroutine ! get_ao_sp2m
  
!
!
!
subroutine get_atom2norb(uc, atom2norb)
  type(uc_skeleton_t), intent(in) :: uc
  integer, allocatable, intent(inout) :: atom2norb(:)
  !! internal
  integer:: natoms, atm
  integer, allocatable :: sp2norb(:)
  
  call get_sp2norbs(uc, sp2norb)

  natoms = get_natoms(uc)  
  allocate(atom2norb(natoms));
  do atm=1,natoms; 
    atom2norb(atm)= sp2norb(get_sp(uc, atm));
  enddo
  
  _dealloc(sp2norb)
end subroutine !get_atom2norb


!
!
!
subroutine get_sp2norbs(ucs, sp2norbs)
  
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  integer, intent(inout), allocatable ::  sp2norbs(:)
  !! internal
  integer :: isp, nsp

  call check_uc_skeleton(ucs)
  nsp = size(ucs%sp2nmult)
  
  _dealloc(sp2norbs)
  allocate(sp2norbs(nsp))
  do isp=1,nsp;
    sp2norbs(isp) = sum(2 * ucs%mu_sp2j(1:ucs%sp2nmult(isp),isp)+1);
  enddo
  
end subroutine ! get_sp2norbs

!
!
!
subroutine get_orb2(ucs, cdata, orb2data)
  use m_upper, only : upper
  implicit none
  !! external
  type(uc_skeleton_t), intent(in) :: ucs
  character(*), intent(in) :: cdata
  integer, intent(inout), allocatable :: orb2data(:)
  !! internal
  integer :: natoms, norbs,mu,nmu,step,m,j,orb,atom,sp, ao
  character(10) :: cdataupper
  
  _dealloc(orb2data)
  call check_uc_skeleton(ucs)
  natoms = get_natoms(ucs)
  
  cdataupper = upper(cdata)
  
  do step=1,2
    norbs = 0
    orb = 0 
    do atom=1,natoms
      sp = get_sp(ucs,atom)
      nmu = get_nmult(ucs,sp)
      ao = 0
      do mu=1,nmu
        j = get_j(ucs,mu,sp)
        if(step==1) norbs = norbs + 2*j+1
        if(step==2) then
          do m=-j,j
            ao = ao + 1
            orb = orb + 1
          
            select case(cdataupper)
            case("M")
              orb2data(orb) = m
            case("AO") 
              orb2data(orb) = ao
            case("ATOM") 
              orb2data(orb) = atom
            case("J","L") 
              orb2data(orb) = j
            case default
              _die('unknown idat')  
            end select
            
          enddo ! m
        endif ! step==2  
      enddo ! mu
    enddo ! atom
    if(step==1) then
      allocate(orb2data(norbs))
      orb2data = -999
    endif  
  enddo ! step
  
end subroutine ! get_orb2m


end module !m_uc_skeleton
