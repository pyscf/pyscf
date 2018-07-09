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

module m_dft_hsx
  
#include "m_define_macro.F90"

  use m_die, only : die
  
  implicit none 
  private die

  type dft_hsx_t
    logical              :: is_gamma = .false.  ! whether it is a gamma-only calculation, i.e. finite system
    real(8), allocatable :: hamiltonian(:,:,:)  ! (orbital, supercell_orbital, spin)
    real(8), allocatable :: overlap(:,:)        ! (orbital, supercell_orbital)
    integer, allocatable :: sc_orb2uc_orb(:)    ! (supercell_orbital)
    real(8), allocatable :: aB_to_RB_m_Ra(:,:,:)! (xyz, orbital, supercell_orbital) to difference vector between centers
    real(8) :: sc_vectors(3,3) = -999           ! super cell vectors (xyz,i)
    integer :: nunit_cells(3) = -999            ! number of unit cells in super cell along each direction
    real(8) :: tot_electr_chrg = -999           ! total electronic charge   
  end type ! dft_hsx_t

  contains

!
!
!
subroutine dealloc(hsx)
  implicit none
  type(dft_hsx_t), intent(inout) :: hsx

  _dealloc(hsx%hamiltonian)
  _dealloc(hsx%overlap)
  _dealloc(hsx%sc_orb2uc_orb)
  _dealloc(hsx%aB_to_RB_m_Ra)

!hsx%sc_vectors(3,3) = -999 
!hsx% nunit_cells(3) = -999
!hsx% tot_electr_chrg = -999
!hsx% ienergy_units   = -999

end subroutine ! dealloc  

!
!
!
subroutine init_dft_hsx_finite_size(uc_orb2atom, atom_sc2coord, dft_hsx)
  implicit none 

  real(8), intent(in):: atom_sc2coord(:,:)
  integer, intent(in):: uc_orb2atom(:)
  type(dft_hsx_t), intent(inout):: dft_hsx
  !! in
  integer :: orb1, orb2, atom1, atom2, norbs
  norbs = size(uc_orb2atom)
  if(norbs<1) _die('norbs<1')
  
  dft_hsx%is_gamma = .true.
  dft_hsx%nunit_cells = 1
  dft_hsx%sc_vectors = 0
  dft_hsx%sc_vectors(1,1)  =  -999! this should not be used normally...
  dft_hsx%sc_vectors(2,2)  =  -999! this should not be used normally...
  dft_hsx%sc_vectors(3,3)  =  -999! this should not be used normally...
  
  allocate(dft_hsx%overlap(norbs,norbs))
  allocate(dft_hsx%hamiltonian(norbs,norbs,1))
  allocate(dft_hsx%sc_orb2uc_orb(norbs))
  allocate(dft_hsx%aB_to_RB_m_Ra(3,norbs, norbs))
  dft_hsx%overlap = -999
  dft_hsx%hamiltonian = -999
  
  do orb1=1,norbs; dft_hsx%sc_orb2uc_orb(orb1) = orb1; enddo  
  
  do orb2=1,norbs;
    atom2 = uc_orb2atom(orb2)
    do orb1=1,norbs;
      atom1 = uc_orb2atom(orb1)
      dft_hsx%aB_to_RB_m_Ra(:,orb1,orb2) = atom_sc2coord(:,atom2)-atom_sc2coord(:,atom1)
    enddo ! 
  enddo !     
  
    
  ! should also be initialized preferably

end subroutine init_dft_hsx_finite_size

!
!
!
subroutine modify_rhs_mx_mat(orb2atom_ao, ao_sp2m, atom2sp, sco2uco, x_data)
  implicit none
  !! external
  integer, intent(in) :: orb2atom_ao(:,:), ao_sp2m(:,:), atom2sp(:)
  real(8), intent(inout) :: x_data(:,:)
  integer, intent(in) :: sco2uco(:)
  !! internal
  integer :: norbs_sc, norbs, o2_sc, o2, m2, m1, a(3), o1
  !! Dimensions
  a(1) = size(x_data,2)
  a(2) = size(sco2uco,1)
  norbs_sc = a(1)
  if(any(a(1:2)/=norbs_sc)) _die('any(a/=norbs_sc)')
  
  a(1) = size(orb2atom_ao,2)
  a(2) = maxval(sco2uco)
  a(3) = size(x_data,1)
  norbs = a(1)
  if(any(a(1:3)/=norbs)) _die('any(a/=norbs)')
  !! END of Dimensions
  
  do o2_sc=1,norbs_sc
    o2 = sco2uco(o2_sc);
    m2  = ao_sp2m(orb2atom_ao(2,o2), atom2sp(orb2atom_ao(1,o2)))
    do o1=1,norbs
      m1  = ao_sp2m(orb2atom_ao(2,o1), atom2sp(orb2atom_ao(1,o1)))
      x_data(o1,o2_sc) = x_data(o1,o2_sc) * (-1D0)**(m1) * (-1D0)**(m2)
    enddo !
  enddo !
  
end subroutine ! modify_rhs_mx_mat

!
!
!
function get_natoms_sc_hsx(dft_hsx, natoms)  result(natoms_sc)
  implicit none
  !! external
  integer, intent(in) :: natoms
  type(dft_hsx_t), intent(in) :: dft_hsx
  integer :: natoms_sc
  !! internal
  integer :: norbs, norbs_sc

  norbs_sc = get_norbs_sc_hsx(dft_hsx)
  norbs = get_norbs_hsx(dft_hsx)
  
  natoms_sc = (norbs_sc / norbs) * natoms
  if(natoms_sc /= ((1.0D0*norbs_sc) / norbs) * natoms) then
    write(0,*)'error: something is changed '
    write(0,*)'(natoms_sc /= ((norbs_sc*1.0D0) / norbs) * natoms)'
    write(0,*) 'natoms_sc, norbs_sc, natoms, norbs'
    write(0,*) natoms_sc, norbs_sc, natoms, norbs
    _die('assumptions are not valid anymore ?')
  end if
  
end function ! get_natoms_sc_hsx


!
!
!
function get_nspin(dft_hsx) result(n)
  implicit none
  type(dft_hsx_t), intent(in) :: dft_hsx
  integer :: n
  n = size(dft_hsx%hamiltonian,3)
  if(n<1 .or. n>2) _die('n<1 .or. n>2')
end function ! get_nspin

!
! get_uc_vecs
!
function get_uc_vecs(dft_hsx) result(uc_vecs)
  implicit none
  !! external
  type(dft_hsx_t), intent(in) :: dft_hsx
  real(8) :: uc_vecs(3,3)
  !! internal
  call hsx_2_uc_vectors(dft_hsx, uc_vecs, 0, 6)
end function !  get_uc_vecs

!
!
!
subroutine check_dft_hsx(natoms, dft_hsx)  
  implicit none
  !! external
  integer, intent(in) :: natoms
  type(dft_hsx_t), intent(in) :: dft_hsx
  !! internal
  integer :: norbs_sc_hsx, norbs, natoms_sc_hsx

  norbs_sc_hsx = get_norbs_sc_hsx(dft_hsx)
  norbs = get_norbs_hsx(dft_hsx)
  
  natoms_sc_hsx = (norbs_sc_hsx / norbs) * natoms
  if(natoms_sc_hsx /= ((1.0D0*norbs_sc_hsx) / norbs) * natoms) then
    write(0,*)'error: something is changed '
    write(0,*)'(natoms_sc_hsx /= ((norbs_sc*1.0D0) / norbs) * natoms)'
    write(0,*) 'natoms_sc_hsx, norbs_sc, natoms, norbs'
    write(0,*) natoms_sc_hsx, norbs_sc_hsx, natoms, norbs
    _die('assumptions are not valid anymore ?')
  end if
  
end subroutine ! check_dft_hsx

!
!
!
function get_norbs_sc_hsx(dft_hsx) result(no) 
  implicit none
  !! external
  type(dft_hsx_t), intent(in) :: dft_hsx
  integer :: no
  !! internal
  integer :: n(4)
  n(1) = -1; if(allocated(dft_hsx%hamiltonian)) n(1) = size(dft_hsx%hamiltonian,2)
  n(2) = -1; if(allocated(dft_hsx%overlap)) n(2) = size(dft_hsx%overlap,2)
  n(3) = -1; if(allocated(dft_hsx%sc_orb2uc_orb)) n(3) = size(dft_hsx%sc_orb2uc_orb,1)
  n(4) = -1; if(allocated(dft_hsx%aB_to_RB_m_Ra)) n(4) = size(dft_hsx%aB_to_RB_m_Ra,3)
  no = n(1)
  if(any(n/=no)) _die('any(n/=no)')
  if(no<1) _die('no<1')
  
end function ! get_norbs_sc_hsx 

!
!
!
function get_norbs_hsx(dft_hsx) result(no) 
  implicit none
  !! external
  type(dft_hsx_t), intent(in) :: dft_hsx
  integer :: no
  !! internal
  integer :: n(4)
  n(1) = -1; if(allocated(dft_hsx%hamiltonian)) n(1) = size(dft_hsx%hamiltonian,1)
  n(2) = -1; if(allocated(dft_hsx%overlap)) n(2) = size(dft_hsx%overlap,1)
  n(3) = -1; if(allocated(dft_hsx%sc_orb2uc_orb)) n(3) = maxval(dft_hsx%sc_orb2uc_orb)
  n(4) = -1; if(allocated(dft_hsx%aB_to_RB_m_Ra)) n(4) = size(dft_hsx%aB_to_RB_m_Ra,2)
  no = n(1)
  if(any(n/=no)) _die('any(n/=no)')
  if(no<1) _die('no<1')
  
end function ! get_norbs_hsx 
  

!
!
!
subroutine print_message_folding()
  implicit none
  
  write(6,*)"natoms==natoms_sc_dim .and. natoms_sc>natoms"
  write(6,*)"The systems seems to be a 'molecule' according to SIESTA's conventions."
  write(6,*)"However, there are some atoms from super cell, which contribute to Hamiltonian."
  write(6,*)"Normally, this would not happen, but it could happen in principle."
  write(6,*)"For instance, the initial geometry was choosen too tight and automatic"
  write(6,*)"lattice vectors allowed for atoms from neighboring unit cells to interact."
  write(6,*)"I will stop now. In order to continue there are two possible solutions:"
  write(6,*)"1) Choose larger LatticeConstant in .fdf file;"
  write(6,*)"2) Modify your start geometry in such a way that no messages about "
  write(6,*)"possible folding appears in SIESTA's output."

end subroutine ! print_message_folding

!
!
!
subroutine get_atom_sc2(dft_hsx, uc_orb2atom, atom2coord, &
  atom_sc2coord, atom_sc2atom_uc, atom_sc2atom_sc_hsx, atom_sc2atoms_uc)
  
  implicit none
  !! external
  type(dft_hsx_t), intent(in) :: dft_hsx
  integer, intent(in) :: uc_orb2atom(:)
  real(8), intent(in) :: atom2coord(:,:)
  real(8), intent(inout), allocatable :: atom_sc2coord(:,:)
  integer, intent(inout), allocatable :: atom_sc2atom_uc(:)
  integer, intent(inout), allocatable :: atom_sc2atom_sc_hsx(:)
  integer, intent(inout), allocatable :: atom_sc2atoms_uc(:,:)
  !! internal
  integer :: norbs_sc_hsx, norbs, natoms, o1, o2, o2_sc, natoms_sc
  integer :: natoms_sc_mx, atom_sc_found, atom_sc, natoms_sc_hsx, atom2_sc, atom1
  real(8), allocatable :: atom_sc_mx2coord(:,:)
  integer, allocatable :: atom_sc_mx2atom_uc(:), atom_sc_mx2atom_sc_hsx(:), atom_sc2nconn(:)
  logical, allocatable :: atom_sc_mx2atoms_uc(:,:)
  real(8) :: RBa(3), Ra(3), RB(3)

  _dealloc(atom_sc2coord)
  _dealloc(atom_sc2atom_uc)
  _dealloc(atom_sc2atom_sc_hsx)
 
  natoms = maxval(uc_orb2atom)
  norbs  = size(uc_orb2atom)
  norbs_sc_hsx = size(dft_hsx%aB_to_RB_m_Ra,3)
  natoms_sc_hsx = (norbs_sc_hsx / norbs) * natoms
!  write(6,*)'norbs_sc_hsx', norbs_sc_hsx, __FILE__, __LINE__
  if(natoms_sc_hsx /= ((norbs_sc_hsx*1.0D0) / norbs) * natoms) then
    write(0,*)'error: something is changed '
    write(0,*)'(natoms_sc_hsx /= ((norbs_sc*1.0D0) / norbs) * natoms)'
    write(0,*) 'natoms_sc_hsx, norbs_sc, natoms, norbs'
    write(0,*) natoms_sc_hsx, norbs_sc_hsx, natoms, norbs
    _die('assumptions are not valid anymore ?')
  end if
  natoms_sc_mx = natoms_sc_hsx*30

!  write(6,*) 'norbs_sc_hsx, norbs  ', norbs, norbs_sc_hsx
!  write(6,*) 'natoms_sc_mx, natoms ', natoms_sc_mx, natoms
!  write(6,*) 'uc_orb2atom'
!  write(6,'(10i4)') uc_orb2atom

!  atom = 0
!  do atom_sc=1,natoms_sc_hsx
!    atom = atom + 1
!    if(atom>natoms) atom=1;
!    atom_sc_hsx2atom_uc(atom_sc) = atom;
!  end do
  
  allocate(atom_sc_mx2coord(3,natoms_sc_mx))
  allocate(atom_sc_mx2atom_uc(natoms_sc_mx))
  allocate(atom_sc_mx2atom_sc_hsx(natoms_sc_mx))
  allocate(atom_sc_mx2atoms_uc(natoms,natoms_sc_mx))
  !write(6,*) 'uc_orb2atom'
  !write(6,'(1000i5)') uc_orb2atom
  atom_sc_mx2atoms_uc = .false.
  natoms_sc = 0
  do o2_sc=1,norbs_sc_hsx
    o2 = dft_hsx%sc_orb2uc_orb(o2_sc)
    atom2_sc = ( (o2_sc-1) / norbs ) * natoms + uc_orb2atom(o2)
!    write(6,*) o2, atom2_sc
    do o1=1,norbs
      atom1 = uc_orb2atom(o1)
      Ra = atom2coord(:,atom1)
      RBa = dft_hsx%aB_to_RB_m_Ra(:,o1,o2_sc)
      if(all(RBa==-999)) cycle
      RB = RBa + Ra
      !! Look whether we have already such RB in the list atom_sc_mx2coord
      atom_sc_found = -1
      do atom_sc=1,natoms_sc
        if(sum(abs(atom_sc_mx2coord(:,atom_sc) - RB ))>1D-4) cycle !! we have this atom already
        atom_sc_found = atom_sc
        exit
      enddo ! atom_sc
      !! END of Look whether we have already such RB in the list atom_sc_mx2coord
      if(atom_sc_found<1) then
        natoms_sc = natoms_sc + 1
        if(natoms_sc>natoms_sc_mx) then
          write(6,'(a35,2i6)') 'natoms_sc, natoms_sc_mx', natoms_sc, natoms_sc_mx
          _die('natoms_sc>natoms_sc_mx')
        endif  ! natoms_sc>natoms_sc_mx
        atom_sc_mx2coord(:,natoms_sc) = RB
        atom_sc_mx2atom_uc(natoms_sc) = uc_orb2atom(o2)
        atom_sc_mx2atom_sc_hsx(natoms_sc) = atom2_sc
        atom_sc_mx2atoms_uc(atom1, natoms_sc) = .true.
      else if (atom_sc_found>0) then
        atom_sc_mx2atoms_uc(atom1, atom_sc_found) = .true.
      endif ! atom_sc_found<1 
    enddo ! o1
  enddo ! o2_sc
!  write(6,*) 'natoms_sc, natoms_sc_mx', natoms_sc, natoms_sc_mx
!  write(6,*) 'atom_sc_mx2coord(:,1:natoms_sc)'
!  write(6,*) atom_sc_mx2coord(:,1:natoms_sc)
!  write(6,'(10i5)') atom_sc_mx2atom_sc_hsx(1:natoms_sc)
!  write(6,*) '(norbs_sc/norbs)*natoms', (norbs_sc_hsx/norbs)*natoms
!  write(6,*) 'natoms_sc, natoms_sc_mx', natoms_sc, natoms_sc_mx
  
  if(natoms_sc<1) _die('natoms_sc<1')
  allocate(atom_sc2coord(3,natoms_sc))
  allocate(atom_sc2atom_uc(natoms_sc))
  allocate(atom_sc2atom_sc_hsx(natoms_sc))
  atom_sc2coord(1:3,1:natoms_sc) = atom_sc_mx2coord(1:3,1:natoms_sc)
  atom_sc2atom_uc(1:natoms_sc) = atom_sc_mx2atom_uc(1:natoms_sc)
  atom_sc2atom_sc_hsx(1:natoms_sc) = atom_sc_mx2atom_sc_hsx(1:natoms_sc) 

  !! Compress logical connectivity map
  allocate(atom_sc2atoms_uc(natoms, natoms_sc))
  allocate(atom_sc2nconn(natoms_sc))
  atom_sc2nconn = 0
  atom_sc2atoms_uc = -999
  do atom_sc=1,natoms_sc
    do atom1=1,natoms
      if(atom_sc_mx2atoms_uc(atom1,atom_sc)) then
        atom_sc2nconn(atom_sc) = atom_sc2nconn(atom_sc) + 1
        atom_sc2atoms_uc(atom_sc2nconn(atom_sc), atom_sc) = atom1
      endif
    enddo ! atom
  enddo ! atom_sc
  !! END of Compress logical connectivity map
  
  write(6,*)'atom_sc2coord'
  write(6,'(3f20.10)')atom_sc2coord
  write(6,*)'atom_sc2atom_uc'
  write(6,'(10i5)')atom_sc2atom_uc
  write(6,*)'atom_sc2atom_sc_hsx'
  write(6,'(10i5)')atom_sc2atom_sc_hsx
  write(6,*)'atom_sc2atom_uc'
  write(6,'(10i5)')atom_sc2atom_uc
 
!  atom_sc2atoms_uc(1:natoms, 1:natoms_sc) = atom_sc_mx2atoms_uc(1:natoms,1:natoms_sc)

!  do atom2_sc=1,natoms_sc
!    write(6,'(300000l2)')atom_sc_mx2atoms_uc(:,atom2_sc)
!  enddo! atom2_sc
  
!  write(6,*) '37, 38'
!  write(6,'(300000l2)')atom_sc_mx2atoms_uc(:,37)
!  write(6,'(300000l2)')atom_sc_mx2atoms_uc(:,38)

!  do atom2_sc=1,natoms_sc
!    write(6,'(300000i6)')atom_sc2atoms_uc(1:count(atom_sc2atoms_uc(:,atom2_sc)>0),atom2_sc)
!  enddo! atom2_sc

!  write(6,*) '37, 38'
!  atom2_sc = 37
!  write(6,'(300000i6)')atom_sc2atoms_uc(1:count(atom_sc2atoms_uc(:,atom2_sc)>0),atom2_sc)

!  atom2_sc = 38
!  write(6,'(300000i6)')atom_sc2atoms_uc(1:count(atom_sc2atoms_uc(:,atom2_sc)>0),atom2_sc)
  
!  _die('connectivity ?')
end subroutine ! get_atom_sc2

!
!
!
subroutine conv_dft_hsx(atom2sp, mu_sp2j, sp2nmult, &
  atom_sc2coord, atom_sc2atom_uc, atom_sc2atom_sc_hsx1, atom_sc2atoms_uc, hsx1, hsx2)
  use m_sp2norbs, only : get_sp2norbs_arg
  implicit none
  !! external
  integer, intent(in) :: atom2sp(:), mu_sp2j(:,:), sp2nmult(:)
  real(8), intent(in) :: atom_sc2coord(:,:)
  integer, intent(in) :: atom_sc2atom_uc(:), atom_sc2atom_sc_hsx1(:), atom_sc2atoms_uc(:,:)
  type(dft_hsx_t), intent(in) :: hsx1
  type(dft_hsx_t), intent(inout) :: hsx2
  !! internal  
  integer :: natoms_sc1, natoms_sc2, natoms_sc3, natoms_sc, norbs_sc
  integer :: nsp1, nsp2, nsp3, nsp, atom_sc, norbs, atom, nspin, atom2_sc, atom1, atom2
  integer :: natoms1, natoms2, natoms, s1,f1,s2,f2,s21,f21,n1,n2,i, orb2, orb2_uc
  integer :: natoms_sc_hsx1, natoms_sc_hsx2, natoms_sc_hsx,o1,o2,natoms_conn,ind
  integer :: norbs_sc_hsx1, norbs_sc_hsx2, norbs_sc_hsx3, norbs_sc_hsx
  integer, allocatable :: sp2norbs(:), atom_sc_hsx2start_orb(:)
  real(8) :: RBa(3)
  !! Dimensions
  natoms_sc1 = size(atom_sc2coord,2)
  natoms_sc2 = size(atom_sc2atom_uc)
  natoms_sc3 = size(atom_sc2atom_sc_hsx1)
  if(natoms_sc1/=natoms_sc2) _die('natoms_sc1/=natoms_sc2')
  if(natoms_sc2/=natoms_sc3) _die('natoms_sc2/=natoms_sc3')
  natoms_sc = natoms_sc1
  if(natoms_sc<1) _die('natoms_sc<1')

  nsp1 = maxval(atom2sp)
  nsp2 = size(sp2nmult)
  nsp3 = size(mu_sp2j,2)
  if(nsp1/=nsp2) _die('nsp1/=nsp2')
  if(nsp2/=nsp3) _die('nsp2/=nsp3')
  nsp = nsp1
  if(nsp<1) _die('nsp<1')
  
  natoms1 = size(atom2sp)
  natoms2 = maxval(atom_sc2atom_uc)
  if(natoms1/=natoms2) _die('natoms1/=natoms2')
  natoms = natoms1
  if(natoms<1) _die('natoms<1')

  allocate(sp2norbs(nsp))
  call get_sp2norbs_arg(nsp, mu_sp2j, sp2nmult, sp2norbs)
  norbs_sc = 0
  do atom_sc=1,natoms_sc
    norbs_sc = norbs_sc + sp2norbs(atom2sp(atom_sc2atom_uc(atom_sc)))
  enddo

  norbs = 0
  do atom1=1,natoms; norbs = norbs + sp2norbs(atom2sp(atom1)); enddo
  if(norbs>norbs_sc) _die('norbs>norbs_sc')

  norbs_sc_hsx1 = size(hsx1%aB_to_RB_m_Ra,3)
  norbs_sc_hsx2 = size(hsx1%hamiltonian,2)
  norbs_sc_hsx3 = size(hsx1%overlap,2)
  if(norbs_sc_hsx1/=norbs_sc_hsx2)_die('norbs_sc_hsx1/=norbs_sc_hsx2')
  if(norbs_sc_hsx2/=norbs_sc_hsx3)_die('norbs_sc_hsx2/=norbs_sc_hsx3')
  norbs_sc_hsx = norbs_sc_hsx1
  if(norbs_sc_hsx<1) _die('norbs_sc_hsx<1')
  
  natoms_sc_hsx1 = norbs_sc_hsx / norbs * natoms
  natoms_sc_hsx2 = maxval(atom_sc2atom_sc_hsx1)
  if(natoms_sc_hsx1<natoms_sc_hsx2) then
    write(6,'(a35,3i7)') 'norbs_sc_hsx, norbs, natoms', norbs_sc_hsx, norbs, natoms
    write(6,'(a35,2i7)') 'natoms_sc_hsx1, natoms_sc_hsx2', natoms_sc_hsx1, natoms_sc_hsx2
    write(6,'(a35)') 'atom_sc2atom_sc_hsx1'
    write(6,'(200000000i7)') atom_sc2atom_sc_hsx1
    _die('natoms_sc_hsx1<natoms_sc_hsx2')
  endif  
  natoms_sc_hsx = natoms_sc_hsx1
  if(natoms_sc_hsx<1) _die('natoms_sc_hsx<1')
  allocate(atom_sc_hsx2start_orb(natoms_sc_hsx))
  atom = 0
  f2 = 0
  do atom2_sc=1,natoms_sc_hsx
    atom=atom+1; if(atom>natoms) atom=1
    s2 = f2 + 1; n2 = sp2norbs(atom2sp(atom)); f2 = s2 + n2 - 1
    atom_sc_hsx2start_orb(atom2_sc) = s2
  enddo

  nspin = size(hsx1%hamiltonian,3)
  if(nspin<1 .or. nspin>2) _die('nspin<1 .or. nspin>2')
  !! END of Dimensions
  
  hsx2 = hsx1
  
  _dealloc(hsx2%aB_to_RB_m_Ra)
  _dealloc(hsx2%hamiltonian)
  _dealloc(hsx2%overlap)
  _dealloc(hsx2%sc_orb2uc_orb)
  allocate(hsx2%aB_to_RB_m_Ra(3,norbs,norbs_sc))
  allocate(hsx2%hamiltonian(norbs,norbs_sc,nspin))
  allocate(hsx2%overlap(norbs,norbs_sc))
  allocate(hsx2%sc_orb2uc_orb(norbs_sc))
  hsx2%aB_to_RB_m_Ra = -999D0
  hsx2%hamiltonian = 0
  hsx2%overlap = 0
  hsx2%sc_orb2uc_orb = -999
  f2 = 0
  write(6,*) 'natoms_sc', natoms_sc
  do atom2_sc=1,natoms_sc
    atom2 = atom_sc2atom_uc(atom2_sc)
    n2 = sp2norbs(atom2sp(atom2));
    s2 = f2 + 1; f2 = s2 + n2 - 1
    s21 = atom_sc_hsx2start_orb(atom_sc2atom_sc_hsx1(atom2_sc)); f21 = s21 + n2 - 1;

    orb2_uc = atom_sc_hsx2start_orb(atom2)
    do orb2=s2,f2 
      hsx2%sc_orb2uc_orb(orb2) = orb2_uc
      orb2_uc = orb2_uc + 1
    enddo ! orb2
          
    natoms_conn = count(atom_sc2atoms_uc(:,atom2_sc)>0)
!    write(6,*) 'atom2_sc, natoms_conn', atom2_sc, natoms_conn
    do ind=1,natoms_conn
      atom1 = atom_sc2atoms_uc(ind,atom2_sc)
!      write(6,*) 'ind, atom1', ind, atom1
      s1 = atom_sc_hsx2start_orb(atom1); n1 = sp2norbs(atom2sp(atom1)); f1 = s1 + n1 - 1

      hsx2%overlap(s1:f1,s2:f2) = hsx1%overlap(s1:f1,s21:f21)
      
      do i=1,nspin
        hsx2%hamiltonian(s1:f1,s2:f2,i) = hsx1%hamiltonian(s1:f1,s21:f21,i)
      enddo

      do i=1,3
        hsx2%aB_to_RB_m_Ra(i,s1:f1,s2:f2) = hsx1%aB_to_RB_m_Ra(i,s1:f1,s21:f21)
      enddo
      
      RBa = atom_sc2coord(:,atom2_sc) - atom_sc2coord(:,atom1) 
      do o2=s2,f2
        do o1=s1,f1
          if(all(hsx2%aB_to_RB_m_Ra(:,o1,o2)==-999)) cycle
          if(sum(abs(hsx2%aB_to_RB_m_Ra(:,o1,o2)-RBa))>1d-5) then
            write(6,*) atom1, atom2_sc
            write(6,*) o1, o2
            write(6,*) hsx2%aB_to_RB_m_Ra(:,o1,o2)
            write(6,*) s1,s21
            write(6,*) hsx1%aB_to_RB_m_Ra(:,s1,s21)
            write(6,*) RBa
            write(6,*) sum(hsx1%hamiltonian(s1:f1,s21:f21,:))
            _die('>1d-5')
          endif  
        enddo ! o1
      enddo ! o2    

    enddo ! atom1
  enddo ! atom2_sc  
  !_die('conv_dft_hsx')
  
end subroutine ! conv_dft_hsx

!!
!! Construct list of atom pairs for which Hamiltonian is non zero
!! It is different to constr_spair_list because Hamiltonian is sligthly 
!! more non local than overlap, due to Kleinman-Bylander projectors
!! from pseudo-potentials
!!
subroutine constr_hpair_list(atom_sc2coord, sp2norbs, atom2specie, &
  atom_sc2atom_uc, atom_sc2start_orb, dft_hsx, hpair2atoms)
  implicit none
  !! external
  real(8), intent(in) :: atom_sc2coord(:,:)
  integer, intent(in) :: sp2norbs(:), atom2specie(:), atom_sc2atom_uc(:)
  integer, intent(in) :: atom_sc2start_orb(:)
  type(dft_hsx_t), intent(in) :: dft_hsx
  integer, allocatable, intent(inout) :: hpair2atoms(:,:)
  !! internal
  integer :: atom1,atom2,step,npairs_hnzero,pair,n1,n2,o1,o2,orb1,orb2
  real(8) :: DeltaR(3)
  logical :: is_full
  !! Dimensions
  integer :: natoms, natoms_sc
  natoms = size(atom2specie)
  natoms_sc = size(atom_sc2coord,2)
  !! END of Dimensions

  !! List of overlapping atom pairs is created
  npairs_hnzero = 0
  do step=1,2
    pair = 0
    do atom2=1,natoms_sc
      n2 = sp2norbs(atom2specie(atom_sc2atom_uc(atom2)))
      do atom1=1,natoms
        n1 = sp2norbs(atom2specie(atom_sc2atom_uc(atom1)))

        is_full = .false.
        do o2=1,n2; orb2 = atom_sc2start_orb(atom2)+o2-1
        do o1=1,n1; orb1 = atom_sc2start_orb(atom1)+o1-1
          DeltaR = dft_hsx%aB_to_RB_m_Ra(:,orb1,orb2)
          if(.not. count(DeltaR==-999.0D0)==3) then
            is_full = .true.
            exit
          endif  
        enddo
        enddo
        if(.not. is_full) write(0,*) 'constr_hpair_list: wau! hamiltonian is sparse!'
        
        if(step==1 .and. is_full) npairs_hnzero=npairs_hnzero+1
        if(step==2 .and. is_full) then
          pair=pair+1
          hpair2atoms(:,pair) = (/atom1, atom2/)
        endif    
      enddo ! atom_a
    enddo ! atom_B
    if(step==1) allocate(hpair2atoms(2,npairs_hnzero))
  enddo ! step
  !! END of List of overlapping atom pairs is created

end subroutine ! constr_hpair_list

!
!
!
subroutine hsx_2_uc_vectors(dft_hsx, uc_vectors, iv, ilog)
#define _SNAME 'hsx_2_uc_vectors'
  implicit none 
  ! external
  type(dft_hsx_t), intent(in) :: dft_hsx
  real(8), intent(out)        :: uc_vectors(:,:) ! xyz, 123
  integer, intent(in)         :: iv, ilog
  ! internal
  integer :: i
  
  if(iv>0)write(ilog,'(a)') _SNAME//': dft_hsx%sc_vectors';
  if(iv>0)write(ilog,'(3g25.15)') dft_hsx%sc_vectors
  if(iv>0)write(ilog,'(a)') _SNAME//': dft_hsx%nunit_cells';
  if(iv>0)write(ilog,'(3i6)') dft_hsx%nunit_cells
  if(sum(dft_hsx%nunit_cells)==0) then
    if(iv>0)write(ilog,'(a)')'sum(dft_hsx%nunit_cells)==0. Is it a finite system ? '
    if(iv>0)write(ilog,'(a)')_SNAME//': ==> will return zeros uc_vectors';
    uc_vectors = 0
    return;
  endif
  do i=1,3; uc_vectors(:,i) = dft_hsx%sc_vectors(:,i)/dft_hsx%nunit_cells(i); enddo;
  if(iv>0)write(ilog,'(a)') _SNAME//': uc_vectors';
  if(iv>0)write(ilog,'(3g20.10)') uc_vectors
  
!  _die('what about the vectors?')
#undef _SNAME  
end subroutine ! hsx_2_uc_vectors

!
!
!
subroutine build_atom_sc2atom_uc_v2(hsx, atom2sp, sp2norbs, sp2Z, coord, &
  atom_sc2atom_uc, atom_sc2coord, atom_sc2start_orb, iv, ilog)

  use m_log, only : log_size_note, die
  use m_z2sym, only : z2sym
  use m_xyz, only : write_xyz_file
  
  implicit none 
  !! External
  type(dft_hsx_t), intent(inout) :: hsx
  integer, intent(in) :: atom2sp(:), sp2norbs(:), sp2Z(:)
  real(8), intent(in) :: coord(:,:)
  integer, intent(inout), allocatable :: atom_sc2atom_uc(:)
  real(8), intent(inout), allocatable :: atom_sc2coord(:,:)
  integer, intent(inout), allocatable :: atom_sc2start_orb(:)
  integer, intent(in) :: iv, ilog

  !! internal
  integer :: atom, orb_sc, atom_sc, natoms_sc, orbital_sc, natoms_sc999, no
  integer :: orbital_uc, atom_sc_uc, norb_atom_sc, orb, step, norbs_sc
  integer :: atom_sc999, n2, n1, s2, s1, f2, f1, s2999, f2999, i, n2999, o1, o2
  real(8) :: DeltaR(3), Ra(3), Rb(3)

  integer, allocatable :: atom_sc999_2atom_uc(:)
  real(8), allocatable :: atom_sc999_2coord(:,:)
  integer, allocatable :: atom_sc999_2start_orb(:)
  integer, allocatable :: atom_sc2atom_sc999(:)
  real(8), allocatable :: hamilt(:,:,:), overlap(:,:)
  integer, allocatable :: sc_orb2uc_orb(:), atom_sc999_2atom(:)
  real(8), allocatable :: aB2RB_m_Ra(:,:,:)
  logical, allocatable :: atom_sc999_2remains(:)
  
  character(2), allocatable :: atm2sym(:)
  complex(8), allocatable :: ztensor(:,:)
  real(8) :: kvector(3), R_Ba(3), error
  integer :: orb2, orb1
  complex(8) :: phase
  
  !! Dimensions
  integer :: norbs_sc999, natoms, norbs, nspin, na
  norbs_sc999 = size(hsx%sc_orb2uc_orb)
  norbs = maxval(hsx%sc_orb2uc_orb)
  natoms = size(atom2sp)
  nspin = size(hsx%hamiltonian,3)
  !! END of Dimensions
  
  if(iv>0) write(ilog,*) 'build_atom_sc2atom_uc_v2: enter'
  natoms_sc999 = (norbs_sc999 / norbs) * natoms
  if(natoms_sc999 /= ((norbs_sc999*1.0D0) / norbs) * natoms) then
    write(0,*)'error: build_atom_sc2atom_uc_v2: something is changed '
    write(0,*)'(natoms_sc999 /= ((norbs999_sc*1.0D0) / norbs) * natoms)'
    write(0,*) 'natoms_sc999, norbs_sc999, natoms, norbs'
    write(0,*) natoms_sc999, norbs_sc999, natoms, norbs
    _die('error: build_atom_sc2atom_uc_v2: something is changed')
  end if

  call log_size_note('build_atom_sc2atom_uc_v2: natoms_sc999', natoms_sc999, iv);
  allocate(atom_sc999_2atom_uc(natoms_sc999))
  atom = 0
  do atom_sc999=1, natoms_sc999
    atom = atom + 1
    if(atom>natoms) atom=1;
    atom_sc999_2atom_uc(atom_sc999) = atom;
  end do

  allocate(atom_sc999_2coord(3,natoms_sc999))
  if( norbs_sc999 == norbs ) then

    atom_sc999_2coord = coord
    
  else
    
    allocate(atom_sc999_2remains(natoms_sc999))
    atom_sc999_2remains = .false.
    orbital_sc = 0
    do atom_sc999=1, natoms_sc999
      atom_sc_uc = atom_sc999_2atom_uc(atom_sc999)
      norb_atom_sc = sp2norbs(atom2sp(atom_sc_uc))
      do orb_sc=1,norb_atom_sc
        orbital_sc = orbital_sc + 1;
        orbital_uc = 0
        do atom=1,natoms
          do orb=1,sp2norbs(atom2sp(atom))
            orbital_uc=orbital_uc + 1
            DeltaR = hsx%aB_to_RB_m_Ra(:,orbital_uc, orbital_sc)
            !write(6,'(3i6)') atom, atom_sc999, count(DeltaR==-999.0D0)
            atom_sc999_2remains(atom_sc999) = (atom_sc999_2remains(atom_sc999) .or. &
              .not. all(DeltaR==-999D0))
          end do
        end do
      end do
    end do
    
    write(6,*) 'atom_sc999_2remains', atom_sc999_2remains(:)
    write(6,*) 'count(atom_sc999_2remains)', count(atom_sc999_2remains)
    write(6,*) 'size(atom_sc999_2remains)', size(atom_sc999_2remains)
    
    allocate(atom_sc999_2atom(natoms_sc999))
    atom_sc999_2atom = -999
    atom_sc999_2coord = -999.0D0
    orbital_sc = 0
    do atom_sc999=1, natoms_sc999
      atom_sc_uc = atom_sc999_2atom_uc(atom_sc999)
      norb_atom_sc = sp2norbs(atom2sp(atom_sc_uc))
      Rb = coord(:,atom_sc_uc)
      do orb_sc=1,norb_atom_sc
        orbital_sc = orbital_sc + 1;
        orbital_uc = 0
        do atom=1,natoms
          do orb=1,sp2norbs(atom2sp(atom))
            orbital_uc=orbital_uc + 1
            DeltaR = hsx%aB_to_RB_m_Ra(:,orbital_uc, orbital_sc)
            !write(6,'(3i6)') atom, atom_sc999, count(DeltaR==-999.0D0)
            if(any(DeltaR==-999.0D0)) cycle ! zero orbitals with Hamiltonian non zero
            Ra = coord(:,atom)
!            if( .not. all(atom_sc999_2coord(:,atom_sc999)==-999)) then
!              if(sum(abs(atom_sc999_2coord(:,atom_sc999)-(Ra + DeltaR)))>1D-5) then
!                write(6,*) atom_sc999_2coord(:,atom_sc999)+Ra-Rb
!                write(6,*) DeltaR-Rb
!                write(6,*) atom_sc999, atom_sc999_2atom(atom_sc999), atom
!                _die('atom_sc will be rewritten')
!              endif  
!            endif 
            atom_sc999_2coord(:,atom_sc999) = Ra + DeltaR;
            atom_sc999_2atom(atom_sc999) = atom
          end do
        end do
      end do
    end do

    na = 0
    do atom_sc999=1, natoms_sc999
      if(all(atom_sc999_2coord(:,atom_sc999)==-999D0)) na = na+1
    enddo
    write(6,*) 'na', na
    if(size(atom_sc999_2remains)-count(atom_sc999_2remains)/=na) &
      _die('* /=na')
  endif

  !! Export .xyz files before removing atoms
  _dealloc(atm2sym)
  allocate(atm2sym(natoms_sc999))
  do atom=1,natoms_sc999
    atm2sym(atom) = z2sym(sp2Z(atom2sp(atom_sc999_2atom_uc(atom))))
  enddo 

  call write_xyz_file('domiprod-coord_999.xyz', natoms_sc999, atm2sym, &
    atom_sc999_2coord, 'before removing atoms from sc', iv, ilog);
  !! END of Export .xyz files before removing atoms from unit cell

  allocate(atom_sc999_2start_orb(natoms_sc999))
  f2 = 0
  do atom_sc999=1, natoms_sc999
    s2 = f2 + 1; f2 = s2 + sp2norbs(atom2sp(atom_sc999_2atom_uc(atom_sc999))) - 1
    atom_sc999_2start_orb(atom_sc999) = s2
  end do

!  !! Check assumption on the construction of real-space matrices
!  atom_sc=37
!  n2 = sp2norbs(atom2sp(atom_sc999_2atom_uc(atom_sc)))
!  s2=atom_sc999_2start_orb(atom_sc); f2 = s2+n2-1
!  Rb = coord(:,atom_sc999_2atom_uc(atom_sc))
!  atom = 22  
!  n1 = sp2norbs(atom2sp(atom));
!  s1 = atom_sc999_2start_orb(atom); f1=s1+n1-1;
!  DeltaR = atom_sc999_2coord(:,atom_sc) - coord(:,atom)
!  Ra = coord(:,atom)
!  do o2=s2,f2
!    do o1=s1,f1
!      if(all(hsx%aB_to_RB_m_Ra(:,o1,o2)==-999)) cycle
!      error = sum(abs(DeltaR-hsx%aB_to_RB_m_Ra(:,o1,o2)))
!      write(6,'(3f10.5,3x,3f10.5,3x,1e20.10,3x,2i5)') &
!        DeltaR+Ra-Rb, hsx%aB_to_RB_m_Ra(:,o1,o2)+Ra-Rb, error, atom_sc, atom
!    enddo ! o1
!  enddo ! o2  
!  !! END of Check assumption on the construction of real-space matrices

!  _die('37 -- 22') 
  
  !! Check assumption on the construction of real-space matrices
  do atom_sc=1,natoms_sc999
    n2 = sp2norbs(atom2sp(atom_sc999_2atom_uc(atom_sc)))
    s2=atom_sc999_2start_orb(atom_sc); f2 = s2+n2-1
    Rb = coord(:,atom_sc999_2atom_uc(atom_sc))
    do atom=1,natoms
      n1 = sp2norbs(atom2sp(atom));
      s1 = atom_sc999_2start_orb(atom); f1=s1+n1-1;
      DeltaR = atom_sc999_2coord(:,atom_sc) - coord(:,atom)
      Ra = coord(:,atom)
      do o2=s2,f2
        do o1=s1,f1
          !if(all(DeltaR==-999)) cycle
          if(all(hsx%aB_to_RB_m_Ra(:,o1,o2)==-999)) cycle
          error = sum(abs(DeltaR-hsx%aB_to_RB_m_Ra(:,o1,o2)))
          if(error>1D-5) then
            write(6,'(3f10.5,3x,3f10.5,3x,1e20.10,3x,2i5)') &
              DeltaR+Ra-Rb, hsx%aB_to_RB_m_Ra(:,o1,o2)+Ra-Rb, error, atom_sc, atom
            _die('not continuous ?')
          endif  
        enddo ! o1
      enddo ! o2  
    enddo
  enddo
  !! END of Check assumption on the construction of real-space matrices

  !! Remove gaps from atom_sc999_2atom_uc...
  do step=1,2
  
    natoms_sc = 0 
    do atom_sc999=1,natoms_sc999
      if(all(atom_sc999_2coord(:,atom_sc999)==-999D0)) cycle
      natoms_sc = natoms_sc + 1
      if(step==2) then
        atom_sc2atom_uc(natoms_sc) = atom_sc999_2atom_uc(atom_sc999)
        atom_sc2coord(:,natoms_sc) = atom_sc999_2coord(:,atom_sc999)
        atom_sc2atom_sc999(natoms_sc) = atom_sc999
      endif   
    end do
  
    if(step==1) then
      _dealloc(atom_sc2atom_uc)
      _dealloc(atom_sc2coord)
      _dealloc(atom_sc2atom_sc999)
      allocate(atom_sc2atom_uc(natoms_sc))
      allocate(atom_sc2coord(3,natoms_sc))
      allocate(atom_sc2atom_sc999(natoms_sc))
    endif
    
  enddo ! step

!  !! Correct coordinates in the unit cell
!  error_coord_sc = sum(abs(coord(1:3,1:natoms) - atom_sc2coord(1:3,1:natoms)))
!  if(iv>0)write(ilog,'(a,es20.12)') 'sum(abs(coord - atom_sc2coord(:,1:natoms)))', error_coord_sc
!  if(error_coord_sc>1D-14) then
!    write(ilog,'(a,es20.10)')'sum(abs(coord-atom_sc2coord(:,1:natoms)))/=0--> convention is broken!', error_coord_sc
!    !stop 'import_from_siesta_arg: sum(abs(coord - atom_sc2coord(:,1:natoms))) /= 0';
!    write(ilog,'(a)') 'warning: insisting on the convention: atom_sc2coord(:,1:natoms) = coord(1:3,1:natoms)'
!    write(ilog,'(a)') 'why I have to do this at ALL ?'
!    atom_sc2coord(1:3,1:natoms) = coord(1:3,1:natoms); ! why I have to do this at ALL ?
!  endif
!  !! END of Correct coordinates in the unit cell
  
!  !! Correct the coordinates in siesta's super cell
!  call hsx_2_uc_vectors(hsx, uc_vecs, iv, ilog);
!  uc_cross(:,1) = cross_product(uc_vecs(:,2),uc_vecs(:,3))
!  uc_cross(:,2) = cross_product(uc_vecs(:,1),uc_vecs(:,3))
!  uc_cross(:,3) = cross_product(uc_vecs(:,1),uc_vecs(:,2))
!  do i=1,3; uuc(i) = sum(uc_cross(:,i)*uc_vecs(:,i)); enddo;
  
!  allocate(atom_sc2coord1(3,natoms_sc))
!  do atom_sc=1,natoms_sc
!    atom = atom_sc2atom_uc(atom_sc)
!    RB_m_Ra = atom_sc2coord(:,atom_sc)-coord(:,atom)
!    do i=1,3; coeff(i) = sum(RB_m_Ra*uc_cross(:,i))/uuc(i); enddo
!    coeff = nint(coeff)
!    atom_sc2coord1(:,atom_sc) = coord(:,atom)
!    do i=1,3 
!      atom_sc2coord1(:,atom_sc) = atom_sc2coord1(:,atom_sc) + coeff(i)*uc_vecs(:,i)
!    enddo
!  enddo
!  error_coord_sc = sum(abs(atom_sc2coord1 - atom_sc2coord))
!  if(iv>0)write(ilog,'(a,es20.12)') 'error_coord_sc among atoms in entire sc ', error_coord_sc
!  atom_sc2coord = atom_sc2coord1
!  !! END of Correct the coordinates in siesta's super cell

  allocate(atom_sc2start_orb(natoms_sc))
  f2 = 0
  do atom_sc=1,natoms_sc
    s2 = f2 + 1; n2 = sp2norbs(atom2sp(atom_sc2atom_uc(atom_sc))); f2 = s2 + n2 - 1;
    atom_sc2start_orb(atom_sc) = s2
  enddo
  !! END of Remove gaps from atom_sc999_2atom_uc...
  
  !! Remove -999 gaps from hsx%hamiltonian and hsx%overlap  
  norbs_sc = 0
  do atom_sc=1,natoms_sc
    no = sp2norbs(atom2sp(atom_sc2atom_uc(atom_sc)))
    norbs_sc = norbs_sc + no
  enddo
  allocate(hamilt(norbs,norbs_sc,nspin))
  allocate(overlap(norbs,norbs_sc))
  allocate(sc_orb2uc_orb(norbs_sc))
  allocate(aB2RB_m_Ra(3,norbs,norbs_sc));
  hamilt = -999
  overlap = -999
  aB2RB_m_Ra = -999
  do atom_sc=1,natoms_sc
    atom_sc999=atom_sc2atom_sc999(atom_sc)
    n2 = sp2norbs(atom2sp(atom_sc2atom_uc(atom_sc)))
    n2999 = sp2norbs(atom2sp(atom_sc999_2atom_uc(atom_sc999)))
    if(n2/=n2999) _die('n2/=n2999')
    s2=atom_sc2start_orb(atom_sc); f2 = s2+n2-1
    s2999=atom_sc999_2start_orb(atom_sc999); f2999=s2999+n2-1
    sc_orb2uc_orb(s2:f2) = hsx%sc_orb2uc_orb(s2999:f2999)
    !write(6,'(i5,3x,1000i5)')atom_sc,sc_orb2uc_orb(s2:f2)
    do atom=1,natoms
      n1 = sp2norbs(atom2sp(atom)); s1 = atom_sc2start_orb(atom); f1=s1+n1-1;
      hamilt(s1:f1,s2:f2,:) = hsx%hamiltonian(s1:f1,s2999:f2999,:)
      overlap(s1:f1,s2:f2)  = hsx%overlap(s1:f1,s2999:f2999)
      do i=1,3
        aB2RB_m_Ra(i,s1:f1,s2:f2) = atom_sc2coord(i,atom_sc)-coord(i,atom)
      enddo
    enddo
  enddo

  !! Compute k-dep overlap right on the spot
  _dealloc(ztensor)
  allocate(ztensor(norbs,norbs))
  kvector = 1D0
  ztensor = 0
  do orb_sc=1,norbs_sc999
    orb2 = hsx%sc_orb2uc_orb(orb_sc)
    do orb1=1, norbs
      if(hsx%overlap(orb1,orb_sc)==0) cycle
      R_Ba(1:3) = hsx%aB_to_RB_m_Ra(1:3, orb1, orb_sc)
      if(all(R_Ba==-999.0D0)) then
        if(hsx%overlap(orb1,orb_sc)/=0D0) _die('why /= 0 ?')
!        cycle
      endif  
      phase = exp(cmplx(0.0D0, sum(R_Ba*kvector),8))
      !write(6,*) o1, o2, O2_sc, phase, kvector
      ztensor(orb1,orb2) = ztensor(orb1,orb2) + phase * hsx%overlap(orb1,orb_sc);
      write(400,'(2i6,3e20.10,2x,e20.10)') orb_sc, orb1, R_Ba, hsx%overlap(orb1,orb_sc)
    end do
  end do
  write(6,*) '|S-S^H|', sum(abs(ztensor - conjg(transpose(ztensor))))

  ztensor = 0
  do orb_sc=1,norbs_sc
    orb2 = sc_orb2uc_orb(orb_sc)
    do orb1=1, norbs
      R_Ba(1:3) = aB2RB_m_Ra(1:3, orb1, orb_sc)
      if(overlap(orb1,orb_sc)==0) cycle
      phase = exp(cmplx(0.0D0, sum(R_Ba*kvector),8))
      !write(6,*) o1, o2, O2_sc, phase, kvector
      ztensor(orb1,orb2) = ztensor(orb1,orb2) + phase * overlap(orb1,orb_sc);
      write(401,'(2i6,3e20.10,2x,e20.10)') orb_sc, orb1, R_Ba, overlap(orb1,orb_sc)
    end do
  end do
  write(6,*) '|S-S^H|', sum(abs(ztensor - conjg(transpose(ztensor))))
  !! END of Compute k-dep overlap right on the spot
  
!  _die('what happens?')


  !write(6,*) sum(hsx%hamiltonian), sum(hamilt)
  !write(6,*) sum(hsx%overlap), sum(overlap)
  !write(6,'(100000i6)') atom_sc2start_orb
  !_die('equal?')
  
  deallocate(hsx%hamiltonian);   allocate(hsx%hamiltonian(norbs,norbs_sc,nspin))
  hsx%hamiltonian = hamilt
  deallocate(hsx%overlap);       allocate(hsx%overlap(norbs,norbs_sc))
  hsx%overlap = overlap
  deallocate(hsx%sc_orb2uc_orb); allocate(hsx%sc_orb2uc_orb(norbs_sc))
  hsx%sc_orb2uc_orb = sc_orb2uc_orb
  deallocate(hsx%ab_to_RB_m_Ra); allocate(hsx%ab_to_RB_m_Ra(3,norbs,norbs_sc))
  hsx%ab_to_RB_m_Ra = ab2RB_m_Ra
  !! Remove -999 gaps from hsx%hamiltonian and hsx%overlap

  if(iv>1) then
    write(ilog,*) 'atom_sc2coord'
    write(ilog,'(3g24.14)')  atom_sc2coord
    write(ilog,*) 'natoms', natoms
    write(ilog,*) 'norbs', norbs
    write(ilog,*) 'norbs_sc', norbs_sc
    write(ilog,*) 'natoms_sc', natoms_sc
  endif
  !! END of Build atom_sc2atom_uc table

  !! Export .xyz files after removing atoms
  _dealloc(atm2sym)
  allocate(atm2sym(natoms_sc))
  do atom=1,natoms_sc
    atm2sym(atom) = z2sym(sp2Z(atom2sp(atom_sc2atom_uc(atom))))
  enddo 

  call write_xyz_file('domiprod-coord_sc.xyz', natoms_sc, atm2sym, &
    atom_sc2coord, 'after removing atoms from sc', iv, ilog);
  !! END of Export .xyz files after removing atoms from unit cell

  if(iv>0) write(ilog,*) 'build_atom_sc2atom_uc_v2: exit'

end subroutine ! build_atom_sc2atom_uc_v2


!
!
!
subroutine siesta_hsx2dft_hsx(hsx, dft_hsx) ! This seems to be spin-ready
  use m_siesta_hsx, only : siesta_hsx_t, sparse2full
  use m_die, only : die
  implicit none
  type(siesta_hsx_t), intent(in) :: hsx
  type(dft_hsx_t), intent(inout)   :: dft_hsx
  !! internal
  integer :: icol, nspin, n, n_sc, ispin, ixyz, orb, i, irow, sparse_ind
  integer, allocatable :: row2displ(:)
  integer, allocatable :: row2nnzero(:);
  integer, allocatable :: sparse_ind2column(:)
  real(8), allocatable :: m_sparse_d(:)
  
  if(.not. allocated(hsx%h_sparse)) _die('!h_sparse')
  if(.not. allocated(hsx%s_sparse)) _die('!s_sparse')
  if(.not. allocated(hsx%aB2RaB_sparse)) _die('!aB2RaB_sparse')
!  if(.not. allocated(hsx%sc_orb2uc_orb)) _die('!sc_orb2uc_orb')

  n_sc = hsx%norbitals_sc
  n    = hsx%norbitals
  nspin= hsx%nspin

  !! Initialize the DFT structure
_dealloc(dft_hsx%hamiltonian)
_dealloc(dft_hsx%overlap)
_dealloc(dft_hsx%ab_to_RB_m_Ra)

  allocate(dft_hsx%hamiltonian(n, n_sc, nspin));
  allocate(dft_hsx%overlap(n, n_sc));
  allocate(dft_hsx%sc_orb2uc_orb(n_sc));

  !! Initialize the index array supercell orbital --> unit cell orbital 
  if(.not. hsx%is_gamma) then 
    dft_hsx%sc_orb2uc_orb = hsx%sc_orb2uc_orb;
  else 
    if(n/=n_sc) _die('(n/=n_sc)')
    do orb=1,n; dft_hsx%sc_orb2uc_orb(orb) = orb; enddo;
  endif
  !! END of Initialize the index array supercell orbital --> unit cell orbital

  !! Fill the displacements (according to row2nnzero) row2displ
  allocate( row2displ(hsx%norbitals) );
  row2displ(1)=0
  do icol=2, hsx%norbitals
    row2displ(icol) = row2displ(icol-1) + hsx%row2nnzero(icol-1)
  enddo
  !! END of Fill the displacements (according to row2nnzero) row2displ
  allocate(m_sparse_d(size(hsx%h_sparse,1)))
  allocate(row2nnzero(size(hsx%row2nnzero)));
  allocate(sparse_ind2column(size(hsx%sparse_ind2column)));
  row2nnzero = hsx%row2nnzero;
  sparse_ind2column = hsx%sparse_ind2column;
  do ispin=1,nspin;
    m_sparse_d = hsx%h_sparse(:,ispin)
    call sparse2full(n, n_sc, dft_hsx%hamiltonian(1,1,ispin), n, &
      m_sparse_d, row2nnzero, row2displ, sparse_ind2column);
  end do;

  m_sparse_d = hsx%s_sparse
  call sparse2full(n, n_sc, dft_hsx%overlap, n, m_sparse_d, row2nnzero, row2displ, sparse_ind2column)

  allocate(dft_hsx%ab_to_RB_m_Ra(3,n,n_sc))
  dft_hsx%ab_to_RB_m_Ra = -999
  do ixyz=1,3
    do irow=1,n
      do i=1,row2nnzero(irow);
        sparse_ind = row2displ(irow)+i;
        icol = sparse_ind2column(sparse_ind);
        dft_hsx%ab_to_RB_m_Ra(ixyz,irow,icol) = hsx%aB2RaB_sparse(ixyz,sparse_ind) !! a bit mess with columns and rows
      enddo
    enddo
  enddo ! ixyz
  !! END of Initialize the DFT structure
  dft_hsx%tot_electr_chrg = hsx%total_electronic_charge
  dft_hsx%is_gamma = hsx%is_gamma

_dealloc(row2displ)
_dealloc(row2nnzero)
_dealloc(sparse_ind2column)
_dealloc(m_sparse_d)
  
end subroutine ! siesta_hsx2dft_hsx

end module !m_dft_hsx
