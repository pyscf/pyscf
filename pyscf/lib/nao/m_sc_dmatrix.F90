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

module m_sc_dmatrix

#include "m_define_macro.F90"
  use m_log, only : die
  use m_arrays, only : d_array2_t
  
  implicit none

  type sc_dmatrix_t
    type(d_array2_t), allocatable :: pair2submat(:) !             1..npairs
    real(8), allocatable :: pair2cells(:,:,:)       ! 1..3, 1..2, 1..npairs -- integers!
    integer, allocatable :: pair2atoms(:,:)         ! 1..2,       1..npairs
    integer :: irr_trans_inv_sym = -1
    integer :: trans_op = -1
  end type ! sc_dmatrix_t

contains 

!
!
!
subroutine dealloc(rmat)
  implicit none
  type(sc_dmatrix_t), intent(inout) :: rmat
  
  _dealloc(rmat%pair2submat)
  _dealloc(rmat%pair2cells)
  _dealloc(rmat%pair2atoms)
!  _dealloc(rmat%)
  
end subroutine ! dealloc

!
!
!
real(8) function get_sum(rmat)
  implicit none
  type(sc_dmatrix_t), intent(in), target :: rmat
  !! internal
  integer :: np, p
  real(8), pointer :: dat(:,:)
  get_sum = 0
  np = get_stored_npairs(rmat)
  do p=1,np
    dat => get_stored_data_ptr(rmat, p)
    get_sum = get_sum + sum(dat)
  enddo 
  
end function ! get_sum


! 
! Initialize sc_t with information from sv%hs%overlap and other
! i.e. build a custom super cell
!
subroutine get_sc_list(rmat, sc_list, pair2atoms_sc)
  use m_sets, only : dlist2set
  implicit none
  !! external
  type(sc_dmatrix_t), intent(in) :: rmat
  real(8), allocatable, intent(inout) :: sc_list(:,:) ! (atom,cell(1:3); atom_sc)
  integer, allocatable, intent(inout) :: pair2atoms_sc(:,:) ! pair; 1:2 atoms
  !! internal
  integer :: np, p, atoms(2), cells(3,2), ind, natoms_sc
  real(8), allocatable :: ind2atom_cell(:,:)
  integer, allocatable :: l2s(:)
  
  np = get_stored_npairs(rmat)
  allocate(ind2atom_cell(4,2*np))
  ind2atom_cell = 0
  ind = 0
  do p=1,np
    atoms = get_stored_atoms(rmat, p)
    cells = int(get_stored_cells(rmat, p))
    ind = ind + 1
    ind2atom_cell(:,ind) = (/atoms(1), cells(1:3,1) /)
    ind = ind + 1
    ind2atom_cell(:,ind) = (/atoms(2), cells(1:3,2) /)
  enddo ! p

  call dlist2set(ind2atom_cell, natoms_sc, sc_list, l2s)

  _dealloc(pair2atoms_sc)
  allocate(pair2atoms_sc(np,2))
  do p=1,np
    pair2atoms_sc(p,1:2) = (/ l2s(2*p-1), l2s(2*p) /)
  enddo ! p
  
end subroutine ! get_sc_list


!
! Allocates top-level arrays in sc_dmatrix_t
!
subroutine alloc_sc_dmatrix(npairs, irr_trans_inv_sym, trans_op, rmat)
  !! external
  implicit none
  integer, intent(in) :: npairs, irr_trans_inv_sym, trans_op
  type(sc_dmatrix_t), intent(inout) :: rmat
  !! internal
 
  _dealloc(rmat%pair2atoms)
  _dealloc(rmat%pair2submat)
  _dealloc(rmat%pair2cells)
  allocate(rmat%pair2atoms(2,npairs))
  allocate(rmat%pair2submat(npairs))
  allocate(rmat%pair2cells(3,2,npairs))
  rmat%pair2atoms = -999
  rmat%pair2cells = -999
  rmat%irr_trans_inv_sym = irr_trans_inv_sym
  rmat%trans_op = trans_op
  
end subroutine ! alloc_sc_dmatrix

!
! Allocate or reallocate the submatrix for a given pair with a given sizes
!
subroutine alloc_submat( p, nn, rmat)
  !! external
  implicit none
  integer, intent(in) :: p, nn(2)
  type(sc_dmatrix_t), intent(inout) :: rmat
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  if(p<1 .or. p>np) _die('p<1 .or. p>np')
  if(any(nn<1)) _die('any(nn<1)')

  if(allocated(rmat%pair2submat(p)%array)) then
    if(any(ubound(rmat%pair2submat(p)%array)/=nn))deallocate(rmat%pair2submat(p)%array)
  endif
  if(.not. allocated(rmat%pair2submat(p)%array)) &
    allocate(rmat%pair2submat(p)%array(nn(1),nn(2)))

  rmat%pair2submat(p)%array=0
end subroutine !  alloc_submat 

!
! This returns a symmetry index 
!   1 -- inversion symmetry is reduced in the stored data
!   0 -- inversion symmetry is present in the stored data
!
function get_sym(sc_mat) result(a)
  implicit none
  type(sc_dmatrix_t), intent(in) :: sc_mat
  integer :: a
  
  integer :: np
  np = get_stored_npairs(sc_mat)
  if(np<1) _die('np<1')
  if(sc_mat%irr_trans_inv_sym<0 .or. sc_mat%irr_trans_inv_sym>1) _die('sym?')
  a = sc_mat%irr_trans_inv_sym

end function ! get_sym

!
! This returns a symmetry index 
!   1 -- inversion symmetry is reduced in the stored data
!   0 -- inversion symmetry is present in the stored data
!
function get_trans_op(sc_mat) result(a)
  implicit none
  type(sc_dmatrix_t), intent(in) :: sc_mat
  integer :: a
  
  integer :: np
  np = get_stored_npairs(sc_mat)
  if(np<1) _die('np<1')
  if(sc_mat%trans_op<0 .or. sc_mat%trans_op>1) _die('trans_op?')
  a = sc_mat%trans_op

end function ! get_trans_op

!
! This returns the number of pairs which are stored in the structure
!
function get_stored_npairs(sc_mat) result(a)
  implicit none
  type(sc_dmatrix_t), intent(in) :: sc_mat
  integer :: a
  
  integer :: aa(3)
  aa = 0
  if(allocated(sc_mat%pair2submat)) aa(1) = size(sc_mat%pair2submat)
  if(allocated(sc_mat%pair2cells))  aa(2) = size(sc_mat%pair2cells,3)
  if(allocated(sc_mat%pair2atoms))  aa(3) = size(sc_mat%pair2atoms,2)
  a = aa(1)
  if(any(aa/=a)) _die('any(aa/=a)')
  if(a<1) _die('(a<1)')
  
end function ! get_stored_npairs

!
! This returns atom indices (within unit cell) of a given stored pair
!
function get_stored_atoms(rmat, p) result(a)
  implicit none
  type(sc_dmatrix_t), intent(in) :: rmat
  integer, intent(in) :: p![air] stored
  integer :: a(2)
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  if(p<1 .or. p>np) _die('p<1 .or. p>np')
  a = rmat%pair2atoms(:,p)
  
end function ! get_stored_atoms

!
! This returns cells integer translations for a given stored pair
!
function get_stored_cells(rmat, p) result(cells)
  implicit none
  type(sc_dmatrix_t), intent(in) :: rmat
  integer, intent(in) :: p![air] stored
  real(8) :: cells(1:3,1:2) ! translation along 1..3 direction, local atom index 1..2
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  if(p<1 .or. p>np) _die('p<1 .or. p>np')
  cells = rmat%pair2cells(:,:,p)
end function ! get_stored_cells

!
! Get a pointer to stored data
!
function get_stored_data_ptr(rmat, p) result(dat)
  implicit none
  type(sc_dmatrix_t), intent(in), target :: rmat
  integer, intent(in) :: p![air] stored
  real(8), pointer    :: dat(:,:)
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  if(p<1 .or. p>np) then 
    write(6,*) 'p, np ', p, np
    _die('p<1 .or. p>np')
  endif  
  if(.not. allocated(rmat%pair2submat(p)%array)) _die('data not allocated')
  dat => rmat%pair2submat(p)%array
  
end function ! get_stored_data_ptr

!
! Get dimensions of a given stored submatrix
!
function get_stored_data_dims(rmat, p) result(dims)
  implicit none
  type(sc_dmatrix_t), intent(in), target :: rmat
  integer, intent(in) :: p![air] stored
  integer :: dims(2)
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  if(p<1 .or. p>np) _die('p<1 .or. p>np')
  dims = ubound(rmat%pair2submat(p)%array)
  
end function ! get_stored_data_dims

!
! Sets the atom indices for a given pair
!
subroutine set_stored_atoms(p, atoms, rmat)
  !! external
  implicit none
  integer, intent(in) :: p, atoms(2)
  type(sc_dmatrix_t), intent(inout) :: rmat
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  if(p<1 .or. p>np) _die('p<1 .or. p>np')

  rmat%pair2atoms(1:2,p) = atoms(1:2)
end subroutine ! set_stored_atoms

!
! Sets the atom indices for a given pair
!
subroutine set_stored_cells(p, cells, rmat)
  !! external
  implicit none
  integer, intent(in) :: p
  real(8), intent(in) :: cells(3,2)
  type(sc_dmatrix_t), intent(inout) :: rmat
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  if(p<1 .or. p>np) _die('p<1 .or. p>np')

  rmat%pair2cells(1:3,1:2,p) = cells(1:3,1:2)
  
end subroutine ! set_stored_atoms

!
!
!
subroutine modify_rhs_sc_dmatrix(ao_sp2m,atom2sp,sc_mat)
  implicit none
  !! external
  integer, intent(in) :: ao_sp2m(:,:)
  integer, intent(in) :: atom2sp(:)
  type(sc_dmatrix_t), intent(inout) :: sc_mat
  !! internal
  integer :: pair, npairs, sp(2), ao1, ao2, m1, m2, nao(2)
  
  npairs = get_stored_npairs(sc_mat)
  do pair=1, npairs
    sp(1:2) = atom2sp(sc_mat%pair2atoms(1:2,pair))
    nao(1:2) = ubound(sc_mat%pair2submat(pair)%array)
    do ao2=1,nao(2)
      m2 = ao_sp2m(ao2, sp(2))
      do ao1=1,nao(1)
        m1 = ao_sp2m(ao1, sp(1))
        sc_mat%pair2submat(pair)%array(ao1,ao2) = &
          sc_mat%pair2submat(pair)%array(ao1,ao2) * (-1D0)**m1* (-1D0)**m2
      enddo ! ao1    
    enddo ! ao2
      
  enddo ! pair

!    type(d_array2_t), allocatable :: pair2submat(:) !       1..npairs
!    integer, allocatable :: pair2cells(:,:,:)       ! 1..3, 1..2, 1..npairs
!    integer, allocatable :: pair2atoms(:,:)         ! 1..2, 1..npairs
!    integer :: irr_trans_inv_sym = -1
end subroutine ! modify_rhs_sc_dmatrix  
 
!
! The rect_mat2sc_mat is obsolete because it is not general!!!
!
! Initialization of the super cell tensor type(sc_dmatrix_t) from 
! rectangular full matrix, rect_mat:
!
subroutine rect_mat2sc_mat(rect_mat, atom_sc2atom_uc, atom_sc2start_orb, sp2norbs, &
  atom2specie, uc_vecs, atom_sc2coord, sc_mat)
  use m_bulk_uc_vecs, only : get_coeff_vecs
  implicit none
  !! external
  real(8), intent(in) :: rect_mat(:,:)
  integer, intent(in) :: atom_sc2atom_uc(:)
  integer, intent(in) :: atom_sc2start_orb(:)
  integer, intent(in) :: sp2norbs(:)
  integer, intent(in) :: atom2specie(:)
  real(8), intent(in) :: atom_sc2coord(:,:)
  real(8), intent(in) :: uc_vecs(3,3)
  type(sc_dmatrix_t), intent(inout) :: sc_mat
  !! internal
  integer :: atom_sc, sp2,n2,si2,fi2, sp1,n1,si1,fi1, atom
  integer :: pair, npairs, irr_trans_inv_sym
  real(8) :: sa, shift_of_atom2(3), coeff2(3), coeff_vecs(3,3)
  !! Dimensions
  integer :: natoms_sc, natoms
  natoms_sc = size(atom_sc2atom_uc)
  if(natoms_sc/=size(atom_sc2coord,2)) _die("natoms_sc/=size(atom_sc2coord,2)")
  natoms = size(atom2specie)
  !! END of Dimensions

  !! Count atom pairs which have some non-zero matrix elements  
  npairs = 0
  do atom_sc=1,natoms_sc
    sp2=atom2specie(atom_sc2atom_uc(atom_sc));
    si2=atom_sc2start_orb(atom_sc); n2 = sp2norbs(sp2); fi2 = si2 + n2 - 1;
   
    do atom=1,natoms
      sp1=atom2specie(atom_sc2atom_uc(atom))
      si1=atom_sc2start_orb(atom);  n1 = sp2norbs(sp1); fi1 = si1 + n1 - 1;
      sa=sum(abs(rect_mat(si1:fi1,si2:fi2)))
      if(sa>0) npairs = npairs + 1
    enddo ! atom
  enddo ! atom_sc
  !! END of Count atom pairs which have some non-zero matrix elements 

  irr_trans_inv_sym = 0
  call alloc_sc_dmatrix(npairs, irr_trans_inv_sym, 0, sc_mat)
  
  !! Fill the structure with data in two loops: first a<=B; second a>B (if at all);
  !! First loop a<B
  coeff_vecs = get_coeff_vecs(uc_vecs)
  pair = 0
  do atom_sc=1,natoms_sc
    sp2=atom2specie(atom_sc2atom_uc(atom_sc));
    si2=atom_sc2start_orb(atom_sc); n2 = sp2norbs(sp2); fi2 = si2 + n2 - 1
    
    shift_of_atom2 = atom_sc2coord(:,atom_sc) - atom_sc2coord(:,atom_sc2atom_uc(atom_sc))
    coeff2 = nint(matmul(coeff_vecs,shift_of_atom2))
    
    do atom=1,min(atom_sc,natoms)
      sp1=atom2specie(atom_sc2atom_uc(atom))
      si1=atom_sc2start_orb(atom);  n1 = sp2norbs(sp1); fi1 = si1 + n1 - 1
      sa=sum(abs(rect_mat(si1:fi1,si2:fi2)))
      if(sa==0) cycle
      pair = pair + 1
      sc_mat%pair2atoms(1,pair) = atom_sc2atom_uc(atom)
      sc_mat%pair2atoms(2,pair) = atom_sc2atom_uc(atom_sc)
      sc_mat%pair2cells(:,1,pair) = 0 ! first atom is unit cell always
      sc_mat%pair2cells(:,2,pair) = nint(coeff2) ! second atom can be outside of unit cell
      allocate(sc_mat%pair2submat(pair)%array(n1,n2))
      sc_mat%pair2submat(pair)%array = rect_mat(si1:fi1,si2:fi2)
    enddo ! atom
  enddo ! atom_sc
  !! END of First loop a<B

  !! Complementary loop
  do atom_sc=1,natoms_sc
    sp2=atom2specie(atom_sc2atom_uc(atom_sc));
    si2=atom_sc2start_orb(atom_sc); n2 = sp2norbs(sp2); fi2 = si2 + n2 - 1
    
    shift_of_atom2 = atom_sc2coord(:,atom_sc) - atom_sc2coord(:,atom_sc2atom_uc(atom_sc))
    coeff2 = nint(matmul(coeff_vecs,shift_of_atom2))
    
    do atom=min(atom_sc,natoms)+1,natoms
      sp1=atom2specie(atom_sc2atom_uc(atom))
      si1=atom_sc2start_orb(atom);  n1 = sp2norbs(sp1); fi1 = si1 + n1 - 1
      sa=sum(abs(rect_mat(si1:fi1,si2:fi2)))
      if(sa==0) cycle
      pair = pair + 1
      sc_mat%pair2atoms(1,pair) = atom_sc2atom_uc(atom)
      sc_mat%pair2atoms(2,pair) = atom_sc2atom_uc(atom_sc)
      sc_mat%pair2cells(:,1,pair) = 0 ! first atom is unit cell always
      sc_mat%pair2cells(:,2,pair) = nint(coeff2) ! second atom can be outside of unit cell
      allocate(sc_mat%pair2submat(pair)%array(n1,n2))
      sc_mat%pair2submat(pair)%array = rect_mat(si1:fi1,si2:fi2)
    enddo ! atom
  enddo ! atom_sc
  !! END of !! Complementary loop

  !! END of Fill the structure with data
  
  sc_mat%irr_trans_inv_sym = 0 
  sc_mat%trans_op = 0
  
end subroutine ! rect_mat2sc_mat

!
!
!
subroutine get_bloch_matrix_irr(rmat, kvec, atom2coord, uc_vecs, atom2start_orb,atom2specie,sp2norbs, zmat)
  implicit none
  !! external
  type(sc_dmatrix_t), intent(in) :: rmat
  real(8), intent(in) :: uc_vecs(3,3), atom2coord(:,:)
  real(8), intent(in) :: kvec(3)
  integer, intent(in) :: atom2start_orb(:), atom2specie(:), sp2norbs(:)
  complex(8), intent(inout) :: zmat(:,:)
  !! internal
  integer :: pair, atoms(2), cells(3,2), norb_mx, s(2), f(2), n(2)
  real(8) :: svecs(3,2), R21(3)
  real(8), allocatable :: submat(:,:)
  complex(8) :: phase
  
  norb_mx = get_submat_maxdim(rmat)
  allocate(submat(norb_mx,norb_mx))
  
  zmat = 0
  do pair=1, get_stored_npairs(rmat)
    
    atoms = rmat%pair2atoms(:,pair)
    s = atom2start_orb(atoms(:))
    n = sp2norbs(atom2specie(atoms))
    f = s + n - 1

    cells = int(rmat%pair2cells(:,:,pair))
    submat(1:n(1),1:n(2)) = rmat%pair2submat(pair)%array
    
    svecs = matmul(uc_vecs(1:3,1:3), cells(1:3,1:2))
    R21 = atom2coord(:,atoms(2)) + svecs(:, 2) - atom2coord(:,atoms(1)) - svecs(:, 1)
    phase = exp(cmplx(0.0D0, sum(R21*kvec),8))
    zmat(s(1):f(1), s(2):f(2)) = zmat(s(1):f(1), s(2):f(2)) + submat(1:n(1),1:n(2))*phase


    atoms(1) = rmat%pair2atoms(2,pair)
    atoms(2) = rmat%pair2atoms(1,pair)
    s = atom2start_orb(atoms(:))
    n = sp2norbs(atom2specie(atoms))
    f = s + n - 1
    submat(1:n(1), 1:n(2)) = transpose(rmat%pair2submat(pair)%array)
    cells(:,1) = int(rmat%pair2cells(:,2,pair))
    cells(:,2) = int(rmat%pair2cells(:,1,pair))


    svecs = matmul(uc_vecs(1:3,1:3), cells(1:3,1:2))
    R21 = atom2coord(:,atoms(2)) + svecs(:, 2) - atom2coord(:,atoms(1)) - svecs(:, 1)
    phase = exp(cmplx(0.0D0, sum(R21*kvec),8))

    if(any(cells/=0)) then
      zmat(s(1):f(1), s(2):f(2)) = zmat(s(1):f(1), s(2):f(2)) + submat(1:n(1),1:n(2))*phase
    endif
    
    if(all(cells==0) .and. atoms(1)/=atoms(2)) then
      zmat(s(1):f(1), s(2):f(2)) = zmat(s(1):f(1), s(2):f(2)) + submat(1:n(1),1:n(2))*phase
    endif  

  enddo ! pair

end subroutine ! get_bloch_matrix_irr

!
!
!
function get_submat_maxdim(rmat) result(a)
  implicit none
  !! external
  type(sc_dmatrix_t), intent(in) :: rmat
  integer :: a
  !! internal
  integer :: s(2), nstored, pair
  a = -1
  nstored = get_stored_npairs(rmat)
  do pair = 1, nstored
    if(.not. allocated(rmat%pair2submat(pair)%array)) cycle
    s(1) = size(rmat%pair2submat(pair)%array,1)
    s(2) = size(rmat%pair2submat(pair)%array,2)
    a = max(a, s(1))
    a = max(a, s(2))
  enddo
  
end function !get_submat_maxdim 


!
!
!
function get_natoms(rmat) result(na)
  implicit none
  !! external
  type(sc_dmatrix_t), intent(in) :: rmat
  integer :: na
  !! internal
  integer :: np
  np = get_stored_npairs(rmat)
  na = maxval(rmat%pair2atoms(1:2,1:np))
  if(na<1) _die('na<1') 
  
end function ! get_natoms  

!
! 
!
function get_abs_dcells_max(rmat) result(n123)
  
  implicit none
  !! external
  type(sc_dmatrix_t), intent(in) :: rmat
  integer :: n123(3)
  !! internal
  integer :: np, i
  
  np = get_stored_npairs(rmat) 
  do i=1,3
    n123(i) = maxval(abs(int(rmat%pair2cells(i,2,1:np)-rmat%pair2cells(i,1,1:np)))) ! 1..3, 1..2, 1..npairs 
  enddo ! i
  
  if(any(n123<0)) _die('any(n123<0)')
  
end function ! get_abs_n123_sc_max


end module !m_sc_dmatrix
