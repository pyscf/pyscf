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

module m_system_vars

#include "m_define_macro.F90"

  use m_arrays, only : d_array2_t
  use m_dft_hsx, only : dft_hsx_t, dealloc_dft_hsx=>dealloc
  use m_siesta_dipo_types, only : dft_dipo_t, dealloc_dipo=>dealloc
  use m_hs, only : hs_t, dealloc_hs=>dealloc
  use m_hsx, only : hsx_t, dealloc_hsx=>dealloc
  use m_die, only : die
  use m_warn, only : warn
  use m_dft_wf4, only : dft_wf4_t, dealloc_dft_wf4=>dealloc
  use m_dft_wf8, only : dft_wf8_t, dealloc_dft_wf8=>dealloc
  use m_uc_skeleton, only : uc_skeleton_t, dealloc_uc=>dealloc
  use m_siesta_ion, only : siesta_ion_t
    
  implicit none

  private d_array2_t, dft_hsx_t, dft_wf8_t, uc_skeleton_t

  type system_vars_t
     type(dft_hsx_t)      :: dft_hsx  !
     type(dft_wf4_t)      :: dft_wf4  !
     type(dft_wf8_t)      :: dft_wf8  !
     type(dft_dipo_t)     :: dft_dipo ! not ready for read/write in sv type
     type(hsx_t)          :: hsx      ! sparse H and S and X (X -- coordinate differences)
     type(hs_t)           :: hs
     type(uc_skeleton_t)  :: uc
     integer, allocatable :: atom_sc2start_orb(:)
     integer, allocatable :: atom_sc2atom_uc(:) ! 
     real(8), allocatable :: atom_sc2coord(:,:) ! xyz,atom

     real(8), allocatable :: rr(:)
     real(8), allocatable :: pp(:)
     real(8), allocatable :: psi_log_rl(:,:,:) ! nr, nmult_max, nspecies 
     real(8), allocatable :: psi_log(:,:,:) ! nr, nmult_max, nspecies 
     
     real(8), allocatable :: nsk2occ(:,:,:) ! molecular orbital, spin 2 occupation... maybe changed! nsk2occ ?
     real(8) :: Temp = -999
     
     !! all these integers in principle reducible
     integer :: norbs = -999
     integer :: norb_max = -999
     integer :: natoms = -999
     integer :: nspin  = -999
     integer :: jmx    = -999
     integer :: nocc   = -999

     real(8), allocatable :: core_log(:,:) ! nr, nspecies 
     integer, allocatable :: sp2has_core(:)
     real(8), allocatable :: sp2rcut_core(:)
     type(siesta_ion_t), allocatable :: sp2ion(:)

     logical :: is_dft_wf4 = .True. !Use single precision by default
  end type !system_vars_t


  interface get_fullmat_overlap
    module procedure get_fullmat_overlap_real4
    module procedure get_fullmat_overlap_real8
  end interface !get_fullmat_overlap
  
  
contains

!
!
!
subroutine dealloc(sv)
  implicit none
  type(system_vars_t), intent(inout) :: sv

  call dealloc_dft_hsx(sv%dft_hsx)
  call dealloc_dft_wf4(sv%dft_wf4)
  call dealloc_dft_wf8(sv%dft_wf8)
  call dealloc_dipo(sv%dft_dipo)
  call dealloc_hsx(sv%hsx)
  call dealloc_hs(sv%hs)
  call dealloc_uc(sv%uc)
  _dealloc(sv%atom_sc2start_orb)
  _dealloc(sv%atom_sc2atom_uc)
  _dealloc(sv%atom_sc2coord)
  _dealloc(sv%rr)
  _dealloc(sv%pp)
  _dealloc(sv%psi_log_rl)
  _dealloc(sv%psi_log)
  _dealloc(sv%nsk2occ)
  _dealloc(sv%core_log)
  _dealloc(sv%sp2has_core)
  _dealloc(sv%sp2rcut_core)
  _dealloc(sv%sp2ion)

  sv%Temp = -999
  sv%norbs = -999 
  sv%norb_max = -999
  sv%natoms = -999
  sv%nspin = -999
  sv%jmx = -999
  sv%nocc = -999
  sv%is_dft_wf4 = .true.

end subroutine ! dealloc

!
!
!
subroutine init_size_dft_wf_X(size_x, sv)
  use m_dft_wf4, only : init_size_X

  implicit none
  type(system_vars_t), intent(inout) :: sv
  integer, intent(in) :: size_x(1:5)

  if (sv%is_dft_wf4) then
    call init_size_X(size_x, sv%dft_wf4)
  else
    _die("implemented only for dft_wf4")
  endif

end subroutine !init_size_dft_wf_x
!
!
!
function get_uc_vecs_ptr(sv) result(uc_vecs)
  implicit none
  !! external
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: uc_vecs(:,:)
  !! internal
  
  uc_vecs => sv%uc%uc_vecs
end function !get_uc_vecs_ptr
  

!
! 
!
function get_fermi_energy(sv) result(E)
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8) :: E

  if (sv%is_dft_wf4) then
    E = sv%dft_wf4%fermi_energy
  else 
    E = sv%dft_wf8%fermi_energy
  endif
end function !get_fermi_energy


!
! 
!
function get_eigenvalues_shift(sv) result(E)
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8) :: E
  if (sv%is_dft_wf4) then
    E = sv%dft_wf4%eigenvalues_shift
  else
    E = sv%dft_wf8%eigenvalues_shift
  endif
    
end function !get_eigenvalues_shift


!
!
!
function get_sp2ion_ptr(sv) result(ptr)
  use m_siesta_ion, only : siesta_ion_t
!  use m_system_vars, only : system_vars_t
  implicit none
  type(system_vars_t), intent(in), target :: sv
  type(siesta_ion_t), pointer :: ptr(:)

  ptr => sv%sp2ion
  if(.not. allocated(sv%sp2ion)) _warn('!alloc sv%sp2ion(:)')      
  
end function !get_sp2ion_ptr  

!
!
!
function get_nsk2occ_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:,:)
  !! internal
  integer :: no, ns
  no = get_norbs(sv)
  ns = get_nspin(sv)
  ptr=>sv%nsk2occ
  if(no/=size(ptr,1)) _die('!E norbs?')
  if(ns/=size(ptr,2)) _die('!E nspin?')
end function ! get_nsk2occ_ptr

!
!
!
function get_eigvecs_ptr4(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(4), pointer :: ptr(:,:,:,:,:)
  !! internal
  
  ptr => null()
  if (.not. allocated(sv%dft_wf4%X))_die('!')
  ptr => sv%dft_wf4%X
  
end function ! get_eigvecs_ptr

!
!
!
function get_eigvecs_ptr8(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:,:,:,:)
  !! internal
  
  ptr => null()
  if (.not. allocated(sv%dft_wf8%X)) _die('!')
  ptr => sv%dft_wf8%X
  
end function ! get_eigvecs_ptr

!
!
!
function get_eigvals_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:,:)
  !! internal
  integer :: no, ns
  no = get_norbs(sv)
  ns = get_nspin(sv)
  if (sv%is_dft_wf4) then
    ptr=>sv%dft_wf4%E
  else
    ptr=>sv%dft_wf8%E 
  endif
  if(no/=size(ptr,1)) _die('!E norbs?')
  if(ns/=size(ptr,2)) _die('!E nspin?')
end function ! get_eigvals_ptr

!
!
!
function get_core_log_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:)
  !! internal
  integer :: nn(2), nn_ref(2)
  if(.not. allocated(sv%core_log)) _die('!%core_log')
  nn = ubound(sv%core_log)
  nn_ref = [get_nr(sv), get_nspecies(sv)]
  if(any(nn/=nn_ref)) _die('nn/=nn_ref')
  ptr=>sv%core_log
end function ! get_core_log_ptr  

!
!
!
logical function has_nonlin_core_correct(sv)
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  !! internal
  integer :: nsp
  if(.not. allocated(sv%sp2has_core)) _die('!%sp2has_core')
  nsp = get_nspecies(sv)
  if(nsp/=size(sv%sp2has_core)) _die('nsp/=size(%sp2has_core)')
  if(any(sv%sp2has_core<0)) _die('!init %sp2has_core')
  has_nonlin_core_correct = any(sv%sp2has_core>0)
  
end function ! has_nonlin_core_correct


!
!
!
real(8) function get_rcut_core_max(sv) 
  implicit none
  type(system_vars_t), intent(in) :: sv
  if(.not. allocated(sv%sp2rcut_core)) _die('!%sp2rcut_core')
  get_rcut_core_max = maxval(sv%sp2rcut_core)
end function ! get_rcut_core_max  

!
!
!
function get_rcuts_core(sv, atoms) result(rcuts)
  use m_uc_skeleton, only : get_atom2sp_ptr
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: atoms(:)
  real(8) :: rcuts(size(atoms))
  !! internal
  integer :: ia, nsp, sp
  integer, pointer :: atom2sp(:)
  
  if(.not. allocated(sv%sp2rcut_core)) _die('!%sp2rcut_core')
  nsp = get_nspecies(sv)
  if(nsp/=size(sv%sp2rcut_core)) _die('nsp/=size(sv%sp2rcut_core)')
  
  atom2sp => get_atom2sp_ptr(sv%uc)
  
  do ia=1,size(atoms)
    sp = atom2sp(atoms(ia))
    rcuts(ia) = sv%sp2rcut_core(sp)
  enddo
    
end function ! get_rcuts_core


!
!
!
function get_atom_sc2coord_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:)
  !! internal
  integer :: nsp
  nsp = get_natoms_sc(sv)
  if(nsp<1) _die('nsp<1')
  ptr => sv%atom_sc2coord
end function ! get_atom_sc2coord_ptr

!
!
!
function get_atom_sc2atom_uc_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: ptr(:)
  !! internal
  integer :: nsp
  nsp = get_natoms_sc(sv)
  if(nsp<1) _die('nsp<1')
  ptr => sv%atom_sc2atom_uc
end function ! get_atom_sc2atom_uc_ptr

!
!
!
function get_atom_sc2start_orb_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: ptr(:)
  !! internal
  integer :: nsp
  nsp = get_natoms_sc(sv)
  if(nsp<1) _die('nsp<1')  
  ptr => sv%atom_sc2start_orb
end function ! get_atom_sc2start_orb_ptr

!
! Computes a mean radii for each specie (average over multipletts)
!
subroutine comp_sp2radii(psi_log, nmultipletts, mu_sp2j, rr, sp2radii)
  use m_functs_l_mult_type, only : get_radii
  !!  external
  implicit none
  real(8), intent(in) :: psi_log(:,:,:)
  integer, intent(in) :: nmultipletts(:)
  integer, intent(in) :: mu_sp2j(:,:)
  real(8), intent(in) :: rr(:)
  real(8), intent(out) :: sp2radii(:)
  !! internal
  integer :: sp, mu, norb, j
  real(8) :: dlambda, mean_radii, radii
  !! Dimensions
  integer :: nr, nspecies
  nr = size(rr)
  nspecies = size(psi_log,3)

  dlambda = log(rr(nr)/rr(1))/(nr-1)
  do sp=1, nspecies
    mean_radii = 0
    norb = 0
    do mu=1,nmultipletts(sp)
      j = mu_sp2j(mu,sp)
      radii = get_radii(psi_log(:,mu,sp), rr, dlambda)
      mean_radii = mean_radii + radii*(2*j+1D0)/2D0
      norb = norb + (2*j+1)
    end do
    mean_radii = mean_radii / norb
    sp2radii(sp) = mean_radii
  end do;
end subroutine ! comp_sp2radii

!
!
!
function get_psi_log_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:,:)
  !! internal
  integer :: nsp
  nsp = get_nspecies(sv)
  if(.not. allocated(sv%psi_log)) _die('!psi_log')
  if(size(sv%psi_log,3)/=nsp) _die('? size3')
  ptr => sv%psi_log
end function ! get_psi_log_ptr

!
!
!
function get_psi_log_rl_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:,:)
  !! internal
  integer :: nsp
  nsp = get_nspecies(sv)
  if(.not. allocated(sv%psi_log_rl)) _die('!psi_log_rl')
  if(size(sv%psi_log_rl,3)/=nsp) _die('? size3')
  ptr => sv%psi_log_rl
end function ! get_psi_log_rl_ptr


!
!
!
function get_atom2sfo_ptr(sv) result(ptr)
  use m_uc_skeleton, only : get_atom2sfo_ptr_uc=>get_atom2sfo_ptr
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: ptr(:,:)
  ptr => get_atom2sfo_ptr_uc(sv%uc)
end function ! get_atom2sfo_ptr

!
!
!
function get_sp2norbs_ptr(sv) result(ptr)
  use m_uc_skeleton, only : get_sp2norbs_ptr_uc=>get_sp2norbs_ptr
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: ptr(:)
  ptr => get_sp2norbs_ptr_uc(sv%uc)
end function ! get_sp2norbs_ptr

!
!
!
function get_mu_sp2start_ao_ptr(sv) result(ptr)
  use m_uc_skeleton, only : get_mu_sp2start_ao_ptr_uc=>get_mu_sp2start_ao_ptr
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: ptr(:,:)
  ptr => get_mu_sp2start_ao_ptr_uc(sv%uc)
end function ! get_mu_sp2start_ao_ptr

!
!
!
function get_mu_sp2j_ptr(sv) result(ptr)
  use m_uc_skeleton, only : get_mu_sp2j_ptr_uc=>get_mu_sp2j_ptr
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: ptr(:,:)
  ptr => get_mu_sp2j_ptr_uc(sv%uc)
end function ! get_mu_sp2j_ptr

!
!
!
function get_sp2nmult_ptr(sv) result(ptr)
  use m_uc_skeleton, only : get_sp2nmult_ptr_uc=>get_sp2nmult_ptr
  implicit none
  !! external
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: ptr(:)
  !! internal
  ptr => get_sp2nmult_ptr_uc(sv%uc)
end function ! get_sp2nmult_ptr


!
!
!
function get_mu_sp2rcut_ptr(sv) result(ptr)
  use m_uc_skeleton, only : get_mu_sp2rcut_ptr_uc=>get_mu_sp2rcut_ptr
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:)
  ptr => get_mu_sp2rcut_ptr_uc(sv%uc)
end function ! get_mu_sp2rcut_ptr


!
!
!
function get_atom2sp_ptr(sv) result(atom2specie_ptr)
  use m_uc_skeleton, only : get_atom2sp_ptr_uc=>get_atom2sp_ptr
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, pointer :: atom2specie_ptr(:)
  atom2specie_ptr =>get_atom2sp_ptr_uc(sv%uc)
end function ! get_atom2sp_ptr 

!
!
!
subroutine get_atom2elem(sv, atom2elem)
  use m_uc_skeleton, only : get_sp2element_ptr
  implicit none
  type(system_vars_t), intent(in), target :: sv
  integer, allocatable, intent(inout) :: atom2elem(:)
  !!
  integer :: natoms, a
  integer, pointer :: atom2sp(:)
  integer, pointer :: sp2elem(:)
  atom2sp=>get_atom2sp_ptr(sv)
  sp2elem=>get_sp2element_ptr(sv%uc)
  natoms = get_natoms(sv)
  
  _dealloc(atom2elem)
  allocate(atom2elem(natoms))
  atom2elem = 0
  
  do a=1,natoms
    atom2elem(a) = sp2elem(atom2sp(a))
  enddo ! a  
  
end subroutine ! get_atom2znuc
    

!
!
!
character(100) function get_BlochPhaseConv(sv) 
  implicit none
  type(system_vars_t), intent(in) :: sv

  if (sv%is_dft_wf4) then
    get_BlochPhaseConv = get_BlochPhaseConv4(sv)
  else
    get_BlochPhaseConv = get_BlochPhaseConv8(sv)
  endif
end function ! get_BlochPhaseConv

!
!
!
character(100) function get_BlochPhaseConv4(sv) 
  use m_dft_wf4, only : get_BlochPhaseConv_wf=>get_BlochPhaseConv
  implicit none
  type(system_vars_t), intent(in) :: sv
  get_BlochPhaseConv4 = get_BlochPhaseConv_wf(sv%dft_wf4)
end function ! get_BlochPhaseConv

!
!
!
character(100) function get_BlochPhaseConv8(sv) 
  use m_dft_wf8, only : get_BlochPhaseConv_wf=>get_BlochPhaseConv
  implicit none
  type(system_vars_t), intent(in) :: sv
  get_BlochPhaseConv8 = get_BlochPhaseConv_wf(sv%dft_wf8)
end function ! get_BlochPhaseConv


!
!
!
logical function is_init(sv) 
  implicit none
  type(system_vars_t), intent(in) :: sv
  is_init = .true.

  is_init = is_init .and. (get_nr(sv)>0)
  is_init = is_init .and. (get_jmx(sv)>=0)
  is_init = is_init .and. (get_nspecies(sv)>0)
  is_init = is_init .and. (get_natoms(sv)>0)
  is_init = is_init .and. (get_natoms_sc(sv)>0)
  is_init = is_init .and. (get_nspin(sv)>0 .and. get_nspin(sv)<3)
  is_init = is_init .and. (get_nmult_max(sv)>0)
  is_init = is_init .and. (get_norbs(sv)>0)
  is_init = is_init .and. (get_norbs_sc(sv)>0)
  is_init = is_init .and. (get_norbs_max(sv)>0)
  is_init = is_init .and. (get_basis_type(sv)>0 .and. get_basis_type(sv)<3)
  is_init = is_init .and. (get_tot_electr_chrg(sv)>0)
  is_init = is_init .and. (get_nkpoints(sv)>0)
  
end function ! is_init

!
!
!
function get_pp_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:)
  !! 
  if(.not. allocated(sv%pp)) _die('.not. allocated(sv%pp)')
  ptr => sv%pp
end function ! get_pp_ptr

!
!
!
function get_rr_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:)
  !! 
  if(.not. allocated(sv%rr)) _die('.not. allocated(sv%rr)')
  ptr => sv%rr
end function ! get_rr_ptr


!
!
!
function get_atom2coord_ptr(sv) result(ptr)
  implicit none
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:)
  !! 
  if(.not. allocated(sv%uc%atom2coord)) _die('.not. allocated(sv%uc%atom2coord)')
  ptr => sv%uc%atom2coord
end function ! get_atom2coord_ptr 
  
!
!
!
function get_uc_ptr(sv) result(ptr)
  implicit none
  !! external
  type(system_vars_t), intent(in), target :: sv
  type(uc_skeleton_t), pointer :: ptr

  ptr => sv%uc
end function !  get_uc_ptr
 
!
!
!
function get_rcuts(sv, atoms) result(rcuts)
  use m_uc_skeleton, only : get_rcuts_ucs=>get_rcuts
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: atoms(:)
  real(8) :: rcuts(size(atoms))
  !! internal
  rcuts = get_rcuts_ucs(sv%uc, atoms)
end function ! get_rcuts

!
!
!
function get_coords(sv, atoms, cells, n)    result(coords)
  use m_uc_skeleton, only : get_coords_ucs=>get_coords
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: n
  integer, intent(in) :: atoms(n)
  real(8), intent(in) :: cells(3,n)
  real(8) :: coords(3,n)
  !! internal
  coords = get_coords_ucs(sv%uc, atoms, cells, n)
  
end function ! get_coords



!
!
!
function get_rcut_max(sv) result(rc)
  use m_uc_skeleton, only : get_rcut_max_ucs=>get_rcut_max
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  real(8) :: rc
  !! internal
  rc = get_rcut_max_ucs(sv%uc)
  
end function ! get_rcut_max

!
!
!
function get_overlap_sc_ptr(sv) result(overlap_sc)
  use m_sc_dmatrix, only : sc_dmatrix_t
  implicit none
  !! external
  type(system_vars_t), intent(in), target :: sv
  type(sc_dmatrix_t), pointer :: overlap_sc
  !! internal
  overlap_sc => sv%hs%overlap
end function ! get_overlap_sc_ptr

!
!
!
function get_hamiltonian_sc_ptr(sv) result(h_sc)
  use m_sc_dmatrix, only : sc_dmatrix_t
  implicit none
  !! external
  type(system_vars_t), intent(in), target :: sv
  type(sc_dmatrix_t), pointer :: h_sc(:)
  !! internal
  
  h_sc => sv%hs%spin2hamilt
  
end function ! get_hamiltonian_sc_ptr

!
!
!
subroutine shift_mean_center(sv, iv, ilog)
#define _sname 'shift_mean_center'
  implicit none
  !! external
  type(system_vars_t), intent(inout) :: sv
  integer, intent(in) :: iv, ilog
  !! internal
  real(8) :: center(3)
  integer :: i, natoms
  natoms = get_natoms(sv)
  
  do i=1,3; center(i) = sum(sv%atom_sc2coord(i,1:natoms))/natoms; enddo
  do i=1, natoms; sv%atom_sc2coord(:,i) = sv%atom_sc2coord(:,i)-center; enddo;

  !!! This is bad! There m ust be just one array with coordinates!!!
  sv%uc%atom2coord(1:3,1:natoms) = sv%atom_sc2coord(1:3,1:natoms)
  
  call create_xyz('S', sv, 'atom_sc2coord_shifted.txt', iv,ilog)
  if(iv>0)write(ilog,*)_sname//': shift by: '
  if(iv>0)write(ilog,*) real(center),' Bohr, done.'
#undef _sname
end subroutine ! shift_mean_center

!
!
!
subroutine create_xyz(ccell, sv, fname, iv,ilog)
  use m_upper, only : upper
  use m_xyz, only : create_xyz_xyz=>create_xyz
  implicit none
  !! external
  character :: ccell
  character(*), intent(in) :: fname
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: iv, ilog
  !! internal
  integer :: natoms, natoms_sc

  if(upper(ccell)=='U') then
  
    natoms = get_natoms(sv)
    call create_xyz_xyz(trim(fname), sv%uc%atom2sp, sv%uc%sp2element, &
      sv%atom_sc2atom_uc, natoms, sv%uc%atom2coord, &
      trim(sv%uc%systemlabel)//' unit cell', iv,ilog)
      
  else if (upper(ccell)=='S') then
  
    natoms_sc = get_natoms_sc(sv)
    call create_xyz_xyz(trim(fname), sv%uc%atom2sp, sv%uc%sp2element, &
      sv%atom_sc2atom_uc, natoms_sc, sv%atom_sc2coord, &
      trim(sv%uc%systemlabel)//' super cell', iv,ilog)
      
  else
    _die('unknown ccell')
  endif      

end subroutine ! create_xyz

!
!
!
subroutine get_atom_sc2start_orb(sp2norbs, atom2sp, atom_sc2atom_uc, atom_sc2start_orb)
  implicit none
  !! external 
  integer, allocatable, intent(in) :: sp2norbs(:), atom2sp(:), atom_sc2atom_uc(:)
  integer, allocatable, intent(inout) :: atom_sc2start_orb(:)
  !! internal
  integer :: nsp1, nsp2, nsp, natoms_sc, s,f,n, atom_sc
  _dealloc(atom_sc2start_orb)
  nsp1 = 0; if(allocated(sp2norbs)) nsp1 = size(sp2norbs)
  nsp2 = 0; if(allocated(atom2sp)) nsp2 = maxval(atom2sp)
  if(nsp1/=nsp2) _die('nsp1/=nsp2')
  nsp = nsp1
  if(nsp<1) _die('nsp<1')
  natoms_sc = 0
  if(allocated(atom_sc2atom_uc)) natoms_sc = size(atom_sc2atom_uc)
  if(natoms_sc<0) _die('natoms_sc<0')
  
  allocate(atom_sc2start_orb(natoms_sc))
  atom_sc2start_orb = -999
  f = 0
  do atom_sc=1,natoms_sc
    s = f + 1; n = sp2norbs(atom2sp(atom_sc2atom_uc(atom_sc))); f = s + n - 1
    atom_sc2start_orb(atom_sc) = s
  enddo ! atom_sc  

end subroutine ! get_atom_sc2start_orb  

!
!
!
subroutine report_info_sv(fname, sv, iv)
  use m_log, only : log_size_note
  use m_io, only : get_free_handle
  !! external
  implicit none
  character(*), intent(in) :: fname
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: iv
  !! internal
  integer :: ifile, ios

  ifile = get_free_handle()
  open(ifile, file=fname, action='write', iostat=ios)
  if(ios/=0) _die('ios/=0')
  call print_info_sv(ifile, sv)
  close(ifile)
  call log_size_note('written: ', trim(fname), iv)

end subroutine ! report_info_sv

!
!
!
subroutine print_info_sv(ifile, sv)
  use m_algebra, only : cross_product
  implicit none
  !! external
  integer, intent(in) :: ifile
  type(system_vars_t), intent(in) :: sv
  !! internal
  integer :: norbs, nocc, nvirt, bt, nsp, sp
  real(8) :: v_uc, uc_vecs(3,3)
  
  write(ifile,'(a43,i8)') 'get_nr(sv)', get_nr(sv)
  write(ifile,'(a43,i8)') 'get_jmx(sv)', get_jmx(sv)
  nsp = get_nspecies(sv)
  write(ifile,'(a43,i8)') 'get_nspecies(sv)', nsp
  write(ifile,'(a43,i8)') 'get_natoms(sv)', get_natoms(sv)
  write(ifile,'(a43,i8)') 'get_natoms_sc(sv)', get_natoms_sc(sv)
  write(ifile,'(a43,i8)') 'get_nspin(sv)', get_nspin(sv)
  write(ifile,'(a43,i8)') 'get_nmult_max(sv)', get_nmult_max(sv)
  norbs = get_norbs(sv)
  write(ifile,'(a43,i8)') 'get_norbs(sv)', norbs
  write(ifile,'(a43,i8)') 'get_norbs_sc(sv)', get_norbs_sc(sv)
  write(ifile,'(a43,i8)') 'get_norbs_max(sv)', get_norbs_max(sv)
  write(ifile,'(a43,e20.12)') 'get_vnn(sv)', get_vnn(sv)

  bt = get_basis_type(sv)
  write(ifile,'(a43,i8)') 'get_basis_type(sv)', bt
  if(bt==1) then
    nocc = get_nocc(sv)
    write(ifile,'(a43,i8)') 'get_nocc(sv)', nocc
    nvirt = get_nvirt(sv)
    write(ifile,'(a43,i8)') 'get_nvirt(sv)', nvirt
  endif

  write(ifile,'(a43,g15.8)') 'get_tot_electr_chrg(sv) ', get_tot_electr_chrg(sv)
  write(ifile,'(a43,i8)')    'get_nkpoints(sv)', get_nkpoints(sv)
  write(ifile,'(a43,g15.8)') 'get_temperature(sv) ', get_temperature(sv)
  write(ifile,'(a43,g15.8)') 'get_fermi_energy(sv) ', get_fermi_energy(sv)

  write(ifile,'(a43)') 'sv%uc%uc_vecs'
  write(ifile,'(3g16.8)') sv%uc%uc_vecs
  V_uc = 0.0D0
  V_uc = abs(dot_product( uc_vecs(:,1),cross_product(uc_vecs(:,2),uc_vecs(:,3))))
  !write(ifile,'(a43,f12.8)') 'Volume of unit cell: ',  V_uc
  write(ifile, *) 'Volume of unit cell: ',  V_uc

  write(ifile,'(a35)') 'sv%uc%sp2nmult(1:nspecies)'
  write(ifile,'(200i4)') sv%uc%sp2nmult
  write(ifile,'(a35)') 'sv%uc%mu_sp2j(1:nmult(sp),sp)'
  do sp=1,nsp
    write(ifile,'(i6,a,3x,200i4)') sp,' : ', sv%uc%mu_sp2j(1:sv%uc%sp2nmult(sp),sp)
  enddo ! sp
  write(ifile,'(a35)') 'sv%uc%mu_sp2rcut(1:nmult(sp),sp)'
  do sp=1,nsp
    write(ifile,'(i6,a,3x,200f10.5)') sp,' : ', sv%uc%mu_sp2rcut(1:sv%uc%sp2nmult(sp),sp)
  enddo ! sp
  write(ifile,'(a35)') 'sv%uc%atom2sp(1:natoms)'
  write(ifile,'(20000000i4)') sv%uc%atom2sp

  write(ifile,'(a35)') 'sv%uc%sp2element'
  write(ifile,'(20000000i4)') sv%uc%sp2element
   
end subroutine ! print_info_sv

!
! Calculate nuclear repulsion energy
!
function get_vnn(sv) result(vnn)
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8) :: vnn
  ! internal
  integer :: ii,ij, nnuc
  real(8) :: rx, Z1, Z2

  vnn = 0
  nnuc = get_natoms(sv)
  if (nnuc .gt.1) then
    do ii=1,(nnuc-1)
      do ij=(ii+1),nnuc 
        rx=sqrt(sum((sv%atom_sc2coord(:,ii)-sv%atom_sc2coord(:,ij))**2))
        if(rx<1d-12) cycle
        Z1 = sv%uc%sp2element(sv%uc%atom2sp(ii))
        Z2 = sv%uc%sp2element(sv%uc%atom2sp(ij))
        vnn = vnn + Z1*Z2/rx 
      enddo
    enddo
  endif
  
end function ! get_vnn

!
! Initialize mu2si(:) helper array
!
subroutine mu2si_from_mu2j(mu2j, mu2si)
  implicit none
  !! external
  integer, intent(in), allocatable :: mu2j(:)
  integer, intent(inout), allocatable :: mu2si(:)
  !! internal
  integer :: nmu, imu
  nmu = 0
  if(allocated(mu2j)) nmu = size(mu2j)
  _dealloc(mu2si)
  if(nmu<1) return
  allocate(mu2si(nmu))
  
  mu2si = -999
  mu2si = 1
  do imu=1,nmu-1
    mu2si(imu+1) = mu2si(imu) + 2*mu2j(imu)+1
  enddo ! imu 
end subroutine ! mu2si_from_mu2j

!
! Counts maximal number of l-multipletts, does also a number of cross checks
!
function get_nmult_max(sv) result(n)
  use m_uc_skeleton, only : get_nmult_max_ucs=>get_nmult_max
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = get_nmult_max_ucs(sv%uc)
    
end function ! get_nmult_max  

!
! Gets coordinates of atoms in unit cell
!
subroutine get_atom2coord(sv, atom2coord)
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  real(8), allocatable, intent(inout) :: atom2coord(:,:)
  !! internal
  integer :: natoms
  _dealloc(atom2coord)
  natoms = get_natoms(sv)
  allocate(atom2coord(3,natoms))
  atom2coord = sv%atom_sc2coord(1:3,1:natoms)
end subroutine ! get_atom2coord

!
! Finds out whether basis functions are local orbitals or periodically extended orbitals.
! Periodically extended orbitals are Bloch's orbitals.
! They are used in bulk calculations.
!
function get_basis_type(sv) result(isys_type)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: isys_type
  
  isys_type = -999
  
  if (allocated(sv%dft_wf4%X)) then
      
    isys_type = size(sv%dft_wf4%X,1)
    if(isys_type==2 .and. sv%dft_hsx%is_gamma) &
      _die("hsx and wf are not consistent.")

    if(isys_type==1 .and. (.not. sv%dft_hsx%is_gamma)) &
      _die("hsx and wf are not consistent.")

  elseif (allocated(sv%dft_wf8%X)) then
      
    isys_type = size(sv%dft_wf8%X,1)
    if(isys_type==2 .and. sv%dft_hsx%is_gamma) &
      _die("hsx and wf are not consistent.")

    if(isys_type==1 .and. (.not. sv%dft_hsx%is_gamma)) &
      _die("hsx and wf are not consistent.")
  else
      _die("can't determine basis type.")
  endif


    
end function ! get_basis_type

!    
! function returns number of occupied molecular orbitals
!
function get_nocc(sv) result(n)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = int((get_tot_electr_chrg(sv)+1)/2)
  if(n/=sv%nocc) then
    write(0,*) 'get_nocc:    nocc', n
    write(0,*) 'get_nocc: sv%nocc', sv%nocc
    stop 'get_nocc: (nocc/=sv%nocc)'
  endif  
end function !get_nocc

!
!
!
function get_zelem(sv, specie) result(zelem)
  use m_uc_skeleton, only : get_zelem_ucs=>get_zelem
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: specie
  integer :: zelem
  
  zelem = get_zelem_ucs(sv%uc, specie)
  
end function ! get_zelem  
  
!
! This should return a full-matrix-stored overlap matrix between orbitals
! For now it just should return the already-stored overlap
! 
subroutine get_fullmat_overlap_real4(sv, o)
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  real(4), allocatable, intent(inout) :: o(:,:)
  integer :: norbs, norbs_sc, nn(2)
  if(.not. allocated(sv%dft_hsx%overlap)) _die('could calc, but...')

  norbs = get_norbs(sv)
  norbs_sc = get_norbs_sc(sv)
  nn = [norbs, norbs_sc]
_dealloc_u(o, nn)
  if(.not. allocated(o)) allocate(o(nn(1),nn(2)))
  o = real(sv%dft_hsx%overlap, 4)
end subroutine ! get_fullmat_overlap  


!
! This should return a full-matrix-stored overlap matrix between orbitals
! For now it just should return the already-stored overlap
! 
subroutine get_fullmat_overlap_real8(sv, o)
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  real(8), allocatable, intent(inout) :: o(:,:)
  integer :: norbs, norbs_sc, nn(2)
  if(.not. allocated(sv%dft_hsx%overlap)) _die('could calc, but...')

  norbs = get_norbs(sv)
  norbs_sc = get_norbs_sc(sv)
  nn = [norbs, norbs_sc]
_dealloc_u(o, nn)
  if(.not. allocated(o)) allocate(o(nn(1),nn(2)))
  o = sv%dft_hsx%overlap
end subroutine ! get_fullmat_overlap  

!
! This should return a full-matrix-stored overlap matrix between orbitals
! For now it just should return the already-stored overlap
! 
function get_fullmat_overlap_ptr(sv) result(ptr)
  implicit none
  !! external
  type(system_vars_t), intent(in), target :: sv
  real(8), pointer :: ptr(:,:)
  if(.not. allocated(sv%dft_hsx%overlap)) _die('could calc, but...')
  ptr => sv%dft_hsx%overlap
end function ! get_fullmat_overlap_ptr

! 
! Returns the total electronic charge of the system
! For now the number already-stored in sv%dft_hsx%tot_electr_chrg
!
function get_tot_electr_chrg(sv) result(tot_electr_chrg)
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  real(8) :: tot_electr_chrg
  if(sv%dft_hsx%tot_electr_chrg==-999) _die('N_e=-999 suspicious!')
  tot_electr_chrg = sv%dft_hsx%tot_electr_chrg
  
end function ! get_tot_electr_chrg
 

!
! Initializes xmin=0; xmax=0; ymin=0; ymax=0; zmin=0; zmax=0;
!
subroutine get_box(sv, box)
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(inout) :: box(3,2)
  integer :: natoms, ixyz
  real(8) :: rcut
  natoms = get_natoms(sv)
  rcut = maxval(sv%uc%mu_sp2rcut)
  do ixyz=1,3
    box(ixyz,1) = minval(sv%atom_sc2coord(ixyz,1:natoms)-rcut)
    box(ixyz,2) = maxval(sv%atom_sc2coord(ixyz,1:natoms)+rcut)
  enddo
end subroutine !get_box

!
!
!
function get_nkpoints(sv) result(n)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = 0

  if (sv%is_dft_wf4) then
    if( allocated(sv%dft_wf4%kpoints)) then
      n=size(sv%dft_wf4%kpoints,2)
      if(n/=size(sv%dft_wf4%X,5)) stop 'get_nkpoints: nkpoints/=size(sv%dft_wf4%X,5)'
    else
      if(allocated(sv%dft_wf4%X) ) then
        if(size(sv%dft_wf4%X,5)/=1) then
          write(0,*) 'get_nkpoints: ubound(sv%dft_wf4%X)', ubound(sv%dft_wf4%X)
          stop 'get_nkpoints: size(sv%dft_wf4%X,5)/=1'
        endif
      endif
    endif
  else
    if( allocated(sv%dft_wf8%kpoints)) then
      n=size(sv%dft_wf8%kpoints,2)
      if(n/=size(sv%dft_wf8%X,5)) stop 'get_nkpoints: nkpoints/=size(sv%dft_wf8%X,5)'
    else
      if(allocated(sv%dft_wf8%X) ) then
        if(size(sv%dft_wf8%X,5)/=1) then
          write(0,*) 'get_nkpoints: ubound(sv%dft_wf8%X)', ubound(sv%dft_wf8%X)
          stop 'get_nkpoints: size(sv%dft_wf8%X,5)/=1'
        endif
      endif
    endif
  endif
  if(n<1) _die('n<1')
end function !get_nkpoints

!
!
!
function get_kvec(sv, ik) result(kvec)
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: ik
  real(8) :: kvec(3)
  !! internal
  integer :: nk
  kvec = 0
  nk = get_nkpoints(sv)
  
  if (sv%is_dft_wf4) then
    if(0<ik .and. ik<=nk) kvec = sv%dft_wf4%kpoints(:,ik)
  else
    if(0<ik .and. ik<=nk) kvec = sv%dft_wf8%kpoints(:,ik)
  endif
  
end function ! get_kvec

!
! function returns number of radial points on logarithmic mesh
!
function get_nr(sv) result(n)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n=size(sv%rr)
  if(n/=size(sv%rr)) stop 'get_nr: nr/=size(sv%rr)'
  if(n/=size(sv%pp)) stop 'get_nr: nr/=size(sv%pp)'
  if(n/=size(sv%psi_log,1)) stop 'get_nr: nr/=size(sv%psi_log,1)'
  if(n<1) _die('n<1')
end function !get_nr


!
! function returns maximal number of angular momenta in the calculation
!
function get_jmx(sv) result(n)
  use m_uc_skeleton, only : get_jmx_ucs=>get_jmx
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = get_jmx_ucs(sv%uc)
end function !get_jmx


!
! function returns number of species (of atoms) in the calculation
!
function get_nspecies(sv) result(n)
  use m_uc_skeleton, only : get_nspecies_ucs=>get_nspecies
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = get_nspecies_ucs(sv%uc)
end function !get_nspecies

!
! function returns number of atoms in the calculation
!
function get_natoms(sv) result(n)
  use m_uc_skeleton, only : get_natoms_ucs=>get_natoms
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = get_natoms_ucs(sv%uc)
  
end function !get_natoms

! function returns number of atoms in the calculation
!
function get_natoms_sc(sv) result(n)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  integer :: nn(3)
  nn(1) = size(sv%atom_sc2coord,2)
  nn(2) = size(sv%atom_sc2start_orb)
  nn(3) = size(sv%atom_sc2atom_uc)
  if(any(nn/=nn(1))) _die('any(nn/=nn(1))')
  n = nn(1)
  if(n<1) _die('(natoms_sc<1')
end function !get_natoms_sc

!
! function returns (electronic) temperature
!
function get_temperature(sv) result(T)
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8) :: T
  T = sv%temp
end function !get_temperature

!
! function returns number of spins in the calculation
!
function get_nspin(sv) result(n)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = sv%nspin

  if (sv%is_dft_wf4) then
    if(n/= sv%dft_wf4%size_X(2)) then
      print*, "n = ", n, "sv%dft_wf4%size_X = ", sv%dft_wf4%size_X
      _die('(nspin/= sv%dft_wf4%size_X(2))')
    endif
  else
    _die("only for dft_wf4!")
  endif
  if(n<1) _die('nspin<1')
end function !get_nspin 

!
! function returns number of localized orbitals in sv
!
function get_norbs(sv) result(n)
  use m_uc_skeleton, only : get_norbs_ucs=>get_norbs
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n, a(2)
  a(1) = sv%norbs
  if(a(1)<1) _die('get_norbs<1: sv not initialized.')
  a(2) = get_norbs_ucs(sv%uc)
  n = a(1)
  if(any(a/=n)) _die('any(a/=n)')
  if(n<1) _die('n<1')

end function !get_norbs

!
! function returns maximal number of localized orbitals per atom in sv
!
function get_norbs_max(sv) result(n)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  integer, allocatable :: sp2norbs(:)
  call get_sp2norbs(sv, sp2norbs)
  n = maxval(sp2norbs)
  if(sv%norb_max/=n) then
    write(0,*) 'get_norbs_max: sv%norb_max, norbs_max', sv%norb_max, n
    stop 'get_norbs_max: (norbs_max/=sv%norbs_max)'
  endif
  if(n<1) _die('n<1')
end function !get_norbs_max

!
! function returns number of localized orbitals in sv
!
function get_norbs_sc(sv) result(n)
  use m_uc_skeleton, only : get_sp
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n, atom, natoms_sc, sp
  integer, allocatable :: sp2norbs(:)
  call get_sp2norbs(sv, sp2norbs)
  natoms_sc = get_natoms_sc(sv)
  n = 0
  do atom=1,natoms_sc
    sp = get_sp(sv%uc, sv%atom_sc2atom_uc(atom))
    n = n + sp2norbs(sp)
  enddo
  if(n<1) _die('n<1')
! a cross check ?  
!  if(n/=ncount) then
!    write(0,*)'get_norbs: norbs, ncount', n, ncount 
!    stop 'get_norbs: (norbs/=ncount)'
!  endif  
  
end function !get_norbs_sc

!    
! function returns number of unoccupied (virtual) molecular orbitals
!
function get_nvirt(sv) result(n)
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer :: n
  n = get_norbs(sv)-get_nocc(sv) 
  if(n/=sv%norbs-sv%nocc) stop 'get_nvirt: (nvirt/=sv%norbs-sv%nocc)'
  if(n<1) _die('n<1')
end function !get_nvirt

!
!
!
subroutine get_atom2rcut(sv, atm2rcut)
  use m_uc_skeleton, only : get_sp
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(inout), allocatable :: atm2rcut(:)
  
  integer:: natoms, atom, sp
  
  natoms = get_natoms(sv)
  _dealloc(atm2rcut)
  allocate(atm2rcut(natoms))
  do atom=1,natoms
    sp = get_sp(sv%uc, atom)
    atm2rcut(atom) = maxval(sv%uc%mu_sp2rcut(:,sp))
  enddo
 
end subroutine !get_atm2rcut

!
!
!
subroutine get_atom2rcut_atom2rcut2(sv, atm2rcut, atm2rcut2)
  use m_uc_skeleton, only : get_sp
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(inout), allocatable :: atm2rcut(:), atm2rcut2(:)
  
  integer:: natoms, atom, sp
  
  natoms = get_natoms(sv)
  allocate(atm2rcut(natoms))
  allocate(atm2rcut2(natoms))
  do atom=1,natoms
    sp = get_sp(sv%uc, atom)
    atm2rcut(atom) = maxval(sv%uc%mu_sp2rcut(:,sp))
  enddo
  atm2rcut2 = atm2rcut**2;
  
end subroutine !get_atm2rcut_atm2rcut2

!
!
!
subroutine get_atom2rcut2(sv, atom2rcut2)
  use m_uc_skeleton, only : get_sp
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(inout), allocatable :: atom2rcut2(:)
  integer:: natoms, atom, sp
  natoms = get_natoms(sv)
  if(allocated(atom2rcut2))deallocate(atom2rcut2); allocate(atom2rcut2(natoms))
  do atom=1,natoms
    sp = get_sp(sv%uc, atom)
    atom2rcut2(atom) = maxval(sv%uc%mu_sp2rcut(:,sp))**2
  enddo
end subroutine !get_atom2rcut2

!
!
!
subroutine get_sp2rcut2(sv, sp2rcut2)
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(inout), allocatable :: sp2rcut2(:)
  integer:: nspecies, sp
  nspecies = get_nspecies(sv)
  _dealloc(sp2rcut2)
  allocate(sp2rcut2(nspecies))
  do sp=1,nspecies;
    sp2rcut2(sp) = maxval(sv%uc%mu_sp2rcut(:,sp))**2;
  enddo
end subroutine !get_sp2rcut2

!
!
!
subroutine get_sp2rcut(sv, sp2rcut)
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(inout), allocatable :: sp2rcut(:)
  integer:: nspecies, sp
  nspecies = get_nspecies(sv)
  _dealloc(sp2rcut)
  allocate(sp2rcut(nspecies))
  do sp=1,nspecies; sp2rcut(sp) = maxval(sv%uc%mu_sp2rcut(:,sp)); enddo
end subroutine !get_sp2rcut


!
!
!
subroutine get_sp2norbs(sv, sp2norb)
  use m_uc_skeleton, only : get_sp2norbs_ucs=>get_sp2norbs
  implicit none
  type(system_vars_t), intent(in) :: sv
  integer, allocatable, intent(inout) :: sp2norb(:)  
  integer:: nsp
  nsp = get_nspecies(sv)
  _dealloc(sp2norb)
  allocate(sp2norb(nsp))
  call get_sp2norbs_ucs(sv%uc, sp2norb)
  
end subroutine !get_sp2norbs

!
!
!
subroutine get_atom2norbs_sp2norbs(sv, atom2norbs, sp2norbs)
  use m_uc_skeleton, only : get_sp
  type(system_vars_t), intent(in) :: sv    
  integer, allocatable, intent(inout) :: atom2norbs(:), sp2norbs(:)  
  integer:: natoms, atm
  
  call get_sp2norbs(sv, sp2norbs)

  natoms = get_natoms(sv)  
  _dealloc(atom2norbs)
  allocate(atom2norbs(natoms));
  do atm=1,natoms; atom2norbs(atm)= sp2norbs(get_sp(sv%uc, atm)); enddo
  
end subroutine !get_atom2norbs_sp2norbs

!
!
!
subroutine get_atom2norbs(sv, atom2norbs)
  use m_uc_skeleton, only : get_sp
  type(system_vars_t), intent(in) :: sv    
  integer, allocatable, intent(inout) :: atom2norbs(:)  
  !! internal
  integer:: natoms, atm
  integer, allocatable :: sp2norbs(:)
  
  call get_sp2norbs(sv, sp2norbs)

  natoms = get_natoms(sv)  
  _dealloc(atom2norbs)
  allocate(atom2norbs(natoms))
  do atm=1,natoms; 
    atom2norbs(atm)= sp2norbs(get_sp(sv%uc, atm))
  enddo
  
  _dealloc(sp2norbs)
end subroutine !get_atom2norbs

!
! Computes geometrical center of unit cell
!
function get_geom_center_uc(sv) result(geom_center)
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  real(8) :: geom_center(3)
  !! internal
  integer :: natoms, i
  natoms = get_natoms(sv)
  do i=1,3; geom_center(i) = sum(sv%atom_sc2coord(i,1:natoms)) / natoms; enddo

end function ! get_geom_center_uc

!
! Computes the k-dependent tensor (bloch's tensor) from block-stored rectangular
! tensor. The set of blocks must include only inversion symmetry irreducible 
! atom pairs.
!
subroutine block_irr2bloch(pair2atoms, atom_sc2atom_uc, atom2start_orb, &
  pair2block, kvector, atom_sc2coord, norbs, ztenzor, ldz, zblock );
  implicit none
  !! external
  integer, intent(in) :: pair2atoms(:,:)
  integer, intent(in) :: atom_sc2atom_uc(:)
  integer, intent(in) :: atom2start_orb(:)
  type(d_array2_t), intent(in) :: pair2block(:)
  real(8), intent(in) :: kvector(3)
  real(8), intent(in) :: atom_sc2coord(3,*)
  integer, intent(in) :: norbs, ldz
  complex(8), intent(out) :: ztenzor(ldz,*)
  complex(8), intent(inout) :: zblock(:,:) ! norb_max,norb_max
  !! internal
  integer :: pair, atom_a, atom_B, atom_b_uc, n1, n2, s1, s2, f1, f2
  real(8) :: RB_m_Ra(3)
  !! Dimensions
  integer :: npairs
  complex(8), parameter :: ci = (0D0, 1D0);
  npairs = size(pair2block)
  if(size(pair2atoms,2)/=npairs) stop 'block_irr2bloch: size(pair2atoms,2)/=npairs'  
  !! END of Dimensions

  ztenzor(1:norbs, 1:norbs) = 0
  do pair=1, npairs
    if( .not. allocated(pair2block(pair)%array)) cycle
    atom_a = pair2atoms(1, pair);
    atom_B = pair2atoms(2, pair);
    atom_b_uc = atom_sc2atom_uc(atom_B);
    n1 = size(pair2block(pair)%array,1)
    n2 = size(pair2block(pair)%array,2)
    s1 = atom2start_orb(atom_a);    f1 = s1 + n1 - 1;
    s2 = atom2start_orb(atom_b_uc); f2 = s2 + n2 - 1;
    RB_m_Ra = atom_sc2coord(1:3,atom_B) - atom_sc2coord(1:3,atom_a)
    zblock(1:n1,1:n2) = exp(ci*sum(RB_m_Ra*kvector))*pair2block(pair)%array
    ztenzor(s1:f1,s2:f2) = ztenzor(s1:f1,s2:f2) + zblock(1:n1,1:n2)
    
    if(atom_a/=atom_B) then
      ztenzor(s2:f2,s1:f1) = ztenzor(s2:f2,s1:f1) + conjg(transpose(zblock(1:n1,1:n2)))
    endif
  enddo

end subroutine ! block_irr2bloch

!
! Computes the k-dependent tensor (bloch's tensor) from block-stored rectangular
! tensor. The set of blocks must include all atom pairs, also reducible by 
! inversion symmetry
!
subroutine block_red2bloch(pair2atoms, atom_sc2atom_uc, atom2start_orb, &
  pair2block, kvector, atom_sc2coord, norbs, ztenzor, ldz );
  implicit none
  !! external
  integer, intent(in) :: pair2atoms(:,:)
  integer, intent(in) :: atom_sc2atom_uc(:)
  integer, intent(in) :: atom2start_orb(:)
  type(d_array2_t), intent(in) :: pair2block(:)
  real(8), intent(in) :: kvector(3)
  real(8), intent(in) :: atom_sc2coord(3,*)
  integer, intent(in) :: norbs, ldz
  complex(8), intent(out) :: ztenzor(ldz,*)
  !! internal
  integer :: pair, atom_a, atom_B, atom_b_uc, n1, n2, s1, s2, f1, f2
  real(8) :: RB_m_Ra(3), phase, coeff_re, coeff_im
  !! Dimensions
  integer :: npairs
  npairs = size(pair2block)
  if(size(pair2atoms,2)/=npairs) stop 'block_red2bloch: size(pair2atoms,2)/=npairs'
  !! END of Dimensions

  ztenzor(1:norbs, 1:norbs) = 0
  do pair=1, npairs
    atom_a = pair2atoms(1, pair);
    atom_B = pair2atoms(2, pair);
    atom_b_uc = atom_sc2atom_uc(atom_B)
    n1 = size(pair2block(pair)%array,1)
    n2 = size(pair2block(pair)%array,2)
    s1 = atom2start_orb(atom_a);    f1 = s1 + n1 - 1;
    s2 = atom2start_orb(atom_b_uc); f2 = s2 + n2 - 1;
    RB_m_Ra = atom_sc2coord(1:3,atom_B) - atom_sc2coord(1:3,atom_a)
    phase = sum(RB_m_Ra*kvector)
    coeff_re = cos(phase); coeff_im = sin(phase);
    ztenzor(s1:f1,s2:f2) = ztenzor(s1:f1,s2:f2) + &
      cmplx(coeff_re*pair2block(pair)%array, coeff_im*pair2block(pair)%array,8);
  enddo

end subroutine ! block_red2bloch

!!
!! get_uc_vecs
!!
!subroutine get_uc_vecs_subr(sv, uc_vecs, iv, ilog)
!  use m_dft_hsx, only : hsx_2_uc_vectors
!  implicit none
!  !! external
!  type(system_vars_t), intent(in) :: sv
!  real(8), intent(inout) :: uc_vecs(3,3)
!  integer, intent(in) :: iv, ilog
!  !! internal  
!  call hsx_2_uc_vectors(sv%dft_hsx, uc_vecs, iv, ilog)
!end subroutine !  get_uc_vecs_subr
  
!
!
!
subroutine mu_sp2rcut_to_atom2rcut(mu_sp2rcut, atom2specie, atom2rcut)
  implicit none
  real(8), intent(in) :: mu_sp2rcut(:,:)
  integer, intent(in) :: atom2specie(:)
  real(8), intent(inout), allocatable :: atom2rcut(:)
  ! internal
  integer :: atom
  !! Dimensions
  integer :: natoms
  natoms = size(atom2specie)
  !! END of Dimensions

  if(allocated(atom2rcut) .and. natoms/=size(atom2rcut)) deallocate(atom2rcut)
  if(.not. allocated(atom2rcut)) allocate(atom2rcut(natoms))
  do atom=1,natoms; atom2rcut(atom) = maxval(mu_sp2rcut(:,atom2specie(atom))); enddo;
  
end subroutine ! mu_sp2rcut_to_atom2rcut

!
! Initialize mu_sp2start_ao array
!
subroutine init_mu_sp2start_ao( nmultipletts, mu_sp2j, mu_sp2start_ao )
  implicit none
  !! external
  integer, intent(in) :: nmultipletts(:)
  integer, intent(in) :: mu_sp2j(:,:)
  integer, intent(inout), allocatable :: mu_sp2start_ao(:,:)

  !! internal
  integer :: sp, mu, j
  !! Dimensions
  integer :: nspecies, max_pao
  max_pao = size(mu_sp2j,1)
  nspecies = size(mu_sp2j,2)
  !! END of Dimensions
  
  allocate(mu_sp2start_ao(max_pao, nspecies))
  mu_sp2start_ao = -999
  do sp=1,nspecies

    mu_sp2start_ao(1,sp)=1
    do mu=2, nmultipletts(sp)
      j = mu_sp2j(mu-1,sp)
      mu_sp2start_ao(mu,sp) = mu_sp2start_ao(mu-1,sp) + (2*j+1);
    end do

  end do

end subroutine ! init_mu_sp2start_ao

!
! Initializes xmin=0; xmax=0; ymin=0; ymax=0; zmin=0; zmax=0;
!
subroutine determine_box(atom_sc2coord, atom2sp, nmultipletts, mu_sp2rcut, &
  R_min, R_max, iv, ilog, atom_sc2atom_uc)
#define _SNAME 'determine_box'
  implicit none
  real(8), intent(in) :: atom_sc2coord(:,:)
  integer, intent(in) :: atom2sp(:)
  integer, intent(in) :: nmultipletts(:)
  real(8), intent(in) :: mu_sp2rcut(:,:)
  real(8), intent(out) :: R_min(3), R_max(3)
  integer, intent(in) :: iv, ilog
  integer, intent(in), optional :: atom_sc2atom_uc(:)
  ! internal
  integer :: atom_sc, atom_uc, sp, mu
  real(8) :: r, br(3)
  real(8) :: xmin, xmax, ymin, ymax, zmin, zmax;
  !! Dimensions
  integer :: natoms_sc
  natoms_sc = size(atom_sc2coord,2)
  !! END of Dimensions
  
  !--------------------------------------------------------------------
  ! Determine the box, which contain all the electronic density
  !--------------------------------------------------------------------
  xmin=0; xmax=0; ymin=0; ymax=0; zmin=0; zmax=0;
  do atom_sc=1,natoms_sc
    atom_uc=atom_sc
    if(present(atom_sc2atom_uc)) atom_uc=atom_sc2atom_uc(atom_sc)
    sp=atom2sp(atom_uc)
    br=atom_sc2coord(:,atom_sc)
    do mu=1,nmultipletts(sp)
      r = mu_sp2rcut(mu,sp);
      xmin=min(xmin,br(1)-r); ymin=min(ymin, br(2)-r); zmin=min(zmin,br(3)-r);
      xmax=max(xmax,br(1)+r); ymax=max(ymax, br(2)+r); zmax=max(zmax,br(3)+r);
    enddo
  enddo

  R_min = (/xmin,ymin,zmin/);
  R_max = (/xmax,ymax,zmax/);

  if(iv>1)then 
    write(ilog, *) _SNAME//': box containing the density will be';
    write(ilog, *) _SNAME//': R_min'
    write(ilog, *) R_min
    write(ilog, *) _SNAME//': R_max'
    write(ilog, *) R_max
  endif
#undef _SNAME
end subroutine !determine_box

!
! Initializes the orb2* arrays
!
subroutine init_orb2(specie2nmult, atom2specie, mu_sp2j, orb2j, orb2m, orb2atm, orb2sp, orb2mu, iv, ilog);
#define _SNAME 'init_orb2'
  implicit none
  integer, intent(in) :: iv, ilog
  integer, intent(in) :: specie2nmult(:) ! number of multipletts (pseudo atomic orbitals, or radial parts of pao) per specie
  integer, intent(in) :: atom2specie(:)  ! atom --> specie
  integer, intent(in) :: mu_sp2j(:,:) ! pao,specie-->angular momentum
  integer, intent(out) :: orb2j(:), orb2m(:), orb2atm(:), orb2sp(:), orb2mu(:)

  !! internal
  integer :: atm, sp, mu, j, m, orb
  integer :: natoms 
  natoms = size(atom2specie)

  orb2j = -999; orb2m = -999; orb2atm = -999; orb2sp = -999; orb2mu = -999;
  if(iv>1) write(ilog, *) _SNAME//': natoms ', natoms;

  orb=0
  do atm=1,natoms
    sp = atom2specie(atm)
    do mu=1, specie2nmult(sp)
      j = mu_sp2j(mu,sp)
      do m=-j,j
        orb = orb + 1
        orb2j(orb)   = j
        orb2m(orb)   = m
        orb2sp(orb)  = sp
        orb2mu(orb)  = mu
        orb2atm(orb) = atm
      end do
    end do
  end do

  if(iv>1) write(ilog, *) _SNAME//': init: orb2j orb2m orb2atm orb2sp orb2mu'
#undef _SNAME
end subroutine !init_orb2

!
!
!
subroutine init_atom2start_orb(atom2specie, specie2norb, atom2start_orb)
  implicit none
  ! external
  integer, intent(in)  :: atom2specie(:)
  integer, intent(in)  :: specie2norb(:)
  integer, intent(inout), allocatable :: atom2start_orb(:) 
  ! internal
  integer :: atom, norb
  integer :: natoms
  natoms = size(atom2specie)

  _dealloc(atom2start_orb)
  allocate(atom2start_orb(natoms))
  atom2start_orb(1) = 1
  do atom=2,natoms;
    norb = specie2norb(atom2specie(atom-1))
    atom2start_orb(atom) = atom2start_orb(atom-1)+norb;
  enddo

end subroutine ! init_atom2start_orb

!
!
!
subroutine total_electronic_charge2fermi_energy(q, DFT_E, fermi_energy, nocc, ilog)
#define _SNAME 'total_electronic_charge2fermi_energy'
  ! external
  real(8), intent(in)  :: q
  real(8), intent(in)  :: DFT_E(:,:,:)
  real(8), intent(out) :: fermi_energy
  integer, intent(out) :: nocc
  integer, intent(in)  :: ilog
  
  ! internal
  integer :: homo, lumo
  !! Dimensions
  integer :: norbs, nspin
  norbs = size(DFT_E,1)
  nspin = size(DFT_E,2)
  !! END of Dimensions

  if(nspin==1) then
    if(nint(q/2)*2/=nint(q)) then
      homo = int(q/2)+1
      lumo = min(homo+1, norbs);
      fermi_energy = DFT_E(homo,1,1);
    else
      homo = nint(q/2)
      lumo = min(homo+1, norbs);
      fermi_energy = (DFT_E(lumo,1,1)+DFT_E(homo,1,1))/2;
    endif
    nocc = homo

    if(homo*2/=nint(q)) then
      write(ilog,*) _SNAME//': warn: homo*2/=int(q)'
      write(ilog,*) _SNAME//': homo, q:', homo, q
    endif
    
  else
    homo = nint(q)
    lumo = min(homo+1, norbs)
    fermi_energy = (DFT_E(lumo,1,1)+DFT_E(homo,1,1))/2;
    write(ilog,*) _SNAME//': warn: nspin/=1 not implemented'
    stop
  endif
#undef _SNAME
end subroutine !total_electronic_charge2fermi_energy

!
! Init simple fields in prod_basis_t
!
subroutine alloc_init_rr_pp_uc_vecs_coords(sv, uc_vecs, rr, pp, coords)
  !! external
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(inout) :: uc_vecs(3,3)
  real(8), allocatable, intent(inout) :: rr(:), pp(:)
  real(8), allocatable, intent(inout) :: coords(:,:)
  !! internal
  integer :: nr, natoms
  
  uc_vecs = sv%uc%uc_vecs
  nr = get_nr(sv)
  _dealloc(rr)
  _dealloc(pp)
  allocate(rr(nr), pp(nr))
  rr = sv%rr;
  pp = sv%pp;
  natoms = get_natoms(sv)
  _dealloc(coords)
  allocate(coords(3,natoms))
  coords(1:3,1:natoms) = sv%atom_sc2coord(:,1:natoms)

end subroutine ! alloc_init_rr_pp_uc_vecs

!
!
!
subroutine  init_psi_log_rl(psi_log, rr, mu_sp2j, sp2nmult, psi_log_rl)
  implicit none
  !! external
  real(8), intent(in) :: psi_log(:,:,:)
  real(8), intent(in) :: rr(:)
  integer, intent(in) :: mu_sp2j(:,:)
  integer, intent(in) :: sp2nmult(:)
  real(8), intent(inout), allocatable :: psi_log_rl(:,:,:)
  !! internal
  integer :: nsp, nr, nmu_mx, sp, mu, l, nn(3), nmu
  nsp = size(psi_log,3)
  nmu_mx = size(psi_log,2)
  nr = size(psi_log,1)
  nn = [nr, nmu_mx, nsp]
  if(any(nn<1)) _die('any(nn<1)')
  
  _dealloc(psi_log_rl)
  allocate(psi_log_rl(nr,nmu_mx,nsp))
  do sp=1,nsp
    nmu = sp2nmult(sp)
    do mu=1,nmu
      l = mu_sp2j(mu,sp)
      psi_log_rl(:, mu, sp) = psi_log(:, mu, sp) / rr(:)**l
    enddo ! mu  
  enddo ! sp 
  
end subroutine !  init_psi_log_rl 

!
!
!
function are_ok_dims_fini48(X,E,fname,fline) result(ok)
  implicit none
  !! external
  real(4), intent(in) :: X(:,:,:,:,:)
  real(8), intent(in) :: E(:,:,:)
  character(*), intent(in) :: fname
  integer, intent(in) :: fline
  logical :: ok
  ! internal
  integer :: nnx(5), nne(3)
  ok = .false.
  nnx = ubound(X)
  nne = ubound(E)
  if(any(nnx<1)) call die('!nnx<1 '//trim(fname), fline)
  if(any(nne<1)) call die('!nne<1 '//trim(fname), fline)
  if(nnx(1)/=1) call die('!nnx/=1 '//trim(fname), fline)
  if(nnx(5)/=1) call die('!nnx(5)/=1 '//trim(fname), fline)
  if(nnx(4)>2) call die('!nnx(4)>2 '//trim(fname), fline)
  if(nne(3)/=1) call die('!nne(3)/=1 '//trim(fname), fline)
  if(nnx(2)/=nnx(3)) call die('!nnx(2)/=nnx(3) '//trim(fname), fline)
  if(nnx(2)/=nne(1)) call die('!nnx(2)/=nne(1) '//trim(fname), fline)
  if(nnx(4)/=nne(2)) call die('!nnx(4)/=nne(2) '//trim(fname), fline)
  ok = .true.
end function ! are_ok_dims_fini 

!
!
!
function are_ok_dims_fini88(X,E,fname,fline) result(ok)
  implicit none
  !! external
  real(8), intent(in) :: X(:,:,:,:,:)
  real(8), intent(in) :: E(:,:,:)
  character(*), intent(in) :: fname
  integer, intent(in) :: fline
  logical :: ok
  ! internal
  integer :: nnx(5), nne(3)
  ok = .false.
  nnx = ubound(X)
  nne = ubound(E)
  if(any(nnx<1)) call die('!nnx<1 '//trim(fname), fline)
  if(any(nne<1)) call die('!nne<1 '//trim(fname), fline)
  if(nnx(1)/=1) call die('!nnx/=1 '//trim(fname), fline)
  if(nnx(5)/=1) call die('!nnx(5)/=1 '//trim(fname), fline)
  if(nnx(4)>2) call die('!nnx(4)>2 '//trim(fname), fline)
  if(nne(3)/=1) call die('!nne(3)/=1 '//trim(fname), fline)
  if(nnx(2)/=nnx(3)) call die('!nnx(2)/=nnx(3) '//trim(fname), fline)
  if(nnx(2)/=nne(1)) call die('!nnx(2)/=nne(1) '//trim(fname), fline)
  if(nnx(4)/=nne(2)) call die('!nnx(4)/=nne(2) '//trim(fname), fline)
  ok = .true.
end function ! are_ok_dims_fini 

end module !m_system_vars
