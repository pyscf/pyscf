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

module m_orb_rspace_type

#include "m_define_macro.F90"
  use m_die, only : die
  use m_system_vars, only : system_vars_t
    
  implicit none

  private die

  type orb_rspace_aux_t
    type(system_vars_t), pointer :: sv => null()
    integer :: natoms = -1
    integer :: norbs = -1
    integer :: nr = -1
    integer :: jmx = -1
    real(8) :: dr_jt = -999
    real(8) :: rr1 = -999
    real(8) :: rr2 = -999
    real(8) :: rho_min_jt = -999
    real(8) :: one_over_dr_jt = -999
    integer, pointer :: mu_sp2start_ao(:,:) => null()
    integer, pointer :: mu_sp2j(:,:) => null()
    integer, pointer :: atom2sp(:) => null()
    integer, pointer :: sp2nmult(:) => null()
    integer, pointer :: atom2sfo(:,:) => null()
    integer, allocatable :: sp2jmx(:)
    real(8), pointer :: atom2coord(:,:) => null()    
    real(8), pointer :: mu_sp2rcut(:,:) => null()
    real(8), allocatable :: sp2rcut(:)
    
    real(8), pointer :: psi_log(:,:,:) => null() ! !robust
    real(8), allocatable :: psi_log_rl(:,:,:) ! ... robust, but don't forget to multiply with r^l
  end type !orb_rspace_aux_t

contains

!!
!! The values of all orbitals in the unit cell for a cartesian coordinate
!! Storage of orbitals is sparse, i.e. only non-zero orbitals are going to be stored
!!
subroutine comp_orb_rspace_sprs(a, coord, inz2vo, slm)
  use m_interpolation, only : get_fval
  implicit none
  type(orb_rspace_aux_t), intent(in) :: a !auxiliary
  real(8), intent(in)  :: coord(3)
  real(8), intent(inout), allocatable :: inz2vo(:,:) ! array of non-zero orbital values
  real(8), intent(inout) :: slm(0:) ! auxiliary: must be thread-private!
  !! internal
  integer :: atm, spa, mu, nmu, j, jmjp1, so, k0, m, nnzo
  real(8) :: rho, fr_val, coeff(-2:3)
  logical :: lcycle
  
  nnzo = 0
  !! Calculate values of localized orbitals
  do atm=1,a%natoms;
    call get_adep_params(a,coord,atm, spa,rho,k0,coeff,slm,so,nmu,lcycle)
    if(lcycle) cycle
    do mu=1,nmu
      if(rho>a%mu_sp2rcut(mu,spa)) cycle
      fr_val = sum(coeff*a%psi_log_rl(k0-2:k0+3,mu,spa))
      j = a%mu_sp2j(mu,spa);
      jmjp1=j*(j+1)
      if(j==0) then
        inz2vo(nnzo+1:nnzo+2*j+1,1) = fr_val*slm(jmjp1-j:jmjp1+j)
      else
        inz2vo(nnzo+1:nnzo+2*j+1,1) = rho**j * fr_val*slm(jmjp1-j:jmjp1+j)  
      endif
        
      do m =-j,j
        nnzo = nnzo + 1
        inz2vo(nnzo,2) = a%mu_sp2start_ao(mu,spa)+j+m+so-1
      end do ! m      
    enddo ! mu
  enddo ! atm
  !! END of Calculate values of localized orbitals
  inz2vo(0,1:2) = nnzo
  
end subroutine !calc_orb_rspace_sprs

!
!
!
subroutine init_orb_rspace_aux(sv, a, ul_slm)
  use m_system_vars, only : system_vars_t, get_rr_ptr, get_uc_ptr, get_norbs
  use m_system_vars, only : get_atom2coord_ptr, get_psi_log_ptr, get_sp2rcut
  use m_system_vars, only : get_nr, get_jmx, get_natoms, init_psi_log_rl
  use m_uc_skeleton, only : uc_skeleton_t, get_mu_sp2start_ao_ptr, get_mu_sp2j_ptr
  use m_uc_skeleton, only : get_atom2sp_ptr, get_mu_sp2rcut_ptr, get_atom2sfo_ptr
  use m_uc_skeleton, only : get_sp2nmult_ptr, get_sp2jmx
  use m_harmonics, only : get_lu_slm
  use m_interpolation, only : grid_2_dr_and_rho_min
  implicit none
  type(system_vars_t), intent(in), target :: sv
  type(orb_rspace_aux_t), intent(inout) :: a
  integer, intent(out) :: ul_slm
  !! internal
  integer :: lub(2)
  real(8), pointer :: rr(:)
  type(uc_skeleton_t), pointer :: uc
  
  a%sv => sv
  a%norbs  = get_norbs(sv)
  a%natoms = get_natoms(sv)
  a%nr = get_nr(sv)
  a%jmx = get_jmx(sv)
  rr => get_rr_ptr(sv)
  
  call grid_2_dr_and_rho_min(a%nr, rr, a%dr_jt, a%rho_min_jt)
!  write(6,*) __FILE__, __LINE__
!  write(6,*) a%nr, a%dr_jt, a%rho_min_jt
  
  a%one_over_dr_jt = 1D0/a%dr_jt
  lub = get_lu_slm(a%jmx)
  ul_slm = lub(2)
  
  uc=>get_uc_ptr(sv)
  a%mu_sp2start_ao =>get_mu_sp2start_ao_ptr(uc)
  a%mu_sp2j => get_mu_sp2j_ptr(uc)
  a%atom2coord => get_atom2coord_ptr(sv)
  a%atom2sp => get_atom2sp_ptr(uc)
  a%mu_sp2rcut => get_mu_sp2rcut_ptr(uc)
  a%sp2nmult => get_sp2nmult_ptr(uc)
  a%atom2sfo => get_atom2sfo_ptr(uc)
  call get_sp2rcut(sv, a%sp2rcut)
  call get_sp2jmx(uc, a%sp2jmx)

  a%psi_log => get_psi_log_ptr(sv)
  a%rr1 = rr(1)
  a%rr2 = rr(2)
  call init_psi_log_rl(a%psi_log, rr, a%mu_sp2j, a%sp2nmult, a%psi_log_rl)
  
end subroutine ! init_orb_rspace_aux  

!
!
!
integer function get_ulimit_slm(a)
  use m_harmonics, only : get_lu_slm
  implicit none
  !! external
  type(orb_rspace_aux_t), intent(in) :: a
  !! internal
  integer :: lub(2)
  
  if(a%jmx<0) _die('%jmx<0')
  lub = get_lu_slm(a%jmx)
  get_ulimit_slm = lub(2)
  
end function !  get_ulimit_slm 
  

!
!
!
subroutine alloc_orb_rspace_rsh_vo(a, inz2vo, slm)
  use m_harmonics, only : get_lu_slm
  implicit none
  !! external
  type(orb_rspace_aux_t), intent(in) :: a
  real(8), allocatable, intent(inout) :: inz2vo(:,:)
  real(8), allocatable, intent(inout), optional :: slm(:)  
  !! internal
  integer :: lub(2), jmx, norbs
  
  norbs = get_norbs(a)
  _dealloc(inz2vo)
  if(.not. allocated(inz2vo)) allocate(inz2vo(0:norbs,2))

  if(present(slm)) then 
    jmx = get_jmx(a)
    lub = get_lu_slm(jmx)
    _dealloc(slm)
    if(.not.allocated(slm)) allocate(slm(lub(1):lub(2)))
  endif  
  

end subroutine ! alloc_orb_rspace_rsh_vo  

!
!
!
integer function get_jmx(a)
  use m_system_vars, only : get_jmx_sv=>get_jmx
  implicit none
  type(orb_rspace_aux_t), intent(in) :: a
  if(.not. associated(a%sv)) _die('!%sv')
  get_jmx = get_jmx_sv(a%sv)
end function !  get_jmx 
  
!
!
!
integer function get_norbs(a)
  use m_system_vars, only : get_norbs_sv=>get_norbs
  implicit none
  type(orb_rspace_aux_t), intent(in) :: a
  if(.not. associated(a%sv)) _die('!%sv')
  get_norbs = get_norbs_sv(a%sv)
end function !  get_norbs

!
! The values of all orbitals in the unit cell for a cartesian coordinate
!
subroutine comp_orb_rspace(a, coord, o2v, slm)
  use m_interpolation, only : get_fval
  implicit none
  type(orb_rspace_aux_t), intent(in) :: a !auxiliary
  real(8), intent(in)  :: coord(3)
  real(8), intent(out) :: o2v(:)  !array of orbital values
  real(8), intent(inout) :: slm(0:) ! auxiliary: must be thread private!
  !! internal
  integer :: atm, spa, s, mu, nmu, j, jmjp1, so,k0
  real(8) :: rho, fr_val, coeff(-2:3)
  logical :: lcycle
  
  o2v = 0
  !! Calculate values of localized orbitals
  do atm=1,a%natoms;
    call get_adep_params(a,coord,atm, spa,rho,k0,coeff,slm,so,nmu,lcycle)
    if(lcycle) cycle
    do mu=1,nmu
      if(rho>a%mu_sp2rcut(mu,spa)) cycle
      fr_val = sum(coeff*a%psi_log_rl(k0-2:k0+3,mu,spa))
      j = a%mu_sp2j(mu,spa);
      jmjp1=j*(j+1)
      s = so+a%mu_sp2start_ao(mu,spa)-1
      if(j==0) then
        o2v(s:s+2*j) = fr_val * slm(jmjp1-j:jmjp1+j)
      else
        o2v(s:s+2*j) = rho**j * fr_val * slm(jmjp1-j:jmjp1+j)  
      endif  
    enddo ! mu
  enddo ! atm
  !! END of Calculate values of localized orbitals
end subroutine !calc_orb_rspace

!
!
!
subroutine get_adep_params(a, coord, atm, spa, rho, k, coeff, slm, so, nmu, lcycle)
  use m_harmonics, only : rsphar
  implicit none
  !! external
  type(orb_rspace_aux_t), intent(in) :: a ! auxiliary
  real(8), intent(in) :: coord(3)
  integer, intent(in) :: atm
  integer, intent(inout) :: spa, so, nmu, k
  real(8), intent(inout) :: coeff(-2:3)
  logical, intent(inout) :: lcycle
  real(8), intent(inout) :: rho
  real(8), intent(inout) :: slm(0:)
  !! internal
  real(8) :: br0(3)
  
  spa  = a%atom2sp(atm)
  br0  = coord - a%atom2coord(1:3, atm)  !!print *, 'br01',br,br01,br1;
  rho = sqrt(sum(br0**2))
  lcycle = .false.
  if (rho>a%sp2rcut(spa)) then; lcycle=.true.; return; endif
  call rsphar(br0, slm, a%sp2jmx(spa))
  so = a%atom2sfo(1,atm)
  nmu = a%sp2nmult(spa)
  call comp_coeff(a, rho, coeff, k)
  
end subroutine ! get_adep_params

!
!
!
subroutine comp_coeff(a, rho, coeff, k)
  implicit none
  ! external
  type(orb_rspace_aux_t), intent(in) :: a
  real(8), intent(in)  :: rho
  real(8), intent(out) :: coeff(-2:3)
  integer, intent(out) :: k
  ! internal
  real(8), parameter :: one120 = 1.0D0/120D0, one12=1.0D0/12D0, one24 = 1.0D0/24D0;
  real(8) :: logr, dy, dy2, dym3;

  if(rho<a%rr1) then
    k = 3
    coeff = 0
    coeff(-2) = (rho - a%rr2) / (a%rr1 - a%rr2)
    coeff(-1) = (rho - a%rr1) / (a%rr2 - a%rr1)
    return
  else 
    logr=log(rho)
    k=int((logr-a%rho_min_jt)*a%one_over_dr_jt+1)
    k=max(k,3)
    k=min(k,a%nr-3)
  endif
    
  dy= (logr-a%rho_min_jt-(k-1)*a%dr_jt)*a%one_over_dr_jt;
  dy2  = dy*dy;
  dym3 = (dy-3.0d0);
  coeff(-2) = -dy*(dy2-1.0d0)*(dy-2.0d0) *dym3*one120;
  coeff(-1) =  dy*(dy-1.0d0) *(dy2-4.0d0)*dym3*one24;
  coeff(0)  =-(dy2-1.0d0)    *(dy2-4.0d0)*dym3*one12;
  coeff(1)  =  dy*(dy+1.0d0)*(dy2-4.0d0)*dym3*one12;
  coeff(2)  = -dy*(dy2-1.0d0)*(dy+2.0d0)*dym3*one24;
  coeff(3)  = dy*(dy2-1.0d0)*(dy2-4.0d0)*one120;

end subroutine !comp_coeff

!
!
!
real(8) function get_rho_min_jt(orb_a)
  implicit none
  type(orb_rspace_aux_t), intent(in) :: orb_a
  if(orb_a%rho_min_jt==-999) _die('%rho_min_jt==-999')
  if(orb_a%rho_min_jt==0) _die('%rho_min_jt==0')
  get_rho_min_jt = orb_a%rho_min_jt
end function ! get_rho_min_jt

!
!
!
real(8) function get_dr_jt(orb_a)
  implicit none
  type(orb_rspace_aux_t), intent(in) :: orb_a
  if(orb_a%dr_jt<0) _die('%dr_jt<0?')
  get_dr_jt = orb_a%dr_jt
end function ! get_dr_jt

!
!
!
function get_mu_sp2start_ao_ptr(a) result(ptr)
  implicit none
  type(orb_rspace_aux_t), intent(in), target :: a
  integer, pointer :: ptr(:,:)
  
  if(.not. associated(a%mu_sp2start_ao)) _die('!mu_sp2start_ao')
  ptr=>a%mu_sp2start_ao
end function ! get_mu_sp2start_ao_ptr
  
end module !m_orb_rspace_type
