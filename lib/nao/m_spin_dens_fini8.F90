module m_spin_dens_fini8

#include "m_define_macro.F90"

  implicit none
  !! For charge density
  real(8), allocatable, private :: dm_f2(:), inz2vo(:,:)
  real(8), allocatable, private :: slm_loc(:)
  !$OMP THREADPRIVATE(dm_f2,inz2vo,slm_loc)

  type spin_dens_aux_t
    real(8), pointer :: coord(:,:) =>null()
    real(8), pointer :: DM(:,:,:) =>null()
    integer, pointer :: atom2sp(:) =>null()
    integer, pointer :: sp2nmult(:) =>null()
    integer, pointer :: mu_sp2j(:,:) =>null()
    integer, pointer :: mu_sp2start_ao(:,:) =>null()
    integer, allocatable :: atom2start_orb(:)
    integer, allocatable :: sp2jmx(:)
    real(8), allocatable :: atom2rcut(:)
    real(8), pointer :: mu_sp2rcut(:,:) =>null()
    real(8), pointer :: psi_log(:,:,:) =>null()
    integer :: natoms = -999
    integer :: nspin =-999
    integer :: nr =-999
    integer :: jmx =-999
    real(8) :: dr_jt=-999, one_over_dr_jt=-999, rho_min_jt=-999
    real(8) :: one120 = 1.0D0/120D0, one12=1.0D0/12D0, one24 = 1.0D0/24D0;

  end type !spin_dens_aux_t  

contains

!
!
!
subroutine dealloc(sda)
  implicit none
  type(spin_dens_aux_t), intent(inout) :: sda

  sda%coord =>null()
  sda%DM =>null()
  sda%atom2sp =>null()
  sda%sp2nmult =>null()
  sda%mu_sp2j =>null()
  sda%mu_sp2start_ao =>null()
  _dealloc(sda%atom2start_orb)
  _dealloc(sda%sp2jmx)
  _dealloc(sda%atom2rcut)
  sda%mu_sp2rcut =>null()
  sda%psi_log =>null()
  sda%natoms = -999
  sda%nspin =-999
  sda%nr =-999
  sda%jmx =-999
  sda%dr_jt=-999
  sda%one_over_dr_jt=-999
  sda%rho_min_jt=-999
  sda%one120 = 1.0D0/120D0
  sda%one12=1.0D0/12D0
  sda%one24 = 1.0D0/24D0  
  
end subroutine ! dealloc
  

!!
!!
!!
subroutine init_spin_dens(sv, DM, sda)
  use m_system_vars, only : system_vars_t, get_norbs, get_jmx, get_rr_ptr, get_nr
  use m_system_vars, only : get_atom2coord_ptr, get_atom2sp_ptr, get_sp2nmult_ptr
  use m_system_vars, only : get_natoms, get_psi_log_ptr, get_mu_sp2rcut_ptr, get_nspin
  use m_system_vars, only : get_mu_sp2j_ptr, get_uc_ptr, get_atom2rcut
  use m_uc_skeleton, only : get_mu_sp2start_ao_ptr, uc_skeleton_t, get_sp2jmx
  use m_uc_skeleton, only : get_atom2start_orb
  use m_interpolation, only : grid_2_dr_and_rho_min
  implicit none
  type(system_vars_t), intent(inout), target :: sv
  real(8), intent(in), target :: DM(:,:,:)
  type(spin_dens_aux_t), intent(inout) :: sda
  ! internal
  integer :: norbs, jdim
  real(8), pointer :: rr(:)
  type(uc_skeleton_t), pointer :: uc

  call dealloc(sda)
  
  uc=>get_uc_ptr(sv) 
  norbs = get_norbs(sv)
  sda%nr = get_nr(sv)
  sda%nspin = get_nspin(sv)
  rr => get_rr_ptr(sv)
  call grid_2_dr_and_rho_min(sda%nr, rr, sda%dr_jt, sda%rho_min_jt)
  sda%one_over_dr_jt = 1D0/sda%dr_jt

  sda%coord => get_atom2coord_ptr(sv)
  sda%DM => DM
  sda%atom2sp => get_atom2sp_ptr(sv)
  sda%sp2nmult => get_sp2nmult_ptr(sv)
  sda%mu_sp2j => get_mu_sp2j_ptr(sv)
  sda%mu_sp2start_ao => get_mu_sp2start_ao_ptr(uc)
  call get_atom2start_orb(uc, sda%atom2start_orb)
  call get_sp2jmx(uc, sda%sp2jmx)
  call get_atom2rcut(sv, sda%atom2rcut)
  sda%mu_sp2rcut=>get_mu_sp2rcut_ptr(sv)
  sda%psi_log=>get_psi_log_ptr(sv)
  sda%natoms = get_natoms(sv)
  sda%jmx = get_jmx(sv)
    
  jdim = (2*sda%jmx+1)**2+1
  !$OMP PARALLEL
_dealloc(inz2vo)
  if(.not. allocated(inz2vo)) allocate(inz2vo(0:norbs,2))

_dealloc_u(dm_f2, norbs)
  if(.not. allocated(dm_f2)) allocate(dm_f2(norbs))

_dealloc_u(slm_loc,jdim)
  if(.not. allocated(slm_loc)) allocate(slm_loc(0:jdim))
  !$OMP END PARALLEL
end subroutine !init_spin_dens

!
!
!
subroutine spin_dens(sda, br, dens)
  use m_harmonics, only : rsphar

  implicit none
  type(spin_dens_aux_t), intent(in) :: sda
  real(8), intent(in) :: br(3)
  real(8), intent(out) :: dens(:)

  ! Interne Variable:
  real(8) :: br0(3), rho
  real(8) :: fr_val
  integer  :: atm, spa, mu, nnzo, j, m, jjp1,k,so,start_ao
  real(8)  :: coeff(-2:3)

  nnzo  = 0; !! index to write non-zero orbitals to f(:)
           !! latter -- the maximal index in f(:) to treat

  !! values of localized orbitals
  do atm=1,sda%natoms;
    spa  = sda%atom2sp(atm); 
    br0  = br - sda%coord(:,atm);  !!print *, 'br01',br,br01,br1;
    rho = sqrt(sum(br0**2));
    if (rho>sda%atom2rcut(atm)) cycle
    call comp_coeff(sda, coeff, k, rho)
    call rsphar(br0, slm_loc(0:), sda%sp2jmx(spa));
    so = sda%atom2start_orb(atm)
    
    do mu=1,sda%sp2nmult(spa); 
      if(rho>sda%mu_sp2rcut(mu,spa)) cycle;
      start_ao = sda%mu_sp2start_ao(mu, spa)
      fr_val = sum(coeff*sda%psi_log(k-2:k+3,mu,spa));
      j = sda%mu_sp2j(mu,spa);
      jjp1 = j*(j+1);
      inz2vo(nnzo+1:nnzo+2*j+1,1)= fr_val*slm_loc(jjp1-j:jjp1+j)
      do m =-j,j;
        nnzo = nnzo + 1
        inz2vo(nnzo,2) = start_ao + j + m + so - 1
      end do ! m
    enddo; ! mu
  enddo; ! atom
  inz2vo(0,1:2) = nnzo
  
  call spin_dens_orb_sprs(inz2vo, sda%nspin, sda%DM, dens, dm_f2)
  
end subroutine ! spin_dens

!
!
!
subroutine spin_dens_orb_sprs(inz2vo, nspin,DM, dens, dm_f2)
  use m_sparse_vec, only : get_nnz
  implicit none
  !! external
  real(8), intent(in), allocatable  :: inz2vo(:,:)
  integer, intent(in)  :: nspin
  real(8), intent(in)  :: DM(:,:,:)
  real(8), intent(out) :: dens(:)
  real(8), intent(inout) :: dm_f2(:) ! auxiliary
  !! internal
  integer :: ispin, inzo1, inzo2, nnzo, o1, o2
  nnzo = get_nnz(inz2vo)
  
  if(nnzo==0) then; dens = 0; return; endif;

  !!-------------------------------------------------------------------
  !! f(:) contains the values of atomic orbitals, which contribute to the density
  !! (at a given point). 
  !!-------------------------------------------------------------------
  dens = 0
  do ispin=1, nspin
    do inzo2=1,nnzo
      o2 = int(inz2vo(inzo2,2))
      do inzo1=1,nnzo
        o1 = int(inz2vo(inzo1,2))
        dm_f2(inzo1) = DM(o1,o2,ispin)*inz2vo(inzo2,1)
      enddo
        
      dens(ispin) = dens(ispin)+sum( dm_f2(1:nnzo) * inz2vo(1:nnzo,1) )
      
    end do ! inzo2
  enddo ! ispin

end subroutine ! spin_dens

!
!
!
subroutine comp_coeff(sda, coeff, k, rho)
  implicit none
  type(spin_dens_aux_t), intent(in) :: sda
  real(8), intent(out) :: coeff(-2:3)
  integer, intent(out) :: k
  real(8), intent(in)  :: rho

  ! interpolation
  real(8) :: logr, dy, dy2, dym3;

  logr= log(rho);
  k=int((logr-sda%rho_min_jt)*sda%one_over_dr_jt+1)
  k=max(k,3)
  k=min(k,sda%nr-3);
  dy= (logr-sda%rho_min_jt-(k-1)*sda%dr_jt)*sda%one_over_dr_jt;
  dy2  = dy*dy;
  dym3 = (dy-3.0d0);
  coeff(-2) = -dy*(dy2-1.0d0)*(dy-2.0d0) *dym3*sda%one120;
  coeff(-1) =  dy*(dy-1.0d0) *(dy2-4.0d0)*dym3*sda%one24;
  coeff(0)  =-(dy2-1.0d0)    *(dy2-4.0d0)*dym3*sda%one12;
  coeff(1)  =  dy*(dy+1.0d0)*(dy2-4.0d0)*dym3*sda%one12;
  coeff(2)  = -dy*(dy2-1.0d0)*(dy+2.0d0)*dym3*sda%one24;
  coeff(3)  = dy*(dy2-1.0d0)*(dy2-4.0d0)*sda%one120;

end subroutine !comp_coeff


end module !m_spin_dens_fini8
