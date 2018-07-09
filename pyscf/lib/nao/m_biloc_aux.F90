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

module m_biloc_aux

  use m_precision, only : blas_int
#include "m_define_macro.F90" 
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use m_prod_basis_param, only : prod_basis_param_t
  use m_orb_rspace_type, only : orb_rspace_aux_t
  use m_parallel, only : para_t
  use m_sph_bes_trans, only : Talman_plan_t
  use m_interp, only : interp_t
  use m_log, only : log_memory_note
  
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  type biloc_aux_t
    type(system_vars_t), pointer :: sv =>null()
    type(prod_basis_param_t), pointer :: pb_p =>null()
    type(para_t), pointer :: para =>null()
    type(orb_rspace_aux_t), pointer :: orb_a =>null()
    type(Talman_plan_t) :: Talman_plan
    type(interp_t)  :: interp_log
    real(8), pointer :: rr(:) =>null()
    real(8), pointer :: pp(:) =>null()
    integer, pointer :: mu_sp2j(:,:) =>null()
    integer, pointer :: mu_sp2start_ao(:,:) =>null()
    integer, pointer :: sp2nmult(:) =>null()
    real(8), pointer :: psi_log(:,:,:) =>null()
    real(8), pointer :: psi_log_rl(:,:,:) =>null()
    real(8), pointer :: mu_sp2rcut(:,:) =>null()
    
    integer :: nr = -999
    integer :: nf_max = -999
    integer :: norbs_max = -999
    integer :: nterm_max = -999
    integer(blas_int) :: lwork =-999
    integer :: jmx = -999
    integer :: jcutoff = -999
    integer :: ord = -999
    integer :: jmax_pl = -999
    
    complex(8), allocatable :: ylm_cr(:)
    complex(8), allocatable :: c2r(:,:)
    complex(8), allocatable :: hc_c2r(:,:)
    real(8), allocatable :: mu_sp2inv_wght(:,:)
    real(8), allocatable :: dkappa_pp(:)
    real(8), allocatable :: plval(:,:)
    real(8), allocatable :: xgla(:)
    real(8), allocatable :: wgla(:)
    real(8), allocatable :: sqrt_wgla(:)
    integer, allocatable :: sp2norbs(:)
    
  end type ! biloc_aux_t

  contains

!
!
!
subroutine init_biloc_aux(sv, pb_p, para, orb_a, a)
  use m_system_vars, only : get_norbs_max, get_jmx
  use m_system_vars, only : get_psi_log_ptr, get_sp2nmult_ptr, get_mu_sp2j_ptr, get_nr
  use m_system_vars, only : get_rr_ptr, get_pp_ptr, get_sp2norbs
  use m_system_vars, only : get_mu_sp2rcut_ptr, get_psi_log_rl_ptr
  use m_orb_rspace_type, only : get_mu_sp2start_ao_ptr
  use m_prod_basis_param, only : get_jcutoff, get_metric_type, get_bilocal_center, get_GL_ord_bilocal
  use m_prod_basis_param, only : get_bilocal_center_coeff, get_bilocal_center_pow
  use m_sph_bes_trans, only : sbt_plan
  use m_prod_talman, only : csphar_talman
  use m_harmonics, only : init_c2r_hc_c2r
  use m_interp, only : init_interp
  use m_all_legendre_poly, only : all_legendre_poly
  use m_numint, only : gl_knts, gl_wgts
  
  implicit none
  !! external
  type(system_vars_t), intent(in), target :: sv
  type(prod_basis_param_t), intent(in), target :: pb_p
  type(para_t), intent(in), target :: para
  type(orb_rspace_aux_t), intent(in), target :: orb_a
  type(biloc_aux_t), intent(inout) :: a
  !! internal
  integer :: jmax, metric_type, nn(2)
  real(8), parameter :: unitvec_z(3) = [0D0,0D0,1D0]
  character(100) :: ccenter, ccoeff
  real(8) :: pow, spos
  
  a%sv => sv
  a%pb_p => pb_p
  a%para => para
  a%orb_a => orb_a
  a%mu_sp2start_ao => get_mu_sp2start_ao_ptr(orb_a)
  a%mu_sp2j=> get_mu_sp2j_ptr(sv)
  a%sp2nmult => get_sp2nmult_ptr(sv)
  a%psi_log => get_psi_log_ptr(sv)
  a%psi_log_rl => get_psi_log_rl_ptr(sv)
  a%mu_sp2rcut => get_mu_sp2rcut_ptr(sv)
  
          
  nn = ubound(a%mu_sp2j)
_dealloc_u(a%mu_sp2inv_wght, nn)  
  if(.not. allocated(a%mu_sp2inv_wght)) allocate(a%mu_sp2inv_wght(nn(1),nn(2)))
  ccenter = get_bilocal_center(pb_p)
  if(ccenter=="POW&COEFF" .or. ccenter=="MAXLOC") then
    pow = get_bilocal_center_pow(pb_p)
    ccoeff = get_bilocal_center_coeff(pb_p)
    call comp_mu_sp2radii_pow(sv, pow, ccoeff, ccenter, a%mu_sp2inv_wght)
  else if (ccenter=="LIBERI") then
    call comp_mu_sp2inv_wght_Toyoda_Ozaki(sv, a%mu_sp2inv_wght)
  else if (ccenter=="MAXLOC_BILOCAL") then
    !! everything will be done later in get_inv_wghts
    continue  
  else  
    write(6,*) ccenter
    _die('unknown bilocal_center')
  endif  
      
  
  a%norbs_max = get_norbs_max(sv)
  a%nf_max = get_nf_max(sv)
  a%lwork = 5*a%nf_max**2;
  
  a%jcutoff = get_jcutoff(pb_p)
  a%jmx = get_jmx(sv)
  a%nterm_max = get_nterm_max(a%jmx, a%jcutoff)
  jmax = a%jcutoff+a%jmx*2
_dealloc_u(a%ylm_cr, (jmax+1)**2)
  if(.not. allocated(a%ylm_cr)) allocate(a%ylm_cr(0:(jmax+1)**2));
  call csphar_talman(unitvec_z, a%ylm_cr(0:), jmax);
  call init_c2r_hc_c2r(max(a%jcutoff,2*a%jmx), a%c2r, a%hc_c2r);
  a%nr = get_nr(sv)
  a%rr => get_rr_ptr(sv)
  a%pp => get_pp_ptr(sv)
  call sbt_plan(a%Talman_plan, a%nr, max(a%jcutoff,2*a%jmx+1), a%rr, a%pp, .true.)
  call init_interp(a%rr, a%interp_log)

  _dealloc(a%xgla)
  _dealloc(a%wgla)
  _dealloc(a%sqrt_wgla)
  a%ord = get_GL_ord_bilocal(pb_p)
  allocate(a%xgla(a%ord))
  allocate(a%wgla(a%ord))
  allocate(a%sqrt_wgla(a%ord))
  call GL_knts(a%xgla, -1.0D0, 1.0D0, a%ord, 1)
  call GL_wgts(a%wgla, -1.0D0, 1.0D0, a%ord, 1)
  spos = sum(abs(a%wgla - abs(a%wgla)))
  if(spos>1d-14) _die('!spos>1d-14')
  a%sqrt_wgla = sqrt(a%wgla)
  !write(6,*) __FILE__, __LINE__, sum(abs(a%wgla - abs(a%wgla)))
  a%jmax_pl = 2*(a%jcutoff+a%jmx)
  _dealloc(a%plval)
  allocate(a%plval(a%ord,0:2*(a%jcutoff+a%jmx)))
  call all_legendre_poly(a%xgla,a%ord,a%plval,2*(a%jcutoff+a%jmx))
  
  metric_type = get_metric_type(pb_p)
_dealloc_u(a%dkappa_pp, a%nr)
  if(.not. allocated(a%dkappa_pp)) allocate(a%dkappa_pp(a%nr))
  if(metric_type==1) then
     !! Cartesian metric
     a%dkappa_pp = log(a%pp(a%nr)/a%pp(1))/(a%nr-1)*a%pp**3;
  else if (metric_type==2) then
     !! Coulomb metric
     a%dkappa_pp = log(a%pp(a%nr)/a%pp(1))/(a%nr-1)*a%pp;
  else
     !! Other are not implemented
     write(0,*) 'err: make_bilocal_vertex: metric_type', metric_type;
     _die('!metric_type ?')
  endif

  call get_sp2norbs(sv, a%sp2norbs)

end subroutine ! init_biloc_aux  


!
! Calculate      nf_max
!
function get_nf_max(sv) result(nf_max)
  use m_system_vars, only : system_vars_t, get_jmx, get_natoms
  use m_uc_skeleton, only : get_j, get_nmult, get_sp
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer :: nf_max
  ! internal
  integer :: atm1, atm2, sp1, sp2, mu1, mu2, nmult1, nmult2, m1, m2, j1, j2
  integer, allocatable :: bilocal_levels(:);
  ! Dimensions
  integer :: natoms, jmx
  jmx = get_jmx(sv)
  natoms = get_natoms(sv)
  ! END of dimensions

  allocate(bilocal_levels(-jmx*2:jmx*2))
  nf_max = 0;
  do atm2=1,natoms
    sp2=get_sp(sv%uc, atm2);
    nmult2=get_nmult(sv%uc, sp2);

    do atm1=1, atm2;
      bilocal_levels = 0;
      sp1=get_sp(sv%uc, atm1);
      nmult1=get_nmult(sv%uc, sp1);

      do mu1=1,nmult1;
        j1=get_j(sv%uc, mu1,sp1);
        do m1=-j1,j1       ! Atom 1
          do mu2=1,nmult2;
            j2=get_j(sv%uc, mu2,sp2);
            do m2=-j2,j2   ! Atom 2
              bilocal_levels(m1 + m2)=bilocal_levels(m1 + m2)+1;
            enddo; ! m2
          enddo; ! mu2
        enddo; ! m1
      enddo; ! mu1
    nf_max = max(maxval(bilocal_levels),nf_max);
    enddo; ! atm1
  enddo; ! atm2

end function !get_nf_max

!
! allocate--init of vertex variable
!
function get_nterm_max(jmx, jcutoff) result(nterm_max)
  implicit none
  ! external
  integer, intent(in) :: jmx
  integer, intent(in) :: jcutoff
  integer :: nterm_max

  ! internal
  integer :: ij,clbd,lbd

  !! Initializations for product reduction in Talman's way
  nterm_max=0
  do ij=0,jmx*2
    do clbd=0,jcutoff
      do lbd=abs(clbd-ij),clbd+ij
        nterm_max = nterm_max + 1
      enddo
    enddo
  enddo

end function !get_nterm_max


!
! Computes a mean radii for each multiplett
!
subroutine comp_mu_sp2radii_pow(sv, pow, ccoeff, ccenter, mu_sp2radii)
  use m_functs_l_mult_type, only : get_radii_sqr
  use m_system_vars, only : get_rr_ptr, get_sp2nmult_ptr, get_nspecies
  use m_system_vars, only : get_mu_sp2j_ptr, get_psi_log_ptr
  use m_interpolation, only : get_dr_jt
  !!  external
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(in) :: pow
  character(*), intent(in) :: ccoeff, ccenter
  real(8), intent(out) :: mu_sp2radii(:,:)
  !! internal
  real(8), pointer :: psi_log(:,:,:), rr(:)
  integer, pointer :: sp2nmult(:), mu_sp2j(:,:)
  integer :: sp, mu, j
  real(8) :: dlambda, r2
  !! Dimensions
  integer :: nspecies
  rr=>get_rr_ptr(sv)
  psi_log => get_psi_log_ptr(sv)
  sp2nmult=> get_sp2nmult_ptr(sv)
  mu_sp2j=>get_mu_sp2j_ptr(sv)
  nspecies = get_nspecies(sv)
  dlambda = get_dr_jt(rr)

  mu_sp2radii = -999
  do sp=1, nspecies
    do mu=1,sp2nmult(sp)
      j = mu_sp2j(mu, sp)

      r2 = -1.0d0
      if(ccenter=="POW&COEFF") then
        r2 = get_radii_sqr(psi_log(:,mu,sp), rr, dlambda)
      else if (ccenter=="MAXLOC") then  
        r2 = rr(maxloc(psi_log(:,mu,sp)**2,1))**2
      else
        _die('unknown ccenter')
      endif
        
      if(ccoeff=='1') then
        mu_sp2radii(mu, sp) = sqrt(r2)**pow
      else if(ccoeff=='2*L+1') then
        mu_sp2radii(mu, sp) = sqrt(r2)**pow * (2*j+1)
      else
        _die('some unknwn recipe ?')
      endif    
        
    end do ! mu
  end do ! sp
  
end subroutine ! comp_mu_sp2radii_sqr

!
! Computes a mean radii for each multiplett
!
subroutine comp_mu_sp2inv_wght_Toyoda_Ozaki(sv, mu_sp2iwght)
  use m_functs_l_mult_type, only : get_radii_sqr
  use m_system_vars, only : get_rr_ptr, get_sp2nmult_ptr, get_nspecies
  use m_system_vars, only : get_mu_sp2j_ptr, get_psi_log_ptr, get_nr
  use m_interpolation, only : get_dr_jt, diff
  !!  external
  implicit none
  type(system_vars_t), intent(in) :: sv
  real(8), intent(out) :: mu_sp2iwght(:,:)
  !! internal
  real(8), pointer :: psi_log(:,:,:), rr(:)
  integer, pointer :: sp2nmult(:), mu_sp2j(:,:)
  integer :: sp, mu, j
  real(8) :: dr_jt, r2, lapl
  real(8), allocatable :: ff_diff(:)
  real(8), pointer :: ff(:)
  !! Dimensions
  integer :: nspecies, nr
  nr = get_nr(sv)
  rr=>get_rr_ptr(sv)
  psi_log => get_psi_log_ptr(sv)
  sp2nmult=> get_sp2nmult_ptr(sv)
  mu_sp2j=>get_mu_sp2j_ptr(sv)
  nspecies = get_nspecies(sv)
  dr_jt = get_dr_jt(rr)
  allocate(ff_diff(nr))

  mu_sp2iwght = -999
  do sp=1, nspecies
    do mu=1,sp2nmult(sp)
      j = mu_sp2j(mu, sp)
      ff => psi_log(:,mu,sp)
      r2 = get_radii_sqr(ff, rr, dr_jt)

      call diff(j, nr, ff, rr, dr_jt, ff_diff)
      lapl=(sum(ff_diff**2 * rr**3) + j*(j+1)*sum(ff**2 *rr) ) * dr_jt
      mu_sp2iwght(mu, sp) = r2 / lapl
    end do ! mu
  end do ! sp

!    !! Compute tkcmplx()
!    tkcmplx = 0
!    call diff(j1, nr, f1, rr, dr_jt, f1_diff)
!    call diff(j2, nr, f2, rr, dr_jt, f2_diff)
!    sum2=sum(f1_diff*f2_diff*dlambda_rr_cube)+j1*(j1+1)*sum(f1*f2*rr)*dr_jt
!!    write(6,*) sum(f1_diff*f2_diff), sum(dlambda_rr_cube), __LINE__
!    do m=-j1,j1; tkcmplx(m,m)=sum2; enddo;
!    !! END of Compute tkcmplx()
  
  _dealloc(ff_diff)
end subroutine ! comp_mu_sp2inv_wght_Toyoda_Ozaki


end module !m_biloc_aux
