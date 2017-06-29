module m_sv_prod_log

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use m_prod_basis_param, only : prod_basis_param_t
  use m_orb_rspace_type, only : orb_rspace_aux_t
  use m_parallel, only : para_t
  use m_pair_info, only : pair_info_t
  use m_prod_basis_type, only : prod_basis_t
  use iso_c_binding, only: c_double, c_int64_t
  use m_biloc_aux, only : biloc_aux_t
  use m_dp_aux, only : dp_aux_t
 
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  type(system_vars_t) :: sv
  type(prod_basis_t) :: pb
  type(para_t) :: para
  type(orb_rspace_aux_t) :: orb_a
  type(pair_info_t), allocatable :: bp2info(:)
  type(biloc_aux_t) :: a
  type(dp_aux_t) :: dp_a

  real(8), allocatable :: ff2(:,:,:,:,:), evals(:,:,:), vertex_real2(:,:,:,:,:)
  real(8), allocatable :: rhotb(:,:), sp2rcut(:)
  integer, allocatable :: oo2num(:,:), m2nf(:), i2s(:), p2n(:)
  complex(8), allocatable :: vertex_cmplx2(:,:,:,:,:)
  logical :: lready, lcheck_cpy

  contains

!
! 
!
subroutine sv_prod_log(dinp,ninp) bind(c, name='sv_prod_log')

  use m_fact, only : init_fact
  use m_sv_prod_log_get, only : sv_prod_log_get
  use m_biloc_aux, only : init_biloc_aux
  use m_orb_rspace_type, only : init_orb_rspace_aux
  use m_system_vars, only : get_nr, get_jmx, get_norbs_max, get_natoms, get_sp2rcut
  use m_prod_basis_param, only : get_jcutoff
  use m_parallel, only : init_parallel
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  real(c_double), intent(in) :: dinp(ninp)
  !! internal

  integer :: ul, natoms, nr, jcutoff, jmx, norbs_max, nf_max, nterm_max
  real(c_double) :: ac_rcut
  real(c_double), allocatable :: sp2rcut(:)
  
  !! executable statements
  call init_parallel(para, 0)
  call init_fact()  !! Initializations for product reduction in Talman's way
  call sv_prod_log_get(dinp,ninp, sv, pb)
  call init_orb_rspace_aux(sv, orb_a, ul)
  call init_biloc_aux(sv, pb%pb_p, bp2info, para, orb_a, a)

  allocate(bp2info(1))

  natoms = get_natoms(a%sv)
  nr = get_nr(a%sv)
  jcutoff = get_jcutoff(a%pb_p)
  jmx = get_jmx(a%sv)
  norbs_max = get_norbs_max(a%sv)
  nf_max = a%nf_max
  nterm_max = a%nterm_max
  ac_rcut   = a%pb_p%ac_rcut
  call get_sp2rcut(sv, sp2rcut)

  allocate(ff2(nr,0:jcutoff,nf_max,-jmx*2:jmx*2,2))
  allocate(vertex_cmplx2(-jmx*2:jmx*2,nf_max,norbs_max,norbs_max,2))
  allocate(vertex_real2(-jmx*2:jmx*2,nf_max,norbs_max,norbs_max,2))
  allocate(evals(nf_max,nf_max+1,-jmx*2:jmx*2))
  allocate(rhotb(nr,nterm_max))


end subroutine ! sv_prod_log


end module !m_sv_prod_log
