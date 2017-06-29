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

!
! This is generation of vertex coefficients and conversion coefficients
! at the same time.
!  

  contains

!
! 
!
subroutine sv_prod_log(ninp,dinp) bind(c, name='sv_prod_log')

  use m_fact, only : init_fact
  use m_sv_prod_log_get, only : sv_prod_log_get
  use m_biloc_aux, only : init_biloc_aux
  use m_orb_rspace_type, only : init_orb_rspace_aux
  use m_parallel, only : init_parallel
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  real(c_double), intent(in) :: dinp(ninp)
  !! internal
  integer :: ul

  !! executable statements
  call init_parallel(para, 0)
  call init_fact()  !! Initializations for product reduction in Talman's way
  call sv_prod_log_get(ninp,dinp, sv, pb)

  call init_orb_rspace_aux(sv, orb_a, ul)

  allocate(bp2info(1))

  call init_biloc_aux(sv, pb%pb_p, bp2info, para, orb_a, a)

end subroutine ! sv_prod_log


end module !m_sv_prod_log
