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
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  type(system_vars_t) :: sv
  type(prod_basis_t) :: pb
  type(para_t) :: para
  
!
! This is generation of vertex coefficients and conversion coefficients
! at the same time.
!  

  contains

!
! 
!
subroutine sv_prod_log(ninp,dinp) bind(c, name='sv_prod_log')

  use m_system_vars, only : system_vars_t
  use m_fact, only : init_fact
  use m_parallel, only : para_t
  use m_prod_basis_type, only : prod_basis_t
  use m_sv_prod_log_get, only : sv_prod_log_get
  
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  real(c_double), intent(in) :: dinp(ninp)
  !! internal

  !! executable statements
  call init_fact()  !! Initializations for product reduction in Talman's way
  call sv_prod_log_get(ninp,dinp, sv)
    
end subroutine ! sv_prod_log


end module !m_sv_prod_log
