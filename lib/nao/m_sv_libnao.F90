module m_sv_libnao

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use iso_c_binding, only: c_double, c_double_complex, c_int64_t
 
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  type(system_vars_t) :: sv
  
  contains

!
! 
!
subroutine init_sv_libnao(dinp,ninp) bind(c, name='init_sv_libnao')

  use m_fact, only : init_fact
  use m_sv_get, only : sv_get
  use m_system_vars, only : get_nr
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  real(c_double), intent(in) :: dinp(ninp)
  !! internal

  !! executable statements
  call init_fact()  !! Initializations for product reduction/spherical harmonics/wigner3j in Talman's way
  call sv_get(dinp,ninp, sv)

end subroutine ! init_sv_libnao


end module !m_sv_libnao
