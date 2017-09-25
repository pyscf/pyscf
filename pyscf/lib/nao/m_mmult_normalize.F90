module m_mmult_normalize
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  implicit none

contains

!
!
!
subroutine mmult_normalize(rr, functs, vmm)
  use m_die, only : die
  use m_functs_m_mult_type, only : comp_norm_mmult, get_nfunct_mmult, get_m
  use m_functs_m_mult_type, only : get_ff_ptr_mmult=>get_ff_ptr, get_jmax_mmult
  use m_functs_m_mult_type, only : functs_m_mult_t
  use m_vertex_3cent, only :  vertex_3cent_t
  
  implicit none
  real(8), intent(in) :: rr(:)
  type(functs_m_mult_t), intent(inout) :: functs
  type(vertex_3cent_t), intent(inout) :: vmm
  !! internal
  integer ::  j, f, nf, m, jmax
  real(8), pointer :: ff(:)
  real(8), allocatable :: norms(:)

  call comp_norm_mmult(rr, functs, norms)    
  nf = get_nfunct_mmult(functs)
  jmax = get_jmax_mmult(functs)
  do f=1,nf
    m = get_m(functs, f)
    do j=abs(m),jmax
      ff => get_ff_ptr_mmult(functs, j, f)
      ff = ff / norms(f)
    enddo ! j  
    vmm%vertex(:,:,f) = vmm%vertex(:,:,f) * norms(f)
  enddo ! mu

end subroutine !mmult_normalize

end module !m_mmult_normalize
