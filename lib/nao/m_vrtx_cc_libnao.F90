module m_vrtx_cc_libnao

  use m_precision, only : blas_int
#include "m_define_macro.F90" 
  use m_die, only : die
  use m_warn, only : warn
  use m_system_vars, only : system_vars_t
  use m_prod_basis_param, only : prod_basis_param_t
  use m_orb_rspace_type, only : orb_rspace_aux_t
  use m_parallel, only : para_t
  use m_pair_info, only : pair_info_t
  use m_log, only : log_memory_note
  
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime

!
! This is generation of vertex coefficients and conversion coefficients
! at the same time.
!  

  contains

!
! The subroutine is generating the dominant product vertices and conversion coefficiens
! for a given atom pair
!
subroutine vrtx_cc_libnao(ninp,dinp, nout,dout)

  use m_system_vars, only : system_vars_t
  use m_log, only : log_memory_note, log_timing_note
  use m_fact, only : init_fact
  use m_parallel, only : para_t
  use m_prod_basis_param, only : prod_basis_param_t
  use m_orb_rspace_type, only : init_orb_rspace_aux
  use m_pair_info, only : pair_info_t, distr_atom_pairs_info
  use m_biloc_aux, only : biloc_aux_t, init_biloc_aux
  use m_dp_aux, only : dp_aux_t
  use m_prod_basis_type, only : prod_basis_t
  use m_sv_get, only : sv_get
  
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  real(c_double), intent(in) :: dinp(ninp)
  integer(c_int64_t), intent(inout) :: nout
  real(c_double), intent(inout) :: dout(nout)
  !! internal
  type(system_vars_t) :: sv
  type(prod_basis_param_t) :: pb_p
  type(pair_info_t), allocatable :: bp2info(:)
  type(dp_aux_t) :: dp_a
  type(prod_basis_t) :: pb
  type(para_t) :: para
  type(biloc_aux_t) :: a
  type(orb_rspace_aux_t) :: orb_a
  integer :: ul, nbp_node
  real(8) :: t1,t2,t=0
  !! executable statements
  call init_fact()  !! Initializations for product reduction in Talman's way
  
  call sv_get(ninp,dinp, sv)
  call init_orb_rspace_aux(sv, orb_a, ul)
  allocate(bp2info(1))
  
  !call init_biloc_aux(sv, pb_p, bp2info, para, orb_a, a)
  
  !nbp_node = size(bp2info)
  !call make_bilocal_vertex(a, nbp_node, bp2info, dp_a, pb, iv)
    
end subroutine ! vrtx_cc_libnao


end module !m_vrtx_cc_libnao
