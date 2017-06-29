module m_vrtx_cc_apair

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use iso_c_binding, only: c_double, c_int64_t
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime

  contains

!
! The subroutine is generating the dominant product vertices and conversion coefficiens
! for a given atom pair
!
subroutine vrtx_cc_apair(sp12_c,rc12,dout,nout) bind(c, name='vrtx_cc_apair')
  use m_sv_prod_log, only : a, bp2info, ff2, evals, vertex_real2, oo2num, m2nf, vertex_cmplx2, rhotb 
  use m_bilocal_vertex, only : make_bilocal_vertex_rf
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: sp12_c(2)
  real(c_double), intent(in) :: rc12(3,2)
  integer(c_int64_t), intent(inout) :: nout
  real(c_double), intent(inout) :: dout(nout)
  !! internal
  integer :: nrfmx, ls, rf
  integer(c_int64_t) :: sp12(2)
  real(8) :: ttt(9), rcut, center(3)
  logical :: lready 

  if(.not. allocated(bp2info)) then; write(6,*) __FILE__, __LINE__; stop '!bp2info'; endif

  sp12 = sp12_c+1 ! getting zero based indices !
  write(6,*) ' a%sv%uc%sp2nmult ' , a%sv%uc%sp2nmult
  write(6,*) ' sp12 ', sp12

  bp2info(1)%atoms = -1
  bp2info(1)%species(1:2) = int(sp12(1:2))
  bp2info(1)%coords(1:3,1:2) = rc12(1:3,1:2)
  bp2info(1)%cells = 0
  bp2info(1)%ls2nrf(1:2) = a%sv%uc%sp2nmult(sp12(1:2))
  nrfmx = maxval(bp2info(1)%ls2nrf)
  _dealloc(bp2info(1)%rf_ls2mu)
  allocate(bp2info(1)%rf_ls2mu(nrfmx,2))
  bp2info(1)%rf_ls2mu = -999
  do ls=1,2; do rf=1,bp2info(1)%ls2nrf(ls); bp2info(1)%rf_ls2mu(rf,ls) = rf; enddo; enddo 

  call make_bilocal_vertex_rf(a, bp2info(1), &
    ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, &
    vertex_cmplx2, rhotb, ttt)

  !call apair_put(bp2info, ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, dout, nout)
 
end subroutine ! vrtx_cc_apair


end module !m_vrtx_cc_apair
