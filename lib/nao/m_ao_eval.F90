module m_ao_eval
  use iso_c_binding, only: c_double, c_int64_t

  implicit none

#include "m_define_macro.F90"

  contains

!
! Compute values of atomic orbitals for a given specie
!
subroutine ao_eval(nmu, &
  ir_mu2v_rl, & ! (nr,nmu)
  nr, &
  rhomin_jt, &
  dr_jt, &
  mu2j, & 
  mu2s, &
  mu2rcut, &
  rcen, & ! rcen(3)
  ncoords, &
  coords, & ! (3,ncoords)
  norbs, & 
  res, & ! (ldres,norbs)
  ldres ) bind(c, name='ao_eval')

  use m_rsphar, only : rsphar  

  implicit none 
  !! external
  integer(c_int64_t), intent(in)  :: nmu
  integer(c_int64_t), intent(in)  :: nr
  real(c_double), intent(in)  :: ir_mu2v_rl(nr,nmu)
  real(c_double), intent(in)  :: rhomin_jt
  real(c_double), intent(in)  :: dr_jt
  integer(c_int64_t), intent(in)  :: mu2j(nmu)
  integer(c_int64_t), intent(in)  :: mu2s(nmu+1)
  real(c_double), intent(in)  :: mu2rcut(nmu)
  real(c_double), intent(in)  :: rcen(3)
  integer(c_int64_t), intent(in)  :: ncoords
  real(c_double), intent(in)  :: coords(3,ncoords)
  integer(c_int64_t), intent(in)  :: norbs 
  integer(c_int64_t), intent(in)  :: ldres        ! must be >=ncoords
  real(c_double), intent(out) :: res(ldres,norbs) ! norbs is unknown.
  !! internal
  real(c_double), allocatable :: rsh(:)
  real(c_double) :: coeffs(6), r, fval, coord(3), rcutmx
  integer(c_int64_t) :: jmx_sp, icrd, mu, j, s,f,k
  !write(6,*) nmu
  !write(6,*) nr
  !write(6,*) ir_mu2v_rl(1:3,1)
  !write(6,*) ir_mu2v_rl(1:3,2)
  !write(6,*) ir_mu2v_rl(1:3,3)
  !write(6,*) rhomin_jt
  !write(6,*) dr_jt
  !write(6,*) mu2j
  !write(6,*) mu2s(1:nmu+1)
  !write(6,*) rcen
  !write(6,*) ncoords
  !write(6,*) coords(:,1)
  !write(6,*) norbs
  !write(6,*) ldres

  rcutmx = maxval(mu2rcut)
  jmx_sp = maxval(mu2j)
  allocate(rsh(0:(jmx_sp+1)**2-1))
  res = 0
  do icrd = 1,ncoords
    coord = coords(:,icrd)-rcen
    call rsphar(coord, int(jmx_sp), rsh)
    r = sqrt(sum(coord**2))
    if(r>rcutmx) cycle
    call comp_coeffs(r, nr, rhomin_jt, dr_jt, k, coeffs)
    do mu=1,nmu
      ! if(r>mu2rcut(mu)) cycle
      j=mu2j(mu)
      s=mu2s(mu)+1
      f=mu2s(mu+1)
      fval = sum(ir_mu2v_rl(k:k+5,mu)*coeffs)
      if (j>0) fval = fval * (r**j)
      res(icrd,s:f) = fval * rsh(j*(j+1)-j:j*(j+1)+j)
    enddo ! mu
  enddo ! icrd
  
  _dealloc(rsh)
  
end subroutine !ao_eval


!
! Compute values of atomic orbitals for a given specie
!
subroutine comp_coeffs(r, nr, gammin_jt, dg_jt, k, coeffs)
  implicit none 
  !! external
  real(c_double), intent(in) :: r, gammin_jt, dg_jt
  integer(c_int64_t), intent(in) :: nr
  integer(c_int64_t), intent(out) :: k
  real(c_double), intent(out) :: coeffs(:)
  !! internal
  real(c_double) :: dy, lr
  
  if (r<=0) then
    coeffs = 0
    coeffs(1) = 1
    k = 1
    return
  endif  

  lr = log(r)
  k  = int((lr-gammin_jt)/dg_jt+1)
  k  = min(max(k,3_c_int64_t), nr-3_c_int64_t)
  dy = (lr-gammin_jt-(k-1_c_int64_t)*dg_jt)/dg_jt
  
  coeffs(1) =     -dy*(dy**2-1.0D0)*(dy-2.0D0)*(dy-3.0D0)/120.0D0
  coeffs(2) = +5.0D0*dy*(dy-1.0D0)*(dy**2-4.0D0)*(dy-3.0D0)/120.0D0
  coeffs(3) = -10.0D0*(dy**2-1.0D0)*(dy**2-4.0D0)*(dy-3.0D0)/120.0D0
  coeffs(4) = +10.0D0*dy*(dy+1.0D0)*(dy**2-4.0D0)*(dy-3.0D0)/120.0D0
  coeffs(5) = -5.0D0*dy*(dy**2-1.0D0)*(dy+2.0D0)*(dy-3.0D0)/120.0D0
  coeffs(6) =      dy*(dy**2-1.0D0)*(dy**2-4.0D0)/120.0D0

  k = k - 2
  return
   
end subroutine !comp_coeffs

end module !m_ao_eval
