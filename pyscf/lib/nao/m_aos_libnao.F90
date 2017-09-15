module m_aos_libnao
  use iso_c_binding, only: c_double, c_int64_t
  use m_system_vars, only: system_vars_t
  use m_spin_dens_aux, only : spin_dens_aux_t, comp_coeff
  implicit none
#include "m_define_macro.F90"

  type(system_vars_t), pointer :: sv => null()
  type(spin_dens_aux_t) :: sda
  real(c_double), allocatable :: rsh(:)

  contains

!
! Compute values of atomic orbitals for the whole molecule and a given set of coordinates
!
subroutine aos_libnao(ncoords, coords, norbs, oc2val, ldc ) bind(c, name='aos_libnao')
  use m_rsphar, only : rsphar
  use m_die, only : die
  implicit none
  !! external
  integer(c_int64_t), intent(in)  :: ncoords
  real(c_double), intent(in)  :: coords(3,ncoords)
  integer(c_int64_t), intent(in)  :: norbs
  integer(c_int64_t), intent(in)  :: ldc
  real(c_double), intent(inout)  :: oc2val(ldc,ncoords)

  ! Interne Variable:
  real(8) :: br0(3), rho, br(3)
  real(8) :: fr_val
  integer(c_int64_t) :: jmx_sp
  integer  :: atm, spa, mu, j, m, jjp1,k,so,start_ao,icoord
  real(8)  :: coeff(-2:3)

  !! values of localized orbitals
  if (norbs/=sda%norbs) then
    write(6,*) norbs, sda%norbs
    _die('norbs/=sda%norbs')
  endif
  
  do icoord=1,ncoords
    br = coords(1:3,icoord)
    do atm=1,sda%natoms;
      spa  = sda%atom2sp(atm);
      jmx_sp = maxval(sda%mu_sp2j(:,spa))
      br0  = br - sda%coord(:,atm);  !!print *, 'br01',br,br01,br1;
      rho = sqrt(sum(br0**2));
      if (rho>sda%atom2rcut(atm)) cycle
      
      call comp_coeff(sda, coeff, k, rho)
      call rsphar(br0, jmx_sp, rsh(0:));
      so = sda%atom2start_orb(atm)
    
      do mu=1,sda%sp2nmult(spa); 
        if(rho>sda%mu_sp2rcut(mu,spa)) cycle;
        start_ao = sda%mu_sp2start_ao(mu, spa)
        fr_val = sum(coeff*sda%psi_log(k-2:k+3,mu,spa));
        j = sda%mu_sp2j(mu,spa);
        jjp1 = j*(j+1);
        do m =-j,j; oc2val(start_ao + j + m + so - 1, icoord)= fr_val*rsh(start_ao + j + m + so - 1); end do ! m
      enddo; ! mu
    enddo; ! atom
  enddo ! icoord
 
end subroutine ! aos_libnao

!
!
!
subroutine init_aos_libnao(norbs, info) bind(c, name='init_aos_libnao') 
  use m_system_vars, only : get_norbs, get_nspin, get_jmx
  use m_sv_libnao, only : sv_libnao=>sv
  use m_spin_dens_aux, only : init_spin_dens_withoutdm
  use m_rsphar, only : rsphar, init_rsphar
  use m_die, only : die
  implicit none
  ! external
  integer(c_int64_t), intent(in) :: norbs
  integer(c_int64_t), intent(inout) :: info
  ! internal
  integer(c_int64_t) :: n, nspin, jmx
  sv => null()
  
  n = get_norbs(sv_libnao)
  !write(6,*) 'n ', n
  nspin = sv_libnao%nspin
  if ( nspin < 1 .or. nspin > 2 ) then
    write(6,*) nspin
    info = 1
    _die('nspin < 1 .or. nspin > 2')
  endif

  if ( n /= norbs ) then
    write(6,*) n, norbs
    info = 2
    _die('n /= norbs')
  endif

  sv => sv_libnao
  call init_spin_dens_withoutdm(sv, sda)
  jmx = get_jmx(sv)
  call init_rsphar(jmx)
  _dealloc(rsh)
  allocate(rsh(0:(jmx+1)**2-1))
  info = 0
 
end subroutine !init_dens_libnao

end module !m_aos_libnao
