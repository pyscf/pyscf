!
! File rkb_giao_vhf_coul.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
!  HF potential of Dirac-Coulomb Hamiltonian represented in GIAOs.
!       J = (i^L i^L|\mu g\nu) + (i^S i^S|\mu g\nu)
!         + (i^L i^L|p\mu gp\nu) + (i^S i^S|p\mu gp\nu)
!         + (i^L gi^L|\mu \nu) + (i^S gi^S|\mu \nu)
!         + (i^L gi^L|p\mu p\nu) + (i^S gi^S|p\mu p\nu)
!       K = (\mu gi^L|i^L \nu) + (\mu gi^L|i^S p\nu)
!         + (p\mu gi^S|i^L \nu) + (p\mu gi^S|i^S p\nu)
!         + (\mu i^L|i^L g\nu) + (p\mu i^S|i^L g\nu)
!         + (\mu i^L|i^S gp\nu) + (p\mu i^S|i^S gp\nu)
!  Density matrix *is* assumed to be *Hermitian*
!

!**************************************************
! J
!    (gi^L i^L|\mu \nu) + (gi^S i^S|\mu \nu)
!  + (gi^L i^L|p\mu p\nu) + (gi^S i^S|p\mu p\nu) = 0
! += (\mu g\nu|i^L i^L) + (\mu g\nu|i^S i^S)
!  + (p\mu gp\nu|i^L i^L) + (p\mu gp\nu|i^S i^S)
!  = (\mu g\nu|i^L i^L) + (\mu g\nu|i^S i^S)
!  + (p\mu gp\nu|i^L i^L) + (p\mu gp\nu|i^S i^S)

! K = (\mu gi^L|i^L \nu) + (\mu gi^L|i^S p\nu)
!   + (p\mu gi^S|i^L \nu) + (p\mu gi^S|i^S p\nu)
!   + (\mu i^L|i^L g\nu) + (p\mu i^S|i^L g\nu)
!   + (\mu i^L|i^S gp\nu) + (p\mu i^S|i^S gp\nu)
!   = (\mu gi^L|i^L \nu) + (\mu gi^L|i^S p\nu)
!   + (p\mu gi^S|i^L \nu) + (p\mu gi^S|i^S p\nu)
!   + h.c.
!

! tas_ts_dm2 gives correct J because the density-1 vanishes
! If call tas_ts_dm2, then do K + h.c., otherwise using rkb_vhf_after to scale
! the SS block
subroutine rkb_giao_vhf_after(vj, vk, n4c, nset, nset_dm, atm, natm, bas, nbas, env)
  use cint_const_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(inout)  ::  vj(n4c,n4c,3)
  double complex,intent(inout)  ::  vk(n4c,n4c,3)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer                       ::  n2c, is

  n2c = n4c / 2

  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c,:) = vk(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2

  do is = 1, 3
    call zvpluscc_inplace(vk(1,1,is), n4c, n4c)
  end do
  return
end subroutine rkb_giao_vhf_after

subroutine rkb_giao_vhf_LL_after(vjll, vkll, n2c, nset, nset_dm, atm, natm, bas, nbas, env)
  use cint_const_mod
  implicit none
  integer,intent(in)            ::  n2c, nset, nset_dm
  double complex,intent(inout)  ::  vjll(n2c,n2c,3)
  double complex,intent(inout)  ::  vkll(n2c,n2c,3)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer                       ::  is
  do is = 1, 3
    call zvpluscc_inplace(vkll(1,1,is), n2c, n2c)
  end do
end subroutine rkb_giao_vhf_LL_after

subroutine zvpluscc_inplace(v, n, ldv)
  implicit none
  integer,intent(in)            ::  n, ldv
  double complex,intent(inout)  ::  v(ldv,*)
  integer       ::  i, j
  double complex,allocatable    ::  dag(:,:)
  allocate (dag(n,n))
  do j = 1, n
    do i = 1, n
      dag(j,i) = conjg(v(i,j))
    end do
  end do
  v(:n,:n) = v(:n,:n) + dag
  deallocate (dag)
  return
end subroutine zvpluscc_inplace
