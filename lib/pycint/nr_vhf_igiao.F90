!
! File nr_vhf_igiao.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
! *suitable for the intor which is time-reversal anti-symmetric*
!
!  non-relativistic HF coulomb and exchange potential in GIAOs
!       J = i[(i i|\mu g\nu) + (i gi|\mu \nu)]
!       K = i[(\mu gi|i \nu) + (\mu i|i g\nu)]
!         = (\mu g i|i \nu) - h.c.
!  Density matrix should be Hermitian
!  no vj + h.c. because 1st order density vanishes

! no vj_kl because of the 1st density vanishes due to anti time-reversal symm.
! If call has_hs_dm2, then do K - h.c., otherwise skip this step
subroutine nr_vhf_igiao_after(vj, vk, ndim, nset, nset_dm, atm, natm, bas, nbas, env)
  use cint_const_mod
  implicit none
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(inout)::  vj(ndim,ndim,3)
  double precision,intent(inout)::  vk(ndim,ndim,3)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer               ::  i, j
  double precision      ::  tmp(3)

  do j = 1, ndim
    do i = 1, j - 1
      tmp = vk(i,j,:) - vk(j,i,:)
      vk(i,j,:) = tmp
      vk(j,i,:) = -tmp
    end do
    vk(j,j,:) = 0.d0
  end do
end subroutine nr_vhf_igiao_after
