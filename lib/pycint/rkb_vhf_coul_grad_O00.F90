!
! File rkb_vhf_coul_grad_O1.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
!  Relativistic coulomb and exchange potential
!  and gradients of basis
!       J = (i i|\nabla\mu \nu)
!       K = (\nabla\mu i|i \nu)
!

subroutine rkb_vhf_coul_grad_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
!**************************************************
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(out)    ::  vj(ndim,ndim,3)
  double complex,intent(out)    ::  vk(ndim,ndim,3)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

! ==========
!
!  HF potential of Dirac-Coulomb Hamiltonian (in AO representation)
!      vg = J - K
!       J = (ii|\mu \nu)
!       K = (\mu i|i\nu)
!  Density matrix is assumed to be Hermitian
!
! ==========
!
! INPUT:
!
!
! OUTPUT:
!
!
!==================================================

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, i, j
  integer               ::  n2c, n4c, s, is
  integer               ::  shls(4)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
  double complex,allocatable    ::  tdm(:,:)

  vj = 0d0
  vk = 0d0

  n2c = CINTtot_cgto_spinor(bas, nbas)
  n4c = n2c * 2
  s = n2c + 1

  allocate (ao_loc(nbas))
  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  allocate (tdm(n4c,n4c))
  tdm(1 :n2c,1 :n2c) = dm(1 :n2c,1 :n2c)
  tdm(1 :n2c,n2c+1:) = dm(1 :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  tdm(n2c+1:,1 :n2c) = dm(n2c+1:,1 :n2c) * (.5d0/env(PTR_LIGHT_SPEED))
  tdm(n2c+1:,n2c+1:) = dm(n2c+1:,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))**2

  do lshell = 0, nbas - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = 0, nbas - 1
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, nbas - 1
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = 0, nbas - 1
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl,3))
  call cint2e_ip1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(1,1), vj(1,1,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(1,1), vk(1,1,is), n4c, shls, ao_loc)
  end do

  call cint2e_ipspsp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(1,1), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(s,1,is), n4c, shls, ao_loc)
  end do

  call cint2e_ip1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(1,1,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(1,s), vk(1,s,is), n4c, shls, ao_loc)
  end do

  call cint2e_ipspsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc)
  end do
  deallocate (fijkl)

        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(1 :n2c,n2c+1:,:) = vk(1 :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,1 :n2c,:) = vk(n2c+1:,1 :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  deallocate (tdm)
  deallocate (ao_loc)
  return
end subroutine rkb_vhf_coul_grad_O00
