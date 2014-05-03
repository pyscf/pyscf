subroutine rkb_vhf_coul_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
!**************************************************
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(out)    ::  vj(ndim,ndim)
  double complex,intent(out)    ::  vk(ndim,ndim)
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
  integer               ::  n2c, n4c, s
  integer               ::  shls(4)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:)
  double complex,allocatable    ::  tdm(:,:)

  vj = 0d0
  vk = 0d0

  n2c = CINTtot_cgto_spinor(bas, nbas)
  n4c = n2c * 2
  s = n2c + 1

  allocate (ao_loc(nbas))
  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  allocate (tdm(n4c,n4c))
  tdm(  :n2c,  :n2c) = dm(  :n2c,  :n2c)
  tdm(  :n2c,n2c+1:) = dm(  :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  tdm(n2c+1:,  :n2c) = dm(n2c+1:,  :n2c) * (.5d0/env(PTR_LIGHT_SPEED))
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

  allocate (fijkl(di,dj,dk,dl))
  call cint2e(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_kl_O0(fijkl, tdm(1,1), vj(1,1), ndim, shls, ao_loc)
  call HFBLK_il_O0(fijkl, tdm(1,1), vk(1,1), ndim, shls, ao_loc)

  call cint2e_spsp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_kl_O0(fijkl, tdm(s,s), vj(1,1), ndim, shls, ao_loc)
  call HFBLK_ij_O0(fijkl, tdm(1,1), vj(s,s), ndim, shls, ao_loc)
  call HFBLK_il_O0(fijkl, tdm(s,1), vk(s,1), ndim, shls, ao_loc)

  call cint2e_spsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_kl_O0(fijkl, tdm(s,s), vj(s,s), ndim, shls, ao_loc)
  call HFBLK_il_O0(fijkl, tdm(s,s), vk(s,s), ndim, shls, ao_loc)
  deallocate (fijkl)

        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  vj(n2c+1:,n2c+1:) = vj(n2c+1:,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(n2c+1:,1 :n2c) = vk(n2c+1:,1 :n2c) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:) = vk(n2c+1:,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  ! LS-block of exchange part
  do j = n2c+1, n4c
    do i = 1, n2c
      vk(i,j) = conjg(vk(j,i))
    end do
  end do
  deallocate (tdm)
  deallocate (ao_loc)
end subroutine rkb_vhf_coul_O00
