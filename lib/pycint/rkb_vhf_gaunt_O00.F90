subroutine rkb_vhf_gaunt_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s
  integer               ::  shls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:)
  double complex,allocatable    ::  tijkl(:,:,:,:)
  double complex,allocatable    ::  ijtkl(:,:,:,:)
  double complex,allocatable    ::  tijtkl(:,:,:,:)
  double complex,allocatable    ::  tdm(:,:)

  vj = 0d0
  vk = 0d0

  n2c = CINTtot_cgto_spinor(bas, nbas)
  n4c = n2c * 2
  s = n2c + 1

  allocate (ao_loc(nbas))
  call CINTshells_spinor_offset(ao_loc, bas, nbas)
  call time_reversal_spinor(tao, bas, nbas)

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

  allocate (fijkl(di,dj,dk,dl))
  allocate (ijtkl(di,dj,dl,dk))
  allocate (tijkl(dj,di,dk,dl))
  allocate (tijtkl(dj,di,dl,dk))
  !\sigma \sigma\cdot p | \sigma \sigma\cdot p
  shls = (/ishell, jshell, kshell, lshell/)
  call cint2e_ssp1ssp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_kl_O0(fijkl, tdm(s,1), vj(1,s), ndim, shls, ao_loc)
  call HFBLK_il_O0(fijkl, tdm(s,1), vk(1,s), ndim, shls, ao_loc)

  !\sigma \sigma\cdot p | \sigma\cdot p \sigma
  call HFBLK_atimerev_ijtltk(fijkl, ijtkl, shls, ao_loc, tao)
  shls = (/ishell, jshell, lshell, kshell/)
  call HFBLK_kl_O0(ijtkl, tdm(s,1), vj(s,1), ndim, shls, ao_loc)
  call HFBLK_il_O0(ijtkl, tdm(s,s), vk(1,1), ndim, shls, ao_loc)

  !\sigma\cdot p \sigma | \sigma \sigma\cdot p
  shls = (/ishell, jshell, kshell, lshell/)
  call HFBLK_atimerev_tjtikl(fijkl, tijkl, shls, ao_loc, tao)
  shls = (/jshell, ishell, kshell, lshell/)
  call HFBLK_kl_O0(tijkl, tdm(1,s), vj(1,s), ndim, shls, ao_loc)
  call HFBLK_il_O0(tijkl, tdm(1,1), vk(s,s), ndim, shls, ao_loc)

  !\sigma\cdot p \sigma | \sigma\cdot p \sigma
  call HFBLK_atimerev_ijtltk(tijkl, tijtkl, shls, ao_loc, tao)
  shls = (/jshell, ishell, lshell, kshell/)
  call HFBLK_kl_O0(tijtkl, tdm(1,s), vj(s,1), ndim, shls, ao_loc)
  call HFBLK_il_O0(tijtkl, tdm(1,s), vk(s,1), ndim, shls, ao_loc)
  deallocate (fijkl, ijtkl, tijkl, tijtkl)

        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  vj(n2c+1:,  :n2c) = vj(n2c+1:,  :n2c) * (.5d0/env(PTR_LIGHT_SPEED))
  vj(  :n2c,n2c+1:) = vj(  :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c) = vk(n2c+1:,  :n2c) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:) = vk(  :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:) = vk(n2c+1:,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  deallocate (tdm)
  deallocate (ao_loc)
end subroutine rkb_vhf_gaunt_O00

subroutine rkb_vhf_gaunt_O01(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:)
  double complex,allocatable    ::  tijkl(:,:,:,:)
  double complex,allocatable    ::  tdm(:,:)

  vj = 0d0
  vk = 0d0

  n2c = CINTtot_cgto_spinor(bas, nbas)
  n4c = n2c * 2
  s = n2c + 1

  allocate (ao_loc(nbas))
  call CINTshells_spinor_offset(ao_loc, bas, nbas)
  call time_reversal_spinor(tao, bas, nbas)

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

  allocate (fijkl(di,dj,dk,dl))
  allocate (tijkl(dj,di,dk,dl))
  !\sigma \sigma\cdot p | \sigma \sigma\cdot p
  shls = (/ishell, jshell, kshell, lshell/)
  call cint2e_ssp1ssp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_kl_O0(fijkl, tdm(s,1), vj(1,s), ndim, shls, ao_loc)
  call HFBLK_il_O0(fijkl, tdm(s,1), vk(1,s), ndim, shls, ao_loc)

  !\sigma\cdot p \sigma | \sigma \sigma\cdot p
  call HFBLK_atimerev_tjtikl(fijkl, tijkl, shls, ao_loc, tao)
  shls = (/jshell, ishell, kshell, lshell/)
  call HFBLK_kl_O0(tijkl, tdm(1,s), vj(1,s), ndim, shls, ao_loc)
  call HFBLK_il_O0(tijkl, tdm(1,1), vk(s,s), ndim, shls, ao_loc)
  call HFBLK_kj_O0(tijkl, tdm(s,s), vk(1,1), ndim, shls, ao_loc)
  deallocate (fijkl, tijkl)

        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  vj(  :n2c,n2c+1:) = vj(  :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:) = vk(  :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:) = vk(n2c+1:,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  do j = 1, n2c
    do i = n2c+1, n4c
      vj(i,j) = conjg(vj(j,i))
      vk(i,j) = conjg(vk(j,i))
    end do
  end do
  deallocate (tdm)
  deallocate (ao_loc)
end subroutine rkb_vhf_gaunt_O01

subroutine rkb_vhf_gaunt_O02(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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
  integer               ::  shls(4), tijshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:)
  double complex,allocatable    ::  tijkl(:,:,:,:)
  double complex,allocatable    ::  ijtkl(:,:,:,:)
  double complex,allocatable    ::  fklij(:,:,:,:)
  double complex,allocatable    ::  tdm(:,:)

  vj = 0d0
  vk = 0d0

  n2c = CINTtot_cgto_spinor(bas, nbas)
  n4c = n2c * 2
  s = n2c + 1

  allocate (ao_loc(nbas))
  call CINTshells_spinor_offset(ao_loc, bas, nbas)
  call time_reversal_spinor(tao, bas, nbas)

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
          if (ishell+jshell*nbas > kshell+lshell*nbas) then
            cycle
          end if

  allocate (fijkl(di,dj,dk,dl))
  allocate (tijkl(dj,di,dk,dl))
  shls = (/ishell, jshell, kshell, lshell/)
  tijshls = (/jshell, ishell, kshell, lshell/)
  call cint2e_ssp1ssp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_kl_O0(fijkl, tdm(s,1), vj(1,s), ndim, shls, ao_loc)
  call HFBLK_il_O0(fijkl, tdm(s,1), vk(1,s), ndim, shls, ao_loc)
  ! There would be no vj in closed shell sys because the density vanishes due
  ! to the anti time-reversal symm.

  call HFBLK_atimerev_tjtikl(fijkl, tijkl, shls, ao_loc, tao)
  call HFBLK_kl_O0(tijkl, tdm(1,s), vj(1,s), ndim, tijshls, ao_loc)
  call HFBLK_il_O0(tijkl, tdm(1,1), vk(s,s), ndim, tijshls, ao_loc)
  call HFBLK_kj_O0(tijkl, tdm(s,s), vk(1,1), ndim, tijshls, ao_loc)
  deallocate (tijkl)

  if (ishell /= kshell .or. jshell /= lshell) then
    call HFBLK_ij_O0(fijkl, tdm(s,1), vj(1,s), ndim, shls, ao_loc)
    call HFBLK_kj_O0(fijkl, tdm(s,1), vk(1,s), ndim, shls, ao_loc)

    allocate (ijtkl(di,dj,dl,dk))
    call HFBLK_atimerev_ijtltk(fijkl, ijtkl, shls, ao_loc, tao)
    shls = (/ishell, jshell, lshell, kshell/)
    call HFBLK_ij_O0(ijtkl, tdm(1,s), vj(1,s), ndim, shls, ao_loc)
    call HFBLK_il_O0(ijtkl, tdm(s,s), vk(1,1), ndim, shls, ao_loc)
    call HFBLK_kj_O0(ijtkl, tdm(1,1), vk(s,s), ndim, shls, ao_loc)
    deallocate (ijtkl)
  end if
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell
  deallocate (tdm)
  deallocate (ao_loc)

  vj(  :n2c,n2c+1:) = vj(  :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:) = vk(  :n2c,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:) = vk(n2c+1:,n2c+1:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  do j = 1, n2c
    do i = n2c+1, n4c
      vj(i,j) = conjg(vj(j,i))
      vk(i,j) = conjg(vk(j,i))
    end do
  end do
end subroutine rkb_vhf_gaunt_O02
