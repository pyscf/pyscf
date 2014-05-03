!
! File rmb_vhf_coul_O1.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
!  HF potential of Dirac-Coulomb Hamiltonian (in RMB basis)
!       J = (A10i^S i^S|\mu\nu)   + (i^S A10i^S|\mu\nu)
!         + (A10i^S i^S|p\mu p\nu) + (i^S A10i^S|p\mu p\nu)
!         + (i^L i^L|A10\mu p\nu) + (i^L i^L|p\mu A10\nu)
!         + (i^S i^S|A10\mu p\nu) + (i^S i^S|p\mu A10\nu)
!       K = (A10\mu i^S|i^L \nu) + (A10\mu i^S|i^S p\nu)
!         + (p\mu A10i^S|i^L \nu) + (p\mu A10i^S|i^S p\nu)
!         + (\mu i^L|A10i^Sp\nu) + (p\mu i^S|A10i^S p\nu)
!         + (\mu i^L|i^S A10\nu) + (p\mu i^S|i^S A10\nu)
!  Density matrix *is* assumed to be *Hermitian*
!

!**************************************************
!  J
!    (A10i^S i^S|\mu\nu)   + (i^S A10i^S|\mu\nu)
!  + (A10i^S i^S|p\mu p\nu) + (i^S A10i^S|p\mu p\nu) = 0
! +J^SS
!  = (i^L i^L|A10\mu p\nu) + (i^L i^L|p\mu A10\nu)
!  + (i^S i^S|A10\mu p\nu) + (i^S i^S|p\mu A10\nu)
!  = (ii|A10\mu p\nu) + h.c.
!  = D_lk*(Ai pj|kl) + h.c.
!  K
!  = (A10\mu i^S|i^L \nu) + (A10\mu i^S|i^S p\nu)
!  + (p\mu A10i^S|i^L \nu) + (p\mu A10i^S|i^S p\nu)
!  + (\mu i^L|A10i^Sp\nu) + (p\mu i^S|A10i^S p\nu)
!  + (\mu i^L|i^S A10\nu) + (p\mu i^S|i^S A10\nu)
!  K^SL = (A10\mu i^S|i^L \nu) + (p\mu A10i^S|i^L \nu)
!    = D_jk^SL*(Ai pj|kl) + D_jk^SL*(pi Aj|kl)
!  K^SS = (A10\mu i^S|i^S p\nu) + (p\mu A10i^S|i^S p\nu)
!       + (p\mu i^S|i^S A10\nu) + (p\mu i^S|A10i^S p\nu)
!    = (A10\mu i^S|i^S p\nu) + h.c. + (p\mu A10i^S|i^S p\nu) + h.c.
!    =[D_jk^SS*(Ai pj|kl) + D_jk^SS*(pi Aj|kl)] + h.c.
!  K^LS = (\mu i^L|A10i^S p\nu) + (\mu i^L|i^S A10\nu)
!    = D_jk^LS*(ij|Ak pl) + D_jk^LS*(ij|pk Al)
!    = K^SL\dagger
!**************************************************

!subroutine rmb_vhf_SSLL_O1(intor, tdm, vj, vk, &
!                           n4c, nset, nset_dm, pair_kl, ao_loc, &
!                           atm, natm, bas, nbas, env, opt)
!  use cint_const_mod
!  use cint_interface
!  use hf_block_mod
!  implicit none
!
!  external                      ::  intor
!  integer,intent(in)            ::  n4c, nset, nset_dm
!  double complex,intent(in)     ::  tdm(n4c,n4c)
!  double complex,intent(inout)  ::  vj(n4c,n4c,3)
!  double complex,intent(inout)  ::  vk(n4c,n4c,3)
!  integer,intent(in)            ::  pair_kl
!  integer,intent(in)            ::  ao_loc(*)
!  integer,intent(in)            ::  natm, nbas
!  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
!  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
!  double precision,intent(in)   ::  env(*)
!
!  integer                       ::  ishell, jshell, kshell, lshell
!  integer                       ::  di, dj, dk, dl, is, s
!  integer                       ::  shls(4), tijshls(4)
!  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)
!
!  s = n4c / 2 + 1
!  lshell = pair_kl / nbas
!  kshell = pair_kl - lshell * nbas
!  if (kshell > lshell) then
!    return
!  end if
!
!  dk = CINTcgto_spinor(kshell, bas)
!  dl = CINTcgto_spinor(lshell, bas)
!
!  do jshell = 0, nbas - 1
!    dj = CINTcgto_spinor(jshell, bas)
!
!    do ishell = 0, nbas - 1
!      di = CINTcgto_spinor(ishell, bas)
!      shls = (/ishell, jshell, kshell, lshell/)
!      tijshls = (/jshell, ishell, kshell, lshell/)
!
!      allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
!      call intor(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
!      do is = 1, 3
!        call HFBLK_ij_O1(fijkl(:,:,:,:,is), tdm(1,1), vj(s,s,is), n4c, shls, ao_loc, ao_loc(nbas+1))
!        ! no _kl_O* because 1st order density vanishes
!        call HFBLK_il_O1(fijkl(:,:,:,:,is), tdm(s,1), vk(s,1,is), n4c, shls, ao_loc, ao_loc(nbas+1))
!        ! (sp sA10| )
!        call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
!        call HFBLK_il_O1(tijkl, tdm(s,1), vk(s,1,is), n4c, tijshls, ao_loc, ao_loc(nbas+1))
!      end do
!      deallocate (fijkl, tijkl)
!    end do
!  end do
!  return
!end subroutine rmb_vhf_SSLL_O1
!
!subroutine rmb_vhf_SSSS_O1(intor, tdm, vj, vk, &
!                           n4c, nset, nset_dm, pair_kl, ao_loc, &
!                           atm, natm, bas, nbas, env, opt)
!  use cint_const_mod
!  use cint_interface
!  use hf_block_mod
!  implicit none
!
!  external                      ::  intor
!  integer,intent(in)            ::  n4c, nset, nset_dm
!  double complex,intent(in)     ::  tdm(n4c,n4c)
!  double complex,intent(inout)  ::  vj(n4c,n4c,3)
!  double complex,intent(inout)  ::  vk(n4c,n4c,3)
!  integer,intent(in)            ::  pair_kl
!  integer,intent(in)            ::  ao_loc(*)
!  integer,intent(in)            ::  natm, nbas
!  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
!  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
!  double precision,intent(in)   ::  env(*)
!
!  integer                       ::  ishell, jshell, kshell, lshell
!  integer                       ::  di, dj, dk, dl, is, s
!  integer                       ::  shls(4), tijshls(4)
!  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)
!
!  s = n4c / 2 + 1
!  lshell = pair_kl / nbas
!  kshell = pair_kl - lshell * nbas
!  if (kshell > lshell) then
!    return
!  end if
!
!  dk = CINTcgto_spinor(kshell, bas)
!  dl = CINTcgto_spinor(lshell, bas)
!
!  do jshell = 0, nbas - 1
!    dj = CINTcgto_spinor(jshell, bas)
!
!    do ishell = 0, nbas - 1
!      di = CINTcgto_spinor(ishell, bas)
!      shls = (/ishell, jshell, kshell, lshell/)
!      tijshls = (/jshell, ishell, kshell, lshell/)
!
!      allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
!      call intor(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
!      do is = 1, 3
!        call HFBLK_ij_O1(fijkl(:,:,:,:,is), tdm, vj(s,s,is), n4c, shls, ao_loc, ao_loc(nbas+1))
!        ! no _kl_O* because 1st order density vanishes
!        call HFBLK_il_O1(fijkl(:,:,:,:,is), tdm, vk(s,s,is), n4c, shls, ao_loc, ao_loc(nbas+1))
!        ! (sp sA10| )
!        call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
!        call HFBLK_il_O1(tijkl, tdm, vk(s,s,is), n4c, tijshls, ao_loc, ao_loc(nbas+1))
!      end do
!      deallocate (fijkl, tijkl)
!    end do
!  end do
!  return
!end subroutine rmb_vhf_SSSS_O1
!
!subroutine rmb_vhf_after(vj, vk, n4c, nset, nset_dm, atm, natm, bas, nbas, env)
!  use cint_const_mod
!  implicit none
!  integer,intent(in)            ::  n4c, nset, nset_dm
!  double complex,intent(inout)  ::  vj(n4c,n4c,3)
!  double complex,intent(inout)  ::  vk(n4c,n4c,3)
!  integer,intent(in)            ::  natm, nbas
!  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
!  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
!  double precision,intent(in)   ::  env(*)
!
!  integer                       ::  i, j, n2c, is
!  double complex                ::  tmp
!
!  n2c = n4c / 2
!
!  vj(1 :n2c,1 :n2c,:) = 0d0
!  vj(1 :n2c,n2c+1:,:) = 0d0
!  vj(n2c+1:,1 :n2c,:) = 0d0
!  do is = 1, 3
!    do j = n2c+1, n4c
!      do i = n2c+1, j
!        tmp = (vj(i,j,is) + conjg(vj(j,i,is))) * (.5d0/env(PTR_LIGHT_SPEED))**2
!        vj(i,j,is) = tmp
!        vj(j,i,is) = conjg(tmp)
!      end do
!    end do
!  end do
!
!  vk(:n2c,:n2c,:) = 0d0
!  do is = 1, 3
!    do j = 1, n2c
!      do i = n2c+1, n4c
!        tmp = vk(i,j,is) * (.5d0/env(PTR_LIGHT_SPEED))
!        vk(i,j,is) = tmp
!        vk(j,i,is) = conjg(tmp)
!      end do
!    end do
!    do j = n2c+1, n4c
!      do i = n2c+1, j
!        tmp = (vk(i,j,is) + conjg(vk(j,i,is))) * (.5d0/env(PTR_LIGHT_SPEED))**2
!        vk(i,j,is) = tmp
!        vk(j,i,is) = conjg(tmp)
!      end do
!    end do
!  end do
!  return
!end subroutine rmb_vhf_after

! If call tas_ts_dm12, then do K + c.c., J + c.c.,
! otherwise using rkb_vhf_after to scale the SS block
subroutine rmb_vhf_after(vj, vk, n4c, nset, nset_dm, atm, natm, bas, nbas, env)
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
    call zvpluscc_inplace(vj(1,1,is), n4c, n4c)
    call zvpluscc_inplace(vk(1,1,is), n4c, n4c)
  end do
  return
end subroutine rmb_vhf_after

subroutine rmb_vhf_gaunt_iter(intor, fscreen, dm, vj, vk, &
                              n4c, nset, nset_dm, pair_kl, ao_loc, &
                              atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(in)     ::  dm(n4c,n4c)
  double complex,intent(inout)  ::  vj(n4c,n4c,3)
  double complex,intent(inout)  ::  vk(n4c,n4c,3)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, s
  integer                       ::  shls(4), tijshls(4), tklshls(4), ttshls(4)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
  double complex,allocatable    ::  tijkl(:,:,:,:)
  double complex,allocatable    ::  ijtkl(:,:,:,:)
  double complex,allocatable    ::  tijtkl(:,:,:,:)

  s = n4c / 2 + 1
  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas

  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    do ishell = 0, nbas - 1
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)
      tklshls = (/ishell, jshell, lshell, kshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl), ijtkl(di,dj,dl,dk))
  if (0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    do is = 1, 3
      call HFBLK_ij_O0(fijkl(:,:,:,:,is), dm(s,1), vj(1,s,is), n4c, shls, ao_loc)
      call HFBLK_kl_O0(fijkl(:,:,:,:,is), dm(s,1), vj(1,s,is), n4c, shls, ao_loc)
      call HFBLK_il_O0(fijkl(:,:,:,:,is), dm(s,1), vk(1,s,is), n4c, shls, ao_loc)
      call HFBLK_kj_O0(fijkl(:,:,:,:,is), dm(s,1), vk(1,s,is), n4c, shls, ao_loc)

      call HFBLK_timerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
      call HFBLK_kl_O0(tijkl, dm(1,s), vj(1,s,is), n4c, tijshls, ao_loc)
      call HFBLK_il_O0(tijkl, dm(1,1), vk(s,s,is), n4c, tijshls, ao_loc)
      call HFBLK_kj_O0(tijkl, dm(s,s), vk(1,1,is), n4c, tijshls, ao_loc)

      call HFBLK_atimerev_ijtltk(fijkl(:,:,:,:,is), ijtkl, shls, ao_loc, ao_loc(nbas+1))
      call HFBLK_ij_O0(ijtkl, dm(1,s), vj(1,s,is), n4c, tklshls, ao_loc)
      call HFBLK_il_O0(ijtkl, dm(s,s), vk(1,1,is), n4c, tklshls, ao_loc)
      call HFBLK_kj_O0(ijtkl, dm(1,1), vk(s,s,is), n4c, tklshls, ao_loc)
    end do
  end if
  deallocate (fijkl, tijkl, ijtkl)
    end do
  end do
end subroutine rmb_vhf_gaunt_iter

subroutine rmb_vhf_gaunt_after(vj, vk, n4c, nset, nset_dm, &
                               atm, natm, bas, nbas, env)
  use cint_const_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(inout)  ::  vj(n4c,n4c,nset*nset_dm)
  double complex,intent(inout)  ::  vk(n4c,n4c,nset*nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  !call rkb_vhf_gaunt_after(vj, vk, n4c, nset, nset_dm, atm, natm, bas, nbas, env)

  integer                       ::  i, j, n2c

  n2c = n4c / 2
  vj(  :n2c,n2c+1:,:) = vj(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  do j = 1, n2c
    do i = n2c+1, n4c
      vj(i,j,:) = conjg(vj(j,i,:))
      vk(i,j,:) = conjg(vk(j,i,:))
    end do
  end do
end subroutine rmb_vhf_gaunt_after
