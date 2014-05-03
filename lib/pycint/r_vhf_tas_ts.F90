!
! File rkb_vhf_coul.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
! in _O1, _O2 and _O3, the time-reversal operator is applied to electron 1 (ij|
! and electron 2 |kl) to interchange the bra and ket. In these interchanges,
! the electron-1 operator (between (ij|) should be *time-reversal anti-symmetric*, 
! the electron-2 operator (between |kl)) should be *time-reversal symmetric*, 
!
!  LL_LL can be either 2C or 4C
!  LL_SS, SS_LL and SS_SS are 4C
!  r_..._dm2  vj/vk is based on the density of electron 2
!  r_..._dm12 vj/vk is based on the density of e_1 + density of e_2
!
! requirement
!  Density matrix should be *Hermitian*

! ************************************************
! nset_dm of (dm => nset of vj/vk)
subroutine r_tasLL_tsLL_dm2_O0(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsLL_dm2_O0(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsLL_dm2_O0

subroutine r_tasLL_tsLL_dm2_O1(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsLL_dm2_O1(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsLL_dm2_O1

subroutine r_tasLL_tsLL_dm2_O2(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*) ! tao = ao_loc(nbas+1:)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, id
  integer                       ::  shls(4), tijshls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    ! avoid double counting if ishell == jshell
    do ishell = 0, jshell-1
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset), tijkl(dj,di,dk,dl))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_ij_O1(tijkl, dm(1,1,id), vj(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(tijkl, dm(1,1,id), vk(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
          end do
        end do
      end if
      deallocate (fijkl, tijkl)
    end do

    shls = (/jshell, jshell, kshell, lshell/)
    allocate (fijkl(dj,dj,dk,dl,nset))
    if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
        0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
      do is = 1, nset
        do id = 1, nset_dm
          if (0 /= do_vj(id)) then
            call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
          if (0 /= do_vk(id)) then
            call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
        end do
      end do
    end if
    deallocate (fijkl)
  end do
end subroutine r_tasLL_tsLL_dm2_O2

! ************************************************
subroutine r_tasSS_tsSS_dm2_O0(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsSS_dm2_O0(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsSS_dm2_O0

subroutine r_tasSS_tsSS_dm2_O1(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsSS_dm2_O1(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsSS_dm2_O1

subroutine r_tasSS_tsSS_dm2_O2(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer                       ::  s
  s = ndim / 2 + 1
  call r_tasLL_tsLL_dm2_O2(intor, fscreen, dm(s,s,1), &
                           vj(s,s,1,1), vk(s,s,1,1), &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsSS_dm2_O2

! ************************************************
subroutine r_tasSS_tsLL_dm2_O0(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsLL_dm2_O0(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsLL_dm2_O0

subroutine r_tasSS_tsLL_dm2_O1(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsLL_dm2_O1(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsLL_dm2_O1

subroutine r_tasSS_tsLL_dm2_O2(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, id
  integer                       ::  shls(4), tijshls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  integer                       ::  s
  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  s = ndim / 2 + 1

  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    ! avoid double counting if ishell == jshell
    do ishell = 0, jshell-1
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset), tijkl(dj,di,dk,dl))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(s,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(s,1,id), vk(s,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_ij_O1(tijkl, dm(1,1,id), vj(s,s,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(tijkl, dm(s,1,id), vk(s,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
          end do
        end do
      end if
      deallocate (fijkl, tijkl)
    end do

    shls = (/jshell, jshell, kshell, lshell/)
    allocate (fijkl(dj,dj,dk,dl,nset))
    if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
        0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
      do is = 1, nset
        do id = 1, nset_dm
          if (0 /= do_vj(id)) then
            call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(s,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
          if (0 /= do_vk(id)) then
            call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(s,1,id), vk(s,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
        end do
      end do
    end if
    deallocate (fijkl)
  end do
end subroutine r_tasSS_tsLL_dm2_O2

! ************************************************
subroutine r_tasLL_tsSS_dm2_O0(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsSS_dm2_O0(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsSS_dm2_O0

subroutine r_tasLL_tsSS_dm2_O1(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsSS_dm2_O1(intor, fscreen, dm, vj, vk, &
                          ndim, nset, nset_dm, pair_kl, ao_loc, &
                          atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsSS_dm2_O1

subroutine r_tasLL_tsSS_dm2_O2(intor, fscreen, dm, vj, vk, &
                               ndim, nset, nset_dm, pair_kl, ao_loc, &
                               atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, id
  integer                       ::  shls(4), tijshls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  integer                       ::  s
  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  s = ndim / 2 + 1

  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    ! avoid double counting if ishell == jshell
    do ishell = 0, jshell-1
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset), tijkl(dj,di,dk,dl))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(s,s,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,s,id), vk(1,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_ij_O1(tijkl, dm(s,s,id), vj(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(tijkl, dm(1,s,id), vk(1,s,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
          end do
        end do
      end if
      deallocate (fijkl, tijkl)
    end do

    shls = (/jshell, jshell, kshell, lshell/)
    allocate (fijkl(dj,dj,dk,dl,nset))
    if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
        0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
      do is = 1, nset
        do id = 1, nset_dm
          if (0 /= do_vj(id)) then
            call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(s,s,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
          if (0 /= do_vk(id)) then
            call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,s,id), vk(1,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
        end do
      end do
    end if
    deallocate (fijkl)
  end do
end subroutine r_tasLL_tsSS_dm2_O2
! ************************************************


! ************************************************
! FIXME SSLL and LLSS
subroutine r_tasLL_tsLL_dm12_O0(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsLL_dm12_O0(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsLL_dm12_O0

subroutine r_tasLL_tsLL_dm12_O1(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsLL_dm12_O1(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsLL_dm12_O1

subroutine r_tasLL_tsLL_dm12_O2(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*) ! tao = ao_loc(nbas+1:)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, id
  integer                       ::  shls(4), tijshls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    ! avoid double counting if ishell == jshell
    do ishell = 0, jshell-1
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset), tijkl(dj,di,dk,dl))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_ij_O1(tijkl, dm(1,1,id), vj(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              ! There would be no HFBLK_kl_* in closed shell sys. The density
              ! vanishes due to the anti time-reversal symm.
              call HFBLK_kl_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kl_O1(tijkl, dm(1,1,id), vj(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kj_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(tijkl, dm(1,1,id), vk(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kj_O1(tijkl, dm(1,1,id), vk(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
          end do
        end do
      end if
      deallocate (fijkl, tijkl)
    end do

    shls = (/jshell, jshell, kshell, lshell/)
    allocate (fijkl(dj,dj,dk,dl,nset))
    if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
        0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
      do is = 1, nset
        do id = 1, nset_dm
          if (0 /= do_vj(id)) then
            call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
          if (0 /= do_vk(id)) then
            call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
            call HFBLK_kj_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
        end do
      end do
    end if
    deallocate (fijkl)
  end do
end subroutine r_tasLL_tsLL_dm12_O2

! ************************************************
subroutine r_tasSS_tsSS_dm12_O0(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsSS_dm12_O0(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsSS_dm12_O0

subroutine r_tasSS_tsSS_dm12_O1(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsSS_dm12_O1(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsSS_dm12_O1

subroutine r_tasSS_tsSS_dm12_O2(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer                       ::  s
  s = ndim / 2 + 1
  call r_tasLL_tsLL_dm12_O2(intor, fscreen, dm(s,s,1), &
                            vj(s,s,1,1), vk(s,s,1,1), &
                            ndim, nset, nset_dm, pair_kl, ao_loc, &
                            atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsSS_dm12_O2

! ************************************************
subroutine r_tasSS_tsLL_dm12_O0(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsLL_dm12_O0(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsLL_dm12_O0

subroutine r_tasSS_tsLL_dm12_O1(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsSS_tsLL_dm12_O1(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasSS_tsLL_dm12_O1

subroutine r_tasSS_tsLL_dm12_O2(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, id, s
  integer                       ::  shls(4), tijshls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  s = ndim / 2 + 1
  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    do ishell = 0, jshell-1
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset), tijkl(dj,di,dk,dl))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(s,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kl_O1(fijkl(:,:,:,:,is), dm(s,s,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_ij_O1(tijkl, dm(1,1,id), vj(s,s,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kl_O1(tijkl, dm(s,s,id), vj(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(s,1,id), vk(s,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kj_O1(fijkl(:,:,:,:,is), dm(1,s,id), vk(1,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(tijkl, dm(s,1,id), vk(s,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kj_O1(tijkl, dm(1,s,id), vk(1,s,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
          end do
        end do
      end if
      deallocate (fijkl, tijkl)
    end do

    shls = (/jshell, jshell, kshell, lshell/)
    allocate (fijkl(dj,dj,dk,dl,nset))
    if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
        0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
      do is = 1, nset
        do id = 1, nset_dm
          if (0 /= do_vj(id)) then
            call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(s,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
            call HFBLK_kl_O1(fijkl(:,:,:,:,is), dm(s,s,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
          if (0 /= do_vk(id)) then
            call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(s,1,id), vk(s,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
            call HFBLK_kj_O1(fijkl(:,:,:,:,is), dm(1,s,id), vk(1,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
        end do
      end do
    end if
    deallocate (fijkl)
  end do
end subroutine r_tasSS_tsLL_dm12_O2

! ************************************************
subroutine r_tasLL_tsSS_dm12_O0(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsSS_dm12_O0(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsSS_dm12_O0

subroutine r_tasLL_tsSS_dm12_O1(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  call r_tsLL_tsSS_dm12_O1(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
end subroutine r_tasLL_tsSS_dm12_O1

subroutine r_tasLL_tsSS_dm12_O2(intor, fscreen, dm, vj, vk, &
                                ndim, nset, nset_dm, pair_kl, ao_loc, &
                                atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double complex,intent(in)     ::  dm(ndim,ndim,nset_dm)
  double complex,intent(inout)  ::  vj(ndim,ndim,nset,nset_dm)
  double complex,intent(inout)  ::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, id, s
  integer                       ::  shls(4), tijshls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  double complex,allocatable    ::  fijkl(:,:,:,:,:), tijkl(:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  s = ndim / 2 + 1
  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    do ishell = 0, jshell-1
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset), tijkl(dj,di,dk,dl))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, ao_loc(nbas+1))
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(s,s,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kl_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(s,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_ij_O1(tijkl, dm(s,s,id), vj(1,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kl_O1(tijkl, dm(1,1,id), vj(s,s,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,s,id), vk(1,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kj_O1(fijkl(:,:,:,:,is), dm(s,1,id), vk(s,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_il_O1(tijkl, dm(1,s,id), vk(1,s,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kj_O1(tijkl, dm(s,1,id), vk(s,1,is,id), ndim, tijshls, ao_loc, ao_loc(nbas+1))
            end if
          end do
        end do
      end if
      deallocate (fijkl, tijkl)
    end do

    shls = (/jshell, jshell, kshell, lshell/)
    allocate (fijkl(dj,dj,dk,dl,nset))
    if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
        0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
      do is = 1, nset
        do id = 1, nset_dm
          if (0 /= do_vj(id)) then
            call HFBLK_ij_O1(fijkl(:,:,:,:,is), dm(s,s,id), vj(1,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
            call HFBLK_kl_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(s,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
          if (0 /= do_vk(id)) then
            call HFBLK_il_O1(fijkl(:,:,:,:,is), dm(1,s,id), vk(1,s,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
            call HFBLK_kj_O1(fijkl(:,:,:,:,is), dm(s,1,id), vk(s,1,is,id), ndim, shls, ao_loc, ao_loc(nbas+1))
          end if
        end do
      end do
    end if
    deallocate (fijkl)
  end do
end subroutine r_tasLL_tsSS_dm12_O2
