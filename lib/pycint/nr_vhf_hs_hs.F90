!
! File nr_vhf_hs_hs.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
!  a set of non-relativistic HF coulomb and exchange potential (in AO representation)
!       J = (ii|\mu \nu)
!       K = (\mu i|i\nu)
!
! in _O1, _O2 and _O3, the time-reversal operator is applied to electron 1
! (ij| and electron 2 |kl) to interchange the bra and ket. In these
! interchanges,
! the electron-1 operator (between (ij|) should be *Hermitian*, 
! the electron-2 operator (between |kl)) should be *Hermitian*.
!
!  nr_..._dm2  vj/vk is based on the density of electron 2
!  nr_..._dm12 vj/vk is based on the density of electron 1 and 2 (in case the
!       integral is not symmetric for electron 1 and electron 2)
!
! requirement
!  * Density matrix should be *Hermitian*
!  * The operator in intor should be *Hermitian*

subroutine nr_hs_hs_dm2_O0(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  double precision,intent(inout)::  vj(ndim,ndim,nset,nset_dm)
  double precision,intent(inout)::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(nbas)

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, is, id
  integer               ::  shls(4)
  integer               ::  do_vj(nset_dm), do_vk(nset_dm)
  double precision,allocatable  ::  fijkl(:,:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas

  dl = CINTcgto_spheric(lshell, bas)
  dk = CINTcgto_spheric(kshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spheric(jshell, bas)
    do ishell = 0, nbas - 1
      di = CINTcgto_spheric(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_nr_ij_O0(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_nr_il_O0(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
            end if
          end do
        end do
      end if
      deallocate (fijkl)
    end do
  end do
end subroutine nr_hs_hs_dm2_O0

subroutine nr_hs_hs_dm2_O1(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  double precision,intent(inout)::  vj(ndim,ndim,nset,nset_dm)
  double precision,intent(inout)::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(nbas)

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, is, id
  integer               ::  shls(4)
  integer               ::  do_vj(nset_dm), do_vk(nset_dm)
  double precision,allocatable  ::  fijkl(:,:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dl = CINTcgto_spheric(lshell, bas)
  dk = CINTcgto_spheric(kshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spheric(jshell, bas)
    do ishell = 0, nbas - 1
      di = CINTcgto_spheric(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_nr_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_nr_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
            end if
          end do
        end do
      end if
      deallocate (fijkl)
    end do
  end do
end subroutine nr_hs_hs_dm2_O1

subroutine nr_hs_hs_dm2_O2(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  double precision,intent(inout)::  vj(ndim,ndim,nset,nset_dm)
  double precision,intent(inout)::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(nbas)

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, is, id
  integer               ::  shls(4)
  integer               ::  do_vj(nset_dm), do_vk(nset_dm)
  double precision,allocatable  ::  fijkl(:,:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dl = CINTcgto_spheric(lshell, bas)
  dk = CINTcgto_spheric(kshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spheric(jshell, bas)
    do ishell = 0, jshell
      di = CINTcgto_spheric(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_nr_ij_O2(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_nr_il_O2(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
            end if
          end do
        end do
      end if
      deallocate (fijkl)
    end do
  end do
end subroutine nr_hs_hs_dm2_O2

subroutine nr_hs_hs_dm2_O3(intor, fscreen, dm, vj, vk, &
                           ndim, nset, nset_dm, pair_kl, ao_loc, &
                           atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  double precision,intent(inout)::  vj(ndim,ndim,nset,nset_dm)
  double precision,intent(inout)::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(nbas)

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, is, id
  integer               ::  shls(4)
  integer               ::  do_vj(nset_dm), do_vk(nset_dm)
  double precision,allocatable  ::  fijkl(:,:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dl = CINTcgto_spheric(lshell, bas)
  dk = CINTcgto_spheric(kshell, bas)

  do jshell = 0, lshell
    dj = CINTcgto_spheric(jshell, bas)

    do ishell = 0, jshell
      if (jshell == lshell .and. ishell > kshell) then
        cycle
      end if

      di = CINTcgto_spheric(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_nr_ij_O3(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_nr_il_O3(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
            end if
          end do
        end do
      end if
      deallocate (fijkl)
    end do
  end do
end subroutine nr_hs_hs_dm2_O3
! ************************************************


! ************************************************
subroutine nr_hs_hs_dm12_O0(intor, fscreen, dm, vj, vk, &
                            ndim, nset, nset_dm, pair_kl, ao_loc, &
                            atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  double precision,intent(inout)::  vj(ndim,ndim,nset,nset_dm)
  double precision,intent(inout)::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(nbas)

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, is, id
  integer               ::  shls(4)
  integer               ::  do_vj(nset_dm), do_vk(nset_dm)
  double precision,allocatable  ::  fijkl(:,:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas

  dl = CINTcgto_spheric(lshell, bas)
  dk = CINTcgto_spheric(kshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spheric(jshell, bas)
    do ishell = 0, nbas - 1
      di = CINTcgto_spheric(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_nr_ij_O0(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
              call HFBLK_nr_kl_O0(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_nr_il_O0(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
              call HFBLK_nr_kj_O0(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
            end if
          end do
        end do
      end if
      deallocate (fijkl)
    end do
  end do
end subroutine nr_hs_hs_dm12_O0

subroutine nr_hs_hs_dm12_O1(intor, fscreen, dm, vj, vk, &
                             ndim, nset, nset_dm, pair_kl, ao_loc, &
                             atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  double precision,intent(inout)::  vj(ndim,ndim,nset,nset_dm)
  double precision,intent(inout)::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(nbas)

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, is, id
  integer               ::  shls(4)
  integer               ::  do_vj(nset_dm), do_vk(nset_dm)
  double precision,allocatable  ::  fijkl(:,:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dl = CINTcgto_spheric(lshell, bas)
  dk = CINTcgto_spheric(kshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spheric(jshell, bas)
    do ishell = 0, nbas - 1
      di = CINTcgto_spheric(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_nr_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
              call HFBLK_nr_kl_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_nr_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
              call HFBLK_nr_kj_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
            end if
          end do
        end do
      end if
      deallocate (fijkl)
    end do
  end do
end subroutine nr_hs_hs_dm12_O1

subroutine nr_hs_hs_dm12_O2(intor, fscreen, dm, vj, vk, &
                             ndim, nset, nset_dm, pair_kl, ao_loc, &
                             atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  double precision,intent(inout)::  vj(ndim,ndim,nset,nset_dm)
  double precision,intent(inout)::  vk(ndim,ndim,nset,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(nbas)

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, is, id
  integer               ::  shls(4), tijshls(4)
  integer               ::  do_vj(nset_dm), do_vk(nset_dm)
  double precision,allocatable  ::  fijkl(:,:,:,:,:), fjikl(:,:,:,:)

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dl = CINTcgto_spheric(lshell, bas)
  dk = CINTcgto_spheric(kshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spheric(jshell, bas)
    do ishell = 0, jshell - 1
      di = CINTcgto_spheric(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset), fjikl(dj,di,dk,dl))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          call HFBLK_nr_swap_ij(fijkl(:,:,:,:,is), fjikl, shls)
          call dscal(size(fjikl), -1d0, fjikl, 1)
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_nr_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
              call HFBLK_nr_ij_O1(fjikl, dm(1,1,id), vj(1,1,is,id), ndim, tijshls, ao_loc)
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_nr_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
              call HFBLK_nr_kj_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
              call HFBLK_nr_il_O1(fjikl, dm(1,1,id), vk(1,1,is,id), ndim, tijshls, ao_loc)
              call HFBLK_nr_kj_O1(fjikl, dm(1,1,id), vk(1,1,is,id), ndim, tijshls, ao_loc)
            end if
          end do
        end do
      end if
      deallocate (fijkl, fjikl)
    end do

    shls = (/jshell, jshell, kshell, lshell/)
    allocate (fijkl(dj,dj,dk,dl,nset))
    if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
        0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
      do is = 1, nset
        do id = 1, nset_dm
          if (0 /= do_vj(id)) then
            call HFBLK_nr_ij_O1(fijkl(:,:,:,:,is), dm(1,1,id), vj(1,1,is,id), ndim, shls, ao_loc)
          end if
          if (0 /= do_vk(id)) then
            call HFBLK_nr_il_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
            call HFBLK_nr_kj_O1(fijkl(:,:,:,:,is), dm(1,1,id), vk(1,1,is,id), ndim, shls, ao_loc)
          end if
        end do
      end do
    end if
    deallocate (fijkl)
  end do
end subroutine nr_hs_hs_dm12_O2
