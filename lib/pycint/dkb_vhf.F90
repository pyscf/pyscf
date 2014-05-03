!
! File dkb_vhf_coul.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!

#define QLL     (1)
#define QLP     (2)
#define QPL     (3)
#define QPP     (4)

subroutine dkb_vhf_pre(tdm, dm, n4c, nset, nset_dm, &
                       atm, natm, bas, nbas, env)
  use cint_const_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(out)    ::  tdm(n4c,n4c,nset_dm)
  double complex,intent(in)     ::  dm(n4c,n4c,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  tdm = dm
end subroutine dkb_vhf_pre

subroutine dkb_vhf_pre_and_screen(tdm, dm, n4c, nset, nset_dm, &
                                  atm, natm, bas, nbas, env)
  use cint_const_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(out)    ::  tdm(n4c,n4c,nset_dm)
  double complex,intent(in)     ::  dm(n4c,n4c,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  tdm = dm
  call dkb_vhf_init_screen(dm, n4c, nset, nset_dm, &
                           atm, natm, bas, nbas, env)
end subroutine dkb_vhf_pre_and_screen

subroutine dkb_vhf_after(vj, vk, n4c, nset, nset_dm, &
                         atm, natm, bas, nbas, env)
  call dkb_vhf_del_screen()
end subroutine dkb_vhf_after

subroutine init_dkb_direct_scf(atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer,external              ::  cint2e, cint2e_spsp1spsp2, &
                                    cint2e_spv1spv2, cint2e_vsp1vsp2
  integer                       ::  i, j, iloc, jloc, di, dj
  integer                       ::  shls(4)
  integer                       ::  ao_loc(nbas)
  double complex,allocatable    ::  fijij(:,:,:,:)

  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  if (allocated(HFBLK_q_cond)) then
    deallocate (HFBLK_q_cond)
  end if
  allocate (HFBLK_q_cond(nbas,nbas,4))
  HFBLK_q_cond = 0
  do j = 1, nbas
    jloc = ao_loc(j)
    dj = CINTcgto_spinor(j-1, bas)
    do i = 1, j
      iloc = ao_loc(i)
      di = CINTcgto_spinor(i-1, bas)
      allocate (fijij(di,dj,di,dj))
      shls = (/i-1, j-1, i-1, j-1/)

      if (0 /= cint2e(fijij, shls, atm, natm, bas, nbas, env, 0_8)) then
        HFBLK_q_cond(i,j,QLL) = sqrt(max_ijij(fijij, di, dj))
      end if

      if (0 /= cint2e_spsp1spsp2(fijij, shls, atm, natm, bas, nbas, env, 0_8)) then
        HFBLK_q_cond(i,j,QPP) = \
                sqrt(max_ijij(fijij, di, dj)) * (.5d0/env(PTR_LIGHT_SPEED))**2
      end if

      if (0 /= cint2e_vsp1vsp2(fijij, shls, atm, natm, bas, nbas, env, 0_8)) then
        HFBLK_q_cond(i,j,QLP) = \
                sqrt(max_ijij(fijij, di, dj)) * (.5d0/env(PTR_LIGHT_SPEED))
      end if

      if (0 /= cint2e_spv1spv2(fijij, shls, atm, natm, bas, nbas, env, 0_8)) then
        HFBLK_q_cond(i,j,QPL) = \
                sqrt(max_ijij(fijij, di, dj)) * (.5d0/env(PTR_LIGHT_SPEED))
      end if
      deallocate (fijij)
      HFBLK_q_cond(j,i,QLL) = HFBLK_q_cond(i,j,QLL)
      HFBLK_q_cond(j,i,QLP) = HFBLK_q_cond(i,j,QLP)
      HFBLK_q_cond(j,i,QPL) = HFBLK_q_cond(i,j,QPL)
      HFBLK_q_cond(j,i,QPP) = HFBLK_q_cond(i,j,QPP)
    end do
  end do

  call set_direct_scf_cutoff(1d-12)

contains
double precision function max_ijij(fijij, di, dj)
  double complex        ::  fijij(:,:,:,:)
  integer               ::  i0, j0, di, dj
  max_ijij = 0
  do j0 = 1, dj
    do i0 = 1, di
      max_ijij = max(max_ijij, abs(fijij(i0,j0,i0,j0)))
    end do
  end do
end function max_ijij
end subroutine init_dkb_direct_scf

subroutine del_dkb_direct_scf()
  use hf_block_mod
  if (allocated(HFBLK_q_cond)) then
    deallocate (HFBLK_q_cond)
  end if
  call turnoff_direct_scf()
end subroutine del_dkb_direct_scf

subroutine dkb_vhf_init_screen(dm, n4c, nset, nset_dm, &
                               atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(in)     ::  dm(n4c,n4c,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer                       ::  i, j, iloc, jloc, di, dj, id
  integer                       ::  ao_loc(nbas)

  if (HFBLK_direct_scf_cutoff < 0) then
    return
  end if

  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  if (allocated(HFBLK_dm_cond)) then
    deallocate (HFBLK_dm_cond)
  end if
  allocate (HFBLK_dm_cond(nbas,nbas,nset_dm))
  do j = 1, nbas
    jloc = ao_loc(j)
    dj = CINTcgto_spinor(j-1, bas)
    do i = 1, nbas
      iloc = ao_loc(i)
      di = CINTcgto_spinor(i-1, bas)
      do id = 1, nset_dm
        HFBLK_dm_cond(i,j,id) = maxval(abs(dm(iloc+1:iloc+di,jloc+1:jloc+dj,id)))
      end do
    end do
  end do
end subroutine dkb_vhf_init_screen

subroutine dkb_vhf_del_screen()
  use hf_block_mod
  if (allocated(HFBLK_dm_cond)) then
    deallocate (HFBLK_dm_cond)
  end if
end subroutine dkb_vhf_del_screen

!**************************************************
subroutine dkb_vhf_coul_iter(intor, fscreen, dm, vj, vk, &
                             n4c, nset, nset_dm, pair_kl, ao_loc, &
                             atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(in)     ::  dm(n4c,n4c,nset_dm)
  double complex,intent(inout)  ::  vj(n4c,n4c,nset,nset_dm)
  double complex,intent(inout)  ::  vk(n4c,n4c,nset,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*) ! ao_loc(nbas+1) = ao_loc(nbas+1:)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, i, j, k, l
  integer                       ::  shls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  integer                       ::  nbasL, n2c
  double complex,allocatable    ::  fijkl(:,:,:,:)
  integer,external              ::  cint2e, cint2e_spsp1, &
                                    cint2e_spsp2, cint2e_spsp1spsp2, &
                                    cint2e_spv1, cint2e_vsp1, &
                                    cint2e_spv1spsp2, cint2e_vsp1spsp2, &
                                    cint2e_spv1spv2, cint2e_vsp1spv2, &
                                    cint2e_spv1vsp2, cint2e_vsp1vsp2

  n2c = n4c / 2

  ! determine the number of large component basis sets
  nbasL = 0
  do i = 1, nbas
    if (ao_loc(i) == n2c) then
      nbasL = i - 1
      exit
    end if
  end do

  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  if (lshell < nbasL .and. kshell <= lshell) then
    do jshell = 0, lshell
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = 0, jshell
        if (jshell == lshell .and. ishell > kshell) then
          cycle
        end if
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QLL, QLL) .and. &
      0 /= cint2e(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QPP, QPP) .and. &
      0 /= cint2e_spsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**4
    if (0 /= do_vj(1)) then
      call HFBLK_kl_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do

    do jshell = 0, nbasL - 1
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = 0, jshell
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QPP, QLL) .and. &
      0 /= cint2e_spsp1(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O2(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
      call HFBLK_kl_O2(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O2(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
      call HFBLK_kj_O2(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do

    do jshell = 0, nbasL - 1
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = nbasL, nbas - 1
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QPL, QLL) .and. &
      0 /= cint2e_spv1(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QLP, QLL) .and. &
      0 /= cint2e_vsp1(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QPL, QPP) .and. &
      0 /= cint2e_spv1spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))**3
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QLP, QPP) .and. &
      0 /= cint2e_vsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**3
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do

    do jshell = nbasL, nbas - 1
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = nbasL, jshell
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QLL, QLL) .and. &
      0 /= cint2e(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QLL, QPP) .and. &
      0 /= cint2e_spsp1(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QPP, QLL) .and. &
      0 /= cint2e_spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QPP, QPP) .and. &
      0 /= cint2e_spsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**4
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do

  else if (lshell < nbasL .and. nbasL <= kshell) then

    do jshell = 0, lshell
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = nbasL, nbas - 1
        if (jshell == lshell .and. ishell > kshell) then
          cycle
        end if
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QPL, QPL) .and. &
      0 /= cint2e_spv1spv2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QLP, QPL) .and. &
      0 /= cint2e_vsp1spv2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (-(.5d0/env(PTR_LIGHT_SPEED))**2)
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QPL, QLP) .and. &
      0 /= cint2e_spv1vsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (-(.5d0/env(PTR_LIGHT_SPEED))**2)
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QLP, QLP) .and. &
      0 /= cint2e_vsp1vsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do

  else if (nbasL <= kshell .and. kshell <= lshell ) then

    do jshell = 0, nbasL - 1
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = nbasL, nbas - 1
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QPL, QLL) .and. &
      0 /= cint2e_spv1(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QLP, QLL) .and. &
      0 /= cint2e_vsp1(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QPL, QPP) .and. &
      0 /= cint2e_spv1spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))**3
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QLP, QPP) .and. &
      0 /= cint2e_vsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**3
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do

    do jshell = nbasL, lshell
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = nbasL, jshell
        if (jshell == lshell .and. ishell > kshell) then
          cycle
        end if
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QLL, QLL) .and. &
      0 /= cint2e(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if

  if (0 /= prescreen(shls, QPP, QPP) .and. &
      0 /= cint2e_spsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**4
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do

    do jshell = nbasL, nbas - 1
      dj = CINTcgto_spinor(jshell, bas)
      do ishell = nbasL, jshell
        di = CINTcgto_spinor(ishell, bas)
        shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  if (0 /= prescreen(shls, QPP, QLL) .and. &
      0 /= cint2e_spsp1(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
    fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
    if (0 /= do_vj(1)) then
      call HFBLK_ij_O2(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
      call HFBLK_kl_O2(fijkl, dm, vj, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
    if (0 /= do_vk(1)) then
      call HFBLK_il_O2(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
      call HFBLK_kj_O2(fijkl, dm, vk, n4c, shls, ao_loc, ao_loc(nbas+1))
    end if
  end if
  deallocate (fijkl)
      end do
    end do
  end if

contains
integer function prescreen(shls, qij, qkl)
  integer               ::  shls(4), qij, qkl
  integer               ::  id
  double precision      ::  dm_max, qijkl
  integer,external      ::  no_screen

  if (HFBLK_direct_scf_cutoff < 0) then
    prescreen = no_screen(shls, do_vj, do_vk, nset_dm)
  else
    i = shls(1) + 1
    j = shls(2) + 1
    k = shls(3) + 1
    l = shls(4) + 1
    prescreen = 0
    qijkl = HFBLK_q_cond(i,j,qij) * HFBLK_q_cond(i,j,qkl)
    do id = 1, nset_dm
      dm_max = max(HFBLK_dm_cond(j,i,id), HFBLK_dm_cond(l,k,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vj(id) = 0
      else
        do_vj(id) = 1
        prescreen = 1
      end if
      dm_max = max(HFBLK_dm_cond(j,k,id), HFBLK_dm_cond(j,l,id), &
                   HFBLK_dm_cond(i,k,id), HFBLK_dm_cond(i,l,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vk(id) = 0
      else
        do_vk(id) = 1
        prescreen = 1
      end if
    end do
  end if
end function prescreen
end subroutine dkb_vhf_coul_iter
