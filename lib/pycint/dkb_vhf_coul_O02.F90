subroutine dkb_vhf_coul_O02(dm, vj, vk, n4c, atm, natm, bas, nbas, env)
!**************************************************
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,intent(in)            ::  n4c
  double complex,intent(in)     ::  dm(n4c,n4c)
  double complex,intent(out)    ::  vj(n4c,n4c)
  double complex,intent(out)    ::  vk(n4c,n4c)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,*)
  double precision,intent(in)   ::  env(*)

! ==========
!
!  HF potential of Dirac-Coulomb Hamiltonian (in AO representation) with DKB
!  as basis sets
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
  integer               ::  n2c
  integer               ::  shls(4)
  integer               ::  nbasL
  integer               ::  ao_loc(nbas), tao(n4c)
  double complex,allocatable    ::  fijkl(:,:,:,:)

  vj = 0d0
  vk = 0d0

  call CINTshells_spinor_offset(ao_loc, bas, nbas)
  call time_reversal_spinor(tao, bas, nbas)

  n2c = n4c / 2

  ! determine the number of large component basis sets
  nbasL = 0
  do i = 1, nbas
    if (ao_loc(i) == n2c) then
      nbasL = i - 1
      exit
    end if
  end do

  do lshell = 0, nbasL - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = 0, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, lshell
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = 0, jshell
          if (jshell == lshell .and. ishell > kshell) then
            cycle
          end if
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl))
  call cint2e(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**4
  call HFBLK_kl_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  do lshell = 0, nbasL - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = 0, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, nbasL - 1
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = 0, jshell
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl))
  call cint2e_spsp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
  call HFBLK_ij_O2(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_kl_O2(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O2(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  call HFBLK_kj_O2(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  do lshell = 0, nbasL - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = 0, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, nbasL - 1
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = nbasL, nbas - 1
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  call cint2e_spv1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_vsp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spv1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))**3
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_vsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**3
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  do lshell = 0, nbasL - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = 0, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = nbasL, nbas - 1
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = nbasL, jshell
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  call cint2e(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spsp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**4
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  do lshell = 0, nbasL - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = nbasL, nbas - 1
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, lshell
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = nbasL, nbas - 1
          if (jshell == lshell .and. ishell > kshell) then
            cycle
          end if
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  call cint2e_spv1spv2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_vsp1spv2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (-(.5d0/env(PTR_LIGHT_SPEED))**2)
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spv1vsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (-(.5d0/env(PTR_LIGHT_SPEED))**2)
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_vsp1vsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  do lshell = nbasL, nbas - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = nbasL, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, nbasL - 1
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = nbasL, nbas - 1
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  call cint2e_spv1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_vsp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spv1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (-.5d0/env(PTR_LIGHT_SPEED))**3
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_vsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**3
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  do lshell = nbasL, nbas - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = nbasL, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = nbasL, lshell
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = nbasL, jshell
          if (jshell == lshell .and. ishell > kshell) then
            cycle
          end if
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  call cint2e(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)

  call cint2e_spsp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**4
  call HFBLK_ij_O3(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O3(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  do lshell = nbasL, nbas - 1
    dl = CINTcgto_spinor(lshell, bas)
    do kshell = nbasL, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = nbasL, nbas - 1
        dj = CINTcgto_spinor(jshell, bas)
        do ishell = nbasL, jshell
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
  allocate (fijkl(di,dj,dk,dl))
  call cint2e_spsp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  fijkl = fijkl * (.5d0/env(PTR_LIGHT_SPEED))**2
  call HFBLK_ij_O2(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_kl_O2(fijkl, dm, vj, n4c, shls, ao_loc, tao)
  call HFBLK_il_O2(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  call HFBLK_kj_O2(fijkl, dm, vk, n4c, shls, ao_loc, tao)
  deallocate (fijkl)
        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell
end subroutine dkb_vhf_coul_O02
