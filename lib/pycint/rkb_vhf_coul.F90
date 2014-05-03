!
! File rkb_vhf_coul.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
!  J_{LL} = (SS|LL) * D_{LL}
!  J_{SS} = (SS|LL) * D_{SS}
!  K_{SL} = (i^S j^S|k^L l^L) * D_{j^S k^L}

#define QLL     (1)
#define QSS     (2)
#define QSL     (3)
#define DLL     (-3)
#define DLS     (-2)
#define DSL     (-1)
#define DSS     (0)

subroutine rkb_vhf_pre(tdm, dm, n4c, nset, nset_dm, &
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

  integer       ::  n2c

  n2c = n4c / 2
  tdm(1 :n2c,1 :n2c,:) = dm(1 :n2c,1 :n2c,:)
  tdm(1 :n2c,n2c+1:,:) = dm(1 :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  tdm(n2c+1:,1 :n2c,:) = dm(n2c+1:,1 :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  tdm(n2c+1:,n2c+1:,:) = dm(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
end subroutine rkb_vhf_pre

subroutine rkb_vhf_pre_and_screen(tdm, dm, n4c, nset, nset_dm, &
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

  call rkb_vhf_pre(tdm, dm, n4c, nset, nset_dm, &
                   atm, natm, bas, nbas, env)
  call rkb_vhf_init_screen(dm, n4c, nset, nset_dm, &
                           atm, natm, bas, nbas, env)
end subroutine rkb_vhf_pre_and_screen

subroutine rkb_vhf_after(vj, vk, n4c, nset, nset_dm, &
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

  integer                       ::  n2c

  n2c = n4c / 2
  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(1 :n2c,n2c+1:,:) = vk(1 :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,1 :n2c,:) = vk(n2c+1:,1 :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  call rkb_vhf_del_screen()
end subroutine rkb_vhf_after
! ************************************************


! ************************************************
! optimize for tsSS_tsLL, if vj/vk is Hermitian
subroutine rkb_vhf_SL_O2(intor, fscreen, dm, vj, vk, &
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
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, is, id, s
  integer                       ::  shls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)

  s = n4c / 2 + 1
  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas
  if (kshell > lshell) then
    return
  end if

  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    do ishell = 0, jshell
      di = CINTcgto_spinor(ishell, bas)
      shls = (/ishell, jshell, kshell, lshell/)

      allocate (fijkl(di,dj,dk,dl,nset))
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= intor(fijkl, shls, atm, natm, bas, nbas, env, opt)) then
        do is = 1, nset
          do id = 1, nset_dm
            if (0 /= do_vj(id)) then
              call HFBLK_ij_O2(fijkl(:,:,:,:,is), dm(1,1,id), vj(s,s,is,id), n4c, shls, ao_loc, ao_loc(nbas+1))
              call HFBLK_kl_O2(fijkl(:,:,:,:,is), dm(s,s,id), vj(1,1,is,id), n4c, shls, ao_loc, ao_loc(nbas+1))
            end if
            if (0 /= do_vk(id)) then
              call HFBLK_il_O2(fijkl(:,:,:,:,is), dm(s,1,id), vk(s,1,is,id), n4c, shls, ao_loc, ao_loc(nbas+1))
            end if
          end do
        end do
      end if
      deallocate (fijkl)
    end do
  end do
  return
end subroutine rkb_vhf_SL_O2

subroutine rkb_vhf_SL_after(vj, vk, n4c, nset, nset_dm, atm, natm, bas, nbas, env)
  use cint_const_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(inout)  ::  vj(n4c,n4c,nset*nset_dm)
  double complex,intent(inout)  ::  vk(n4c,n4c,nset*nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer                       ::  i, j, n2c

  n2c = n4c / 2

  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2

  vk(n2c+1:,:n2c  ,:) = vk(n2c+1:,:n2c  ,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  ! K_LS = K_SL^\dagger
  do j = n2c+1, n4c
    do i = 1, n2c
      vk(i,j,:) = conjg(vk(j,i,:))
    end do
  end do
  call rkb_vhf_del_screen()
end subroutine rkb_vhf_SL_after
! ************************************************


! ************************************************
subroutine rkb_vhf_LL_pre(tdm, dm, n4c, nset, nset_dm, &
                          atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(out)    ::  tdm(n4c,n4c,nset_dm)
  double complex,intent(in)     ::  dm(n4c,n4c,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer                       ::  ao_loc(nbas)

  integer                       ::  n2c

  n2c = CINTtot_cgto_spinor(bas, nbas)
  tdm(1 :n2c,1 :n2c,:) = dm(1 :n2c,1 :n2c,:)
end subroutine rkb_vhf_LL_pre

subroutine rkb_vhf_LL_pre_and_screen(tdm, dm, n4c, nset, nset_dm, &
                                     atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(out)    ::  tdm(n4c,n4c,nset_dm)
  double complex,intent(in)     ::  dm(n4c,n4c,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer                       ::  i, j, iloc, jloc, di, dj, id
  integer                       ::  n2c
  integer                       ::  ao_loc(nbas)

  call rkb_vhf_LL_pre(tdm, dm, n4c, nset, nset_dm, &
                      atm, natm, bas, nbas, env)

  if (HFBLK_direct_scf_cutoff < 0) then
    return
  end if

  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  if (allocated(HFBLK_dm_cond)) then
    deallocate (HFBLK_dm_cond)
  end if
  allocate (HFBLK_dm_cond(nbas,nbas*4,nset_dm))
  do j = 1, nbas
    jloc = ao_loc(j)
    dj = CINTcgto_spinor(j-1, bas)
    do i = 1, nbas
      iloc = ao_loc(i)
      di = CINTcgto_spinor(i-1, bas)
      do id = 1, nset_dm
        HFBLK_dm_cond(i,j*4+DLL,id) = maxval(abs(dm(iloc+1:iloc+di,jloc+1:jloc+dj,id)))
      end do
    end do
  end do
end subroutine rkb_vhf_LL_pre_and_screen

!subroutine rkb_vhf_SS_pre(dmss_t, dmss, n2c, nset, nset_dm, &
!                          atm, natm, bas, nbas, env)
!  use cint_const_mod
!  implicit none
!  integer,intent(in)            ::  n2c, nset, nset_dm
!  double complex,intent(out)    ::  dmss_t(n2c,n2c,nset_dm)
!  double complex,intent(in)     ::  dmss(n2c,n2c,nset_dm)
!  integer,intent(in)            ::  natm, nbas
!  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
!  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
!  double precision,intent(in)   ::  env(*)
!
!  integer       ::  i, j, is
!
!  do is = 1, nset_dm
!    do j = 1, n2c
!      do i = 1, n2c
!        dmss_t(i,j,is) = dmss(j,i,is) * (.5d0/env(PTR_LIGHT_SPEED))**2
!      end do
!    end do
!  end do
!  return
!end subroutine rkb_vhf_SS_pre
!
!subroutine rkb_vhf_SS_after(vjss, vkss, n2c, nset, nset_dm, &
!                            atm, natm, bas, nbas, env)
!  use cint_const_mod
!  implicit none
!  integer,intent(in)            ::  n2c, nset, nset_dm
!  double complex,intent(inout)  ::  vjss(n2c,n2c,nset,nset_dm)
!  double complex,intent(inout)  ::  vkss(n2c,n2c,nset,nset_dm)
!  integer,intent(in)            ::  natm, nbas
!  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
!  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
!  double precision,intent(in)   ::  env(*)
!
!  vjss = vjss * (.5d0/env(PTR_LIGHT_SPEED))**2
!  vkss = vkss * (.5d0/env(PTR_LIGHT_SPEED))**2
!end subroutine rkb_vhf_SS_after


subroutine init_rkb_direct_scf(atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer,external              ::  cint2e, cint2e_spsp1spsp2
  integer                       ::  i, j, i0, j0, iloc, jloc, di, dj
  integer                       ::  shls(4)
  integer                       ::  ao_loc(nbas)
  double precision              ::  qtmp
  double complex,allocatable    ::  fijij(:,:,:,:)

  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  if (allocated(HFBLK_q_cond)) then
    deallocate (HFBLK_q_cond)
  end if
  allocate (HFBLK_q_cond(nbas,nbas,3))
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
        qtmp = 0
        do j0 = 1, dj
          do i0 = 1, di
            qtmp = max(qtmp, abs(fijij(i0,j0,i0,j0)))
          end do
        end do
        HFBLK_q_cond(i,j,QLL) = sqrt(qtmp)
      end if
      if (0 /= cint2e_spsp1spsp2(fijij, shls, atm, natm, bas, nbas, env, 0_8)) then
        qtmp = 0
        do j0 = 1, dj
          do i0 = 1, di
            qtmp = max(qtmp, abs(fijij(i0,j0,i0,j0)))
          end do
        end do
        HFBLK_q_cond(i,j,QSS) = sqrt(qtmp) * (.5d0/env(PTR_LIGHT_SPEED))**2
      end if
      deallocate (fijij)
      HFBLK_q_cond(j,i,QLL) = HFBLK_q_cond(i,j,QLL)
      HFBLK_q_cond(j,i,QSS) = HFBLK_q_cond(i,j,QSS)
    end do
  end do

  call set_direct_scf_cutoff(1d-13)
end subroutine init_rkb_direct_scf

subroutine del_rkb_direct_scf()
  use hf_block_mod
  if (allocated(HFBLK_q_cond)) then
    deallocate (HFBLK_q_cond)
  end if
  call turnoff_direct_scf()
end subroutine del_rkb_direct_scf

subroutine rkb_vhf_init_screen(dm, n4c, nset, nset_dm, &
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
  integer                       ::  n2c
  integer                       ::  ao_loc(nbas)

  if (HFBLK_direct_scf_cutoff < 0) then
    return
  end if

  n2c = n4c / 2

  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  if (allocated(HFBLK_dm_cond)) then
    deallocate (HFBLK_dm_cond)
  end if
  allocate (HFBLK_dm_cond(nbas,nbas*4,nset_dm))
  do j = 1, nbas
    jloc = ao_loc(j)
    dj = CINTcgto_spinor(j-1, bas)
    do i = 1, nbas
      iloc = ao_loc(i)
      di = CINTcgto_spinor(i-1, bas)
      do id = 1, nset_dm
        HFBLK_dm_cond(i,j*4+DLL,id) = maxval(abs(dm(iloc+1    :iloc+di    ,jloc+1    :jloc+dj    ,id)))
        HFBLK_dm_cond(i,j*4+DLS,id) = maxval(abs(dm(iloc+1    :iloc+di    ,jloc+1+n2c:jloc+dj+n2c,id)))
        HFBLK_dm_cond(i,j*4+DSL,id) = maxval(abs(dm(iloc+1+n2c:iloc+di+n2c,jloc+1    :jloc+dj    ,id)))
        HFBLK_dm_cond(i,j*4+DSS,id) = maxval(abs(dm(iloc+1+n2c:iloc+di+n2c,jloc+1+n2c:jloc+dj+n2c,id)))
      end do
    end do
  end do
end subroutine rkb_vhf_init_screen

subroutine rkb_vhf_del_screen()
  use hf_block_mod
  if (allocated(HFBLK_dm_cond)) then
    deallocate (HFBLK_dm_cond)
  end if
end subroutine rkb_vhf_del_screen

integer function rkb_vhf_LL_prescreen(shls, do_vj, do_vk, nset)
  use hf_block_mod
  implicit none
  integer,intent(in)    ::  shls(4), nset
  integer,intent(out)   ::  do_vj(nset), do_vk(nset)
  integer               ::  i, j, k, l, id
  double precision      ::  dm_max, qijkl
  integer,external      ::  no_screen

  if (HFBLK_direct_scf_cutoff < 0) then
    rkb_vhf_LL_prescreen = no_screen(shls, do_vj, do_vk, nset)
  else
    i = shls(1) + 1
    j = shls(2) + 1
    k = shls(3) + 1
    l = shls(4) + 1
    rkb_vhf_LL_prescreen = 0
    qijkl = HFBLK_q_cond(i,j,QLL) * HFBLK_q_cond(k,l,QLL)
    do id = 1, nset
      dm_max = max(HFBLK_dm_cond(j,i*4+DLL,id), HFBLK_dm_cond(l,k*4+DLL,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vj(id) = 0
      else
        do_vj(id) = 1
        rkb_vhf_LL_prescreen = 1
      end if
    ! (ij|kl) -> K_{il},   K_{iTk}, K_{Tjl}, K_{TjTk} = K1
    !            K_{TlTi}, K_{kTi}, K_{Tlj}, K_{kj}  == T(K1)
      dm_max = max(HFBLK_dm_cond(j,k*4+DLL,id), HFBLK_dm_cond(j,l*4+DLL,id), &
                   HFBLK_dm_cond(i,k*4+DLL,id), HFBLK_dm_cond(i,l*4+DLL,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vk(id) = 0
      else
        do_vk(id) = 1
        rkb_vhf_LL_prescreen = 1
      end if
    end do
  end if
end function rkb_vhf_LL_prescreen

integer function rkb_vhf_SL_prescreen(shls, do_vj, do_vk, nset)
  use hf_block_mod
  implicit none
  integer,intent(in)    ::  shls(4), nset
  integer,intent(out)   ::  do_vj(nset), do_vk(nset)
  integer               ::  i, j, k, l, id
  double precision      ::  dm_max, qijkl
  integer,external      ::  no_screen

  if (HFBLK_direct_scf_cutoff < 0) then
    rkb_vhf_SL_prescreen = no_screen(shls, do_vj, do_vk, nset)
  else
    i = shls(1) + 1
    j = shls(2) + 1
    k = shls(3) + 1
    l = shls(4) + 1
    rkb_vhf_SL_prescreen = 0
    qijkl = HFBLK_q_cond(i,j,QSS) * HFBLK_q_cond(k,l,QLL)
    do id = 1, nset
      dm_max = max(HFBLK_dm_cond(j,i*4+DSS,id), HFBLK_dm_cond(l,k*4+DLL,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vj(id) = 0
      else
        do_vj(id) = 1
        rkb_vhf_SL_prescreen = 1
      end if
      dm_max = max(HFBLK_dm_cond(j,k*4+DSL,id), HFBLK_dm_cond(j,l*4+DSL,id), &
                   HFBLK_dm_cond(i,k*4+DSL,id), HFBLK_dm_cond(i,l*4+DSL,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vk(id) = 0
      else
        do_vk(id) = 1
        rkb_vhf_SL_prescreen = 1
      end if
    end do
  end if
end function rkb_vhf_SL_prescreen

integer function rkb_vhf_SS_prescreen(shls, do_vj, do_vk, nset)
  use hf_block_mod
  implicit none
  integer,intent(in)    ::  shls(4), nset
  integer,intent(out)   ::  do_vj(nset), do_vk(nset)
  integer               ::  i, j, k, l, id
  double precision      ::  dm_max, qijkl
  integer,external      ::  no_screen

  if (HFBLK_direct_scf_cutoff < 0) then
    rkb_vhf_SS_prescreen = no_screen(shls, do_vj, do_vk, nset)
  else
    i = shls(1) + 1
    j = shls(2) + 1
    k = shls(3) + 1
    l = shls(4) + 1
    rkb_vhf_SS_prescreen = 0
    qijkl = HFBLK_q_cond(i,j,QSS) * HFBLK_q_cond(k,l,QSS)
    do id = 1, nset
      dm_max = max(HFBLK_dm_cond(j,i*4+DSS,id), HFBLK_dm_cond(l,k*4+DSS,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vj(id) = 0
      else
        do_vj(id) = 1
        rkb_vhf_SS_prescreen = 1
      end if
      dm_max = max(HFBLK_dm_cond(j,k*4+DSS,id), HFBLK_dm_cond(j,l*4+DSS,id), &
                   HFBLK_dm_cond(i,k*4+DSS,id), HFBLK_dm_cond(i,l*4+DSS,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vk(id) = 0
      else
        do_vk(id) = 1
        rkb_vhf_SS_prescreen = 1
      end if
    end do
  end if
end function rkb_vhf_SS_prescreen
