#define QLL     (1)
#define QSS     (2)
#define QSL     (3)
#define DLL     (-3)
#define DLS     (-2)
#define DSL     (-1)
#define DSS     (0)

subroutine rkb_vhf_gaunt_iter(intor, fscreen, dm, vj, vk, &
                              n4c, nset, nset_dm, pair_kl, ao_loc, &
                              atm, natm, bas, nbas, env, opt)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,external              ::  intor, fscreen
  integer,intent(in)            ::  n4c, nset, nset_dm
  double complex,intent(in)     ::  dm(n4c,n4c,nset_dm)
  double complex,intent(inout)  ::  vj(n4c,n4c,nset_dm)
  double complex,intent(inout)  ::  vk(n4c,n4c,nset_dm)
  integer,intent(in)            ::  pair_kl
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)
  integer(8)                    ::  opt

  integer,external              ::  cint2e_ssp1ssp2
  integer                       ::  ishell, jshell, kshell, lshell
  integer                       ::  di, dj, dk, dl, id, s
  integer                       ::  shls(4), tijshls(4)
  integer                       ::  do_vj(nset_dm), do_vk(nset_dm)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
  double complex,allocatable    ::  tijkl(:,:,:,:)
  double complex,allocatable    ::  ijtkl(:,:,:,:)

  s = n4c / 2 + 1
  lshell = pair_kl / nbas
  kshell = pair_kl - lshell * nbas

  dk = CINTcgto_spinor(kshell, bas)
  dl = CINTcgto_spinor(lshell, bas)

  do jshell = 0, nbas - 1
    dj = CINTcgto_spinor(jshell, bas)

    do ishell = 0, nbas - 1
      di = CINTcgto_spinor(ishell, bas)
      if (ishell+jshell*nbas > pair_kl) then
        cycle
      end if

      allocate (fijkl(di,dj,dk,dl,nset_dm))
      shls = (/ishell, jshell, kshell, lshell/)
      tijshls = (/jshell, ishell, kshell, lshell/)
      if (0 /= fscreen(shls, do_vj, do_vk, nset_dm) .and. &
          0 /= cint2e_ssp1ssp2(fijkl, shls, atm, natm, bas, nbas, env, opt)) then

      allocate (tijkl(dj,di,dk,dl))
  do id = 1, nset_dm
    ! There would be no vj in closed shell sys because the density vanishes due
    ! to the anti time-reversal symm.
    call HFBLK_kl_O0(fijkl(:,:,:,:,id), dm(s,1,id), vj(1,s,id), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,id), dm(s,1,id), vk(1,s,id), n4c, shls, ao_loc)

    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,id), tijkl, shls, ao_loc, ao_loc(nbas+1))
    call HFBLK_kl_O0(tijkl, dm(1,s,id), vj(1,s,id), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, dm(1,1,id), vk(s,s,id), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, dm(s,s,id), vk(1,1,id), n4c, tijshls, ao_loc)
  end do
  deallocate (tijkl)

  if (ishell /= kshell .or. jshell /= lshell) then
    allocate (ijtkl(di,dj,dl,dk))
    tijshls = (/ishell, jshell, lshell, kshell/)
    do id = 1, nset_dm
      call HFBLK_ij_O0(fijkl(:,:,:,:,id), dm(s,1,id), vj(1,s,id), n4c, shls, ao_loc)
      call HFBLK_kj_O0(fijkl(:,:,:,:,id), dm(s,1,id), vk(1,s,id), n4c, shls, ao_loc)

      call HFBLK_atimerev_ijtltk(fijkl(:,:,:,:,id), ijtkl, shls, ao_loc, ao_loc(nbas+1))
      call HFBLK_ij_O0(ijtkl, dm(1,s,id), vj(1,s,id), n4c, tijshls, ao_loc)
      call HFBLK_il_O0(ijtkl, dm(s,s,id), vk(1,1,id), n4c, tijshls, ao_loc)
      call HFBLK_kj_O0(ijtkl, dm(1,1,id), vk(s,s,id), n4c, tijshls, ao_loc)
    end do
    deallocate (ijtkl)
  end if

      end if
      deallocate (fijkl)
    end do
  end do
  return
end subroutine rkb_vhf_gaunt_iter


subroutine rkb_vhf_gaunt_after(vj, vk, n4c, nset, nset_dm, &
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
  call rkb_vhf_del_screen()
end subroutine rkb_vhf_gaunt_after


! the gaunt term must be called after coulomb term which is initialized in
! init_rkb_direct_scf
subroutine init_rkb_gaunt_direct_scf(atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer,external              ::  cint2e_ssp1ssp2
  integer                       ::  i, j, i0, j0, iloc, jloc, di, dj
  integer                       ::  shls(4)
  integer                       ::  ao_loc(nbas)
  double precision              ::  qtmp
  double complex,allocatable    ::  fijij(:,:,:,:)

  call CINTshells_spinor_offset(ao_loc, bas, nbas)

  ! HFBLK_q_cond should be partially initialized in init_rkb_direct_scf
  do j = 1, nbas
    jloc = ao_loc(j)
    dj = CINTcgto_spinor(j-1, bas)
    do i = 1, nbas
      iloc = ao_loc(i)
      di = CINTcgto_spinor(i-1, bas)
      allocate (fijij(di,dj,di,dj))
      shls = (/i-1, j-1, i-1, j-1/)
      if (0 /= cint2e_ssp1ssp2(fijij, shls, atm, natm, bas, nbas, env, 0_8)) then
        qtmp = 0
        do j0 = 1, dj
          do i0 = 1, di
            qtmp = max(qtmp, abs(fijij(i0,j0,i0,j0)))
          end do
        end do
        HFBLK_q_cond(i,j,QSL) = sqrt(qtmp)
      end if
      deallocate (fijij)
    end do
  end do
end subroutine init_rkb_gaunt_direct_scf

subroutine rkb_vhf_gaunt_pre_and_screen(tdm, dm, n4c, nset, nset_dm, &
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
  ! Initialize dm_cond. It's the same in coulomb/gaunt terms.
  call rkb_vhf_init_screen(dm, n4c, nset, nset_dm, &
                           atm, natm, bas, nbas, env)
end subroutine rkb_vhf_gaunt_pre_and_screen

integer function rkb_vhf_gaunt_prescreen(shls, do_vj, do_vk, nset)
  use hf_block_mod
  implicit none
  integer,intent(in)    ::  shls(4), nset
  integer,intent(out)   ::  do_vj(nset), do_vk(nset)
  integer               ::  i, j, k, l
  double precision      ::  dm_max, qijkl
  integer,external      ::  no_screen

  if (HFBLK_direct_scf_cutoff == 0) then
    rkb_vhf_gaunt_prescreen = no_screen(shls, do_vj, do_vk, nset)
  else
    i = shls(1) + 1
    j = shls(2) + 1
    k = shls(3) + 1
    l = shls(4) + 1
    rkb_vhf_gaunt_prescreen = 0
    qijkl = HFBLK_q_cond(i,j,QSL) * HFBLK_q_cond(k,l,QSL)
    !dm_max = max(maxval(HFBLK_dm_cond(j,i*4+DSL,:)), &
    !             maxval(HFBLK_dm_cond(i,j*4+DLS,:)), &
    !             maxval(HFBLK_dm_cond(l,k*4+DSL,:)), &
    !             maxval(HFBLK_dm_cond(k,l*4+DLS,:)))
    !if (dm_max * qijkl > HFBLK_direct_scf_cutoff) then
    !  rkb_vhf_gaunt_prescreen = 1
    !end if
    dm_max = max(maxval(HFBLK_dm_cond(j,k*4+DSL,:)), &
                 maxval(HFBLK_dm_cond(l,i*4+DSL,:)), &
                 maxval(HFBLK_dm_cond(i,k*4+DLL,:)), &
                 maxval(HFBLK_dm_cond(l,j*4+DSS,:)), &
                 maxval(HFBLK_dm_cond(j,l*4+DSS,:)), &
                 maxval(HFBLK_dm_cond(k,i*4+DLL,:)))
    if (dm_max * qijkl > HFBLK_direct_scf_cutoff) then
      rkb_vhf_gaunt_prescreen = 1
    end if
  end if
end function rkb_vhf_gaunt_prescreen
