subroutine rmb4cg_vhf_coul_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s, is
  integer               ::  shls(4), tijshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
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
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
  call cint2e_cg_sa10sp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(1,1), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(1,1,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(s,1,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(1,s), vk(1,s,is), n4c, shls, ao_loc)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(tijkl, tdm(1,1), vj(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kl_O0(tijkl, tdm(s,s), vj(1,1,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(s,1), vk(s,1,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(1,s), vk(1,s,is), n4c, tijshls, ao_loc)
  end do
  call cint2e_cg_sa10sp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(tijkl, tdm(s,s), vj(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kl_O0(tijkl, tdm(s,s), vj(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc)
  end do
  deallocate (fijkl, tijkl)
        end do
      end do
    end do ! kshell
  end do ! lshell

  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c,:) = vk(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  deallocate (tdm)
  deallocate (ao_loc)
  return
end subroutine rmb4cg_vhf_coul_O00

subroutine rmb4giao_vhf_coul_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl, i, j
  integer               ::  n2c, n4c, s, is
  integer               ::  shls(4), tijshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
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
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
  call cint2e_giao_sa10sp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(1,1), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(1,1,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(s,1,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(1,s), vk(1,s,is), n4c, shls, ao_loc)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(tijkl, tdm(1,1), vj(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kl_O0(tijkl, tdm(s,s), vj(1,1,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(s,1), vk(s,1,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(1,s), vk(1,s,is), n4c, tijshls, ao_loc)
  end do
  call cint2e_giao_sa10sp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,s), vj(s,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(tijkl, tdm(s,s), vj(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kl_O0(tijkl, tdm(s,s), vj(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc)
  end do
  deallocate (fijkl, tijkl)
        end do
      end do
    end do ! kshell
  end do ! lshell

  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c,:) = vk(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  deallocate (tdm)
  deallocate (ao_loc)
  return
end subroutine rmb4giao_vhf_coul_O00

subroutine rmb4cg_vhf_coul_O01(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s, is, i, j
  integer               ::  shls(4), tijshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
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
    do kshell = 0, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, nbas - 1
        dj = CINTcgto_spinor(jshell, bas)

        do ishell = 0, nbas - 1
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
  call cint2e_cg_sa10sp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O1(fijkl(:,:,:,:,is), tdm(1,1), vj(s,s,is), n4c, shls, ao_loc, tao)
    call HFBLK_il_O1(fijkl(:,:,:,:,is), tdm(s,1), vk(s,1,is), n4c, shls, ao_loc, tao)
    call HFBLK_kj_O1(fijkl(:,:,:,:,is), tdm(1,s), vk(1,s,is), n4c, shls, ao_loc, tao)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O1(tijkl, tdm(1,1), vj(s,s,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_il_O1(tijkl, tdm(s,1), vk(s,1,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_kj_O1(tijkl, tdm(1,s), vk(1,s,is), n4c, tijshls, ao_loc, tao)
  end do
  call cint2e_cg_sa10sp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O1(fijkl(:,:,:,:,is), tdm(s,s), vj(s,s,is), n4c, shls, ao_loc, tao)
    call HFBLK_il_O1(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc, tao)
    call HFBLK_kj_O1(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc, tao)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O1(tijkl, tdm(s,s), vj(s,s,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_il_O1(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_kj_O1(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc, tao)
  end do
  deallocate (fijkl, tijkl)
        end do
      end do
    end do ! kshell
  end do ! lshell

  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c,:) = vk(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  do j = 1, n4c
    do i = 1, j-1
      vj(j,i,:) = conjg(vj(i,j,:))
      vk(j,i,:) = conjg(vk(i,j,:))
    end do
  end do
  deallocate (tdm)
  deallocate (ao_loc)
  return
end subroutine rmb4cg_vhf_coul_O01

subroutine rmb4giao_vhf_coul_O01(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s, is, i, j
  integer               ::  shls(4), tijshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
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
    do kshell = 0, lshell
      dk = CINTcgto_spinor(kshell, bas)
      do jshell = 0, nbas - 1
        dj = CINTcgto_spinor(jshell, bas)

        do ishell = 0, nbas - 1
          di = CINTcgto_spinor(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
  call cint2e_giao_sa10sp1(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O1(fijkl(:,:,:,:,is), tdm(1,1), vj(s,s,is), n4c, shls, ao_loc, tao)
    call HFBLK_il_O1(fijkl(:,:,:,:,is), tdm(s,1), vk(s,1,is), n4c, shls, ao_loc, tao)
    call HFBLK_kj_O1(fijkl(:,:,:,:,is), tdm(1,s), vk(1,s,is), n4c, shls, ao_loc, tao)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O1(tijkl, tdm(1,1), vj(s,s,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_il_O1(tijkl, tdm(s,1), vk(s,1,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_kj_O1(tijkl, tdm(1,s), vk(1,s,is), n4c, tijshls, ao_loc, tao)
  end do
  call cint2e_giao_sa10sp1spsp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O1(fijkl(:,:,:,:,is), tdm(s,s), vj(s,s,is), n4c, shls, ao_loc, tao)
    call HFBLK_il_O1(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc, tao)
    call HFBLK_kj_O1(fijkl(:,:,:,:,is), tdm(s,s), vk(s,s,is), n4c, shls, ao_loc, tao)
    ! (sp sA10| )
    call HFBLK_atimerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O1(tijkl, tdm(s,s), vj(s,s,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_il_O1(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc, tao)
    call HFBLK_kj_O1(tijkl, tdm(s,s), vk(s,s,is), n4c, tijshls, ao_loc, tao)
  end do
  deallocate (fijkl, tijkl)
        end do
      end do
    end do ! kshell
  end do ! lshell

  vj(n2c+1:,n2c+1:,:) = vj(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c,:) = vk(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  do j = 1, n4c
    do i = 1, j-1
      vj(j,i,:) = conjg(vj(i,j,:))
      vk(j,i,:) = conjg(vk(i,j,:))
    end do
  end do
  deallocate (tdm)
  deallocate (ao_loc)
  return
end subroutine rmb4giao_vhf_coul_O01


subroutine rmb4cg_vhf_gaunt_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s, is, i, j
  integer               ::  shls(4), tijshls(4), tklshls(4), ttshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
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
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)
          tklshls = (/ishell, jshell, lshell, kshell/)
          ttshls  = (/jshell, ishell, lshell, kshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
  allocate (ijtkl(di,dj,dl,dk), tijtkl(dj,di,dl,dk))
  call cint2e_cg_ssa10ssp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)

    call HFBLK_timerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(tijkl, tdm(s,1), vj(s,1,is), n4c, tijshls, ao_loc)
    call HFBLK_kl_O0(tijkl, tdm(1,s), vj(1,s,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(1,1), vk(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(s,s), vk(1,1,is), n4c, tijshls, ao_loc)

    call HFBLK_atimerev_ijtltk(fijkl(:,:,:,:,is), ijtkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(ijtkl, tdm(1,s), vj(1,s,is), n4c, tklshls, ao_loc)
    call HFBLK_kl_O0(ijtkl, tdm(s,1), vj(s,1,is), n4c, tklshls, ao_loc)
    call HFBLK_il_O0(ijtkl, tdm(s,s), vk(1,1,is), n4c, tklshls, ao_loc)
    call HFBLK_kj_O0(ijtkl, tdm(1,1), vk(s,s,is), n4c, tklshls, ao_loc)

    call HFBLK_atimerev_ijtltk(tijkl, tijtkl, tijshls, ao_loc, tao)
    call HFBLK_ij_O0(tijtkl, tdm(1,s), vj(s,1,is), n4c, ttshls, ao_loc)
    call HFBLK_kl_O0(tijtkl, tdm(1,s), vj(s,1,is), n4c, ttshls, ao_loc)
    call HFBLK_il_O0(tijtkl, tdm(1,s), vk(s,1,is), n4c, ttshls, ao_loc)
    call HFBLK_kj_O0(tijtkl, tdm(1,s), vk(s,1,is), n4c, ttshls, ao_loc)
  end do
  deallocate (fijkl, tijkl, ijtkl, tijtkl)
        end do
      end do
    end do ! kshell
  end do ! lshell
  deallocate (tdm)
  deallocate (ao_loc)

  vj(  :n2c,n2c+1:,:) = vj(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vj(n2c+1:,  :n2c,:) = vj(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c,:) = vk(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  return
end subroutine rmb4cg_vhf_gaunt_O00

subroutine rmb4giao_vhf_gaunt_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s, is
  integer               ::  shls(4), tijshls(4), tklshls(4), ttshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
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
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)
          tklshls = (/ishell, jshell, lshell, kshell/)
          ttshls  = (/jshell, ishell, lshell, kshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl))
  allocate (ijtkl(di,dj,dl,dk), tijtkl(dj,di,dl,dk))
  call cint2e_giao_ssa10ssp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)

    call HFBLK_timerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(tijkl, tdm(s,1), vj(s,1,is), n4c, tijshls, ao_loc)
    call HFBLK_kl_O0(tijkl, tdm(1,s), vj(1,s,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(1,1), vk(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(s,s), vk(1,1,is), n4c, tijshls, ao_loc)

    call HFBLK_atimerev_ijtltk(fijkl(:,:,:,:,is), ijtkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(ijtkl, tdm(1,s), vj(1,s,is), n4c, tklshls, ao_loc)
    call HFBLK_kl_O0(ijtkl, tdm(s,1), vj(s,1,is), n4c, tklshls, ao_loc)
    call HFBLK_il_O0(ijtkl, tdm(s,s), vk(1,1,is), n4c, tklshls, ao_loc)
    call HFBLK_kj_O0(ijtkl, tdm(1,1), vk(s,s,is), n4c, tklshls, ao_loc)

    call HFBLK_atimerev_ijtltk(tijkl, tijtkl, tijshls, ao_loc, tao)
    call HFBLK_ij_O0(tijtkl, tdm(1,s), vj(s,1,is), n4c, ttshls, ao_loc)
    call HFBLK_kl_O0(tijtkl, tdm(1,s), vj(s,1,is), n4c, ttshls, ao_loc)
    call HFBLK_il_O0(tijtkl, tdm(1,s), vk(s,1,is), n4c, ttshls, ao_loc)
    call HFBLK_kj_O0(tijtkl, tdm(1,s), vk(s,1,is), n4c, ttshls, ao_loc)
  end do
  deallocate (fijkl, tijkl, ijtkl, tijtkl)
        end do
      end do
    end do ! kshell
  end do ! lshell
  deallocate (tdm)
  deallocate (ao_loc)

  vj(  :n2c,n2c+1:,:) = vj(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vj(n2c+1:,  :n2c,:) = vj(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,  :n2c,:) = vk(n2c+1:,  :n2c,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  return
end subroutine rmb4giao_vhf_gaunt_O00

subroutine rmb4cg_vhf_gaunt_O01(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s, is, i, j
  integer               ::  shls(4), tijshls(4), tklshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
  double complex,allocatable    ::  tijkl(:,:,:,:)
  double complex,allocatable    ::  ijtkl(:,:,:,:)
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
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)
          tklshls = (/ishell, jshell, lshell, kshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl), ijtkl(di,dj,dl,dk))
  call cint2e_cg_ssa10ssp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)

    call HFBLK_timerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_kl_O0(tijkl, tdm(1,s), vj(1,s,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(1,1), vk(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(s,s), vk(1,1,is), n4c, tijshls, ao_loc)

    call HFBLK_atimerev_ijtltk(fijkl(:,:,:,:,is), ijtkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(ijtkl, tdm(1,s), vj(1,s,is), n4c, tklshls, ao_loc)
    call HFBLK_il_O0(ijtkl, tdm(s,s), vk(1,1,is), n4c, tklshls, ao_loc)
    call HFBLK_kj_O0(ijtkl, tdm(1,1), vk(s,s,is), n4c, tklshls, ao_loc)
  end do
  deallocate (fijkl, tijkl, ijtkl)
        end do
      end do
    end do ! kshell
  end do ! lshell
  deallocate (tdm)
  deallocate (ao_loc)

  vj(  :n2c,n2c+1:,:) = vj(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  do j = 1, n2c
    do i = n2c+1, n4c
      vj(i,j,:) = conjg(vj(j,i,:))
      vk(i,j,:) = conjg(vk(j,i,:))
    end do
  end do
  return
end subroutine rmb4cg_vhf_gaunt_O01

subroutine rmb4giao_vhf_gaunt_O01(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
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

  integer               ::  ishell, jshell, kshell, lshell
  integer               ::  di, dj, dk, dl
  integer               ::  n2c, n4c, s, is, i, j
  integer               ::  shls(4), tijshls(4), tklshls(4)
  integer               ::  tao(ndim)
  integer,allocatable           ::  ao_loc(:)
  double complex,allocatable    ::  fijkl(:,:,:,:,:)
  double complex,allocatable    ::  tijkl(:,:,:,:)
  double complex,allocatable    ::  ijtkl(:,:,:,:)
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
          shls = (/ishell, jshell, kshell, lshell/)
          tijshls = (/jshell, ishell, kshell, lshell/)
          tklshls = (/ishell, jshell, lshell, kshell/)

  allocate (fijkl(di,dj,dk,dl,3), tijkl(dj,di,dk,dl), ijtkl(di,dj,dl,dk))
  call cint2e_giao_ssa10ssp2(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do is = 1, 3
    call HFBLK_ij_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kl_O0(fijkl(:,:,:,:,is), tdm(s,1), vj(1,s,is), n4c, shls, ao_loc)
    call HFBLK_il_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)
    call HFBLK_kj_O0(fijkl(:,:,:,:,is), tdm(s,1), vk(1,s,is), n4c, shls, ao_loc)

    call HFBLK_timerev_tjtikl(fijkl(:,:,:,:,is), tijkl, shls, ao_loc, tao)
    call HFBLK_kl_O0(tijkl, tdm(1,s), vj(1,s,is), n4c, tijshls, ao_loc)
    call HFBLK_il_O0(tijkl, tdm(1,1), vk(s,s,is), n4c, tijshls, ao_loc)
    call HFBLK_kj_O0(tijkl, tdm(s,s), vk(1,1,is), n4c, tijshls, ao_loc)

    call HFBLK_atimerev_ijtltk(fijkl(:,:,:,:,is), ijtkl, shls, ao_loc, tao)
    call HFBLK_ij_O0(ijtkl, tdm(1,s), vj(1,s,is), n4c, tklshls, ao_loc)
    call HFBLK_il_O0(ijtkl, tdm(s,s), vk(1,1,is), n4c, tklshls, ao_loc)
    call HFBLK_kj_O0(ijtkl, tdm(1,1), vk(s,s,is), n4c, tklshls, ao_loc)
  end do
  deallocate (fijkl, tijkl, ijtkl)
        end do
      end do
    end do ! kshell
  end do ! lshell
  deallocate (tdm)
  deallocate (ao_loc)

  vj(  :n2c,n2c+1:,:) = vj(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(  :n2c,n2c+1:,:) = vk(  :n2c,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))
  vk(n2c+1:,n2c+1:,:) = vk(n2c+1:,n2c+1:,:) * (.5d0/env(PTR_LIGHT_SPEED))**2
  do j = 1, n2c
    do i = n2c+1, n4c
      vj(i,j,:) = conjg(vj(j,i,:))
      vk(i,j,:) = conjg(vk(j,i,:))
    end do
  end do
  return
end subroutine rmb4giao_vhf_gaunt_O01
