!
! File nr_vhf.F90
! Author: Qiming Sun <osirpt.sun@gmail.com>
!
!  a set of non-relativistic HF coulomb and exchange potential (in AO representation)
!       J = (ii|\mu \nu)
!       K = (\mu i|i\nu)
!  no requirement on Density matrix
!

!subroutine nr_vhf_pre(dm_t, dm, ndim, nset_dm, atm, natm, bas, nbas, env)
!  use cint_const_mod
!  implicit none
!  integer,intent(in)            ::  ndim, nset_dm
!  double precision,intent(out)  ::  dm_t(ndim,ndim,nset_dm)
!  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
!  integer,intent(in)            ::  natm, nbas
!  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
!  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
!  double precision,intent(in)   ::  env(*)
!
!  integer       ::  i, j, is
!
!  do is = 1, nset_dm
!    do j = 1, ndim
!      do i = 1, ndim
!        dm_t(i,j,is) = dm(j,i,is)
!      end do
!    end do
!  end do
!end subroutine nr_vhf_pre

subroutine init_nr_direct_scf(atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer,external              ::  cint2e_sph
  integer                       ::  i, j, i0, j0, iloc, jloc, di, dj
  integer                       ::  shls(4)
  integer                       ::  ao_loc(nbas)
  double precision              ::  qtmp
  double precision,allocatable  ::  fijij(:,:,:,:)

  call CINTshells_spheric_offset(ao_loc, bas, nbas)

  if (allocated(HFBLK_q_cond)) then
    deallocate (HFBLK_q_cond)
  end if
  allocate (HFBLK_q_cond(nbas,nbas,1))
  HFBLK_q_cond = 0
  do j = 1, nbas
    jloc = ao_loc(j)
    dj = CINTcgto_spheric(j-1, bas)
    do i = 1, j
      iloc = ao_loc(i)
      di = CINTcgto_spheric(i-1, bas)
      allocate (fijij(di,dj,di,dj))
      shls = (/i-1, j-1, i-1, j-1/)
      if (0 /= cint2e_sph(fijij, shls, atm, natm, bas, nbas, env, 0_8)) then
        qtmp = 0
        do j0 = 1, dj
          do i0 = 1, di
            qtmp = max(qtmp, abs(fijij(i0,j0,i0,j0)))
          end do
        end do
        HFBLK_q_cond(i,j,1) = sqrt(qtmp)
      end if
      deallocate (fijij)
      HFBLK_q_cond(j,i,1) = HFBLK_q_cond(i,j,1)
    end do
  end do

  call set_direct_scf_cutoff(1d-13)
end subroutine init_nr_direct_scf

subroutine del_nr_direct_scf()
  use hf_block_mod
  if (allocated(HFBLK_q_cond)) then
    deallocate (HFBLK_q_cond)
  end if
  call turnoff_direct_scf()
end subroutine del_nr_direct_scf

subroutine turnoff_direct_scf()
  use hf_block_mod
  HFBLK_direct_scf_cutoff = -1
end subroutine turnoff_direct_scf

subroutine set_direct_scf_cutoff(c)
  use hf_block_mod
  double precision      ::  c
  HFBLK_direct_scf_cutoff = c
end subroutine set_direct_scf_cutoff

! ****************************************
! conditions for direct-SCF. Ref. JCC, 10, 104
subroutine nr_vhf_init_screen(dm, ndim, nset, nset_dm, &
                              atm, natm, bas, nbas, env)
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none
  integer,intent(in)            ::  ndim, nset, nset_dm
  double precision,intent(in)   ::  dm(ndim,ndim,nset_dm)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

  integer                       ::  i, j, iloc, jloc, di, dj, id
  integer                       ::  ao_loc(nbas)

  if (HFBLK_direct_scf_cutoff < 0) then
    return
  end if

  call CINTshells_spheric_offset(ao_loc, bas, nbas)

  if (allocated(HFBLK_dm_cond)) then
    deallocate (HFBLK_dm_cond)
  end if
  allocate (HFBLK_dm_cond(nbas,nbas,nset_dm))
  do j = 1, nbas
    jloc = ao_loc(j)
    dj = CINTcgto_spheric(j-1, bas)
    do i = 1, nbas
      iloc = ao_loc(i)
      di = CINTcgto_spheric(i-1, bas)
      do id = 1, nset_dm
        HFBLK_dm_cond(i,j,id) = .25d0 * maxval(abs(dm(iloc+1:iloc+di,jloc+1:jloc+dj,id)))
      end do
    end do
  end do
end subroutine nr_vhf_init_screen

subroutine nr_vhf_del_screen()
  use hf_block_mod
  if (allocated(HFBLK_dm_cond)) then
    deallocate (HFBLK_dm_cond)
  end if
end subroutine nr_vhf_del_screen

! nr_vhf_prescreen = 0 means small value which can be ignored
integer function nr_vhf_prescreen(shls, do_vj, do_vk, nset)
  use hf_block_mod
  implicit none
  integer,intent(in)    ::  shls(4), nset
  integer,intent(out)   ::  do_vj(nset), do_vk(nset)
  integer               ::  i, j, k, l, id
  double precision      ::  dm_max, qijkl
  integer,external      ::  no_screen

  if (HFBLK_direct_scf_cutoff < 0) then
    nr_vhf_prescreen = no_screen(shls, do_vj, do_vk, nset)
  else
    i = shls(1) + 1
    j = shls(2) + 1
    k = shls(3) + 1
    l = shls(4) + 1
    nr_vhf_prescreen = 0
    qijkl = HFBLK_q_cond(i,j,1) * HFBLK_q_cond(k,l,1)
    do id = 1, nset
      dm_max = 4 * max(HFBLK_dm_cond(j,i,id), HFBLK_dm_cond(l,k,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vj(id) = 0
      else
        do_vj(id) = 1
        nr_vhf_prescreen = 1
      end if
      dm_max = max(HFBLK_dm_cond(j,k,id), HFBLK_dm_cond(j,l,id), &
                   HFBLK_dm_cond(i,k,id), HFBLK_dm_cond(i,l,id))
      if (dm_max * qijkl < HFBLK_direct_scf_cutoff) then
        do_vk(id) = 0
      else
        do_vk(id) = 1
        nr_vhf_prescreen = 1
      end if
    end do
  end if
end function nr_vhf_prescreen

integer function no_screen(shls, do_vj, do_vk, nset)
  integer,intent(in)    ::  shls(4), nset
  integer,intent(out)   ::  do_vj(nset), do_vk(nset)
  do_vj = 1
  do_vk = 1
  no_screen = 1
end function no_screen


