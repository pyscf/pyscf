subroutine nr_vhf_igiao_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
!**************************************************
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(out)  ::  vj(ndim,ndim,3)
  double precision,intent(out)  ::  vk(ndim,ndim,3)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

! ==========
!
!  non-relativistic HF coulomb and exchange potential in GIAOs
!       J = i[(i i|\mu g\nu) + (i gi|\mu \nu)]
!       K = i[(\mu gi|i \nu) + (\mu i|i g\nu)]
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
  integer               ::  shls(4)
  integer,allocatable           ::  ao_loc(:)
  double precision,allocatable  ::  fijkl(:,:,:,:,:)

  vj = 0d0
  vk = 0d0

  allocate (ao_loc(nbas))
  call CINTshells_spheric_offset(ao_loc, bas, nbas)

  do lshell = 0, nbas - 1
    dl = CINTcgto_spheric(lshell, bas)
    do kshell = 0, nbas - 1
      dk = CINTcgto_spheric(kshell, bas)
      do jshell = 0, nbas - 1
        dj = CINTcgto_spheric(jshell, bas)
        do ishell = 0, nbas - 1
          di = CINTcgto_spheric(ishell, bas)
          shls = (/ishell, jshell, kshell, lshell/)

  allocate (fijkl(di,dj,dk,dl,3))
  call cint2e_ig1_sph(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  do i = 1, 3
    call HFBLK_nr_ij_O0(fijkl(:,:,:,:,i), dm, vj(1,1,i), ndim, shls, ao_loc)
    call HFBLK_nr_il_O0(fijkl(:,:,:,:,i), dm, vk(1,1,i), ndim, shls, ao_loc)
    call HFBLK_nr_kj_O0(fijkl(:,:,:,:,i), dm, vk(1,1,i), ndim, shls, ao_loc)
  end do
  deallocate (fijkl)
        end do
      end do
    end do
  end do

  deallocate (ao_loc)
  return
end subroutine nr_vhf_igiao_O00
