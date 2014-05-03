subroutine nr_vhf_O00(dm, vj, vk, ndim, atm, natm, bas, nbas, env)
!**************************************************
  use cint_const_mod
  use cint_interface
  use hf_block_mod
  implicit none

  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(out)  ::  vj(ndim,ndim)
  double precision,intent(out)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  natm, nbas
  integer,intent(in)            ::  atm(ATM_SLOTS,natm)
  integer,intent(in)            ::  bas(BAS_SLOTS,nbas)
  double precision,intent(in)   ::  env(*)

! ==========
!
!  non-relativistic HF coulomb and exchange potential (in AO representation)
!      vg = J - K
!       J = (ii|\mu \nu)
!       K = (\mu i|i\nu)
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
  double precision,allocatable  ::  fijkl(:,:,:,:)

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

  allocate (fijkl(di,dj,dk,dl))
  call cint2e_sph(fijkl, shls, atm, natm, bas, nbas, env, 0_8)
  call HFBLK_nr_kl_O0(fijkl, dm, vj, ndim, shls, ao_loc)
  call HFBLK_nr_il_O0(fijkl, dm, vk, ndim, shls, ao_loc)
  deallocate (fijkl)

        end do ! ishell
      end do ! jshell
    end do ! kshell
  end do ! lshell

  deallocate (ao_loc)
  return
end subroutine nr_vhf_O00
