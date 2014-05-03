!*******************************************************************************
!
!  File: hf_block_mod.F90
!  Author: Qiming Sun <osirpt.sun@gmail.com>
!  Last Change:
!  Description:
!       generate HF blocks with cint.
!       subscripts 1234 correspond to ijkl; fijkl == <ki|jl> == (ij|kl)
!       both e1 and e2 operator in (ij|kl) should be TIME-REVERSAL SYMM.
!       time-reversal anti-symm. of e1 will affect *_O2 and *_O3
!       time-reversal anti-symm. of e2 will affect *_O1, *_O2 and *_O3
!
!*******************************************************************************

module hf_block_mod
  use cint_const_mod
  use cint_interface

  private
  ! conditions for direct-SCF. Ref. JCC, 10, 104
  double precision,allocatable,public   ::  HFBLK_dm_cond(:,:,:)
  double precision,allocatable,public   ::  HFBLK_q_cond(:,:,:)
  ! if HFBLK_direct_scf_cutoff < 0, switch off direct scf
  double precision,public       ::  HFBLK_direct_scf_cutoff = -1

  public :: HFBLK_nr_ij_O0, &
            HFBLK_nr_ij_O1, &
            HFBLK_nr_ij_O2, &
            HFBLK_nr_ij_O3, &
            HFBLK_nr_il_O0, &
            HFBLK_nr_il_O1, &
            HFBLK_nr_il_O2, &
            HFBLK_nr_il_O3, &
            HFBLK_nr_kj_O0, &
            HFBLK_nr_kj_O1, &
            HFBLK_nr_kj_O2, &
            HFBLK_nr_kj_O3, &
            HFBLK_nr_kl_O0, &
            HFBLK_nr_kl_O1, &
            HFBLK_nr_kl_O2, &
            HFBLK_nr_kl_O3, &
            HFBLK_nr_swap_ij, &
            HFBLK_nr_swap_ik_jl, &
            HFBLK_ij_O0, &
            HFBLK_ij_O1, &
            HFBLK_ij_O2, &
            HFBLK_ij_O3, &
            HFBLK_il_O0, &
            HFBLK_il_O1, &
            HFBLK_il_O2, &
            HFBLK_il_O3, &
            HFBLK_kj_O0, &
            HFBLK_kj_O1, &
            HFBLK_kj_O2, &
            HFBLK_kj_O3, &
            HFBLK_kl_O0, &
            HFBLK_kl_O1, &
            HFBLK_kl_O2, &
            HFBLK_kl_O3, &
            HFBLK_swap_ik_jl, &
            HFBLK_atimerev_ijtltk, &
            HFBLK_atimerev_tjtikl, &
            HFBLK_timerev_ijtltk, &
            HFBLK_timerev_tjtikl
contains
subroutine HFBLK_nr_ij_O0(fijkl, dm, vj, ndim, shls, ao_loc)
!**************************************************
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_ij = (ij|kl) * D_lk
!
!==================================================

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tdm(:,:)
  double precision,allocatable  ::  tmp(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(di,dj))
  allocate (tdm(dk,dl))
  do k = 1, dk
    tdm(k,:) = dm(lloc+1:lloc+dl,kloc+k)
  end do
  call dgemv("N", di*dj, dk*dl, 1d0, fijkl, di*dj, tdm, 1, 0d0, tmp, 1)
  vj(iloc+1:iloc+di,jloc+1:jloc+dj) = vj(iloc+1:iloc+di,jloc+1:jloc+dj) + tmp

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_ij_O0
!**************************************************


!**************************************************
subroutine HFBLK_nr_ij_O1(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_ij = (ij|kl) * D_lk + (ij|lk) * D_{kl}
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tdm(:,:)
  double precision,allocatable  ::  tmp(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (ij|kl) * D_lk + (ij|lk) * D_{kl}
  ! = (ij|kl) * (D_lk + D_{kl})
  allocate (tmp(di,dj))
  allocate (tdm(dk,dl))
  if (shls(3) == shls(4)) then
    do k = 1, dk
      tdm(k,:) = dm(lloc+1:lloc+dl,kloc+k)
    end do
  else
    do k = 1, dk
      tdm(k,:) = dm(kloc+k,lloc+1:lloc+dl) + dm(lloc+1:lloc+dl,kloc+k)
    end do
  end if
  call dgemv("N", di*dj, dk*dl, 1d0, fijkl, di*dj, tdm, 1, 0d0, tmp, 1)
  vj(iloc+1:iloc+di,jloc+1:jloc+dj) = vj(iloc+1:iloc+di,jloc+1:jloc+dj) + tmp

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_ij_O1
!**************************************************


!**************************************************
subroutine HFBLK_nr_ij_O2(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_ij
!  (i,j) = (ij|kl) * D_lk + (ij|kl) * D_{kl}
!  (j,i) = (ji|kl) * D_lk + (ji|kl) * D_{kl}
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (ij|kl) * D_lk + (ij|lk) * D_{kl}
  ! = (ij|kl) * (D_lk +/- D_{kl})
  allocate (tmp(di,dj))
  allocate (tdm(dk,dl))
  if (shls(3) == shls(4)) then
    do k = 1, dk
      tdm(k,:) = dm(lloc+1:lloc+dl,kloc+k)
    end do
  else
    do k = 1, dk
      tdm(k,:) = dm(kloc+k,lloc+1:lloc+dl) + dm(lloc+1:lloc+dl,kloc+k)
    end do
  end if
  call dgemv("N", di*dj, dk*dl, 1d0, fijkl, di*dj, tdm, 1, 0d0, tmp, 1)
  vj(iloc+1:iloc+di,jloc+1:jloc+dj) = vj(iloc+1:iloc+di,jloc+1:jloc+dj) + tmp

  if (shls(1) /= shls(2)) then
    ! (ji|kl) * D_lk
    do j = 1, dj
      do i = 1, di
        vj(jloc+j,iloc+i) = vj(jloc+j,iloc+i) + tmp(i,j)
      end do
    end do
  end if

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_ij_O2
!**************************************************


!**************************************************
subroutine HFBLK_nr_ij_O3(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
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

  call HFBLK_nr_ij_O2(fijkl, dm, vj, ndim, shls, ao_loc)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_nr_kl_O2(fijkl, dm, vj, ndim, shls, ao_loc)
  end if
end subroutine HFBLK_nr_ij_O3
!**************************************************


!**************************************************
subroutine HFBLK_nr_il_O0(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_il = (ij|kl) * D_jk
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(di,dl))
  allocate (tdm(dj,dk))
  do k = 1, dk
    tdm(:,k) = dm(jloc+1:jloc+dj,kloc+k)
  end do
  tmp = 0d0
  do l = 1, dl
    call dgemv("N", di, dj*dk, 1d0, fijkl(1,1,1,l), di, &
        tdm, 1, 1d0, tmp(1,l), 1)
  end do
  vk(iloc+1:iloc+di,lloc+1:lloc+dl) = vk(iloc+1:iloc+di,lloc+1:lloc+dl) + tmp

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_il_O0
!**************************************************


!**************************************************
subroutine HFBLK_nr_il_O1(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_il
!  (i,l) = (ij|kl) * D_jk
!  (i,k) = (ij|lk) * D_jl
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (i,l) = (ij|kl) * D_jk
  call HFBLK_nr_il_O0(fijkl, dm, vk, ndim, shls, ao_loc)

  allocate (tmp(di,dk))
  allocate (tdm(dj,dl))
  ! (i,k)= (ij|lk) * D_{lj}
  if (shls(3) /= shls(4)) then
    do l = 1, dl
      tdm(:,l) = dm(jloc+1:jloc+dj,lloc+l)
    end do
    tmp = 0d0
    do l = 1, dl
      do k = 1, dk
        call dgemv("N", di, dj, 1d0, fijkl(1,1,k,l), di, &
            tdm(1,l), 1, 1d0, tmp(1,k), 1)
      end do
    end do
    vk(iloc+1:iloc+di,kloc+1:kloc+dk) = vk(iloc+1:iloc+di,kloc+1:kloc+dk) + tmp
  end if

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_il_O1
!**************************************************


!**************************************************
subroutine HFBLK_nr_il_O2(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_il
!  (i,l) = (ij|kl) * D_jk
!  (i,k) = (ij|lk) * D_{j,l}
!  (j,l) = (ji|kl) * D_{i,k}
!  (j,k) = (ji|lk) * D_{i,l}
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

  integer                       ::  di, dj, dk, dl
  integer                       ::  shls_swp(4)
  double precision,allocatable  ::  fjikl(:,:,:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)

  call HFBLK_nr_il_O1(fijkl, dm, vk, ndim, shls, ao_loc)

  ! (j,l) = (ji|kl) * D_{i,k}
  ! (j,k) = (ji|lk) * D_{i,l}
  if (shls(1) /= shls(2)) then
    allocate (fjikl(dj,di,dk,dl))
    call HFBLK_nr_swap_ij(fijkl, fjikl, shls)
    shls_swp = (/shls(2), shls(1), shls(3), shls(4)/)
    call HFBLK_nr_il_O1(fjikl, dm, vk, ndim, shls_swp, ao_loc)
    deallocate (fjikl)
  end if
end subroutine HFBLK_nr_il_O2
!**************************************************


!**************************************************
subroutine HFBLK_nr_il_O3(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
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

  call HFBLK_nr_il_O2(fijkl, dm, vk, ndim, shls, ao_loc)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_nr_kj_O2(fijkl, dm, vk, ndim, shls, ao_loc)
  end if
end subroutine HFBLK_nr_il_O3
!**************************************************


!**************************************************
subroutine HFBLK_nr_kj_O0(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_kj = (ij|kl) * D_li
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (ij|kl) * D_li
  allocate (tmp(dj,dk))
  allocate (tdm(di,dl))
  do i = 1, di
    tdm(i,:) = dm(lloc+1:lloc+dl,iloc+i)
  end do
  tmp = 0d0
  do l = 1, dl
    call dgemv("T", di, dj*dk, 1d0, fijkl(1,1,1,l), di, &
        tdm(1,l), 1, 1d0, tmp, 1)
  end do
  do j = 1, dj
    vk(kloc+1:kloc+dk,jloc+j) = vk(kloc+1:kloc+dk,jloc+j) + tmp(j,:)
  end do

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_kj_O0
!**************************************************


!**************************************************
subroutine HFBLK_nr_kj_O1(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_kj
!  (k,j) = (ij|kl) * D_li
!  (l,j) = (ij|lk) * D_{ki}
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (k,j)  = (ij|kl) * D_li
  call HFBLK_nr_kj_O0(fijkl, dm, vk, ndim, shls, ao_loc)

  allocate (tmp(dj,dl))
  allocate (tdm(di,dk))
  if (shls(3) /= shls(4)) then
    ! (l,j) = (ij|lk) * D_{ki}
    do i = 1, di
      tdm(i,:) = dm(kloc+1:kloc+dk,iloc+i)
    end do

    tmp = 0d0
    do l = 1, dl
      do k = 1, dk
        call dgemv("T", di, dj, 1d0, fijkl(1,1,k,l), di, &
            tdm(1,k), 1, 1d0, tmp(1,l), 1)
      end do
    end do
    do j = 1, dj
        vk(lloc+1:lloc+dl,jloc+j) = vk(lloc+1:lloc+dl,jloc+j) + tmp(j,:)
    end do
  end if

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_kj_O1
!**************************************************


!**************************************************
subroutine HFBLK_nr_kj_O2(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_kj
!  (k,j) = (ij|kl) * D_li
!  (l,j) = (ij|lk) * D_{ki}
!  (k,i) = (ji|kl) * D_{lj}
!  (l,i) = (ji|lk) * D_{kj}
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

  integer                       ::  di, dj, dk, dl
  integer                       ::  shls_swp(4)
  double precision,allocatable  ::  fjikl(:,:,:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)

  call HFBLK_nr_kj_O1(fijkl, dm, vk, ndim, shls, ao_loc)

  ! (j,l) = (ji|kl) * D_{ki}
  ! (j,k) = (ji|lk) * D_{li}
  if (shls(1) /= shls(2)) then
    allocate (fjikl(dj,di,dk,dl))
    call HFBLK_nr_swap_ij(fijkl, fjikl, shls)
    shls_swp = (/shls(2), shls(1), shls(3), shls(4)/)
    call HFBLK_nr_kj_O1(fjikl, dm, vk, ndim, shls_swp, ao_loc)
    deallocate (fjikl)
  end if
end subroutine HFBLK_nr_kj_O2
!**************************************************


!**************************************************
subroutine HFBLK_nr_kj_O3(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
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

  call HFBLK_nr_kj_O2(fijkl, dm, vk, ndim, shls, ao_loc)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_nr_il_O2(fijkl, dm, vk, ndim, shls, ao_loc)
  end if
end subroutine HFBLK_nr_kj_O3
!**************************************************


!**************************************************
subroutine HFBLK_nr_kl_O0(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_kl = (ij|kl) * D_ji
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(dk,dl))
  allocate (tdm(di,dj))
  do i = 1, di
    tdm(i,:) = dm(jloc+1:jloc+dj,iloc+i)
  end do
  call dgemv("T", di*dj, dk*dl, 1d0, fijkl, di*dj, tdm, 1, 0d0, tmp, 1)
  vj(kloc+1:kloc+dk,lloc+1:lloc+dl) = vj(kloc+1:kloc+dk,lloc+1:lloc+dl) + tmp

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_kl_O0
!**************************************************


!**************************************************
subroutine HFBLK_nr_kl_O1(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_kl
!  (k,l) = (ij|kl) * D_ji
!  (l,k) = (ij|lk) * D_ji
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(dk,dl))
  allocate (tdm(di,dj))
  do i = 1, di
    tdm(i,:) = dm(jloc+1:jloc+dj,iloc+i)
  end do
  call dgemv("T", di*dj, dk*dl, 1d0, fijkl, di*dj, tdm, 1, 0d0, tmp, 1)
  vj(kloc+1:kloc+dk,lloc+1:lloc+dl) = vj(kloc+1:kloc+dk,lloc+1:lloc+dl) + tmp

  if (shls(3) /= shls(4)) then
    ! (ij|lk) * D_ji
    do l = 1, dl
      do k = 1, dk
        vj(lloc+l,kloc+k) = vj(lloc+l,kloc+k) + tmp(k,l)
      end do
    end do
  end if

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_kl_O1
!**************************************************


!**************************************************
subroutine HFBLK_nr_kl_O2(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  v_kl
!  (k,l) = (ij|kl) * D_ji + (ji|kl) * D_{ij}
!  (l,k) = (ij|lk) * D_ji + (ji|lk) * D_{ij}
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double precision,allocatable  ::  tmp(:,:)
  double precision,allocatable  ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(dk,dl))
  allocate (tdm(di,dj))
  if (shls(1) == shls(2)) then
    do i = 1, di
      tdm(i,:) = dm(jloc+1:jloc+dj,iloc+i)
    end do
  else
    ! (ij|kl) * D_lk + (ij|lk) * D_{k,l}
    ! = (ij|kl) * (D_lk +/- D_{k,l})
    do i = 1, di
      tdm(i,:) = dm(jloc+1:jloc+dj,iloc+i) + dm(iloc+i,jloc+1:jloc+dj)
    end do
  end if
  call dgemv("T", di*dj, dk*dl, 1d0, fijkl, di*dj, tdm, 1, 0d0, tmp, 1)
  vj(kloc+1:kloc+dk,lloc+1:lloc+dl) = vj(kloc+1:kloc+dk,lloc+1:lloc+dl) + tmp

  if (shls(3) /= shls(4)) then
    ! (ij|lk) * D_{k,l}
    do l = 1, dl
      do k = 1, dk
        vj(lloc+l,kloc+k) = vj(lloc+l,kloc+k) + tmp(k,l)
      end do
    end do
  end if

  deallocate (tmp, tdm)
end subroutine HFBLK_nr_kl_O2
!**************************************************


!**************************************************
subroutine HFBLK_nr_kl_O3(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double precision,intent(in)   ::  dm(ndim,ndim)
  double precision,intent(inout)::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  non-relativistic
!  use permutation symmetry
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

  call HFBLK_nr_kl_O2(fijkl, dm, vj, ndim, shls, ao_loc)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_nr_ij_O2(fijkl, dm, vj, ndim, shls, ao_loc)
  end if
end subroutine HFBLK_nr_kl_O3
!**************************************************


!**************************************************
subroutine HFBLK_nr_swap_ij(fijkl, fjikl, shls)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  double precision,intent(out)  ::  fjikl(:,:,:,:)
  integer,intent(in)            ::  shls(4)

! ==========
!
!  (ji|kl) from time-reversal symmetric (ij|kl)
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)

  do j = 1, dj
    do i = 1, di
      fjikl(j,i,:,:) = fijkl(i,j,:,:)
    end do
  end do
end subroutine HFBLK_nr_swap_ij
!**************************************************


!**************************************************
subroutine HFBLK_nr_swap_ik_jl(fijkl, fklij, shls)
  implicit none

  double precision,intent(in)   ::  fijkl(:,:,:,:)
  double precision,intent(out)  ::  fklij(:,:,:,:)
  integer,intent(in)            ::  shls(4)

! ==========
!
!  swap two electrons
!  (ij|kl) -> (kl|ij)
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

  integer                       ::  di, i, &
                                    dj, j, &
                                    dk, k, &
                                    dl, l

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  do l = 1, dl
    do k = 1, dk
      fklij(:,:,k,l) = fijkl(k,l,:,:)
    end do
  end do
end subroutine HFBLK_nr_swap_ik_jl
!**************************************************


subroutine HFBLK_ij_O0(fijkl, dm, vj, ndim, shls, ao_loc)
!**************************************************
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  v_ij = (ij|kl) * D'_kl
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

  double complex,parameter      ::  Z1 = 1d0
  double complex,parameter      ::  Z0 = 0d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(di,dj))
  allocate (tdm(dk,dl))
  do k = 1, dk
    tdm(k,:) = dm(lloc+1:lloc+dl,kloc+k)
  end do
  call zgemv("N", di*dj, dk*dl, Z1, fijkl, di*dj, tdm, 1, Z0, tmp, 1)
  vj(iloc+1:iloc+di,jloc+1:jloc+dj) = vj(iloc+1:iloc+di,jloc+1:jloc+dj) + tmp

  deallocate (tmp, tdm)
end subroutine HFBLK_ij_O0
!**************************************************


!**************************************************
subroutine HFBLK_ij_O1(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_ij = (ij|kl) * D'_kl + <Tl i|j Tk> * D'_{Tl,Tk}
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

  double complex,parameter      ::  Z1 = 1d0
  double complex,parameter      ::  Z0 = 0d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(di,dj))
  allocate (tdm(dk,dl))

  ! (ij|kl) * D_lk + (ji|Tl Tk) * D_{Tk,Tl}
  ! = (ij|kl) * (D_lk +/- D_{Tk,Tl})
  if (shls(3) == shls(4)) then
    do k = 1, dk
      tdm(k,:) = dm(lloc+1:lloc+dl,kloc+k)
    end do
  else
    do l = 1, dl
      if (tao(lloc+l) > 0) then
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            tdm(k,l) = dm(lloc+l,kloc+k) + dm( tao(kloc+k),tao(lloc+l))
          else
            tdm(k,l) = dm(lloc+l,kloc+k) - dm(-tao(kloc+k),tao(lloc+l))
          end if
        end do
      else
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            tdm(k,l) = dm(lloc+l,kloc+k) - dm( tao(kloc+k),-tao(lloc+l))
          else
            tdm(k,l) = dm(lloc+l,kloc+k) + dm(-tao(kloc+k),-tao(lloc+l))
          end if
        end do
      end if
    end do
  end if
  call zgemv("N", di*dj, dk*dl, Z1, fijkl, di*dj, tdm, 1, Z0, tmp, 1)
  vj(iloc+1:iloc+di,jloc+1:jloc+dj) = vj(iloc+1:iloc+di,jloc+1:jloc+dj) + tmp

  deallocate (tmp, tdm)
end subroutine HFBLK_ij_O1
!**************************************************


!**************************************************
subroutine HFBLK_ij_O2(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_ij
!  (i,j) = (ij|kl) * D'_kl + (ij|Tl Tk) * D'_{Tl,Tk}
!  (Tj,Ti) = (Tj Ti|kl) * D'_kl + (Tj Ti|Tl Tk) * D'_{Tl,Tk}
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

  double complex,parameter      ::  Z1 = 1d0
  double complex,parameter      ::  Z0 = 0d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(di,dj))
  allocate (tdm(dk,dl))

  ! (ij|kl) * D_lk + (ji|Tl Tk) * D_{Tk,Tl}
  ! = (ij|kl) * (D_lk +/- D_{Tk,Tl})
  if (shls(3) == shls(4)) then
    do k = 1, dk
      tdm(k,:) = dm(lloc+1:lloc+dl,kloc+k)
    end do
  else
    do l = 1, dl
      if (tao(lloc+l) > 0) then
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            tdm(k,l) = dm(lloc+l,kloc+k) + dm( tao(kloc+k),tao(lloc+l))
          else
            tdm(k,l) = dm(lloc+l,kloc+k) - dm(-tao(kloc+k),tao(lloc+l))
          end if
        end do
      else
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            tdm(k,l) = dm(lloc+l,kloc+k) - dm( tao(kloc+k),-tao(lloc+l))
          else
            tdm(k,l) = dm(lloc+l,kloc+k) + dm(-tao(kloc+k),-tao(lloc+l))
          end if
        end do
      end if
    end do
  end if
  call zgemv("N", di*dj, dk*dl, Z1, fijkl, di*dj, tdm, 1, Z0, tmp, 1)
  vj(iloc+1:iloc+di,jloc+1:jloc+dj) = vj(iloc+1:iloc+di,jloc+1:jloc+dj) + tmp

  ! v(Tj,Ti)
  if (shls(1) /= shls(2)) then
    do j = 1, dj
      if (tao(jloc+j) > 0) then
        do i = 1, di
          if (tao(iloc+i) > 0) then
            vj(tao(jloc+j), tao(iloc+i)) = vj(tao(jloc+j), tao(iloc+i)) + tmp(i,j)
          else
            vj(tao(jloc+j),-tao(iloc+i)) = vj(tao(jloc+j),-tao(iloc+i)) - tmp(i,j)
          end if
        end do
      else
        do i = 1, di
          if (tao(iloc+i) > 0) then
            vj(-tao(jloc+j), tao(iloc+i)) = vj(-tao(jloc+j), tao(iloc+i)) - tmp(i,j)
          else
            vj(-tao(jloc+j),-tao(iloc+i)) = vj(-tao(jloc+j),-tao(iloc+i)) + tmp(i,j)
          end if
        end do
      end if
    end do
  end if

  deallocate (tmp, tdm)
end subroutine HFBLK_ij_O2
!**************************************************


!**************************************************
subroutine HFBLK_ij_O3(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!
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

  call HFBLK_ij_O2(fijkl, dm, vj, ndim, shls, ao_loc, tao)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_kl_O2(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  end if
end subroutine HFBLK_ij_O3
!**************************************************


!**************************************************
subroutine HFBLK_il_O0(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  v_il = (ij|kl) * D'_kj
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

  double complex,parameter      ::  Z1 = 1d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  !allocate (tmp(di,dl))
  allocate (tdm(dj,dk))
  do k = 1, dk
    tdm(:,k) = dm(jloc+1:jloc+dj,kloc+k)
  end do
  do l = 1, dl
    call zgemv("N", di, dj*dk, Z1, fijkl(1,1,1,l), di, &
        tdm, 1, Z1, vk(iloc+1,lloc+l), 1)
  end do

  deallocate (tdm)
end subroutine HFBLK_il_O0
!**************************************************


!**************************************************
subroutine HFBLK_il_O1(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_il
!  (i,l)  = (ij|kl) * D'_kj
!  (i,Tk) = <Tl i|j Tk> * D'_{Tl,j}
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

  double complex,parameter      ::  Z1 = 1d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  integer                       ::  shls_swp(4)
  double complex,allocatable    ::  tijkl(:,:,:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (i,l) = (ij|kl) * D_jk
  call HFBLK_il_O0(fijkl, dm, vk, ndim, shls, ao_loc)


  ! (i,Tk)= +/-(ij|kl) * D_{j,Tl}
  if (shls(3) /= shls(4)) then
    allocate (tmp(di,dk))
    tmp = 0d0
    do l = 1, dl
      if (tao(lloc+l) > 0) then
        do k = 1, dk
          call zgemv("N", di, dj, Z1, fijkl(1,1,k,l), di, &
              dm(jloc+1,tao(lloc+l)), 1, Z1, tmp(1,k), 1)
        end do
      else
        do k = 1, dk
          call zgemv("N", di, dj, -Z1, fijkl(1,1,k,l), di, &
              dm(jloc+1,-tao(lloc+l)), 1, Z1, tmp(1,k), 1)
        end do
      end if
    end do

    do k = 1, dk
      if (tao(kloc+k) > 0) then
        vk(iloc+1:iloc+di, tao(kloc+k)) = vk(iloc+1:iloc+di, tao(kloc+k)) + tmp(:,k)
      else
        vk(iloc+1:iloc+di,-tao(kloc+k)) = vk(iloc+1:iloc+di,-tao(kloc+k)) - tmp(:,k)
      end if
    end do
    deallocate (tmp)
  end if
end subroutine HFBLK_il_O1
!**************************************************


!**************************************************
subroutine HFBLK_il_O2(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_il
!  (i,l)  = (ij|kl) * D_jk
!  (i,Tk) = (ij|Tl Tk) * D_{j,Tl}
!  (Tj,l) = (Tj Ti|kl) * D_{Ti,k}
!  (Tj,Tk)= (Tj Ti|Tl Tk) * D_{Ti,Tl}
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

  integer                       ::  di, dj, dk, dl
  integer                       ::  shls_swp(4)
  double complex,allocatable    ::  tijkl(:,:,:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)

  call HFBLK_il_O1(fijkl, dm, vk, ndim, shls, ao_loc, tao)

  if (shls(1) /= shls(2)) then
    allocate (tijkl(dj,di,dk,dl))
    call HFBLK_timerev_tjtikl(fijkl, tijkl, shls, ao_loc, tao)
    shls_swp = (/shls(2), shls(1), shls(3), shls(4)/)
    call HFBLK_il_O1(tijkl, dm, vk, ndim, shls_swp, ao_loc, tao)
    deallocate (tijkl)
  end if
end subroutine HFBLK_il_O2
!**************************************************


!**************************************************
subroutine HFBLK_il_O3(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!
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

  call HFBLK_il_O2(fijkl, dm, vk, ndim, shls, ao_loc, tao)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_kj_O2(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  end if
end subroutine HFBLK_il_O3
!**************************************************


!**************************************************
subroutine HFBLK_kj_O0(fijkl, dm, vk, ndim, shls, ao_loc)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  v_kj = (ij|kl) * D_li
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

  double complex,parameter      ::  Z1 = 1d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (ij|kl) * D_li
  allocate (tmp(dj,dk))
  allocate (tdm(di,dl))
  do i = 1, di
    tdm(i,:) = dm(lloc+1:lloc+dl,iloc+i)
  end do
  tmp = 0d0
  do l = 1, dl
    call zgemv("T", di, dj*dk, Z1, fijkl(1,1,1,l), di, &
        tdm(1,l), 1, Z1, tmp, 1)
  end do
  do j = 1, dj
    vk(kloc+1:kloc+dk,jloc+j) = vk(kloc+1:kloc+dk,jloc+j) + tmp(j,:)
  end do

  deallocate (tmp, tdm)
end subroutine HFBLK_kj_O0
!**************************************************


!**************************************************
subroutine HFBLK_kj_O1(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_kj
!  (k,j)  = (ij|kl) * D_li
!  (Tl,j) = (ij|Tl Tk) * D_{Tk,i}
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

  double complex,parameter      ::  Z1 = 1d0, Z0 = 0d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  ! (k,j)  = (ij|kl) * D'_il
  call HFBLK_kj_O0(fijkl, dm, vk, ndim, shls, ao_loc)

  if (shls(3) /= shls(4)) then
    ! (Tl,j) = +/-(ij|kl) * D_{Tk,i}
    allocate (tmp(dj,dl))
    allocate (tdm(di,dk))
    do k = 1, dk
      if (tao(kloc+k) > 0) then
        tdm(:,k) =  dm( tao(kloc+k),iloc+1:iloc+di)
      else
        tdm(:,k) = -dm(-tao(kloc+k),iloc+1:iloc+di)
      end if
    end do

    tmp = 0d0
    do l = 1, dl
      do k = 1, dk
        call zgemv("T", di, dj, Z1, fijkl(1,1,k,l), di, &
            tdm(1,k), 1, Z1, tmp(1,l), 1)
      end do
    end do

    do l = 1, dl
      if (tao(lloc+l) > 0) then
        vk( tao(lloc+l),jloc+1:jloc+dj) = vk( tao(lloc+l),jloc+1:jloc+dj) + tmp(:,l)
      else
        vk(-tao(lloc+l),jloc+1:jloc+dj) = vk(-tao(lloc+l),jloc+1:jloc+dj) - tmp(:,l)
      end if
    end do
    deallocate (tmp, tdm)
  end if
end subroutine HFBLK_kj_O1
!**************************************************


!**************************************************
subroutine HFBLK_kj_O2(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_kj
!  (k,j)  = (ij|kl) * D_li
!  (k,Ti) = (Tj Ti|kl) * D_{l,Tj}
!  (Tl,j) = (ij|Tl Tk) * D_{Tk,i}
!  (Tl,Ti)= (Tj Ti|Tl Tk) * D_{Tk,Tj}
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

  integer                       ::  di, dj, dk, dl
  integer                       ::  shls_swp(4)
  double complex,allocatable    ::  tijkl(:,:,:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)

  call HFBLK_kj_O1(fijkl, dm, vk, ndim, shls, ao_loc, tao)

  if (shls(1) /= shls(2)) then
    allocate (tijkl(dj,di,dk,dl))
    call HFBLK_timerev_tjtikl(fijkl, tijkl, shls, ao_loc, tao)
    shls_swp = (/shls(2), shls(1), shls(3), shls(4)/)
    call HFBLK_kj_O1(tijkl, dm, vk, ndim, shls_swp, ao_loc, tao)
    deallocate (tijkl)
  end if
end subroutine HFBLK_kj_O2
!**************************************************


!**************************************************
subroutine HFBLK_kj_O3(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vk(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!
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

  call HFBLK_kj_O2(fijkl, dm, vk, ndim, shls, ao_loc, tao)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_il_O2(fijkl, dm, vk, ndim, shls, ao_loc, tao)
  end if
end subroutine HFBLK_kj_O3
!**************************************************


!**************************************************
subroutine HFBLK_kl_O0(fijkl, dm, vj, ndim, shls, ao_loc)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)

! ==========
!
!  v_kl = (ij|kl) * D_ji
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

  double complex,parameter      ::  Z1 = 1d0
  double complex,parameter      ::  Z0 = 0d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(dk,dl))
  allocate (tdm(di,dj))
  do i = 1, di
    tdm(i,:) = dm(jloc+1:jloc+dj,iloc+i)
  end do
  call zgemv("T", di*dj, dk*dl, Z1, fijkl, di*dj, tdm, 1, Z0, tmp, 1)
  vj(kloc+1:kloc+dk,lloc+1:lloc+dl) = vj(kloc+1:kloc+dk,lloc+1:lloc+dl) + tmp

  deallocate (tmp, tdm)
end subroutine HFBLK_kl_O0
!**************************************************


!**************************************************
subroutine HFBLK_kl_O1(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_kl
!  (k,l) = (ij|kl) * D_ji
!  (Tl,Tk) = (ij|Tl Tk) * D_ji
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

  double complex,parameter      ::  Z1 = 1d0
  double complex,parameter      ::  Z0 = 0d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(dk,dl))
  allocate (tdm(di,dj))
  do i = 1, di
    tdm(i,:) = dm(jloc+1:jloc+dj,iloc+i)
  end do
  call zgemv("T", di*dj, dk*dl, Z1, fijkl, di*dj, tdm, 1, Z0, tmp, 1)
  vj(kloc+1:kloc+dk,lloc+1:lloc+dl) = vj(kloc+1:kloc+dk,lloc+1:lloc+dl) + tmp

  if (shls(3) /= shls(4)) then
    do l = 1, dl
      if (tao(lloc+l) > 0) then
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            vj(tao(lloc+l), tao(kloc+k)) = vj(tao(lloc+l), tao(kloc+k)) + tmp(k,l)
          else
            vj(tao(lloc+l),-tao(kloc+k)) = vj(tao(lloc+l),-tao(kloc+k)) - tmp(k,l)
          end if
        end do
      else
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            vj(-tao(lloc+l), tao(kloc+k)) = vj(-tao(lloc+l), tao(kloc+k)) - tmp(k,l)
          else
            vj(-tao(lloc+l),-tao(kloc+k)) = vj(-tao(lloc+l),-tao(kloc+k)) + tmp(k,l)
          end if
        end do
      end if
    end do
  end if

  deallocate (tmp, tdm)
end subroutine HFBLK_kl_O1
!**************************************************


!**************************************************
subroutine HFBLK_kl_O2(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  v_kl
!  (k,l) = (ij|kl) * D_ji + (Tj Ti|kl) * D_{Ti,Tj}
!  (Tl,Tk) = (ij|Tl Tk) * D_ji + (Tj Ti|Tk Tl) * D_{Ti,Tj}
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

  double complex,parameter      ::  Z1 = 1d0
  double complex,parameter      ::  Z0 = 0d0
  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l
  double complex,allocatable    ::  tmp(:,:)
  double complex,allocatable    ::  tdm(:,:)

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  allocate (tmp(dk,dl))
  allocate (tdm(di,dj))

  if (shls(1) == shls(2)) then
    do i = 1, di
      tdm(i,:) = dm(jloc+1:jloc+dj,iloc+i)
    end do

  else
    do j = 1, dj
      if (tao(jloc+j) > 0) then
        do i = 1, di
          if (tao(iloc+i) > 0) then
            tdm(i,j) = dm(jloc+j,iloc+i) + dm( tao(iloc+i),tao(jloc+j))
          else
            tdm(i,j) = dm(jloc+j,iloc+i) - dm(-tao(iloc+i),tao(jloc+j))
          end if
        end do
      else
        do i = 1, di
          if (tao(iloc+i) > 0) then
            tdm(i,j) = dm(jloc+j,iloc+i) - dm( tao(iloc+i),-tao(jloc+j))
          else
            tdm(i,j) = dm(jloc+j,iloc+i) + dm(-tao(iloc+i),-tao(jloc+j))
          end if
        end do
      end if
    end do
  end if
  call zgemv("T", di*dj, dk*dl, Z1, fijkl, di*dj, tdm, 1, Z0, tmp, 1)
  vj(kloc+1:kloc+dk,lloc+1:lloc+dl) = vj(kloc+1:kloc+dk,lloc+1:lloc+dl) + tmp

  if (shls(3) /= shls(4)) then
    do l = 1, dl
      if (tao(lloc+l) > 0) then
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            vj(tao(lloc+l), tao(kloc+k)) = vj(tao(lloc+l), tao(kloc+k)) + tmp(k,l)
          else
            vj(tao(lloc+l),-tao(kloc+k)) = vj(tao(lloc+l),-tao(kloc+k)) - tmp(k,l)
          end if
        end do
      else
        do k = 1, dk
          if (tao(kloc+k) > 0) then
            vj(-tao(lloc+l), tao(kloc+k)) = vj(-tao(lloc+l), tao(kloc+k)) - tmp(k,l)
          else
            vj(-tao(lloc+l),-tao(kloc+k)) = vj(-tao(lloc+l),-tao(kloc+k)) + tmp(k,l)
          end if
        end do
      end if
    end do
  end if

  deallocate (tmp)
end subroutine HFBLK_kl_O2
!**************************************************


!**************************************************
subroutine HFBLK_kl_O3(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  integer,intent(in)            ::  ndim
  double complex,intent(in)     ::  dm(ndim,ndim)
  double complex,intent(inout)  ::  vj(ndim,ndim)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  use permutation symmetry
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

  call HFBLK_kl_O2(fijkl, dm, vj, ndim, shls, ao_loc, tao)

  ! permutation symmetry (ij|kl) -> (kl|ij)
  if (shls(1) /= shls(3) .or. shls(2) /= shls(4)) then
    call HFBLK_ij_O2(fijkl, dm, vj, ndim, shls, ao_loc, tao)
  end if
end subroutine HFBLK_kl_O3
!**************************************************


!**************************************************
subroutine HFBLK_atimerev_ijtltk(fijkl, ijtkl, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  double complex,intent(out)    ::  ijtkl(:,:,:,:)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  (ij|Tl Tk) from time-reversal anti-symmetric (ij|kl)
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  do l = 1, dl
    if (tao(lloc+l) > 0) then
      do k = 1, dk
        if (tao(kloc+k) > 0) then
          ijtkl(:,:,tao(lloc+l)-lloc, tao(kloc+k)-kloc) =-fijkl(:,:,k,l)
        else
          ijtkl(:,:,tao(lloc+l)-lloc,-tao(kloc+k)-kloc) = fijkl(:,:,k,l)
        end if
      end do
    else
      do k = 1, dk
        if (tao(kloc+k) > 0) then
          ijtkl(:,:,-tao(lloc+l)-lloc, tao(kloc+k)-kloc) = fijkl(:,:,k,l)
        else
          ijtkl(:,:,-tao(lloc+l)-lloc,-tao(kloc+k)-kloc) =-fijkl(:,:,k,l)
        end if
      end do
    end if
  end do
end subroutine HFBLK_atimerev_ijtltk
!**************************************************


!**************************************************
subroutine HFBLK_atimerev_tjtikl(fijkl, tijkl, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  double complex,intent(out)    ::  tijkl(:,:,:,:)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  (Tj Ti|kl) from time-reversal anti-symmetric (ij|kl)
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)

  do j = 1, dj
    if (tao(jloc+j) > 0) then
      do i = 1, di
        if (tao(iloc+i) > 0) then
          tijkl(tao(jloc+j)-jloc, tao(iloc+i)-iloc,:,:) =-fijkl(i,j,:,:)
        else
          tijkl(tao(jloc+j)-jloc,-tao(iloc+i)-iloc,:,:) = fijkl(i,j,:,:)
        end if
      end do
    else
      do i = 1, di
        if (tao(iloc+i) > 0) then
          tijkl(-tao(jloc+j)-jloc, tao(iloc+i)-iloc,:,:) = fijkl(i,j,:,:)
        else
          tijkl(-tao(jloc+j)-jloc,-tao(iloc+i)-iloc,:,:) =-fijkl(i,j,:,:)
        end if
      end do
    end if
  end do
end subroutine HFBLK_atimerev_tjtikl
!**************************************************


!**************************************************
subroutine HFBLK_swap_ik_jl(fijkl, fklij, shls)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  double complex,intent(out)    ::  fklij(:,:,:,:)
  integer,intent(in)            ::  shls(4)

! ==========
!
!  swap two electrons
!  (ij|kl) -> (kl|ij)
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

  integer                       ::  di, i, &
                                    dj, j, &
                                    dk, k, &
                                    dl, l

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  do l = 1, dl
    do k = 1, dk
      fklij(k,l,:,:) = fijkl(:,:,k,l)
    end do
  end do
end subroutine HFBLK_swap_ik_jl
!**************************************************


!**************************************************
subroutine HFBLK_timerev_tjtikl(fijkl, tijkl, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  double complex,intent(out)    ::  tijkl(:,:,:,:)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  (Tj Ti|kl) from time-reversal symmetric (ij|kl)
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  iloc = ao_loc(shls(1)+1)
  jloc = ao_loc(shls(2)+1)

  do j = 1, dj
    if (tao(jloc+j) > 0) then
      do i = 1, di
        if (tao(iloc+i) > 0) then
          tijkl(tao(jloc+j)-jloc, tao(iloc+i)-iloc,:,:) = fijkl(i,j,:,:)
        else
          tijkl(tao(jloc+j)-jloc,-tao(iloc+i)-iloc,:,:) =-fijkl(i,j,:,:)
        end if
      end do
    else
      do i = 1, di
        if (tao(iloc+i) > 0) then
          tijkl(-tao(jloc+j)-jloc, tao(iloc+i)-iloc,:,:) =-fijkl(i,j,:,:)
        else
          tijkl(-tao(jloc+j)-jloc,-tao(iloc+i)-iloc,:,:) = fijkl(i,j,:,:)
        end if
      end do
    end if
  end do
end subroutine HFBLK_timerev_tjtikl
!**************************************************


!**************************************************
subroutine HFBLK_timerev_ijtltk(fijkl, ijtkl, shls, ao_loc, tao)
  implicit none

  double complex,intent(in)     ::  fijkl(:,:,:,:)
  double complex,intent(out)    ::  ijtkl(:,:,:,:)
  integer,intent(in)            ::  shls(4)
  integer,intent(in)            ::  ao_loc(*)
  integer,intent(in)            ::  tao(*)

! ==========
!
!  (ij|Tl Tk) from time-reversal symmetric (ij|kl)
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

  integer                       ::  iloc, di, i, &
                                    jloc, dj, j, &
                                    kloc, dk, k, &
                                    lloc, dl, l

  di = size(fijkl, 1)
  dj = size(fijkl, 2)
  dk = size(fijkl, 3)
  dl = size(fijkl, 4)
  kloc = ao_loc(shls(3)+1)
  lloc = ao_loc(shls(4)+1)

  do l = 1, dl
    if (tao(lloc+l) > 0) then
      do k = 1, dk
        if (tao(kloc+k) > 0) then
          ijtkl(:,:,tao(lloc+l)-lloc, tao(kloc+k)-kloc) = fijkl(:,:,k,l)
        else
          ijtkl(:,:,tao(lloc+l)-lloc,-tao(kloc+k)-kloc) =-fijkl(:,:,k,l)
        end if
      end do
    else
      do k = 1, dk
        if (tao(kloc+k) > 0) then
          ijtkl(:,:,-tao(lloc+l)-lloc, tao(kloc+k)-kloc) =-fijkl(:,:,k,l)
        else
          ijtkl(:,:,-tao(lloc+l)-lloc,-tao(kloc+k)-kloc) = fijkl(:,:,k,l)
        end if
      end do
    end if
  end do
end subroutine HFBLK_timerev_ijtltk
!**************************************************
end module hf_block_mod
