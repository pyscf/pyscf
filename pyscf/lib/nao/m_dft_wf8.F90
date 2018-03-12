! Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

module m_dft_wf8

#include "m_define_macro.F90"
  use m_die, only : die
  use m_precision, only : mpi_int  
  implicit none

  type dft_wf8_t
    integer(mpi_int)     :: desc(50)= -999 ! Scalapack descriptor of eigen vectors X
    real(8), allocatable :: kpoints(:,:)   ! (xyz, kpoint)
    real(8), allocatable :: X(:,:,:,:,:)   ! (reim, orbital, state, spin, kpoint)
    real(8), allocatable :: E(:,:,:)       !                (state, spin, kpoint)
    real(8) :: fermi_energy  = -999        ! Fermi energy
    real(8) :: eigenvalues_shift = -999    ! E = E_orig + eigenvalues_shift
                                           ! H X = E_orig S X
    character(100) :: BlochPhaseConv = ""  ! TextBook or Simple
        
  end type ! dft_wf8_t

contains



subroutine dealloc(v)
  implicit none
  type(dft_wf8_t), intent(inout) :: v
  _dealloc(v%kpoints)
  _dealloc(v%E)
  _dealloc(v%X)
  v%desc = -999
  v%fermi_energy = -999
  v%eigenvalues_shift = -999
  v%BlochPhaseConv = ""
  
end subroutine ! dealloc

!
!
! 
subroutine get_X4_noxv_8(wf, wf_4, i1, i2, i3, no, Fmin, svX_padd, ptr, ptr_aF, orb2padd)
  use m_dft_wf4, only : dft_wf4_t
  use m_precision, only : c_integer

  implicit none
  type(dft_wf8_t), intent(inout) :: wf
  type(dft_wf4_t), intent(inout), target :: wf_4
  integer(c_integer), intent(in) :: no
  integer, intent(in) :: i1, i2, i3, Fmin
  real(4), allocatable, intent(inout), target :: svX_padd(:,:,:,:,:)
  real(4), intent(inout), pointer :: ptr(:, :), ptr_aF(:, :)
  real(4), intent(in), optional :: orb2padd(:)

  integer :: i
  integer :: uu(5)

  if(.not. present(orb2padd) .and. associated(ptr)) then
    return
  endif

  ptr => null()
  ptr_aF => null()
  if (allocated(wf%X) .and. (.not. allocated(wf_4%X))) then
    _dealloc(wf_4%X)
    allocate(wf_4%X(lbound(wf%X, 1):ubound(wf%X, 1), lbound(wf%X, 2):ubound(wf%X, 2), &
      lbound(wf%X, 3):ubound(wf%X, 3), lbound(wf%X, 4):ubound(wf%X, 4), &
      lbound(wf%X, 5):ubound(wf%X, 5)))

    wf_4%X = real(wf%X, 4)
    _dealloc(wf%X)
  else if ((.not. allocated(wf%X)) .and. (.not. allocated(wf_4%X))) then
    _die('wf%X and wf_4%X not allocated')
  endif

  if(present(orb2padd)) then
    uu = ubound(wf_4%X)
    _dealloc(svX_padd)
    allocate(svX_padd(uu(1),uu(2),uu(3),uu(4),uu(5)))
    call SCOPY(size(wf_4%X),wf_4%X,1,svX_padd,1)
    do i=1,no; svX_padd(i1,1:no,i,i2,i3) = svX_padd(i1,1:no,i,i2,i3)*orb2padd(i); enddo
    ptr => svX_padd(i1, :, :, i2, i3)
  else
    ptr => wf_4%X(i1, :, :, i2, i3)
  endif
  ptr_aF => wf_4%X(i1,:,Fmin:no,i2,i3)

end subroutine !get_X4_noxv

!
!
!
subroutine get_eigvec_cmplx(wf, n, spin, k, zX)
  implicit none
  !! external
  type(dft_wf8_t), intent(in) :: wf
  integer, intent(in) :: n, spin, k
  complex(8), intent(inout) :: zX(:)
  !! internal
  integer :: nreim, norbs, neigv, nspin, nkp
  call check_dims(n, spin, k, wf)
  call get_dims(wf, nreim, norbs, neigv, nspin, nkp)
  if(size(zX)<norbs) _die('size(zX)<norbs')

  if(nreim==1) then  
    zX(1:norbs) = wf%X(1,1:norbs, n, spin, k)
  else if (nreim == 2) then
    zX(1:norbs) = cmplx(wf%X(1,1:norbs, n, spin, k), wf%X(2,1:norbs, n, spin, k), 8)
  else 
    _die('wrong nreim')
  endif    
      
end subroutine ! get_eigvec_cmplx

!
!
!
subroutine get_eigvec_real(wf, n, spin, k, dX)
  implicit none
  !! external
  type(dft_wf8_t), intent(in) :: wf
  integer, intent(in) :: n, spin, k
  real(8), intent(inout) :: dX(:)
  !! internal
  integer :: nreim, norbs, neigv, nspin, nkp
  call check_dims(n, spin, k, wf)
  call get_dims(wf, nreim, norbs, neigv, nspin, nkp)
  if(nreim/=1) _die('!nreim/=1')
  if(size(dX)<norbs) _die('size(zX)<norbs')

  dX(1:norbs) = wf%X(1,1:norbs, n, spin, k)
      
end subroutine ! get_eigvec_real


!
!
!
subroutine set_eigvec_cmplx(zX, n, spin, k, wf)
  implicit none
  !! external
  complex(8), intent(in) :: zX(:)  
  integer, intent(in) :: n, spin, k
  type(dft_wf8_t), intent(inout) :: wf

  !! internal
  integer :: nreim, norbs, neigv, nspin, nkp
  call check_dims(n, spin, k, wf)
  call get_dims(wf, nreim, norbs, neigv, nspin, nkp)
  if(size(zX)<norbs) _die('size(zX)<norbs')
  
  if(nreim==1) then
    wf%X(1,1:norbs, n, spin, k) = real(zX(1:norbs),8)
  else if (nreim==2) then
    wf%X(1,1:norbs, n, spin, k) = real(zX(1:norbs),8)
    wf%X(2,1:norbs, n, spin, k) = aimag(zX(1:norbs))
  else
    _die('wrong nreim')
  endif  
      
end subroutine ! set_eigvec_cmplx

!
!
!
subroutine check_dims(n, spin, k, wf)
  implicit none
  integer, intent(in) :: n, spin, k
  type(dft_wf8_t), intent(in) :: wf
  !! internal
  integer :: norbs, nspin, nkp, neigv, nreim

  call get_dims(wf, nreim, norbs, neigv, nspin, nkp)
  if(nreim<1 .or. nreim>2) _die('nreim<1 .or. nreim>2')
  if(nspin<1 .or. nspin>2) _die('nspin<1 .or. nspin>2')
  if(norbs<1) _die('norbs<1')
  if(nkp<1) _die('nkp<1')
  if(neigv<1) _die('neigv<1')
    
  nkp   = get_nkpoints(wf)
  if(n<1 .or. n>neigv) _die('n<1 .or. n>neigv')
  if(k<1 .or. k>nkp) _die('k<1 .or. k>nkp')
  if(spin<1 .or. spin>nspin) _die('spin<1 .or. spin>nspin')

end subroutine ! check_dims

!
!
!
integer function get_basis_type(wf)
  implicit none
  type(dft_wf8_t), intent(in) :: wf
  !! internal
  integer :: n
  if(.not. allocated(wf%X)) _die(".not. allocated(X)")
  n = size(wf%X,1)
  if(n<1 .or. n>2) _die('n<1 .or. n>2')
  get_basis_type = n
end function ! get_basis_type
  
!
!
!
subroutine set_BlochPhaseConv(conv, wf)
  use m_upper, only : upper
  implicit none
  character(*), intent(in) :: conv
  type(dft_wf8_t), intent(inout) :: wf
  if(len_trim(conv)<1) _die('!BlochPhaseConv')
  wf%BlochPhaseConv = upper(conv)
end subroutine  ! set_BlochPhaseConv


!
!
!
character(100) function get_BlochPhaseConv(wf) 
  use m_upper, only : upper
  implicit none
  type(dft_wf8_t), intent(in) :: wf
  if(len_trim(wf%BlochPhaseConv)<1) _die('!BlochPhaseConv')
  get_BlochPhaseConv = upper(wf%BlochPhaseConv)
end function ! get_BlochPhaseConv  


!
!
!
function get_kvec(wf, k) result(kvec)
  implicit none
  !! external
  type(dft_wf8_t), intent(in) :: wf
  integer, intent(in) :: k
  real(8) :: kvec(3)
  !! internal
  integer :: nkp
  nkp = get_nkpoints(wf)
  if(k<1 .or. k>nkp) _die('k<1 .or. k>nkp')
  
  kvec = wf%kpoints(:,k)
end function ! get_kvec


!!
!! Get Dims -- probable we will have several subroutines of this type
!! when we have second subroutine of this type, we will define interface...
!!
subroutine get_dims(wf, nreim, norbs, neigv, nspin, nkp)
  implicit none
  !! external
  type(dft_wf8_t), intent(in) :: wf
  integer, intent(out) :: nreim, norbs, neigv, nspin, nkp
  
  if(.not. allocated(wf%X)) _die('!X')
  if(.not. allocated(wf%E)) _die('!E')
  
  nreim = size(wf%X,1)
  norbs = size(wf%X,2)
  neigv = size(wf%X,3)
  nspin = size(wf%X,4)
  nkp   = size(wf%X,5)
  
  if(nspin<1 .or. nspin>2) _die('nspin<1 .or. nspin>2')
  
  if(nkp/=size(wf%E,3)) _die('nkp/=size(E,3)')
  if(nspin/=size(wf%E,2)) _die('nspin/=size(E,2)')
  if(neigv/=size(wf%E,1)) _die('neigv/=size(wf%E,1)')

end subroutine ! get_dims

!
!
!
function get_nkpoints(a) result(n)
  implicit none
  !! external
  type(dft_wf8_t), intent(in) :: a
  integer :: n
  !! internal
  integer :: na(3)
  
  na = 0
  if(allocated(a%kpoints)) na(1) = size(a%kpoints,2)
  if(allocated(a%X)) na(2) = size(a%X,5)
  if(allocated(a%E)) na(3) = size(a%E,3)
  n = na(1)
  if(any(na/=n)) _die('any(na/=n)')
  if(n<1) _die('n<1')

end function !  get_nkpoints 
  
!
! Computes DFT DOS as a simple sum over poles
!
subroutine dft_wf_sum_poles2dos(dft_wf, fermi_energy, d_omega, eps, dos_output)
  implicit none
  !! external
  type(dft_wf8_t), intent(in) :: dft_wf
  real(8), intent(in) :: d_omega, eps, fermi_energy
  real(8), allocatable, intent(inout) :: dos_output(:)

  !! internal
  integer :: nff_min, nff_max, kpoint, spin, eigenvalue, nkpoints, nspin, neigen, f
  real(8) :: dos, omega
  real(8) :: pi
  pi = 4D0 * atan(1D0)

  nff_max  = ubound(dos_output,1)
  nff_min  = lbound(dos_output,1)
  neigen   = size(dft_wf%E,1)
  nspin    = size(dft_wf%E,2)
  nkpoints = size(dft_wf%E,3)

  do f=nff_min, nff_max
    omega = f*d_omega

    dos = 0
    do kpoint=1,nkpoints
      do spin=1,nspin
        do eigenvalue=1,neigen
          dos = dos + aimag(1.0D0/(cmplx(omega - dft_wf%E(eigenvalue,spin,kpoint) + fermi_energy, eps,8)))
        end do
      end do
    end do
    dos_output(f) = -dos/pi

  end do

end subroutine !dft_wf_sum_poles2dos


!
! Computes DFT DOS as a simple sum over poles
!
subroutine comp_dos_via_poles(wf_k, d_omega, eps, dos_output)
  implicit none
  !! external
  type(dft_wf8_t), intent(in) :: wf_k
  real(8), intent(in)   :: d_omega, eps
  real(8), allocatable, intent(inout) :: dos_output(:)

  !! internal
  integer :: f, kp, i, spin
  real(8) :: omega, dos
  !! Dimensions
  integer :: neigen, nspin, nkpoints
  real(8) :: pi
  pi = 4D0 * atan(1D0)
  neigen   = size(wf_k%E,1)
  nspin    = size(wf_k%E,2)
  nkpoints = size(wf_k%E,3)
  !! END of Dimensions
  
  do f=lbound(dos_output,1), ubound(dos_output,1)
    omega = f*d_omega

    dos = 0
    do kp=1,nkpoints
      do spin=1,nspin
        do i=1,neigen
          dos = dos + &
            aimag(1.0D0/(cmplx( omega- (wf_k%E(i,spin,kp)-wf_k%fermi_energy-&
              wf_k%eigenvalues_shift), eps,8)));
        end do
      end do
    end do
    dos_output(f) = -dos/pi
  end do ! f
  
  dos_output = dos_output / nkpoints
  
end subroutine !dft_wf_sum_poles2dos


end module !m_dft_wf8
