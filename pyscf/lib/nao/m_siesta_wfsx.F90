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

module m_siesta_wfsx
#include "m_define_macro.F90"  
  use m_die, only : die
  use m_warn, only : warn
  use m_precision, only : siesta_int
  use iso_c_binding, only: c_char, c_double, c_float, c_int64_t, c_int

  implicit none
  private die
  private warn

  !! structure holds eigenvalues and eigenvectors a other info from a .WFSX file
  type siesta_wfsx_t
    integer(siesta_int)              :: nkpoints=-999
    integer(siesta_int)              :: nspin = -999
    integer(siesta_int)              :: norbs = -999
    logical(siesta_int)              :: gamma = .false.
    integer(siesta_int), allocatable :: orb2atm(:)
    character(len=20), allocatable   :: orb2strspecie(:)
    integer(siesta_int), allocatable :: orb2ao(:)
    integer(siesta_int), allocatable :: orb2n(:)
    character(len=20), allocatable   :: orb2strsym(:) 
    real(8), allocatable             :: kpoints(:,:)  ! (xyz, kpoint)
    real(8), allocatable             :: DFT_E(:,:,:)  ! (orbital,spin,kpoint)
    real(4), allocatable             :: DFT_X(:,:,:,:,:)  ! (reim,orbital,eigenvalue,spin,kpoint)
    logical(siesta_int), allocatable :: mo_spin_kpoint_2_is_read(:,:,:) ! (eigenvalue,spin,kpoint)
  end type ! siesta_wfsx_t
  !! END of structure holds eigenvalues and eigenvectors a other info from a .WFSX file

  contains

!
!
!
subroutine siesta_wfsx_book_size(fname_in, nreim, isize, ios) bind(c, name='siesta_wfsx_book_size')
  use m_null2char, only : null2char
  implicit none
  !! external 
  character(c_char), intent(in) :: fname_in(*)
  integer(c_int64_t), intent(in)  :: nreim
  integer(c_int64_t), intent(inout)  :: isize
  integer(c_int64_t), intent(inout)  :: ios
  !! internal
  type(siesta_wfsx_t) :: wfsx
  character(1000) :: fname
  integer :: strsym_len, strspecie_len
  
  isize = 0
  call null2char(fname_in, fname)
  call siesta_get_wfsx(fname, nreim, wfsx, ios)
  if(ios/=0) return
  
  isize = isize + 1 !  integer(siesta_int)              :: nkpoints=-999
  isize = isize + 1 !  integer(siesta_int)              :: nspin = -999
  isize = isize + 1 !  integer(siesta_int)              :: norbs = -999
  isize = isize + 1 !  logical(siesta_int)              :: gamma = .false.
  isize = isize + wfsx%norbs    !  integer(siesta_int), allocatable :: orb2atm(:)
  isize = isize + wfsx%norbs    !  integer(siesta_int), allocatable :: orb2ao(:)
  isize = isize + wfsx%norbs    !  integer(siesta_int), allocatable :: orb2n(:)
  isize = isize + 1             !  strspecie_len
  strspecie_len = len(wfsx%orb2strspecie(1))
  isize = isize + wfsx%norbs*strspecie_len !  character(len=20), allocatable   :: orb2strspecie(:)
  isize = isize + 1             !  strsym_len
  strsym_len = len(wfsx%orb2strsym(1))
  isize = isize + wfsx%norbs*strsym_len !  character(len=20), allocatable   :: orb2strsym(:) 

end subroutine ! siesta_wfsx_book_size

!
!
!
subroutine siesta_wfsx_book_read(fname_in, nreim, dat, ios) bind(c)
  use m_null2char, only : null2char
  implicit none
  !! external
  character(c_char), intent(in) :: fname_in(*)
  integer(c_int64_t), intent(inout) :: nreim
  integer(c_int64_t), intent(inout) :: dat(*)
  integer(c_int64_t), intent(inout) :: ios
  !! internal
  type(siesta_wfsx_t) :: wfsx
  character(1000) :: fname
  integer :: i, n, j, splen, symlen, k
  
  call null2char(fname_in, fname)

  call siesta_get_wfsx(fname, nreim, wfsx, ios)
  if(ios/=0) return
  
  i = 1
  n = wfsx%norbs
  if(n<1) _die('!n<1')
  dat(i) = wfsx%nkpoints; i=i+1;!  integer(siesta_int)              :: nkpoints=-999
  dat(i) = wfsx%nspin; i=i+1;   !  integer(siesta_int)              :: nspin = -999
  dat(i) = wfsx%norbs; i=i+1;   !  integer(siesta_int)              :: norbs = -999
  dat(i) = l2i(logical(wfsx%gamma)); i=i+1; !  logical(siesta_int)              :: gamma = .false.
  
  dat(i:i+n-1) = wfsx%orb2atm(1:n); i=i+n;   !  integer(siesta_int), allocatable :: orb2atm(:)
  dat(i:i+n-1) = wfsx%orb2ao(1:n); i=i+n;   !  integer(siesta_int), allocatable :: orb2atm(:)
  dat(i:i+n-1) = wfsx%orb2n(1:n); i=i+n;   !  integer(siesta_int), allocatable :: orb2atm(:)

  splen = len(wfsx%orb2strspecie(1))
  dat(i) = splen; i=i+1;   !   strspecie_len
  do j=1,n
    do k=1,splen
      dat(i) = ichar(wfsx%orb2strspecie(j)(k:k)); i=i+1;
    enddo
  enddo

  symlen = len(wfsx%orb2strsym(1))
  dat(i) = symlen; i=i+1;   !   strsym_len

  do j=1,n
    do k=1,splen
      dat(i) = ichar(wfsx%orb2strsym(j)(k:k)); i=i+1;
    enddo
  enddo

end subroutine ! siesta_wfsx_book_read

!
!
!
subroutine siesta_wfsx_dread(fname_in, nreim, dat, ios) bind(c)
  use m_null2char, only : null2char
  implicit none
  !! external
  character(c_char), intent(in) :: fname_in(*)
  integer(c_int64_t), intent(in) :: nreim 
  real(c_double), intent(inout) :: dat(*)
  integer(c_int64_t), intent(inout) :: ios 
  !! internal
  type(siesta_wfsx_t) :: wfsx
  character(1000) :: fname
  integer :: i, k, j, s
  
  call null2char(fname_in, fname)

  call siesta_get_wfsx(fname, nreim, wfsx, ios)
  if(ios/=0) return
  i = 1
  do k=1,wfsx%nkpoints
    do s=1,wfsx%nspin
      do j=1,wfsx%norbs
        dat(i) = wfsx%DFT_E(j,s,k); i=i+1;
      enddo
    enddo
  enddo

  do k=1,wfsx%nkpoints
    do j=1,3
      dat(i) = wfsx%kpoints(j,k); i=i+1;
    enddo
  enddo

end subroutine ! siesta_wfsx_dread

!
!
!
subroutine siesta_wfsx_sread(fname_in, nreim, dat, ios) bind(c, name='siesta_wfsx_sread')
  use m_null2char, only : null2char
  implicit none
  !! external
  character(c_char), intent(in) :: fname_in(*)
  real(c_float), intent(inout)  :: dat(*)
  integer(c_int64_t), intent(in) :: nreim
  integer(c_int64_t), intent(inout) :: ios
  !! internal
  type(siesta_wfsx_t) :: wfsx
  character(1000) :: fname
  integer(c_int64_t) :: l, r, i, k, j, s
  
  call null2char(fname_in, fname)

  call siesta_get_wfsx(fname, nreim, wfsx, ios)
  if(ios/=0) return
  
  i = 1

  do k=1,wfsx%nkpoints
    do s=1,wfsx%nspin
      do j=1,wfsx%norbs
        do l=1,wfsx%norbs
          do r=1,nreim
            dat(i) = wfsx%DFT_X(r,l,j,s,k); i=i+1;
            !write(6,*) r,l,j,s,k, wfsx%DFT_X(r,l,j,s,k)
          enddo
        enddo 
      enddo
    enddo
  enddo

end subroutine ! siesta_wfsx_sread

!
!
!
integer(c_int) function l2i(l)
  implicit none
  logical, intent(in) :: l
  if(l) then; l2i=1; else; l2i=0; endif
end function !l2i


!
! Open a .WFSX file and reads the eigenvectors and eigenenergies from this file
! structure siesta_wfsx_t will be filled
!
subroutine siesta_get_wfsx(fname, nreim, wfsx, ios)
  use m_io, only : get_free_handle
  implicit none
  character(len=*), intent(in) :: fname
  integer(c_int64_t), intent(in) :: nreim
  type(siesta_wfsx_t), intent(inout)  :: wfsx
  integer(c_int64_t), intent(inout) :: ios

  ! internal
  integer :: ifile, i, natoms, norb_max
  real(8) :: real_E_eV   !! reading eigenvectors
  integer(siesta_int) :: ispin_in, ikpoint_in, imolecular_orb_in
  integer(siesta_int) :: norbitals_in !! number of saved molecular (!) orbitals
  integer :: ikpoint, ispin, imolecular_orb
  
  ifile = get_free_handle()
  open(ifile, file=fname, form='unformatted', action='read', status='old',iostat=ios);
  if(ios/=0) return
  
  rewind(ifile)
  read(ifile) wfsx%nkpoints, wfsx%gamma
  read(ifile) wfsx%nspin
  read(ifile) wfsx%norbs

  allocate(wfsx%orb2atm(       wfsx%norbs))
  allocate(wfsx%orb2strspecie( wfsx%norbs))
  allocate(wfsx%orb2ao(        wfsx%norbs))
  allocate(wfsx%orb2n(         wfsx%norbs))
  allocate(wfsx%orb2strsym(    wfsx%norbs))
  read(ifile)(wfsx%orb2atm(i),wfsx%orb2strspecie(i),wfsx%orb2ao(i),wfsx%orb2n(i),wfsx%orb2strsym(i),i=1,wfsx%norbs);
  natoms = maxval(wfsx%orb2atm);
  norb_max = maxval(wfsx%orb2ao);

  !! Debug output
  !if(iv>1) then
  !  write(ilog,'(a)') ' orbital,orb2atm,trim(orb2strspecie),orb2ao,  orb2n,trim(orb2strsym)'
  !  do i=1, wfsx%norbs
  !    write(ilog,*) i, wfsx%orb2atm(i), trim(wfsx%orb2strspecie(i)),wfsx%orb2ao(i),wfsx%orb2n(i),trim(wfsx%orb2strsym(i))
  !  end do
  !  write(ilog,*) 'wfsx%norbs, wfsx%nspin, natoms'
  !  write(ilog,*)  wfsx%norbs, wfsx%nspin, natoms
  !endif
  !! END of Debug output

  !! Allocate wfsx%kpoints, wfsx%DFT_E, wfsx%DFT_X
  allocate(wfsx%kpoints(3,wfsx%nkpoints))
  allocate(wfsx%DFT_E(wfsx%norbs, wfsx%nspin, wfsx%nkpoints))
  if(nreim<0 .or. nreim>2) then
    if(wfsx%gamma) then ! gamma calculation k={0,0,0} --> eigenvectors are real
      allocate(wfsx%DFT_X(1,wfsx%norbs,wfsx%norbs, wfsx%nspin, wfsx%nkpoints))
    else
      allocate(wfsx%DFT_X(2,wfsx%norbs,wfsx%norbs, wfsx%nspin, wfsx%nkpoints))
    endif
  else
    allocate(wfsx%DFT_X(nreim,wfsx%norbs,wfsx%norbs, wfsx%nspin, wfsx%nkpoints))
    if(nreim==1) wfsx%gamma = .true.
    if(nreim==2) wfsx%gamma = .false.
  endif
    
  wfsx%kpoints = -999
  wfsx%DFT_X   = -999
  wfsx%DFT_E   = -999
  !! END of Allocate wfsx%kpoints, wfsx%DFT_E, wfsx%DFT_X

  !! Additional check for everything read
  allocate(wfsx%mo_spin_kpoint_2_is_read(wfsx%norbs,wfsx%nspin,wfsx%nkpoints))
  wfsx%mo_spin_kpoint_2_is_read = .false.
  !! END of Additional check for everything read

  !! Loop over k-points to get the data
  do ikpoint = 1, wfsx%nkpoints
    do ispin = 1, wfsx%nspin
      read(ifile)ikpoint_in,wfsx%kpoints(1:3,ikpoint)
      if (ikpoint/=ikpoint_in) stop 'siesta_get_wfsx: ikpoint .ne. ikpoint_in'

      read(ifile) ispin_in
      if(ispin_in>wfsx%nspin) then
        write(0,*)'siesta_get_wfsx: err: ispin_in>wfsx%nspin', ispin_in, wfsx%nspin
        write(0,*)'siesta_get_wfsx: ikpoint, ispin, ispin_in, norbitals_in',  ikpoint, ispin, ispin_in
        stop 'siesta_get_wfsx'
      endif

      read(ifile) norbitals_in
      if(norbitals_in>wfsx%norbs) then
        write(0,*)'siesta_get_wfsx: err: norbitals_in>wfsx%norbs', norbitals_in, wfsx%norbs
        write(0,*)'siesta_get_wfsx: ikpoint, ispin, ispin_in, norbitals_in',  ikpoint, ispin, ispin_in, norbitals_in
        stop 'siesta_get_wfsx'
      endif

      do imolecular_orb=1,norbitals_in

        read(ifile) imolecular_orb_in
        if(imolecular_orb_in>wfsx%norbs) then
          write(0,*)'siesta_get_wfsx: err: imolecular_orb_in>wfsx%norbs', imolecular_orb_in, wfsx%norbs
          write(0,*)'siesta_get_wfsx: ikpoint, ispin, ispin_in, norbitals_in',  ikpoint, ispin, ispin_in, norbitals_in
          stop 'siesta_get_wfsx'
        endif

        read(ifile) real_E_eV
        read(ifile) wfsx%DFT_X(:,:,imolecular_orb_in, ispin_in, ikpoint)
        wfsx%DFT_E(imolecular_orb_in, ispin_in, ikpoint) = real_E_eV / (13.60580d0*2);
        wfsx%mo_spin_kpoint_2_is_read(imolecular_orb_in,ispin_in,ikpoint) = .true.
      enddo;
    enddo;
  enddo;
  !! END of Loop over k-points to get the data
  close(ifile);

  if(.not. all(wfsx%mo_spin_kpoint_2_is_read)) then
    write(6,*) 'siesta_get_wfsx: warn: .not. all(wfsx%mo_spin_k_2_is_read) '
    write(6,*) wfsx%mo_spin_kpoint_2_is_read
  endif
  

end subroutine ! siesta_get_wfsx

!
! Computes number of species (different atoms in molecule) & initializes a sp2label array
!
subroutine siesta_wfsx_to_sp2label_atm2sp(wfsx, sp2label, atm2sp)
  implicit none
  type(siesta_wfsx_t), intent(in) :: wfsx
  character(20), allocatable, intent(inout) :: sp2label(:)  !! character(20) must be 
  integer, allocatable, intent(inout)      :: atm2sp(:)

  integer :: nspecies, iunique_sp, iorb, atm, isp, i, natoms, step
  character(20), allocatable :: orb2strspecie_unique(:)
  logical :: one_more_specie

  !! Compute number of species (number of unique entities in orb2strspecie)
  if( .not. allocated(wfsx%orb2strspecie)) stop '.not. allocated(wfsx%orb2strspecie)'
  if (.not. allocated(wfsx%orb2atm)) stop '.not. allocated(wfsx%orb2atm)'

  if(allocated(sp2label)) deallocate(sp2label)

  allocate(orb2strspecie_unique(wfsx%norbs));

  do step=1,2
    orb2strspecie_unique = '';
    nspecies = 0
    do iorb=1, wfsx%norbs
      one_more_specie = .true.
      do iunique_sp=1,nspecies
        if(orb2strspecie_unique(iunique_sp)/=wfsx%orb2strspecie(iorb)) cycle
        one_more_specie = .false.
        exit
      end do
      if(.not. one_more_specie) cycle
      nspecies = nspecies + 1
      orb2strspecie_unique(nspecies) = wfsx%orb2strspecie(iorb);
      if(step==2) sp2label(nspecies) = wfsx%orb2strspecie(iorb);
    end do ! iorb
    if(step==1) allocate(sp2label(nspecies))
  end do
  !! END of Compute number of species (number of unique entities in orb2strspecie)

  if(allocated(atm2sp)) deallocate(atm2sp)
  natoms = maxval(wfsx%orb2atm)
  allocate(atm2sp(natoms))

  !! Fill atm2sp  (atom to specie)
  do iorb=1, wfsx%norbs
    atm = wfsx%orb2atm(iorb)
    isp = 0
    do i=1, nspecies
      if(sp2label(i)/=wfsx%orb2strspecie(iorb)) cycle
      isp = i; exit;
    enddo
    if(isp==0) then; write(0,*) 'wfsx_to_sp2label_atm2sp: isp==0', isp; stop; endif;
    atm2sp(atm) = isp;
  enddo

end subroutine !wfsx2sp2label

end module !m_siesta_wfsx
