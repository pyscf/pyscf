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

module m_hsx
#include "m_define_macro.F90"  
  use m_die, only : die
  implicit none
  private die

  !!
  !! Holds info about hamiltonian and overlap in sparse form
  !! 
  type hsx_t
    !! First index belongs to unit cell, second to super cell
    integer              :: norbs = -999       !! Number of orbitals in (unit cell)
    integer              :: norbs_sc = -999    !! Number of orbitals in super cell
    integer              :: nspin  = -999      !! Number of spins in calculation
    integer              :: nnz = -999         !! Number of nonzero matrix elements in H and S
    logical              :: is_gamma = .false. !! whether it is gamma calculation
 
    integer, allocatable :: orb_sc2orb_uc(:)   !! Index "super cell orbital --> unit cell orbital" indxuo in SIESTA

    ! CRS (Compressed Row Storage) control arrays. Data arrays below
    integer, allocatable :: row_ptr(:)   !! (norbs+1)
    integer, allocatable :: col_ind(:)   !! (nnz)
    ! END of CRS (Compressed Row Storage) control arrays. Data arrays below

    real(4), allocatable :: H4(:,:)      !! (nnz,nspin)
    real(4), allocatable :: S4(:)        !! (nnz)
    real(4), allocatable :: X4(:,:)      !! (xyz,nnz) Spatial vectors between orbital centers aB2RaB4() 

    real(8), allocatable :: H8(:,:)      !! (nnz,nspin)
    real(8), allocatable :: S8(:)        !! (nnz)
    real(8), allocatable :: X8(:,:)      !! (xyz,nnz) Spatial vectors between orbital centers

    real(8) :: Ne = -999  !! Number of electrons: Total electronic charge
    real(8) :: Te = -999  !! Temperature of electrons
  end type !hsx_t

  contains  

!
!
!
subroutine dealloc(hsx)
  implicit none
  type(hsx_t), intent(inout) :: hsx
  
  _dealloc(hsx%orb_sc2orb_uc)
  _dealloc(hsx%row_ptr)
  _dealloc(hsx%col_ind)
  _dealloc(hsx%H4)
  _dealloc(hsx%S4)
  _dealloc(hsx%X4)
  _dealloc(hsx%H8)
  _dealloc(hsx%S8)
  _dealloc(hsx%X8)
  hsx%norbs = -999
  hsx%norbs_sc = -999
  hsx%nspin = -999
  hsx%nnz   = -999
  hsx%Ne    = -999
  hsx%Te    = -999
  
end subroutine !dealloc 
  

!!
!!
!!
subroutine read_siesta_hsx(fname, force_basis_type, hsx)
  use m_precision, only : siesta_int
  use m_io, only : get_free_handle
  implicit none
  character(len=*), intent(in) :: fname
  integer, intent(in)          :: force_basis_type
  type(hsx_t), intent(inout)   :: hsx

  !! internal
  logical(siesta_int) :: gamma_si
  integer(siesta_int) :: n, n_sc, nspin, nnz
  integer :: ifile, ios, i, spin, sum_row2nnz
  integer(siesta_int), allocatable :: row2nnz(:), ref(:), iaux(:)

  !! executable
  
  call dealloc(hsx)
  
  ifile = get_free_handle();
  open(ifile,file=trim(fname),form='unformatted',action='read',status='old',iostat=ios);
  if(ios/=0) _die('file '//trim(fname)//' ?')
  rewind(ifile, iostat=ios)
  if(ios/=0)_die('ios /= 0')
  read(ifile,iostat=ios) n, n_sc, nspin, nnz

  hsx%norbs = n
  hsx%norbs_sc = n_sc
  hsx%nspin = nspin
  hsx%nnz = nnz
  
  if(ios/=0) then
    write(6,*) "ios = ", ios; _die('ios /= 0')
  endif
  read(ifile,iostat=ios) gamma_si
  if(ios/=0)then
    write(6,*) "ios = ", ios; _die('ios /= 0')
    _die('ios /= 0')
  endif
  hsx%is_gamma = gamma_si

  select case (force_basis_type)
  case(1)
    hsx%is_gamma = .true.
  case(2)
    hsx%is_gamma = .false.
  case default
    continue
  end select
  
  !! Read index array super cell orbital --> unit cell orbital. Specific for periodic case...
  if(.not. hsx%is_gamma) then
    !if(iv>0)write(ilog,*) '.not. hsx%is_gamma', hsx%is_gamma, (.not. hsx%is_gamma)
    allocate(iaux(hsx%norbs_sc))
    allocate(hsx%orb_sc2orb_uc(hsx%norbs_sc))
    read(ifile,iostat=ios) (iaux(i), i=1,hsx%norbs_sc)
    if (ios/=0) _die("ios/=0")
    hsx%orb_sc2orb_uc = iaux

    !! Consistency check to catch a read error issue observed with ifort 12.0.0 compiler
    allocate(ref(hsx%norbs))
    do i=1,hsx%norbs; ref(i)=i; enddo
    if(any(hsx%orb_sc2orb_uc(1:hsx%norbs)/=ref)) then
      write(0,'(a20,a20,a20)') 'o ', ' hsx%sc_orb2uc_orb(o)', ' ref(o)'
      do i=1,hsx%norbs
        write(0,'(3i20)') i, hsx%orb_sc2orb_uc(i), ref(i)
      enddo  
      write(0,*) 'Current conventions for .HSX files are violated.'
      write(0,*) 'This may indicate a problem (i) with (intel) fortran compiler'
      write(0,*) 'http://software.intel.com/en-us/forums/topic/499093'
      write(0,*) '(ii) with incompatibility of compiler conventions on logical'
      write(0,*) 'variables representations. In the latter case try using '
      write(0,*) '  force_basis_type  1 -- for finite size systems (molecules) '
      write(0,*) '  force_basis_type  2 -- for extenden systems (solids, slabs etc.) '
      _die('any(hsx%orb_sc2orb_uc(1:norbs)/=ref)')
    endif
    !! END of Consistency check to catch a read error issue observed with ifort 12.0.0 compiler
  endif
  !! END of Read index array super cell orbital --> unit cell orbital. Specific for periodic case...

  _dealloc(row2nnz)
  allocate(row2nnz(hsx%norbs))
  read(ifile,iostat=ios) (row2nnz(i), i=1,hsx%norbs)
  if (ios/=0) _die("ios/=0")
  sum_row2nnz = sum(row2nnz)
  if (sum_row2nnz /= hsx%nnz) then
    write(0,*) __FILE__, __LINE__, sum_row2nnz, hsx%nnz, hsx%norbs, hsx%is_gamma
    write(0,*) row2nnz
    _die('sum_row2nnz /= hsx%nnz');
  endif

  !! Fill the displacements (according to row2nnzero) row_ptr
  allocate(hsx%row_ptr(hsx%norbs+1))
  hsx%row_ptr = -999
  hsx%row_ptr(1) = 1
  do i=1, hsx%norbs; hsx%row_ptr(i+1)=hsx%row_ptr(i)+row2nnz(i); enddo
  !! END of Fill the displacements (according to row2nnz) row_ptr

  !! Fill the columns for each row index
  allocate(hsx%col_ind(hsx%nnz))
  _dealloc(iaux)
  allocate(iaux(hsx%norbs_sc))
  
  do i=1, hsx%norbs
    if (hsx%row_ptr(i+1)-hsx%row_ptr(i)/=row2nnz(i)) then
      write(0,*) i, hsx%row_ptr(i+1), hsx%row_ptr(i) 
      _die('hsx%row_ptr(i+1)-hsx%row_ptr(i)/=f')
    endif

    read(ifile,iostat=ios)iaux(1:row2nnz(i)) ! read set of columns for given row
    if (ios/=0) then
      write(0,*) 'error: ios=', ios
      _die('ios/=0')
    endif
    hsx%col_ind(hsx%row_ptr(i):hsx%row_ptr(i+1)-1) = iaux(1:row2nnz(i))

  enddo
  !! END of Fill the columns for each row index

  allocate(hsx%H4(hsx%nnz,hsx%nspin)); ! Hamiltonian matrix in sparse form
  allocate(hsx%S4(hsx%nnz));           ! Overlap matrix in sparse form
  allocate(hsx%X4(3,hsx%nnz));         ! Distances between atoms from unit cell and super cell.
  !! END of Allocate H, S and X matrices

  !! Read the data to H_sparse array
  do spin=1,hsx%nspin
    do i=1,hsx%norbs
      read(ifile,iostat=ios)hsx%H4(hsx%row_ptr(i):hsx%row_ptr(i+1)-1, spin)
      if (ios/=0) _die('ios/=0')
    enddo
  enddo
  ! unfortunately we do not know m-numbers at this point and the conversion must happen after
  !! END of Read the data to H_sparse array

  !! Read the data to S_sparse array
  do i=1,hsx%norbs
    read(ifile,iostat=ios)hsx%S4(hsx%row_ptr(i):hsx%row_ptr(i+1)-1)
    if (ios/=0) _die('ios/=0')
  enddo
  !! END of Read the data to S_sparse array
  ! unfortunately we do not know m-numbers at this point and the conversion must happen after

  read(ifile,iostat=ios) hsx%Ne, hsx%Te  ! Total electronic charge and Temperature
  if (ios/=0) _die('ios/=0')

  !! Read the data to ab_2_RB_minus_Ra array
  do i=1,hsx%norbs
    read(ifile,iostat=ios)hsx%X4(1:3,hsx%row_ptr(i):hsx%row_ptr(i+1)-1)
    if (ios/=0) _die('ios/=0')
  enddo
  !! END of Read the data to ab_2_RB_minus_Ra array
  close(ifile)

  _dealloc(iaux)
  _dealloc(ref)
  _dealloc(row2nnz)
   
end subroutine !read_siesta_hsx

!
!
!
subroutine crs2dns4(m, crs, row_ptr, col_ind, dns, emptyvalue)
  !! external
  implicit none
  integer, intent(in)    :: m
  real(4), intent(in)    :: crs(:)
  integer, intent(in)    :: row_ptr(:), col_ind(:)
  real(4), intent(inout) :: dns(:,:)
  real(4), intent(in), optional :: emptyvalue

  !! internal
  integer :: i,si

  if(present(emptyvalue)) then; dns = emptyvalue; else; dns = 0; endif

  do i=1,m
    do si=row_ptr(i),row_ptr(i+1)-1
      
      dns(i,col_ind(si)) = crs(si)

    enddo ! sparse index
  enddo ! row
  
end subroutine !crs2dns4


end module !m_hsx
