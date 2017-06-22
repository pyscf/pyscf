module m_siesta_hsx
#include "m_define_macro.F90"  
  use m_die, only : die
  use m_precision, only : siesta_int
  implicit none
  private die

  !!
  !! type to hold information from .HSX file
  type siesta_hsx_t
    integer(siesta_int)              :: norbitals = -999     !! Number of orbitals in (unit cell)
    integer(siesta_int)              :: norbitals_sc = -999  !! Number of orbitals in super cell
    integer(siesta_int)              :: nspin  = -999        !! Number of spins in calculation
    integer(siesta_int)              :: nnonzero = -999      !! Number of nonzero matrix elements in H and S
    logical(siesta_int)              :: is_gamma = .false.   !! whether it is gamma calculation
    integer(siesta_int), allocatable :: sc_orb2uc_orb(:)     !! Index "super cell orbital --> unit cell orbital" indxuo in SIESTA
    integer(siesta_int), allocatable :: row2nnzero(:)        !! (norbitals)
    integer(siesta_int), allocatable :: sparse_ind2column(:) !! (nnonzero)
    real(4), allocatable    :: H_sparse(:,:)        !! (nnonzero,nspin)
    real(4), allocatable    :: S_sparse(:)          !! (nnonzero)
    real(4), allocatable    :: aB2RaB_sparse(:,:)   !! (3,nnonzero) Spatial vectors between orbital centers. 
                                                    !! First index belongs to unit cell, second to super cell 
    real(8) :: Total_electronic_charge = -999       !! Total electronic charge
    real(8) :: Temp                    = -999       !! Temperature
    !! ??? X
  end type !siesta_hsx_t

  interface realloc_if_necessary
    module procedure realloc_if_necessary_i8_d
    module procedure realloc_if_necessary_i4_d
    module procedure realloc_if_necessary_i4_i4
    module procedure realloc_if_necessary_i8_i8
  end interface  ! realloc_if_necessary

  contains  

!
!
!
subroutine dealloc(hsx)
  implicit none
  type(siesta_hsx_t), intent(inout) :: hsx
  
  _dealloc(hsx%sc_orb2uc_orb)
  _dealloc(hsx%row2nnzero)
  _dealloc(hsx%sparse_ind2column)
  _dealloc(hsx%H_sparse)
  _dealloc(hsx%S_sparse)
  _dealloc(hsx%aB2RaB_sparse)
  
end subroutine !dealloc 
  


!!
!!
!!
subroutine siesta_get_hsx(fname, force_basis_type, hsx, iv, ilog)
  use m_log, only : die, log_size_note
  use m_precision, only : siesta_int
  use m_io, only : get_free_handle
  implicit none
  character(len=*), intent(in)      :: fname
  integer, intent(in)               :: force_basis_type
  type(siesta_hsx_t), intent(inout) :: hsx
  integer, intent(in)               :: iv, ilog

  !! internal
  integer(siesta_int), allocatable  ::  int_buff(:)   !! buffer for pointers (to nonzero elements) within a column
  real(4),    allocatable  :: sp_buff(:)     !! buffer for vector values (of nonzero elements) within a column
  real(4),    allocatable  :: sp_buff3(:, :) !! buffer for vector values (of nonzero elements) for reading coord differences
  integer :: ifile, ios, irow, ispin, sum_row2nnzero, maxnnzero, d,f,i
  integer, allocatable :: row2displ(:), ref(:)
 
  if(iv>0) write(ilog,*)'siesta_get_hsx: enter'
  ifile = get_free_handle();
  open(ifile,file=trim(fname),form='unformatted',action='read',status='old',iostat=ios);
  if(ios/=0) _die('file '//trim(fname)//' ?')
  rewind(ifile, iostat=ios)
  if(ios/=0)_die('ios /= 0')
  read(ifile,iostat=ios) hsx%norbitals, hsx%norbitals_sc, hsx%nspin, hsx%nnonzero
  if(ios/=0)_die('ios /= 0')
  call log_size_note("hsx%norbitals", hsx%norbitals, iv); 
  call log_size_note("hsx%norbitals_sc", hsx%norbitals_sc, iv); 
  call log_size_note("hsx%nspin", hsx%nspin, iv); 
  call log_size_note("hsx%nnonzero", hsx%nnonzero, iv); 

  read(ifile,iostat=ios) hsx%is_gamma
  if(ios/=0)_die('ios /= 0')

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
    if(iv>0)write(ilog,*) '.not. hsx%is_gamma', hsx%is_gamma, (.not. hsx%is_gamma)
    allocate(hsx%sc_orb2uc_orb(hsx%norbitals_sc))
    read(ifile,iostat=ios) (hsx%sc_orb2uc_orb(i), i=1,hsx%norbitals_sc)
    if (ios/=0) _die("ios/=0")

    !! Consistency check to catch a read error issue observed with ifort 12.0.0 compiler
    allocate(ref(hsx%norbitals))
    do i=1,hsx%norbitals; ref(i)=i; enddo
    if(any(hsx%sc_orb2uc_orb(1:hsx%norbitals)/=ref)) then
      write(0,'(a20,a20,a20)') 'o ', ' hsx%sc_orb2uc_orb(o)', ' ref(o)'
      do i=1,hsx%norbitals
        write(0,'(3i20)') i, hsx%sc_orb2uc_orb(i), ref(i)
      enddo  
      write(0,*) 'Current conventions for .HSX files are violated.'
      write(0,*) 'This may indicate a problem (i) with (intel) fortran compiler'
      write(0,*) 'http://software.intel.com/en-us/forums/topic/499093'
      write(0,*) '(ii) with incompatibility of compiler conventions on logical'
      write(0,*) 'variables representations. In the latter case try using '
      write(0,*) '  force_basis_type  1 -- for finite size systems (molecules) '
      write(0,*) '  force_basis_type  2 -- for extenden systems (solids, slabs etc.) '
      _die('any(hsx%sc_orb2uc_orb(1:norbs)/=ref)')
    endif
    !! END of Consistency check to catch a read error issue observed with ifort 12.0.0 compiler
  endif
  !! END of Read index array super cell orbital --> unit cell orbital. Specific for periodic case...

  !! allocate the buffers
  if(allocated(hsx%row2nnzero)) deallocate(hsx%row2nnzero)
  allocate(hsx%row2nnzero(hsx%norbitals));
  if(allocated(hsx%sparse_ind2column)) deallocate(hsx%sparse_ind2column)
  allocate(hsx%sparse_ind2column(hsx%nnonzero));

  read(ifile,iostat=ios) (hsx%row2nnzero(i), i=1,hsx%norbitals);
  if (ios/=0) stop "siesta_get_hsx: (ios/=0) hsx%row2nnzero"
  sum_row2nnzero = sum(hsx%row2nnzero)

  if (sum_row2nnzero /= hsx%nnonzero) then
    write(0,*) 'siesta_get_hsx: sum_col2nnzero /= nnonzero ', &
      sum_row2nnzero, hsx%nnonzero, hsx%norbitals, hsx%is_gamma;
    write(0,*) hsx%row2nnzero;
    _die('sum_row2nnzero /= hsx%nnonzero');
  endif

  !! Fill the displacements (according to row2nnzero) row2displ
  allocate(row2displ(hsx%norbitals))
  row2displ(1)=0
  do irow=2, hsx%norbitals
    row2displ(irow) = row2displ(irow-1) + hsx%row2nnzero(irow-1)
  enddo
  !! END of Fill the displacements (according to row2nnzero) row2displ

  maxnnzero = maxval(hsx%row2nnzero)
  allocate(int_buff(maxnnzero));

  !! Fill the rows for each index in *_sparse arrays
  do irow=1, hsx%norbitals
    f = hsx%row2nnzero(irow)
    read(ifile,iostat=ios)int_buff(1:f) ! read set of rows where nonzero elements reside
    if (ios/=0) then
      write(0,*) 'error: ios=', ios
      stop "siesta_get_hsx: (ios/=0) int_buff 1"
    endif
    d = row2displ(irow)
    hsx%sparse_ind2column(d+1:d+f) = int_buff(1:f);
  enddo
  !! END of Fill the rows for each index in *_sparse arrays

  !! Allocate H, S and X matrices
  allocate(sp_buff(maxnnzero));
  allocate(sp_buff3(3,maxnnzero));

  if(allocated(hsx%H_sparse)) deallocate(hsx%H_sparse)
  allocate(hsx%H_sparse(hsx%nnonzero,hsx%nspin)); ! Hamiltonian matrix in sparse form

  if(allocated(hsx%S_sparse)) deallocate(hsx%S_sparse)
  allocate(hsx%S_sparse(hsx%nnonzero));       ! Overlap matrix in sparse form

  if(allocated(hsx%aB2RaB_sparse)) deallocate(hsx%aB2RaB_sparse)
  allocate(hsx%aB2RaB_sparse(3,hsx%nnonzero)); ! Distances between atoms from unit cell and super cell.
  !! END of Allocate H, S and X matrices

  !! Read the data to H_sparse array
  do ispin=1,hsx%nspin
    do irow=1,hsx%norbitals
      d = row2displ(irow)
      f = hsx%row2nnzero(irow)
      read(ifile,iostat=ios)sp_buff(1:f)
      if (ios/=0) stop "siesta_get_hsx: (ios/=0) Hamiltonian matrix"
      hsx%H_sparse(d+1:d+f, ispin) = sp_buff(1:f); 
    enddo
  enddo

  !! END of Read the data to H_sparse array
  if(iv>1) write(ilog,*) 'sum(abs(hsx%H_sparse))', sum(abs(hsx%H_sparse))

  !! Read the data to S_sparse array
  do irow=1,hsx%norbitals
    d = row2displ(irow)
    f = hsx%row2nnzero(irow)
    read(ifile,iostat=ios)sp_buff(1:f)
    if (ios/=0) then
      write(ilog,*)ios, d, f, irow
      stop "siesta_get_hsx: (ios/=0) overlap matrix"
    endif
    hsx%S_sparse(d+1:d+f) = sp_buff(1:f);
  enddo
  !! END of Read the data to S_sparse array

  read(ifile,iostat=ios) hsx%total_electronic_charge, hsx%temp  ! Total electronic charge and Temperature
  if (ios/=0) stop "siesta_get_hsx: (ios/=0) total_electronic_charge, temp"
  call log_size_note("siesta_get_hsx: hsx%total_electronic_charge", hsx%total_electronic_charge, iv);
  call log_size_note("siesta_get_hsx: hsx%temp, Ry", hsx%temp, iv);

  !! Read the data to ab_2_Ra_minus_RB array
  do irow=1,hsx%norbitals
    d = row2displ(irow)
    f = hsx%row2nnzero(irow)
    read(ifile,iostat=ios)sp_buff3(1:3,1:f)
    if (ios/=0) stop "siesta_get_hsx: (ios/=0) overlap matrix"
    hsx%aB2RaB_sparse(1:3,d+1:d+f) = sp_buff3(1:3,1:f);
  enddo
  !! END of Read the data to S_sparse array

  close(ifile);
  if(iv>1) write(ilog,*)'siesta_get_hsx: ', trim(fname);
  if(iv>1) write(ilog,*)'siesta_get_hsx: exit'

end subroutine !siesta_get_hsx

!
!
!
subroutine siesta_write_sparse(fname, numh, listhptr, listh, matrix_sparse, iv, ilog)
  use m_io, only : get_free_handle
  use m_precision, only : siesta_int
  !! external
  character(*), intent(in) :: fname
  integer, intent(in) :: numh(:), listh(:), listhptr(:)
  real(8), intent(in) :: matrix_sparse(:)
  integer, intent(in) :: iv, ilog

  !! internal
  integer :: ifile, ios

  ifile = get_free_handle()
  open(ifile, file=fname, action='write', form='unformatted', iostat=ios)
  if(ios/=0) then
    write(0,'(a,a,a,i6,a)') 'siesta_write_sparse: ', trim(fname), ' io error', ios, ' ==> return'
    return
  endif

  write(ifile) size(numh)
  write(ifile) numh
  write(ifile) size(listhptr)
  write(ifile) listhptr
  write(ifile) size(listh)
  write(ifile) listh
  write(ifile) size(matrix_sparse)
  write(ifile) matrix_sparse
  close(ifile)
  if(iv>0) write(ilog,*)'siesta_write_sparse: ', trim(fname);
  
end subroutine !! siesta_write_sparse

!
!
!
subroutine siesta_read_sparse(fname, numh_out, listhptr_out, listh_out, matrix_sparse_out, iv, ilog)
  use m_io, only : get_free_handle
  use m_precision, only : siesta_int
  !! external
  character(*), intent(in) :: fname
  integer, intent(inout), allocatable :: numh_out(:), listh_out(:), listhptr_out(:)
  real(8), intent(inout), allocatable :: matrix_sparse_out(:)
  integer, intent(in) :: iv, ilog
  
  !! internal
  integer :: ifile, ios
  integer(siesta_int) :: s
  integer(siesta_int), allocatable :: numh(:), listh(:), listhptr(:)
  real(8), allocatable :: matrix_sparse(:)
  
  ifile = get_free_handle()
  open(ifile, file=fname, action='read', form='unformatted', iostat=ios)
  if(ios/=0) then
    write(0,'(a,a,a,i6,a)') 'siesta_read_sparse: ', trim(fname), ' io error', ios, ' ==> return'
    return
  endif

  read(ifile) s; 
  allocate(numh(s))
  read(ifile) numh(1:s)
  call realloc_if_necessary(int(s), numh_out)
  numh_out = numh

  read(ifile) s; 
  allocate(listhptr(s))
  read(ifile) listhptr(1:s)
  call realloc_if_necessary(int(s), listhptr_out)
  listhptr_out = listhptr 

  read(ifile) s; 
  allocate(listh(s))
  read(ifile) listh(1:s)
  call realloc_if_necessary(int(s), listh_out)
  listh_out = listh 

  read(ifile) s; 
  allocate(matrix_sparse(s))
  read(ifile) matrix_sparse(1:s)
  call realloc_if_necessary(s, matrix_sparse_out)
  matrix_sparse_out = matrix_sparse

  close(ifile)
  if(iv>0) write(ilog,*)'siesta_read_sparse: ', trim(fname);
  
end subroutine !! siesta_read_sparse

!
!

subroutine sparse2full(ndim1, ndim2, M_full, ldm, &
M_sparse, row2nnzero, row2displ, sparse_ind2column, emptyvalue)
!! external
implicit none
integer, intent(in)    :: ndim1, ndim2, ldm
real(8), intent(inout) :: M_full(ldm,ndim2)
real(8), intent(in)    :: M_sparse(*)
integer, intent(in)    :: row2nnzero(*), row2displ(*), sparse_ind2column(*)
real(8), intent(in), optional :: emptyvalue

!! internal
integer :: icol, i, irow, sparse_ind

if(present(emptyvalue)) then
M_full = emptyvalue
else
M_full = 0
endif

do irow=1,ndim1
do i=1,row2nnzero(irow);
sparse_ind = row2displ(irow)+i;
icol = sparse_ind2column(sparse_ind);
M_full(irow,icol) = M_sparse(sparse_ind) !! a bit mess with columns and rows
enddo
enddo

end subroutine !sparse2full


!
!
!
subroutine realloc_if_necessary_i4_i4(minsize, array)
  implicit none
  integer(4), intent(in) :: minsize
  integer(4), allocatable, intent(inout) :: array(:)
  
  if( .not. allocated(array) ) allocate(array(minsize) )
  if( allocated(array) .and. minsize>size(array) ) then; deallocate(array); allocate(array(minsize) ); endif
end subroutine ! reallocate_if_necessary_integer

subroutine realloc_if_necessary_i8_i8(minsize, array)
  integer(8), intent(in) :: minsize
  integer(8), allocatable, intent(inout) :: array(:)
  
  if( .not. allocated(array) ) allocate(array(minsize) )
  if( allocated(array) .and. minsize>size(array) ) then; deallocate(array); allocate(array(minsize) ); endif
end subroutine ! reallocate_if_necessary_integer

!
!
!
subroutine realloc_if_necessary_i4_d(minsize, array)
  implicit none
  integer(4), intent(in) :: minsize
  real(8), allocatable, intent(inout) :: array(:)
  
  if( .not. allocated(array) ) allocate(array(minsize) )
  if( allocated(array) .and. minsize>size(array) ) then; deallocate(array); allocate(array(minsize) ); endif
end subroutine ! reallocate_if_necessary_integer

!
!
!
subroutine realloc_if_necessary_i8_d(minsize, array)
  implicit none
  integer(8), intent(in) :: minsize
  real(8), allocatable, intent(inout) :: array(:)
  
  if( .not. allocated(array) ) allocate(array(minsize) )
  if( allocated(array) .and. minsize>size(array) ) then; deallocate(array); allocate(array(minsize) ); endif
end subroutine ! reallocate_if_necessary_integer

end module !m_siesta_hsx
