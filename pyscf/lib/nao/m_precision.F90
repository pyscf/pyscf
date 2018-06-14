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

module m_precision

#include "m_define_macro.F90"

  use, intrinsic :: iso_c_binding

  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 60)

! BLAS/LAPACK library precision of integer variables
#ifdef BLAS_INT
  integer, parameter     :: blas_int   = BLAS_INT;
#else
  integer, parameter     :: blas_int   = 4; 
#endif

#ifdef C_INTEGER
  integer, parameter     :: c_integer   = C_INTEGER;
#else
  integer, parameter     :: c_integer   = C_INT; 
#endif


! FFTW library precision of real variables 
#ifdef FFTW_REAL
  integer, parameter     :: fftw_real  = FFTW_REAL;
#else
  integer, parameter     :: fftw_real  = 8; 
#endif

! MPI library precision of integer variables
#ifdef MPI_INT
  integer, parameter     :: mpi_int   = MPI_INT;
#else
  integer, parameter     :: mpi_int   = 4; 
#endif

! SIESTA package precision of integer variables
#ifdef SIESTA_INT
  integer, parameter     :: siesta_int   = SIESTA_INT;
#else
  integer, parameter     :: siesta_int   = 4; 
#endif

! SIESTA package precision of integer variables
#ifdef LIBXC_INT
  integer, parameter     :: libxc_int   = LIBXC_INT;
#else
  integer, parameter     :: libxc_int   = 4; 
#endif
  
contains

!
!
!
subroutine print_precision(ifile)
  implicit none
  !! external
  integer, intent(in) :: ifile
  !! internal
  integer :: i=0
  write(ifile,'(a35,i5)') 'precision of variables (internal):'
  write(ifile,'(a35,2x,i5)') 'blas_int', blas_int
  write(ifile,'(a35,2x,i5)') 'fftw_real', fftw_real
  write(ifile,'(a35,2x,i5)') 'mpi_int', mpi_int
  write(ifile,'(a35,2x,i5)') 'siesta_int', siesta_int
  write(ifile,'(a35,2x,i5)') 'fortran_int', kind(i)
  
end subroutine ! print_precision

!
! Reports precision parameters to a file fname 
!
subroutine report_precision(fname, iv)
  use m_io, only : get_free_handle
  implicit none
  !! external  
  character(*), intent(in) :: fname
  integer, intent(in) :: iv
  !! internal
  integer :: ifile, ios
  
  ifile = get_free_handle()
  open(ifile, file=fname, action='write', iostat=ios)
  if(ios/=0) then
    write(6,*)'io error at ',__FILE__,__LINE__
    stop 'io error '
  endif  
  call print_precision(ifile)
  close(ifile)
  if(iv>0)write(6,*)'written: ', trim(fname)
end subroutine ! report_precision

end module !m_precision
