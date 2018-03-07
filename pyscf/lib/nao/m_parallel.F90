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

module m_parallel

#ifdef _OPENMP
  use omp_lib, only : omp_get_num_threads
#endif 

  implicit none

#ifdef MPI
  include 'mpif.h'
#endif

  public init_parallel
  public end_parallel

#ifdef MPI
  public MPI_COMM_WORLD
  public MPI_INTEGER
  public MPI_REAL4
  public MPI_REAL8
  public MPI_MAX
  public MPI_SUM
  public MPI_Complex16 ! (complex(8))
  public MPI_IN_PLACE
  public MPI_LOGICAL
  
#endif

  type para_t
    integer       :: rank=0
    integer       :: nodes=1
    logical       :: master = .true.
    integer       :: num_threads = 1
    character(20) :: paratype=""
    integer       :: mpi_thread_support=-999
  end type ! para_t
  
  public para_t
  
  contains

!
! Tells the number of nodes in the MPI world
!
integer function get_nnodes(para)
  implicit none
  type(para_t), intent(in) :: para

  if(para%nodes<1) then
    write(6,*) __FILE__, __LINE__, para%nodes
    stop 'para%nodes<1'
  endif

  get_nnodes = para%nodes

end function !get_nnodes


!
!
!
subroutine init_parallel(para, iv)
  use m_precision, only : report_precision
  !use m_input, only : init_parameter, input_t
  !use m_arch, only : report_arch
  implicit none
  !! external
  !type(input_t), intent(in) :: inp
  type(para_t), intent(inout) :: para
  integer, intent(in) :: iv

#ifdef MPI  
  integer :: ierr, s
#endif
  integer :: ilog
  para%paratype =""
  ilog = 6
    
#ifdef MPI
    call MPI_Init_thread(MPI_THREAD_FUNNELED, para%mpi_thread_support, ierr)
    if(ierr/=0) then 
      write(6,*) __FILE__, __LINE__
      stop '0'
    endif
    if(iv>0)then
      s = get_mpi_thread_support(para)
      write(ilog,*) "get_mpi_thread_support(para) ", s
      if(s==MPI_THREAD_SINGLE) then
        write(ilog,*) 's==MPI_THREAD_SINGLE'
      else if (s==MPI_THREAD_FUNNELED) then
        write(ilog,*) 's==MPI_THREAD_FUNNELED'
      else if (s==MPI_THREAD_SERIALIZED) then
        write(ilog,*) 's==MPI_THREAD_SERIALIZED'
      else if (s==MPI_THREAD_MULTIPLE) then
        write(ilog,*) 's==MPI_THREAD_MULTIPLE'
      else 
        write(ilog,*) 'not recognized mpi_thread_support'
      endif
    endif    
    
    call MPI_Comm_size( MPI_Comm_World, para%nodes, ierr)
    if(iv>0)write(ilog,*) "MPI_Comm_Size mpi_ierr nodes ", ierr, para%nodes
    if(ierr/=0) stop '1'
    call MPI_Comm_Rank( MPI_Comm_World, para%rank, ierr)
    if(iv>1)write(ilog,*) "MPI_Comm_Rank mpi_ierr mynode   ", ierr, para%rank
    if(ierr/=0) stop '2'
    para%master = (para%rank==0)
    para%paratype="MPI"
#else
    para%nodes = 1
    para%rank   = 0
    para%master   = .true.
    if(iv>1)write(ilog,*) "init_parallel: ", para%rank, para%nodes
#endif


#ifdef _OPENMP
  !$omp parallel 
  !$omp master
  !call init_parameter('num_threads', inp, omp_get_num_threads(), para%num_threads, iv)
  para%num_threads = omp_get_num_threads()
  !$omp end master
  !$omp end parallel 
  if(len_trim(para%paratype)>0)para%paratype = trim(para%paratype) // "-"
  para%paratype = trim(para%paratype) // "OPENMP"
#else
  para%num_threads = 1
  if(len_trim(para%paratype)>0)para%paratype = trim(para%paratype) // "-" 
  para%paratype = trim(para%paratype) // "SEQ"
#endif

  if (para%master .and. iv>0) then
    write(ilog,'(a,a)')  "Type of parallelisation:    ", para%paratype
    write(ilog,'(a,i4)') "Number of nodes:            ", para%nodes
    write(ilog,'(a,i4)') "Number of threads per node: ", para%num_threads
  end if
#ifdef MPI
  call MPI_Barrier(MPI_Comm_World, ierr)
  if(ierr/=0) stop '3'
#endif
  !call report_arch('report_arch.txt', iv)
  call report_precision('report_precision.txt', iv)

end subroutine !init_parallel

!
!
!
integer function get_mpi_thread_support(para)
  implicit none
  type(para_t), intent(in) :: para
  if(para%mpi_thread_support==-999) then
    write(6,*) __FILE__, __LINE__
    stop 'probable mpi_thread_support was not initialized'
  endif
  get_mpi_thread_support = para%mpi_thread_support
end function !  get_mpi_thread_support

!
!
!
subroutine  end_parallel()
#ifdef  MPI
  integer :: ierr
  call mpi_finalize(ierr)
  if(ierr/=0) stop 'ierr/=0'
#endif
end subroutine !end_parallel

end module !m_parallel


