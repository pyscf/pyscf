module m_sparse_vec

  implicit none

  interface get_nnz
    module procedure get_nnz_real4
    module procedure get_nnz_real8
  end interface !get_nnz


  contains

!
!
!
integer function get_nnz_real4(inz2vo)
  implicit none
  ! external
  real(4), intent(in), allocatable :: inz2vo(:,:)
  ! internal
  integer :: nn(2)
  if(.not. allocated(inz2vo)) then
    write(6,*) __FILE__, __LINE__
    stop '!inz2vo... OMP ?'
  endif  
    
  nn = lbound(inz2vo)
  if(any(nn/=[0,1])) then
    write(6,*) __FILE__, __LINE__
    stop 'any(nn/=[0,1])'
  endif  
  if(ubound(inz2vo,2)/=2) then
    write(6,*) __FILE__, __LINE__
    stop 'ubound(inz2vo,2)/=2'
  endif
    
  nn = int(inz2vo(0,1:2))
  if(nn(1)/=nn(2)) then
    write(6,*) __FILE__, __LINE__
    stop 'nn(1)/=nn(2)'
  endif  
  get_nnz_real4 = nn(1)
  if(get_nnz_real4<0) then
    write(6,*) __FILE__, __LINE__
    stop 'get_nnz<0 ? can this be true?'
  endif  
  
end function ! get_nnz  



!
!
!
integer function get_nnz_real8(inz2vo)
  implicit none
  ! external
  real(8), intent(in), allocatable :: inz2vo(:,:)
  ! internal
  integer :: nn(2)
  if(.not. allocated(inz2vo)) then
    write(6,*) __FILE__, __LINE__
    stop '!inz2vo... OMP ?'
  endif  
    
  nn = lbound(inz2vo)
  if(any(nn/=[0,1])) then
    write(6,*) __FILE__, __LINE__
    stop 'any(nn/=[0,1])'
  endif  
  if(ubound(inz2vo,2)/=2) then
    write(6,*) __FILE__, __LINE__
    stop 'ubound(inz2vo,2)/=2'
  endif
    
  nn = int(inz2vo(0,1:2))
  if(nn(1)/=nn(2)) then
    write(6,*) __FILE__, __LINE__
    stop 'nn(1)/=nn(2)'
  endif  
  get_nnz_real8 = nn(1)
  if(get_nnz_real8<0) then
    write(6,*) __FILE__, __LINE__
    stop 'get_nnz<0 ? can this be true?'
  endif  
  
end function ! get_nnz  


end module !m_sparse_vec
