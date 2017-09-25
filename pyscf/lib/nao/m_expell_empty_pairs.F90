module m_expell_empty_pairs
!
! Expells the empty pairs
!
#include "m_define_macro.F90"

  implicit none

contains

!
! Initialization of the aux variables
!
subroutine expell_empty_pairs(book_dp, p2n, coeffs)
  use m_book_pb, only : book_pb_t, print_info_book
  use m_coeffs_type, only : d_coeffs_ac_dp_t, dealloc
  use m_die, only : die
  
  implicit  none
  ! external
  type(book_pb_t), intent(inout), allocatable :: book_dp(:)
  integer, intent(in) :: p2n(:)
  type(d_coeffs_ac_dp_t), intent(inout), allocatable :: coeffs(:)
  !! internal
  type(book_pb_t), allocatable :: book_cont(:)
  type(d_coeffs_ac_dp_t), allocatable :: coeffs_cont(:)
  integer :: npairs_orig, npairs, p, top, step, pair, s,f
  
  _size_alloc(book_dp, npairs_orig)
  if(npairs_orig/=size(p2n)) then
    write(6,*) npairs_orig, size(p2n) 
    _die('!p2n?')
  endif

  ! init global counting in dominant products
  f = 0
  do p=1,npairs_orig
    top = book_dp(p)%top
    select case(top)
    case(-1);  cycle
    case(1:2); continue
    case default; write(6,*) top, p; _die('!type of pair?');
    end select
    
    s = f + 1
    if(p2n(p)<1) then
      write(6,*) p, p2n(p), top
      _die('!p2n')
    endif
    f = s + p2n(p) - 1
    book_dp(p)%si(3) = s
    book_dp(p)%fi(3) = f
  enddo ! ibp  
  ! END of init global counting in dominant products  

  !call print_info_book(6, book_dp)

  npairs = 0
  do step=1,2
    pair = 0
    do p=1,npairs_orig
      top = book_dp(p)%top
      select case(top)
      case(-1);  cycle
      case(1:2); continue
      case default
        write(6,*) top, p; _die('!type of pair?');
      end select
      pair = pair + 1
      if(step<2) cycle
      book_cont(pair) = book_dp(p)
      book_cont(pair)%ic = pair
      coeffs_cont(pair) = coeffs(p)
      call dealloc(coeffs(p))
    enddo ! p
    
    if(step>1) cycle
    npairs = pair
    if(npairs==npairs_orig) exit ! it looks as we don't have to restore data 
    allocate(book_cont(npairs))
    allocate(coeffs_cont(npairs))
  enddo ! step
  
  if(npairs/=npairs_orig) then
    _dealloc(book_dp)
    _dealloc(coeffs)
    allocate(book_dp(npairs))
    allocate(coeffs(npairs))
  
    book_dp = book_cont
    coeffs = coeffs_cont
  endif  
     
  !call print_info_book(6, book_dp)
!  write(6,*) __FILE__, __LINE__, npairs_orig, npairs
!  _die('removed ?')    

  _dealloc(book_cont)
  _dealloc(coeffs_cont)

end subroutine ! init_aux_pb_cp  


end module !m_expell_empty_pairs
