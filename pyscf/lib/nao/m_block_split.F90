module m_block_split
!
! 
!

  implicit none

  contains

!
!  The subroutine  splits a range 1:size_total
!  into segments of size not more than block_size
!  Example can be fount in the module m_rf0_on4_ac_ia.
!
subroutine block_split(size_total, block_size, nblocks, block2start)
  implicit none
  !! external
  integer, intent(in) :: size_total !! dimension of matrix
  integer, intent(in) :: block_size
  integer, intent(inout) :: nblocks
  integer, intent(inout), allocatable :: block2start(:)
  !! internal
  integer :: block_size_mod, block_size_real, ib, s, f, block_size_local
  
  
  
  block_size_real = min(block_size,size_total)
  nblocks = int(size_total/block_size_real)
  block_size_mod  = modulo(size_total,block_size_real)
  if(block_size_mod/=0) nblocks = nblocks + 1
  
  if(allocated(block2start)) deallocate(block2start)
  allocate(block2start(nblocks+1))
  f = 0
  do ib=1,nblocks
    
    block_size_local = block_size_real
    if (ib==nblocks .and. (block_size_mod .gt. 0) ) then
      block_size_local = block_size_mod
    endif
    
    s = f + 1
    f = s + block_size_local - 1
    block2start(ib) = s
  enddo
  block2start(nblocks+1) = size_total+1

end subroutine ! block_split


end module !m_block_split


