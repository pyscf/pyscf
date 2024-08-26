subroutine mnsol_interface(c,atnum,nat,sola,solb,solc,solg,solh,soln,icds,gcds,areacds,dcds)
      IMPLICIT none
      integer(kind=4), intent(in) :: nat,icds
      real(kind=8), intent(in) :: sola,solb,solc,solg,solh,soln
      real(kind=8), intent(in) :: c(3,nat)
      real(kind=8), intent(out):: dcds(3,nat)
      integer(kind=4), intent(in) :: atnum(nat)
      real(kind=8), intent(out) :: gcds, areacds

      real(kind=8), allocatable :: x(:)

      integer :: ixmem
      call mnsol_xmem(nat, ixmem)
      allocate(x(ixmem))
      call cdsset(icds,gcds,areacds,nat,c,atnum,dcds,x,sola,solb,solc,solg,solh,soln)
      deallocate(x)

end subroutine mnsol_interface
