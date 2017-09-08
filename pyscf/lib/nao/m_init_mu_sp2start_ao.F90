module m_init_mu_sp2start_ao


#include "m_define_macro.F90"
  use m_die, only : die
  use m_warn, only : warn
  
  implicit none
    
  contains

subroutine init_mu_sp2start_ao(sp2nmult, mu_sp2j, mu_sp2start_ao)
  implicit none
  ! external
  integer, intent(in) :: sp2nmult(:), mu_sp2j(:,:)
  integer, intent(inout), allocatable :: mu_sp2start_ao(:,:)
  !internal
  integer :: nmult_mx, nsp, s,f,j,sp,mu
  
  nmult_mx = size(mu_sp2j,1)
  nsp = size(mu_sp2j,2)
  if(nsp<1) _die('!nsp')
  if(nmult_mx<1) _die('!nmult_mx')
  if(nsp/=size(sp2nmult)) _die('!sp2nmult')
  
  _dealloc(mu_sp2start_ao)
  allocate(mu_sp2start_ao(nmult_mx, nsp))
  mu_sp2start_ao =-999
  
  !! Updating %mu_sp2start_ao
  do sp=1,nsp
    f = 0
    do mu=1,sp2nmult(sp)
      j = mu_sp2j(mu,sp)
      s = f + 1
      f = s + 2*j
      mu_sp2start_ao(mu,sp) = s
    enddo ! mu
  enddo ! sp
  !! END of Updating %mu_sp2start_ao

end subroutine ! init_mu_sp2start_ao

end module !m_init_mu_sp2start_ao


