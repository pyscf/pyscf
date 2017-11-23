module m_conv_lims

  use m_precision, only : fftw_real
 
  contains

!
! Computes 6 numbers out of limits 
!
subroutine conv_lims(s1,f1,s2,f2,s3,f3, n1,n2,ncaux,saux,salloc_aux,falloc_aux)
  implicit none
  ! external
  integer, intent(in) :: s1, f1, s2, f2, s3, f3
  integer, intent(out) :: n1,n2,ncaux,saux,salloc_aux,falloc_aux
  
  n1 = f1-s1+1
  n2 = f2-s2+1
  saux = s1+s2
  salloc_aux = min(saux,s3)
  falloc_aux = f3
  ncaux = min(falloc_aux-saux+1, f1+f2-saux+1)
  
end subroutine ! conv_lims

end module ! m_conv_lims
