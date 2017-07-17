module m_rsphar

  implicit none

#include "m_define_macro.F90"
  
  integer :: lmx = -1
  real(8), allocatable  :: lm2aa(:), lm2bb(:), l2tlp1(:), l2tlm1(:), l21mhl(:)

  contains

!
!
!
subroutine init_rsphar(lmx, lm2aa, lm2bb, l2tlp1, l2tlm1, l21mhl)
  ! external
  integer, intent(in) :: lmx
  real(8), intent(inout), allocatable :: lm2aa(:), lm2bb(:), l2tlp1(:), l2tlm1(:), l21mhl(:)
  ! internal
  integer :: l,m,ind
  _dealloc(lm2aa)
  _dealloc(lm2bb)
  _dealloc(l2tlp1)
  _dealloc(l2tlm1)
  _dealloc(l21mhl)
  
  allocate(lm2aa(0:(lmx+1)**2-1))
  allocate(lm2bb(0:(lmx+1)**2-1))
  allocate(l2tlp1(0:lmx))
  allocate(l2tlm1(0:lmx))
  allocate(l21mhl(0:lmx))
  
  do l=0,lmx
    l2tlp1(l) = sqrt(2D0*l+1)
    if(l<1) cycle
    l21mhl(l) = sqrt((l-0.5D0)/l)
    l2tlm1(l) = sqrt(2D0*l-1)
  enddo
  
  do l=0,lmx
    do m=-l,l
      ind = l*(l+1)+m
      lm2aa(ind) = sqrt(1d0*l**2-m**2)
      lm2bb(ind) = sqrt(1d0*(l+1)**2-m**2)
    enddo
  enddo
  
  
end subroutine !  

!
! real spherical harmonics fast and tested
! needs m_fact: call init_fact()
! warn: sgn in m_fact :-)
!
subroutine rsphar(r,lmax,res) bind(c, name='rsphar')
  use m_fact, only : onediv4pi, rttwo, sgn, pi, init_fact
  use iso_c_binding, only: c_char, c_double, c_float, c_int64_t, c_int

  implicit none 
  real(c_double), intent(in)  :: r(3)
  integer(c_int), intent(in)  :: lmax
  real(c_double), intent(out) :: res((lmax+1)**2) 
  ! internal
  integer :: l,m,il1,il2,ind,ll2,twol,l2
  real(8) :: dd,phi,cc,ss,zz,cs,P,rt2lp1, xxpyy

  if(lmx<lmax) then
    call init_rsphar(lmax, lm2aa, lm2bb, l2tlp1, l2tlm1, l21mhl)
    lmx = lmax
  endif
  
  if(.not. allocated(sgn)) then
    call init_fact()
    !write(0,*) __FILE__, __LINE__
    !stop '!sgn'
  endif

  xxpyy = r(1)*r(1)+r(2)*r(2);
  dd=sqrt(xxpyy+r(3)*r(3))

  if (dd < 1D-10) then
    res=0; 
    res(1)=onediv4pi
    return
  endif

  if (r(1) .eq. 0D0) then; 
    phi=0.5D0*pi; if (r(2).lt.0D0) phi=-phi;
  else; 
    phi=atan(r(2)/r(1)); if (r(1).lt. 0D0) phi=phi+pi 
  endif

  ss=sqrt(xxpyy)/dd 
  cc=r(3)/dd
  res(1)=onediv4pi;
  if (lmax.eq.0) return

  do l=1,lmax 
    twol = l+l;
    l2   = l*l;
    il2  = l2+twol;
    il1  = l2-1
    res(il2+1)=-ss*l21mhl(l)*res(il1+1) 
    res(il2)=cc*l2tlm1(l)*res(il1+1)
  end do

  if (lmax.ge.2) then
    do m=0,lmax-2
      if (m.lt.lmax) then
        do l=m+1,lmax-1
          ind=l*(l+1)+m 
          zz=(l+l+1)*cc*res(ind+1)-lm2aa(ind)*res(ind-l-l+1) 
          res(ind+l+l+2+1)=zz/lm2bb(ind)
        end do
      endif
    end do
  endif

  do l=0,lmax
    ll2=l*(l+1)
    res(ll2+1) = res(ll2+1)*l2tlp1(l)
    rt2lp1 = l2tlp1(l)*rttwo
    do m=1,l
      cs=sin(m*phi)
      cc=cos(m*phi)
      P = res(ll2+m+1)*sgn(m)*rt2lp1;
      res(ll2+m+1)=cc*P;
      res(ll2-m+1)=cs*P;
    enddo
  enddo
  return
end subroutine !rsphar


!
! real spherical harmonics 
!
subroutine rsphar_vec(r,nc,lmax,res) bind(c, name='rsphar_vec')
  use iso_c_binding, only: c_char, c_double, c_float, c_int64_t, c_int

  implicit none
  integer(c_int64_t), intent(in)  :: nc
  real(c_double), intent(in)      :: r(3,nc)
  integer(c_int64_t), intent(in)  :: lmax
  real(c_double), intent(inout)   :: res((lmax+1)**2,nc) 
  ! internal
  integer(c_int64_t) :: ic
  
  if (nc<1) then
    write(6,*) nc
    stop 'nc<1'
  endif
  
  do ic=1,nc
    call rsphar(r(:,ic), int(lmax), res(:,ic))
  enddo

end subroutine !rsphar_vec


end module !m_rsphar
