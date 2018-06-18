# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import numpy

#
#
#
def spline_diff2(h,yin,yp1,ypn):
  """
subroutine spline(delt,y,n,yp1,ypn,y2) 
!! Cubic Spline Interpolation.
!! Adapted from Numerical Recipes routines for a uniform grid
!! D. Sanchez-Portal, Oct. 1996.
!! Alberto Garcia, June 2000
!! Peter Koval, Dec 2009

  implicit none
  !! external
  integer, intent(in)    :: n
  real(8), intent(in)   :: delt, yp1, ypn, y(:)
  real(8), intent(out)  :: y2(:)

  !! internal
  integer i, k
  real(8) sig, p, qn, un

  real(8), allocatable  :: u(:)
  allocate(u(n));

  if (yp1.eq. huge(1D0)) then
    y2(1)=0
    u(1)=0
  else
    y2(1)=-0.5D0
    u(1)=(3.0D0/delt)*((y(2)-y(1))/delt-yp1)
  endif

  do i=2,n-1
    sig=0.5D0
    p=sig*y2(i-1)+2
    y2(i)=(sig-1)/p
    u(i)=(3*( y(i+1)+y(i-1)-2*y(i) )/(delt*delt)-sig*u(i-1))/p
  enddo

  if (ypn.eq.huge(1D0)) then
    qn=0; un=0
  else
    qn=0.5D0; un=(3/delt)*(ypn-(y(n)-y(n-1))/delt)
  endif

  y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1)
  do k=n-1,1,-1
    y2(k)=y2(k)*y2(k+1)+u(k)
  enddo
end subroutine !spline
  """
  assert(type(yin)==numpy.ndarray)
  
  h2 = h*h
  n = len(yin)
  u = numpy.zeros((n), dtype='float64')
  yout = numpy.zeros((n), dtype='float64')
  
  if yp1<1e300 : yout[0],u[0]=-0.5, (3.0/h)*((yin[1]-yin[0])/h-yp1)

  for i in range(1,n-1):
    p = 0.5*yout[i-1]+2.0
    yout[i] = -0.5 / p
    u[i]=(3*( yin[i+1]+yin[i-1]-2*yin[i] )/h2-0.5*u[i-1])/p

  qn,un = 0.0,0.0
  if ypn<1e300 : qn,un = 0.5,(3.0/h)*( ypn-(yin[n-1]-yin[n-2])/h)

  yout[n-1]=(un-qn*u[n-2])/(qn*yout[n-2]+1)

  for k in range(n-2,-1,-1): yout[k]=yout[k]*yout[k+1]+u[k]
  
  return(yout)
