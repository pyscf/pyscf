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
def spline_interp(h,yy,yy_diff2,x) :
  """
!subroutine splint(delt,ya,y2a,n,x,y,dydx)
!! Cubic Spline Interpolation.
!! Adapted from Numerical Recipes for a uniform grid.

  implicit none
  !! external
  integer,  intent(in) :: n
  real(8), intent(in)  :: delt, ya(n), y2a(n), x
  real(8), intent(out) :: y
!  real(dp), intent(out) :: y, dydx

  !! internal
  integer  :: nlo, nhi
  real(8) :: a, b

  nlo=max(int(x/delt)+1,1)
  if(nlo>n-1) then; y=0; return; endif
  !if(nlo>n-1) then; y=0; dydx=0; return; endif
  nhi=min(nlo+1,n)
  a=nhi-x/delt-1
  b=1.0D0-a
  y=a*ya(nlo)+b*ya(nhi)+((a**3-a)*y2a(nlo)+(b**3-b)*y2a(nhi))*(delt**2)/6D0
!  dydx=(ya(nhi)-ya(nlo))/delt + (-((3*(a**2)-1._dp)*y2a(nlo))+ (3*(b**2)-1._dp)*y2a(nhi))*delt/6._dp
end subroutine ! splint
  """
  assert(type(yy)==numpy.ndarray)
  
  n=yy.shape[0]
  nlo=max(int(x/h),0)
  if nlo>n-1: return(0.0)
  nhi=min(nlo+1,n-1)
  a=nhi-x/h # This is checked... different to Fortran version due to 0-based arrays
  b=1.0-a
  y=a*yy[nlo]+b*yy[nhi]+((a**3-a)*yy_diff2[nlo]+(b**3-b)*yy_diff2[nhi])*(h**2)/6.0
  return(y)
