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

from __future__ import division, print_function
import numpy as np

# Generates the Gauss-Legendre knots and weights for an arbitrary integration interval (a..b)
def leggauss_ab(n=96, a=-1.0, b=1.0):
  assert(n>0)
  x,w = np.polynomial.legendre.leggauss(n)
  x = (b-a) * 0.5 * x+(b+a) * 0.5
  w = w * (b-a) * 0.5
  return x,w

def gauss_legendre(n, charge=None, atom_id=None, **kwargs):
    '''
    Generates the Gauss-Legendre knots and weights
    '''
    if 'atom2rcut' in kwargs and atom_id is not None:
        rcut = kwargs['atom2rcut'][atom_id]
    else:
        rcut = 1.
    return leggauss_ab(n, 0, rcut)

# Generates the Gauss-Legendre knots and weights
#   Fortran version is written by James Talman
def gauleg_ab(n=96, a=-1.0, b=1.0, eps=1.0e-15, cmx=15):
  assert(n>0)
  x,w = np.zeros((n)),np.zeros((n))
  
  m=(n+1)//2
  for i in range(m):
    z=np.cos(np.pi*(i+1-0.25)/(n+0.5))
    for c in range(cmx):
      p1,p2=1.0,0.0
      for j in range(1,n+1):
        p3=p2
        p2=p1
        p1=((2.0*j-1)*z*p2-(j-1)*p3)/j
      pp=n*(z*p1-p2)/(z*z-1.0)
      z1=z
      z=z1-p1/pp
      if abs(z-z1)<eps : exit

    x[ i],x[-i-1] = -z,z
    w[ i]=w[-i-1] = 2.0/((1.0-z*z)*pp*pp)

  x = (b-a) * 0.5 * x+(b+a) * 0.5
  w = w * (b-a) * 0.5
  return x,w

if __name__ == '__main__':
  from pyscf.nao.m_gauleg import gauleg_ab
  from numpy.polynomial.legendre import leggauss
  for n in range(1,320):
    xx1,ww1=leggauss(n)
    xx2,ww2=gauleg_ab(n)
    if np.allclose(xx1,xx2) and np.allclose(ww1,ww2):
      print(n, 'allclose')
    else:
      print(n, (xx1-xx2).sum(), (ww1-ww2).sum())
