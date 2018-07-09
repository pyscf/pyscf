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

from __future__ import print_function, division
import os,unittest,numpy as np
from scipy.sparse import linalg
from scipy.linalg import inv

n = 3
A = np.array([1.0+1j*0.1, 2.6+1j*0.1, 3.0+1j*0.1, \
              1.2+1j*0.1, -2.0+1j*0.1, 3.0+1j*0.1, \
              -5.8+1j*0.1, 2.0+1j*0.1, 3.0+1j*0.1], dtype=np.complex64).reshape([n,n])

b = np.array([1.0+1j*0.2, 2.5+1j*0.2, 4.0+1j*0.2], dtype=np.complex64)
x_ref = np.dot(inv(A), b)

def mvop(v): return np.dot(A, v)


class vext2veff_c():
  def __init__(self, omega, eps, n):
    self.omega = np.float32(omega)
    self.eps = np.float32(eps)
    self.shape = (n,n)
    self.dtype = np.complex64

  def matvec(self, v):
    return np.dot((self.omega+1j*self.eps)*A, v)
    

class KnowValues(unittest.TestCase):

  def test_scipy_gmres_den(self):
    """ This is a test on gmres method with dense matrix in scipy """
    x_itr,info = linalg.lgmres(A, b)
    derr = abs(x_ref-x_itr).sum()/x_ref.size
    self.assertLess(derr, 1e-6)

  def test_scipy_gmres_linop(self):
    """ This is a test on gmres method with linear operators in scipy """
    linop = linalg.LinearOperator((n,n), matvec=mvop, dtype=np.complex64)
    x_itr,info = linalg.lgmres(linop, b)
    derr = abs(x_ref-x_itr).sum()/x_ref.size
    self.assertLess(derr, 1e-6)

  def test_scipy_gmres_linop_parameter(self):
    """ This is a test on gmres method with a parameter-dependent linear operator """
    for omega in np.linspace(-10.0, 10.0, 10):
      for eps in np.linspace(-10.0, 10.0, 10):
        
        linop_param = linalg.aslinearoperator(vext2veff_c(omega, eps, n))
        
        Aparam = np.zeros((n,n), np.complex64)
        for i in range(n):
          uv = np.zeros(n, np.complex64); uv[i] = 1.0
          Aparam[:,i] = linop_param.matvec(uv)
        x_ref = np.dot(inv(Aparam), b)
    
        x_itr,info = linalg.lgmres(linop_param, b)
        derr = abs(x_ref-x_itr).sum()/x_ref.size
        self.assertLess(derr, 1e-6)

if __name__ == "__main__": unittest.main()
