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

import numpy as np
from pyscf.nao.m_xjl import xjl

#
#
#
class sbt_c():
  '''
  Spherical Bessel Transform by James Talman. Functions are given on logarithmic mesh
  See m_log_mesh
  Args:
    nr : integer, number of points on radial mesh
    rr : array of points in coordinate space
    kk : array of points in momentum space
    lmax : integer, maximal angular momentum necessary
    with_sqrt_pi_2 : if one, then transforms will be multiplied by sqrt(pi/2)
    fft_flags : ??
  Returns:
    a class preinitialized to perform the spherical Bessel Transform
  
  Examples:
    label = 'siesta'
    sv = system_vars_c(label)
    sbt = sbt_c(sv.ao_log.rr, sv.ao_log.pp)
    print(sbt.exe(sv.ao_log.psi_log[0,0,:], 0))
  '''
  def __init__(self, rr, kk, lmax=12, with_sqrt_pi_2=True, fft_flags=None):
    assert(type(rr)==np.ndarray)
    assert(rr[0]>0.0)
    assert(type(kk)==np.ndarray)
    assert(kk[0]>0.0)
    self.nr = len(rr)
    n = self.nr
    assert(self.nr>1)
    assert(lmax>-1)
    self.rr,self.kk = rr,kk
    nr2, self.rr3, self.kk3 = self.nr*2, rr**3, kk**3
    self.rmin,self.kmin = rr[0],kk[0]
    self.rhomin,self.kapmin= np.log(self.rmin),np.log(self.kmin)

    self.dr_jt = np.log(rr[1]/rr[0])
    dr = self.dr_jt
    dt = 2.0*np.pi/(nr2*dr)
    
    self._smallr = self.rmin*np.array([np.exp(-dr*(n-i)) for i in range(n)], dtype='float64')
    self._premult = np.array([np.exp(1.5*dr*(i-n)) for i in range(2*n)], dtype='float64')

    coeff = 1.0/np.sqrt(np.pi/2.0) if with_sqrt_pi_2  else 1.0
    self._postdiv = np.array([coeff*np.exp(-1.5*dr*i) for i in range(n)], dtype='float64')
  
    temp1 = np.zeros((nr2), dtype='complex128')
    temp2 = np.zeros((nr2), dtype='complex128')
    temp1[0] = 1.0
    temp2 = np.fft.fft(temp1)
    xx = sum(np.real(temp2))
    if abs(nr2-xx)>1e-10 : raise SystemError('err: sbt_plan: problem with fftw sum(temp2):')
 
    self._mult_table1 = np.zeros((lmax+1, self.nr), dtype='complex128')
    for it in range(n):
      tt = it*dt                           # Define a t value
      phi3 = (self.kapmin+self.rhomin)*tt  # See Eq. (33)
      rad,phi = np.sqrt(10.5**2+tt**2),np.arctan((2.0*tt)/21.0)
      phi1 = -10.0*phi-np.log(rad)*tt+tt+np.sin(phi)/(12.0*rad) \
        -np.sin(3.0*phi)/(360.0*rad**3)+np.sin(5.0*phi)/(1260.0*rad**5) \
        -np.sin(7.0*phi)/(1680.0*rad**7)
        
      for ix in range(1,11): phi1=phi1+np.arctan((2.0*tt)/(2.0*ix-1))  # see Eqs. (27) and (28)

      phi2 = -np.arctan(1.0) if tt>200.0 else -np.arctan(np.sinh(np.pi*tt/2)/np.cosh(np.pi*tt/2))  # see Eq. (20)
      phi = phi1+phi2+phi3

      self._mult_table1[0,it] = np.sqrt(np.pi/2)*np.exp(1j*phi)/n  # Eq. (18)
      if it==0 : self._mult_table1[0,it] = 0.5*self._mult_table1[0,it]
      phi = -phi2 - np.arctan(2.0*tt)
      if lmax>0 : self._mult_table1[1,it] = np.exp(2.0*1j*phi)*self._mult_table1[0,it] # See Eq. (21)

      #    Apply Eq. (24)
      for lk in range(1,lmax):
        phi = -np.arctan(2*tt/(2*lk+1))
        self._mult_table1[lk+1,it] = np.exp(2.0*1j*phi)*self._mult_table1[lk-1,it]
    # END of it in range(n):

    # make the initialization for the calculation at small k values for 2N mesh values
    self._mult_table2 = np.zeros((lmax+1, self.nr+1), dtype='complex128')
    j_ltable = np.zeros((lmax+1,nr2), dtype='float64')

    for i in range(nr2): j_ltable[0:lmax+1,i] = xjl( np.exp(self.rhomin+self.kapmin+i*dr), lmax )

    for ll in range(lmax+1):
      self._mult_table2[ll,:] = np.fft.rfft(j_ltable[ll,:]) /nr2
    if with_sqrt_pi_2 : self._mult_table2 = self._mult_table2/np.sqrt(np.pi/2)
    
  # 
  # The calculation of the Sperical Bessel Transform for a given data...
  #
  def sbt(self, ff, am, direction=1, npow=0) :
    """
  Args:
    ff : numpy array containing radial orbital (values of radial orbital on logarithmic grid) to be transformed. The data must be on self.rr grid or self.kk grid provided during initialization.
    am : angular momentum of the radial orbital ff[:]
    direction : 1 -- real-space --> momentum space transform; -1 -- momentum space --> real-space transform.
    npow : additional power for the shape of the orbital
      f(xyz) = rr[i]**npow * ff[i] * Y_lm( xyz )
  Result:
    gg : numpy array containing the result of the Spherical Bessel Transform
    gg(k) = int_0^infty  ff(r) j_{am}(k*r) r**2  dr  ( direction ==  1 )
    gg(r) = int_0^infty  ff(k) j_{am}(k*r) k**2  dk  ( direction == -1 )
    """
    assert(type(ff)==np.ndarray)
    assert(len(ff)==self.nr)
    assert(am > -1)
    assert(am < self._mult_table1.shape[0])
  
    if direction==1 :
      rmin, kmin, ptr_rr3 = self.rmin, self.kmin, self.rr3
      dr = np.log(self.rr[1]/self.rr[0])
      C = ff[0]/self.rr[0]**(npow+am)
    elif direction==-1 :
      rmin, kmin, ptr_rr3 = self.kmin, self.rmin, self.kk3
      dr = np.log(self.kk[1]/self.kk[0])
      C = ff[0]/self.kk[0]**(npow+am)
    else:
      raise SystemError('!direction=+/-1')

    gg = np.zeros((self.nr), dtype='float64')     # Allocate the result
    
    # make the calculation for LARGE k values extend the input to the doubled mesh, extrapolating the input as C r**(np+li)
    nr2 = self.nr*2
    r2c_in = np.zeros((nr2), dtype='float64')
    r2c_in[0:self.nr] = C*self._premult[0:self.nr]*self._smallr[0:self.nr]**(npow+am)
    r2c_in[self.nr:nr2] = self._premult[self.nr:nr2]*ff[0:self.nr]
    r2c_out = np.fft.rfft(r2c_in)
    
    temp1 = np.zeros((nr2), dtype='complex128')
    temp1[0:self.nr] = np.conj(r2c_out[0:self.nr])*self._mult_table1[am,0:self.nr]
    temp2 = np.fft.ifft(temp1)*nr2
    gg[0:self.nr] = (rmin/kmin)**1.5 * (temp2[self.nr:nr2]).real * self._postdiv[0:self.nr]
        
    # obtain the SMALL k results in the array c2r_out
    r2c_in[0:self.nr] = ptr_rr3[0:self.nr] * ff[0:self.nr]
    r2c_in[self.nr:nr2] = 0.0
    r2c_out = np.fft.rfft(r2c_in)

    c2r_in = np.conj(r2c_out[0:self.nr+1]) * self._mult_table2[am,0:self.nr+1]
    c2r_out = np.fft.irfft(c2r_in)*dr*nr2

    r2c_in[0:self.nr] = abs(gg[0:self.nr]-c2r_out[0:self.nr])
    kdiv = np.argmin(r2c_in[0:self.nr])

    gg[0:kdiv] = c2r_out[0:kdiv]
    return gg
