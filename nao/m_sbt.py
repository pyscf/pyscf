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
  '''
  def __init__(self, rr=None, kk=None, lmax=12, with_sqrt_pi_2=True, fft_flags=None):
    assert(type(rr)==np.ndarray)
    assert(rr[0]>0.0)
    assert(type(kk)==np.ndarray)
    assert(kk[0]>0.0)
    self.nr = len(rr)
    n = self.nr
    assert(self.nr>1)
    self.lmax = lmax
    assert(self.lmax>-1)
    self.rr,self.kk = rr,kk
    self.with_sqrt_pi_2 = with_sqrt_pi_2
    self.fft_flags = fft_flags
    self.nr2, self.rr3, self.kk3 = self.nr*2, rr**3, kk**3
    self.rmin,self.kmin = rr[0],kk[0]
    self.rhomin,self.kapmin= np.log(self.rmin),np.log(self.kmin)

    dr = np.log(rr[2]/rr[1])
    dt = 2.0*np.pi/(self.nr2*dr)
    
    self.smallr = self.rmin*np.array([np.exp(-dr*(n-i)) for i in range(n)], dtype='float64')
    self.premult = np.array([np.exp(1.5*dr*(i-n)) for i in range(2*n)], dtype='float64')

    coeff = 1.0/np.sqrt(np.pi/2.0) if with_sqrt_pi_2  else 1.0
    self.postdiv = np.array([coeff*np.exp(-1.5*dr*i) for i in range(n)], dtype='float64')
  
    self.temp1 = np.zeros((self.nr2), dtype='complex128')
    self.temp2 = np.zeros((self.nr2), dtype='complex128')
    self.temp1[0] = 1.0
    self.temp2 = np.fft.fft(self.temp1)
    xx = sum(np.real(self.temp2))
    if abs(self.nr2-xx)>1e-10 : raise SystemError('err: sbt_plan: problem with fftw sum(temp2):')
 
    self.mult_table1 = np.zeros((self.lmax+1, self.nr), dtype='complex128')
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
     
      self.mult_table1[0,it] = np.sqrt(np.pi/2)*np.exp(1j*phi)/n  # Eq. (18)
      if it==0 : self.mult_table1[0,it] = 0.5*self.mult_table1[0,it]
      phi = -phi2 - np.arctan(2.0*tt)
      if self.lmax>0 : self.mult_table1[1,it] = np.exp(2.0*1j*phi)*self.mult_table1[0,it] # See Eq. (21)

      #    Apply Eq. (24)
      for lk in range(1,self.lmax-1):
        phi = -np.arctan(2*tt/(2*lk+1))
        self.mult_table1[lk+1,it] = np.exp(2.0*1j*phi)*self.mult_table1[lk-1,it]
    # END of it in range(n):

  #!   make the initialization for the calculation at small k values
  #!   for 2N mesh values
    self.mult_table2 = np.zeros((self.lmax+1, self.nr+1), dtype='complex128')
    j_ltable = np.zeros((self.lmax+1,self.nr2), dtype='float64')

    for i in range(self.nr2):
      xx = np.exp(self.rhomin+self.kapmin+(i-1)*dr)
      j_ltable[0:self.lmax+1,i] = xjl(xx,self.lmax)

    for ll in range(self.lmax+1):
      self.mult_table2[ll,:] = np.fft.rfft(j_ltable[ll,:])
    if with_sqrt_pi_2 : self.mult_table2 = self.mult_table2/np.sqrt(np.pi/2)
