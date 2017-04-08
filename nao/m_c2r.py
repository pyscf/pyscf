import numpy as np
from pyscf.nao.m_fact import sgn

#
#
#
class c2r_c():
  """
    Conversion from complex to real harmonics
  """
  def __init__(self, j):
    self._j = j
    self.c2r = np.zeros( (2*j+1, 2*j+1), dtype='complex128')
    self.hc_c2r = np.zeros( (2*j+1, 2*j+1), dtype='complex128')
    
    self.c2r[j,j]=1.0
    for m in range(1,j+1):
      self.c2r[m+j, m+j] = sgn[m] * np.sqrt(0.5) 
      self.c2r[m+j,-m+j] = np.sqrt(0.5) 
      self.c2r[-m+j,-m+j]= 1j*np.sqrt(0.5)
      self.c2r[-m+j, m+j]= -sgn[m] * 1j * np.sqrt(0.5)

    self.conj_c2r = np.conjugate(self.c2r)
    self.tr_c2r = np.transpose(self.c2r)

  #
  #
  #
  def c2r_(self, j1,j2, jm,cmat,rmat,mat):
    
    mat.fill(0.0)
    
    for mm1 in range(-j1,j1+1):
      for mm2 in range(-j2,j2+1):
        if mm2 == 0 :
          mat[mm1+jm,mm2+jm] = cmat[mm1+jm,mm2+jm]*self.tr_c2r[mm2+self._j,mm2+self._j]
        else :
          mat[mm1+jm,mm2+jm] = \
            (cmat[mm1+jm,mm2+jm]*self.tr_c2r[mm2+self._j,mm2+self._j] + \
             cmat[mm1+jm,-mm2+jm]*self.tr_c2r[-mm2+self._j,mm2+self._j]).real

    for mm2 in range(-j2,j2+1):
      for mm1 in range(-j1,j1+1):
        if mm1 == 0 :
          rmat[mm1+jm, mm2+jm] = (self.conj_c2r[mm1+self._j,mm1+self._j]*mat[mm1+jm,mm2+jm]).real
        else :
          rmat[mm1+jm, mm2+jm] = \
            (self.conj_c2r[mm1+self._j,mm1+self._j] * mat[mm1+jm,mm2+jm] + \
             self.conj_c2r[mm1+self._j,-mm1+self._j] * mat[-mm1+jm,mm2+jm]).real
