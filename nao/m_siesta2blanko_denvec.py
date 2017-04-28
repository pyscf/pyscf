import numpy
from numpy import empty 

#
#
#
def _siesta2blanko_denvec(orb2m, vec, orb_sc2orb_uc=None):

  nreim,n = vec.shape
  if(nreim==1):
    for i in range(n): 
      j = i
      if(j>=n): j = orb_sc2orb_uc[i]
      vec[0,i] = vec[0,i]*(-1.0)**orb2m[j]
  elif(nreim==2):
    for i in range(n):
      j = i
      if(j>=n): j = orb_sc2orb_uc[i]
      v = numpy.complex_(vec[0:1,i])*(-1.0)**orb2m[j]
      vec[0,i] = v.real
      vec[1,i] = v.imag
  else:
    raise SystemError('!nreim')

  return(0)
