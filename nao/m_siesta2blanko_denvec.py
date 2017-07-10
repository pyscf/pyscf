import numpy
from numpy import empty 

#
#
#
def _siesta2blanko_denvec(orb2m, vec, orb_sc2orb_uc=None):

  n,nreim = vec.shape
  if(nreim==1):
    for i in range(n): 
      j = i
      if(j>=n): j = orb_sc2orb_uc[i]
      vec[i,0] = vec[i,0]*(-1.0)**orb2m[j]
  elif(nreim==2):
    for i in range(n):
      j = i
      if(j>=n): j = orb_sc2orb_uc[i]
      v = numpy.complex_(vec[i,0],vec[i,1])*(-1.0)**orb2m[j]
      vec[i,0],vec[i,1] = v[0],v[1]
  else:
    raise SystemError('!nreim')

  return(0)
