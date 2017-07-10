import numpy
from numpy import empty 

#
#
#
def _siesta2blanko_denvec(orb2m, vec, orb_sc2orb_uc=None):

  n,nreim = vec.shape

  if orb_sc2orb_uc is None:
    orb_sc2m = orb2m
  else:
    orb_sc2m = np.zeros_like(orb_sc2orb_uc)
    for orb_sc,orb_uc in enumerate(orb_sc2orb_uc): orb_sc2m[orb_sc] = orb2m[orb_uc]

  orb2ph = (-1)**orb_sc2m
  
  if(nreim==1):
    vec[:,0] = vec[:,0]*orb2ph[:]

  elif(nreim==2):
    
    for i in range(n):
      v = numpy.complex_(vec[i,0],vec[i,1])*orb2ph[i]
      vec[i,0],vec[i,1] = v[0],v[1]

  else:
    raise SystemError('!nreim')

  return(0)
