import numpy as np

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
    
    v = np.zeros(n, dtype=np.complex128)
    for i,reim in enumerate(vec): v[i] = reim[0]+1j*reim[1]
    v = v * orb2ph
    vec[:,0] = v.real
    vec[:,1] = v.imag

  else:
    raise SystemError('!nreim')

  return(0)
