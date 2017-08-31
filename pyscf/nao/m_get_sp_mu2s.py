import numpy as np

#
#
#
def get_sp_mu2s(sp2nmult,sp_mu2j):
  """
    Generates list of start indices for atomic orbitals, based on the counting arrays
  """
  sp_mu2s = []
  for sp,nmu in enumerate(sp2nmult):
    mu2s = np.zeros((nmu+1), dtype='int64')
    for mu in range(nmu): mu2s[mu+1] = sum(2*sp_mu2j[sp][0:mu+1]+1)
    sp_mu2s.append(mu2s)
    
  return sp_mu2s
