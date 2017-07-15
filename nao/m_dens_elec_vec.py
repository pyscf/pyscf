from __future__ import print_function, division
from numpy import require, zeros, linalg, compress
from scipy.spatial.distance import cdist

def dens_elec_vec(sv, crds, dm):
  """ Compute the electronic density using vectorized oprators """
  assert crds.ndim==2  
  assert crds.shape[-1]==3  
  nc = crds.shape[0]
  
  for ia,[ra,sp] in enumerate(zip(sv.atom2coord,sv.atom2sp)):
    lngs = cdist(crds, sv.atom2coord[ia:ia+1,:])
    bmask = lngs<sv.ao_log.sp2rcut[sp]
    crds_selec = compress(bmask[:,0], crds, axis=0)
    lngs_selec = compress(bmask[:,0], lngs)
    print(ia, lngs.shape, crds.shape, crds_selec.shape, lngs_selec.shape)
    
  return 0
