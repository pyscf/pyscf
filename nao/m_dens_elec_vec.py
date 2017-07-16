from __future__ import print_function, division
from numpy import require, zeros, linalg, compress
from scipy.spatial.distance import cdist
from timeit import default_timer as timer

def dens_elec_vec(sv, crds, dm):
  """ Compute the electronic density using vectorized oprators """
  from pyscf.nao.m_rsphar_vec import rsphar_vec as rsphar_vec_python

  assert crds.ndim==2  
  assert crds.shape[-1]==3  
  nc = crds.shape[0]
  
  lmax = sv.ao_log.jmx
  for ia,[ra,sp] in enumerate(zip(sv.atom2coord,sv.atom2sp)):
    lngs = cdist(crds, sv.atom2coord[ia:ia+1,:])
    bmask = lngs<sv.ao_log.sp2rcut[sp]
    crds_selec = compress(bmask[:,0], crds, axis=0)
    t1 = timer()
    rsh1 = rsphar_vec_python(crds_selec, lmax)
    t2 = timer(); print(t2-t1); t1 = timer()
        
  return 0
