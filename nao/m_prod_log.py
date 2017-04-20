from __future__ import print_function
from __future__ import division
import numpy as np
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_local_vertex import local_vertex_c
#
#
#
class prod_log_c(ao_log_c):
  '''
  Holder of product functions and vertices.
  Args:
  Returns:
  Examples:
  '''
  def __init__(self, ao_log, tol=1e-10):
    
    self.sp2nmult = np.zeros((ao_log.nspecies), dtype='int64')
    self.nmultmax = max(self.sp2nmult)
    
    lvc = local_vertex_c(ao_log) # constructor of local vertices
    self.psi_log = []  # it is impossible to use constructor of ao_log, no ? Therefore, I define myself...
    self.sp_mu2rcut = [] # list of numpy arrays containing the maximal radii
    self.sp_mu2j = []    # list of numpy arrays containing the angular momentum of the radial function
    
    for sp in range(ao_log.nspecies):
      ldp = lvc.get_local_vertex(sp)
      self.sp2nmult[sp] = sum([sum(evs>tol) for evs in ldp['j2eva']])

      nam, nmult = len(ldp['j2eva']), self.sp2nmult[sp]
      mu2ff = np.zeros((nmult, lvc.nr), dtype='float64')
      mu2j,mu2rcut  = np.zeros((nmult), dtype='int64'),np.zeros((nmult), dtype='float64')
      mu = -1
      for evs,xff,j in zip(ldp['j2eva'],ldp['j2xff'],range(nam)):
        for domi,ev in enumerate(evs):
          if ev<=tol: continue
          mu+=1
          mu2ff[mu,:],mu2j[mu],mu2rcut[mu] = xff[domi,:], j, np.amax(ao_log.sp_mu2rcut[sp])

      mu2s = np.zeros((len(mu2j)+1), dtype='int64')
      for mu in range(len(mu2j)): mu2s[mu+1] = sum(2*mu2j[0:mu+1]+1) # counting within specie

      self.psi_log.append(mu2ff);  self.sp_mu2j.append(mu2j);  self.sp_mu2rcut.append(mu2rcut)

      no,npf = sum(2*ao_log.sp_mu2j[sp]+1), sum(2*mu2j+1)  # count number of orbitals and product functions
      mu2ww = np.zeros((no,no,npf), dtype='float64')
      mu = -1
      for evs,xww,j in zip(ldp['j2eva'],ldp['j2xww'],range(nam)):
        jmx_prd = (xww.shape[3]-1)//2
        for domi,ev in enumerate(evs):
          if ev<=tol: continue
          for mu1,j1 in enumerate(ao_log.sp_mu2j[sp]):
            for mu2,j2 in enumerate(ao_log.sp_mu2j[sp]):
              for m1,jm1 in zip( range(-j1,j1+1), range(j2*(j2+1)-j2,j2*(j2+1)+j2+1) ) :
                for m2,jm2 in zip( range(-j2,j2+1), range(j2*(j2+1)-j2,j2*(j2+1)+j2+1) ):
                  for m3 in range(-j,j+1):
                    mu2ww[0,0,0] = xww[domi,jm1,jm2,jmx_prd+m3]

      print(sp, no, npf)
      
      



