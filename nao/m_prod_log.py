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
    self.sp_mu2s = []    # list of numpy arrays containing the starting index for each radial multiplett
    self.sp2vertex = []  # list of numpy arrays containing the vertex coefficients
    
    for sp in range(ao_log.nspecies):
      ldp = lvc.get_local_vertex(sp)

      mu2mjd = []
      for j,evs in enumerate(ldp['j2eva']):
        for domi,ev in enumerate(evs):
          if ev>tol: mu2mjd.append([len(mu2mjd),j,domi])

      nmult=len(mu2mjd)
      mu2j = np.array([mjd[1] for mjd in mu2mjd], dtype='int64')
      mu2s = np.array([0]+[sum(2*mu2j[0:mu+1]+1) for mu in range(nmult)], dtype='int64')
      mu2rcut = np.array([[ao_log.sp2rcut[sp]]*nmult], dtype='float64')
      
      self.sp2nmult[sp]=nmult
      self.sp_mu2j.append(mu2j)
      self.sp_mu2rcut.append(mu2rcut)
      self.sp_mu2s.append(mu2s)

      mu2ff = np.zeros((nmult, lvc.nr), dtype='float64')
      for mu,j,domi in mu2mjd: mu2ff[mu,:] = ldp['j2xff'][j][domi,:]
      self.psi_log.append(mu2ff)

      no,npf,nmua = sum(2*ao_log.sp_mu2j[sp]+1), sum(2*mu2j+1), len(ao_log.sp_mu2j[sp])  # count number of orbitals and product functions
      jmx_prd = max(mu2j)
      mu2ww = np.zeros((no,no,npf), dtype='float64')
      for mu,j,domi in mu2mjd:
        xww = ldp['j2xww'][j]
        s=mu2s[mu]
        for mu1,j1,s1 in zip(range(nmua), ao_log.sp_mu2j[sp], ao_log.sp_mu2s[sp]):
          for mu2,j2,s2 in zip(range(nmua), ao_log.sp_mu2j[sp], ao_log.sp_mu2s[sp]):
            for m1,jm1 in zip( range(-j1,j1+1), range(j1*(j1+1)-j1,j1*(j1+1)+j1+1) ) :
              for m2,jm2 in zip( range(-j2,j2+1), range(j2*(j2+1)-j2,j2*(j2+1)+j2+1) ):
                mu2ww[s1+j1+m1,s2+j2+m2,s:s+2*j+1] = xww[domi,jm1,jm2,0:2*j+1]

      self.sp2vertex.append(mu2ww)

  ################################################
  
