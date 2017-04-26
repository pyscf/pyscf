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
    
    self.rr,self.pp = ao_log.rr,ao_log.pp
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

      mu2jd = []
      for j,evs in enumerate(ldp['j2eva']):
        for domi,ev in enumerate(evs):
          if ev>tol: mu2jd.append([j,domi])

      nmult=len(mu2jd)
      mu2j = np.array([jd[0] for jd in mu2jd], dtype='int64')
      mu2s = np.array([0]+[sum(2*mu2j[0:mu+1]+1) for mu in range(nmult)], dtype='int64')
      mu2rcut = np.array([ao_log.sp2rcut[sp]]*nmult, dtype='float64')
      
      self.sp2nmult[sp]=nmult
      self.sp_mu2j.append(mu2j)
      self.sp_mu2rcut.append(mu2rcut)
      self.sp_mu2s.append(mu2s)

      mu2ff = np.zeros((nmult, lvc.nr), dtype='float64')
      for mu,[j,domi] in enumerate(mu2jd): mu2ff[mu,:] = ldp['j2xff'][j][domi,:]
      self.psi_log.append(mu2ff)

      no,npf= lvc.sp2norbs[sp], sum(2*mu2j+1)  # count number of orbitals and product functions
      mu2ww = np.zeros((npf,no,no), dtype='float64')
      for [j,domi],s in zip(mu2jd,mu2s): mu2ww[s:s+2*j+1,:,:] = ldp['j2xww'][j][domi,0:2*j+1,:,:]

      self.sp2vertex.append(mu2ww)

  #
  #
  #
  def _moments(self):
    rr3dr = self.rr**3*np.log(self.rr[1]/self.rr[0])
    rr4dr = self.rr*rr3dr
    self.sp2mom0,self.sp2mom1,cs,cd = [],[],np.sqrt(4*np.pi),np.sqrt(4*np.pi/3.0)
    for sp,nmu in enumerate(self.sp2nmult):
      nfunct=sum(2*self.sp_mu2j[sp]+1)
      mom0 = np.zeros((nfunct), dtype='float64')
      d = np.zeros((nfunct,3), dtype='float64')
      for mu,[j,s] in enumerate(zip(self.sp_mu2j[sp],self.sp_mu2s[sp])):
        if j==0:                 mom0[s]  = cs*sum(self.psi_log[sp][mu,:]*rr3dr)
        if j==1: d[s,1]=d[s+1,2]=d[s+2,0] = cd*sum(self.psi_log[sp][mu,:]*rr4dr)
      self.sp2mom0.append(mom0)
      self.sp2mom1.append(d)
