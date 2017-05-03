from __future__ import print_function, division
import numpy as np
from pyscf.nao.m_dipole_ni import dipole_ni
from pyscf.nao.m_overlap_ni import overlap_ni
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_local_vertex import local_vertex_c


def comp_moments(self):
  """
    Computes the scalar and dipole moments of the product functions
    Args:
      argument can be  prod_log_c    or   ao_log_c
  """
  rr3dr = self.rr**3*np.log(self.rr[1]/self.rr[0])
  rr4dr = self.rr*rr3dr
  sp2mom0,sp2mom1,cs,cd = [],[],np.sqrt(4*np.pi),np.sqrt(4*np.pi/3.0)
  for sp,nmu in enumerate(self.sp2nmult):
    nfunct=sum(2*self.sp_mu2j[sp]+1)
    mom0 = np.zeros((nfunct), dtype='float64')
    d = np.zeros((nfunct,3), dtype='float64')
    for mu,[j,s] in enumerate(zip(self.sp_mu2j[sp],self.sp_mu2s[sp])):
      if j==0:                 mom0[s]  = cs*sum(self.psi_log[sp][mu,:]*rr3dr)
      if j==1: d[s,1]=d[s+1,2]=d[s+2,0] = cd*sum(self.psi_log[sp][mu,:]*rr4dr)
    sp2mom0.append(mom0)
    sp2mom1.append(d)
  return sp2mom0,sp2mom1

#
#
#
def overlap_check(prod_log, overlap_funct=overlap_ni, **kvargs):
  """ Computes the allclose(), mean absolute error and maximal error of the overlap reproduced by the (local) vertex."""
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  me = ao_matelem_c(prod_log.ao_log)
  sp2mom0,sp2mom1 = comp_moments(prod_log)
  mael,mxel,acl=[],[],[]
  for sp,[vertex,mom0] in enumerate(zip(prod_log.sp2vertex,sp2mom0)):
    oo_ref = overlap_funct(me,sp,sp,np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),**kvargs)
    oo = np.einsum('ijk,i->jk', vertex, mom0)
    ac = np.allclose(oo_ref, oo, atol=prod_log.tol*10, rtol=prod_log.tol)
    mae = abs(oo_ref-oo).sum()/oo.size
    mxe = abs(oo_ref-oo).max()
    acl.append(ac); mael.append(mae); mxel.append(mxe)
    if not ac: print('overlap check:', sp, mae, mxe, prod_log.tol) 
  return mael,mxel,acl

#
#
#
def dipole_check(sv, prod_log, dipole_funct=dipole_ni, **kvargs):
  """ Computes the allclose(), mean absolute error and maximal error of the dipoles reproduced by the (local) vertex. """
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  me = ao_matelem_c(prod_log.ao_log)
  sp2mom0,sp2mom1 = comp_moments(prod_log)
  mael,mxel,acl=[],[],[]
  for atm,[sp,coord] in enumerate(zip(sv.atom2sp,sv.atom2coord)):
    dip_moms = np.einsum('j,k->jk', sp2mom0[sp],coord)+sp2mom1[sp]
    koo2dipme = np.einsum('pab,pc->cab', prod_log.sp2vertex[sp],dip_moms) 
    dipme_ref = dipole_funct(me,sp,sp,coord,coord, **kvargs)
    ac = np.allclose(dipme_ref, koo2dipme, atol=prod_log.tol*10, rtol=prod_log.tol)
    mae = abs(koo2dipme-dipme_ref).sum()/koo2dipme.size
    mxe = abs(koo2dipme-dipme_ref).max()
    acl.append(ac); mael.append(mae); mxel.append(mxe)
    if not ac: print('dipole check:', sp, mae, mxe, prod_log.tol) 
  return mael,mxel,acl

#
#
#
class prod_log_c(ao_log_c):
  '''
  Holder of product functions and vertices.
  Args:
    ao_log, i.e. holder of the numerical orbitals
    tol : tolerance to exclude the linear combinations
  Returns:
    for each specie returns a set of radial functions defining a product basis
    These functions are sufficient to represent the products of original atomic orbitals
    via a product vertex coefficients.
  Examples:
    
  '''
  def __init__(self, ao_log, tol=1e-10):
    
    self.ao_log = ao_log
    self.tol = tol
    self.rr,self.pp,self.nr = ao_log.rr,ao_log.pp,ao_log.nr
    self.interp_rr = ao_log.interp_rr
    self.sp2nmult = np.zeros((ao_log.nspecies), dtype='int64')
    self.nmultmax = max(self.sp2nmult)
    
    lvc = local_vertex_c(ao_log) # constructor of local vertices
    self.psi_log    = [] # radial orbitals: list of arrays
    self.psi_log_rl = [] # radial orbitals times r**j: list of arrays
    self.sp_mu2rcut = [] # list of numpy arrays containing the maximal radii
    self.sp_mu2j    = [] # list of numpy arrays containing the angular momentum of the radial function
    self.sp_mu2s    = [] # list of numpy arrays containing the starting index for each radial multiplett
    self.sp2vertex  = [] # list of numpy arrays containing the vertex coefficients
    self.sp2norbs   = [] # number of orbitals per specie
    self.sp2charge  = ao_log.sp2charge # copy of nuclear charges from atomic orbitals
    
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
      self.sp2norbs.append(mu2s[-1])

      mu2ff = np.zeros((nmult, lvc.nr), dtype='float64')
      for mu,[j,domi] in enumerate(mu2jd): mu2ff[mu,:] = ldp['j2xff'][j][domi,:]
      self.psi_log.append(mu2ff)
      
      mu2ff = np.zeros((nmult, lvc.nr), dtype='float64')
      for mu,[j,domi] in enumerate(mu2jd): mu2ff[mu,:] = ldp['j2xff'][j][domi,:]/lvc.rr**j
      self.psi_log_rl.append(mu2ff)
       
      no,npf= lvc.sp2norbs[sp], sum(2*mu2j+1)  # count number of orbitals and product functions
      mu2ww = np.zeros((npf,no,no), dtype='float64')
      for [j,domi],s in zip(mu2jd,mu2s): mu2ww[s:s+2*j+1,:,:] = ldp['j2xww'][j][domi,0:2*j+1,:,:]

      self.sp2vertex.append(mu2ww)

    self.jmx = np.amax(np.array( [max(mu2j) for mu2j in self.sp_mu2j], dtype='int32'))
