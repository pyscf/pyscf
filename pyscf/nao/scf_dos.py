from __future__ import print_function, division
import numpy as np
from numpy import zeros_like, zeros 


def eigenvalues2dos(ksn2e, zomegas, nkpoints=1): 
  """ Compute the Density of States using the eigenvalues """
  dos = zeros(len(zomegas))
  for iw,zw in enumerate(zomegas): dos[iw] = (1.0/(zw - ksn2e)).sum().imag
  return -dos/np.pi/nkpoints
  

def scf_dos(scf, zomegas, nkpoints=1):
  """ Compute the Density of States using the eigenvalues """
  return eigenvalues2dos(scf.mo_energy, zomegas, nkpoints)


def scf_pdos(scf, zomegas, nkpoints=1): 
  """ Compute the Partial Density of States (resolved in angular momentum of the orbitals) using the eigenvalues and eigenvectors in wfsx """
  from timeit import default_timer as timer
  
  jmx = scf.ao_log.jmx
  jksn2w = zeros([jmx+1]+list(scf.mo_energy.shape))
  over = scf.hsx.s4_csr.toarray()
  
  orb2j = scf.get_orb2j()
  for j in range(jmx+1):
    mask = (orb2j==j)    
    for k in range(scf.nkpoints):
      for s in range(scf.nspin):
        for n in range(scf.norbs):
          jksn2w[j,k,s,n] = np.dot( np.dot(mask*scf.mo_coeff[k,s,n,:,0], over), scf.mo_coeff[k,s,n,:,0])
  
  pdos = zeros((jmx+1,len(zomegas)))
  for j in range(jmx+1):
    for iw,zw in enumerate(zomegas):
      pdos[j,iw] = (jksn2w[j,:,:,:]/(zw - scf.wfsx.ksn2e[:,:,:])).sum().imag
    
  return -pdos/np.pi/nkpoints


def system_vars_ados(sv, zomegas, ls_atom_groups, nkpoints=1): 
  """ Compute a Partial Density of States (resolved in atomic indices) using the eigenvalues and eigenvectors in wfsx """
  from timeit import default_timer as timer
  
  iksn2w = zeros([2]+list(sv.wfsx.ksn2e.shape))
  over = sv.hsx.s4_csr.toarray()
  
  orb2id_group = sv.get_orb2j()
  
  for id_group,group in enumerate(ls_atom_groups):
    mask = (orb2group==id_group)
    for k,kvec in enumerate(sv.wfsx.k2xyz):
      for s in range(sv.nspin):
        for n in range(sv.norbs):
          jksn2w[j,k,s,n] = np.dot( np.dot(mask*sv.wfsx.x[k,s,n,:,0], over), sv.wfsx.x[k,s,n,:,0])
  
  pdos = zeros((jmx+1,len(zomegas)))
  for j in range(jmx+1):
    for iw,zw in enumerate(zomegas):
      pdos[j,iw] = (jksn2w[j,:,:,:]/(zw - sv.wfsx.ksn2e[:,:,:])).sum().imag
    
  return -pdos/np.pi/nkpoints
