from __future__ import print_function, division
import numpy as np
from numpy import zeros_like, zeros 


def eigenvalues2dos(ksn2e, zomegas, nkpoints=1): 
  """ Compute the Density of States using the eigenvalues """
  dos = zeros(len(zomegas))
  for iw,zw in enumerate(zomegas): dos[iw] = (1.0/(zw - ksn2e)).sum().imag
  return -dos/np.pi/nkpoints
  

def system_vars_dos(sv, zomegas, nkpoints=1): 
  """ Compute the Density of States using the eigenvalues """
  return eigenvalues2dos(sv.wfsx.ksn2e, zomegas, nkpoints)


def system_vars_pdos(sv, zomegas, nkpoints=1): 
  """ Compute the Partial Density of States (resolved in angular momentum of the orbitals) using the eigenvalues and eigenvectors in wfsx """
  from timeit import default_timer as timer
  
  jmx = sv.ao_log.jmx
  jksn2w = zeros([jmx+1]+list(sv.wfsx.ksn2e.shape))
  over = sv.hsx.s4_csr.toarray()
  
  orb2j = sv.get_orb2j()
  for j in range(jmx+1):
    mask = (orb2j==j)    
    for k,kvec in enumerate(sv.wfsx.k2xyz):
      for s in range(sv.nspin):
        for n in range(sv.norbs):
          jksn2w[j,k,s,n] = np.dot( np.dot(mask*sv.wfsx.x[k,s,n,:,0], over), sv.wfsx.x[k,s,n,:,0])
  
  pdos = zeros((jmx+1,len(zomegas)))
  for j in range(jmx+1):
    for iw,zw in enumerate(zomegas):
      pdos[j,iw] = (jksn2w[j,:,:,:]/(zw - sv.wfsx.ksn2e[:,:,:])).sum().imag
    
  return -pdos/np.pi/nkpoints
