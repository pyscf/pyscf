from __future__ import print_function, division
import numpy as np
from pyscf.nao import log_mesh_c

#
#
#
class prod_talman_c(log_mesh_c):
  
  def __init__(self, log_mesh, jmx=7, ngl=96, lbdmxa=12):
    """
    Reduction of the products of two atomic orbitals placed at some distance
    [1] Talman JD. Multipole Expansions for Numerical Orbital Products, Int. J. Quant. Chem. 107, 1578--1584 (2007)
      ngl : order of Gauss-Legendre quadrature
    """
    from numpy.polynomial.legendre import leggauss
    assert ngl>2
    assert lbdmxa>0
    assert hasattr(log_mesh, 'rr')
    assert hasattr(log_mesh, 'pp')
    
    self.ngl = ngl
    self.lbdmxa = lbdmxa
    self.xx,self.ww = leggauss(ngl)
    log_mesh_c.__init__(self)
    self.init_log_mesh(log_mesh.rr, log_mesh.pp)

    self.plval=np.zeros([2*(lbdmxa+jmx+1), ngl])
    self.plval[0,:] = 1.0
    self.plval[1,:] = self.xx
    for kappa in range(1,2*(lbdmxa+jmx)+1):
      self.plval[kappa+1, :]= ((2*kappa+1)*self.xx*self.plval[kappa, :]-kappa*self.plval[kappa-1, :])/(kappa+1)

    return
    
  def prdred(self,phia,la,ra, phib,lb,rb,rcen):
    assert la>-1
    assert lb>-1
    
    jtb,clbdtb,lbdtb=self.prdred_terms(la,lb)

    ya=phia/self.rr**la
    yb=phib/self.rr**lb

    return 0


  def prdred_terms(self,la,lb):
    """ Compute term-> Lambda,Lambda',j correspondence """
    nterm=0
    ijmx=la+lb
    for ij in range(abs(la-lb),ijmx+1):
      for clbd in range(self.lbdmxa+1):
          nterm=nterm+ (clbd+ij+1 - abs(clbd-ij))

    jtb = np.zeros(nterm, dtype=np.int32)
    clbdtb = np.zeros(nterm, dtype=np.int32)
    lbdtb = np.zeros(nterm, dtype=np.int32)
    
    ix=-1
    for ij in range(abs(la-lb),ijmx+1):
      for clbd in range(self.lbdmxa+1):
        for lbd in range(abs(clbd-ij),clbd+ij+1):
          ix=ix+1
          jtb[ix]=ij
          clbdtb[ix]=clbd
          lbdtb[ix]=lbd
          
    return jtb,clbdtb,lbdtb
    
#
#
#
if __name__=='__main__':
  from pyscf.nao import prod_basis_c, system_vars_c
  from pyscf import gto
  import numpy as np
  
