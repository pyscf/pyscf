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
    from pyscf.nao.m_log_interp import log_interp_c
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

    self.log_interp = log_interp_c(self.rr)
    return
    
  def prdred(self,phia,la,ra, phib,lb,rb,rcen):
    from numpy import sqrt
    from pyscf.nao.m_thrj import thrj
    from pyscf.nao.m_fact import fact as fac, sgn

    assert la>-1
    assert lb>-1
    
    jtb,clbdtb,lbdtb=self.prdred_terms(la,lb)
    nterm = len(jtb)
    
    ya = phia/self.rr**la
    yb = phib/self.rr**lb
    raa,rbb=sqrt(sum((ra-rcen)**2)),sqrt(sum((rb-rcen)**2))
    ijmx=la+lb
    fval=np.zeros([2*self.lbdmxa+ijmx+1, self.nr])
    yz = np.zeros(self.ngl)
    kpmax = 0
    for ir,r in enumerate(self.rr):
      for igl,x in enumerate(self.xx):
        a1 = sqrt(r*r-2*raa*r*x+raa**2)
        a2 = sqrt(r*r+2*rbb*r*x+rbb**2)
        yz[igl]=self.log_interp(ya,a1)*self.log_interp(yb,a2)

      kpmax = 2*self.lbdmxa+ijmx if raa+rbb>1.0e-5 else 0 
      for kappa in range(kpmax+1):
        fval[kappa,ir]=0.5*(self.plval[kappa,:]*yz*self.ww).sum()

    rhotb=np.zeros([nterm,self.nr])
    for ix,[ij,clbd,clbdp] in enumerate(zip(jtb, clbdtb, lbdtb)):
      for lbd1 in range(la+1):
        lbdp1 = la-lbd1
        aa = thrj(lbd1,lbdp1,la,0,0,0)*fac[lbd1]*fac[lbdp1]*fac[2*la+1] / (fac[2*lbd1]*fac[2*lbdp1]*fac[la])

        for lbd2 in range(lb+1):
          lbdp2=lb-lbd2
          bb=thrj(lbd2,lbdp2,lb,0,0,0)*fac[lbd2]*fac[lbdp2]*fac[2*lb+1] / (fac[2*lbd2]*fac[2*lbdp2]*fac[lb])
          bb=aa*bb
          
          for kappa in range(kpmax+1):
            sumb=0.0
            lcmin=max(abs(lbd1-lbd2),abs(clbd-kappa))
            lcmax=min(lbd1+lbd2,clbd+kappa)
            for lc in range(lcmin,lcmax+1,2):
              lcpmin=max(abs(lbdp1-lbdp2),abs(clbdp-kappa))
              lcpmax=min(lbdp1+lbdp2,clbdp+kappa)
              for lcp in range(lcpmin,lcpmax+1,2):
                if abs(lc-ij)<=lcp and lcp<=lc+ij:
                  sumb = sumb+(2*lc+1)*(2*lcp+1) * \
                    thrj(lbd1,lbd2,lc,0,0,0) * \
                    thrj(lbdp1,lbdp2,lcp,0,0,0) * \
                    thrj(lc,clbd,kappa,0,0,0) * \
                    thrj(lcp,clbdp,kappa,0,0,0) * \
                    sixj(clbd,clbdp,ij,lcp,lc,kappa) * \
                    ninej(la,lb,ij,lbd1,lbd2,lc,lbdp1,lbdp2,lcp)
              
              cc=sgn(lbd1+kappa+lb)*(2*ij+1)*(2*kappa+1) * (2*clbd+1)*(2*clbdp+1)*bb*sumb
              if cc != 0.0:
                lbd1_p_lbd2 = lbd1 + lbd2
                rhotb[ix,:] = rhotb[ix,:] + cc*self.rr[:]**(lbd1_p_lbd2) *(raa**lbdp1)* (rbb**lbdp2)* fval[kappa,:]

    return jtb,clbdtb,lbdtb,rhotb


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
  
