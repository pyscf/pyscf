# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
import numpy as np
from pyscf.nao import log_mesh_c
from pyscf.nao.m_libnao import libnao
from ctypes import POINTER, c_double, c_int, byref

# phia,la,ra,phib,lb,rb,rcen,lbdmxa,rhotb,rr,nr,jtb,clbdtb,lbdtb,nterm,ord,pcs,rho_min_jt,dr_jt

"""
 Reduction of the products of two atomic orbitals placed at some distance
 [1] Talman JD. Multipole Expansions for Numerical Orbital Products, Int. J. Quant. Chem. 107, 1578--1584 (2007)
 ngl : order of Gauss-Legendre quadrature
"""

libnao.prdred.argtypes = (
  POINTER(c_double), # phia(nr)
  POINTER(c_int),    # la
  POINTER(c_double), # ra(3)
  POINTER(c_double), # phib(nr)
  POINTER(c_int),    # lb
  POINTER(c_double), # rb(3)
  POINTER(c_double), # rcen(3)
  POINTER(c_int),    # lbdmxa
  POINTER(c_double), # rhotb(nr,nterm)
  POINTER(c_double), # rr(nr)
  POINTER(c_int),    # nr
  POINTER(c_int),    # jtb(nterm)
  POINTER(c_int),    # clbdtb(nterm)
  POINTER(c_int),    # lbdtb(nterm)
  POINTER(c_int),    # nterm
  POINTER(c_int),    # ord
  POINTER(c_int),    # pcs
  POINTER(c_double), # rho_min_jt
  POINTER(c_double)) # dr_jt

#
#
#
class prod_talman_c(log_mesh_c):
  
  def __init__(self, log_mesh, jmx=7, ngl=96, lbdmx=14):
    """
    Expansion of the products of two atomic orbitals placed at given locations and around a center between these locations 
    [1] Talman JD. Multipole Expansions for Numerical Orbital Products, Int. J. Quant. Chem. 107, 1578--1584 (2007)
      ngl : order of Gauss-Legendre quadrature
      log_mesh : instance of log_mesh_c defining the logarithmic mesh (rr and pp arrays)
      jmx : maximal angular momentum quantum number of each atomic orbital in a product
      lbdmx : maximal angular momentum quantum number used for the expansion of the product phia*phib
    """
    from numpy.polynomial.legendre import leggauss
    from pyscf.nao.m_log_interp import log_interp_c
    from pyscf.nao.m_csphar_talman_libnao import csphar_talman_libnao as csphar_jt
    assert ngl>2 
    assert jmx>-1
    assert hasattr(log_mesh, 'rr') 
    assert hasattr(log_mesh, 'pp')
    
    self.ngl = ngl
    self.lbdmx = lbdmx
    self.xx,self.ww = leggauss(ngl)
    log_mesh_c.__init__(self)
    self.init_log_mesh(log_mesh.rr, log_mesh.pp)

    self.plval=np.zeros([2*(self.lbdmx+jmx+1), ngl])
    self.plval[0,:] = 1.0
    self.plval[1,:] = self.xx
    for kappa in range(1,2*(self.lbdmx+jmx)+1):
      self.plval[kappa+1, :]= ((2*kappa+1)*self.xx*self.plval[kappa, :]-kappa*self.plval[kappa-1, :])/(kappa+1)

    self.log_interp = log_interp_c(self.rr)
    self.ylm_cr = csphar_jt([0.0,0.0,1.0], self.lbdmx+2*jmx)

    return

  def prdred(self,phia,la,ra, phib,lb,rb,rcen):
    """ Reduce two atomic orbitals given by their radial functions phia, phib,  
    angular momentum quantum numbers la, lb and their centers ra,rb.
    The expansion is done around a center rcen."""
    from numpy import sqrt
    from pyscf.nao.m_thrj import thrj
    from pyscf.nao.m_fact import fact as fac, sgn

    assert la>-1 
    assert lb>-1 
    assert len(rcen)==3 
    assert len(ra)==3 
    assert len(rb)==3
    
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
      for clbd in range(self.lbdmx+1):
          nterm=nterm+ (clbd+ij+1 - abs(clbd-ij))

    jtb = np.zeros(nterm, dtype=np.int32)
    clbdtb = np.zeros(nterm, dtype=np.int32)
    lbdtb = np.zeros(nterm, dtype=np.int32)
    
    ix=-1
    for ij in range(abs(la-lb),ijmx+1):
      for clbd in range(self.lbdmx+1):
        for lbd in range(abs(clbd-ij),clbd+ij+1):
          ix=ix+1
          jtb[ix]=ij
          clbdtb[ix]=clbd
          lbdtb[ix]=lbd
          
    return jtb,clbdtb,lbdtb
  
  def prdred_libnao(self,phia,la,ra, phib,lb,rb,rcen):
    """ By calling a subroutine  """
    assert len(phia)==self.nr
    assert len(phib)==self.nr
    
    jtb,clbdtb,lbdtb=self.prdred_terms(la,lb)
    nterm     = len(jtb)
    
    jtb_cp    = np.require(jtb,  dtype=c_int, requirements='C')
    clbdtb_cp = np.require(clbdtb, dtype=c_int, requirements='C')
    lbdtb_cp  = np.require(lbdtb,  dtype=c_int, requirements='C')
    rhotb_cp  = np.require( np.zeros([nterm, self.nr]), dtype=c_double, requirements='CW')
    rr_cp     = np.require(self.rr,dtype=c_double, requirements='C')
    phia_cp   = np.require(phia,dtype=c_double, requirements='C')
    phib_cp   = np.require(phib,dtype=c_double, requirements='C')
    ra_cp     = np.require(ra,dtype=c_double, requirements='C')
    rb_cp     = np.require(rb,dtype=c_double, requirements='C')
    rcen_cp   = np.require(rcen,dtype=c_double, requirements='C')
    
    libnao.prdred(phia_cp.ctypes.data_as(POINTER(c_double)), c_int(la), ra_cp.ctypes.data_as(POINTER(c_double)),
                  phib_cp.ctypes.data_as(POINTER(c_double)), c_int(lb), rb_cp.ctypes.data_as(POINTER(c_double)),
                  rcen_cp.ctypes.data_as(POINTER(c_double)), 
                  c_int(self.lbdmx),
                  rhotb_cp.ctypes.data_as(POINTER(c_double)),
                  rr_cp.ctypes.data_as(POINTER(c_double)),
                  c_int(self.nr),
                  jtb_cp.ctypes.data_as(POINTER(c_int)),
                  clbdtb_cp.ctypes.data_as(POINTER(c_int)),
                  lbdtb_cp.ctypes.data_as(POINTER(c_int)),
                  c_int(nterm),
                  c_int(self.ngl),
                  c_int(1),
                  c_double(self.log_interp.gammin_jt),
                  c_double(self.log_interp.dg_jt) )
    rhotb = rhotb_cp
    return jtb,clbdtb,lbdtb,rhotb

  
  def prdred_further(self, ja,ma,jb,mb,rcen,jtb,clbdtb,lbdtb,rhotb):
    """ Evaluate the Talman's expansion at given Cartesian coordinates"""
    from pyscf.nao.m_thrj import thrj
    from pyscf.nao.m_fact import sgn
    from pyscf.nao.m_csphar_talman_libnao import csphar_talman_libnao as csphar_jt
    from numpy import zeros, sqrt, pi, array
    
    assert all(rcen == zeros(3)) # this works only when center is at the origin
    nterm = len(jtb)
    assert nterm == len(clbdtb)
    assert nterm == len(lbdtb)
    assert nterm == rhotb.shape[0]
    assert self.nr == rhotb.shape[1]

    ffr = zeros([self.lbdmx+1,self.nr], np.complex128)
    m = mb + ma
    ylm_cr = csphar_jt([0.0,0.0,1.0], lbdtb.max())
    for j,clbd,lbd,rho in zip(jtb,clbdtb,lbdtb,rhotb):
      ffr[clbd,:]=ffr[clbd,:] + thrj(ja,jb,j,ma,mb,-m)*thrj(j,clbd,lbd,-m,m,0)*rho*ylm_cr[lbd*(lbd+1)]
    return ffr,m

  def prdred_further_scalar(self, ja,ma,jb,mb,rcen,jtb,clbdtb,lbdtb,rhotb):
    """ Evaluate the Talman's expansion at given Cartesian coordinates"""
    from pyscf.nao.m_thrj import thrj
    from pyscf.nao.m_csphar_talman_libnao import csphar_talman_libnao as csphar_jt
    from numpy import zeros, sqrt, pi, array
    
    assert all(rcen == zeros(3)) # this works only when center is at the origin
    nterm = len(jtb)
    assert nterm == len(clbdtb)
    assert nterm == len(lbdtb)
    assert nterm == len(rhotb)

    ffr = zeros([self.lbdmx+1], np.complex128)
    m = mb + ma
    for j,clbd,lbd,rho in zip(jtb,clbdtb,lbdtb,rhotb):
      ffr[clbd]=ffr[clbd] + thrj(ja,jb,j,ma,mb,-m)*thrj(j,clbd,lbd,-m,m,0)*rho*self.ylm_cr[lbd*(lbd+1)]
    return ffr,m

#
#
#
if __name__=='__main__':
  from pyscf.nao import prod_basis_c, system_vars_c
  from pyscf import gto
  import numpy as np
  
