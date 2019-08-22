from __future__ import print_function, division
import sys, numpy as np
from copy import copy
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l
from pyscf.nao.m_rf0_den import rf0_den, rf0_cmplx_ref_blk, rf0_cmplx_ref, rf0_cmplx_vertex_dp, rf0_cmplx_vertex_ac
from pyscf.nao.m_rf_den import rf_den
from pyscf.nao.m_rf_den_pyscf import rf_den_pyscf
from pyscf.data.nist import HARTREE2EV
from pyscf.nao.m_valence import get_str_fin
from timeit import default_timer as timer
from numpy import stack, dot, zeros, einsum, pi, log, array, require
from pyscf.nao import scf, gw
import time


start_time = time.time()
class gw_iter(gw):
  """ Iterative G0W0 with integration along imaginary axis """

  def __init__(self, **kw):
    gw.__init__(self, **kw)



  def si_c_check (self, tol = 1e-5):
    """
    This compares np.solve and lgmres methods for solving linear equation (1-v\chi_{0}) * W_c = v\chi_{0}v
    """
    import time
    import numpy as np
    from pyscf.nao import gw
    ww = 1j*self.ww_ia
    t = time.time()
    si0_1 = self.gw.si_c(ww)              #method 1:  numpy.linalg.solve
    t1 = time.time() - t
    print('numpy: {} sec'.format(t1))
    t2 = time.time()
    si0_2 = self.si_c_lgmres(ww)       #method 2:  scipy.sparse.linalg.lgmres
    t3 = time.time() - t2
    print('lgmres: {} sec'.format(t3))
    summ = abs(si0_1 + si0_2).sum()
    diff = abs(si0_1 - si0_2).sum() 
    if diff/summ < tol and diff/si0_1.size < tol:
       print('OK! scipy.lgmres methods and np.linalg.solve have identical results')
    else:
       print('Results (W_c) are NOT similar!')     
    return [[diff/summ] , [np.amax(abs(diff))] ,[tol]]



  def snmw2sf_test (self):
    """
     For XVX instead of last precedure: multiplications were done by reshaping matrices in 2D shape in XVX.
    """
    import numpy as np    
    from scipy.sparse.linalg import LinearOperator    
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import lgmres
    from numpy.linalg import solve
    ww = 1j*self.ww_ia
    rf0 = self.rf0(ww)
    
    v_pab = self.pb.get_ac_vertex_array()       #atom-centered product basis: V_{\mu}^{ab}
    v_pab1= v_pab.reshape(self.nprod*self.norbs, self.norbs)          #2D shape of atom-centered product

    v_pd  = self.pb.get_dp_vertex_array()       #dominant product basis: V_{\widetilde{\mu}}^{ab}
    v_pd1 = v_pd.reshape(v_pd.shape[0]*self.norbs, self.norbs)      #2D shape of dominant produc

    c = self.pb.get_da2cc_den()                 #atom_centered functional: C_{\widetilde{\mu}}^{\mu}
                                                #V_{\mu}^{ab}= V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu}  

    #tip
    #v1 = v_pab.T.reshape(self.norbs,-1)                     #reshapes v_pab (norb, norb*nprod), decrease 3d to 2d-matrix
    #v2 = v1.reshape(self.norbs,self.norbs,self.nprod).T     #reshape to initial shape, so v2 is again v_pab=(norb, norb, nprod)
 
    snm2i = []
    for s in range(self.nspin):
        sf_aux = np.zeros((len(self.nn[s]), self.norbs, self.nprod), dtype=self.dtypeComplex)  
        inm = np.zeros((len(self.nn[s]), self.norbs, len(ww)), dtype=self.dtypeComplex)
        
        xna = self.mo_coeff[0,s,self.nn[s],:,0]             #(nstat,norbs)
        xmb = self.mo_coeff[0,s,:,:,0]                      #(nstat,norbs)
        xvx_ref  = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb)  #einsum: direct multiplication 
        xvx_ref2 = np.swapaxes(np.dot(xna, np.dot(v_pab,xmb.T)),1,2)                                      #direct multiplication by using np.dot and swapping between axis
        print('comparison between einsum and dot: ',np.allclose(xvx_ref,xvx_ref2,atol=1e-15))             #einsum=dot

        #atom-centered product basis
        #First step
        vx  = np.dot(v_pab1,xmb.T)                               #multiplications were done one by one in 2D shape
        vx  = vx.reshape(self.nprod,self.norbs, self.norbs) #reshape it into initial 3D shape
        #Second step
        xvx1 = np.swapaxes(vx,0,1)
        xvx1 = xvx1.reshape(self.norbs,-1)
        xvx1 = np.dot(xna,xvx1)
        xvx1 = xvx1.reshape(len(self.nn[s]),self.nprod,self.norbs)
        xvx1 = np.swapaxes(xvx1,1,2)
        print('comparison between ac is directly used and ref: ',np.allclose(xvx1,xvx_ref,atol=1e-15))                #its result is equal to the direct np.dot


        #dominant product basis
        #First step
        size = self.cc_da.shape[0]
        vxdp  = np.dot(v_pd1,xmb.T)
        vxdp  = vxdp.reshape(size,self.norbs, self.norbs)
        #Second step
        xvx2 = np.swapaxes(vxdp,0,1)
        xvx2 = xvx2.reshape(self.norbs,-1)
        xvx2 = np.dot(xna,xvx2)
        xvx2 = xvx2.reshape(len(self.nn[s]),size,self.norbs)
        xvx2 = np.swapaxes(xvx2,1,2)
        xvx2 = np.dot(xvx2,c)
        print('comparison between dp directly used and ref: ',np.allclose(xvx2,xvx_ref,atol=1e-15))


        #dominant product basis in COO-format
        #First step
        data = v_pd.reshape(-1)
        i0,i1,i2 = np.mgrid[0:v_pd.shape[0],0:v_pd.shape[1],0:v_pd.shape[2] ].reshape((3,data.size))
        from pyscf.nao import ndcoo
        nc = ndcoo((data, (i0, i1, i2)))
        m0 = nc.tocoo_pa_b('p,a,b->ap,b')
        vx1 = m0*(xmb.T)
        vx1 = vx1.reshape(size,self.norbs,self.norbs)  #shape (p,a,b)
        vx1 = vx1.reshape(self.norbs,-1)             #shape(a,p*b)  
        #Second Step
        xvx3 = np.dot(xna,vx1)                            #xna(ns,a).V(a,p*b)=xvx(ns,p*b)
        xvx3 = xvx3.reshape(len(self.nn[s]),size,self.norbs)     #xvx(ns,p,b)
        xvx3 = np.swapaxes(xvx3,1,2)                         #xvx(ns,b,p)
        xvx3 = np.dot(xvx3,c)                                #XVX=xvx.c
        print('comparison between Sparse_dp by using ndcoo clase and ref: ',np.allclose(xvx3, xvx_ref, atol=1e-15))
  
        for iw,w in enumerate(ww):                              #iw is number of grid and w is complex plane                                
            k_c = np.dot(self.kernel_sq, rf0[iw,:,:])           #v\chi_{0}
            self.comega_current = w                             #appropriate ferequency for self.vext2veff_matvec
            k_c_opt = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=self.dtypeComplex)    #convert k_c as full matrix into Operator
            for n in range(len(self.nn[s])):    
                for m in range(self.norbs):
                    a = np.dot(self.kernel_sq, xvx3[n,m,:])     #v XVX
                    b = self.apply_rf0(a,self.comega_current)   #\chi_{0}v XVX by using matrix vector 
                    a = np.dot(self.kernel_sq, b)               #v\chi_{0}v XVX, this should be aquals to bxvx in last approach
                    sf_aux[n,m,:] ,exitCode = lgmres(k_c_opt, a, tol=1e-06)
            if exitCode != 0: print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
            inm[:,:,iw]=np.einsum('nmp,nmp->nm',xvx3, sf_aux)   #I= XVX I_aux
        snm2i.append(np.real(inm))
    return snm2i



  def get_snmw2sf_iter(self):
    from scipy.sparse import csr_matrix, coo_matrix
    from scipy.linalg import blas
    from scipy.sparse.linalg import LinearOperator, lgmres    
    from scipy.sparse import csc_matrix

    ww = 1j*self.ww_ia
        
    v_pab = self.pb.get_ac_vertex_array()       #atom-centered product basis: V_{\mu}^{ab}
    v_pab1= v_pab.reshape(self.nprod*self.norbs, self.norbs)          #2D shape of atom-centered product

    v_pd  = self.pb.get_dp_vertex_array()       #dominant product basis: V_{\widetilde{\mu}}^{ab}
    v_pd1 = v_pd.reshape(v_pd.shape[0]*self.norbs, self.norbs)      #2D shape of dominant produc

    c = self.pb.get_da2cc_den()                 #atom_centered functional: C_{\widetilde{\mu}}^{\mu}
                                                #V_{\mu}^{ab}= V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu}  

    snm2i = []
    for s in range(self.nspin):
        sf_aux = np.zeros((len(self.nn[s]), self.norbs, self.nprod), dtype=self.dtypeComplex)  
        inm = np.zeros((len(self.nn[s]), self.norbs, len(ww)), dtype=self.dtypeComplex)
        
        xna = self.mo_coeff[0,s,self.nn[s],:,0]             #(nstat,norbs)
        xmb = self.mo_coeff[0,s,:,:,0]                      #(nstat,norbs)


        #dominant product basis in COO-format
        #First step
        size = self.cc_da.shape[0]
        data = v_pd.reshape(-1)
        i0,i1,i2 = np.mgrid[0:v_pd.shape[0],0:v_pd.shape[1],0:v_pd.shape[2] ].reshape((3,data.size))
        from pyscf.nao import ndcoo
        nc = ndcoo((data, (i0, i1, i2)))
        m0 = nc.tocoo_pa_b('p,a,b->ap,b')
        vx1 = m0*(xmb.T)
        vx1 = vx1.reshape(size,self.norbs,self.norbs)  #shape (p,a,b)
        vx1 = vx1.reshape(self.norbs,-1)             #shape(a,p*b)  
        #Second Step
        xvx3 = np.dot(xna,vx1)                            #xna(ns,a).V(a,p*b)=xvx(ns,p*b)
        xvx3 = xvx3.reshape(len(self.nn[s]),size,self.norbs)     #xvx(ns,p,b)
        xvx3 = np.swapaxes(xvx3,1,2)                         #xvx(ns,b,p)
        xvx3 = np.dot(xvx3,c)                                #XVX=xvx.c
        xvx_ref  = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb)
        #print('comparison between Sparse_dp by using ndcoo clase and ref: ',np.allclose(xvx3, xvx_ref, atol=1e-15))
        
        for iw,w in enumerate(ww):                              #iw is number of grid and w is complex plane                                
            self.comega_current = w                            
            k_c_opt = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec, dtype=self.dtypeComplex)    #convert k_c as full matrix into Operator
            #print('k_c_opt',k_c_opt.shape)
            for n in range(len(self.nn[s])):    
                for m in range(self.norbs):
                    a = np.dot(self.kernel_sq, xvx3[n,m,:])     #v XVX
                    #print('first aaaaaaa',a.shape)
                    b = self.gw_applyrf0(a,self.comega_current)   #\chi_{0}v XVX by using matrix vector 
                    b = np.split(b, self.nspin)
                    #print('bbbbbbb',len(b),len(b[s]))
                    a = np.dot(self.kernel_sq, b[s])               #v\chi_{0}v XVX, this should be aquals to bxvx in last approach
                    #print('second aaaaaaa',a.shape)
                    sf_aux[n,m,:] ,exitCode = lgmres(k_c_opt, a, tol=1e-06)
                    #print('done')
            if exitCode != 0: print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
            inm[:,:,iw]=np.einsum('nmp,nmp->nm',xvx3, sf_aux)   #I= XVX I_aux
        snm2i.append(np.real(inm))
    return snm2i



  def gw_vext2veffmatvec(self,vin):
    dn0 = self.gw_applyrf0(vin, self.comega_current)
    #print('dnnnnnnnnnnnn',dn0.shape)
    vcre,vcim = self.gw_applykernelspin(dn0)
    #print('-'*40)
    return vin - (vcre + 1.0j*vcim)


  def gw_applyrf0(self,sp2v, comega=1j*0.0):
    if (self.nspin==2): sp2v = np.concatenate((sp2v, sp2v), axis=None)
    return self.gw_chi0_mv(sp2v, comega)


  def gw_applykernelspin(self,dn):
    dn = np.split(dn, self.nspin)    
    if self.nspin==1:
      return self.gw_applykernel_nspin1(dn[0])
    elif self.nspin==2:
      return self.gw_applykernel_nspin1(dn[0])


  def gw_applykernel_nspin1(self,dn):
    
    daux  = np.zeros(self.nprod, dtype=self.dtype)
    daux[:] = np.require(dn.real, dtype=self.dtype, requirements=["A","O"])
    vcre = self.spmv(self.nprod, 1.0, self.kernel, daux)
    
    daux[:] = np.require(dn.imag, dtype=self.dtype, requirements=["A","O"])
    vcim = self.spmv(self.nprod, 1.0, self.kernel, daux)
    return vcre,vcim


  def gw_applykernel_nspin2(self,dn):

    vcre = np.zeros((2,self.nspin,self.nprod), dtype=self.dtype)
    daux = np.zeros((self.nprod), dtype=self.dtype)
    s2dn = dn.reshape((self.nspin,self.nprod))

    for s in range(self.nspin):
      for t in range(self.nspin):
        for ireim,sreim in enumerate(('real', 'imag')):
          daux[:] = np.require(getattr(s2dn[t], sreim), dtype=self.dtype, requirements=["A","O"])
          vcre[ireim,s] += self.spmv(self.nprod, 1.0, self.ss2kernel[s][t], daux)
    #print('*'*150,vcre,vcre.shape)
    return vcre[0,0].reshape(-1),vcre[1,0].reshape(-1)


  def gw_chi0_mv(self, dvin, comega=1j*0.0, dnout=None):
    from scipy.sparse import csr_matrix, coo_matrix
    from scipy.linalg import blas
    from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec, csc_matvecs
    import math
   
    if dnout is None: dnout = np.zeros_like(dvin, dtype=self.dtypeComplex)
    
    sp2v  = dvin.reshape((self.nspin,self.nprod))
    #print('sp2v', sp2v,sp2v.shape)
    sp2dn = dnout.reshape((self.nspin,self.nprod))
    
    for s in range(self.nspin):
      vdp = csr_matvec(self.cc_da, sp2v[s].real)  # real part
      sab = (vdp*self.v_dab).reshape((self.norbs,self.norbs))
    
      nb2v = self.gemm(1.0, self.xocc[s], sab)
      nm2v_re = self.gemm(1.0, nb2v, self.xvrt[s].T)
    
      vdp = csr_matvec(self.cc_da, sp2v[s].imag)  # imaginary
      sab = (vdp*self.v_dab).reshape((self.norbs, self.norbs))
      
      nb2v = self.gemm(1.0, self.xocc[s], sab)
      nm2v_im = self.gemm(1.0, nb2v, self.xvrt[s].T)

      vs,nf = self.vstart[s],self.nfermi[s]
    
      if self.use_numba:
        self.div_numba(self.ksn2e[0,s], self.ksn2f[0,s], nf, vs, comega, nm2v_re, nm2v_im)
      else:
        for n,(en,fn) in enumerate(zip(self.ksn2e[0,s,:nf], self.ksn2f[0,s,:nf])):
          for m,(em,fm) in enumerate(zip(self.ksn2e[0,s,vs:],self.ksn2f[0,s,vs:])):
            nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
            nm2v = nm2v * (fn - fm) * \
              ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
            nm2v_re[n, m] = nm2v.real
            nm2v_im[n, m] = nm2v.imag

        for n in range(vs+1,nf): #padding m<n i.e. negative occupations' difference
          for m in range(n-vs):  nm2v_re[n,m],nm2v_im[n,m] = 0.0,0.0

      nb2v = self.gemm(1.0, nm2v_re, self.xvrt[s]) # real part
      ab2v = self.gemm(1.0, self.xocc[s].T, nb2v).reshape(self.norbs*self.norbs)
      vdp = csr_matvec(self.v_dab, ab2v)
      chi0_re = vdp*self.cc_da

      nb2v = self.gemm(1.0, nm2v_im, self.xvrt[s]) # imag part
      ab2v = self.gemm(1.0, self.xocc[s].T, nb2v).reshape(self.norbs*self.norbs)
      vdp = csr_matvec(self.v_dab, ab2v)    
      chi0_im = vdp*self.cc_da
      
      sp2dn[s] = chi0_re + 1.0j*chi0_im
    #print('dnout', dnout.size)
    return dnout



  def check_veff(self):
    """ This computes an effective field (scalar potential) given the external scalar potential as follows:
        (1-v\chi_{0})V_{eff}=V_{ext}=X_{a}^{n}V_{\mu}^{ab}X_{b}^{m} * v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
        returns V_{eff} as list for all n states(self.nn[s]).
    """
    import numpy as np
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import lgmres
    from numpy.linalg import solve
    ww = 1j*self.ww_ia
    rf0 = self.rf0(ww)
    v_pab = self.pb.get_ac_vertex_array()                   #V_{\mu}^{ab}
    for s in range(self.nspin):
      v_eff = np.zeros((len(self.nn[s]), self.nprod), dtype=self.dtype)
      v_eff_ref = np.zeros((len(self.nn[s]), self.nprod), dtype=self.dtype)
      xna = self.mo_coeff[0,s,self.nn[s],:,0]               #X_{a}^{n}
      xmb = self.mo_coeff[0,s,:,:,0]                        #X_{b}^{m}
      xvx = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb)    #X_{a}^{n}V_{\mu}^{ab}X_{b}^{m}
      for iw,w in enumerate(ww):     
          k_c = np.dot(self.kernel_sq, rf0[iw,:,:])         # v\chi_{0} 
          b = np.dot(k_c, self.kernel_sq)                   # v\chi_{0}v 
          k_c = np.eye(self.nprod)-k_c                      #(1-v\chi_{0})
          bxvx = np.einsum('pq,nmq->nmp', b, xvx)           #v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
          xvxbxvx = np.einsum ('nmp,nlp->np',xvx,bxvx)      #V_{ext}=X_{a}^{n}V_{\mu}^{ab}X_{b}^{m} * v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
            
          for n in range (len(self.nn[s])):
              v_eff_ref[n,:] = self.comp_veff(xvxbxvx[n,:]) #compute v_eff in tddft_iter class as referance
              v_eff[n,:]=solve(k_c, xvxbxvx[n,:])           #linear eq. for finding V_{eff} --> (1-v\chi_{0})V_{eff}=V_{ext}
    if np.allclose(v_eff,v_eff_ref,atol=1e-4)== True:       #compares both V_{eff}
      return v_eff

  def gw_corr_int_iter(self, sn2w, eps=None):
    """ This computes an integral part of the GW correction at energies sn2e[spin,len(self.nn)] """
    if not hasattr(self, 'snmw2sf'): self.snmw2sf = self.get_snmw2sf_iter()

    sn2int = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]
    eps = self.dw_excl if eps is None else eps
    #print(__name__, 'self.dw_ia', self.dw_ia, sn2w)
    for s,ww in enumerate(sn2w):
      for n,w in enumerate(ww):
        #print(__name__, 's,n,w int corr', s,n,w)
        for m in range(self.norbs):
          if abs(w-self.ksn2e[0,s,m])<eps : continue
          state_corr = ((self.dw_ia*self.snmw2sf[s][n,m,:] / (w + 1j*self.ww_ia-self.ksn2e[0,s,m])).sum()/pi).real
          #print(n, m, -state_corr, w-self.ksn2e[0,s,m])
          sn2int[s][n] -= state_corr
    return sn2int


  def gw_corr_res_iter(self, sn2w):
    """This computes a residue part of the GW correction at energies sn2w[spin,len(self.nn)]"""
    v_pab = self.pb.get_ac_vertex_array()
    sn2res = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]
    for s,ww in enumerate(sn2w):
      x = self.mo_coeff[0,s,:,:,0]
      for nl,(n,w) in enumerate(zip(self.nn[s],ww)):
        lsos = self.lsofs_inside_contour(self.ksn2e[0,s,:],w,self.dw_excl)
        zww = array([pole[0] for pole in lsos])
        #print(__name__, s,n,w, 'len lsos', len(lsos))
        si_ww = self.si_c(ww=zww)
        xv = dot(v_pab,x[n])
        for pole,si in zip(lsos, si_ww.real):
          xvx = dot(xv, x[pole[1]])
          contr = dot(xvx, dot(si, xvx))
          #print(pole[0], pole[2], contr)
          sn2res[s][nl] += pole[2]*contr
    return sn2res

  def g0w0_eigvals_iter(self):
    """ This computes the G0W0 corrections to the eigenvalues """
    sn2eval_gw = [np.copy(self.ksn2e[0,s,nn]) for s,nn in enumerate(self.nn) ]  #self.ksn2e = self.mo_energy
    sn2eval_gw_prev = copy(sn2eval_gw)

    self.nn_conv = []           # self.nn_conv -- list of states to converge, spin-resolved.
    for nocc_0t,nocc_conv,nvrt_conv in zip(self.nocc_0t, self.nocc_conv, self.nvrt_conv):
      self.nn_conv.append( range(max(nocc_0t-nocc_conv,0), min(nocc_0t+nvrt_conv,self.norbs)))

    # iterations to converge the 
    if self.verbosity>0: 
        print('-'*48,'|  G0W0 corrections of eigenvalues  |','-'*48+'\n')
        print('MAXIMUM number of iterations (Input file): {} and number of grid points: {}'.format(self.niter_max_ev,self.nff_ia))
        print('Number of GW correction at energies calculated by integral part: {} and by sigma part: {}\n'.format(np.size(self.gw_corr_int_iter(sn2eval_gw)), np.size(self.gw_corr_int_iter(sn2eval_gw))))
        print('GW corection for eigenvalues STARTED:\n')    
    for i in range(self.niter_max_ev):
      sn2i = self.gw_corr_int_iter(sn2eval_gw)
      sn2r = self.gw_corr_res_iter(sn2eval_gw)
      
      sn2eval_gw = []
      for s,(evhf,n2i,n2r,nn) in enumerate(zip(self.h0_vh_x_expval,sn2i,sn2r,self.nn)):
        sn2eval_gw.append(evhf[nn]+n2i+n2r)
        
      sn2mismatch = zeros((self.nspin,self.norbs))
      for s, nn in enumerate(self.nn): sn2mismatch[s,nn] = sn2eval_gw[s][:]-sn2eval_gw_prev[s][:]
      sn2eval_gw_prev = copy(sn2eval_gw)
      err = 0.0
      for s,nn_conv in enumerate(self.nn_conv): err += abs(sn2mismatch[s,nn_conv]).sum()/len(nn_conv)

      if self.verbosity>0:
        np.set_printoptions(linewidth=1000, suppress=True, precision=5)
        print('Iteration #{}  Relative Error: {:.7f}'.format(i+1, err))
      if self.verbosity>1:
        #print(sn2mismatch)
        for s,n2ev in enumerate(sn2eval_gw):
          print('Spin{}: {}'.format(s+1, n2ev[:]*HARTREE2EV)) #, sn2i[s][:]*HARTREE2EV, sn2r[s][:]*HARTREE2EV))   
      if err<self.tol_ev : 
        if self.verbosity>0: print('-'*43,' |  Convergence has been reached at iteration#{}  | '.format(i+1),'-'*43,'\n')
        break
      if err>=self.tol_ev and i+1==self.niter_max_ev:
        if self.verbosity>0: print('-'*30,' |  TAKE CARE! Convergence to tolerance {} not achieved after {}-iterations  | '.format(self.tol_ev,self.niter_max_ev),'-'*30,'\n')
    return sn2eval_gw



  def make_mo_g0w0_iter(self):
    """ This creates the fields mo_energy_g0w0, and mo_coeff_g0w0 """

    self.h0_vh_x_expval = self.get_h0_vh_x_expval()
    if self.verbosity>2:
      if self.nspin==1:
        print(__name__,'\t\t====> Expectation values of Hartree-Fock Hamiltonian (eV):\n %3s  %16s'%('no.','<H>'))
        for i, ab in enumerate(zip(self.h0_vh_x_expval[0].T*HARTREE2EV)):
            print (' %3d  %16.6f'%(i,ab[0]))
      if self.nspin==2:
        print(__name__,'\t\t====> Expectation values of Hartree-Fock Hamiltonian (eV):\n %3s  %16s  | %12s'%('no.','<H_up>','<H_dn>'))        
        for i , (ab) in enumerate(zip(self.h0_vh_x_expval[0].T* HARTREE2EV,self.h0_vh_x_expval[1].T* HARTREE2EV)):
	        print(' %3d  %16.6f  | %12.6f'%(i, ab[0],ab[1]))
      print('mean-field Total energy   (eV):%16.6f'%(self.mf.e_tot*HARTREE2EV))
      S = self.spin/2
      S0 = S*(S+1)
      SS = self.mf.spin_square()
      print('<S^2> and  2S+1               :%16.7f %16.7f'%(SS[0],SS[1]))
      print('Instead of                    :%16.7f %16.7f'%(S0, 2*S+1))
      elapsed_time = time.time() - start_time
      print('\nRunning time is:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),'\n\n') 
    if not hasattr(self,'sn2eval_gw'): self.sn2eval_gw=self.g0w0_eigvals_iter() # Comp. GW-corrections
    
    # Update mo_energy_gw, mo_coeff_gw after the computation is done
    self.mo_energy_gw = np.copy(self.mo_energy)
    self.mo_coeff_gw = np.copy(self.mo_coeff)
    #print(len(self.sn2eval_gw), type(self.sn2eval_gw))
    #print(self.nn, type(self.nn))
    #print(self.mo_energy_gw.shape, type(self.mo_energy_gw))
    self.argsort = []
    for s,nn in enumerate(self.nn):
      self.mo_energy_gw[0,s,nn] = self.sn2eval_gw[s]
      nn_occ = [n for n in nn if n<self.nocc_0t[s]]
      nn_vrt = [n for n in nn if n>=self.nocc_0t[s]]
      scissor_occ = (self.mo_energy_gw[0,s,nn_occ] - self.mo_energy[0,s,nn_occ]).sum()/len(nn_occ)
      scissor_vrt = (self.mo_energy_gw[0,s,nn_vrt] - self.mo_energy[0,s,nn_vrt]).sum()/len(nn_vrt)
      #print(scissor_occ, scissor_vrt)
      mm_occ = list(set(range(self.nocc_0t[s]))-set(nn_occ))
      mm_vrt = list(set(range(self.nocc_0t[s],self.norbs)) - set(nn_vrt))
      self.mo_energy_gw[0,s,mm_occ] +=scissor_occ
      self.mo_energy_gw[0,s,mm_vrt] +=scissor_vrt
      #print(self.mo_energy_g0w0)
      argsrt = np.argsort(self.mo_energy_gw[0,s,:])
      self.argsort.append(argsrt)
      if self.verbosity>0: print(__name__, '\t\t====> Spin {}: energy-sorted MO indices: {}'.format(str(s+1),argsrt))
      self.mo_energy_gw[0,s,:] = np.sort(self.mo_energy_gw[0,s,:])
      for n,m in enumerate(argsrt): self.mo_coeff_gw[0,s,n] = self.mo_coeff[0,s,m]
 
    self.xc_code = 'GW'
    if self.verbosity>1:
      print(__name__,'\t\t====> Performed xc_code: {}\n '.format(self.xc_code))
      print('\nConverged GW-corrected eigenvalues:\n',self.mo_energy_gw*HARTREE2EV)
    
    return self.etot_gw()
        
  kernel_gw_iter = make_mo_g0w0_iter



if __name__=='__main__':
    from pyscf import gto, scf
    from pyscf.nao import gw_iter   

    mol = gto.M(atom='''O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978''', basis='ccpvdz',spin=2)
    mf = scf.RHF(mol)
    mf.kernel()

    gw = gw_iter(mf=mf, gto=mol, verbosity=3, niter_max_ev=20, kmat_algo='sm0_sum')
    gwiter = gw.get_snmw2sf_iter()
    gwnon = gw.get_snmw2sf()
    print(np.allclose(gwiter[0],gwnon[0],atol=1e-5)) #compares matrix element of W obtained by gw_iter with gw 
    gw.kernel_gw_iter()
    gw.report()
