from __future__ import print_function, division
import sys, numpy as np
from copy import copy
from pyscf.data.nist import HARTREE2EV
from timeit import default_timer as timer
from numpy import stack, dot, zeros, einsum, pi, log, array, require
from pyscf.nao import scf, gw
import time


start_time = time.time()
class gw_iter(gw):
  """ Iterative G0W0 with integration along imaginary axis """

  def __init__(self, **kw):
    gw.__init__(self, **kw)


  def si_c3(self,ww):
    """This computes the correlation part of the screened interaction using lgmres
       1-vchi_0 is computed by linear opt, i.e. self.vext2veff_matvec, in form of vector
    """
    import numpy as np
    from scipy.sparse.linalg import lgmres
    from scipy.sparse.linalg import LinearOperator
    #ww = 1j*self.ww_ia
    si01 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtype)
    si02 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtype)
    rf0 = self.rf0(ww)
    for iw,w in enumerate(ww):
      self.comega_current = w
      a = np.dot(self.kernel_sq, rf0[iw,:,:]) 
      b = np.dot(a, self.kernel_sq)
      c = np.eye(self.nprod)- a  
      k_c_opt = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec, dtype=self.dtypeComplex)
      k_c_opt2 = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec2, dtype=self.dtypeComplex)  
      aa = np.diag(k_c_opt2.matvec(np.ones(self.nprod)))    #this is equal to a=v*chi0
      aaa=np.diag(k_c_opt.matvec(np.ones(self.nprod)))      #equal to c= 1-v*chi0 atol 1e-05
      aaaa= np.dot(aa, self.kernel_sq)                      #equal to b= v*chi0*v atol 1e-03
      for m in range(self.nprod):
         si01[iw,m,:],exitCode = lgmres(c, b[m,:], atol=1e-05) 
         si02[iw,m,:],exitCode = lgmres(k_c_opt, aaaa[m,:], atol=1e-05)   
      if exitCode != 0: print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))   
    return np.allclose(si01,si02,atol=1e-03)


  def gw_xvx (self, algo=None):
    """
     calculates XVX = X_{a}^{n}V_{\nu}^{ab}X_{b}^{m} using 4-methods
     1- direct multiplication by using np.dot and np.einsum via swapping between axis
     2- using atom-centered product basis
     3- using dominant product basis
     4- using dominant product basis in COO-format
    """
    import numpy as np  
    algol = algo.lower() if algo is not None else 'dp_coo'  
    #tip
    #v1 = v_pab.T.reshape(self.norbs,-1)                     #reshapes v_pab (norb, norb*nprod), decrease 3d to 2d-matrix
    #v2 = v1.reshape(self.norbs,self.norbs,self.nprod).T     #reshape to initial shape, so v2 is again v_pab=(norb, norb, nprod)
    xvx=[]
    for s in range(self.nspin):
        xna = self.mo_coeff[0,s,self.nn[s],:,0]             #(nstat,norbs)
        xmb = self.mo_coeff[0,s,:,:,0]                      #(nstat,norbs)
     
        #1-direct multiplication with np and einsum
        if algol=='simple':
            v_pab = self.pb.get_ac_vertex_array()       #atom-centered product basis: V_{\mu}^{ab}
            xvx_ref  = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb)  #einsum: direct multiplication 
            xvx_ref2 = np.swapaxes(np.dot(xna, np.dot(v_pab,xmb.T)),1,2)                          #direct multiplication by using np.dot and swapping between axis
            #print('comparison between einsum and dot: ',np.allclose(xvx_ref,xvx_ref2,atol=1e-15)) #einsum=dot
            xvx.append(xvx_ref)

        #2-atom-centered product basis
        if algol=='ac':
            v_pab = self.pb.get_ac_vertex_array()       #atom-centered product basis: V_{\mu}^{ab}
            v_pab1= v_pab.reshape(self.nprod*self.norbs, self.norbs)          #2D shape of atom-centered product
            #First step
            vx  = np.dot(v_pab1,xmb.T)                          #multiplications were done one by one in 2D shape
            vx  = vx.reshape(self.nprod,self.norbs, self.norbs) #reshape it into initial 3D shape
            #Second step
            xvx1 = np.swapaxes(vx,0,1)
            xvx1 = xvx1.reshape(self.norbs,-1)
            xvx1 = np.dot(xna,xvx1)
            xvx1 = xvx1.reshape(len(self.nn[s]),self.nprod,self.norbs)
            xvx1 = np.swapaxes(xvx1,1,2)
            xvx.append(xvx1)            
        
        #3-dominant product basis
        if algol=='dp':
            v_pd  = self.pb.get_dp_vertex_array()     #dominant product basis: V_{\widetilde{\mu}}^{ab}
            v_pd1 = v_pd.reshape(v_pd.shape[0]*self.norbs, self.norbs)    #2D shape of dominant product
            c = self.pb.get_da2cc_den()             #atom_centered functional: C_{\widetilde{\mu}}^{\mu}
                                     #V_{\mu}^{ab}= V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu}
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
            xvx.append(xvx2)

        #4-dominant product basis in COO-format
        if algol=='dp_coo':
            v_pd  = self.pb.get_dp_vertex_array()   #dominant product basis: V_{\widetilde{\mu}}^{ab}
            v_pd1 = v_pd.reshape(v_pd.shape[0]*self.norbs, self.norbs)    #2D shape of dominant product
            c = self.pb.get_da2cc_den()             #atom_centered functional: C_{\widetilde{\mu}}^{\mu}
                                     #V_{\mu}^{ab}= V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu} 
            #First step
            data = v_pd.reshape(-1)
            i0,i1,i2 = np.mgrid[0:v_pd.shape[0],0:v_pd.shape[1],0:v_pd.shape[2] ].reshape((3,data.size))
            from pyscf.nao import ndcoo
            nc = ndcoo((data, (i0, i1, i2)))
            m0 = nc.tocoo_pa_b('p,a,b->ap,b')
            size = self.cc_da.shape[0]
            vx1 = m0*(xmb.T)
            vx1 = vx1.reshape(size,self.norbs,self.norbs)#shape (p,a,b)
            vx1 = vx1.reshape(self.norbs,-1)             #shape(a,p*b)  
            #Second Step
            xvx3 = np.dot(xna,vx1)                               #xna(ns,a).V(a,p*b)=xvx(ns,p*b)
            xvx3 = xvx3.reshape(len(self.nn[s]),size,self.norbs) #xvx(ns,p,b)
            xvx3 = np.swapaxes(xvx3,1,2)                         #xvx(ns,b,p)
            xvx3 = np.dot(xvx3,c)                                #XVX=xvx.c
            xvx.append(xvx3)

        if algol=='check':
            print('atom-centered with ref: ',np.allclose(self.gw_xvx(algo='simple')[0],self.gw_xvx(algo='ac')[0],atol=1e-15)) #equality with the direct np.dot
            print('dominant product with ref: ',np.allclose(self.gw_xvx(algo='simple')[0],self.gw_xvx(algo='dp')[0],atol=1e-15)) #equality with the direct np.dot
            print('Sparse_dominant product-ndCoo with ref: ',np.allclose(self.gw_xvx(algo='simple')[0], self.gw_xvx(algo='dp_coo')[0], atol=1e-15)) #equality with the direct np.dot
    return xvx


  def get_snmw2sf_iter(self):
    """ 
    This computes a matrix elements of W_c: <\Psi | W_c |\Psi>.
    sf[spin,n,m,w] = X^n V_mu X^m W_mu_nu X^n V_nu X^m,
    where n runs from s...f, m runs from 0...norbs, w runs from 0...nff_ia, spin=0...1 or 2.
    1- XVX is calculated using dominant product in COO format: gw_xvx('dp_coo')
    2- I_nm = W XVX = (1-v\chi_{0})^{-1}v\chi_{0}v
    3- S_nm = XVX W XVX = XVX * I_nm
    """
    from scipy.sparse.linalg import LinearOperator,lgmres
    ww = 1j*self.ww_ia
    xvx= self.gw_xvx('dp_coo')
    snm2i = []
    for s in range(self.nspin):
        sf_aux = np.zeros((len(self.nn[s]), self.norbs, self.nprod), dtype=self.dtypeComplex)  
        inm = np.zeros((len(self.nn[s]), self.norbs, len(ww)), dtype=self.dtypeComplex)
                      
        for iw,w in enumerate(ww):                     #iw is number of grid and w is complex plane                                
            self.comega_current = w                            
            k_c_opt = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec, dtype=self.dtypeComplex)    #convert k_c as full matrix into Operator
            #print('k_c_opt',k_c_opt.shape)
            for n in range(len(self.nn[s])):    
                for m in range(self.norbs):
                    a = np.dot(self.kernel_sq, xvx[s][n,m,:])     #v XVX
                    b = self.gw_chi0_mv(a, self.comega_current) #\chi_{0}v XVX by using matrix vector
                    a = np.dot(self.kernel_sq, b)               #v\chi_{0}v XVX, this should be aquals to bxvx in last approach
                    sf_aux[n,m,:] ,exitCode = lgmres(k_c_opt, a, atol=1e-05)
            if exitCode != 0: print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
            inm[:,:,iw]=np.einsum('nmp,nmp->nm',xvx[s], sf_aux)   #I= XVX I_aux
        snm2i.append(np.real(inm))
    return snm2i


  def gw_vext2veffmatvec(self,vin):
    dn0 = self.gw_chi0_mv(vin, self.comega_current)
    vcre,vcim = self.gw_applykernel_nspin1(dn0)
    return vin - (vcre + 1.0j*vcim)         #1- v\chi_0


  def gw_vext2veffmatvec2(self,vin):
    dn0 = self.gw_chi0_mv(vin, self.comega_current)
    vcre,vcim = self.gw_applykernel_nspin1(dn0)
    return 1- (vin - (vcre + 1.0j*vcim))    #1- (1-v\chi_0)


  def gw_applykernel_nspin1(self,dn):
    daux  = np.zeros(self.nprod, dtype=self.dtype)
    daux[:] = np.require(dn.real, dtype=self.dtype, requirements=["A","O"])
    vcre = self.spmv(self.nprod, 1.0, self.kernel, daux)
    
    daux[:] = np.require(dn.imag, dtype=self.dtype, requirements=["A","O"])
    vcim = self.spmv(self.nprod, 1.0, self.kernel, daux)
    return vcre,vcim


  def gw_chi0_mv(self,dvin, comega=1j*0.0, dnout=None):
    from scipy.linalg import blas
    from pyscf.nao.m_sparsetools import csr_matvec    
    if dnout is None: 
        dnout = np.zeros_like(dvin, dtype=self.dtypeComplex)

    dnout.fill(0.0)

    vdp = csr_matvec(self.cc_da, dvin.real)  # real part
    sab_real = (vdp*self.v_dab).reshape((self.norbs,self.norbs))

    vdp = csr_matvec(self.cc_da, dvin.imag)  # imaginary
    sab_imag = (vdp*self.v_dab).reshape((self.norbs, self.norbs))

    ab2v_re = np.zeros((self.norbs, self.norbs), dtype=self.dtype)
    ab2v_im = np.zeros((self.norbs, self.norbs), dtype=self.dtype)

    for s in range(self.nspin):
        nb2v = self.gemm(1.0, self.xocc[s], sab_real)
        nm2v_re = self.gemm(1.0, nb2v, self.xvrt[s].T)
    
        nb2v = self.gemm(1.0, self.xocc[s], sab_imag)
        nm2v_im = self.gemm(1.0, nb2v, self.xvrt[s].T)

        vs, nf = self.vstart[s], self.nfermi[s]
    
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
                for m in range(n-vs):  
                    nm2v_re[n,m],nm2v_im[n,m] = 0.0,0.0

        nb2v = self.gemm(1.0, nm2v_re, self.xvrt[s]) # real part
        ab2v_re = self.gemm(1.0, self.xocc[s].T, nb2v, 1.0, ab2v_re)

        nb2v = self.gemm(1.0, nm2v_im, self.xvrt[s]) # imag part
        ab2v_im = self.gemm(1.0, self.xocc[s].T, nb2v, 1.0, ab2v_im)
    
    vdp = csr_matvec(self.v_dab, ab2v_re.reshape(self.norbs*self.norbs))
    chi0_re = vdp*self.cc_da

    vdp = csr_matvec(self.v_dab, ab2v_im.reshape(self.norbs*self.norbs))    
    chi0_im = vdp*self.cc_da

    dnout = chi0_re + 1.0j*chi0_im
    return dnout


  def gw_comp_veff(self, vext, comega=1j*0.0):
    """ This computes an effective field (scalar potential) given the external scalar potential as follows:
        (1-v\chi_{0})V_{eff}=V_{ext}=X_{a}^{n}V_{\mu}^{ab}X_{b}^{m} * v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
        returns V_{eff} as list for all n states(self.nn[s]).
    """
    from scipy.sparse.linalg import LinearOperator
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec, dtype=self.dtypeComplex)

    from scipy.sparse.linalg import lgmres
    resgm, info = lgmres(veff_op, np.require(vext, dtype=self.dtypeComplex, requirements='C'), tol=self.tddft_iter_tol, maxiter=self.maxiter)
    if info != 0: print("LGMRES Warning: info = {0}".format(info))
    return resgm


  def check_veff(self):
    """
    This checks the equality of effective field (scalar potential) given the external scalar potential
    obtained from lgmres(linearopt, v_ext) and np.solve(dense matrix, vext). 
    """
    import numpy as np
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
              v_eff_ref[n,:] = self.gw_comp_veff(xvxbxvx[n,:]) #compute v_eff in tddft_iter class as referance
              v_eff[n,:]=solve(k_c, xvxbxvx[n,:])           #linear eq. for finding V_{eff} --> (1-v\chi_{0})V_{eff}=V_{ext}
    if np.allclose(v_eff,v_eff_ref,atol=1e-4)== True:       #compares both V_{eff}
      return v_eff


  def gw_corr_int_iter(self, sn2w, eps=None):
    """ This computes an integral part of the GW correction at GW class while uses get_snmw2sf_iter"""
    if not hasattr(self, 'snmw2sf'): self.snmw2sf = self.get_snmw2sf_iter()
    return self.gw_corr_int(sn2w, eps=None)


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
    if self.verbosity>2: self.report_mf()
      
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
    mf = scf.UHF(mol)
    mf.kernel()

    gw = gw_iter(mf=mf, gto=mol, verbosity=3, niter_max_ev=20, kmat_algo='sm0_sum', nff_ia=5)
    gw_it = gw.get_snmw2sf_iter()
    gw_no = gw.get_snmw2sf()
    print('Comparison between matrix element of W obtained from gw_iter and gw classes: ', np.allclose(gw_it,gw_no,atol=1e-5)) 

