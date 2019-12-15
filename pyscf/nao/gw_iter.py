from __future__ import print_function, division
import sys, numpy as np
from copy import copy
from pyscf.data.nist import HARTREE2EV
from timeit import default_timer as timer
import numpy as np
from numpy import stack, dot, zeros, einsum, pi, log, array, require
from pyscf.nao import scf, gw
import time

def profile(fnc):
    """
    Profiles any function in following class just by adding @profile above function
    """
    import cProfile, pstats, io
    def inner (*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc (*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'   #Ordered
        ps = pstats.Stats(pr,stream=s).strip_dirs().sort_stats(sortby)
        n=20                    #reduced the list to be monitored
        ps.print_stats(n)
        #ps.dump_stats("profile.prof")
        print(s.getvalue())
        return retval
    return inner

start_time = time.time()

class gw_iter(gw):
  """
  Iterative G0W0 with integration along imaginary axis
  """

  def __init__(self, **kw):
    gw.__init__(self, **kw)
    self.gw_iter_tol = kw['gw_iter_tol'] if 'gw_iter_tol' in kw else 1e-4
    self.maxiter = kw['maxiter'] if 'maxiter' in kw else 1000

    self.h0_vh_x_expval = self.get_h0_vh_x_expval()

  def si_c2(self,ww):
    """
    This computes the correlation part of the screened interaction using LinearOpt and lgmres
    lgmres method is much slower than np.linalg.solve !!
    """
    import numpy as np
    from scipy.sparse.linalg import lgmres
    from scipy.sparse.linalg import LinearOperator
    rf0 = si0 = self.rf0(ww)    
    for iw,w in enumerate(ww):                                
      k_c = np.dot(self.kernel_sq, rf0[iw,:,:])                                         
      b = np.dot(k_c, self.kernel_sq)               
      self.comega_current = w
      k_c_opt = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec, dtype=self.dtypeComplex)  
      for m in range(self.nprod): 
         si0[iw,m,:],exitCode = lgmres(k_c_opt, b[m,:], atol=self.gw_iter_tol, maxiter=self.maxiter)   
      if exitCode != 0: print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
      #np.allclose(np.dot(k_c, si0), b, atol=1e-05) == True  #Test   
    return si0

  def si_c_check (self, tol = 1e-5):
    """
    This compares np.solve and LinearOpt-lgmres methods for solving linear equation (1-v\chi_{0}) * W_c = v\chi_{0}v
    """
    import time
    import numpy as np
    ww = 1j*self.ww_ia
    t = time.time()
    si0_1 = self.si_c(ww)      #method 1:  numpy.linalg.solve
    t1 = time.time() - t
    print('numpy: {} sec'.format(t1))
    t2 = time.time()
    si0_2 = self.si_c2(ww)     #method 2:  scipy.sparse.linalg.lgmres
    t3 = time.time() - t2
    print('lgmres: {} sec'.format(t3))
    summ = abs(si0_1 + si0_2).sum()
    diff = abs(si0_1 - si0_2).sum() 
    if diff/summ < tol and diff/si0_1.size < tol:
       print('OK! scipy.lgmres methods and np.linalg.solve have identical results')
    else:
       print('Results (W_c) are NOT similar!')     
    return [[diff/summ] , [np.amax(abs(diff))] ,[tol]]

  #@profile
  def gw_xvx (self, algo=None):
    """
     calculates basis products \Psi(r')\Psi(r') = XVX[spin,(nn, norbs, nprod)] = X_{a}^{n}V_{\nu}^{ab}X_{b}^{m} using 4-methods
     1- direct multiplication by using np.dot and np.einsum via swapping between axis
     2- using atom-centered product basis
     3- using atom-centered product basis and BLAS multiplication
     4- using dominant product basis
     5- using dominant product basis in COOrdinate format
    """
    
    algol = algo.lower() if algo is not None else 'dp_coo'  
    xvx=[]

    # we should write function for each algo
    # this is definitively too long

    #1-direct multiplication with np and einsum
    if algol=='simple':
        v_pab = self.pb.get_ac_vertex_array()       #atom-centered product basis: V_{\mu}^{ab}
        for s in range(self.nspin):
            xna = self.mo_coeff[0,s,self.nn[s],:,0]      #(nstat,norbs)
            xmb = self.mo_coeff[0,s,:,:,0]               #(norbs,norbs)
            xvx_ref  = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb)  #einsum: direct multiplication 
            xvx_ref2 = np.swapaxes(np.dot(xna, np.dot(v_pab,xmb.T)),1,2)  #direct multiplication by using np.dot and swapping between axis
            #print('comparison between einsum and dot: ',np.allclose(xvx_ref,xvx_ref2,atol=1e-15)) #einsum=dot
            xvx.append(xvx_ref)

    #2-atom-centered product basis
    elif algol=='ac':
        v_pab = self.pb.get_ac_vertex_array()       
        #First step
        v_pab1= v_pab.reshape(self.nprod*self.norbs, self.norbs)  #2D shape
        for s in range(self.nspin):
            xna = self.mo_coeff[0,s,self.nn[s],:,0]
            xmb = self.mo_coeff[0,s,:,:,0]
            vx  = np.dot(v_pab1,xmb.T)        #multiplications were done one by one in 2D shape
            vx  = vx.reshape(self.nprod, self.norbs, self.norbs) #reshape into initial 3D shape
            #Second step
            xvx1 = np.swapaxes(vx,0,1)
            xvx1 = xvx1.reshape(self.norbs,-1)
            #Third step
            xvx1 = np.dot(xna,xvx1)
            xvx1 = xvx1.reshape(len(self.nn[s]),self.nprod,self.norbs)
            xvx1 = np.swapaxes(xvx1,1,2)
            xvx.append(xvx1)

    #3-atom-centered product basis and BLAS
    elif algol=='blas':
        from pyscf.nao.m_rf0_den import calc_XVX      #uses BLAS
        v = np.einsum('pab->apb', self.pb.get_ac_vertex_array())
        for s in range(self.nspin):
            vx = np.dot(v, self.mo_coeff[0,s,self.nn[s],:,0].T)
            xvx0 = calc_XVX(self.mo_coeff[0,s,:,:,0], vx)
            xvx.append(xvx0.T)          
        
    #4-dominant product basis
    elif algol=='dp':
        size = self.cc_da.shape[0]
        v_pd  = self.pb.get_dp_vertex_array()   #dominant product basis: V_{\widetilde{\mu}}^{ab}
        c = self.pb.get_da2cc_den()             #atom_centered functional: C_{\widetilde{\mu}}^{\mu}
                                 #V_{\mu}^{ab}= V_{\widetilde{\mu}}^{ab} * C_{\widetilde{\mu}}^{\mu}
        #First step
        v_pd1 = v_pd.reshape(v_pd.shape[0]*self.norbs, self.norbs)    #2D shape
        for s in range(self.nspin):
            xna = self.mo_coeff[0,s,self.nn[s],:,0]
            xmb = self.mo_coeff[0,s,:,:,0]
            vxdp  = np.dot(v_pd1,xmb.T)
            #Second step
            vxdp  = vxdp.reshape(size,self.norbs, self.norbs)
            xvx2 = np.swapaxes(vxdp,0,1)
            xvx2 = xvx2.reshape(self.norbs,-1)
            #Third step
            xvx2 = np.dot(xna,xvx2)
            xvx2 = xvx2.reshape(len(self.nn[s]),size,self.norbs)
            xvx2 = np.swapaxes(xvx2,1,2)
            xvx2 = np.dot(xvx2,c)
            xvx.append(xvx2)


    #5-dominant product basis in COOrdinate-format instead of reshape
    elif algol=='dp_coo':
        size = self.cc_da.shape[0]
        v_pd  = self.pb.get_dp_vertex_array() 
        c = self.pb.get_da2cc_den()
        #First step
        data = v_pd.ravel() #v_pd.reshape(-1)
        #i0,i1,i2 = np.mgrid[0:v_pd.shape[0],0:v_pd.shape[1],0:v_pd.shape[2] ].reshape((3,data.size))   #fails in memory
        i0,i1,i2 = np.ogrid[0:v_pd.shape[0],0:v_pd.shape[1],0:v_pd.shape[2]]
        i00,i11,i22 = np.asarray(np.broadcast_arrays(i0,i1,i2)).reshape((3,data.size))
        from pyscf.nao import ndcoo
        nc = ndcoo((data, (i00, i11, i22)))
        m0 = nc.tocoo_pa_b('p,a,b->ap,b')

        for s in range(self.nspin):
            xna = self.mo_coeff[0,s,self.nn[s],:,0]
            xmb = self.mo_coeff[0,s,:,:,0]
            vx1 = m0*(xmb.T)
            #Second Step
            vx1 = vx1.reshape(size,self.norbs,self.norbs)   #shape (p,a,b)
            vx_ref = vx1.reshape(self.norbs,-1)             #shape (b,p*a)
            #data = vx1.ravel()
            #i0,i1,i2 = np.ogrid[0:vx1.shape[0],0:vx1.shape[1],0:vx1.shape[2]]
            #i00,i11,i22 = np.asarray(np.broadcast_arrays(i0,i1,i2)).reshape((3,data.size))
            #nc1 = ndcoo((data, (i00, i11, i22)))
            #m1 = nc1.tocoo_pa_b('p,a,b->ap,b')  
            #Third Step
            xvx3 = np.dot(xna,vx_ref)                               #xna(ns,a).V(a,p*b)=xvx(ns,p*b)
            xvx3 = xvx3.reshape(len(self.nn[s]),size,self.norbs) #xvx(ns,p,b)
            xvx3 = np.swapaxes(xvx3,1,2)                         #xvx(ns,b,p)
            xvx3 = np.dot(xvx3,c)                                #XVX=xvx.c
            xvx.append(xvx3)
    
    elif algol=='check':
        ref = self.gw_xvx(algo='simple')
        for s in range(self.nspin):
            print('Spin {}, atom-centered with ref: {}'.format(s+1,np.allclose(ref[s],self.gw_xvx(algo='ac')[s],atol=1e-15)))
            print('Spin {}, atom-centered (BLAS) with ref: {}'.format(s+1,np.allclose(ref[s],self.gw_xvx(algo='blas')[s],atol=1e-15)))
            print('Spin {}, dominant product with ref: {}'.format(s+1,np.allclose(ref[s],self.gw_xvx(algo='dp')[s],atol=1e-15)))
            print('Spin {}, sparse_dominant product-ndCoo with ref: {}'.format(s+1,np.allclose(ref[s], self.gw_xvx(algo='dp_coo')[s], atol=1e-15)))
    else:
      raise ValueError("Unknow algp {}".format(algol))

    return xvx

  def get_snmw2sf_iter(self, optimize="greedy"):
    """ 
    This computes a matrix elements of W_c: <\Psi(r)\Psi(r) | W_c(r,r',\omega) |\Psi(r')\Psi(r')>.
    sf[spin,n,m,w] = X^n V_mu X^m W_mu_nu X^n V_nu X^m,
    where n runs from s...f, m runs from 0...norbs, w runs from 0...nff_ia, spin=0...1 or 2.
    1- XVX is calculated using dominant product in COO format: gw_xvx('dp_coo')
    2- I_nm = W XVX = (1-v\chi_0)^{-1}v\chi_0v
    3- S_nm = XVX W XVX = XVX * I_nm
    """

    from scipy.sparse.linalg import LinearOperator,lgmres
    
    ww = 1j*self.ww_ia
    xvx= self.gw_xvx('blas')
    snm2i = []
    #convert k_c as full matrix into Operator
    k_c_opt = LinearOperator((self.nprod,self.nprod),
                             matvec=self.gw_vext2veffmatvec,
                             dtype=self.dtypeComplex)

    for s in range(self.nspin):
        sf_aux = np.zeros((len(self.nn[s]), self.norbs, self.nprod), dtype=self.dtypeComplex)
        inm = np.zeros((len(self.nn[s]), self.norbs, len(ww)), dtype=self.dtypeComplex)
        
        # w is complex plane
        for iw,w in enumerate(ww):
            self.comega_current = w                            
            #print('k_c_opt',k_c_opt.shape)
            for n in range(len(self.nn[s])):    
                for m in range(self.norbs):
                    # v XVX
                    a = np.dot(self.kernel_sq, xvx[s][n,m,:])
                    # \chi_{0}v XVX by using matrix vector
                    b = self.gw_chi0_mv(a, self.comega_current)
                    # v\chi_{0}v XVX, this should be equals to bxvx in last approach
                    a = np.dot(self.kernel_sq, b)
                    sf_aux[n,m,:],exitCode = lgmres(k_c_opt, a,
                                                     atol=self.gw_iter_tol,
                                                     maxiter=self.maxiter)
                    if exitCode != 0:
                      print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
            # I= XVX I_aux
            inm[:,:,iw]=np.einsum('nmp,nmp->nm',xvx[s], sf_aux, optimize=optimize)
        snm2i.append(np.real(inm))

    if (self.write_w==True):
        from pyscf.nao.m_restart import write_rst_h5py
        print(write_rst_h5py(data = snm2i, filename= 'SCREENED_COULOMB.hdf5'))

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

            # padding m<n i.e. negative occupations' difference
            for n in range(vs+1,nf):
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
    """
    This computes an effective field (scalar potential) given the external
    scalar potential as follows:
        (1-v\chi_{0})V_{eff} = V_{ext} = X_{a}^{n}V_{\mu}^{ab}X_{b}^{m} * 
                                         v\chi_{0}v * X_{a}^{n}V_{nu}^{ab}X_{b}^{m}
    
    returns V_{eff} as list for all n states(self.nn[s]).
    """
    
    from scipy.sparse.linalg import LinearOperator
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod),
                             matvec=self.gw_vext2veffmatvec,
                             dtype=self.dtypeComplex)

    from scipy.sparse.linalg import lgmres
    resgm, info = lgmres(veff_op,
                         np.require(vext, dtype=self.dtypeComplex, requirements='C'),
                         atol=self.gw_iter_tol, maxiter=self.maxiter)
    if info != 0:
      print("LGMRES has not achieved convergence: exitCode = {}".format(info))
    return resgm

  def check_veff(self, optimize="greedy"):
    """
    This checks the equality of effective field (scalar potential) given the external
    scalar potential obtained from lgmres(linearopt, v_ext) and np.solve(dense matrix, vext). 
    """

    from numpy.linalg import solve

    ww = 1j*self.ww_ia
    rf0 = self.rf0(ww)
    #V_{\mu}^{ab}
    v_pab = self.pb.get_ac_vertex_array()
    for s in range(self.nspin):
      v_eff = np.zeros((len(self.nn[s]), self.nprod), dtype=self.dtype)
      v_eff_ref = np.zeros((len(self.nn[s]), self.nprod), dtype=self.dtype)
      # X_{a}^{n}
      xna = self.mo_coeff[0,s,self.nn[s],:,0]
      # X_{b}^{m}
      xmb = self.mo_coeff[0,s,:,:,0]
      # X_{a}^{n}V_{\mu}^{ab}X_{b}^{m}
      xvx = np.einsum('na,pab,mb->nmp', xna, v_pab, xmb, optimize=optimize)
      for iw,w in enumerate(ww):     
          # v\chi_{0} 
          k_c = np.dot(self.kernel_sq, rf0[iw,:,:])
          # v\chi_{0}v 
          b = np.dot(k_c, self.kernel_sq)
          #(1-v\chi_{0})
          k_c = np.eye(self.nprod)-k_c
          
          #v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
          bxvx = np.einsum('pq,nmq->nmp', b, xvx, optimize=optimize)
          #V_{ext}=X_{a}^{n}V_{\mu}^{ab}X_{b}^{m} * v\chi_{0}v * X_{a}^{n}V_{\nu}^{ab}X_{b}^{m}
          xvxbxvx = np.einsum ('nmp,nlp->np', xvx, bxvx, optimize=optimize)
          
          for n in range (len(self.nn[s])):
              # compute v_eff in tddft_iter class as referance
              v_eff_ref[n,:] = self.gw_comp_veff(xvxbxvx[n,:])
              # linear eq. for finding V_{eff} --> (1-v\chi_{0})V_{eff}=V_{ext}
              v_eff[n,:]=solve(k_c, xvxbxvx[n,:])

    # compares both V_{eff}
    if np.allclose(v_eff,v_eff_ref,atol=1e-4)== True:
      return v_eff

  def gw_corr_int_iter(self, sn2w, eps=None):
    """
    This computes an integral part of the GW correction at GW class while uses get_snmw2sf_iter
    """

    if self.restart_w is True: 
      from pyscf.nao.m_restart import read_rst_h5py
      self.snmw2sf, msg = read_rst_h5py()
      print(msg)  
    else:
      self.snmw2sf = self.get_snmw2sf_iter()
    
    return self.gw_corr_int(sn2w, eps=None)

  def gw_corr_res_iter(self, sn2w):
    """
    This computes a residue part of the GW correction at energies in iterative procedure
    """
    
    from scipy.sparse.linalg import lgmres, LinearOperator
    v_pab = self.pb.get_ac_vertex_array()
    sn2res = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]   
    k_c_opt = LinearOperator((self.nprod,self.nprod), matvec=self.gw_vext2veffmatvec, dtype=self.dtypeComplex)  
    for s,ww in enumerate(sn2w):
      x = self.mo_coeff[0,s,:,:,0]
      for nl,(n,w) in enumerate(zip(self.nn[s],ww)):
        lsos = self.lsofs_inside_contour(self.ksn2e[0,s,:],w,self.dw_excl)
        zww = array([pole[0] for pole in lsos])
        xv = np.dot(v_pab,x[n])
        for pole, z_real in zip(lsos, zww):
          self.comega_current = z_real
          xvx = np.dot(xv, x[pole[1]])
          a = np.dot(self.kernel_sq, xvx)
          b = self.gw_chi0_mv(a, self.comega_current)
          a = np.dot(self.kernel_sq, b)
          si_xvx, exitCode = lgmres(k_c_opt, a, atol=self.gw_iter_tol, maxiter=self.maxiter)
          if exitCode != 0: print("LGMRES has not achieved convergence: exitCode = {}".format(exitCode))
          contr = np.dot(xvx, si_xvx)
          sn2res[s][nl] += pole[2]*contr.real
    
    return sn2res

  def g0w0_eigvals_iter(self):
    """
    This computes the G0W0 corrections to the eigenvalues
    """

    #self.ksn2e = self.mo_energy
    sn2eval_gw = [np.copy(self.ksn2e[0,s,nn]) for s,nn in enumerate(self.nn) ]
    sn2eval_gw_prev = copy(sn2eval_gw)

    self.nn_conv = []           # self.nn_conv - list of states to converge
    for nocc_0t,nocc_conv,nvrt_conv in zip(self.nocc_0t, self.nocc_conv, self.nvrt_conv):
      self.nn_conv.append( range(max(nocc_0t-nocc_conv,0), min(nocc_0t+nvrt_conv,self.norbs)))

    # iterations to converge the qp-energies 
    if self.verbosity>0: 
        print('='*48,'|  G0W0 corrections of eigenvalues  |','='*48+'\n')
        print('MAXIMUM number of iterations (Input file): {} and number of grid points: {}'.format(self.niter_max_ev,self.nff_ia))
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
          print('Spin{}: {}'.format(s+1, n2ev[:]*HARTREE2EV)) 
      
      if err<self.tol_ev : 
        if self.verbosity>0:
          print('-'*43,
                ' |  Convergence has been reached at iteration#{}  | '.format(i+1),
                '-'*43,'\n')
        break

      if err>=self.tol_ev and i+1==self.niter_max_ev:
        if self.verbosity>0:
          print('='*30,
                ' |  TAKE CARE! Convergence to tolerance {} not achieved after {}-iterations  | '.format(self.tol_ev,self.niter_max_ev),
                '='*30,'\n')
    
    return sn2eval_gw

  #@profile  
  def make_mo_g0w0_iter(self):
    """
    This creates the fields mo_energy_g0w0, and mo_coeff_g0w0
    """

    self.h0_vh_x_expval = self.get_h0_vh_x_expval()
    if self.verbosity>2: self.report_mf()
      
    if not hasattr(self,'sn2eval_gw'): self.sn2eval_gw=self.g0w0_eigvals_iter() # Comp. GW-corrections
    
    # Update mo_energy_gw, mo_coeff_gw after the computation is done
    self.mo_energy_gw = np.copy(self.mo_energy)
    self.mo_coeff_gw = np.copy(self.mo_coeff)
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
    if self.verbosity>3:
      print(__name__,'\t\t====> Performed xc_code: {}\n '.format(self.xc_code))
      print('\nConverged GW-corrected eigenvalues:\n',self.mo_energy_gw*HARTREE2EV)
    
    return self.etot_gw()
        
  # This line is odd !!!
  kernel_gw_iter = make_mo_g0w0_iter

if __name__=='__main__':
    from pyscf import gto, scf
    from pyscf.nao import gw_iter   

    mol = gto.M(atom='''O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978''', basis='ccpvdz',spin=2)
    mf = scf.UHF(mol)
    mf.kernel()

    gw = gw_iter(mf=mf, gto=mol, verbosity=1, niter_max_ev=1, nff_ia=5, nvrt=1, nocc=1, gw_iter_tol=1e-04)

    gw_it = gw.get_snmw2sf_iter()
    gw_ref = gw.get_snmw2sf()
    print('Comparison between matrix element of W obtained from gw_iter and gw classes: ', np.allclose(gw_it, gw_ref, atol= gw.gw_iter_tol)) 

    sn2eval_gw = [np.copy(gw.ksn2e[0,s,nn]) for s,nn in enumerate(gw.nn) ]
    sn2r_it  = gw.gw_corr_res_iter(sn2eval_gw)
    sn2r_ref = gw.gw_corr_res(sn2eval_gw)
    print('Comparison between energies in residue part of the GW correction obtained from gw_iter and gw classes: ', np.allclose(sn2r_it, sn2r_ref, atol= gw.gw_iter_tol))

    #gw.kernel_gw_iter()
    #gw.report()
