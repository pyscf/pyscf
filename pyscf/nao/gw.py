from __future__ import print_function, division
import sys, numpy as np
from numpy import dot, zeros, einsum, pi, log, array
from pyscf.nao import scf
from copy import copy
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l

class gw(scf):
  """ G0W0 with integration along imaginary axis """
  
  def __init__(self, **kw):
    from pyscf.nao.m_log_mesh import log_mesh
    """ Constructor G0W0 class """
    # how to exclude from the input the dtype and xc_code ?
    scf.__init__(self, **kw)
    #print(__name__, ' dtype ', self.dtype)

    self.xc_code_scf = copy(self.xc_code)
    self.niter_max_ev = kw['niter_max_ev'] if 'niter_max_ev' in kw else 15
    self.tol_ev = kw['tol_ev'] if 'tol_ev' in kw else 1e-6
    self.perform_gw = kw['perform_gw'] if 'perform_gw' in kw else False
    self.rescf = kw['rescf'] if 'rescf' in kw else False

    if self.nspin==1: self.nocc_0t = nocc_0t = np.array([int(self.nelec/2)])
    elif self.nspin==2: self.nocc_0t = nocc_0t = self.nelec
    else: raise RuntimeError('nspin>2?')

    if self.verbosity>0: print(__name__, 'nocc_0t =', nocc_0t)

    if 'nocc' in kw:
      s2nocc = [kw['nocc']] if type(kw['nocc'])==int else kw['nocc']
      self.nocc = array([min(i,j) for i,j in zip(s2nocc,nocc_0t)])
    else :
      self.nocc = array([min(6,j) for j in nocc_0t])
      
    if 'nvrt' in kw:
      s2nvrt = [kw['nvrt']] if type(kw['nvrt'])==int else kw['nvrt']
      self.nvrt = array([min(i,j) for i,j in zip(s2nvrt,self.norbs-nocc_0t)])
    else :
      self.nvrt = array([min(6,j) for j in self.norbs-nocc_0t])
    if self.verbosity>0: print(__name__, 'nocc =', self.nocc, 'nvrt =', self.nvrt)

      
    self.start_st,self.finish_st = self.nocc_0t-self.nocc, self.nocc_0t+self.nvrt
    if self.verbosity>0: print(__name__, 'sf_st =', self.start_st, self.finish_st)

    self.nn = [range(self.start_st[s], self.finish_st[s]) for s in range(self.nspin)] # list of states to correct?
    #self.nn = range(self.start_st, self.finish_st) # list of states to correct?

    self.nocc_conv = kw['nocc_conv'] if 'nocc_conv' in kw else self.nocc
    self.nvrt_conv = kw['nvrt_conv'] if 'nvrt_conv' in kw else self.nvrt

    if self.verbosity>0: print(__name__, 'nn =', self.nn)
    
    if self.rescf: self.kernel_scf() # here is rescf with HF functional tacitly assumed
    
    #print(self.nocc, self.nvrt)
    #print(self.start_st, self.finish_st)
    #print('     nn:', self.nn)
    
    self.nff_ia = kw['nff_ia'] if 'nff_ia' in kw else 32
    self.tol_ia = kw['tol_ia'] if 'tol_ia' in kw else 1e-6
    (wmin_def,wmax_def,tmin_def,tmax_def) = self.get_wmin_wmax_tmax_ia_def(self.tol_ia)
    self.wmin_ia = kw['wmin_ia'] if 'wmin_ia' in kw else wmin_def
    self.wmax_ia = kw['wmax_ia'] if 'wmax_ia' in kw else wmax_def
    self.tmin_ia = kw['tmin_ia'] if 'tmin_ia' in kw else tmin_def
    self.tmax_ia = kw['tmax_ia'] if 'tmax_ia' in kw else tmax_def
    self.tt_ia,self.ww_ia = log_mesh(self.nff_ia, self.tmin_ia, self.tmax_ia, self.wmax_ia)
    #print('self.tmin_ia, self.tmax_ia, self.wmax_ia')    
    #print(self.tmin_ia, self.tmax_ia, self.wmax_ia)
    #print(self.ww_ia[0], self.ww_ia[-1])

    self.dw_ia = self.ww_ia*(log(self.ww_ia[-1])-log(self.ww_ia[0]))/(len(self.ww_ia)-1)
    self.dw_excl = self.ww_ia[0]
    
    assert self.cc_da.shape[1]==self.nprod
    self.kernel_sq = self.hkernel_den
    #self.v_dab_ds = self.pb.get_dp_vertex_doubly_sparse(axis=2)

    if self.perform_gw: self.kernel_gw()
    
  def get_h0_vh_x_expval(self):
    if self.nspin==1:
      mat = self.get_hcore()+self.get_j()-0.5*self.get_k()
      mat1 = np.dot(self.mo_coeff[0,0,:,:,0], mat)
      expval = np.einsum('nb,nb->n', mat1, self.mo_coeff[0,0,:,:,0])
    elif self.nspin==2:
      vh = self.get_j()
      mat = self.get_hcore()+vh[0]+vh[1]-self.get_k()
      expval = np.zeros((self.nspin, self.norbs))
      for s in range(self.nspin):
        mat1 = np.dot(self.mo_coeff[0,s,:,:,0], mat[s])
        expval[s] = np.einsum('nb,nb->n', mat1, self.mo_coeff[0,s,:,:,0])
    return expval
    
  def get_wmin_wmax_tmax_ia_def(self, tol):
    from numpy import log, exp, sqrt, where, amin, amax
    """ 
      This is a default choice of the wmin and wmax parameters for a log grid along 
      imaginary axis. The default choice is based on the eigenvalues. 
    """
    E = self.ksn2e[0,0,:]
    E_fermi = self.fermi_energy
    E_homo = amax(E[where(E<=E_fermi)])
    E_gap  = amin(E[where(E>=E_fermi)]) - E_homo  
    E_maxdiff = amax(E) - amin(E)
    d = amin(abs(E_homo-E)[where(abs(E_homo-E)>1e-4)])
    wmin_def = sqrt(tol * (d**3) * (E_gap**3)/(d**2+E_gap**2))
    wmax_def = (E_maxdiff**2/tol)**(0.250)
    tmax_def = -log(tol)/ (E_gap)
    tmin_def = -100*log(1.0-tol)/E_maxdiff
    return wmin_def, wmax_def, tmin_def,tmax_def

  def rf0_cmplx_vertex_dp(self, ww):
    """ Full matrix response in the basis of atom-centered product functions """
    rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
    v_arr = self.pb.get_dp_vertex_array()
    
    zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
    for n,(en,fn) in enumerate(zip(self.ksn2e[0,0,0:self.nfermi], self.ksn2f[0, 0, 0:self.nfermi])):
      vx = dot(v_arr, self.xocc[n,:])
      for m,(em,fm) in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        if (fn - fm)<0 : break
        vxx_a = dot(vx, self.xvrt[m,:]) * self.cc_da
        for iw,comega in enumerate(ww):
          zvxx_a[iw,:] = vxx_a * (fn - fm) * ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
        rf0 = rf0 + einsum('wa,b->wab', zvxx_a, vxx_a)
    return rf0

  def rf0_cmplx_vertex_ac(self, ww):
    """ Full matrix response in the basis of atom-centered product functions """
    rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
    v = self.pb.get_ac_vertex_array()
    
    #print('self.ksn2e', __name__)
    #print(self.ksn2e[0,0,0]-self.ksn2e)
    #print(self.ksn2f)
    #print(' abs(v).sum(), ww.sum(), self.nfermi, self.vstart ')
    #print(abs(v).sum(), ww.sum(), self.nfermi, self.vstart)
    
    zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
    for n,(en,fn) in enumerate(zip(self.ksn2e[0,0,0:self.nfermi], self.ksn2f[0, 0, 0:self.nfermi])):
      vx = dot(v, self.xocc[n,:])
      #print(n, abs(vx).sum(), en)
      for m,(em,fm) in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        if (fn - fm)<0 : break
        vxx_a = dot(vx, self.xvrt[m,:].T)
        for iw,comega in enumerate(ww):
          zvxx_a[iw,:] = vxx_a * (fn - fm) * ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
        rf0 += einsum('wa,b->wab', zvxx_a, vxx_a)
    return rf0
  
  rf0 = rf0_cmplx_vertex_ac
  
  def si_c(self, ww):
    from numpy.linalg import solve
    """ 
    This computes the correlation part of the screened interaction W_c
    by solving <self.nprod> linear equations (1-K chi0) W = K chi0 K
    scr_inter[w,p,q], where w in ww, p and q in 0..self.nprod 
    """
    rf0 = si0 = self.rf0(ww)
    for iw,w in enumerate(ww):
      k_c = dot(self.kernel_sq, rf0[iw,:,:])
      b = dot(k_c, self.kernel_sq)
      k_c = np.eye(self.nprod)-k_c
      si0[iw,:,:] = solve(k_c, b)

    return si0

  def get_snmw2sf(self):
    """ 
    This computes a spectral function of the GW correction.
    sf[spin,n,m,w] = X^n V_mu X^m W_mu_nu X^n V_nu X^m,
    where n runs from s...f, m runs from 0...norbs, w runs from 0...nff_ia, spin=0...1 or 2.
    """
    wpq2si0 = self.si_c(ww = 1j*self.ww_ia).real
    v_pab = self.pb.get_ac_vertex_array()

    snmw2sf = []
    for s in range(self.nspin):
      nmw2sf = zeros((len(self.nn[s]), self.norbs, self.nff_ia), dtype=self.dtype)
      #nmw2sf = zeros((len(self.nn), self.norbs, self.nff_ia), dtype=self.dtype)
      xna = self.mo_coeff[0,s,self.nn[s],:,0]
      #xna = self.mo_coeff[0,s,self.nn,:,0]
      xmb = self.mo_coeff[0,s,:,:,0]
      nmp2xvx = einsum('na,pab,mb->nmp', xna, v_pab, xmb)
      for iw,si0 in enumerate(wpq2si0):
        nmw2sf[:,:,iw] = einsum('nmp,pq,nmq->nm', nmp2xvx, si0, nmp2xvx)
      snmw2sf.append(nmw2sf)
    return snmw2sf

  def gw_corr_int(self, sn2w, eps=None):
    """ This computes an integral part of the GW correction at energies sn2e[spin,len(self.nn)] """
    if not hasattr(self, 'snmw2sf'): self.snmw2sf = self.get_snmw2sf()
    #print(__name__, 'sn2w', sn2w)
    sn2int = np.zeros_like(sn2w, dtype=self.dtype)
    eps = self.dw_excl if eps is None else eps
    #print(__name__, 'self.dw_ia', self.dw_ia, sn2w)
    for s,ww in enumerate(sn2w):
      for n,w in enumerate(ww):
        #print(__name__, 's,n,w int corr', s,n,w)
        for m in range(self.norbs):
          if abs(w-self.ksn2e[0,s,m])<eps : continue
          state_corr = ((self.dw_ia*self.snmw2sf[s][n,m,:] / (w + 1j*self.ww_ia-self.ksn2e[0,s,m])).sum()/pi).real
          #print(n, m, -state_corr, w-self.ksn2e[0,s,m])
          sn2int[s,n] -= state_corr
    return sn2int

  def gw_corr_res(self, sn2w):
    """ This computes a residue part of the GW correction at energies sn2w[spin,len(self.nn)] """
    v_pab = self.pb.get_ac_vertex_array()
    sn2res = np.zeros_like(sn2w, dtype=self.dtype)
    for s,ww in enumerate(sn2w):
      x = self.mo_coeff[0,s,:,:,0]
      for nl,(n,w) in enumerate(zip(self.nn[s],ww)):
      #for nl,(n,w) in enumerate(zip(self.nn,ww)):
        lsos = self.lsofs_inside_contour(self.ksn2e[0,s,:],w,self.dw_excl)
        zww = array([pole[0] for pole in lsos])
        si_ww = self.si_c(ww=zww)
        xv = dot(v_pab,x[n])
        #print(__name__, 's,n,w', s,n,w)
        for pole,si in zip(lsos, si_ww.real):
          xvx = dot(xv, x[pole[1]])
          contr = dot(xvx, dot(si, xvx))
          #print(pole[0], pole[2], contr)
          sn2res[s,nl] += pole[2]*contr
    return sn2res

  def lsofs_inside_contour(self, ee, w, eps):
    """ 
      Computes number of states the eigen energies of which are located inside an integration contour.
      The integration contour depends on w 
    """ 
    nGamma_pos = 0
    nGamma_neg = 0
    for i,e in enumerate(ee):
      zr = e - w
      zi = -np.sign(e-self.fermi_energy)
      if zr>=eps and zi>0 : nGamma_pos +=1
      if abs(zr)<eps and zi>0 : nGamma_pos +=1
    
      if zr<=-eps and zi<0 : nGamma_neg += 1
      if abs(zr)<eps and zi<0 : nGamma_neg += 1

    npol = max(nGamma_pos,nGamma_neg)
    if npol<1: return []

    i2zsc = []
    ipol = 0
    for i,e in enumerate(ee):
      zr = e - w
      zi = -eps*np.sign(e-self.fermi_energy)

      if zr>=eps and zi>0:
        ipol += 1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, -1.0] )
      elif zr<=-eps and zi<0:
        ipol += 1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, +1.0] )
      elif abs(zr)<eps and zi>0:
        ipol += 1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, -0.5] ) #[zr+1j*zi, i, -0.5]
      elif abs(zr)<eps and zi<0:
        ipol +=1
        if ipol>npol: raise RuntimeError('ipol>npol')
        i2zsc.append( [zr+1j*zi, i, +0.5] ) #[zr+1j*zi, i, +0.5]

    if ipol!=npol: raise RuntimeError('loop logics incompat???')
    return i2zsc
  
  def g0w0_eigvals(self):
    """ This computes the G0W0 corrections to the eigenvalues """
    sn2eval_gw = [np.copy(self.ksn2e[0,s,nn]) for s,nn in enumerate(self.nn) ]
    #print(__name__, 'sn2eval_gw', sn2eval_gw)
    
    sn2eval_gw_prev = copy(sn2eval_gw)
    nocc_conv = self.nocc_conv
    nvrt_conv = self.nvrt_conv
    self.nn_close = range(max(self.nocc_0t-nocc_conv,0), 
                          min(self.nocc_0t+nvrt_conv,self.norbs)) # list of states for checking convergence
    
    mo_eigval = np.zeros(self.norbs)
    for i in range(self.niter_max_ev):
      gw_corr_int = self.gw_corr_int(sn2eval_gw)
      gw_corr_res = self.gw_corr_res(sn2eval_gw)
      sn2eval_gw = self.h0_vh_x_expval[self.nn] + gw_corr_int + gw_corr_res
      sn2mismatch = np.zeros(self.norbs)
      sn2mismatch[self.nn] = sn2eval_gw-sn2eval_gw_prev
      mo_eigval[self.nn] = sn2eval_gw
      sn2eval_gw_prev = np.copy(sn2eval_gw)
      err = abs(sn2mismatch[self.nn_close]).sum()/len(self.nn_close)
      if self.verbosity>0: print(__name__, i, err, mo_eigval[self.nn_close], gw_corr_int, gw_corr_res)
      if err<self.tol_ev : break
    
    return sn2eval_gw

  def make_mo_g0w0(self):
    """ This creates the fields mo_energy_g0w0, and mo_coeff_g0w0 """

    self.h0_vh_x_expval = self.get_h0_vh_x_expval()
    if self.verbosity>0:
      print(__name__, '.h0_vh_x_expval: ')
      print(self.h0_vh_x_expval)

    if not hasattr(self, 'sn2eval_gw'): self.sn2eval_gw = self.g0w0_eigvals()

    self.mo_energy_gw = np.copy(self.mo_energy)
    self.mo_coeff_gw = np.copy(self.mo_coeff)
    #print(self.sn2eval_gw.shape, type(self.sn2eval_gw))
    #print(self.nn, type(self.nn))
    #print(self.mo_energy_g0w0.shape, type(self.mo_energy_g0w0))
    for s,nn in enumerate(self.nn):
      self.mo_energy_gw[0,s,nn] = self.sn2eval_gw[s]
      nn_occ = [n for n in nn if n<self.nocc_0t[0]]
      nn_vrt = [n for n in nn if n>=self.nocc_0t[0]]
      scissor_occ = (self.mo_energy_gw[0,s,nn_occ] - self.mo_energy[0,s,nn_occ]).sum()/len(nn_occ)
      scissor_vrt = (self.mo_energy_gw[0,s,nn_vrt] - self.mo_energy[0,s,nn_vrt]).sum()/len(nn_vrt)
      #print(scissor_occ, scissor_vrt)
      mm_occ = list(set(range(self.nocc_0t[s]))-set(nn_occ))
      mm_vrt = list(set(range(self.nocc_0t[s],self.norbs)) - set(nn_vrt))
      #print(mm_occ, mm_vrt)
      self.mo_energy_gw[0,s,mm_occ] +=scissor_occ
      self.mo_energy_gw[0,s,mm_vrt] +=scissor_vrt
      #print(self.mo_energy_g0w0)
      if self.verbosity>0: print(__name__, 'np.argsort(self.mo_energy_gw)', np.argsort(self.mo_energy_gw[0,s,:]))
      argsrt = np.argsort(self.mo_energy_gw[0,s,:])
      self.mo_energy_gw[0,s,:] = np.sort(self.mo_energy_gw[0,s,:])
      for n,m in enumerate(argsrt): self.mo_coeff_gw[0,0,n] = self.mo_coeff[0,0,m]
 
    self.xc_code = 'GW'
    if self.verbosity>0:
      print(__name__, ' self.mo_energy_gw, self.xc_code ', self.xc_code)
      print(self.mo_energy_gw)
        
  kernel_gw = make_mo_g0w0
