from __future__ import print_function, division
import sys, numpy as np
from copy import copy
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l
from pyscf.nao.m_rf0_den import rf0_den, rf0_den_numba, rf0_cmplx_ref_blk, rf0_cmplx_ref, rf0_cmplx_vertex_dp
from pyscf.nao.m_rf0_den import rf0_cmplx_vertex_ac, si_correlation, si_correlation_numba
from pyscf.nao.m_rf_den import rf_den
from pyscf.nao.m_rf_den_pyscf import rf_den_pyscf
from pyscf.data.nist import HARTREE2EV
from pyscf.nao.m_valence import get_str_fin
from timeit import default_timer as timer
from numpy import stack, dot, zeros, einsum, pi, log, array, require
import scipy.sparse as sparse
from pyscf.nao import scf
import time

try:
  import numba as nb
  use_numba = True
except:
  use_numba = False

def __LINE__():
      return sys._getframe(1).f_lineno

start_time = time.time()
class gw(scf):
  """ G0W0 with integration along imaginary axis """
  
  def __init__(self, **kw):
    from pyscf.nao.log_mesh import funct_log_mesh
    """ Constructor G0W0 class """
    # how to exclude from the input the dtype and xc_code ?
    scf.__init__(self, **kw)
    print(__name__, ' dtype ', self.dtype)

    self.xc_code_scf = copy(self.xc_code)
    self.niter_max_ev = kw['niter_max_ev'] if 'niter_max_ev' in kw else 15
    self.tol_ev = kw['tol_ev'] if 'tol_ev' in kw else 1e-6
    self.perform_gw = kw['perform_gw'] if 'perform_gw' in kw else False
    self.rescf = kw['rescf'] if 'rescf' in kw else False
    self.bsize = kw['bsize'] if 'bsize' in kw else min(40, self.norbs)
    self.tdscf = kw['tdscf'] if 'tdscf' in kw else None
    self.frozen_core = kw['frozen_core'] if 'frozen_core' in kw else None
    self.write_w = kw['write_w'] if 'write_w' in kw else False
    self.restart_w = kw['restart_w'] if 'restart_w' in kw else False
    if sum(self.nelec) == 1:
      raise RuntimeError('Not implemented H, sorry :-) Look into scf/__init__.py for HF1e class...')
    
    if self.nspin==1:
        self.nocc_0t = nocc_0t = np.array([int((self.nelec+1)/2)])
    elif self.nspin==2:
        self.nocc_0t = nocc_0t = self.nelec
    else:
        raise RuntimeError('nspin>2?')

    if self.verbosity>0:
        mess = """====> Number of:
    * occupied states = {},
    * states up to fermi level= {},
    * nspin = {}, 
    * magnetization = {}""".format(nocc_0t,self.nfermi,self.nspin,self.spin)
        print(__name__, mess)

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

    if self.verbosity>0: print(__name__,'\t\t====> Number of ocupied states are gonna correct (nocc) = {}, Number of virtual states are gonna correct (nvrt) = {}'.format(self.nocc, self.nvrt))

    #self.start_st,self.finish_st = self.nocc_0t-self.nocc, self.nocc_0t+self.nvrt
    frozen_core = kw['frozen_core'] if 'frozen_core' in kw else self.frozen_core
    if frozen_core is not None: 
        st_fi = get_str_fin (self, algo=frozen_core, **kw)
        self.start_st, self.finish_st = st_fi[0], st_fi[1]
    else: 
        self.start_st = self.nocc_0t-self.nocc
        self.finish_st = self.nocc_0t+self.nvrt
    if self.verbosity>0:
      print(__name__,'\t\t====> Indices of states to be corrected start from {} to {} \n'.format(self.start_st,self.finish_st))
    self.nn = [range(self.start_st[s], self.finish_st[s]) for s in range(self.nspin)] # list of states
    

    if 'nocc_conv' in kw:
      s2nocc_conv = [kw['nocc_conv']] if type(kw['nocc_conv'])==int else kw['nocc_conv']
      self.nocc_conv = array([min(i,j) for i,j in zip(s2nocc_conv,nocc_0t)])
    else :
      self.nocc_conv = self.nocc

    if self.verbosity>0:
      print(__name__, __LINE__())
    if 'nvrt_conv' in kw:
      s2nvrt_conv = [kw['nvrt_conv']] if type(kw['nvrt_conv'])==int else kw['nvrt_conv']
      self.nvrt_conv = array([min(i,j) for i,j in zip(s2nvrt_conv,self.norbs-nocc_0t)])
    else :
      self.nvrt_conv = self.nvrt
    
    print(__name__, __LINE__())
    
    if self.rescf:
      self.kernel_scf() # here is rescf with HF functional tacitly assumed
        
    print(__name__, __LINE__())
    self.nff_ia = kw['nff_ia'] if 'nff_ia' in kw else 32    #number of grid points
    self.tol_ia = kw['tol_ia'] if 'tol_ia' in kw else 1e-10
    (wmin_def,wmax_def,tmin_def,tmax_def) = self.get_wmin_wmax_tmax_ia_def(self.tol_ia)
    self.wmin_ia = kw['wmin_ia'] if 'wmin_ia' in kw else wmin_def
    self.wmax_ia = kw['wmax_ia'] if 'wmax_ia' in kw else wmax_def
    self.tmin_ia = kw['tmin_ia'] if 'tmin_ia' in kw else tmin_def
    self.tmax_ia = kw['tmax_ia'] if 'tmax_ia' in kw else tmax_def
    self.tt_ia,self.ww_ia = funct_log_mesh(self.nff_ia, self.tmin_ia, self.tmax_ia, self.wmax_ia)
    #print('self.tmin_ia, self.tmax_ia, self.wmax_ia')
    #print(self.tmin_ia, self.tmax_ia, self.wmax_ia)
    #print(self.ww_ia[0], self.ww_ia[-1])
    print(__name__, __LINE__())

    self.dw_ia = self.ww_ia*(log(self.ww_ia[-1])-log(self.ww_ia[0]))/(len(self.ww_ia)-1)
    self.dw_excl = self.ww_ia[0]
    
    assert self.cc_da.shape[1]==self.nprod
    self.kernel_sq = self.hkernel_den
    #self.v_dab_ds = self.pb.get_dp_vertex_doubly_sparse(axis=2)

    self.x = require(self.mo_coeff[0,:,:,:,0], dtype=self.dtype, requirements='CW')

    if self.perform_gw: self.kernel_gw()
    self.snmw2sf_ncalls = 0
    print(__name__, __LINE__())
    
  def get_h0_vh_x_expval(self):
    """
    This calculates the expectation value of Hartree-Fock Hamiltonian, when:
    self.get_hcore() = -1/2 del^{2}+ V_{ext}
    self.get_j() = Coloumb operator or Hartree energy vh
    self.get_k() = Exchange operator/energy
    mat1 is product of this Hamiltonian and molecular coefficients and it will be diagonalized in expval by einsum
    """
    if self.nspin==1:
      mat = self.get_hcore()+self.get_j()-0.5*self.get_k()
      mat1 = np.dot(self.mo_coeff[0,0,:,:,0], mat)
      expval = einsum('nb,nb->n', mat1, self.mo_coeff[0,0,:,:,0]).reshape((1,self.norbs))
    elif self.nspin==2:
      vh = self.get_j()
      mat = self.get_hcore()+vh[0]+vh[1]-self.get_k()
      expval = zeros((self.nspin, self.norbs))
      for s in range(self.nspin):
        mat1 = np.dot(self.mo_coeff[0,s,:,:,0], mat[s])
        expval[s] = einsum('nb,nb->n', mat1, self.mo_coeff[0,s,:,:,0])
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
    E_gap  = amin(E[where(E>E_fermi)]) - E_homo  
    E_maxdiff = amax(E) - amin(E)
    d = amin(abs(E_homo-E)[where(abs(E_homo-E)>1e-4)])
    wmin_def = sqrt(tol * (d**3) * (E_gap**3)/(d**2+E_gap**2))
    wmax_def = (E_maxdiff**2/tol)**(0.250)
    tmax_def = -log(tol)/ (E_gap)
    tmin_def = -100*log(1.0-tol)/E_maxdiff
    return wmin_def, wmax_def, tmin_def,tmax_def


  rf0_cmplx_ref_blk = rf0_cmplx_ref_blk         #Full matrix response in the basis of atom-centered product functions
  rf0_cmplx_ref = rf0_cmplx_ref                 #Full matrix response in the basis of atom-centered product functions
  rf0_cmplx_vertex_dp = rf0_cmplx_vertex_dp     #Full matrix response in the basis of atom-centered product functions
  rf0_cmplx_vertex_ac = rf0_cmplx_vertex_ac     #Full matrix response in the basis of atom-centered product functions
  rf0 = rf0_den                                 #Full matrix response in the basis of atom-centered product functions for parallel spins
  rf = rf_den                                   #Full matrix interacting response from NAO GW class
  rf_pyscf = rf_den_pyscf                       #Full matrix interacting response from tdscf class


  def si_c(self, ww):
    from numpy.linalg import solve
    """ 
    This computes the correlation part of the screened interaction W_c
    by solving <self.nprod> linear equations (1-K chi0) W = K chi0 K 
    or v_{ind}\sim W_{c} = (1-v\chi_{0})^{-1}v\chi_{0}v
    scr_inter[w,p,q], where w in ww, p and q in 0..self.nprod 
    """

    if not hasattr(self, 'pab2v_den'):
      self.pab2v_den = einsum('pab->apb', self.pb.get_ac_vertex_array())

    si0 = np.zeros((ww.size, self.nprod, self.nprod), dtype=self.dtypeComplex)
    if use_numba:
        si_correlation_numba(si0, ww, self.x, self.kernel_sq, self.ksn2f, self.ksn2e,
                             self.pab2v_den, self.nprod, self.norbs, self.bsize,
                             self.nspin, self.nfermi, self.vstart)
    else:
        si_correlation(rf0(self, ww), si0, ww, self.kernel_sq, self.nprod)
    return si0

  def si_c_via_diagrpa(self, ww):
    """ 
    This method computes the correlation part of the screened interaction W_c
    via the interacting response function. The interacting response function,
    in turn, is computed by diagonalizing RPA Hamiltonian.
    """
    rf = si0 = self.rf_pyscf(ww)
    for iw,r in enumerate(rf):
      si0[iw] = np.dot(self.kernel_sq, np.dot(r, self.kernel_sq))
    return si0

  def epsilon(self, ww):
    """ 
    This computes the dynamic dielectric function epsilon(r,r'.\omega)=\delta(r,r') - v(r.r") * \chi_0(r",r',\omega)
    """
    rf0 = epsilon = self.rf0(ww)
    for iw,w in enumerate(ww):                                
      epsilon[iw,:,:] = np.eye(self.nprod)- np.dot(self.kernel_sq, rf0[iw,:,:])
    return epsilon

  def get_snmw2sf(self, optimize="greedy"):
    """ 
    This computes a matrix elements of W_c: <\Psi\Psi | W_c |\Psi\Psi>.
    sf[spin,n,m,w] = X^n V_mu X^m W_mu_nu X^n V_nu X^m,
    where n runs from s...f, m runs from 0...norbs, w runs from 0...nff_ia, spin=0...1 or 2.
    """
    wpq2si0 = self.si_c(ww = 1j*self.ww_ia).real
    v_pab = self.pb.get_ac_vertex_array()
    #self.snmw2sf_ncalls += 1
    snmw2sf = []
    for s in range(self.nspin):
      nmw2sf = zeros((len(self.nn[s]), self.norbs, self.nff_ia), dtype=self.dtype)
      #nmw2sf = zeros((len(self.nn), self.norbs, self.nff_ia), dtype=self.dtype)

      # n runs from s...f or states will be corrected:
      # self.nn = [range(self.start_st[s], self.finish_st[s])
      xna = self.mo_coeff[0,s,self.nn[s],:,0]
      #xna = self.mo_coeff[0,s,self.nn,:,0]

      # m runs from 0...norbs
      xmb = self.mo_coeff[0,s,:,:,0]

      # This calculates nmp2xvx= X^n V_mu X^m for each side
      nmp2xvx = einsum('na,pab,mb->nmp', xna, v_pab, xmb, optimize=optimize)
      for iw,si0 in enumerate(wpq2si0):
        # This calculates nmp2xvx(outer loop)*real.W_mu_nu*nmp2xvx 
        nmw2sf[:,:,iw] = einsum('nmp,pq,nmq->nm', nmp2xvx, si0, nmp2xvx, optimize=optimize)
      
      snmw2sf.append(nmw2sf)

      if self.write_w:
        from pyscf.nao.m_restart import write_rst_h5py
        print(write_rst_h5py(data = snmw2sf, filename= 'SCREENED_COULOMB.hdf5'))
    
    return snmw2sf

  def gw_corr_int(self, sn2w, eps=None):
    """
    This computes an integral part of the GW correction at energies sn2e[spin,len(self.nn)]
    -\frac{1}{2\pi}\int_{-\infty}^{+\infty } \sum_m \frac{I^{nm}(i\omega{'})}{E_n + i\omega{'}-E_m^0} d\omega{'}
    see eq (16) within DOI: 10.1021/acs.jctc.9b00436
    """

    if not hasattr(self, 'snmw2sf'):
      if self.restart_w is True:
        from pyscf.nao.m_restart import read_rst_h5py
        self.snmw2sf, msg = read_rst_h5py()
        print(msg)
      else:
        self.snmw2sf = self.get_snmw2sf()

    sn2int = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]
    eps = self.dw_excl if eps is None else eps

    # split into mo_energies
    for s,ww in enumerate(sn2w):
      # split mo_energies into each spin channel
      for n,w in enumerate(ww):
        #print(__name__, 's,n,w int corr', s,n,w)
        # runs over orbitals
        for m in range(self.norbs):
          if abs(w-self.ksn2e[0,s,m])<eps : continue
          state_corr = ((self.dw_ia*self.snmw2sf[s][n,m,:] / \
              (w + 1j*self.ww_ia-self.ksn2e[0,s,m])).sum()/pi).real
          #print(n, m, -state_corr, w-self.ksn2e[0,s,m])
          sn2int[s][n] -= state_corr

    return sn2int

  def gw_corr_res(self, sn2w):
    """
    This computes a residue part of the GW correction at energies sn2w[spin,len(self.nn)]
    """
    v_pab = self.pb.get_ac_vertex_array()
    sn2res = [np.zeros_like(n2w, dtype=self.dtype) for n2w in sn2w ]
    
    for s,ww in enumerate(sn2w):    #split into spin and energies
      x = self.mo_coeff[0,s,:,:,0]
        
      # split into nl=counter, n=number of energy level and relevant w=mo_energy inside gw.nn 
      for nl,(n,w) in enumerate(zip(self.nn[s], ww)):
        lsos = self.lsofs_inside_contour(self.ksn2e[0,s,:],w,self.dw_excl)  #gives G's poles in n level with w energy
        zww = array([pole[0] for pole in lsos]) #pole[0]=energies pole[1]=state in gw.nn and pole[2]=occupation number
        #print(__name__, s,n,w, 'len lsos', len(lsos))
        si_ww = self.si_c(ww=zww) #send pole's frequency to calculate W
        xv = np.dot(v_pab,x[n])
        
        for pole,si in zip(lsos, si_ww.real):
          xvx = np.dot(xv, x[pole[1]]) #XVX for x=n v= ac produvt and x=states of poles
          contr = np.dot(xvx, np.dot(si, xvx))
          #print(pole[0], pole[2], contr)
          sn2res[s][nl] += pole[2]*contr
    
    return sn2res

  def lsofs_inside_contour(self, ee, w, eps):
    """ 
      Computes number of states the eigenenergies of which are located inside an integration contour.
      The integration contour depends on w
      z_n=E^0_n-\omega \pm i\eta 
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
    sn2eval_gw = [np.copy(self.ksn2e[0,s,nn]) for s,nn in enumerate(self.nn) ]  #self.ksn2e = self.mo_energy in range of gw.nn
    sn2eval_gw_prev = copy(sn2eval_gw)

    self.nn_conv = []           # self.nn_conv -- list of states to converge, spin-resolved.
    for nocc_0t,nocc_conv,nvrt_conv in zip(self.nocc_0t, self.nocc_conv, self.nvrt_conv):
      self.nn_conv.append( range(max(nocc_0t-nocc_conv,0), min(nocc_0t+nvrt_conv,self.norbs)))

    # iterations to converge the 
    if self.verbosity>0:
      mess = """
        '='*48' G0W0 corrections of eigenvalues  '='*48
        MAXIMUM number of iterations (Input file): {}
        and number of grid points: {}
        Number of GW correction at energies calculated by integral part: {}
        and by sigma part: {}
        GW corection for eigenvalues STARTED:
        """.format(self.niter_max_ev, self.nff_ia, np.size(self.gw_corr_int(sn2eval_gw)),
                   np.size(self.gw_corr_int(sn2eval_gw)))

    for i in range(self.niter_max_ev):
      sn2i = self.gw_corr_int(sn2eval_gw)
      sn2r = self.gw_corr_res(sn2eval_gw)
      
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
        print('Iteration #{:3d}  Relative Error: {:.8f}'.format(i+1, err))
      if self.verbosity>1:
        #print(sn2mismatch)
        for s,n2ev in enumerate(sn2eval_gw):
          print('Spin{}: {}'.format(s+1, n2ev[:]*HARTREE2EV)) #, sn2i[s][:]*HARTREE2EV, sn2r[s][:]*HARTREE2EV))   
      if err<self.tol_ev : 
        if self.verbosity>0: print('-'*43,' |  Convergence has been reached at iteration#{}  | '.format(i+1),'-'*43,'\n')
        break
      if err>=self.tol_ev and i+1==self.niter_max_ev:
        print('-'*32,' |  TAKE CARE! Convergence to tolerance not achieved after {}-iterations  | '.format(self.niter_max_ev),'-'*32,'\n')
    return sn2eval_gw  
    
  def make_mo_g0w0(self):
    """ This creates the fields mo_energy_g0w0, and mo_coeff_g0w0 """

    self.h0_vh_x_expval = self.get_h0_vh_x_expval()

    if self.verbosity>3:    self.report_mf()

    if not hasattr(self,'sn2eval_gw'): self.sn2eval_gw=self.g0w0_eigvals() # Comp. GW-corrections
    
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
      if self.verbosity>2: print(__name__, '\t\t====> Spin {}: energy-sorted MO indices: {}'.format(str(s+1),argsrt))
      self.mo_energy_gw[0,s,:] = np.sort(self.mo_energy_gw[0,s,:])
      for n,m in enumerate(argsrt): self.mo_coeff_gw[0,s,n] = self.mo_coeff[0,s,m]
 
    self.xc_code = 'GW'
    if self.verbosity>4:
      print(__name__,'\t\t====> Performed xc_code: {}\n '.format(self.xc_code))
      print('\nConverged GW-corrected eigenvalues:\n',self.mo_energy_gw*HARTREE2EV)
    
    return self.etot_gw()
        
  kernel_gw = make_mo_g0w0

  def etot_gw(self):
    dm1 = self.make_rdm1()
    ecore = (self.get_hcore()*dm1[0,...,0]).sum()
    vh,kmat = self.get_jk()
    
    if self.nspin==1:
      etot = ecore + 0.5*((vh-0.5*kmat)*dm1[0,...,0]).sum()
    elif self.nspin==2:
      etot = ecore + 0.5*((vh[0]+vh[1]-kmat)*dm1[0,...,0]).sum()
    else:
      print(__name__, self.nspin)
      raise RuntimeError('nspin?')
    
    ecorr = 0.0
    for spin, (n2egw, m2emf, m2occ, n2m) in enumerate(zip(self.mo_energy_gw[0],self.mo_energy[0],self.mo_occ[0],self.argsort)):
      for n, m in enumerate(n2m):
        ecorr -= 0.5*m2occ[m]*(n2egw[n]-m2emf[m])
    
    self.etot_gw = etot+ecorr+self.energy_nuc()
    return self.etot_gw
    
  def spin_square(self):
    from pyscf.nao.mf import mf
    mo_coeff = self.mo_coeff_gw if hasattr(self, 'mo_energy_gw') else self.mo_coeff
    return mf.spin_square(self, mo_coeff=mo_coeff)


  def report_mf(self,dm=None):
    """ Prints the energy levels of mean-field hamiltonian"""
    from pyscf.nao.m_report import report_mfx
    if dm is None: dm = self.make_rdm1()
    return report_mfx(self,dm)
    
  def report_ex(self):
    """ Prints the self energy components in HF levels """
    from pyscf.nao.m_report import sigma_xc
    return sigma_xc(self)

  def report(self):
    """ Prints the energy levels of meanfield and G0W0"""
    from pyscf.nao.m_report import report_gw
    return report_gw(self)
