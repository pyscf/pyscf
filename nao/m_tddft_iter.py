from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix
from timeit import default_timer as timer
import sys

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import pycuda.gpuarray as gpuarray
    import skcuda.misc as cumisc
    from m_iter_div_eigenenergy_cuda import div_eigenenergy_cuda

    culinalg.init()
    print("Device compute capability: ", cumisc.get_compute_capability(pycuda.autoinit.device))
    GPU_import = True
except:
    GPU_import = False


try:
    import numba
    from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba
    use_numba = True
except:
    use_numba = False


class tddft_iter_c():

  def __init__(self, sv, pb, tddft_iter_tol=1e-2, tddft_iter_broadening=0.00367493,
          nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None, xc_code='LDA,PZ',
          GPU=False):
    """ Iterative TDDFT a la PK, DF, OC JCTC """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    from pyscf.nao.m_comp_dm import comp_dm

    assert tddft_iter_tol>1e-6
    assert type(tddft_iter_broadening)==float
    assert sv.wfsx.x.shape[-1]==1 # i.e. real eigenvectors we accept here


    self.rf0_ncalls = 0
    self.matvec_ncalls = 0
    self.tddft_iter_tol = tddft_iter_tol
    self.eps = tddft_iter_broadening
    self.sv, self.pb, self.norbs, self.nspin = sv, pb, sv.norbs, sv.nspin
    #print('before vertex_coo')
    self.v_dab = pb.get_dp_vertex_coo(dtype=np.float32).tocsr()
    #print('before conversion coefficients coo')
    self.cc_da = pb.get_da2cc_coo(dtype=np.float32).tocsr()
    #print('before moments')
    self.moms0,self.moms1 = pb.comp_moments(dtype=np.float32)
    self.nprod = self.moms0.size
    #print('before kernel')
    t1 = timer()
    self.kernel = pb.comp_coulomb_den(dtype=np.float32)
    t2 = timer()
    print("Time Hartree kernel: ", t2-t1)

    #print("en first part")
    if xc_code.upper()!='RPA' :
      dm = comp_dm(sv.wfsx.x, sv.get_occupations())
      self.kernel = self.kernel + pb.comp_fxc_lil(dm, xc_code, dtype=np.float32).todense()
    #print("end kernel")

    self.telec = sv.hsx.telec if telec is None else telec
    self.nelec = sv.hsx.nelec if nelec is None else nelec
    self.fermi_energy = sv.fermi_energy if fermi_energy is None else fermi_energy
    self.x  = np.require(sv.wfsx.x, dtype=np.float32, requirements='CW')
    self.ksn2e = np.require(sv.wfsx.ksn2e, dtype=np.float32, requirements='CW')
    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
    self.ksn2f = (3-self.nspin)*ksn2fd
    self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
    self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>nfermi_tol)
    #print('before xocc, xvrt')
    self.xocc = self.x[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
    self.xvrt = self.x[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?
    
    if GPU and GPU_import:
        self.GPU=True
        print("GPU initialization")
        self.xvrt_gpu = gpuarray.to_gpu(self.xvrt)
        self.xvrt_gpu_tr = gpuarray.to_gpu(np.transpose(self.xvrt))
        self.ksn2e_gpu = gpuarray.to_gpu(self.ksn2e[0, 0, :])
        self.ksn2f_gpu = gpuarray.to_gpu(self.ksn2f[0, 0, :])

        self.block_size = np.array([32, 32], dtype=int) # threads by block
        self.grid_size = np.array([0, 0], dtype=int) # number of blocks
        dimensions = [self.nfermi, self.ksn2f.shape[2]]
        print("self.ksn2f.shape = ", self.ksn2f.shape)
        for i in range(2):
            if dimensions[i] <= self.block_size[i]:
                self.block_size[i] = dimensions[i]
                self.grid_size[i] = 1
            else:
                self.grid_size[i] = dimensions[i]/self.block_size[i] + 1
        print("Python: block_size = ", self.block_size, " grid_size = ", self.grid_size, "nfermi = ", self.nfermi)
    elif GPU and not GPU_import:
        raise ValueError("GPU lib failed to initialize!")
    else:
        self.GPU = False


  def finalize_gpu(self):
    self.xvrt_gpu.gpudata.free()
    self.ksn2e_gpu.gpudata.free()
    self.ksn2f_gpu.gpudata.free()

    self.nb2v_gpu.gpudata.free()
    self.nm2v_gpu_real.gpudata.free()
    self.nm2v_gpu_imag.gpudata.free()


  def apply_rf0(self, v, comega=1j*0.0):
    """ This applies the non-interacting response function to a vector (a set of vectors?) """
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    self.rf0_ncalls+=1
    vdp = self.cc_da * np.require(v, dtype=np.complex64)
    no = self.norbs
    sab = csr_matrix((np.transpose(vdp)*self.v_dab).reshape([no,no]))
    nb2v = self.xocc*sab
    #nm2v = np.zeros([self.nfermi,len(self.xvrt)], dtype=np.complex64)

    if self.GPU:
        self.nb2v_gpu = gpuarray.to_gpu(nb2v.real)

        self.nm2v_gpu_real = culinalg.dot(self.nb2v_gpu, self.xvrt_gpu_tr)
        self.nb2v_gpu = gpuarray.to_gpu(nb2v.imag)
        self.nm2v_gpu_imag = culinalg.dot(self.nb2v_gpu, self.xvrt_gpu_tr)
        nm2v_bis = np.dot(nb2v, np.transpose(self.xvrt))
        div_eigenenergy_cuda(self.ksn2e_gpu, self.ksn2f_gpu, self.nfermi, self.vstart,
                comega, self.nm2v_gpu_real, self.nm2v_gpu_imag, self.block_size, self.grid_size)

        for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
            for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,n+1:],self.ksn2f[0,0,n+1:])):
                m = j+n+1-self.vstart
                if m >0:
                    nm2v_bis[n,m] = nm2v_bis[n,m] * (fn - fm) * (( 1.0 / (comega - (em - en))) - (1.0 / (comega + (em - en)) ))
        print("nm2v.flags: ", nm2v_bis.flags)
        print("xvrt.flags: ", self.xvrt.flags)
        print("nm2v_gpu.flags: ", self.nm2v_gpu_real.flags)
        print("xvrt_gpu.flags: ", self.xvrt_gpu.flags)
        self.nb2v_gpu = culinalg.dot(self.nm2v_gpu_real, self.xvrt_gpu)
        self.nm2v_gpu_real = self.nb2v_gpu

        self.nb2v_gpu = culinalg.dot(self.nm2v_gpu_imag, self.xvrt_gpu)
        self.nm2v_gpu_imag = self.nb2v_gpu

        nm2v = 1j*self.nm2v_gpu_imag.get()
        nm2v += self.nm2v_gpu_real.get()

        nb2v = np.dot(nm2v_bis,self.xvrt)
        print("sum(nm2v_gpu): = ", np.sum(abs(nm2v.real)), np.sum(abs(nm2v.imag)))
        print("sum(nm2v_cpu): = ", np.sum(abs(nb2v.real)), np.sum(abs(nb2v.imag)))
        print("error: ", np.sum(abs(nb2v-nm2v)))
        sys.exit()

        ab2v = np.dot(np.transpose(self.xocc),nb2v).reshape(no*no)
    else:
        nm2v = np.dot(nb2v, np.transpose(self.xvrt))
        
        if use_numba:
            div_eigenenergy_numba(self.ksn2e, self.ksn2f, self.nfermi, self.vstart, comega, nm2v)
        else:
            for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
              for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,n+1:],self.ksn2f[0,0,n+1:])):
                m = j+n+1-self.vstart
                # m can be negative, I think we should drop the value if m < 0??
                if m > 0:
                    nm2v[n,m] = nm2v[n,m] * (fn-fm) *\
                      ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )

        nb2v = np.dot(nm2v,self.xvrt)
        ab2v = np.dot(np.transpose(self.xocc),nb2v).reshape(no*no)

    vdp = self.v_dab*ab2v
    res = vdp*self.cc_da
    return res

  def comp_veff(self, vext, comega=1j*0.0):
    from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
    
    """ This computes an effective field (scalar potential) given the external scalar potential """
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=np.complex64)
    resgm = gmres_alias(veff_op, np.require(vext, dtype=np.complex64, requirements='C'), tol=self.tddft_iter_tol)
    return resgm
  
  def vext2veff_matvec(self, v):
    self.matvec_ncalls+=1 
    return v-np.dot(self.kernel, self.apply_rf0(v, self.comega_current))

  def comp_polariz_xx(self, comegas):
    """ Polarizability """
    polariz = np.zeros_like(comegas, dtype=np.complex64)
    for iw,comega in enumerate(comegas):
      veff,info = self.comp_veff(self.moms1[:,0], comega)
      #print(iw, info, veff.sum())
      polariz[iw] = np.dot(self.moms1[:,0], self.apply_rf0( veff, comega ))

    if self.GPU:
        self.finalize_gpu()

    return polariz
