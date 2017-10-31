from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from timeit import default_timer as timer
from pyscf.nao import scf
from pyscf.nao.m_tddft_iter_gpu import tddft_iter_gpu_c
from pyscf.nao.m_chi0_noxv import chi0_mv_gpu, chi0_mv
from pyscf.nao.m_blas_wrapper import spmv_wrapper

import scipy
if int(scipy.__version__[0]) > 0:
    scipy_ver = 1
else:
    scipy_ver = 0

try:
    import numba
    from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba
    use_numba = True
except:
    use_numba = False


class tddft_tem(scf):

    def __init__(self, **kw):
        """ 
            Iterative TDDFT a la PK, DF, OC JCTC 
            using moving charge as perturbation.
            The unit of the input are in atomic untis !!!

            Input Parameters:
            -----------------
                velec: xyz component of the electron velocity in atomic unit
                beam_offset: xyz components of the beam offset, must be orthogonal
                        to velec in atomic unit
                dr: spatial resolution for the electron trajectory in atomic unit.
                    Warning: This parameter influence the accuracy of the calculations.
                        if it is taken too large the results will be wrong.
                freq: Frequency range (in atomic unit), freq[0] must be 0.0!!

        """
        from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations


        self.tddft_iter_tol = kw['tddft_iter_tol'] if 'tddft_iter_tol' in kw else 1e-2
        self.eps = kw['iter_broadening'] if 'iter_broadening' in kw else 0.00367493
        self.GPU = GPU = kw['GPU'] if 'GPU' in kw else None
        self.xc_code = xc_code = kw['xc_code'] if 'xc_code' in kw else 'LDA,PZ'
        self.nfermi_tol = nfermi_tol = kw['nfermi_tol'] if 'nfermi_tol' in kw else 1e-5
        self.dtype = kw['dtype'] if 'dtype' in kw else np.float32

        self.velec = kw["velec"] if "velec" in kw else np.array([1.0, 0.0, 0.0])
        self.beam_offset = kw["beam_offset"] if "beam_offset" in kw else np.array([0.0, 0.0, 0.0])
        self.dr = kw["dr"] if "dr" in kw else np.array([0.3, 0.3, 0.3])
        self.freq = kw["freq"] if "freq" in kw else np.arange(0.0, 0.367, 1.5*self.eps)

        self.vnorm = np.sqrt(np.dot(self.velec, self.velec))
        self.vdir = self.velec/self.vnorm

        self.spmv = spmv_wrapper
        if self.dtype == np.float32:
          self.dtypeComplex = np.complex64
          self.gemm = blas.sgemm
          if scipy_ver > 0: self.spmv = blas.sspmv
        elif self.dtype == np.float64:
          self.dtypeComplex = np.complex128
          self.gemm = blas.dgemm
          if scipy_ver > 0: self.spmv = blas.dspmv
        else:
          raise ValueError("dtype can be only float32 or float64")
        self.load_kernel = load_kernel = kw['load_kernel'] if 'load_kernel' in kw else False
        
        assert self.tddft_iter_tol>1e-6
        assert type(self.eps)==float
        assert abs(np.dot(self.velec, self.beam_offset)) < 1e-8 # check orthogonality between beam direction
                                                                # and beam offset
        assert self.freq[0] == 0.0

        
        # heavy calculations after checking !!
        scf.__init__(self, **kw)
        self.telec = kw['telec'] if 'telec' in kw else self.telec
        self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy

        self.check_collision(self.atom2coord)
        self.get_time_range()

        pb = self.pb

        # deallocate hsx
        if hasattr(self, 'hsx'): self.hsx.deallocate()
        
        self.rf0_ncalls = 0
        self.l0_ncalls = 0
        self.matvec_ncalls = 0

        self.v_dab = pb.get_dp_vertex_sparse(dtype=self.dtype, sparseformat=coo_matrix).tocsr()
        self.cc_da = pb.get_da2cc_sparse(dtype=self.dtype, sparseformat=coo_matrix).tocsr()

        self.moms0,self.moms1 = pb.comp_moments(dtype=self.dtype)
        self.nprod = self.moms0.size

        if load_kernel:
            self.load_kernel_method(**kw)
        else:
            self.kernel,self.kernel_dim = pb.comp_coulomb_pack(dtype=self.dtype) # Lower Triangular Part of the kernel
            assert self.nprod==self.kernel_dim, "%r %r "%(self.nprod, self.kernel_dim)
            
            if xc_code.upper()!='RPA' :
              self.comp_fxc_pack(kernel=self.kernel, **kw)


        self.calc_external_potential()
        import sys
        sys.exit()

        # probably unnecessary, require probably does a copy
        # problematic for the dtype, must there should be another option 
        #self.x  = np.require(sv.wfsx.x, dtype=self.dtype, requirements='CW')

        self.ksn2e = np.require(np.zeros((1,self.nspin,self.norbs)), dtype=self.dtype, requirements='CW')
        self.ksn2e[0,0,:] = self.mo_energy
        ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
        self.ksn2f = (3-self.nspin)*ksn2fd
        self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
        self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>nfermi_tol)

        self.xocc = self.mo_coeff[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
        self.xvrt = self.mo_coeff[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?

        self.td_GPU = tddft_iter_gpu_c(GPU, self.mo_coeff[0, 0, :, :, 0], self.ksn2f, self.ksn2e, 
                self.norbs, self.nfermi, self.nprod, self.vstart)

    def check_collision(self, atom2coord):
        """
            Check if the electron collide with an atom
        """

        R0 = -100.0*np.max(atom2coord)*self.vdir + self.beam_offset

        for atm in range(atom2coord.shape[0]):
            vec = R0 - atom2coord[atm, :]
            
            # unit vector to compare to vdir
            vec = abs(vec/np.sqrt(np.dot(vec, vec)))

            if np.sqrt(np.dot(vec-self.vdir, vec-self.vdir)) < 1e-6:
                mess = """
                Electron is collinding with atom {0}:
                velec = [{1:.3f}, {2:.3f}, {3:.3f}]
                beam_offset = [{4:.3f}, {5:.3f}, {6:.3f}]
                atom coord = [{7:.3f}, {8:.3f}, {9:.3f}]
                impact parameter = {10:.9f} > 1e-6
                """.format(atm, self.velec[0], self.velec[1], self.velec[2],
                        self.beam_offset[0], self.beam_offset[1], self.beam_offset[2],
                        atom2coord[atm, 0],  atom2coord[atm, 1],  atom2coord[atm, 2],
                        np.sqrt(np.dot(vec, self.vdir)))
                raise ValueError(mess)

    def get_time_range(self):
        """
            Get the time and symmetric frequency range for the electron passing close
            to the particle. The tim e range is a symmetric array
            around 0.0. At t = 0, the electron is at its closest
            position from the molecule. This array will depend on the
            frequency range and the spatial precision dr.
            To respect the Fourier transform convention, the following
            relationshiip must be fulfill,

            N = 2*pi/(dw*dt)

            with N the number of element of t.
            N must be an odd number in order that t is symmetric
        """

        dt = np.min(self.dr)/self.vnorm
        dw = self.freq[1]-self.freq[0]

        N = int(2*np.pi/(dw*dt))
        if N % 2 == 0:
          N +=1

        wmax = 2.0*np.pi*(N-1)/(N*dt)/2.0

        self.freq_symm = np.arange(-wmax, wmax+dw, dw)

        tmax = (N-1)*dt/2
        self.time = np.arange(-tmax, tmax+dt, dt)
        
    def calc_external_potential(self):
        """
        Calculate the external potential created by a moving charge
        """
        from pyscf.nao.m_tools import find_nearrest_index
        from pyscf.nao.m_ao_matelem import ao_matelem_c
        from pyscf.nao.m_csphar import csphar

        self.V_freq = np.zeros((self.nprod, self.freq.size), dtype=np.complex64)
        V_time = np.zeros((self.time.size), dtype=np.complex64)

        aome = ao_matelem_c(self.ao_log.rr, self.ao_log.pp)
        aome.init_one_set(self.ao_log)
        
        R0 = self.vnorm*self.time[0]*self.vdir + self.beam_offset
        rr = self.ao_log.rr
        dr = (rr[-1]-rr[0])/(rr.size-1)
        dt = self.time[1]-self.time[0]
        dw = self.freq_symm[1] - self.freq_symm[0]
        wmin = self.freq_symm[0]
        tmin = self.time[0]
        nff = self.freq.size
        ub = self.freq_symm.size//2 - 1

        for atm, sp in enumerate(self.atom2sp):
            rcut = self.ao_log.sp2rcut[sp]
            center = self.atom2coord[atm, :]
            rmax = find_nearrest_index(rr, rcut)
            si = self.pb.c2s[sp]

            print(atm, sp, self.nprod, self.pb.c2s[sp], self.pb.c2s[sp+1])
            for mu,l,s,f in aome.ao1.sp2info[sp]:
                inte1 = np.sum(self.ao_log.psi_log_rl[sp][mu, 0:rmax+1]*rr[0:rmax+1]**(l+2)*
                        rr[0:rmax+1]*dr)
                print(mu,l,s,f, "inte1 = ", inte1)

                for k in range(s, f+1):
                    V_time.fill(0.0)

                    for it, t in enumerate(self.time):
                        R_sub = R0 + self.vnorm*self.vdir*(t - self.time[0]) - center
                        norm = np.sqrt(np.dot(R_sub, R_sub))

                        if norm > rcut:
                            I1 = inte1/(norm**(l+1))
                            I2 = 0.0
                        else:
                            rsub_max = find_nearrest_index(rr, norm)

                            I1 = np.sum(self.ao_log.psi_log_rl[sp][mu, 0:rsub_max+1]*
                                    rr[0:rsub_max+1]**(l+2)*rr[0:rsub_max+1])
                            I2 = np.sum(self.ao_log.psi_log_rl[sp][mu, rsub_max+1:]*
                                    rr[rsub_max+1:]/(rr[rsub_max+1:]**(l-1)))

                            I1 = I1*dr/(norm**(l+1))
                            I2 = I2*(norm**l)*dr
                        clm_tem = (4*np.pi/(2*l+1))*csphar(R_sub, 2*aome.jmx+1)*(I1 + I2)
                        rlm_tem = aome.c2r_vector(clm_tem, l, s, f)
                        V_time[it] = rlm_tem[k]

                    V_time *= dt*np.exp(-1.0j*wmin*(self.time-tmin))
                    FT = np.fft.fft(V_time)
                    self.V_freq[si + k, :] = FT[ub+1:ub+nff+1]*np.exp(-1.0j*(wmin*tmin + \
                            self.freq_symm[ub+1:ub+nff+1]-wmin)*tmin)

        print("There is probably mistake!!", np.sum(abs(self.V_freq.real)), np.sum(abs(self.V_freq.imag)))

        raise ValueError("Euh!!! check how to get nc, nfmx, jcut_lmult!!!")

    def load_kernel_method(self, kernel_fname, kernel_format="npy", kernel_path_hdf5=None, **kwargs):

        if kernel_format == "npy":
          self.kernel = self.dtype(np.load(kernel_fname))
        elif kernel_format == "txt":
          self.kernel = np.loadtxt(kernel_fname, dtype=self.dtype)
        elif kernel_format == "hdf5":
          import h5py
          if kernel_path_hdf5 is None:
              raise ValueError("kernel_path_hdf5 not set while trying to read kernel from hdf5 file.")
          self.kernel = h5py.File(kernel_fname, "r")[kernel_path_hdf5].value
        else:
          raise ValueError("Wrong format for loading kernel, must be: npy, txt or hdf5, got " + kernel_format)

        if len(self.kernel.shape) > 1:
          raise ValueError("The kernel must be saved in packed format in order to be loaded!")

        assert self.nprod*(self.nprod+1)//2 == self.kernel.size, "wrong size for loaded kernel: %r %r "%(self.nprod*(self.nprod+1)//2, self.kernel.size)
        self.kernel_dim = self.nprod

    def comp_fxc_lil(self, **kw): 
        """Computes the sparse version of the TDDFT interaction kernel"""
        from pyscf.nao.m_vxc_lil import vxc_lil
        return vxc_lil(self, deriv=2, ao_log=self.pb.prod_log, **kw)
  
    def comp_fxc_pack(self, **kw): 
        """Computes the packed version of the TDDFT interaction kernel """
        from pyscf.nao.m_vxc_pack import vxc_pack
        vxc_pack(self, deriv=2, ao_log=self.pb.prod_log, **kw)

    def apply_rf0(self, v, comega=1j*0.0):
        """ 
            This applies the non-interacting response function to a vector (a set of vectors?) 
        """
    
        assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
        self.rf0_ncalls+=1
        no = self.norbs

        if self.td_GPU.GPU is None:
            return chi0_mv(self, v, comega)
        else:
            return chi0_mv_gpu(self, v, comega) 

    def comp_veff(self, vext, comega=1j*0.0, x0=None):
        """ 
            This computes an effective field (scalar potential) given the external scalar potential
        """
        #from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
        from scipy.sparse.linalg import lgmres, LinearOperator
    
        assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
        self.comega_current = comega
        veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=self.dtypeComplex)
        resgm = lgmres(veff_op, np.require(vext, dtype=self.dtypeComplex, 
            requirements='C'), x0=x0, tol=self.tddft_iter_tol)
        return resgm
  
    def vext2veff_matvec(self, v):
        self.matvec_ncalls+=1 
        chi0 = self.apply_rf0(v, self.comega_current)
    
        # For some reason it is very difficult to pass only one dimension
        # of an array to the fortran routines?? matvec[0, :].ctypes.data_as(POINTER(c_float))
        # is not working!!!

        # real part
        chi0_reim = np.require(chi0.real, dtype=self.dtype, requirements=["A", "O"])
        matvec_real = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim, lower=1)
    
        # imaginary part
        chi0_reim = np.require(chi0.imag, dtype=self.dtype, requirements=["A", "O"])
        matvec_imag = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim, lower=1)

        return v - (matvec_real + 1.0j*matvec_imag)

    def comp_polariz_xx(self, comegas, x0=False):
        """ 
        Compute interacting polarizability

        Inputs:
        -------
            comegas (complex 1D array): frequency range (in Hartree) for which the polarizability is computed.
                                     The imaginary part control the width of the signal.
                                     For example, 
                                     td = tddft_iter_c(...)
                                     comegas = np.arange(0.0, 10.05, 0.05) + 1j*td.eps
            x0 (boolean, optional): determine if a starting guess array should be use to
                                    guess the solution. if True, it will use the non-interacting 
                                    polarizability as guess.
        Output:
        -------
            polariz (complex 1D array): computed polarizability
            self.dn (complex 2D array): computed density change in prod basis
        
        """
        polariz = np.zeros_like(comegas, dtype=np.complex64)
        self.dn = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)
    
        for iw,comega in enumerate(comegas):
            if x0 == True:
                veff,info = self.comp_veff(self.moms1[:,0], comega, x0=self.dn0[iw, :])
            else:
                veff,info = self.comp_veff(self.moms1[:,0], comega, x0=None)

            self.dn[iw, :] = self.apply_rf0(veff, comega)
            polariz[iw] = np.dot(self.moms1[:,0], self.dn[iw, :])

        if self.td_GPU.GPU is not None:
            self.td_GPU.clean_gpu()

        return polariz

    def comp_nonin(self, comegas):
        """ 
        Compute non-interacting polarizability

        Inputs:
        -------
            comegas (complex 1D array): frequency range (in Hartree) for which the polarizability is computed.
                                     The imaginary part control the width of the signal.
                                     For example, 
                                     td = tddft_iter_c(...)
                                     comegas = np.arange(0.0, 10.05, 0.05) + 1j*td.eps
        Output:
        -------
            pxx (complex 1D array): computed non-interacting polarizability
            self.dn0 (complex 2D array): computed non-interacting density change in prod basis
        
        """

        vext = np.transpose(self.moms1)
        pxx = np.zeros(comegas.shape, dtype=np.complex64)
        self.dn0 = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)

        for iw, comega in enumerate(comegas):
            self.dn0[iw, :] = self.apply_rf0(vext[0, :], comega) 
            pxx[iw] = np.dot(self.dn0[iw, :], vext[0,:])
        return pxx
