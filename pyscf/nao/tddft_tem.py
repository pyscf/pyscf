from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from timeit import default_timer as timer
from pyscf.nao import tddft_iter
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


class tddft_tem(tddft_iter):

    def __init__(self, **kw):
        """ 
        Iterative TDDFT a la PK, DF, OC JCTC 
        using moving charge as perturbation.
        The unit of the input are in atomic untis !!!

        Input Parameters:
            dr: spatial resolution for the electron trajectory in atomic unit.
                Warning: This parameter influence the accuracy of the calculations.
                    if it is taken too large the results will be wrong.
            freq: Frequency range (in atomic unit), freq[0] must be 0.0!!

        """
        from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations


        self.dr = kw["dr"] if "dr" in kw else np.array([0.3, 0.3, 0.3])

        # heavy calculations after checking !!
        tddft_iter.__init__(self, **kw)
        self.freq = kw["freq"] if "freq" in kw else np.arange(0.0, 0.367, 1.5*self.eps)

    def get_spectrum_nonin(self, velec = np.array([1.0, 0.0, 0.0]), beam_offset = np.array([0.0, 0.0, 0.0])):
        """
        Calculate the non interacting TEM spectra for an electron trajectory
        
        Input Parameters:
            velec: xyz component of the electron velocity in atomic unit
            beam_offset: xyz components of the beam offset, must be orthogonal
                    to velec in atomic unit
        """
        
        assert velec.size == 3
        assert beam_offset.size == 3

        self.velec = velec
        self.beam_offset = beam_offset

        self.vnorm = np.sqrt(np.dot(self.velec, self.velec))
        self.vdir = self.velec/self.vnorm

        assert abs(np.dot(self.velec, self.beam_offset)) < 1e-8 # check orthogonality between beam direction
                                                                # and beam offset
        self.check_collision(self.atom2coord)
        self.get_time_range()
        self.calc_external_potential()

        return self.comp_tem_spectrum_nonin()

    def get_spectrum_inter(self, velec = np.array([1.0, 0.0, 0.0]), beam_offset = np.array([0.0, 0.0, 0.0])):
        """
        Calculate the interacting TEM spectra for an electron trajectory
        
        Input Parameters:
            velec: xyz component of the electron velocity in atomic unit
            beam_offset: xyz components of the beam offset, must be orthogonal
                    to velec in atomic unit
        """
        
        assert velec.size == 3
        assert beam_offset.size == 3

        self.velec = velec
        self.beam_offset = beam_offset

        self.vnorm = np.sqrt(np.dot(self.velec, self.velec))
        self.vdir = self.velec/self.vnorm

        assert abs(np.dot(self.velec, self.beam_offset)) < 1e-8 # check orthogonality between beam direction
                                                                # and beam offset
        self.check_collision(self.atom2coord)
        self.get_time_range()
        self.calc_external_potential()

        return self.comp_tem_spectrum()

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
              ######### fancy message does not work in python2
              mess = 'np.sqrt(np.dot(vec-self.vdir, vec-self.vdir))<1e-6:'
              #mess = """
              #Electron is collinding with atom {0}:
              #velec = [{1:.3f}, {2:.3f}, {3:.3f}]
              #beam_offset = [{4:.3f}, {5:.3f}, {6:.3f}]
              #atom coord = [{7:.3f}, {8:.3f}, {9:.3f}]
              #impact parameter = {10:.9f} > 1e-6""".format(atm, *self.velec,*self.beam_offset[0],*atom2coord[atm, :],np.sqrt(np.dot(vec, self.vdir)))
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
        dw = self.freq[1] - self.freq[0]

        N = int(2*np.pi/(dw*dt))
        #if N % 2 == 0:
        #  N +=1
        N += 1
        
        dw_symm = 2.0*np.pi/(N*dt)

        wmax = 2.0*np.pi*(N-1)/(N*dt)/2.0

        self.freq_symm = np.arange(-wmax, wmax+dw_symm, dw_symm)

        tmax = (N-1)*dt/2
        self.time = np.arange(-tmax, tmax+dt, dt)
        
    def calc_external_potential(self):
        """
        Calculate the external potential created by a moving charge
        """
        from pyscf.nao.m_comp_vext_tem import comp_vext_tem

        self.V_freq = np.zeros((self.freq.size, self.nprod), dtype=np.complex64)

        comp_vext_tem(self, self.pb.prod_log)
        if self.verbosity>0: print("sum(V_freq) = ", np.sum(abs(self.V_freq.real)), np.sum(abs(self.V_freq.imag)))

    def comp_tem_spectrum(self, x0=False):
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
        comegas = self.freq + 1.0j*self.eps
        polariz = np.zeros_like(comegas, dtype=np.complex64)
        self.dn = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)
    
        for iw,comega in enumerate(comegas):
            if self.verbosity>0: print("freq = ", iw)

            if x0 == True:
                veff = self.comp_veff(self.V_freq[iw, :], comega, x0=self.dn0[iw, :])
            else:
                veff = self.comp_veff(self.V_freq[iw, :], comega, x0=None)

            self.dn[iw, :] = self.apply_rf0(veff, comega)
            polariz[iw] = np.dot(np.conj(self.V_freq[iw, :]), self.dn[iw, :])
            
        if self.td_GPU.GPU is not None:
            self.td_GPU.clean_gpu()

        return -polariz/np.pi

    def comp_tem_spectrum_nonin(self):
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
        comegas = self.freq + 1.0j*self.eps

        pxx = np.zeros(comegas.shape, dtype=np.complex64)
        self.dn0 = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)

        for iw, comega in enumerate(comegas):
            self.dn0[iw, :] = self.apply_rf0(self.V_freq[iw, :], comega) 
            pxx[iw] = np.dot(self.dn0[iw, :], np.conj(self.V_freq[iw, :]))
        return -pxx/np.pi
