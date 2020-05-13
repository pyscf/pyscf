from __future__ import print_function, division
import numpy as np
from timeit import default_timer as timer
from pyscf.nao import tddft_iter

class tddft_tem(tddft_iter):

    def __init__(self, **kw):
        """ 
        Iterative TDDFT using the electostatic potential of a moving charge as perturbation.
        The units of the input are in Hartree atomic units...

        Input Parameters:
            dr: spatial resolution for the electron trajectory in atomic unit.
                Warning: This parameter influence the accuracy of the calculations.
                    if it is taken too large the results will be wrong.

            freq: Frequency range (in atomic unit), freq[0] must be 0.0!!
            
        """
        
        tddft_iter.__init__(self, **kw)

        self.freq = kw["freq"] if "freq" in kw else np.arange(0.0, 0.367, 1.5*self.eps)
        self.dr = kw["dr"] if "dr" in kw else np.array([0.3, 0.3, 0.3])
        self.V_freq = None
        self.velec = None
        self.beam_offset = None


    def get_spectrum_nonin(self, velec = np.array([1.0, 0.0, 0.0]), beam_offset = np.array([0.0, 0.0, 0.0]),
            tmp_fname=None, calc_Vext=True):
        """
        Calculate the non interacting TEM spectra for an electron trajectory
        
        Input Parameters:
            velec: xyz component of the electron velocity in atomic unit
            beam_offset: xyz components of the beam offset, must be orthogonal
                    to velec in atomic unit
        """
        
        assert velec.size == 3
        assert beam_offset.size == 3
        if tmp_fname is not None:
            if not isinstance(tmp_fname, str):
                raise ValueError("tmp_fname must be a string")

        if not calc_Vext and any(self.velec != velec):
            calc_Vext = True
        self.velec = velec
        
        if not calc_Vext and any(self.beam_offset != beam_offset):
            calc_Vext = True
        self.beam_offset = beam_offset

        self.vnorm = np.sqrt(np.dot(self.velec, self.velec))
        self.vdir = self.velec/self.vnorm

        self.check_collision(self.atom2coord)
        self.get_time_range()
        if calc_Vext:
            self.calc_external_potential()
        else:
            if self.V_freq is None:
                self.calc_external_potential()

        return self.comp_tem_spectrum_nonin(tmp_fname=tmp_fname)

    def get_spectrum_inter(self, velec = np.array([1.0, 0.0, 0.0]), 
                                 beam_offset = np.array([0.0, 0.0, 0.0]),
                                 tmp_fname=None, calc_Vext=True):
        """
        Calculate the interacting TEM spectra for an electron trajectory
        
        Input Parameters:
            velec: xyz component of the electron velocity in atomic unit
            beam_offset: xyz components of the beam offset, must be orthogonal
                    to velec in atomic unit
        """
        
        assert velec.size == 3
        assert beam_offset.size == 3
        if tmp_fname is not None:
            if not isinstance(tmp_fname, str):
                raise ValueError("tmp_fname must be a string")


        if not calc_Vext and any(self.velec != velec):
            calc_Vext = True
        self.velec = velec
        
        if not calc_Vext and any(self.beam_offset != beam_offset):
            calc_Vext = True
        self.beam_offset = beam_offset

        self.vnorm = np.sqrt(np.dot(self.velec, self.velec))
        self.vdir = self.velec/self.vnorm
       
        self.check_collision(self.atom2coord)
        self.get_time_range()
        #print(__name__, calc_Vext)
        if calc_Vext:
            self.calc_external_potential()
        else:
            if self.V_freq is None:
                print("self.V_freq is None")
                self.calc_external_potential()

        return self.comp_tem_spectrum(tmp_fname=tmp_fname)

    def check_collision(self, atom2coord):
        """
            Check if the electron collide with an atom
        """

        if self.verbosity>0:
            print("tem parameters:")
            print("vdir: ", self.vdir)
            print("vnorm: ", self.vnorm)
            print("beam_offset: ", self.beam_offset)

        assert abs(np.dot(self.velec, self.beam_offset)) < 1e-8 # check orthogonality between beam direction
                                                                # and beam offset
 
        R0 = -100.0*np.max(atom2coord)*self.vdir + self.beam_offset

        for atm in range(atom2coord.shape[0]):
            vec = R0 - atom2coord[atm, :]
            
            # unit vector to compare to vdir
            vec = abs(vec/np.sqrt(np.dot(vec, vec)))

            if np.sqrt(np.dot(vec-self.vdir, vec-self.vdir)) < 1e-6:
              ######### fancy message does not work in python2
              mess = 'np.sqrt(np.dot(vec-self.vdir, vec-self.vdir))<1e-6:'
              print("atoms {0} coordinate: ".format(atm), atom2coord[atm, :])
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
        from pyscf.nao.m_tools import is_power2

        dt = np.min(self.dr)/self.vnorm
        dw = self.freq[1] - self.freq[0]

        N_org = int(2*np.pi/(dw*dt))

        # to improve performance, N must be a power of 2
        if not is_power2(N_org):
            power = 1
            while 2**power < N_org:
                power +=1

            minima = np.argmin(np.array([abs(2**(power-1) - N_org), abs(2**power - N_org)]))
            if minima == 0:
                N = 2**(power-1)
            else:
                N = 2**power

            if self.verbosity>0: print("N_org = {0}, N_new = {1}".format(N_org, N))
            dt = 2*np.pi/(N*dw)
            dr = dt*self.vnorm
            self.dr = np.array([dr, dr, dr])
        else:
            N = N_org


        dw_symm = 2.0*np.pi/(N*dt)

        wmax = 2.0*np.pi*(N-1)/(N*dt)/2.0

        self.freq_symm = np.arange(-wmax, wmax+dw_symm, dw_symm)[0:N]

        tmax = (N-1)*dt/2
        self.time = np.arange(-tmax, tmax+dt, dt)[0:N]
        
    def calc_external_potential(self):
        """
        Calculate the external potential created by a moving charge
        """
        from pyscf.nao.m_comp_vext_tem import comp_vext_tem

        self.V_freq = comp_vext_tem(self, self.pb.prod_log, self.numba_parallel)
        if self.verbosity>0: print("sum(V_freq) = ", np.sum(abs(self.V_freq.real)), np.sum(abs(self.V_freq.imag)))

    def comp_tem_spectrum(self, x0=False, tmp_fname=None):
        """ 
        Compute interacting polarizability

        Inputs:
        -------
            * comegas (complex 1D array): frequency range (in Hartree) for which the polarizability is computed.
                                     The imaginary part control the width of the signal.
                                     For example, 
                                     td = tddft_iter_c(...)
                                     comegas = np.arange(0.0, 10.05, 0.05) + 1j*td.eps
            * x0 (boolean, optional): determine if a starting guess array should be use to
                                    guess the solution. if True, it will use the non-interacting 
                                    polarizability as guess.
            * tmp_fname (string, default None): temporary file to save polarizability
                                    at each frequency. Can be a life saver for large systems.
                    The format of the file is the following,
                    # energy (Hartree)    Re(gamma)    Im(gamma)
        Output:
        -------
            gamma (complex 1D array): computed eels spectrum
            self.dn (complex 2D array): computed density change in prod basis
        
        """
        comegas = self.freq + 1.0j*self.eps
        gamma = np.zeros_like(comegas, dtype=np.complex64)
        self.dn = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)
    
        for iw,comega in enumerate(comegas):
            if self.verbosity>0: print("freq = ", iw)

            if x0 == True:
                veff = self.comp_veff(self.V_freq[iw, :], comega, x0=self.dn0[iw, :])
            else:
                veff = self.comp_veff(self.V_freq[iw, :], comega, x0=None)

            self.dn[iw, :] = self.apply_rf0(veff, comega)
            gamma[iw] = np.dot(np.conj(self.V_freq[iw, :]), self.dn[iw, :])
            if tmp_fname is not None:
                tmp = open(tmp_fname, "a")
                tmp.write("{0}   {1}   {2}\n".format(comega.real, -gamma[iw].real/np.pi, 
                                                                  -gamma[iw].imag/np.pi))
                tmp.close() # Need to open and close the file at every freq, otherwise
                            # tmp is written only at the end of the calculations, therefore,
                            # it is useless
            
        return -gamma/np.pi

    def comp_tem_spectrum_nonin(self, tmp_fname = None):
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
            gamma (complex 1D array): computed non-interacting eels spectrum
            self.dn0 (complex 2D array): computed non-interacting density change in prod basis
        
        """
        comegas = self.freq + 1.0j*self.eps

        gamma = np.zeros(comegas.shape, dtype=np.complex64)
        self.dn0 = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)
        
        for iw, comega in enumerate(comegas):
            self.dn0[iw, :] = self.apply_rf0(self.V_freq[iw, :], comega) 
            gamma[iw] = np.dot(self.dn0[iw, :], np.conj(self.V_freq[iw, :]))
            if tmp_fname is not None:
                tmp = open(tmp_fname, "a")
                tmp.write("{0}   {1}   {2}\n".format(comega.real, -gamma[iw].real/np.pi, 
                                                                  -gamma[iw].imag/np.pi))
                tmp.close() # Need to open and close the file at every freq, otherwise
                            # tmp is written only at the end of the calculations, therefore,
                            # it is useless
 
        return -gamma/np.pi
