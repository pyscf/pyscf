'''

@author: Linus Bjarne Dittmer

'''

from pyscf import scf
import pyscf
import numpy
import scipy


class DIIS_M3:
    '''
    This class performs hybrid DIIS/M3 SCF convergence.
    '''

    m3 = None
    mf = None
    threads = 0
    purge_subconvergers = 0.0
    convergence_thresh = 0
    init_scattering = 0
    trust_scale_range = None
    mem_size = 0
    mem_scale = 0.0

    def __init__(self, mf, threads, purgeSolvers=0.5, convergence=8, init_scattering=0.1,
            trust_scale_range=(0.01, 0.2, 8), mem_size=1, mem_scale=0.2):
        '''
        Constructor for the DIIS_M3 method.
        
        Args:
            mf: an instance of SCF class
                SCF object on which M3 is to be constructed.
            threads: int
                The number of threads used in the M3 calculation.
        Kwargs:
            purgeSolvers: float
                The percentage of solvers which are to be annihilated and reassigned in every step of M3.
            convergence: float
                10^-convergence is the convergence threshold for M3.
            init_scattering: float
                Initial Scattering value for the M3 calculation.
            trust_scale_range: float[3]
                Array of 3 floats consisting of min, max and gamma for the trust scale.
            mem_size: int
                Number of past values that should be considered in the M3 calculation. Default is strongly 
                recommended.
            mem_scale: float
                Scaling used for past iterations in the M3 calculation. Default is strongly recommended.
        '''
        self.mf = mf
        self.threads = threads
        self.purge_subconvergers = purgeSolvers
        self.convergence_thresh = convergence
        self.init_scattering = init_scattering
        self.trust_scale_range = trust_scale_range
        self.mem_size = mem_size
        self.mem_scale = mem_scale

    def kernel(self, bufferSize=10, switchThresh=10**-6, hardSwitch=100):
        '''
        Main driver for DIIS/M3.

        Args:
            None
        Kwargs:
            bufferSize: int
                Minimum number of DIIS iterations. Strongly recommended to be at least the size of the DIIS
                buffer.
            switchThresh: float
                Maximum difference of energy that is tolerated between two macro-iterations of DIIS before
                a switch to M3 is enforced.
            hardSwitch: int
                Maximum number of DIIS iterations (not macro-iterations) that are allowed before a switch to
                M3 is enforced.

        Returns: 
            conv: bool
                Whether the SCF is converged.
            energy: float
                Single-point energy of the final result (including nuclear repulsion)
            mo_energy: ndarray
                Molecular orbital energies
            mo_coeff: ndarray
                Molecular orbital coefficients
            mo_occ: ndarray
                Molecular orbital occupancies

        '''
        converged = False
        self.mf.max_cycle = bufferSize
        old_energy = self.mf.kernel()
        diis_conv = self.mf.converged
        mo_energy = self.mf.mo_energy
        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff
        new_energy = self.mf.kernel()
        dm = self.mf.make_rdm1(mo_coeff, mo_occ)
        counter = 0

        while not converged:

            new_energy = self.mf.kernel(dm0=dm)
            counter += 1
            diis_conv = self.mf.converged
            mo_energy = self.mf.mo_energy
            mo_occ = self.mf.mo_occ
            mo_coeff = self.mf.mo_coeff

            denergy = new_energy - old_energy
            old_energy = new_energy
            dm = self.mf.make_rdm1(mo_coeff, mo_occ)
            if not denergy > 0 and abs(denergy) > switchThresh and not counter*bufferSize >= hardSwitch:
                continue
            self.m3 = scf.M3SOSCF(self.mf, self.threads, purgeSolvers=self.purge_subconvergers,
                    convergence=self.convergence_thresh, init_scattering=self._init_scattering,
                    trust_scale_range=self.trust_scale_range, mem_size=self.mem_size, mem_scale=self.mem_scale,
                    initGuess=mo_coeff)

            diis_conv, new_energy, mo_energy, mo_coeff, mo_occ = self.m3.converge()
            converged = diis_conv


        return diis_conv, new_energy, mo_energy, mo_coeff, mo_occ
