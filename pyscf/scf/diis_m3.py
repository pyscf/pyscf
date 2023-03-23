'''

@author: Linus Bjarne Dittmer

'''

from pyscf import scf
import pyscf
import numpy
import scipy


class DIIS_M3:

    m3 = None
    mf = None
    threads = 0
    purge_subconvergers = 0.0
    convergence_thresh = 0
    init_scattering = 0
    trustScaleRange = None
    memSize = 0
    memScale = 0.0

    def __init__(self, mf, threads, purgeSolvers=0.5, convergence=8, init_scattering=0.1, trustScaleRange=(0.01, 0.2, 8), memSize=1, memScale=0.2):
        self.mf = mf
        self.threads = threads
        self.purge_subconvergers = purgeSolvers
        self.convergence_thresh = convergence
        self._init_scattering = init_scattering
        self._trustScaleRange = trustScaleRange
        self._memSize = memSize
        self._memScale = memScale

    def kernel(self, bufferSize=1, switchThresh=10**-6, hardSwitch=100):
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
            self.m3 = scf.M3SOSCF(self.mf, self.threads, purgeSolvers=self.purge_subconvergers, convergence=self.convergence_thresh, init_scattering=self._init_scattering, \
                    trustScaleRange=self.trustScaleRange, memSize=self.memSize, memScale=self.memScale, initGuess=mo_coeff)

            diis_conv, new_energy, mo_energy, mo_coeff, mo_occ = self.m3.converge()
            converged = diis_conv


        return diis_conv, new_energy, mo_energy, mo_coeff, mo_occ 

















