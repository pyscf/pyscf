'''

@author: Linus Bjarne Dittmer

'''

from pyscf import scf
import pyscf
import numpy
import scipy


class DIIS_M3:

    _m3 = None
    _mf = None
    _threads = 0
    _purgeSolvers = 0.0
    _convergence = 0
    initScattering = 0
    trustScaleRange = None
    memSize = 0
    memScale = 0.0

    def __init__(self, mf, threads, purgeSolvers=0.5, convergence=8, initScattering=0.1, trustScaleRange=(0.01, 0.2, 8), memSize=1, memScale=0.2):
        self._mf = mf
        self._threads = threads
        self._purgeSolvers = purgeSolvers
        self._convergence = convergence
        self._initScattering = initScattering
        self._trustScaleRange = trustScaleRange
        self._memSize = memSize
        self._memScale = memScale

    def kernel(self, bufferSize=1, switchThresh=10**-6, hardSwitch=100):
        #max_cycle = self._mf.max_cycle
        #bufferCursor = 0
        converged = False
        #cycle = 0
        self._mf.max_cycle = bufferSize
        old_energy = self._mf.kernel()
        diis_conv = self._mf.converged
        mo_energy = self._mf.mo_energy
        mo_occ = self._mf.mo_occ
        mo_coeff = self._mf.mo_coeff
        new_energy = self._mf.kernel()
        dm = self._mf.make_rdm1(mo_coeff, mo_occ)
        counter = 0

        while not converged:
            
            new_energy = self._mf.kernel(dm0=dm)
            counter += 1
            diis_conv = self._mf.converged
            mo_energy = self._mf.mo_energy
            mo_occ = self._mf.mo_occ
            mo_coeff = self._mf.mo_coeff
        
            #if diis_conv:
            #    break
            denergy = new_energy - old_energy
            old_energy = new_energy
            print("D Energy: " + str(denergy))
            dm = self._mf.make_rdm1(mo_coeff, mo_occ)
            if not denergy > 0 and abs(denergy) > switchThresh and not counter*bufferSize >= hardSwitch:
                continue
            self._m3 = scf.M3SOSCF(self._mf, self._threads, purgeSolvers=self._purgeSolvers, convergence=self._convergence, initScattering=self._initScattering, \
                    trustScaleRange=self._trustScaleRange, memSize=self._memSize, memScale=self._memScale, initGuess=mo_coeff)

            diis_conv, new_energy, mo_energy, mo_coeff, mo_occ = self._m3.converge()
            converged = diis_conv


        return diis_conv, new_energy, mo_energy, mo_coeff, mo_occ 

















