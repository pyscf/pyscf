'''

author: Linus Bjarne Dittmer

'''

import numpy
import numpy.linalg
import scipy
import scipy.linalg
import scipy.special
import pyscf.scf
import pyscf.dft
import pyscf.soscf.newton_ah as newton_ah
import pyscf.soscf.sigma_utils as sigutils
import multiprocessing
import multiprocessing.managers
import os
import itertools
import copy

from pyscf.lib import logger


class M3SOSCF:
    '''
    Attributes for M3SOSCF:
        mf: SCF Object
            SCF Object that is to be converged. Currently only RHF permissible.
        threads: int > 0
            Number of theoretically parallel threads that is to be used. Generally increases speed
        purgeSolvers: float, optional
            Partition of solvers that is to be purged and each iteration. Between 0 and 1
        convergence: int
            Convergence Threshold for Trust is 10**-convergence
        initScattering: float
            Initial statistical distribution of subconverger guesses. The original initial guess (e.g. minao, huckel, 1e, ...) is always conserved as the 0th guess, the rest is scattered around with uniform radius distribution and uniform angular distribution on a box

        Examples:

        >>> mol = gto.M('C 0.0 0.0 0.0; O 0.0 0.0 1.1')
        >>> mf = scf.RHF(mol)
        >>> threads = 5
        >>> m3 = M3SOSCF(threads)
        >>> m3.converge()
    '''

    # scf object, currently only RHF, RKS, UHF and UKS implemented
    _mf = None
    # density matrix of the current state. Used for import/export and to get the size of the system
    _currentDm = None
    # Subconverger Redistribution Handler
    _subconvergerRM = None
    # String identifier of used method. currently supported: RHF
    _method = ''
    # Lowest Energy that is to be accepted as valid. Useful for ESM3-SOSCF
    _lowestEnergy = -1.7*10**302

    # Trust array (threads,)
    _currentTrusts = None
    # Array of Indices where the individual Subconvergers put their solution and trust
    # _subconvergerIndices[(Converger)] = (Index of Converger in _currentTrusts/_moCoeffs)
    _subconvergerIndices = None
    # MO Coeffs of each subconverger (threads,n,n). Used for storage of local solutions
    _moCoeffs = None
    # MO Coeff index that scrolls through the _moCoeffs array for memory purposes. Irrelevant if meM3 is turned off.
    _moCursor = 0
    # MO Coeffs used for initialisation (n,n)
    _moBasisCoeffs = None
    # Energy array (threads,)
    _currentEnergies = None
    # Number of Subconvergers
    _threads = 0
    # Array ob Subconverger Objects
    _subconvergers = None
    # Gradient threshold for convergence
    _convergenceThresh = 10**-5
    # Percentage / 100 of Subconvergers that are deleted and redone according to solution density each iteration
    _purgeSubconvergers = 0.0
    # Initial scattering of Subconvergers
    _initialScattering = 0.0
    # Stepsize in NR step
    _nr_stepsize = 0.0

    _maxcycles = 200


    # Sigma utils variables
    _int2e = None


    def __init__(self, mf, threads, purgeSolvers=0.5, convergence=8, initScattering=0.3, trustScaleRange=(0.05, 0.5, 0.5), memSize=1, memScale=0.2, initGuess='minao', stepsize=0.2):

        self._mf = mf

        if isinstance(self._mf, pyscf.dft.uks.UKS):
            self._method = 'uks'
        elif isinstance(self._mf, pyscf.dft.rks.KohnShamDFT):
            self._method = 'rks'
        elif isinstance(self._mf, pyscf.scf.uhf.UHF):
            self._method = 'uhf'
        elif isinstance(self._mf, pyscf.scf.rohf.ROHF):
            self._method = 'rohf'
        elif isinstance(self._mf, pyscf.scf.hf.RHF):
            self._method = 'rhf'
        else:
            raise Exception('Only HF permitted in current version.')
       
        if self._method == 'uhf' or self._method == 'uks':
            self._currentDm = numpy.zeros((2, self._mf.mol.nao_nr(), self._mf.mol.nao_nr()))
            self._moCoeffs = numpy.zeros((memSize, threads, 2, self._mf.mol.nao_nr(), self._mf.mol.nao_nr()))
        else:
            self._currentDm = numpy.zeros((self._mf.mol.nao_nr(), self._mf.mol.nao_nr()))
            self._moCoeffs = numpy.zeros((memSize, threads, self._mf.mol.nao_nr(), self._mf.mol.nao_nr()))
        self._threads = threads
        self._currentTrusts = numpy.zeros((memSize, threads))
        self._currentEnergies = numpy.zeros(threads)
        self._moBasisCoeffs = numpy.zeros(self._currentDm.shape)
        self._subconvergers = []
        self._subconvergerRM = SubconvergerReassigmentManager(self)
        self._initialScattering = initScattering
        self._subconvergerRM._trustScaleRange = trustScaleRange
        self._subconvergerRM._memScale = memScale
        self._nr_stepsize = stepsize
        self._moCursor = 0
        self._maxcycles = self._mf.max_cycle

        for i in range(threads):
            self._subconvergers.append(Subconverger(self))


        self._subconvergerIndices = numpy.arange(len(self._subconvergers))
        self._purgeSubconvergers = purgeSolvers
        self._convergenceThresh = 10**-convergence


        if not type(initGuess) is numpy.ndarray:
            self.initDensityMatrixWithRothaanStep(self._mf.get_init_guess(key=initGuess))
        else:
            self.initDensityMatrixDirectly(initGuess)

    def initialiseEnergySelectiveM3(self, minenergy, maxenergy, mo_coeffs):
        # Save relevant variables

        self._lowestEnergy = minenergy
        self._moBasisCoeffs = numpy.copy(mo_coeffs)

        # Reinitialise Subconvergers

        for i in range(len(self._subconvergers)):
            self._subconvergers[i] = ESM3_Subconverger(self, minenergy, maxenergy)



    def getDegreesOfFreedom(self):
        '''
        Returns the number of Degrees Of Freedom: N(N-1)/2
        '''
        return int(0.5 * len(self._currentDm[0]) * (len(self._currentDm[0])-1))

    def initDensityMatrixDirectly(self, idc):
        self._moBasisCoeffs = idc
        mo_pe = None
        if self._method == 'uks' or self._method == 'uhf':
            mo_pe = numpy.array((numpy.arange(len(idc[0])), numpy.arange(len(idc[0]))))
        else:
            mo_pe = numpy.array(numpy.arange(len(idc[0])))
        mo_occ = self._mf.get_occ(mo_pe, idc)
        self._mf.mo_occ = mo_occ
        self.setCurrentDm(self._mf.make_rdm1(idc, mo_occ))

    def initDensityMatrixWithRothaanStep(self, idm=None):
        '''
        Initialises the M3SOSCF-Solver with a given density matrix. One Rothaan step is performed afterwards to ensure DM properties.

        Arguments:
            idm: 2D array
                Density Matrix used for initialisation
        '''

        mf = self._mf
        it_num = 1
        mo_coeff = None
        if self._method == 'rohf':
            it_num = 2
        for i in range(it_num):
            fock = mf.get_fock(dm=idm)
            mo_energy, mo_coeff = mf.eig(fock, mf.get_ovlp())
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            self._mf.mo_occ = mo_occ
            idm = mf.make_rdm1(mo_coeff, mo_occ)
        self._moBasisCoeffs = mo_coeff
        self.setCurrentDm(idm)


    def setCurrentDm(self, dm, convertToMO=False):
        '''
        Overrides the current density matrix.

        Arguments:
            dm: 2D array
                New density matrix
            convertToMO: Boolean, optional, default False
                If True, converts the density matrix to MO space via D_{MO} = 0.5 SD_{AO}
        '''
        if convertToMO:
            self._currentDm = 0.5 * self._mf.get_ovlp() @ dm
        else:
            self._currentDm = dm

    def getCurrentDm(self, convertToAO=False):
        '''
        Returns the current density matrix. Possibility of desync, since density matrix is not regularly updated during the SCF procedure. Only use after SCF procedure is complete.

        Arguments:
            convertToAO: Boolean, optional, default False
                If True, converts the density matrix from MO space to AO space via D_{AO} = 2 S^{-1}D_{MO}
        
        Returns:
            currentDm: 2D array
                current density matrix
        '''
        if convertToAO:
            return 2 * numpy.linalg.inv(self._mf.get_ovlp()) @ self._currentDm
        return self._currentDm

    def set(self, currentDm=None, purgeSolvers=-1, convergence=-1, initScatting=-1, trustScaleRange=None, memSize=-1, memScale=-1, mo_coeffs=None):
        if type(currentDm) is numpy.ndarray:
            self.setCurrentDm(currentDm)
        if purgeSolvers >= 0:
            self.purgeSolvers = purgeSolvers



    def kernel(self, purgeSolvers=0.5, convergence=8, initScattering=0.1, trustScaleRange=(0.01, 0.2, 8), memScale=0.2, dm0=None):
        self._purgeSolvers = purgeSolvers
        self._convergenceThresh = 10**(-convergence)
        self._initialScattering = initScattering
        self._subconvergerRM._trustScaleRange = trustScaleRange 
        self._memScale = memScale
        
        if type(dm0) is numpy.ndarray:
            self.initDensityMatrixDirectly(initGuess)
        else:
            raise Exception('Illegal initial matrix: dm0 is not a numpy.ndarray.')

        return self.converge()


    def converge(self):
        '''
        Starts the SCF procedure. 

        Returns:
            scf_conv: boolean
                Whether the SCF managed to converge within the set amount of cycles to the given precision.
            final_energy: float
                Total SCF energy of the converged solution.
            final_mo_energy: 1D array
                Orbital energies of the converged MOs.
            final_mo_coeffs: 2D array
                MO coefficient matrix of the converged MOs.
            mo_occs: 1D array
                Absolute occupancies of the converged MOs.

        Examples:
        >>> mol = gto.M('H 0.0 0.0 0.0; F 0.0 0.0 1.0', basis='6-31g')
        >>> mf = scf.RHF(mol)
        >>> threads = 5
        >>> m3 = scf.M3SOSCF(mf, threads)
        >>> result = m3.converge()
        >>> log.info(result[1]) # Print SCF energy
        -99.9575044930158
        '''
        log = logger.new_logger(self._mf, self._mf.mol.verbose)

        if numpy.einsum('i->', self._moCoeffs.flatten()) == 0:
            for sc in self._subconvergers:
                sc.setMoCoeffs(self._moBasisCoeffs)

        #basis = sigutils.getCanonicalBasis(len(self._currentDm[0]))

        self._subconvergers[0].setMoCoeffs(self._moBasisCoeffs)
        self._moCoeffs[0,0] = self._moBasisCoeffs

        if self._threads >= 2:
            for j in range(1, self._threads):
                if self._method == 'uhf' or self._method == 'uks':
                    mo_pert_a = numpy.random.random(1)[0] * self._initialScattering * sigutils.vectorToMatrix(numpy.random.uniform(low=-0.5, high=0.5, size=(self.getDegreesOfFreedom(),)))
                    mo_pert_b = numpy.random.random(1)[0] * self._initialScattering * sigutils.vectorToMatrix(numpy.random.uniform(low=-0.5, high=0.5, size=(self.getDegreesOfFreedom(),)))
                    mo_coeffs_l = numpy.array((self._moBasisCoeffs[0] @ scipy.linalg.expm(mo_pert_a), self._moBasisCoeffs[1] @ scipy.linalg.expm(mo_pert_b)))
                    self._subconvergers[j].setMoCoeffs(mo_coeffs_l)
                    self._moCoeffs[0,j] = mo_coeffs_l
                else:
                    mo_pert = numpy.random.random(1)[0] * self._initialScattering * sigutils.vectorToMatrix(numpy.random.uniform(low=-0.5, high=0.5, size=(self.getDegreesOfFreedom(),)))
                    mo_coeffs_l = self._moBasisCoeffs @ scipy.linalg.expm(mo_pert)
                    self._subconvergers[j].setMoCoeffs(mo_coeffs_l)
                    self._moCoeffs[0,j] = mo_coeffs_l

        total_cycles = self._maxcycles
        final_energy = 0.0
        scf_conv = False
        final_mo_coeffs = None
        final_mo_energy = None
        last_energy = 10**50

        s1e = self._mf.get_ovlp()
        h1e = self._mf.get_hcore()

        mo_occs = self._mf.mo_occ
        #multiprocessing.set_start_method('fork')
        debug_mo_coeffs = numpy.zeros(self._currentDm.shape)

        guess_energy = self._mf.energy_elec(self._mf.make_rdm1(self._moCoeffs[0,0,:], mo_occs))[0]
        log.info("Guess energy: " + str(guess_energy))

        for cycle in range(self._maxcycles):

            # handle MO Coefficient cursor for I/O

            writeCursor = self._moCursor
            readCursor = self._moCursor-1
            if readCursor < 0:
                readCursor = len(self._moCoeffs)-1

            if cycle == 0:
                writeCursor = 0
                readCursor = 0

            # edit subconverges according to solution density
            # a certain number of subconvergers get purged each iteration
            # purge = 0.3 - 0.8

            log.info("Iteration: " + str(cycle))

            purge_indices = None

            if cycle > 0 and len(self._subconvergers) > 1:
                sorted_indices = numpy.argsort(self._currentTrusts[readCursor])
                purge_indices = sorted_indices[0:min(int(len(sorted_indices) * (self._purgeSubconvergers)), len(sorted_indices))]
                uniquevals, uniqueindices = numpy.unique(self._currentTrusts[readCursor], return_index=True)
                nonuniqueindices = []

                for i in range(self._threads):
                    if not i in uniqueindices:
                        nonuniqueindices.append(i)

                nui = numpy.array(nonuniqueindices, dtype=numpy.int32)
                zero_indices = numpy.where(self._currentTrusts[readCursor,:] <= self._convergenceThresh)[0]
                low_indices = numpy.where(self._currentEnergies < self._lowestEnergy)[0]
                if not type(self._subconvergers[0]) is ESM3_Subconverger:
                    purge_indices = numpy.unique(numpy.concatenate((purge_indices, nui, zero_indices, low_indices)))
                else:
                    purge_indices = numpy.unique(numpy.concatenate((purge_indices, nui, zero_indices, low_indices)))


                purge_indices = numpy.sort(purge_indices)
                if purge_indices[0] == 0 and self._currentTrusts[readCursor,0] > 0.0 and self._currentEnergies[purge_indices[0]] > self._lowestEnergy:
                    purge_indices = purge_indices[1:] 

                max_written = min(cycle, len(self._currentTrusts))
                log.info("Purge Indices: " + str(purge_indices))
                log.info("Purging: " + str(len(purge_indices)) + " / " + str(len(self._subconvergers)))
                new_shifts = self._subconvergerRM.generateNewShifts(self._currentTrusts[:max_written], self._moCoeffs[:max_written], len(purge_indices), readCursor, log)

                for j in range(len(purge_indices)):
                    self._moCoeffs[writeCursor,purge_indices[j]] = new_shifts[j]
                    self._currentEnergies[purge_indices[j]] = numpy.finfo(dtype=numpy.float32).max


            for j in range(len(self._subconvergers)):
                newCursor = readCursor
                if type(purge_indices) is numpy.ndarray:
                    if self._subconvergerIndices[j] in purge_indices:
                        newCursor = writeCursor
                self._subconvergers[j].setMoCoeffs(self._moCoeffs[newCursor,self._subconvergerIndices[j]])
                self._subconvergers[j].setEnergy(self._currentEnergies[j])


            # generate local solutions and trusts

            # buffer array for new mocoeffs
            newMoCoeffs = numpy.copy(self._moCoeffs[readCursor])

            sorted_trusts = numpy.zeros(1, dtype=numpy.int32)
            if len(self._subconvergers) > 1:
                sorted_trusts = numpy.argsort(self._currentTrusts[readCursor, 1:]) + 1


            #M3MultiprocessingManager.register("SubconvergerContainer", SubconvergerContainer)


            #with M3MultiprocessingManager() as mpr_manager:
            #    shared_container = mpr_manager.SubconvergerContainer(self._moCoeffs, self._mf)
            #    
            #    processes = []
            #    for k in range(len(self._subconvergers)):
            #        processes.append(multiprocessing.Process(target=processSubconvergers, args=(shared_container,)))
            #    
            #    for process in processes:
            #        process.start()

            #container_size = len(os.sched_getaffinity(0))
            #num_containers = int(float(self._threads) / len(os.sched_getaffinity(0))) + 1

            #for j in range(num_containers):
            #    cont_subconvergers = [self._subconvergers[k] for k in range(j*container_size, min((j+1)*container_size, len(self._subconvergers)))]
            #    cont_size = len(cont_subconvergers)
            #    pool = multiprocessing.Pool(cont_size)
            #    pool.map(execSubconverger, [""] * cont_size)
            #    log.info("Subconverger subset length: " + str(len(cont_subconvergers)))


            for j in range(len(self._subconvergers)):

                sol, trust = self._subconvergers[j].getLocalSolAndTrust(h1e, s1e)

                numpy.set_printoptions(linewidth=500, precision=2)
                log.info("J: " + str(j) + " Trust: " + str(trust))

                if trust == 0:
                    continue

                writeTrustIndex = 0
                if j > 0:
                    writeTrustIndex = sorted_trusts[j-1]
                
                # update trust and solution array

                mc_threshold = 1 - self._currentTrusts[readCursor,writeTrustIndex] + trust

                if j == 0 and self._currentTrusts[readCursor,j] > 0.0 or len(self._subconvergers) == 1:
                    self._currentTrusts[writeCursor,j] = trust
                    newMoCoeffs[j] = sol
                    self._subconvergerIndices[j] = j


                elif numpy.random.rand(1) < mc_threshold:
                    self._currentTrusts[writeCursor,writeTrustIndex] = trust
                    newMoCoeffs[writeTrustIndex] = sol
                    self._subconvergerIndices[j] = writeTrustIndex

            # update moCoeff array with buffer
            self._moCoeffs[writeCursor] = numpy.copy(newMoCoeffs)


            # check for convergence

            highestTrustIndex = numpy.argmax(self._currentTrusts[writeCursor])
            log.info("Highest Trust Index: " + str(highestTrustIndex))
            log.info("Lowest Energy: " + str(numpy.min(self._currentEnergies)))
            log.info("Lowest Energy Index: " + str(numpy.argmin(self._currentEnergies)))
            # current energy

            for j in range(len(self._currentEnergies)):
                self._currentEnergies[j] = self._mf.energy_elec(self._mf.make_rdm1(self._moCoeffs[writeCursor,j], self._mf.mo_occ))[0]
                if self._currentEnergies[j] < self._lowestEnergy:
                    log.info("Energy pruning")
                    self._currentTrusts[writeCursor,lowestTrustIndex] = 0.0
                log.info("ENERGY (" + str(j) + "): " + str(self._currentEnergies[j]))

            log.info("")

            #log.info("Gradient L2 norm: " + str(numpy.linalg.norm(self._solGradients[highestTrustIndex])))
            #log.info("Gradient L1 norm: " + str(numpy.linalg.norm(self._solGradients[highestTrustIndex], 1)))

            scf_tconv =  1 - self._currentTrusts[writeCursor,highestTrustIndex]**4 < self._convergenceThresh
            current_energy = numpy.min(self._currentEnergies[numpy.where(self._currentEnergies >= self._lowestEnergy)[0]])
            log.info("Lowest Energy: " + str(current_energy))
            if scf_tconv and current_energy - self._currentEnergies[highestTrustIndex] < -self._convergenceThresh and not type(self._subconvergers[0]) is ESM3_Subconverger:
                del_array1 = numpy.where(self._currentEnergies >= self._currentEnergies[highestTrustIndex])[0]
                del_array2 = numpy.where(1 - self._currentTrusts[writeCursor,:]**4 < self._convergenceThresh)[0]
                log.info("Deletion Array 1 (Too High Energy): " + str(del_array1))
                log.info("Deletion Array 2 (Converged): " + str(del_array2))
                log.info("Intersected Deletion Array: " + str(numpy.intersect1d(del_array1, del_array2)))
                self._currentTrusts[writeCursor,numpy.intersect1d(del_array1, del_array2)] = 0.0
                log.info("### DISREGARDING SOLUTION DUE TO NON VARIATIONALITY ###")
                scf_tconv = False

            log.info("Trust converged: " + str(scf_tconv))


            if scf_tconv:
                self._currentDm = self._mf.make_rdm1(self._moCoeffs[writeCursor,highestTrustIndex], self._mf.mo_occ)

                final_fock = self._mf.get_fock(dm=self._currentDm, h1e=h1e, s1e=s1e)
                final_mo_coeffs = self._moCoeffs[writeCursor,highestTrustIndex]
                final_mo_energy = self.calculateOrbitalEnergies(final_mo_coeffs, final_fock, s1e)
                final_energy = self._mf.energy_tot(self._currentDm, h1e=h1e)
                total_cycles = cycle+1
                #final_mo_energy_buffer = numpy.zeros(final_mo_energy.shape)
                #orbital_overlap_matrix = None
                #if self._method == 'uhf' or self._method == 'uks':
                #    orbital_overlap_matrix = numpy.einsum('aki,kl,ajk->aij', dummy_mo_coeffs, s1e, final_mo_coeffs)
                #else:
                #    orbital_overlap_matrix = numpy.einsum('ki,kl,lj->ij', dummy_mo_coeffs, s1e, final_mo_coeffs)
#
#                orbital_overlap_matrix = numpy.abs(orbital_overlap_matrix)
#
#                for o in range(len(orbital_overlap_matrix[0])):
#                    match_index = None
#                    if self._method == 'uhf' or self._method == 'uks':
#                        match_index_a = numpy.argmax(abs(orbital_overlap_matrix[0,o,:]))
#                        match_index_b = numpy.argmax(abs(orbital_overlap_matrix[1,o,:]))
#                        final_mo_energy_buffer[0,o] = final_mo_energy[0,match_index_a]
#                        final_mo_energy_buffer[1,o] = final_mo_energy[1,match_index_b]
#                    else:
#                        match_index = numpy.array(numpy.argmax(abs(orbital_overlap_matrix[o,:])))
#                        final_mo_energy_buffer[o] = final_mo_energy[match_index]

#               final_mo_energy = final_mo_energy_buffer
                self._mf.mo_energy = final_mo_energy
                self._mf.mo_coeff = final_mo_coeffs
                self._mf.e_tot = final_energy
                self._mf.converged = True

                scf_conv = True
                break

            self._moCursor += 1
            if self._moCursor >= len(self._moCoeffs):
                self._moCursor = 0


        log.info("Final Energy: " + str(final_energy) + " ha")
        log.info("Cycles: " + str(total_cycles))
        
        self.dumpInfo(log, total_cycles)


        return scf_conv, final_energy, final_mo_energy, final_mo_coeffs, mo_occs
    

    def calculateOrbitalEnergies(self, mo_coefficients, fock, s1e):
        # Oribtal energies calculated from secular equation
        
        mo_energies = None
        
        if self._method == 'uhf' or self._method == 'uks':
            s1e_inv = numpy.linalg.inv(s1e)
            f_eff_a = s1e_inv @ fock[0]
            f_eff_b = s1e_inv @ fock[1]
            mo_energies_a = numpy.diag(numpy.linalg.inv(mo_coefficients[0]) @ f_eff_a @ mo_coefficients[0])
            mo_energies_b = numpy.diag(numpy.linalg.inv(mo_coefficients[1]) @ f_eff_b @ mo_coefficients[1])

            mo_energies = numpy.array((mo_energies_a, mo_energies_b))

        else:
            f_eff = numpy.linalg.inv(s1e) @ fock
            mo_energies = numpy.diag(numpy.linalg.inv(mo_coefficients) @ f_eff @ mo_coefficients)


        return mo_energies

    def dumpInfo(self, log, cycles):
        log.info("")
        log.info("==== INFO DUMP ====")
        log.info("")
        log.info("Number of Cycles:         " + str(cycles))
        log.info("Final Energy:             " + str(self._mf.e_tot))
        log.info("Converged:                " + str(self._mf.converged))
        homo_index = None
        lumo_index = None
        if self._method == 'uhf' or self._method == 'uks':
            occs = numpy.where(self._mf.mo_occ[0,:] > 0.5)[0]
            no_occs = numpy.where(self._mf.mo_occ[0,:] < 0.5)[0]
            homo_index = (0, occs[numpy.argmax(self._mf.mo_energy[0,occs])])
            lumo_index = (0, no_occs[numpy.argmin(self._mf.mo_energy[0,no_occs])])
        else:
            occs = numpy.where(self._mf.mo_occ > 0.5)[0]
            no_occs = numpy.where(self._mf.mo_occ < 0.5)[0]
            homo_index = occs[numpy.argmax(self._mf.mo_energy[occs])]
            lumo_index = no_occs[numpy.argmin(self._mf.mo_energy[no_occs])]
        log.info("HOMO Index:               " + str(homo_index))
        log.info("LUMO Index:               " + str(lumo_index))
        homo_energy = 0
        lumo_energy = 0
        homo_energy = self._mf.mo_energy[homo_index]
        lumo_energy = self._mf.mo_energy[lumo_index]
        log.info("HOMO Energy:              " + str(homo_energy))
        log.info("LUMO Energy:              " + str(lumo_energy))
        log.info("Aufbau solution:          " + str(homo_energy < lumo_energy))

        if self._method == 'uhf' or self._method == 'uks':
            ss = self._mf.spin_square()
            log.info("Spin-Square:              " + str(ss[0]))
            log.info("Multiplicity:             " + str(ss[1]))

        log.info("")
        log.info("")
        log.info("ORIBTAL SUMMARY:")
        log.info("")
        log.info("Index:        Energy [ha]:                        Occupation:")
        if self._method == 'uhf' or self._method == 'uks':
            for index in range(len(self._mf.mo_energy[0])):
                mo_e = self._mf.mo_energy[0,index]
                s = str(index) + " (A)" + (9 - len(str(index))) * " "
                if mo_e > 0:
                    s += " "
                s += str(mo_e)
                if mo_e > 0:
                    s += (36 - len(str(mo_e))) * " "
                else:
                    s += (37 - len(str(mo_e))) * " "
                if self._mf.mo_occ[0,index] > 0:
                    s += "A"
                log.info(s)

                mo_e = self._mf.mo_energy[1,index]
                s = str(index) + " (B)" + (9 - len(str(index))) * " "
                if mo_e > 0:
                    s += " "
                s += str(mo_e)
                if mo_e > 0:
                    s += (36 - len(str(mo_e))) * " "
                else:
                    s += (37 - len(str(mo_e))) * " "
                if self._mf.mo_occ[1,index] > 0:
                    s += "  B"
                log.info(s)
        else:
            for index in range(len(self._mf.mo_energy)):
                mo_e = self._mf.mo_energy[index]
                s = str(index) + (13 - len(str(index))) * " "
                if mo_e > 0:
                    s += " "
                s += str(mo_e)
                if mo_e > 0:
                    s += (36 - len(str(mo_e))) * " "
                else:
                    s += (37 - len(str(mo_e))) * " "
                if self._mf.mo_occ[index] > 0:
                    s += "A "
                if self._mf.mo_occ[index] > 1:
                    s += "B"
                log.info(s)



            



###

# GENERAL SUBCONVERGER/SLAVE. FUNCTIONS AS AN EVALUATOR OF TRUST AND LOCAL SOL, FROM WHICH SOL DENSITY IS CONSTRUCTED

###


class Subconverger:
    '''
    Subconverger class used in M3SOSCF. This class calculates local solutions via a local NR step, which are then returned to the Master M3SOSCF class together with a trust value and the local gradient. Instances of this class should not be created in the code and interaction with the code herein should only be performed via the superlevel M3SOSCF instance.

    Arguments:
        lc: M3SOSCF
            Master M3SOSCF class
    '''


    _m3 = None
    _moCoeffs = None
    _energy = 0.0
    _basis = None
    _hess = None
    _newton = None
    _trustScale = 0.0

    def __init__(self, lc):
        self._m3 = lc
        #self._basis = sigutils.getCanonicalBasis(len(lc._currentDm))
        self._hess = None
       
        molv = self._m3._mf.mol.verbose
        self._m3._mf.mol.verbose = 0
        self._newton = newton_ah.newton(self._m3._mf)
        self._m3._mf.mol.verbose = molv
        self._newton.max_cycle = 1
        self._newton.verbose = 0
        self._trustScale = 10**-4

        self._newton.max_stepsize = self._m3._nr_stepsize

    def getLocalSolAndTrust(self, h1e=None, s1e=None):
        '''
        This method is directly invoked by the master M3SOSCF. It solves the local NR step from the previously assigned base MO coefficients and returns the solution as well as a trust and the gradient.

        Arguments:
            h1e: 2D array, optional, default None
                Core Hamiltonian. Inclusion not necessary, but improves performance
            s1e: 2D array, optional, default None
                Overlap matrix. Inclusion not necessary, but improves performance

        Returns:
            sol: 2D array
                local solution as an anti-hermitian MO rotation matrix.
            egrad: 1D array
                local gradient calculated at the base MO coefficients and used in the NR step as a compressed vector that can be expanded to an anti-hermitian matrix via contraction with the canonical basis.
            trust: float
                Trust value of the local solution, always between 0 and 1. 0 indicates that the solution is infinitely far away and 1 indicates perfect convergence.
        '''

        old_dm = self._m3._mf.make_rdm1(self._moCoeffs, self._m3._mf.mo_occ) 
        esol, converged = self.solveForLocalSol()
        new_dm = self._m3._mf.make_rdm1(esol, self._m3._mf.mo_occ)

        trust = self.getTrust(old_dm, new_dm, converged)
       
        #sol = numpy.einsum('kij,k->ij', self._basis, esol)
        sol = esol
        # sol, grad, trust
        return sol, trust


    def solveForLocalSol(self):
        '''
        This method directly solves the NR step. In the current implementation, this is performed via an SVD transformation: An SVD is performed on the hessian, the gradient is transformed into the SVD basis and the individual, uncoupled linear equations are solved. If the hessian singular value is below a threshold of its maximum, this component of the solution is set to 0 to avoid singularities.

        Arguments:
            hess: 2D array
                Electronic Hessian
            grad: 1D array
                Electronic Gradient

        Returns:
            localSol: 1D array
                Local Solution to the NR step.

        '''

        mo_occ = self._m3._mf.mo_occ
        self._newton.kernel(mo_coeff=numpy.copy(self._moCoeffs), mo_occ=mo_occ)
        new_mo_coeffs = self._newton.mo_coeff

        return new_mo_coeffs, self._newton.converged

    def setMoCoeffs(self, moCoeffs):
        '''
        This method should be used for overriding the MO coefficients the subconverger uses as a basis for the NR step.

        Arguments:
            moCoeffs: 2D array
                New MO coefficients
        '''
        self._moCoeffs = moCoeffs

    def getMoCoeffs(self):
        '''
        Returns the current MO coefficients used as a basis for the NR step. Se getLocalSolAndTrust for the solution to the NR step.

        Returns:
            moCoeffs: 2D array
                Stored MO coefficients
        '''
        return self._moCoeffs

    def setEnergy(self, energy):
        '''
        Determine the energy that is possibly used in trust calculations. This is typically the energy of the last step to determine whether a decrease in energy has happened.

        Arguments:
            energy: float
                New energy
        '''
        self._energy = energy

    def elec(self, x, h1e=None):
        '''
        Shorthand method for evaluating the electronic energy.

        Arguments:
            x: 1D array
                MO Rotation coefficients in the given basis relative to the stored base MO coefficients at which the electronic energy should be evaluated.
            h1e: 2D array, optional, default None
                Core Hamiltonian. Inclusion not necessary, but improves performance
        '''
        

        morotation = sigutils.vectorToMatrix(x)
        localdm = self._moCoeffs @ morotation @ self._occs @ morotation.conj().T @ self._moCoeffs.conj().T

        return self._m3._mf.energy_elec(dm=localdm, h1e=h1e)[0]


    def getTrust(self, dm0, dm1, converged):
        '''
        Calculates the trust of a given solution from the solution distance, gradient and difference in energy.

        Arguments:
            sol: 1D array
                local solution in the canonical basis
            grad: 1D array
                electronic gradient in the canonical basis
            denergy: float
                difference in energy to the last step, i. e. E(now) - E(last)

        Returns:
            trust: float
                The trust of the given solution.
        '''

        #if converged:
        #    return 1.0

        e1 = self._m3._mf.energy_elec(dm1)[0]
        e0 = self._m3._mf.energy_elec(dm0)[0]
        denergy = e1 - e0

        l = numpy.linalg.norm(dm1-dm0) * self._trustScale
        return 1.0 / (l + 1.0)**2 * self.auxilliaryXi(denergy)


    def auxilliaryXi(self, x):
        '''
        Auxilliary function that is used to define the effects of energy difference on the trust.

        Arguments:
            x: float
                function input value (here: the difference in energy)
        
        Returns:
            y: float
                function output value (here: scalar effect on trust)
        '''
        if x <= 0:
            return 1
        return numpy.e**(-x)



###

# Energy Selective M3 Subconverger

###


class ESM3_Subconverger(Subconverger):

    _minenergy = -10**308
    _maxenergy = 10**308

    def __init__(self, lc, minenergy, maxenergy):
        super().__init__(lc)
        self._minenergy = minenergy
        self._maxenergy = maxenergy
        self._trustScale = 0.1

    def resetMinEnergy(self):
        self._minenergy = -10**308

    def resetMaxEnergy(self):
        self._maxenergy = 10**308


    def getTrust(self, dm0, dm1, converged):
        t1 = super().getTrust(dm0, dm1, converged)
        if converged:
            log.info("NEWTON RAPHSON CONVERGED")
            return 1.0
        e1 = self._m3._mf.energy_elec(dm1)[0]

        t1 *= self.auxilliaryXi(e1-self._maxenergy)
        t1 *= self.auxilliaryXi(self._minenergy-e1)

        return t1







###

# Manages Reassignment of Subconvergers via Trust densities during the SCF procedures

###



class SubconvergerReassigmentManager:
    '''
    This class regulates the reassignment of subconvergers after each iteration. If a subconverger is either redundant or gives a solution with a low trust, it is reassigned to another place in the electronic phase space where it generates more valuable local solutions.

    Arguments: 
        lc: M3SOSCF
            master M3SOSCF instance
    '''


    _m3 = None
    _alpha = 5.0
    _memScale = 0.2
    _trustScaleRange = None

    def __init__(self, lc):
        self._m3 = lc
        self._trustScaleRange = (0.01, 0.2, 8)
        self._alpha = 1.0 / self._trustScaleRange[1]


    def generateNewShifts(self, trusts, sols, total, cursor, log):
        '''
        This method is directly invoked by the master M3SOSCF class at the start of each SCF iteration to generate new, useful positions for the reassigned subconvergers.

        Arguments:
            trusts: 1D array
                Array of trusts of each subconverger. trusts[i] belongs to the same subconverger as sols[i]
            sols: 3D array
                Array of the current MO coefficents of each subconverger. sols[i] belongs to the same subconverger as trusts[i]
            total: int
                Total number of subconvergers to be reassigned, therefore total number of new shifts that need to be generated

        Returns:
            shifts: 3D array
                Array of new MO coefficient for each subconverger that is to be reassigned. The shape of shifts is ( total, n, n )

        '''

        maxTrust = numpy.max(trusts.flatten())
        self._alpha = 1.0 / ((self._trustScaleRange[1] - self._trustScaleRange[0]) * (1 - maxTrust)**(self._trustScaleRange[2]) + self._trustScaleRange[0])

        log.info("Current Trust Scaling: " + str(self._alpha))

        #basis = sigutils.getCanonicalBasis(len(self._m3._currentDm[0]))
        dim = self._m3.getDegreesOfFreedom()

        for i in range(len(trusts)):
            p = cursor - i
            p %= len(trusts)
            if p < 0:
                p += len(trusts)
            trusts[i] *= self._memScale**p

        trusts = self.flattenTrustMOArray(trusts)
        sols = self.flattenTrustMOArray(sols)


        trustmap = numpy.argsort(trusts)
        normed_trusts = trusts[trustmap] / numpy.einsum('i->', trusts)

        def inverseCDF(x):
            for i in range(len(normed_trusts)):
                if x < normed_trusts[i]:
                    return i
                x -= normed_trusts[i]
            return len(normed_trusts)-1


        selected_indices = numpy.zeros(total, dtype=numpy.int32)

        for i in range(len(selected_indices)):
            rand = numpy.random.random(1)
            selected_indices[i] = trustmap[inverseCDF(rand)]

        # generate Points in each trust region

        if self._m3._method == 'uhf' or self._m3._method == 'uks':
            shifts = numpy.zeros((total, 2, len(self._m3._currentDm[0]), len(self._m3._currentDm[0])))
        else:
            shifts = numpy.zeros((total, len(self._m3._currentDm[0]), len(self._m3._currentDm[0])))

        for i in range(len(selected_indices)):
            if self._m3._method == 'uhf' or self._m3._method == 'uks':
                p_a = self.genTrustRegionPoints(trusts[selected_indices[i]], 1)[0]
                p_b = self.genTrustRegionPoints(trusts[selected_indices[i]], 1)[0]
                shifts[i,0] = sols[selected_indices[i],0] @ scipy.linalg.expm(sigutils.vectorToMatrix(p_a))
                shifts[i,1] = sols[selected_indices[i],1] @ scipy.linalg.expm(sigutils.vectorToMatrix(p_b))
            else:
                p = self.genTrustRegionPoints(trusts[selected_indices[i]], 1)[0]
                shifts[i] = sols[selected_indices[i]] @ scipy.linalg.expm(sigutils.vectorToMatrix(p))



        # return 

        return shifts

    def flattenTrustMOArray(self, array):
        if len(array) == 1:
            return array[0]
        if len(array.shape) == 2: # Trust array
            return array.flatten()
        else:
            molen = len(self._m3._currentDm[0])
            farray = numpy.zeros((len(array)*len(array[0]), molen, molen))
            for i in range(len(array)):
                for j in range(len(array[0])):
                    farray[i*len(array[0])+j] = array[i,j]

            return farray

    def genSpherePoint(self):
        '''
        This method generates random points on any dimensional unit sphere via the box inflation algorithm. This results in approximately uniform point distribution, although slight accumulation at the projected corners is possible.

        Returns:
            point: 1D array
                Random point on a sphere
        '''
        dim = self._m3.getDegreesOfFreedom()

        capt_dim = numpy.random.randint(0, dim, 1)[0]
        capt_sign = numpy.random.randint(0, 1, 1)[0] * 2 - 1

        sub_point = numpy.random.random(dim-1)

        index_correct = 0
        point = numpy.zeros(dim)

        for i in range(dim):
            if i == int(capt_dim):
                point[i] = capt_sign
                index_correct = 1
                continue
            point[i] = sub_point[i-index_correct]

        point /= numpy.linalg.norm(point)
        
        return point

    def genSpherePoints(self, num):
        '''
        Generate multiple random points on any dimensional sphere. See genSpherePoint for futher details.

        Arguments:
            num: int
                number of random points to be generated

        Returns:
            points: 2D array
                array of random points on the sphere.
        '''
        points = numpy.zeros((num, self._m3.getDegreesOfFreedom()))

        for i in range(len(points)):
            points[i] = self.genSpherePoint()

        return points

    def genTrustRegionPoints(self, trust, num):
        '''
        Generates random points in a specific trust region.

        Arguments:
            trust: float
                The trust of the trust region, in which the points should be generated
            num: int
                The number of points to be generated

        Returns:
            points: 2D array
                The random points generated
        '''
        dim = self._m3.getDegreesOfFreedom()
        #norm = 2 * 3.141592**((dim + 1) / 2.0) * trust * math.factorial(int(dim + 1)) / (scipy.special.gamma((dim + 1) / 2.0) * (self._alpha * trust)**(dim + 2))
        norm = 1
        #log.info("Norm: " + str(norm))
        def inverseCDF(x):
            # Exponential Distribution
            return numpy.log(- norm * (x - 1)) / (- self._alpha * trust)
            # Gaussian Distribution
            #return scipy.special.erfinv(x) / (self._alpha * trust)


        radii = inverseCDF(numpy.random.rand(num))

        spoints = self.genSpherePoints(num)
        dpoints = numpy.zeros(spoints.shape)

        for i in range(len(dpoints)):
            dpoints[i] = spoints[i] * radii[i]

        return dpoints






