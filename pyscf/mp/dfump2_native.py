#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
native implementation of DF-MP2/RI-MP2 with a UHF reference
'''

import numpy as np
import scipy

from pyscf import lib
from pyscf import scf
from pyscf import df
from pyscf.scf import ucphf
from pyscf.mp.dfmp2_native import DFRMP2, ints3c_cholesky, orbgrad_from_Gamma


class DFUMP2(DFRMP2):
    '''
    native implementation of DF-MP2/RI-MP2 with a UHF reference
    '''

    def __init__(self, mf, frozen=None, auxbasis=None):
        '''
        Args:
            mf : UHF instance
            frozen : number of frozen orbitals or list of frozen orbitals
            auxbasis : name of auxiliary basis set, otherwise determined automatically
        '''

        if not isinstance(mf, scf.uhf.UHF):
            raise TypeError('Class initialization with non-UHF object')

        # UHF quantities are stored as numpy arrays
        self.mo_coeff = np.array(mf.mo_coeff)
        self.mo_energy = np.array(mf.mo_energy)
        self.nocc = np.array([np.count_nonzero(mf.mo_occ[0]), np.count_nonzero(mf.mo_occ[1])])
        # UHF MO coefficient matrix shape: (2, number of AOs, number of MOs)
        self.nmo = self.mo_coeff.shape[2]
        self.e_scf = mf.e_tot
        self._scf = mf

        # Process the frozen core option correctly as either an integer or two lists (alpha, beta).
        # self.frozen_mask sets a flag for each orbital if it is frozen (True) or not (False).
        # Only occupied orbitals can be frozen.
        self.frozen_mask = np.zeros((2, self.nmo), dtype=bool)
        if frozen is None:
            pass
        elif lib.isinteger(frozen):
            if frozen > min(self.nocc):
                raise ValueError('only occupied orbitals can be frozen')
            self.frozen_mask[:, :frozen] = True
        else:
            try:
                if len(frozen) != 2:
                    raise ValueError
                for s in 0, 1:
                    if not lib.isintsequence(frozen[s]):
                        raise TypeError
            except (TypeError, ValueError):
                raise TypeError('frozen must be an integer or two integer lists')
            if len(frozen[0]) != len(frozen[1]):
                raise ValueError('frozen orbital lists not of equal length')
            for s in 0, 1:
                self.frozen_mask[s, frozen[s]] = True

        # mask for occupied orbitals that are not frozen
        self.occ_mask = np.zeros((2, self.nmo), dtype=bool)
        for s in 0, 1:
            self.occ_mask[s, :self.nocc[s]] = True
        self.occ_mask[self.frozen_mask] = False

        self.mol = mf.mol
        if not auxbasis:
            auxbasis = df.make_auxbasis(self.mol, mp2fit=True)
        self.auxmol = df.make_auxmol(self.mol, auxbasis)

        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = self.mol.max_memory

        # _intsfile will be a list with two elements for the alpha and beta integrals
        self._intsfile = []
        self.e_corr = None

        # Spin component scaling factors
        self.ps = 1.0
        self.pt = 1.0

        self.cphf_max_cycle = 100
        self.cphf_tol = mf.conv_tol

    def dump_flags(self, logger=None):
        '''
        Prints selected information.

        Args:
            logger : Logger object
        '''
        if not logger:
            logger = lib.logger.new_logger(self)
        logger.info('')
        logger.info('******** {0:s} ********'.format(repr(self.__class__)))
        logger.info('nmo = {0:d}'.format(self.nmo))
        logger.info('nocc = {0:d}, {1:d}'.format(self.nocc[0], self.nocc[1]))
        nfrozen = np.count_nonzero(self.frozen_mask[0])
        logger.info('no. of frozen = {0:d}'.format(nfrozen))
        frozen_tmp = np.arange(self.nmo)[self.frozen_mask[0]]
        logger.debug('frozen (alpha) = {0}'.format(frozen_tmp))
        frozen_tmp = np.arange(self.nmo)[self.frozen_mask[1]]
        logger.debug('frozen (beta) = {0}'.format(frozen_tmp))
        logger.info('basis = {0:s}'.format(repr(self.mol.basis)))
        logger.info('auxbasis = {0:s}'.format(repr(self.auxmol.basis)))
        logger.info('max_memory = {0:.1f} MB (current use {1:.1f} MB)'.
                    format(self.max_memory, lib.current_memory()[0]))

    def calculate_energy(self):
        '''
        Calculates the MP2 correlation energy.
        '''
        if not self.has_ints:
            self.calculate_integrals_()

        logger = lib.logger.new_logger(self)
        logger.info('')
        logger.info('Calculating DF-MP2 energy')
        self.e_corr = emp2_uhf(self._intsfile, self.mo_energy, self.frozen_mask,
                               logger, ps=self.ps, pt=self.pt)
        logger.note('DF-MP2 correlation energy: {0:.14f}'.format(self.e_corr))
        return self.e_corr

    def make_rdm1(self, relaxed=False, ao_repr=False):
        '''
        Calculates the MP2 1-RDM.
        - The relaxed density matrix can be used to calculate properties of systems
          for which MP2 is well-behaved.
        - The unrelaxed density is less suited to calculate properties accurately,
          but it can be used to calculate CASSCF starting orbitals.

        Args:
            relaxed : relaxed density if True, unrelaxed density if False
            ao_repr : density in AO or in MO basis

        Returns:
            the 1-RDM
        '''
        logger = lib.logger.new_logger(self)
        if relaxed:
            logger.info('')
            logger.info('DF-MP2 relaxed density calculation')
        else:
            logger.info('')
            logger.info('DF-MP2 unrelaxed density calculation')
        rdm1_mo = make_rdm1(self, relaxed, logger)
        if ao_repr:
            return lib.einsum('sxp,spq,syq->sxy', self.mo_coeff, rdm1_mo, self.mo_coeff)
        else:
            return rdm1_mo

    def make_natorbs(self, rdm1_mo=None, relaxed=False):
        '''
        Calculate natural orbitals.
        Note: the most occupied orbitals come first (left)
              and the least occupied orbitals last (right).

        Args:
            rdm1_mo : 1-RDM in MO basis
                      the function calculates a density matrix if none is provided
            relaxed : calculated relaxed or unrelaxed density matrix

        Returns:
            natural occupation numbers, natural orbitals
        '''
        if rdm1_mo is None:
            dm = self.make_rdm1(ao_repr=False, relaxed=relaxed)
        elif isinstance(rdm1_mo, np.ndarray):
            dm = rdm1_mo
        else:
            raise TypeError('rdm1_mo must be a 3-D array')

        # Transform the beta component to the alpha basis and sum both together.
        SAO = self.mol.intor_symmetric('int1e_ovlp')
        Sab = lib.einsum('xp,xy,yq->pq', self.mo_coeff[0, :, :], SAO, self.mo_coeff[1, :, :])
        rdm1_abas = dm[0, :, :] + lib.einsum('pr,rs,qs->pq', Sab, dm[1, :, :], Sab)

        # Diagonalize the spin-traced 1-RDM in alpha basis to get the natural orbitals.
        eigval, eigvec = np.linalg.eigh(rdm1_abas)
        natocc = np.flip(eigval)
        natorb = lib.dot(self.mo_coeff[0, :, :], np.fliplr(eigvec))
        return natocc, natorb

    def calculate_integrals_(self):
        '''
        Calculates the three center integrals for MP2.
        '''
        intsfile = []
        logger = lib.logger.new_logger(self)
        logger.info('')
        logger.info('Calculating integrals')
        for s in [0, 1]:
            Co = self.mo_coeff[s][:, self.occ_mask[s]]
            Cv = self.mo_coeff[s][:, self.nocc[s]:]
            f = ints3c_cholesky(self.mol, self.auxmol, Co, Cv, self.max_memory, logger)
            intsfile.append(f)
        self._intsfile = intsfile
        logger.info('Stored in files:\n{0:s}\n{1:s}'.
                    format(self._intsfile[0].filename, self._intsfile[1].filename))

    def delete(self):
        '''
        Delete the temporary file(s).
        '''
        self._intsfile = []

    def nuc_grad_method(self):
        raise NotImplementedError


MP2 = UMP2 = DFMP2 = DFUMP2


class SCSDFUMP2(DFUMP2):
    '''
    UHF-DF-MP2 with spin-component scaling
    S. Grimme, J. Chem. Phys. 118 (2003), 9095
    https://doi.org/10.1063/1.1569242
    '''

    def __init__(self, mf, ps=6/5, pt=1/3, *args, **kwargs):
        '''
        mf : UHF instance
        ps : opposite-spin (singlet) scaling factor
        pt : same-spin (triplet) scaling factor
        '''
        super().__init__(mf, *args, **kwargs)
        self.ps = ps
        self.pt = pt

    def dump_flags(self, logger=None):
        if not logger:
            logger = lib.logger.new_logger(self)
        super().dump_flags(logger=logger)
        logger.info('pt(scs) = {0:.6f}'.format(self.pt))
        logger.info('ps(scs) = {0:.6f}'.format(self.ps))


SCSMP2 = SCSUMP2 = SCSDFMP2 = SCSDFUMP2


def emp2_uhf(intsfiles, mo_energy, frozen_mask, logger, ps=1.0, pt=1.0):
    '''
    Calculates the DF-MP2 energy with an UHF reference.

    Args:
        intsfiles : contains the three center integrals in MO basis
        mo_energy : energies of the molecular orbitals
        frozen_mask : boolean mask for frozen orbitals
        logger : Logger instance
        ps : SCS factor for opposite-spin contributions
        pt : SCS factor for same-spin contributions

    Returns:
        the MP2 correlation energy
    '''
    ints_a = intsfiles[0]['ints_cholesky']
    ints_b = intsfiles[1]['ints_cholesky']
    nocc_act = np.array([ints_a.shape[0], ints_b.shape[0]])
    nfrozen = np.count_nonzero(frozen_mask[0])
    if np.count_nonzero(frozen_mask[1]) != nfrozen:
        raise ValueError('number of frozen alpha and beta orbitals differs')
    nocc = nocc_act + nfrozen
    nvirt = np.array([ints_a.shape[2], ints_b.shape[2]])

    logger.debug('    UHF-DF-MP2 energy routine')
    logger.debug('    Occupied orbitals: {0:d}, {1:d}'.format(nocc[0], nocc[1]))
    logger.debug('    Virtual orbitals:  {0:d}, {1:d}'.format(nvirt[0], nvirt[1]))
    logger.debug('    Frozen orbitals:   {0:d}'.format(nfrozen))
    logger.debug('    Integrals (alpha) from file: {0:s}'.format(intsfiles[0].filename))
    logger.debug('    Integrals (beta)  from file: {0:s}'.format(intsfiles[1].filename))

    mo_energy_masked = mo_energy[~frozen_mask].reshape((2, -1))

    energy_total = 0.0

    # loop over spins to calculate same-spin energies
    for s in 0, 1:
        energy_contrib = 0.0
        if s == 0:
            logger.debug('    alpha-alpha pairs')
            ints = ints_a
        else:
            logger.debug('    beta-beta pairs')
            ints = ints_b
        # precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for the denominator
        Eab = np.zeros((nvirt[s], nvirt[s]))
        for a in range(nvirt[s]):
            Eab[a, :] += mo_energy[s, nocc[s]+a]
            Eab[:, a] += mo_energy[s, nocc[s]+a]
        # loop over j < i
        for i in range(nocc_act[s]):
            ints3c_ia = ints[i, :, :]
            for j in range(i):
                ints3c_jb = ints[j, :, :]
                Kab = lib.dot(ints3c_ia.T, ints3c_jb)
                DE = mo_energy_masked[s, i] + mo_energy_masked[s, j] - Eab
                Tab = (Kab - Kab.T) / DE
                energy_contrib += pt * lib.einsum('ab,ab', Tab, Kab)
        logger.debug('      E = {0:.14f}'.format(energy_contrib))
        energy_total += energy_contrib

    # opposite-spin energy
    logger.debug('    alpha-beta pairs')
    # precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for the denominator
    Eab = np.zeros((nvirt[0], nvirt[1]))
    for a in range(nvirt[0]):
        Eab[a, :] += mo_energy[0, nocc[0]+a]
    for b in range(nvirt[1]):
        Eab[:, b] += mo_energy[1, nocc[1]+b]
    # loop over i(alpha), j(beta)
    energy_contrib = 0.0
    for i in range(nocc_act[0]):
        ints3c_ia = ints_a[i, :, :]
        for j in range(nocc_act[1]):
            ints3c_jb = ints_b[j, :, :]
            Kab = lib.dot(ints3c_ia.T, ints3c_jb)
            DE = mo_energy_masked[0, i] + mo_energy_masked[1, j] - Eab
            Tab = Kab / DE
            energy_contrib += ps * lib.einsum('ab,ab', Tab, Kab)
    logger.debug('      E = {0:.14f}'.format(energy_contrib))
    energy_total += energy_contrib

    logger.debug('    DF-MP2 correlation energy: {0:.14f}'.format(energy_total))
    return energy_total


def make_rdm1(mp2, relaxed, logger=None):
    '''
    Calculates the unrelaxed or relaxed MP2 density matrix.

    Args:
        mp2 : DFUMP2 instance
        relaxed : relaxed density if True, unrelaxed density if False
        logger : Logger instance

    Returns:
        the 1-RDM in MO basis
    '''
    if not mp2.has_ints:
        mp2.calculate_integrals_()

    # Calculate the unrelaxed 1-RDM.
    if logger is None:
        logger = lib.logger.new_logger(mp2)
    rdm1, GammaFile = \
        ump2_densities_contribs(mp2._intsfile, mp2.mo_energy, mp2.frozen_mask, mp2.max_memory,
                                logger, calcGamma=relaxed, auxmol=mp2.auxmol, ps=mp2.ps, pt=mp2.pt)

    if relaxed:

        Lvo = [None, None]
        for s, sstr in [(0, 'alpha'), (1, 'beta')]:

            # right-hand side for the CPHF equation
            Gamma = GammaFile['Gamma_'+sstr]
            Lvo[s], Lfo_s = \
                orbgrad_from_Gamma(mp2.mol, mp2.auxmol, Gamma, mp2.mo_coeff[s], mp2.frozen_mask[s],
                                   mp2.max_memory, logger)

            # frozen core orbital relaxation contribution
            frozen_list = np.arange(mp2.nmo)[mp2.frozen_mask[s]]
            for fm, f in enumerate(frozen_list):
                for i in np.arange(mp2.nmo)[mp2.occ_mask[s]]:
                    zfo = Lfo_s[fm, i] / (mp2.mo_energy[s, f] - mp2.mo_energy[s, i])
                    rdm1[s, f, i] += 0.5 * zfo
                    rdm1[s, i, f] += 0.5 * zfo

        # Fock response
        Lvo_a, Lvo_b = fock_response_uhf(mp2._scf, rdm1)
        Lvo[0] -= Lvo_a
        Lvo[1] -= Lvo_b
        # solving the CPHF equations
        minusLvo = [-Lvo[0], -Lvo[1]]
        zvo = solve_cphf_uhf(mp2._scf, minusLvo, mp2.cphf_max_cycle, mp2.cphf_tol, logger)

        # add the relaxation contribution to the density
        for s in 0, 1:
            rdm1[s, mp2.nocc[s]:, :mp2.nocc[s]] += 0.5 * zvo[s]
            rdm1[s, :mp2.nocc[s], mp2.nocc[s]:] += 0.5 * zvo[s].T

    # HF contribution
    for s in 0, 1:
        rdm1[s, :mp2.nocc[s], :mp2.nocc[s]] += np.eye(mp2.nocc[s])

    return rdm1


def ump2_densities_contribs(intsfiles, mo_energy, frozen_mask, max_memory, logger,
                            calcGamma=False, auxmol=None, ps=1.0, pt=1.0):
    '''
    Calculates the unrelaxed DF-MP2 density matrix contribution with a UHF reference.
    Note: this is the difference density, i.e. without HF contribution.A
    lso calculates the three-center two-particle density if requested.

    Args:
        intsfile : contains the three center integrals
        mo_energy : molecular orbital energies
        frozen_mask : boolean mask for frozen orbitals
        max_memory : memory threshold in MB
        logger : Logger instance
        calcGamma : if True, calculate 3c2e density
        auxmol : required if relaxed is True
        ps : SCS factor for opposite-spin contributions
        pt : SCS factor for same-spin contributions

    Returns:
        matrix containing the 1-RDM contribution, file with 3c2e density if requested
    '''
    ints = [intsfiles[s]['ints_cholesky'] for s in (0, 1)]
    nocc_act = np.array([ints[s].shape[0] for s in (0, 1)])
    naux = ints[0].shape[1]
    if ints[1].shape[1] != naux:
        raise ValueError('integrals have inconsistent aux dimensions')
    nvirt = np.array([ints[s].shape[2] for s in (0, 1)])
    nmo = mo_energy.shape[1]
    nfrozen = np.count_nonzero(frozen_mask[0])
    if np.count_nonzero(frozen_mask[0]) != nfrozen:
        raise ValueError('unequal numbers of frozen orbitals for alpha and beta')
    nocc = nfrozen + nocc_act
    if np.any(nocc + nvirt != nmo):
        raise ValueError('numbers of frozen, occupied and virtual orbitals inconsistent')

    logger.debug('    Density matrix contributions for DF-MP2')
    logger.debug('    Occupied orbitals: {0:d}, {1:d}'.format(nocc[0], nocc[1]))
    logger.debug('    Virtual orbitals:  {0:d}, {1:d}'.format(nvirt[0], nvirt[1]))
    logger.debug('    Frozen orbitals:   {0:d}'.format(nfrozen))
    logger.debug('    Three center integrals (alpha) from file: {0:s}'.format(intsfiles[0].filename))
    logger.debug('    Three center integrals (beta) from file: {0:s}'.format(intsfiles[1].filename))

    GammaFile, LT = None, None
    if calcGamma:
        if not auxmol:
            raise RuntimeError('auxmol needs to be specified for relaxed density computation')
        # create temporary file to store the two-body density Gamma
        GammaFile = lib.H5TmpFile(libver='latest')
        GammaFile.create_dataset('Gamma_alpha', (nocc_act[0], naux, nvirt[0]), dtype='f8')
        GammaFile.create_dataset('Gamma_beta', (nocc_act[1], naux, nvirt[1]), dtype='f8')
        logger.debug('    Storing 3c2e density in file: {0:s}'.format(GammaFile.filename))
        # We will need LT = L^T, where L L^T = V
        LT = scipy.linalg.cholesky(auxmol.intor('int2c2e'), lower=False)

    # We start forming P with contiguous frozen, occupied, virtual subblocks.
    P = np.zeros((2, nmo, nmo))
    mo_energy_masked = mo_energy[~frozen_mask].reshape(2, nmo-nfrozen)

    # Loop over all the spin variants
    for s1, s2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        with lib.H5TmpFile(libver='latest') as tfile:

            tiset = \
                tfile.create_dataset('amplitudes', (nocc_act[s2], nvirt[s1], nvirt[s2]), dtype='f8')

            s1_str = ('alpha', 'beta')[s1]
            s2_str = ('alpha', 'beta')[s2]
            logger.debug('    {0:s}-{1:s} pairs'.format(s1_str, s2_str))
            logger.debug('    Storing amplitudes in temporary file: {0:s}'.format(tfile.filename))

            # Precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for division with numpy.
            Eab = np.zeros((nvirt[s1], nvirt[s2]))
            for a in range(nvirt[s1]):
                Eab[a, :] += mo_energy[s1, nocc[s1]+a]
            for b in range(nvirt[s2]):
                Eab[:, b] += mo_energy[s2, nocc[s2]+b]

            # For each occupied spin orbital i, all amplitudes are calculated once and
            # stored on disk. The occupied 1-RDM contribution is calculated in a batched
            # algorithm. More memory -> more efficient I/O.
            # The virtual contribution to the 1-RDM is calculated in memory.
            for i in range(nocc_act[s1]):
                ints3c_ia = ints[s1][i, :, :]

                # Amplitudes T^ij_ab are calculated for a given orbital i with spin s1,
                # and all j (s2), a (s1) and b (s2). These amplitudes are stored on disk.
                for j in range(nocc_act[s2]):
                    ints3c_jb = ints[s2][j, :, :]
                    Kab = lib.dot(ints3c_ia.T, ints3c_jb)
                    DE = mo_energy_masked[s1, i] + mo_energy_masked[s2, j] - Eab
                    if s1 == s2:
                        numerator = Kab - Kab.T
                        prefactor = 0.5 * pt
                    else:
                        numerator = Kab
                        prefactor = ps
                    Tab = numerator / DE
                    tiset[j, :, :] = Tab
                    # virtual 1-RDM contribution
                    P[s1, nocc[s1]:, nocc[s1]:] += prefactor * lib.dot(Tab, Tab.T)
                del ints3c_jb, Kab, DE, numerator, Tab

                # Batches of amplitudes are read from disk to calculate the occupied
                # 1-RDM contribution.
                batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / (nocc_act[s2] * nvirt[s2] * 8))
                batchsize = min(nvirt[s1], batchsize)
                if batchsize < 1:
                    raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
                logger.debug2('    Batch size: {0:d} (of {1:d})'.format(batchsize, nvirt[s1]))
                logger.debug2('      Pij formation - MO {0:d} ({1:s}), batch size {2:d} (of {3:d})'.
                              format(i, s1_str, batchsize, nvirt[s1]))
                for astart in range(0, nvirt[s1], batchsize):
                    aend = min(astart+batchsize, nvirt[s1])
                    tbatch = tiset[:, astart:aend, :]
                    if s1 == s2:
                        prefactor = 0.5 * pt
                    else:
                        prefactor = ps
                    P[s2, nfrozen:nocc[s2], nfrozen:nocc[s2]] -= \
                        prefactor * lib.einsum('iab,jab->ij', tbatch, tbatch)
                del tbatch

                if calcGamma:
                    # This produces (P | Q)^-1 (Q | i a)
                    ints3cV1_ia = scipy.linalg.solve_triangular(LT, ints3c_ia, lower=False)
                    # Here, we construct Gamma for spin s2
                    Gamma = GammaFile['Gamma_'+s2_str]
                    # Read batches of amplitudes from disk and calculate the two-body density Gamma
                    size = nvirt[s1] * nvirt[s2] * 8 + naux * nvirt[s2] * 8
                    batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / size)
                    batchsize = min(nocc_act[s2], batchsize)
                    if batchsize < 1:
                        raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
                    logger.debug2('      Gamma ({0:s}) formation - MO {1:d} ({2:s}), batch size {3:d} (of {4:d})'.
                                  format(s2_str, i, s1_str, batchsize, nocc_act[s2]))
                    if s1 == s2:
                        prefactor = 2.0 * pt
                    else:
                        prefactor = 2.0 * ps
                    for jstart in range(0, nocc_act[s2], batchsize):
                        jend = min(jstart+batchsize, nocc_act[s2])
                        tbatch = tiset[jstart:jend, :, :]
                        # Here, we collect two-body density contributions for spin s2
                        Gbatch = Gamma[jstart:jend, :, :]
                        for jj in range(jend-jstart):
                            Tijab = tbatch[jj]
                            Gbatch[jj] += prefactor * lib.dot(ints3cV1_ia, Tijab)
                        Gamma[jstart:jend, :, :] = Gbatch
                    del tbatch, Gbatch

    # now reorder P such that the frozen orbitals correspond to frozen_mask
    for s in 0, 1:
        idx_reordered = \
            np.concatenate([np.arange(nmo)[frozen_mask[s]], np.arange(nmo)[~frozen_mask[s]]])
        P[s][idx_reordered, :] = P[s].copy()
        P[s][:, idx_reordered] = P[s].copy()

    logger.debug('    Density matrix contributions calculation finished')
    return P, GammaFile


def fock_response_uhf(mf, dm, full=True):
    '''
    Calculate the unrestricted Fock response function for a given density matrix.

    Args:
        mf : UHF instance
        dm : density matrix in MO basis
        full : full MO density matrix if True, [virt. x occ., virt. x occ.] if False

    Returns:
        Fock response in MO basis. Shape: [virt. x occ., virt. x occ.]
    '''
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mf.mol.nao
    dmao = np.zeros((2, nao, nao))
    for s in 0, 1:
        if full:
            dmao[s, :, :] = lib.einsum('xp,pq,yq->xy', mo_coeff[s], dm[s], mo_coeff[s])
        else:
            Ci = mo_coeff[s][:, mo_occ[s]>0]
            Ca = mo_coeff[s][:, mo_occ[s]==0]
            dmao[s, :, :] = lib.einsum('xa,ai,yi->xy', Ca, dm[s], Ci)
    rao = mf.get_veff(dm=dmao+dmao.transpose((0, 2, 1)))
    rvo = [None, None]
    for s in 0, 1:
        Ci = mo_coeff[s][:, mo_occ[s]>0]
        Ca = mo_coeff[s][:, mo_occ[s]==0]
        rvo[s] = lib.einsum('xa,xy,yi->ai', Ca, rao[s], Ci)
    return rvo


def solve_cphf_uhf(mf, Lvo, max_cycle, tol, logger):
    '''
    Solve the CPHF equations.

    Args:
        mf : a UHF object
        Lvo : right-hand side the the response equation
        max_cycle : number of iterations for the CPHF solver
        tol : convergence tolerance for the CPHF solver
        logger : Logger object
    '''
    logger.info('Solving the CPHF response equations')
    logger.info('Max. iterations: {0:d}'.format(max_cycle))
    logger.info('Convergence tolerance: {0:.3g}'.format(tol))

    # Currently we need to make the CPHF solver somewhat more talkative to see anything at all.
    cphf_verbose = logger.verbose
    if logger.verbose == lib.logger.INFO:
        cphf_verbose = lib.logger.DEBUG

    nva, noa = Lvo[0].shape
    nvb, nob = Lvo[1].shape
    def fvind(zflat):
        za = zflat[0, :noa*nva].reshape(nva, noa)
        zb = zflat[0, -nob*nvb:].reshape(nvb, nob)
        ra, rb = fock_response_uhf(mf, [za, zb], full=False)
        rflat = np.hstack([ra.reshape((1, noa*nva)), rb.reshape((1, nob*nvb))])
        return rflat

    zvo = ucphf.solve(fvind, mf.mo_energy, mf.mo_occ, Lvo,
                      max_cycle=max_cycle, tol=tol, verbose=cphf_verbose)[0]
    logger.info('CPHF iterations finished')
    return zvo


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.spin = 2
    mol.basis = 'def2-SVP'
    mol.verbose = lib.logger.INFO
    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()

    with DFUMP2(mf) as pt:
        pt.kernel()
        natocc, _ = pt.make_natorbs()
        print()
        print(natocc)
