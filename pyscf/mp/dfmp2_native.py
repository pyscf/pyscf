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
native implementation of DF-MP2/RI-MP2 with an RHF reference
'''


import numpy as np
import scipy

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import df
from pyscf.scf import cphf


class DFRMP2(lib.StreamObject):
    '''
    native implementation of DF-MP2/RI-MP2 with an RHF reference
    '''

    def __init__(self, mf, frozen=None, auxbasis=None):
        '''
        Args:
            mf : RHF instance
            frozen : number of frozen orbitals or list of frozen orbitals
            auxbasis : name of auxiliary basis set, otherwise determined automatically
        '''

        if not isinstance(mf, scf.rhf.RHF):
            raise TypeError('Class initialization with non-RHF object')
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy
        self.nocc = np.count_nonzero(mf.mo_occ)
        self.nmo = self.mo_coeff.shape[1]
        self.e_scf = mf.e_tot
        self._scf = mf

        # Process the frozen core option correctly as an integer or a list.
        # self.frozen_mask sets a flag for each orbital if it is frozen (True) or not (False).
        # Only occupied orbitals can be frozen.
        self.frozen_mask = np.zeros(self.nmo, dtype=bool)
        if frozen is None:
            pass
        elif lib.isinteger(frozen):
            if frozen > self.nocc:
                raise ValueError('only occupied orbitals can be frozen')
            self.frozen_mask[:frozen] = True
        elif lib.isintsequence(frozen):
            if max(frozen) > self.nocc - 1:
                raise ValueError('only occupied orbitals can be frozen')
            self.frozen_mask[frozen] = True
        else:
            raise TypeError('frozen must be an integer or a list of integers')

        # mask for occupied orbitals that are not frozen
        self.occ_mask = np.zeros(self.nmo, dtype=bool)
        self.occ_mask[:self.nocc] = True
        self.occ_mask[self.frozen_mask] = False

        self.mol = mf.mol
        if not auxbasis:
            auxbasis = df.make_auxbasis(self.mol, mp2fit=True)
        self.auxmol = df.make_auxmol(self.mol, auxbasis)

        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = self.mol.max_memory

        self._intsfile = None
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
        logger.info('nocc = {0:d}'.format(self.nocc))
        nfrozen = np.count_nonzero(self.frozen_mask)
        logger.info('no. of frozen = {0:d}'.format(nfrozen))
        frozen_list = np.arange(self.nmo)[self.frozen_mask]
        logger.debug('frozen = {0}'.format(frozen_list))
        logger.info('basis = {0:s}'.format(repr(self.mol.basis)))
        logger.info('auxbasis = {0:s}'.format(repr(self.auxmol.basis)))
        logger.info('max_memory = {0:.1f} MB (current use {1:.1f} MB)'.
                    format(self.max_memory, lib.current_memory()[0]))

    @property
    def e_tot(self):
        '''
        total energy (SCF + MP2)
        '''
        return self.e_scf + self.e_corr

    def calculate_energy(self):
        '''
        Calculates the MP2 correlation energy.
        '''
        if not self.has_ints:
            self.calculate_integrals_()

        logger = lib.logger.new_logger(self)
        logger.info('')
        logger.info('Calculating DF-MP2 energy')
        self.e_corr = emp2_rhf(self._intsfile, self.mo_energy, self.frozen_mask,
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
            rdm1_ao = lib.einsum('xp,pq,yq->xy', self.mo_coeff, rdm1_mo, self.mo_coeff)
            return rdm1_ao
        else:
            return rdm1_mo

    def make_rdm1_unrelaxed(self, ao_repr=False):
        return self.make_rdm1(relaxed=False, ao_repr=ao_repr)

    def make_rdm1_relaxed(self, ao_repr=False):
        return self.make_rdm1(relaxed=True, ao_repr=ao_repr)

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
            dm = self.make_rdm1(relaxed=relaxed, ao_repr=False)
        elif isinstance(rdm1_mo, np.ndarray):
            dm = rdm1_mo
        else:
            raise TypeError('rdm1_mo must be a 2-D array')

        eigval, eigvec = np.linalg.eigh(dm)
        natocc = np.flip(eigval)
        natorb = lib.dot(self.mo_coeff, np.fliplr(eigvec))
        return natocc, natorb

    @property
    def has_ints(self):
        return bool(self._intsfile)

    def calculate_integrals_(self):
        '''
        Calculates the three center integrals for MP2.
        '''
        Co = self.mo_coeff[:, self.occ_mask]
        Cv = self.mo_coeff[:, self.nocc:]
        logger = lib.logger.new_logger(self)
        logger.info('')
        logger.info('Calculating integrals')
        self._intsfile = ints3c_cholesky(self.mol, self.auxmol, Co, Cv, self.max_memory, logger)
        logger.info('Stored in file: {0:s}'.format(self._intsfile.filename))

    def delete(self):
        '''
        Delete the temporary file(s).
        '''
        self._intsfile = None

    # The class can be used with a context manager (with ... as ...:).
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.delete()

    def kernel(self):
        '''
        Alias for the MP2 energy calculation.
        Does not need to be called to calculate the 1-RDM only.
        '''
        self.dump_flags()
        return self.calculate_energy()

    def nuc_grad_method(self):
        raise NotImplementedError


MP2 = RMP2 = DFMP2 = DFRMP2


class SCSDFRMP2(DFRMP2):
    '''
    RHF-DF-MP2 with spin-component scaling
    S. Grimme, J. Chem. Phys. 118 (2003), 9095
    https://doi.org/10.1063/1.1569242
    '''

    def __init__(self, mf, ps=6/5, pt=1/3, *args, **kwargs):
        '''
        mf : RHF instance
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


SCSMP2 = SCSRMP2 = SCSDFMP2 = SCSDFRMP2


def ints3c_cholesky(mol, auxmol, mo_coeff1, mo_coeff2, max_memory, logger):
    '''
    Calculate the three center electron repulsion integrals in MO basis
    multiplied with the Cholesky factor of the inverse Coulomb metric matrix.
    Only integrals in MO basis are stored on disk; integral-direct with regard to AO integrals.

    Args:
        mol : Mole instance
        auxmol : Mole instance with auxiliary basis
        mo_coeff1 : MO coefficient matrix for the leading MO index, typically occupied
        mo_coeff2 : MO coefficient matrix for the secondary MO index, typically virtual
        max_memory : memory threshold in MB
        logger : Logger instance

    Returns:
        A HDF5 temporary file containing the integrals in the dataset "ints_cholesky".
        Indexing order: [mo1, aux, mo2]
    '''
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,
                                 auxmol._atm, auxmol._bas, auxmol._env)
    nmo1 = mo_coeff1.shape[1]
    nmo2 = mo_coeff2.shape[1]
    nauxfcns = auxmol.nao

    logger.debug('    DF integral transformation')
    logger.debug('    MO dimensions: {0:d} x {1:d}'.format(nmo1, nmo2))
    logger.debug('    Aux functions: {0:d}'.format(nauxfcns))

    intsfile_cho = lib.H5TmpFile(libver='latest')
    with lib.H5TmpFile(libver='latest') as intsfile_tmp:

        logger.debug('    Calculating three center integrals in MO basis.')
        logger.debug('    Temporary file: {0:s}'.format(intsfile_tmp.filename))

        intor = mol._add_suffix('int3c2e')
        logger.debug2('    intor = {0:s}'.format(intor))

        # Loop over shells of auxiliary functions.
        # AO integrals are calculated in memory and directly transformed to MO basis.
        ints_3c = intsfile_tmp.create_dataset('ints_3c', (nauxfcns, nmo1, nmo2), dtype='f8')
        aux_ctr = 0
        for auxsh in range(auxmol.nbas):
            # needs to follow the convention (AO, AO | Aux)
            shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+auxsh, mol.nbas+auxsh+1)
            # AO integrals
            aoints_auxshell = gto.getints(intor, atm, bas, env, shls_slice)
            # loop over aux functions
            for m in range(aoints_auxshell.shape[2]):
                aoints = aoints_auxshell[:, :, m]
                if nmo1 <= nmo2:
                    moints = lib.dot(lib.dot(mo_coeff1.T, aoints), mo_coeff2)
                else:
                    moints = lib.dot(mo_coeff1.T, lib.dot(aoints, mo_coeff2))
                ints_3c[aux_ctr, :, :] = moints
                aux_ctr += 1

        logger.debug('    Calculating fitted three center integrals.')
        logger.debug('    Storage file: {0:s}'.format(intsfile_cho.filename))

        # Typically we need the matrix for a specific occupied MO i in MP2.
        # => i is the leading index for optimal I/O.
        ints_cholesky = intsfile_cho.create_dataset('ints_cholesky', (nmo1, nauxfcns, nmo2), dtype='f8')

        # (P | Q) matrix
        Vmat = auxmol.intor('int2c2e')

        # L L^T = V    <->    L^-T L^-1 = V^-1
        L = scipy.linalg.cholesky(Vmat, lower=True)

        # Buffer only serves to reduce the read operations of the second index in ints_3c
        # (I/O overhead increases from first to third index).
        bufsize = int((max_memory - lib.current_memory()[0]) * 1e6 / (nauxfcns * nmo2 * 8))
        if bufsize < 1:
            raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
        bufsize = max(1, min(nmo1, bufsize))
        logger.debug('    Batch size: {0:d} (of {1:d})'.format(bufsize, nmo1))

        # In batches:
        # - Read integrals from the temporary file.
        # - Instead of multiplying with L^-1, solve linear equation system.
        # - Store the "fitted" integrals in the integrals file.
        for istart in range(0, nmo1, bufsize):
            iend = min(istart+bufsize, nmo1)
            intsbuf = ints_3c[:, istart:iend, :]
            for i in range(istart, iend):
                ints_cholesky[i, :, :] = scipy.linalg.solve_triangular(L, intsbuf[:, i-istart, :], lower=True)

    logger.debug('    DF transformation finished')
    return intsfile_cho


def emp2_rhf(intsfile, mo_energy, frozen_mask, logger, ps=1.0, pt=1.0):
    '''
    Calculates the DF-MP2 energy with an RHF reference.

    Args:
        intsfile : contains the three center integrals in MO basis
        mo_energy : energies of the molecular orbitals
        frozen_mask : boolean mask for frozen orbitals
        logger : Logger instance
        ps : SCS factor for opposite-spin contributions
        pt : SCS factor for same-spin contributions

    Returns:
        the MP2 correlation energy
    '''
    ints = intsfile['ints_cholesky']
    nocc_act, _, nvirt = ints.shape
    nfrozen = np.count_nonzero(frozen_mask)
    nocc = nocc_act + nfrozen

    logger.debug('    RHF-DF-MP2 energy routine')
    logger.debug('    Occupied orbitals: {0:d}'.format(nocc))
    logger.debug('    Virtual orbitals:  {0:d}'.format(nvirt))
    logger.debug('    Frozen orbitals:   {0:d}'.format(nfrozen))
    logger.debug('    Integrals from file: {0:s}'.format(intsfile.filename))

    mo_energy_masked = mo_energy[~frozen_mask]

    # Somewhat awkward workaround to perform division in the MP2 energy expression
    # through numpy routines. We precompute Eab[a, b] = mo_energy[a] + mo_energy[b].
    Eab = np.zeros((nvirt, nvirt))
    for a in range(nvirt):
        Eab[a, :] += mo_energy[nocc + a]
        Eab[:, a] += mo_energy[nocc + a]

    energy = 0.0
    for i in range(nocc_act):
        ints3c_ia = ints[i, :, :]
        # contributions for occupied orbitals j < i
        for j in range(i):
            ints3c_jb = ints[j, :, :]
            Kab = lib.dot(ints3c_ia.T, ints3c_jb)
            DE = mo_energy_masked[i] + mo_energy_masked[j] - Eab
            Tab = Kab / DE
            energy += 2.0 * (ps + pt) * lib.einsum('ab,ab', Tab, Kab)
            energy -= 2.0 * pt * lib.einsum('ab,ba', Tab, Kab)
        # contribution for j == i
        Kab = lib.dot(ints3c_ia.T, ints3c_ia)
        DE = 2.0 * mo_energy_masked[i] - Eab
        Tab = Kab / DE
        energy += ps * lib.einsum('ab,ab', Tab, Kab)

    logger.debug('    DF-MP2 correlation energy: {0:.14f}'.format(energy))
    return energy


def make_rdm1(mp2, relaxed, logger=None):
    '''
    Calculates the unrelaxed or relaxed MP2 density matrix.

    Args:
        mp2 : DFRMP2 instance
        relaxed : relaxed density if True, unrelaxed density if False
        logger : Logger instance

    Returns:
        the 1-RDM in MO basis
    '''
    if not mp2.has_ints:
        mp2.calculate_integrals_()

    if logger is None:
        logger = lib.logger.new_logger(mp2)
    rdm1, GammaFile = \
        rmp2_densities_contribs(mp2._intsfile, mp2.mo_energy, mp2.frozen_mask, mp2.max_memory,
                                logger, calcGamma=relaxed, auxmol=mp2.auxmol, ps=mp2.ps, pt=mp2.pt)

    if relaxed:

        # right-hand side for the CPHF equation
        Lvo, Lfo = orbgrad_from_Gamma(mp2.mol, mp2.auxmol, GammaFile['Gamma'],
                                      mp2.mo_coeff, mp2.frozen_mask, mp2.max_memory, logger)

        # frozen core orbital relaxation contribution
        frozen_list = np.arange(mp2.nmo)[mp2.frozen_mask]
        for fm, f in enumerate(frozen_list):
            for i in np.arange(mp2.nmo)[mp2.occ_mask]:
                zfo = Lfo[fm, i] / (mp2.mo_energy[f] - mp2.mo_energy[i])
                rdm1[f, i] += 0.5 * zfo
                rdm1[i, f] += 0.5 * zfo

        # Fock response
        Lvo -= fock_response_rhf(mp2._scf, rdm1)
        # solving the CPHF equations
        zvo = solve_cphf_rhf(mp2._scf, -Lvo, mp2.cphf_max_cycle, mp2.cphf_tol, logger)

        # add the relaxation contribution to the density
        rdm1[mp2.nocc:, :mp2.nocc] += 0.5 * zvo
        rdm1[:mp2.nocc, mp2.nocc:] += 0.5 * zvo.T

    # SCF part of the density
    rdm1[:mp2.nocc, :mp2.nocc] += 2.0 * np.eye(mp2.nocc)
    return rdm1


def rmp2_densities_contribs(intsfile, mo_energy, frozen_mask, max_memory, logger,
                            calcGamma=False, auxmol=None, ps=1.0, pt=1.0):
    '''
    Calculates the unrelaxed DF-MP2 density matrix contribution with an RHF reference.
    Note: this is the difference density, i.e. without HF contribution.
    Also calculates the three-center two-particle density if requested.

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
    ints = intsfile['ints_cholesky']
    nocc_act, naux, nvirt = ints.shape
    nmo = len(mo_energy)
    nfrozen = np.count_nonzero(frozen_mask)
    nocc = nocc_act + nfrozen
    if nocc + nvirt != nmo:
        raise ValueError('numbers of frozen, occupied and virtual orbitals inconsistent')

    logger.debug('    Density matrix contributions for DF-RMP2')
    logger.debug('    Occupied orbitals: {0:d}'.format(nocc))
    logger.debug('    Virtual orbitals:  {0:d}'.format(nvirt))
    logger.debug('    Frozen orbitals:   {0:d}'.format(nfrozen))
    logger.debug('    Three center integrals from file: {0:s}'.format(intsfile.filename))

    # Precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for division with numpy.
    Eab = np.zeros((nvirt, nvirt))
    for a in range(nvirt):
        Eab[a, :] += mo_energy[nocc + a]
        Eab[:, a] += mo_energy[nocc + a]

    GammaFile, Gamma, LT = None, None, None
    if calcGamma:
        if not auxmol:
            raise RuntimeError('auxmol needs to be specified for relaxed density computation')
        # create temporary file to store the two-body density Gamma
        GammaFile = lib.H5TmpFile(libver='latest')
        Gamma = GammaFile.create_dataset('Gamma', (nocc_act, naux, nvirt), dtype='f8')
        logger.debug('    Storing 3c2e density in file: {0:s}'.format(GammaFile.filename))
        # We will need LT = L^T, where L L^T = V
        LT = scipy.linalg.cholesky(auxmol.intor('int2c2e'), lower=False)

    # We start forming P with contiguous frozen, occupied, virtual subblocks.
    P = np.zeros((nmo, nmo))
    mo_energy_masked = mo_energy[~frozen_mask]

    with lib.H5TmpFile(libver='latest') as tfile:
        logger.debug('    Storing amplitudes in temporary file: {0:s}'.format(tfile.filename))

        # For each occupied orbital i, all amplitudes are calculated once and stored on disk.
        # The occupied 1-RDM contribution is calculated in a batched algorithm.
        # More memory -> more efficient I/O.
        # The virtual contribution to the 1-RDM is calculated in memory.
        tiset = tfile.create_dataset('amplitudes', (nocc_act, nvirt, nvirt), dtype='f8')
        for i in range(nocc_act):
            ints3c_ia = ints[i, :, :]

            # Calculate amplitudes T^ij_ab for a given i and all j, a, b
            # Store the amplitudes in a file.
            for j in range(nocc_act):
                ints3c_jb = ints[j, :, :]
                Kab = lib.dot(ints3c_ia.T, ints3c_jb)
                DE = mo_energy_masked[i] + mo_energy_masked[j] - Eab
                Tab = Kab / DE
                TCab = 2.0 * (ps + pt) * Tab - 2.0 * pt * Tab.T
                tiset[j, :, :] = Tab
                # virtual 1-RDM contribution
                P[nocc:, nocc:] += lib.dot(Tab, TCab.T)
            del ints3c_jb, Kab, DE, Tab, TCab

            # Read batches of amplitudes from disk and calculate the occupied 1-RDM.
            batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / (2 * nocc_act * nvirt * 8))
            batchsize = min(nvirt, batchsize)
            if batchsize < 1:
                raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
            logger.debug2('      Pij formation - MO {0:d}, batch size {1:d} (of {2:d})'.
                          format(i, batchsize, nvirt))
            for astart in range(0, nvirt, batchsize):
                aend = min(astart+batchsize, nvirt)
                tbatch1 = tiset[:, astart:aend, :]
                tbatch2 = tiset[:, :, astart:aend]
                P[nfrozen:nocc, nfrozen:nocc] += \
                    - 2.0 * (ps + pt) * lib.einsum('iab,jab->ij', tbatch1, tbatch1) \
                    + 2.0 * pt * lib.einsum('iab,jba->ij', tbatch1, tbatch2)
            del tbatch1, tbatch2

            if calcGamma:
                # This produces (P | Q)^-1 (Q | i a)
                ints3cV1_ia = scipy.linalg.solve_triangular(LT, ints3c_ia, lower=False)
                # Read batches of amplitudes from disk and calculate the two-body density Gamma
                size = nvirt * nvirt * 8 + naux * nvirt * 8
                batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / size)
                batchsize = min(nocc_act, batchsize)
                if batchsize < 1:
                    raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
                logger.debug2('      Gamma formation - MO {0:d}, batch size {1:d} (of {2:d})'.
                              format(i, batchsize, nocc_act))
                for jstart in range(0, nocc_act, batchsize):
                    jend = min(jstart+batchsize, nocc_act)
                    tbatch = tiset[jstart:jend, :, :]
                    Gbatch = Gamma[jstart:jend, :, :]
                    for jj in range(jend-jstart):
                        TCijab_scal = 4.0 * (pt + ps) * tbatch[jj] - 4.0 * pt * tbatch[jj].T
                        Gbatch[jj] += lib.dot(ints3cV1_ia, TCijab_scal)
                    Gamma[jstart:jend, :, :] = Gbatch
                del ints3cV1_ia, tbatch, Gbatch, TCijab_scal

    # now reorder P such that the frozen orbitals correspond to frozen_mask
    idx_reordered = np.concatenate([np.arange(nmo)[frozen_mask], np.arange(nmo)[~frozen_mask]])
    P[idx_reordered, :] = P.copy()
    P[:, idx_reordered] = P.copy()

    logger.debug('    Density matrix contributions calculation finished')
    return P, GammaFile


class BatchSizeError(Exception):
    pass


def shellBatchGenerator(mol, nao_max):
    '''
    Generates sets of shells with a limited number of functions.

    Args:
        mol : the molecule object
        nao_max : maximum number of AOs in each set

    Returns:
        generator yields ((first shell, last shell+1), (first AO, last AO+1))
    '''
    nbas = mol.nbas
    # ao_loc contains nbas + 1 entries
    ao_loc = mol.ao_loc
    shell_start = 0
    while shell_start < nbas:
        shell_stop = shell_start
        while (ao_loc[shell_stop+1] - ao_loc[shell_start] <= nao_max):
            shell_stop += 1
            if shell_stop == nbas:
                break
        if shell_stop == shell_start:
            raise BatchSizeError('empty batch')
        shell_range = (shell_start, shell_stop)
        ao_range = (ao_loc[shell_start], ao_loc[shell_stop])
        yield shell_range, ao_range
        shell_start = shell_stop


def orbgrad_from_Gamma(mol, auxmol, Gamma, mo_coeff, frozen_mask, max_memory, logger):
    '''
    Calculates the orbital gradient of the two-electron term in the Hylleraas functional.

    Args:
        mol : Mole object
        auxmol : Mole object for the auxiliary functions
        Gamma : h5py dataset with the 3c2e density, order: [occ. orbs., aux. fcns., virt. orbs.]
        mo_coeff : molecular orbital coefficients
        frozen_mask : boolean mask for frozen orbitals
        max_memory : memory limit in MB
        logger : Logger object

    Returns:
        orbital gradient in shape: virt. orbitals x occ. orbitals,
        orbital gradient in shape: froz. orbitals x occ. orbitals
    '''
    nocc_act, _, nvirt = Gamma.shape
    nfrozen = np.count_nonzero(frozen_mask)
    nocc = nfrozen + nocc_act
    nmo = len(mo_coeff)
    if nocc + nvirt != nmo:
        raise ValueError('numbers of frozen, occupied and virtual orbitals inconsistent')

    occ_mask = np.zeros(nmo, dtype=bool)
    occ_mask[:nocc] = True
    occ_mask[frozen_mask] = False

    logger.debug('    Contracting the two-body density with 3c2e integrals in memory')
    logger.debug('    Occupied orbitals: {0:d}'.format(nocc))
    logger.debug('    Virtual orbitals:  {0:d}'.format(nvirt))
    logger.debug('    Frozen orbitals:   {0:d}'.format(nfrozen))

    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,
                                 auxmol._atm, auxmol._bas, auxmol._env)
    intor = mol._add_suffix('int3c2e')
    logger.debug2('    intor = {0:s}'.format(intor))

    Lov_act = np.zeros((nocc_act, nvirt))
    Lof_act = np.zeros((nocc_act, nfrozen))
    Lfv = np.zeros((nfrozen, nvirt))
    # process as many auxiliary functions in a go as possible: may reduce I/O cost
    size_per_aux = (nocc_act * nvirt + mol.nao ** 2) * 8
    naux_max = int((max_memory - lib.current_memory()[0]) * 1e6 / size_per_aux)
    logger.debug2('    Max. auxiliary functions per batch: {0:d}'.format(naux_max))
    try:

        # loop over batches of auxiliary function shells
        for auxsh_range, aux_range in shellBatchGenerator(auxmol, naux_max):
            auxsh_start, auxsh_stop = auxsh_range
            aux_start, aux_stop = aux_range
            logger.debug2('      aux from {0:d} to {1:d}'.format(aux_start, aux_stop))
            # needs to follow the convention (AO, AO | Aux)
            shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+auxsh_start, mol.nbas+auxsh_stop)
            # AO integrals
            aoints_auxshell = gto.getints(intor, atm, bas, env, shls_slice)
            # read 3c2e density elements for the current aux functions
            GiKa = Gamma[:, aux_start:aux_stop, :]
            for m in range(aux_stop - aux_start):
                # Half-transformed Gamma for specific auxiliary function m
                G12 = lib.dot(GiKa[:, m, :], mo_coeff[:, nocc:].T)
                # product of Gamma with integrals: one index still in AO basis
                Gints = lib.dot(G12, aoints_auxshell[:, :, m])
                # 3c2e integrals in occupied MO basis
                intso12 = lib.dot(aoints_auxshell[:, :, m], mo_coeff[:, occ_mask])
                intsoo = lib.dot(mo_coeff[:, occ_mask].T, intso12)
                intsfo = lib.dot(mo_coeff[:, frozen_mask].T, intso12)
                # contributions to the orbital gradient
                Lov_act += lib.dot(intsoo, GiKa[:, m, :]) - lib.dot(Gints, mo_coeff[:, nocc:])
                Lof_act -= lib.dot(Gints, mo_coeff[:, frozen_mask])
                Lfv += lib.dot(intsfo, GiKa[:, m, :])
            del GiKa, aoints_auxshell

    except BatchSizeError:
        raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY)')

    # convert to full matrix with frozen orbitals
    Lvo = np.zeros((nvirt, nocc))
    Lvo[:, occ_mask[:nocc]] = Lov_act.T
    Lvo[:, frozen_mask[:nocc]] = Lfv.T
    Lfo = np.zeros((nfrozen, nocc))
    Lfo[:, occ_mask[:nocc]] = Lof_act.T

    logger.debug('    Finished integral contraction.')

    return Lvo, Lfo


def fock_response_rhf(mf, dm, full=True):
    '''
    Calculate the Fock response function for a given density matrix:
    sum_pq [ 4 (ai|pq) - (ap|iq) - (aq|ip) ] dm[p, q]

    Args:
        mf : RHF instance
        dm : density matrix in MO basis
        full : full MO density matrix if True, virtual x occupied if False

    Returns:
        Fock response in MO basis. Shape: virtual x occupied.
    '''
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    Ci = mo_coeff[:, mo_occ>0]
    Ca = mo_coeff[:, mo_occ==0]
    if full:
        dmao = lib.einsum('xp,pq,yq->xy', mo_coeff, dm, mo_coeff)
    else:
        dmao = lib.einsum('xa,ai,yi->xy', Ca, dm, Ci)
    rao = 2.0 * mf.get_veff(dm=dmao+dmao.T)
    rvo = lib.einsum('xa,xy,yi->ai', Ca, rao, Ci)
    return rvo


def solve_cphf_rhf(mf, Lvo, max_cycle, tol, logger):
    '''
    Solve the CPHF equations.
    (e[i] - e[a]) zvo[a, i] - sum_bj [ 4 (ai|bj) - (ab|ij) - (aj|ib) ] zvo[b, j] = Lvo[a, i]

    Args:
        mf : an RHF object
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

    def fvind(z):
        return fock_response_rhf(mf, z.reshape(Lvo.shape), full=False)

    zvo = cphf.solve(fvind, mf.mo_energy, mf.mo_occ, Lvo,
                     max_cycle=max_cycle, tol=tol, verbose=cphf_verbose)[0]
    logger.info('CPHF iterations finished')
    return zvo


if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. ,  0.    , 0.   )],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. ,  0.757 , 0.587)]]
    mol.basis = 'def2-SVP'
    mol.verbose = lib.logger.INFO
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    with DFMP2(mf) as pt:
        pt.kernel()
        natocc, _ = pt.make_natorbs()
        print()
        print(natocc)
