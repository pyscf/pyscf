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
#
# Author: Peter Pinski, HQS Quantum Simulations

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

        # process the frozen core option correctly as an integer or a list
        if not frozen:
            self.nfrozen = 0
        else:
            if lib.isinteger(frozen):
                self.nfrozen = int(frozen)
            elif lib.isintsequence(frozen):
                self.nfrozen = len(frozen)
                self.mo_coeff, self.mo_energy = order_mos_fc(frozen, mf.mo_coeff, mf.mo_energy, self.nocc)
            else:
                raise TypeError('frozen must be an integer or a list of integers')

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
        if not logger:
            logger = lib.logger.new_logger(self)
        logger.info('')
        logger.info('******** {0:s} ********'.format(repr(self.__class__)))
        logger.info('nmo = {0:d}'.format(self.nmo))
        logger.info('nocc = {0:d}'.format(self.nocc))
        logger.info('nfrozen = {0:d}'.format(self.nfrozen))
        logger.info('basis = {0:s}'.format(repr(self.mol.basis)))
        logger.info('auxbasis = {0:s}'.format(repr(self.auxmol.basis)))
        logger.info('max_memory = {0:.1f} MB (current use {1:.1f} MB)'. \
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
        if not self._intsfile:
            self.calculate_integrals()

        no = self.nocc - self.nfrozen
        emo = self.mo_energy[self.nfrozen:]
        logger = lib.logger.new_logger(self)
        self.e_corr = emp2_rhf(self._intsfile, no, emo, logger, ps=self.ps, pt=self.pt)
        return self.e_corr
    
    def make_rdm1(self, ao_repr=True):
        '''
        Calculates the unrelaxed MP2 density matrix.

        Warning: the MO basis is self.mo_coeff, not mf.mo_coeff. This is relevant if
        __init__ was supplied with a list of frozen orbitals.

        Args:
            ao_repr : density in AO or in MO basis

        Returns:
            the 1-RDM
        '''
        if not self._intsfile:
            self.calculate_integrals()

        # Calculate the unrelaxed 1-RDM.
        nfrz = self.nfrozen
        no = self.nocc - nfrz
        emo = self.mo_energy[nfrz:]
        logger = lib.logger.new_logger(self)
        rdm1_mo = np.zeros((self.nmo, self.nmo))
        rdm1_mo[nfrz:, nfrz:] = mp2_rhf_densities(self._intsfile, no, emo, \
            self.max_memory, logger, ps=self.ps, pt=self.pt)[0]

        # HF contribution
        rdm1_mo[:self.nocc, :self.nocc] += 2.0 * np.eye(self.nocc)

        if ao_repr:
            return np.linalg.multi_dot([self.mo_coeff, rdm1_mo, self.mo_coeff.T])
        else:
            return rdm1_mo
    
    def make_response_dm(self, ao_repr=True):
        '''
        Calculates the relaxed MP2 density matrix.

        Args:
            ao_repr : density in AO or in MO basis
        
        Returns:
            the relaxed 1-RDM
        '''
        # get the unrelaxed 1-RDM and the 3-center-2-electron density
        if not self._intsfile:
            self.calculate_integrals()
        logger = lib.logger.new_logger(self)
        rdm1_ur, GammaFile = mp2_rhf_densities(self._intsfile, self.nocc, self.mo_energy, \
            self.max_memory, logger, relaxed=True, auxmol=self.auxmol, ps=self.ps, pt=self.pt)

        # right-hand side for the CPHF equation
        Lai = orbgrad_from_Gamma(self.mol, self.auxmol, GammaFile['Gamma'], \
            self.mo_coeff, self.nocc, self.max_memory, logger)
        Lai -= fock_response_rhf(self._scf, rdm1_ur)
        zai = solve_cphf_rhf(self._scf, -Lai, self.cphf_max_cycle, self.cphf_tol, logger)

        # add the relaxation contribution to the density
        rdm1_re = rdm1_ur
        rdm1_re[self.nocc:, :self.nocc] += 0.5 * zai
        rdm1_re[:self.nocc, self.nocc:] += 0.5 * zai.T
        rdm1_re[:self.nocc, :self.nocc] += 2.0 * np.eye(self.nocc)

        if ao_repr:
            return np.linalg.multi_dot([self.mo_coeff, rdm1_re, self.mo_coeff.T])
        else:
            return rdm1_re

    def make_natorbs(self, rdm1_mo=None, relaxed=False):
        '''
        Calculate the natural orbitals.
        Perform the entire 1-RDM computation if necessary.
        Note: the most occupied orbitals come first (left)
              and the least occupied orbitals last (right).

        Returns:
            natural occupation numbers, natural orbitals
        '''
        if rdm1_mo is None:
            if relaxed:
                rdm1_mo = self.make_response_dm(ao_repr=False)
            else:
                rdm1_mo = self.make_rdm1(ao_repr=False)
        elif not isinstance(rdm1_mo, np.ndarray):
            raise TypeError('rdm1_mo must be a 2-D array')

        eigval, eigvec = np.linalg.eigh(rdm1_mo)
        natocc = np.flip(eigval)
        natorb = np.dot(self.mo_coeff, np.fliplr(eigvec))
        return natocc, natorb

    def calculate_integrals(self):
        '''
        Calculates the three center integrals for MP2.
        '''
        moc_occ = self.mo_coeff[:, self.nfrozen:self.nocc]
        moc_virt = self.mo_coeff[:, self.nocc:]
        logger = lib.logger.new_logger(self)
        self._intsfile = ints3c_cholesky(self.mol, self.auxmol, moc_occ, moc_virt, \
            self.max_memory, logger)

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
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,\
        auxmol._atm, auxmol._bas, auxmol._env)
    nmo1 = mo_coeff1.shape[1]
    nmo2 = mo_coeff2.shape[1]
    nauxfcns = auxmol.nao

    logger.info('')
    logger.info('*** DF integral transformation')
    logger.info('    MO dimensions: {0:d} x {1:d}'.format(nmo1, nmo2))
    logger.info('    Aux functions: {0:d}'.format(nauxfcns))

    intsfile_cho = lib.H5TmpFile(libver='latest')
    with lib.H5TmpFile(libver='latest') as intsfile_tmp:

        logger.info('  * Calculating three center integrals in MO basis.')
        logger.info('    Temporary file: {0:s}'.format(intsfile_tmp.filename))

        intor = mol._add_suffix('int3c2e')
        logger.debug('    intor = {0:s}'.format(intor))

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
                    moints = np.matmul(np.matmul(mo_coeff1.T, aoints), mo_coeff2)
                else:
                    moints = np.matmul(mo_coeff1.T, np.matmul(aoints, mo_coeff2))
                ints_3c[aux_ctr, :, :] = moints
                aux_ctr += 1

        logger.info('  * Calculating fitted three center integrals.')
        logger.info('    Storage file: {0:s}'.format(intsfile_cho.filename))

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
        bufsize = min(nmo1, bufsize)
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

    logger.info('*** DF transformation finished')
    return intsfile_cho


def emp2_rhf(intsfile, nocc, mo_energy, logger, ps=1.0, pt=1.0):
    '''
    Calculates the DF-MP2 energy with an RHF reference.

    Args:
        intsfile : contains the three center integrals in MO basis
        nocc : number of occupied orbitals
        mo_energy : energies of the molecular orbitals
        logger : Logger instance
        ps : SCS factor for opposite-spin contributions
        pt : SCS factor for same-spin contributions

    Returns:
        the MP2 correlation energy
    '''
    nvirt = len(mo_energy) - nocc
    logger.info('')
    logger.info('*** DF-MP2 energy')
    logger.info('    Occupied orbitals: {0:d}'.format(nocc))
    logger.info('    Virtual orbitals:  {0:d}'.format(nvirt))
    logger.info('    Integrals from file: {0:s}'.format(intsfile.filename))

    # Somewhat awkward workaround to perform division in the MP2 energy expression
    # through numpy routines. We precompute Eab[a, b] = mo_energy[a] + mo_energy[b].
    Eab = np.zeros((nvirt, nvirt))
    for a in range(nvirt):
        Eab[a, :] += mo_energy[nocc + a]
        Eab[:, a] += mo_energy[nocc + a]

    energy = 0.0
    ints = intsfile['ints_cholesky']
    for i in range(nocc):
        ints3c_ia = ints[i, :, :]
        # contributions for occupied orbitals j < i
        for j in range(i):
            ints3c_jb = ints[j, :, :]
            Kab = np.matmul(ints3c_ia.T, ints3c_jb)
            DE = mo_energy[i] + mo_energy[j] - Eab
            Tab = Kab / DE
            energy += 2.0 * (ps + pt) * np.einsum('ab,ab', Tab, Kab)
            energy -= 2.0 * pt * np.einsum('ab,ba', Tab, Kab)
        # contribution for j == i
        Kab = np.matmul(ints3c_ia.T, ints3c_ia)
        DE = 2.0 * mo_energy[i] - Eab
        Tab = Kab / DE
        energy += ps * np.einsum('ab,ab', Tab, Kab)

    logger.note('*** DF-MP2 correlation energy: {0:.14f} Eh'.format(energy))
    return energy


def mp2_rhf_densities(intsfile, nocc, mo_energy, max_memory, logger, \
    relaxed=False, auxmol=None, ps=1.0, pt=1.0):
    '''
    Calculates the unrelaxed DF-MP2 density matrix contribution with an RHF reference.
    Note: this is the difference density, i.e. without HF contribution.
    Also calculates the three-center two-particle density if requested.

    Args:
        intsfile : contains the three center integrals
        nocc : number of occupied orbitals
        mo_energy : molecular orbital energies
        max_memory : memory threshold in MB
        logger : Logger instance
        relaxed : if True, calculate contributions for the relaxed density
        auxmol : required if relaxed is True
        ps : SCS factor for opposite-spin contributions
        pt : SCS factor for same-spin contributions
    
    Returns:
        matrix containing the 1-RDM contribution, file with 3c2e density if requested
    '''
    nmo = len(mo_energy)
    nvirt = nmo - nocc

    logger.info('')
    logger.info('*** Density matrix contributions for DF-MP2')
    logger.info('    Occupied orbitals: {0:d}'.format(nocc))
    logger.info('    Virtual orbitals:  {0:d}'.format(nvirt))
    logger.info('    Three center integrals from file: {0:s}'.format(intsfile.filename))

    # Precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for division with numpy.
    Eab = np.zeros((nvirt, nvirt))
    for a in range(nvirt):
        Eab[a, :] += mo_energy[nocc + a]
        Eab[:, a] += mo_energy[nocc + a]

    GammaFile, Gamma, LT, naux = (None,) * 4
    if relaxed:
        if not auxmol:
            raise RuntimeError('auxmol needs to be specified for relaxed density computation')
        else:
            naux = auxmol.nao
        # create temporary file to store the two-body density Gamma
        GammaFile = lib.H5TmpFile(libver='latest')
        Gamma = GammaFile.create_dataset('Gamma', (nocc, naux, nvirt), dtype='f8')
        logger.info('    Storing 3c2e density in file: {0:s}'.format(GammaFile.filename))
        # We will need LT = L^T, where L L^T = V
        LT = scipy.linalg.cholesky(auxmol.intor('int2c2e'), lower=False)

    P = np.zeros((nmo, nmo))

    with lib.H5TmpFile(libver='latest') as tfile:
        logger.info('    Storing amplitudes in temporary file: {0:s}'.format(tfile.filename))

        # For each occupied orbital i, all amplitudes are calculated once and stored on disk.
        # The occupied 1-RDM contribution is calculated in a batched algorithm.
        # More memory -> more efficient I/O.
        # The virtual contribution to the 1-RDM is calculated in memory.
        ints = intsfile['ints_cholesky']
        tiset = tfile.create_dataset('amplitudes', (nocc, nvirt, nvirt), dtype='f8')
        for i in range(nocc):
                ints3c_ia = ints[i, :, :]

                # Calculate amplitudes T^ij_ab for a given i and all j, a, b
                # Store the amplitudes in a file.
                for j in range(nocc):
                    ints3c_jb = ints[j, :, :]
                    Kab = np.matmul(ints3c_ia.T, ints3c_jb)
                    DE = mo_energy[i] + mo_energy[j] - Eab
                    Tab = Kab / DE
                    TCab = 2.0 * (ps + pt) * Tab - 2.0 * pt * Tab.T
                    tiset[j, :, :] = Tab
                    # virtual 1-RDM contribution
                    P[nocc:, nocc:] += np.matmul(Tab, TCab.T)
                del ints3c_jb, Kab, DE, Tab, TCab

                # Read batches of amplitudes from disk and calculate the occupied 1-RDM.
                batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / (2 * nocc * nvirt * 8))
                batchsize = min(nvirt, batchsize)
                if batchsize < 1:
                    raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
                logger.debug('      Pij formation - MO {0:d}, batch size {1:d} (of {2:d})'. \
                    format(i, batchsize, nvirt))
                for astart in range(0, nvirt, batchsize):
                    aend = min(astart+batchsize, nvirt)
                    tbatch1 = tiset[:, astart:aend, :]
                    tbatch2 = tiset[:, :, astart:aend]
                    P[:nocc, :nocc] -= 2.0 * (ps + pt) * np.einsum('iab,jab->ij', tbatch1, tbatch1)
                    P[:nocc, :nocc] += 2.0 * pt * np.einsum('iab,jba->ij', tbatch1, tbatch2)
                del tbatch1, tbatch2
                
                if relaxed:
                    # This produces (P | Q)^-1 (Q | i a)
                    ints3cV1_ia = scipy.linalg.solve_triangular(LT, ints3c_ia, lower=False)
                    # Read batches of amplitudes from disk and calculate the two-body density Gamma
                    size = nvirt * nvirt * 8 + naux * nvirt * 8
                    batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / size)
                    batchsize = min(nocc, batchsize)
                    if batchsize < 1:
                        raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
                    logger.debug('      Gamma formation - MO {0:d}, batch size {1:d} (of {2:d})'. \
                        format(i, batchsize, nocc))
                    for jstart in range(0, nocc, batchsize):
                        jend = min(jstart+batchsize, nocc)
                        tbatch = tiset[jstart:jend, :, :]
                        Gbatch = Gamma[jstart:jend, :, :]
                        for jj in range(jend-jstart):
                            TCijab_scal = 4.0 * (pt + pt) * tbatch[jj] - 4.0 * pt * tbatch[jj].T
                            Gbatch[jj] += np.matmul(ints3cV1_ia, TCijab_scal)
                        Gamma[jstart:jend, :, :] = Gbatch
                    del ints3cV1_ia, tbatch, Gbatch, TCijab_scal

    logger.info('*** Density matrix contributions calculation finished')
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


def orbgrad_from_Gamma(mol, auxmol, Gamma, mo_coeff, nocc, max_memory, logger):
    '''
    Calculates the orbital gradient of the two-electron term in the Hylleraas functional.

    Args:
        mol : Mole object
        auxmol : Mole object for the auxiliary functions
        Gamma : h5py dataset with the 3c2e density, order: [occ. orbs., aux. fcns., virt. orbs.]
        mo_coeff : molecular orbital coefficients
        nocc : number of occupied orbitals
        max_memory : memory limit in MB
        logger : Logger object

    Returns:
        orbital gradient in shape: virt. orbitals x occ. orbitals
    '''
    nvirt = mo_coeff.shape[1] - nocc
    logger.info('')
    logger.info('*** Contracting the two-body density with 3c2e integrals in memory')
    logger.info('    Occupied orbitals: {0:d}'.format(nocc))
    logger.info('    Virtual orbitals:  {0:d}'.format(nvirt))

    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,\
        auxmol._atm, auxmol._bas, auxmol._env)
    intor = mol._add_suffix('int3c2e')
    logger.debug('    intor = {0:s}'.format(intor))

    Lia = np.zeros((nocc, nvirt))
    # process as many auxiliary functions in a go as possible: may reduce I/O cost
    naux_max = int((max_memory - lib.current_memory()[0]) * 1e6 / (nocc * nvirt + mol.nao ** 2))
    logger.debug('    Max. auxiliary functions per batch: {0:d}'.format(naux_max))
    try:

        # loop over batches of auxiliary function shells
        for auxsh_range, aux_range in shellBatchGenerator(auxmol, naux_max):
            auxsh_start, auxsh_stop = auxsh_range
            aux_start, aux_stop = aux_range
            logger.debug('      aux from {0:d} to {1:d}'.format(aux_start, aux_stop))
            # needs to follow the convention (AO, AO | Aux)
            shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+auxsh_start, mol.nbas+auxsh_stop)
            # AO integrals
            aoints_auxshell = gto.getints(intor, atm, bas, env, shls_slice)
            # read 3c2e density elements for the current aux functions
            GiKa = Gamma[:, aux_start:aux_stop, :]
            for m in range(aux_stop - aux_start):
                # Half-transformed Gamma for specific auxiliary function m
                G12 = np.matmul(GiKa[:, m, :], mo_coeff[:, nocc:].T)
                # contracted integrals: one index still in AO basis
                ints12 = np.matmul(G12, aoints_auxshell[:, :, m])
                # 3c2e integrals in occupied MO basis
                ints_occ = np.matmul(mo_coeff[:, :nocc].T, \
                    np.matmul(aoints_auxshell[:, :, m], mo_coeff[:, :nocc]))
                # contribution to the orbital gradient
                Lia += np.matmul(ints_occ, GiKa[:, m, :]) - np.matmul(ints12, mo_coeff[:, nocc:])

    except BatchSizeError:
        raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY)')

    logger.info('*** Finished integral contraction.')
    Lai = Lia.T
    return Lai


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
        dmao = np.linalg.multi_dot([mo_coeff, dm, mo_coeff.T])
    else:
        dmao = np.linalg.multi_dot([Ca, dm, Ci.T])
    rao = 2.0 * mf.get_veff(dm=dmao+dmao.T)
    rai = np.linalg.multi_dot([Ca.T, rao, Ci])
    return rai


def solve_cphf_rhf(mf, Lai, max_cycle, tol, logger):
    '''
    Solve the CPHF equations.
    (e[i] - e[a]) zai[a, i] - sum_bj [ 4 (ai|bj) - (ab|ij) - (aj|ib) ] zai[b, j] = Lai[a, i]

    Args:
        mf : an RHF object
        Lai : right-hand side the the response equation
        max_cycle : number of iterations for the CPHF solver
        tol : convergence tolerance for the CPHF solver
        logger : Logger object
    '''
    logger.info('')
    logger.info('*** Solving the CPHF response equations')
    logger.info('    Max. iterations: {0:d}'.format(max_cycle))
    logger.info('    Convergence tolerance: {0:.3g}'.format(tol))

    # Currently we need to make the CPHF solver somewhat more talkative to see anything at all.
    cphf_verbose = logger.verbose
    if logger.verbose == lib.logger.INFO:
        cphf_verbose = lib.logger.DEBUG

    def fvind(z):
        return fock_response_rhf(mf, z.reshape(Lai.shape), full=False)

    zai = cphf.solve(fvind, mf.mo_energy, mf.mo_occ, Lai, \
        max_cycle=max_cycle, tol=tol, verbose=cphf_verbose)[0]
    logger.info('*** CPHF iterations finished')
    return zai


def order_mos_fc(frozen, mo_coeff, mo_energy, nocc):
    '''
    Order MO coefficients and energies such that frozen orbitals come leftmost.

    Args:
        frozen : list / array of frozen orbitals
        mo_coeff : MO coefficient matrix
        mo_energy : MO energies
        nocc : number of occupied orbitals (only used for checking)
    
    Returns:
        Ordered MO coefficients, ordered MO energies.
    '''
    _, nmo = mo_coeff.shape
    if mo_energy.shape != (nmo,):
        raise ValueError('MO coefficients and energies have inconsistent shapes.')
    if len(frozen) != len(set(frozen)):
        raise ValueError('Duplicate elements in list of frozen MOs.')
    if np.any(np.array(frozen) >= nocc):
        raise ValueError('Virtual orbital specified as frozen.')
    mo_order = list(frozen)
    mo_order.sort()
    for p in range(nmo):
        if p not in frozen:
            mo_order.append(p)
    return mo_coeff[:, mo_order], mo_energy[mo_order]


if __name__ == '__main__':
    from pyscf import gto, scf, lib

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
