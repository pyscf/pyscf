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
native implementation of DF-MP2/RI-MP2 with a UHF reference
'''

import numpy as np
import scipy

from pyscf import lib
from pyscf import scf
from pyscf import df
from pyscf.mp.dfmp2_fast import DFRMP2, ints3c_cholesky, order_mos_fc


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

        # process the frozen core option correctly as either an integer or two lists (alpha, beta)
        if not frozen:
            self.nfrozen = 0
        else:
            if lib.isinteger(frozen):
                self.nfrozen = int(frozen)
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
                self.nfrozen = len(frozen[0])
                for s in 0, 1:
                    self.mo_coeff[s, :, :], self.mo_energy[s, :] = \
                        order_mos_fc(frozen[s], mf.mo_coeff[s], mf.mo_energy[s], self.nocc[s])

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
        self.rdm1_mo = None

        # Spin component scaling factors
        self.ps = 1.0
        self.pt = 1.0

    def dump_flags(self, logger=None):
        if not logger:
            logger = lib.logger.new_logger(self)
        logger.info('')
        logger.info('******** {0:s} ********'.format(repr(self.__class__)))
        logger.info('nmo = {0:d}'.format(self.nmo))
        logger.info('nocc = {0:d}, {1:d}'.format(self.nocc[0], self.nocc[1]))
        logger.info('nfrozen = {0:d}'.format(self.nfrozen))
        logger.info('basis = {0:s}'.format(repr(self.mol.basis)))
        logger.info('auxbasis = {0:s}'.format(repr(self.auxmol.basis)))
        logger.info('max_memory = {0:.1f} MB (current use {1:.1f} MB)'. \
            format(self.max_memory, lib.current_memory()[0]))

    def calculate_energy(self):
        '''
        Calculates the MP2 correlation energy.
        '''
        if not self._intsfile:
            self.calculate_integrals()

        no = self.nocc - self.nfrozen
        emo = self.mo_energy[:, self.nfrozen:]
        logger = lib.logger.new_logger(self)
        self.e_corr = emp2_uhf(self._intsfile, no, emo, logger, ps=self.ps, pt=self.pt)
        return self.e_corr

    def make_rdm1(self, ao_repr=True):
        '''
        Calculates the unrelaxed MP2 density matrix.

        Warning: the MO basis is self.mo_coeff, not mf.mo_coeff. This is relevant if
        __init__ was supplied with a list of frozen orbitals.

        Args:
            ao_repr : density in AO or in MO basis

        Returns:
            1-RDM in shape (2, nmo, nmo) containing the spin-up and spin-down components
        '''
        if not self._intsfile:
            self.calculate_integrals()

        # Calculate the unrelaxed 1-RDM.
        nfrz = self.nfrozen
        no = self.nocc - nfrz
        emo = self.mo_energy[:, nfrz:]
        logger = lib.logger.new_logger(self)
        rdm1_mo = np.zeros((2, self.nmo, self.nmo))
        rdm1_mo[:, nfrz:, nfrz:] += mp2_uhf_densities(self._intsfile, no, emo, \
            self.max_memory, logger, ps=self.ps, pt=self.pt)[0]

        # Set the UHF density matrix for the frozen orbitals if applicable.
        if nfrz > 0:
            rdm1_mo[0, :nfrz, :nfrz] += np.eye(nfrz)
            rdm1_mo[1, :nfrz, :nfrz] += np.eye(nfrz)

        self.rdm1_mo = rdm1_mo
        if ao_repr:
            return np.einsum('sxp,spq,syq->sxy', self.mo_coeff, rdm1_mo, self.mo_coeff)
        else:
            return rdm1_mo

    def make_natorbs(self):
        '''
        Calculate the natural orbitals.
        Perform the entire 1-RDM computation if necessary.
        Note: the most occupied orbitals come first (left)
              and the least occupied orbitals last (right).

        Returns:
            natural occupation numbers, natural orbitals
        '''
        if self.rdm1_mo is None:
            self.make_rdm1(ao_repr=False)

        # Transform the beta component to the alpha basis and sum both together.
        SAO = self.mol.intor_symmetric('int1e_ovlp')
        Sab = np.linalg.multi_dot([self.mo_coeff[0, :, :].T, SAO, self.mo_coeff[1, :, :]])
        rdm1_abas = self.rdm1_mo[0, :, :] + \
            np.linalg.multi_dot([Sab, self.rdm1_mo[1, :, :], Sab.T])

        # Diagonalize the spin-traced 1-RDM in alpha basis to get the natural orbitals.
        eigval, eigvec = np.linalg.eigh(rdm1_abas)
        natocc = np.flip(eigval)
        natorb = np.dot(self.mo_coeff[0, :, :], np.fliplr(eigvec))
        return natocc, natorb
    
    def calculate_integrals(self):
        '''
        Calculates the three center integrals for MP2.
        '''
        intsfile = []
        logger = lib.logger.new_logger(self)
        for s in 0, 1:
            moc_occ = self.mo_coeff[s, :, self.nfrozen:self.nocc[s]]
            moc_virt = self.mo_coeff[s, :, self.nocc[s]:]
            f = ints3c_cholesky(self.mol, self.auxmol, moc_occ, moc_virt, \
                self.max_memory, logger)
            intsfile.append(f)
        self._intsfile = intsfile
    
    def delete(self):
        '''
        Delete the temporary file(s).
        '''
        self._intsfile = []


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
        logger.info('pt(scs) = {0:f}'.format(self.pt))
        logger.info('ps(scs) = {0:f}'.format(self.ps))


SCSMP2 = SCSUMP2 = SCSDFMP2 = SCSDFUMP2


def emp2_uhf(intsfiles, nocc, mo_energy, logger, ps=1.0, pt=1.0):
    '''
    Calculates the DF-MP2 energy with an UHF reference.

    Args:
        intsfiles : contains the three center integrals in MO basis
        nocc : numbers of occupied orbitals
        mo_energy : energies of the molecular orbitals
        logger : Logger instance
        ps : SCS factor for opposite-spin contributions
        pt : SCS factor for same-spin contributions

    Returns:
        the MP2 correlation energy
    '''
    nvirt = mo_energy.shape[1] - nocc

    logger.info('')
    logger.info('*** DF-MP2 energy')
    logger.info('    Occupied orbitals: {0:d}, {1:d}'.format(nocc[0], nocc[1]))
    logger.info('    Virtual orbitals:  {0:d}, {1:d}'.format(nvirt[0], nvirt[1]))
    logger.info('    Integrals (alpha) from file: {0:s}'.format(intsfiles[0].filename))
    logger.info('    Integrals (beta)  from file: {0:s}'.format(intsfiles[1].filename))

    energy_total = 0.0

    # loop over spins to calculate same-spin energies
    for s in 0, 1:
        energy_contrib = 0.0
        if s == 0:
            logger.info('  * alpha-alpha pairs')
        else:
            logger.info('  * beta-beta pairs')
        ints = intsfiles[s]['ints_cholesky']
        # precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for the denominator
        Eab = np.zeros((nvirt[s], nvirt[s]))
        for a in range(nvirt[s]):
            Eab[a, :] += mo_energy[s, nocc[s]+a]
            Eab[:, a] += mo_energy[s, nocc[s]+a]
        # loop over j < i
        for i in range(nocc[s]):
            ints3c_ia = ints[i, :, :]
            for j in range(i):
                ints3c_jb = ints[j, :, :]
                Kab = np.matmul(ints3c_ia.T, ints3c_jb)
                DE = mo_energy[s, i] + mo_energy[s, j] - Eab
                Tab = (Kab - Kab.T) / DE
                energy_contrib += pt * np.einsum('ab,ab', Tab, Kab)
        logger.info('      E = {0:.14f} Eh'.format(energy_contrib))
        energy_total += energy_contrib

    # opposite-spin energy
    logger.info('  * alpha-beta pairs')
    # precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for the denominator
    Eab = np.zeros((nvirt[0], nvirt[1]))
    for a in range(nvirt[0]):
        Eab[a, :] += mo_energy[0, nocc[0]+a]
    for b in range(nvirt[1]):
        Eab[:, b] += mo_energy[1, nocc[1]+b]
    # loop over i(alpha), j(beta)
    ints_a = intsfiles[0]['ints_cholesky']
    ints_b = intsfiles[1]['ints_cholesky']
    energy_contrib = 0.0
    for i in range(nocc[0]):
        ints3c_ia = ints_a[i, :, :]
        for j in range(nocc[1]):
            ints3c_jb = ints_b[j, :, :]
            Kab = np.matmul(ints3c_ia.T, ints3c_jb)
            DE = mo_energy[0, i] + mo_energy[1, j] - Eab
            Tab = Kab / DE
            energy_contrib += ps * np.einsum('ab,ab', Tab, Kab)
    logger.info('      E = {0:.14f} Eh'.format(energy_contrib))
    energy_total += energy_contrib

    logger.note('*** DF-MP2 correlation energy: {0:.14f} Eh'.format(energy_total))
    return energy_total


def mp2_uhf_densities(intsfiles, nocc, mo_energy, max_memory, logger, \
    relaxed=False, auxmol=None, ps=1.0, pt=1.0):
    '''
    Calculates the unrelaxed DF-MP2 density matrix with a UHF reference.
    Also calculates the three-center two-particle density if requested.

    Args:
        intsfile : contains the three center integrals
        noccs : number of occupied orbitals
        mo_energy : molecular orbital energies
        max_memory : memory threshold in MB
        logger : Logger instance
        relaxed : if True, calculate contributions for the relaxed density
        auxmol : required if relaxed is True
        ps : SCS factor for opposite-spin contributions
        pt : SCS factor for same-spin contributions
    
    Returns:
        matrix containing the unrelaxed 1-RDM, file with 3c2e density if requested
    '''
    if relaxed and not auxmol:
        raise RuntimeError('auxmol needs to be specified for relaxed density computation')

    nmo = mo_energy.shape[1]
    nvirt = nmo - nocc

    logger.info('')
    logger.info('*** Density matrix contributions for DF-MP2')
    logger.info('    Occupied orbitals: {0:d}, {1:d}'.format(nocc[0], nocc[1]))
    logger.info('    Virtual orbitals:  {0:d}, {1:d}'.format(nvirt[0], nvirt[1]))
    logger.info('    Three center integrals (alpha) from file: {0:s}'.format(intsfiles[0].filename))
    logger.info('    Three center integrals (beta) from file: {0:s}'.format(intsfiles[1].filename))

    # Density matrix initialized with the UHF contribution.
    P = np.zeros((2, nmo, nmo))
    P[0, :nocc[0], :nocc[0]] = np.eye(nocc[0])
    P[1, :nocc[1], :nocc[1]] = np.eye(nocc[1])

    GammaFile, LT, naux = (None,) * 3
    if relaxed:
        if not auxmol:
            raise RuntimeError('auxmol needs to be specified for relaxed density computation')
        else:
            naux = auxmol.nao
        # create temporary file to store the two-body density Gamma
        GammaFile = lib.H5TmpFile(libver='latest')
        GammaFile.create_dataset('Gamma_alpha', (nocc[0], naux, nvirt[0]))
        GammaFile.create_dataset('Gamma_beta', (nocc[1], naux, nvirt[1]))
        logger.info('    Storing 3c2e density in file: {0:s}'.format(GammaFile.filename))
        # We will need LT = L^T, where L L^T = V
        LT = scipy.linalg.cholesky(auxmol.intor('int2c2e'), lower=False)

    # Loop over all the spin variants
    for s1, s2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        with lib.H5TmpFile(libver='latest') as tfile:

            tiset = \
                tfile.create_dataset('amplitudes', (nocc[s2], nvirt[s1], nvirt[s2]), dtype='f8')

            s1_str = ('alpha', 'beta')[s1]
            s2_str = ('alpha', 'beta')[s2]
            logger.info('  * {0:s}-{1:s} pairs'.format(s1_str, s2_str))
            logger.info('    Storing amplitudes in temporary file: {0:s}'.format(tfile.filename))

            # Precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for division with numpy.
            Eab = np.zeros((nvirt[s1], nvirt[s2]))
            for a in range(nvirt[s1]):
                Eab[a, :] += mo_energy[s1, nocc[s1]+a]
            for b in range(nvirt[s2]):
                Eab[:, b] += mo_energy[s2, nocc[s2]+b]

            ints1 = intsfiles[s1]['ints_cholesky']
            ints2 = intsfiles[s2]['ints_cholesky']

            # For each occupied spin orbital i, all amplitudes are calculated once and
            # stored on disk. The occupied 1-RDM contribution is calculated in a batched
            # algorithm. More memory -> more efficient I/O.
            # The virtual contribution to the 1-RDM is calculated in memory.
            for i in range(nocc[s1]):
                ints3c_ia = ints1[i, :, :]

                # Amplitudes T^ij_ab are calculated for a given orbital i with spin s1,
                # and all j (s2), a (s1) and b (s2). These amplitudes are stored on disk.
                for j in range(nocc[s2]):
                    ints3c_jb = ints2[j, :, :]
                    Kab = np.matmul(ints3c_ia.T, ints3c_jb)
                    DE = mo_energy[s1, i] + mo_energy[s2, j] - Eab
                    if s1 == s2:
                        numerator = Kab - Kab.T
                        prefactor = 0.5 * pt
                    else:
                        numerator = Kab
                        prefactor = ps
                    Tab = numerator / DE
                    tiset[j, :, :] = Tab
                    # virtual 1-RDM contribution
                    P[s1, nocc[s1]:, nocc[s1]:] += prefactor * np.matmul(Tab, Tab.T)
                del ints3c_jb, Kab, DE, numerator, Tab

                # Batches of amplitudes are read from disk to calculate the occupied
                # 1-RDM contribution.
                batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / (nocc[s2] * nvirt[s2] * 8))
                batchsize = min(nvirt[s1], batchsize)
                if batchsize < 1:
                    raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
                logger.debug('    Batch size: {0:d} (of {1:d})'.format(batchsize, nvirt[s1]))
                logger.debug('      Pij formation - MO {0:d} ({1:s}), batch size {2:d} (of {3:d})'. \
                    format(i, s1_str, batchsize, nvirt[s1]))
                for astart in range(0, nvirt[s1], batchsize):
                    aend = min(astart+batchsize, nvirt[s1])
                    tbatch = tiset[:, astart:aend, :]
                    if s1 == s2:
                        prefactor = 0.5 * pt
                    else:
                        prefactor = ps
                    P[s2, :nocc[s2], :nocc[s2]] -= \
                        prefactor * np.einsum('iab,jab->ij', tbatch, tbatch)
                del tbatch

                if relaxed:
                    # This produces (P | Q)^-1 (Q | i a)
                    ints3cV1_ia = scipy.linalg.solve_triangular(LT, ints3c_ia, lower=False)
                    # Here, we construct Gamma for spin s2
                    Gamma = GammaFile['Gamma_'+s2_str]
                    # Read batches of amplitudes from disk and calculate the two-body density Gamma
                    size = nvirt[s1] * nvirt[s2] * 8 + naux * nvirt[s2] * 8
                    batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / size)
                    batchsize = min(nocc[s2], batchsize)
                    if batchsize < 1:
                        raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
                    logger.debug('      Gamma ({0:s}) formation - MO {1:d} ({2:s}), batch size {3:d} (of {4:d})'. \
                        format(s2_str, i, s1_str, batchsize, nocc[s2]))
                    if s1 == s2:
                        prefactor = 2.0 * pt
                    else:
                        prefactor = ps
                    for jstart in range(0, nocc[s2], batchsize):
                        jend = min(jstart+batchsize, nocc[s2])
                        tbatch = tiset[jstart:jend, :, :]
                        Gbatch = Gamma[jstart:jend, :, :]
                        for jj in range(jend-jstart):
                            Tijab = tbatch[jj]
                            Gbatch[jj] += prefactor * np.matmul(ints3cV1_ia, Tijab)
                        Gamma[jstart:jend, :, :] = Gbatch
                    del tbatch, Gbatch

    logger.info('*** Density matrix contributions calculation finished')
    return P, GammaFile


if __name__ == '__main__':
    from pyscf import gto, scf, lib

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
