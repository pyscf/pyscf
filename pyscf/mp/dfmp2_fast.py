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
# Author: Peter Pinski (HQS Quantum Simulations)

'''
native implementation of DF-MP2/RI-MP2 with an RHF reference
'''

import tempfile
import h5py
import os
import numpy as np
import scipy

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import df


class DFMP2(lib.StreamObject):
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

        # process the frozen core option correctly as an integer or a list
        if not frozen:
            self.nfrozen = 0
        else:
            if isinteger(frozen):
                self.nfrozen = int(frozen)
            elif issequence(frozen, (int, np.integer)):
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
        self.rdm1_mo = None

        # dump flags here, because calling kernel is not mandatory
        self.dump_flags()

    def dump_flags(self):
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
        self.e_corr = emp2_rhf(self._intsfile, no, emo, logger)
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

        self.rdm1_mo = np.zeros((self.nmo, self.nmo))

        # Set the RHF density matrix for the frozen orbitals if applicable.
        nfrz = self.nfrozen
        if nfrz > 0:
            self.rdm1_mo[:nfrz, :nfrz] = 2.0 * np.eye(nfrz)

        # Now calculate the unrelaxed 1-RDM.
        no = self.nocc - nfrz
        emo = self.mo_energy[nfrz:]
        logger = lib.logger.new_logger(self)
        self.rdm1_mo[nfrz:, nfrz:] = \
            rdm1_rhf_unrelaxed(self._intsfile, no, emo, self.max_memory, logger)

        if ao_repr:
            return np.linalg.multi_dot([self.mo_coeff.T, self.rdm1_mo, self.mo_coeff])
        else:
            return self.rdm1_mo

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

        eigval, eigvec = np.linalg.eigh(self.rdm1_mo)
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

    # The class can be used with a context manager (with ... as ...:).
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self._intsfile:
            self._intsfile.delete()

    def kernel(self):
        '''
        Alias for the MP2 energy calculation.
        Does not need to be called to calculate the 1-RDM only.
        '''
        return self.calculate_energy()


class hdf5TempFile:
    '''
    Convenient handling of temporary HDF5 files.
    Instances can be used in a context manager:

        with hdf5TempFile() as ...:
            ...
    
    The "open" method returns a h5py file-like object,
    which can be used in a context manager itself.

    The "delete" method erases the temporary file from disk.
    '''

    def __init__(self, dir=lib.param.TMPDIR):
        '''
        Args:
            dir : directory to store the temporary file
        '''
        # We start by creating an empty temporary file.
        # Storing the file name instead of passing around f directly allows
        # h5py to use whichever driver it wants, instead of forcing it to do everything
        # through python's file interface.
        f = tempfile.NamedTemporaryFile(dir=dir, delete=False)
        f.close()
        self._filename = f.name
        self._libver = 'latest'
        self._file = None
    
    def open(self, mode):
        '''
        Opens the temporary file.
        Note: existing h5py.File instances to this file are closed!

        Args:
            mode : access mode as per h5py convention (e.g. "r", "w", "r+")
        
        Returns:
            an h5py.File instance (which can be used in a context manager itself)
        '''
        self.close()
        # Passing the file name (instead of a file-like object)
        # allows hdf5 to use its own drivers.
        self._file = h5py.File(self._filename, mode=mode, libver=self._libver)
        return self._file

    def close(self):
        '''
        Close the file for access (without deleting it).
        '''
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def delete(self):
        '''
        Delete the actual file on disk.
        '''
        self.close()
        if self._filename:
            os.remove(self._filename)
        self._filename = None

    def __enter__(self):
        '''
        Allows us to use instances in a context manager.
        '''
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        '''
        Delete file when exiting the context.
        '''
        self.delete()

    def __del__(self):
        '''
        Make sure to delete the file no later than at garbage collection.
        '''
        self.delete()


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
        Instance of hdf5TempFile containing the integrals in the dataset "ints_cholesky".
        Indexing order: [mo1, aux, mo2]
    '''
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,\
        auxmol._atm, auxmol._bas, auxmol._env)
    nmo1 = mo_coeff1.shape[1]
    nmo2 = mo_coeff2.shape[1]
    nauxfcns = auxmol.nao

    logger.info('')
    logger.info('*** RI integral transformation')
    logger.info('    MO dimensions: {0:d} x {1:d}'.format(nmo1, nmo2))
    logger.info('    Aux functions: {0:d}'.format(nauxfcns))

    intsfile = hdf5TempFile()
    with hdf5TempFile() as ints_tmp, intsfile.open('w') as h5ints_cholesky:

        h5ints_3c = ints_tmp.open('r+')

        logger.info('  > Calculating three center integrals in MO basis.')
        logger.info('    Temporary file: {0:s}'.format(h5ints_3c.filename))

        intor = mol._add_suffix('int3c2e')
        logger.debug('    intor = {0:s}'.format(intor))

        # Loop over shells of auxiliary functions.
        # AO integrals are calculated in memory and directly transformed to MO basis.
        ints_3c = h5ints_3c.create_dataset('ints_3c', (nauxfcns, nmo1, nmo2), dtype='f8')
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

        logger.info('  > Calculating fitted three center integrals.')
        logger.info('    Storage file: {0:s}'.format(h5ints_cholesky.filename))

        # Typically we need the matrix for a specific occupied MO i in MP2.
        # => i is the leading index for optimal I/O.
        ints_cholesky = h5ints_cholesky.create_dataset('ints_cholesky', (nmo1, nauxfcns, nmo2), dtype='f8')

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
        logger.info('    Batch size: {0:d} (of {1:d})'.format(bufsize, nmo1))

        # In batches:
        # - Read integrals from the temporary file.
        # - Instead of multiplying with L^-1, solve linear equation system.
        # - Store the "fitted" integrals in the integrals file.
        for istart in range(0, nmo1, bufsize):
            iend = min(istart+bufsize, nmo1)
            intsbuf = ints_3c[:, istart:iend, :]
            for i in range(istart, iend):
                ints_cholesky[i, :, :] = scipy.linalg.solve_triangular(L, intsbuf[:, i-istart, :], lower=True)

    logger.info('*** RI transformation finished')

    return intsfile


def emp2_rhf(intsfile, nocc, mo_energy, logger):
    '''
    Calculates the DF-MP2 energy with an RHF reference.

    Args:
        intsfile : contains the three center integrals in MO basis
        nocc : number of occupied orbitals
        mo_energy : energies of the molecular orbitals
        logger : Logger instance

    Returns:
        the MP2 correlation energy
    '''
    nvirt = len(mo_energy) - nocc
    logger.info('')
    logger.info('*** DF-MP2 energy')
    logger.info('    Occupied orbitals: {0:d}'.format(nocc))
    logger.info('    Virtual orbitals:  {0:d}'.format(nvirt))

    # Somewhat awkward workaround to perform division in the MP2 energy expression
    # through numpy routines. We precompute Eab[a, b] = mo_energy[a] + mo_energy[b].
    Eab = np.zeros((nvirt, nvirt))
    for a in range(nvirt):
        Eab[a, :] += mo_energy[nocc + a]
        Eab[:, a] += mo_energy[nocc + a]

    energy = 0.0
    with intsfile.open('r') as h5ints:
        logger.info('  > Calculating the energy')
        logger.info('    Integrals from file: {0:s}'.format(h5ints.filename))

        ints = h5ints['ints_cholesky']

        for i in range(nocc):
            ints3c_ia = ints[i, :, :]
            # contributions for occupied orbitals j < i
            for j in range(i):
                ints3c_jb = ints[j, :, :]
                Kab = np.matmul(ints3c_ia.T, ints3c_jb)
                DE = mo_energy[i] + mo_energy[j] - Eab
                Tab = Kab / DE
                energy += 4.0 * np.einsum('ab,ab', Tab, Kab)
                energy -= 2.0 * np.einsum('ab,ba', Tab, Kab)
            # contribution for j == i
            Kab = np.matmul(ints3c_ia.T, ints3c_ia)
            DE = 2.0 * mo_energy[i] - Eab
            Tab = Kab / DE
            energy += np.einsum('ab,ab', Tab, Kab)

    logger.note('*** DF-MP2 correlation energy: {0:.14f} Eh'.format(energy))
    return energy


def rdm1_rhf_unrelaxed(intsfile, nocc, mo_energy, max_memory, logger):
    '''
    Calculates the unrelaxed DF-MP2 density matrix with an RHF reference.

    Args:
        intsfile : contains the three center integrals
        nocc : number of occupied orbitals
        mo_energy : molecular orbital energies
        max_memory : memory threshold in MB
        logger : Logger instance
    
    Returns:
        matrix containing the unrelaxed 1-RDM
    '''
    nmo = len(mo_energy)
    nvirt = nmo - nocc
    logger.info('')
    logger.info('*** Unrelaxed one-particle density matrix for DF-MP2')
    logger.info('    Occupied orbitals: {0:d}'.format(nocc))
    logger.info('    Virtual orbitals:  {0:d}'.format(nvirt))

    # Precompute Eab[a, b] = mo_energy[a] + mo_energy[b] for division with numpy.
    Eab = np.zeros((nvirt, nvirt))
    for a in range(nvirt):
        Eab[a, :] += mo_energy[nocc + a]
        Eab[:, a] += mo_energy[nocc + a]

    # Density matrix initialized with the RHF contribution.
    P = np.zeros((nmo, nmo))
    P[:nocc, :nocc] = 2.0 * np.eye(nocc)

    with hdf5TempFile() as tfile, intsfile.open('r') as h5ints:
        h5amplitudes = tfile.open('r+')

        logger.info('  > Calculating the 1-RDM')
        logger.info('    Three center integrals from file: {0:s}'.format(h5ints.filename))
        logger.info('    Storing amplitudes in temporary file: {0:s}'.format(h5amplitudes.filename))

        batchsize = int((max_memory - lib.current_memory()[0]) * 1e6 / (2 * nocc * nvirt * 8))
        batchsize = min(nvirt, batchsize)
        if batchsize < 1:
            raise MemoryError('Insufficient memory (PYSCF_MAX_MEMORY).')
        logger.info('    Batch size: {0:d} (of {1:d})'.format(batchsize, nvirt))

        # For each occupied orbital i, all amplitudes are calculated once and stored on disk.
        # The occupied 1-RDM contribution is calculated in a batched algorithm.
        # More memory -> more effocient I/O.
        # The virtual contribution to the 1-RDM is calculated in memory.
        ints = h5ints['ints_cholesky']
        tiset = h5amplitudes.create_dataset('amplitudes', (nocc, nvirt, nvirt), dtype='f8')
        for i in range(nocc):
                ints3c_ia = ints[i, :, :]

                # Calculate amplitudes T^ij_ab for a given i and all j, a, b
                # Store the amplitudes in a file.
                for j in range(nocc):
                    ints3c_jb = ints[j, :, :]
                    Kab = np.matmul(ints3c_ia.T, ints3c_jb)
                    DE = mo_energy[i] + mo_energy[j] - Eab
                    Tab = Kab / DE
                    TCab = 4.0 * Tab - 2.0 * Tab.T
                    tiset[j, :, :] = Tab
                    # virtual 1-RDM contribution
                    P[nocc:, nocc:] += np.matmul(Tab, TCab.T)

                # Read batches of amplitudes from disk and calculate the occupied 1-RDM.
                for astart in range(0, nvirt, batchsize):
                    aend = min(astart+batchsize, nvirt)
                    tbatch1 = tiset[:, astart:aend, :]
                    tbatch2 = tiset[:, :, astart:aend]
                    P[:nocc, :nocc] -= 4.0 * np.tensordot(tbatch1, tbatch1, axes=([1, 2], [1, 2]))
                    P[:nocc, :nocc] += 2.0 * np.tensordot(tbatch1, tbatch2, axes=([1, 2], [2, 1]))
    
    logger.info('*** 1-RDM calculation finished')
    return P


def issequence(obj, datatype=None):
    '''
    Check if an object is a sequence.

    Args:
        obj : object to be tested
        datatype : if provided, check if the elements of the sequence have the given type

    Returns:
        whether the object is a sequence (and its elements have the right type)
    '''
    try:
        # For us, a sequence should:
        # 1) have a length
        # 2) support being iterated over
        len(obj)
        if datatype is None:
            for element in obj:
                pass
        else:
            for element in obj:
                if not isinstance(element, datatype):
                    return False
    except TypeError:
        return False
    else:
        return True


def isinteger(obj):
    '''
    Check if an object is an integer.

    Args:
        obj : object to be tested

    Returns:
        whether the object is an integer
    '''
    # A bool is also an int in python, but we don't want that.
    if isinstance(obj, bool):
        return False
    # These are actual ints we expect to encounter.
    elif isinstance(obj, (int, np.integer)):
        return True
    else:
        return False


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
