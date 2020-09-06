 
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Oliver Backhouse <olbackhouse@gmail.com>
#

'''
Auxiliary space class and helper functions.
'''

import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__


class AuxiliarySpace:
    ''' Simple container to hold the energies, couplings and chemical
        potential associated with an auxiliary space.

    Attributes:
        energy : 1D array
            Energies of the poles
        coupling : 2D array
            Coupling vector of the poles to each physical state
        chempot : float
            Chemical potental associated with the energies
    '''

    def __init__(self, energy, coupling, chempot=0.0):
        self.energy = np.asarray(energy)
        self.coupling = np.asarray(coupling, order='C')
        self.chempot = chempot
        self.sort()

    def sort(self):
        ''' Sort in-place via the energies to make slicing easier.
        '''
        arg = np.argsort(self.energy)
        self.energy = self.energy[arg]
        self.coupling = self.coupling[:,arg]

    def check_sanity(self):
        ''' Perform some debugging checks.
        '''
        #NOTE: convert the expensive calls here to tests later
        assert self.energy.ndim == 1
        assert self.coupling.ndim == 2
        assert self.coupling.shape[1] == self.energy.shape[0]
        assert self.get_occupied().coupling.shape == self.coupling[:,self.energy<self.chempot].shape
        assert self.get_virtual().coupling.shape == self.coupling[:,self.energy>=self.chempot].shape

    def real_freq_spectrum(self, *args, **kwargs):
        ''' See subclasses.
        '''
        raise NotImplementedError

    def compress(self, *args, **kwargs):
        ''' See subclasses.
        '''
        raise NotImplementedError

    def get_occupied(self):
        ''' Returns a copy of the current AuxiliarySpace object
            containing only the poles with energy less than the 
            chemical potential. The object should already be sorted.

        Returns:
            :class:`AuxiliarySpace` with only the occupied auxiliaries
        '''
        nocc = np.searchsorted(self.energy, self.chempot)
        energy = np.copy(self.energy[:nocc])
        coupling = np.copy(self.coupling[:,:nocc])
        return self.__class__(energy, coupling, chempot=self.chempot)
        
    def get_virtual(self):
        ''' Returns a copy of the current AuxiliarySpace object
            containing only the poles with energy greater than the
            chemical potential. The object should already be sorted.

        Returns:
            :class:`AuxiliarySpace` with only the virtual auxiliaries
        '''
        nocc = np.searchsorted(self.energy, self.chempot)
        energy = np.copy(self.energy[nocc:])
        coupling = np.copy(self.coupling[:,nocc:])
        return self.__class__(energy, coupling, chempot=self.chempot)

    def get_array(self, phys, out=None, chempot=None):
        ''' Expresses the auxiliaries as an array, i.e. the extended
            Fock matrix in AGF2 or Hamiltonian of ADC(2).

        Args:
            phys : 2D array
                Physical (1p + 1h) part of the matrix

        Kwargs:
            out : 2D array
                If provided, use to store output
            chempot : float
                If provided, use instead of :attr:`self.chempot`

        Returns:
            Array representing the coupling of the auxiliary space to 
            the physical space
        '''
        #NOTE: memory check

        _check_phys_shape(self, phys)

        if chempot is None:
            chempot = self.chempot

        e_shifted = self.energy - chempot

        if out is None:
            out = np.zeros((self.nphys+self.naux,)*2)

        sp = slice(None, self.nphys)
        sa = slice(self.nphys, None)

        out[sp,sp] = phys
        out[sp,sa] = self.coupling
        out[sa,sp] = self.coupling.conj().T
        out[sa,sa][np.diag_indices(self.naux)] = e_shifted

        return out

    def dot(self, phys, vec, out=None, chempot=None):
        ''' Returns the dot product of :func:`get_array` with a vector.

        Args:
            phys : 2D array
                Physical (1p + 1h) part of the matrix
            vec : ndarray
                Vector to compute dot product with

        Kwargs:
            out : 2D array
                If provided, use to store output
            chempot : float
                If provided, use instead of :attr:`self.chempot`

        Returns:
            ndarray with shape of :attr:`vec`
        '''

        _check_phys_shape(self, phys)

        if chempot is None:
            chempot = self.chempot

        vec = np.asarray(vec)
        input_shape = vec.shape
        vec = vec.reshape((self.nphys+self.naux, -1))
        dtype = np.result_type(self.coupling.dtype, vec.dtype)

        sp = slice(None, self.nphys)
        sa = slice(self.nphys, None)

        if out is None:
            out = np.zeros(vec.shape, dtype=dtype)
        out = out.reshape(vec.shape)

        out[sp]  = np.dot(phys, vec[sp])
        out[sp] += np.dot(self.coupling, vec[sa])

        out[sa]  = np.dot(vec[sp].T, self.coupling).conj().T
        out[sa] += (self.energy[:,None] - chempot) * vec[sa]

        out = out.reshape(input_shape)

        return out

    def eig(self, phys, chempot=None):
        ''' Computes the eigenvalues and eigenvectors of the array
            returned by :func:`get_array`.

        Args:
            phys : 2D array
                Physical (1p + 1h) part of the matrix

        Kwargs:
            chempot : float
                If provided, use instead of :attr:`self.chempot`

        Returns:
            tuple of ndarrays (eigenvalues, eigenvectors)
        '''

        _check_phys_shape(self, phys)

        h = self.get_array(phys, chempot=chempot)
        w, v = np.linalg.eigh(h)

        return w, v

    def moment(self, n, squeeze=True):
        ''' Builds the nth moment of the spectral distribution.

        Args:
            n : int or list of int
                Moment(s) to compute

        Kwargs:
            squeeze : bool
                If True, use :func:`np.squeeze` on output so that in
                the case of :attr:`n` being an int, a 2D array is
                returned. If False, output is always 3D. Default True.

        Returns:
            ndarray of moments
        '''

        n = np.asarray(n)
        n = n.reshape(n.size)

        energy_factored = self.energy[None] ** n[:,None]
        v = self.coupling
        moms = lib.einsum('xk,yk,nk->nxy', v, v.conj(), energy_factored)

        if squeeze:
            moms = np.squeeze(moms)

        return moms

    def save(self, chkfile, key=None):
        ''' Saves the auxiliaries in chkfile

        Args:
            chkfile : str
                Name of chkfile
            key : str
                Key to be used in h5py object. It can contain "/" to
                represent the path in the HDF5 storage structure.
        '''
        
        if key is None:
            key = 'aux'

        lib.chkfile.dump(chkfile, key, self.__dict__)

    @classmethod
    def load(cls, chkfile, key=None):
        ''' Loads the auxiliaries from a chkfile

        Args:
            chkfile : str
                Name of chkfile
            key : str
                Key to be used in h5py object. It can contain "/" to
                represent the path in the HDF5 storage structure.
        '''

        if key is None:
            key = 'aux'

        aux = lib.chkfile.load(chkfile, key)

        return cls(aux['energy'], aux['coupling'], chempot=aux['chempot'])

    def copy(self):
        ''' Returns a copy of the current object.

        Returns:
            AuxiliarySpace
        '''
        energy = np.copy(self.energy)
        coupling = np.copy(self.energy)
        return self.__class__(energy, coupling, chempot=self.chempot)

    @property
    def nphys(self):
        return self.coupling.shape[0]

    @property
    def naux(self):
        return self.coupling.shape[1]


class SelfEnergy(AuxiliarySpace):
    ''' Defines a self-energy represented as a :class:`AuxiliarySpace`
        object.
    '''

    def real_freq_spectrum(self, grid, eta=0.02):
        raise ValueError('Convert SelfEnergy to GreensFunction before '
                         'building a spectrum.')

    def get_greens_function(self, phys, chempot=None):
        ''' Returns a :class:`GreensFunction` by solving the Dyson
            equation.

        Args:
            phys : 2D array
                Physical space (1p + 1h), typically the Fock matrix

        Kwargs:
            chempot : float
                If provided, use instead of :attr:`self.chempot`
        '''

        if chempot is None:
            chempot = self.chempot

        w, v = self.eig(phys, chempot=chempot)
        v = v[:self.nphys]

        return GreensFunction(w, v, chempot=chempot)

    def make_rdm1(self, phys, chempot=None, occupancy=2):
        ''' Returns the first-order reduced density matrix associated
            with the self-energy via the :class:`GreensFunction`.

        Args:
            phys : 2D array
                Physical space (1p + 1h), typically the Fock matrix

        Kwargs:
            chempot : float
                If provided, use instead of :attr:`self.chempot`
            occupancy : int
                Occupancy of the states, i.e. 2 for RHF and 1 for UHF
        '''

        gf = get_greens_function(phys, chempot=chempot)
        return gf.make_rdm1(phys, chempot=chempot, occupancy=occupancy)

    def compress(self, nmoms):
        raise NotImplementedError #TODO


class GreensFunction(AuxiliarySpace):
    ''' Defines a Green's function represented as a
        :class:`AuxiliarySpace` object.
    '''

    def real_freq_spectrum(self, grid, eta=0.02):
        ''' Express the auxiliaries as a spectral function on the real
            frequency axis.

        Args:
            grid : 1D array
                Real frequency grid

        Kwargs:
            eta : float
                Peak broadening factor in Hartrees. Default is 0.02.

        Returns:
            ndarray of the spectrum, with the first index being the
            frequency
        '''

        e_shifted = self.energy - self.chempot
        v = self.coupling
        spectrum = np.zeros((grid.size, self.nphys, self.nphys), dtype=float)
        blksize = 240 #NOTE option for the block size?

        p1 = 0
        for block in range(0, grid.size, blksize):
            p0, p1 = p1, min(p1 + blksize, grid.size)
            denom = grid[p0:p1,None] - (e_shifted + eta*1.0j)[None]
            spectrum[p0:p1] = lib.einsum('xk,yk,wk->wxy', v, v.conj(), 1./denom)

        return -spectrum.imag / np.pi

    def make_rdm1(self, phys, chempot=None, occupancy=2):
        ''' Returns the first-order reduced density matrix associated
            with the Green's function.

        Args:
            phys : 2D array
                Physical space (1p + 1h), typically the Fock matrix

        Kwargs:
            chempot : float
                If provided, use instead of :attr:`self.chempot`
            occupancy : int
                Occupancy of the states, i.e. 2 for RHF and 1 for UHF
        '''

        if chempot is None:
            chempot = self.chempot

        arg = self.energy < chempot
        v_occ = self.coupling[:,arg]
        rdm1 = np.dot(v_occ, v_occ.T.conj()) * occupancy

        return rdm1

    def compress(self, *args, **kwargs):
        raise ValueError('Compression must be performed on SelfEnergy '
                         'rather than GreensFunction.')


def combine(*auxs):
    ''' Combine a set of :class:`AuxiliarySpace` objects. attr:`chempot`
        is inherited from the first element.
    '''

    nphys = [aux.nphys for aux in auxs]
    if not all([x == nphys[0] for x in nphys]):
        raise ValueError('Size of physical space must be the same to '
                         'combine AuxiliarySpace objects.')
    nphys = nphys[0]

    naux = sum([aux.naux for aux in auxs])
    dtype = np.result_type(*[aux.coupling for aux in auxs])

    energy = np.zeros((naux,))
    coupling = np.zeros((nphys[0], naux), dtype=dtype)

    p1 = 0
    for aux in auxs:
        p0, p1 = p1, p1 + aux.naux
        energy[p0:p1] = aux.energy
        coupling[:,p0:p1] = aux.coupling

    aux = auxs[0].__class__(energy, coupling, chempot=auxs[0].chempot)

    return aux


def davidson(aux, phys, chempot=None, nroots=1, which='SM', tol=1e-14, maxiter=None, ntrial=None):
    ''' Diagonalise the result of :func:`AuxiliarySpace.get_array` using
        the sparse :func:`AuxiliarySpace.dot` method, with the Davidson
        algorithm.

        This algorithm may perform poorly for IPs or EAs if they are
        not extremal eigenvalues, which they are not in standard AGF2.

    Args:
        aux : AuxiliarySpace or subclass
            Auxiliary space object to solve for
        phys : 2D array
            Physical space (1p + 1h), typically the Fock matrix

    Kwargs:
        chempot : float
            If provided, use instead of :attr:`self.chempot`
        nroots : int
            Number of roots to solve for. Default 1.
        which : str
            Which eigenvalues to solve for. Options are:
             ‘LM’ : Largest (in magnitude) eigenvalues.
             ‘SM’ : Smallest (in magnitude) eigenvalues.
             ‘LA’ : Largest (algebraic) eigenvalues.
             ‘SA’ : Smallest (algebraic) eigenvalues.
            Default 'SM'.
        tol : float
            Convergence threshold
        maxiter : int
            Maximum number of iterations. Default 10*dim
        ntrial : int
            Maximum number of trial vectors. Default 
            min(dim, max(2*nroots+1, 20))

    Returns:
        tuple of ndarrays (eigenvalues, eigenvectors)
    '''
    #NOTE: I think a lot of this is pulled from the adc code. Can we just inherit anything?

    _check_phys_shape(aux, phys)
    dim = aux.nphys + aux.naux

    if maxiter is None:
        maxiter = 10 * dim

    if ntrial is None:
        ntrial = min(dim, max(2*nroots+1, 20))

    if which not in ['SM', 'LM', 'SA', 'LA']:
        raise ValueError(which)

    if which in ['SM', 'LM']:
        abs_op = np.absolute
    else:
        abs_op = lambda x: x

    if which in ['SM', 'SA']:
        order = 1
    else:
        order = -1

    matvec = lambda x: aux.dot(phys, np.asarray(x))
    diag = np.concatenate([np.diag(phys), aux.energy])
    guess = [np.zeros((dim)) for n in range(nroots)]

    mask = np.argsort(abs_op(diag))[::order]
    for i in range(nroots):
        guess[i][mask[i]] = 1

    def pick(w, v, nroots, callback):
        mask = np.argsort(abs_op(w))
        mask = mask[::order]
        w = w[mask]
        v = v[:,mask]
        return w, v, 0

    conv, w, v = util.davidson(matvec, guess, diag, tol=tol, nroots=nroots,
                               max_space=ntrial, max_cycle=maxiter, pick=pick)

    if not np.all(conv):
        pass #TODO: warn

    return w, v


def _check_phys_shape(aux, phys):
    if phys.shape != (aux.nphys, aux.nphys):
        raise ValueError('Size of physical space must be the same as '
                         'leading dimension of couplings.')
