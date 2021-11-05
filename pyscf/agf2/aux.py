#xx Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#         George Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary space class and helper functions.
'''

import time
import numpy as np
import scipy.linalg.blas
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.lib.parameters import LARGE_DENOM


class AuxiliarySpace(object):
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

    def get_array(self, phys, out=None, chempot=0.0):
        ''' Expresses the auxiliaries as an array, i.e. the extended
            Fock matrix in AGF2 or Hamiltonian of ADC(2).

        Args:
            phys : 2D array
                Physical (1p + 1h) part of the matrix

        Kwargs:
            out : 2D array
                If provided, use to store output
            chempot : float
                Scale energies (by default, :attr:`chempot` is not used
                and energies retain their values). Default 0.0

        Returns:
            Array representing the coupling of the auxiliary space to
            the physical space
        '''

        _check_phys_shape(self, phys)

        dtype = np.result_type(phys.dtype, self.energy.dtype, self.coupling.dtype)

        if out is None:
            out = np.zeros((self.nphys+self.naux,)*2, dtype=dtype)

        sp = slice(None, self.nphys)
        sa = slice(self.nphys, None)

        out[sp,sp] = phys
        out[sp,sa] = self.coupling
        out[sa,sp] = self.coupling.conj().T
        out[sa,sa][np.diag_indices(self.naux)] = self.energy - chempot

        return out

    def dot(self, phys, vec, out=None, chempot=0.0):
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
                Scale energies (by default, :attr:`chempot` is not used
                and energies retain their values). Default 0.0

        Returns:
            ndarray with shape of :attr:`vec`
        '''

        _check_phys_shape(self, phys)

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

    def eig(self, phys, out=None, chempot=0.0):
        ''' Computes the eigenvalues and eigenvectors of the array
            returned by :func:`get_array`.

        Args:
            phys : 2D array
                Physical (1p + 1h) part of the matrix

        Kwargs:
            out : 2D array
                If provided, use to store output
            chempot : float
                Scale energies (by default, :attr:`chempot` is not used
                and energies retain their values). Default 0.0

        Returns:
            tuple of ndarrays (eigenvalues, eigenvectors)
        '''

        _check_phys_shape(self, phys)

        h = self.get_array(phys, chempot=chempot, out=out)
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

    def remove_uncoupled(self, tol):
        ''' Removes poles with very low spectral weight (uncoupled
            to the physical space) in-place.

        Args:
            tol : float
                Threshold for the spectral weight (squared norm)
        '''

        v = self.coupling
        w = np.linalg.norm(v, axis=0) ** 2

        arg = w >= tol

        self.energy = self.energy[arg]
        self.coupling = self.coupling[:,arg]

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

        dct = lib.chkfile.load(chkfile, key)

        return cls(dct['energy'], dct['coupling'], chempot=dct['chempot'])

    def copy(self):
        ''' Returns a copy of the current object.

        Returns:
            AuxiliarySpace
        '''
        energy = np.copy(self.energy)
        coupling = np.copy(self.coupling)
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

    def get_greens_function(self, phys):
        ''' Returns a :class:`GreensFunction` by solving the Dyson
            equation.

        Args:
            phys : 2D array
                Physical space (1p + 1h), typically the Fock matrix

        Returns:
            :class:`GreensFunction`
        '''

        w, v = self.eig(phys)
        v = v[:self.nphys]

        return GreensFunction(w, v, chempot=self.chempot)

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

        gf = self.get_greens_function(phys)
        return gf.make_rdm1(phys, chempot=chempot, occupancy=occupancy)

    def compress(self, phys=None, n=(None, 0), tol=1e-12):
        ''' Compress the auxiliaries via moments of the particle and
            hole Green's function and self-energy. Resulting :attr:`naux`
            depends on the chosen :attr:`n`.

        Kwargs:
            phys : 2D array or None
                Physical space (1p + 1h), typically the Fock matrix.
                Only required if :attr:`n[0]` is not None.
            n : tuple of int
                Compression level of the Green's function and
                self-energy, respectively.
            tol : float
                Linear dependecy tolerance. Default value is 1e-12

        Returns:
            :class:`SelfEnergy` with reduced auxiliary dimension.

        Raises:
            MemoryError if the compression according to Green's
            function moments will exceed the maximum allowed memory.
        '''

        ngf, nse = n
        se = self

        if nse is None and ngf is None:
            return self.copy()

        if nse is not None:
            se = compress_via_se(se, n=nse)

        if ngf is not None:
            se = compress_via_gf(se, phys, n=ngf, tol=tol)

        return se


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
        spectrum = np.zeros((grid.size, self.nphys, self.nphys), dtype=complex)
        blksize = 240

        p1 = 0
        for block in range(0, grid.size, blksize):
            p0, p1 = p1, min(p1 + blksize, grid.size)
            denom = grid[p0:p1,None] - (e_shifted + eta*1.0j)[None]
            spectrum[p0:p1] = lib.einsum('xk,yk,wk->wxy', v, v.conj(), 1./denom)

        return -1/np.pi * np.trace(spectrum.imag, axis1=1, axis2=2)

    def make_rdm1(self, chempot=None, occupancy=2):
        ''' Returns the first-order reduced density matrix associated
            with the Green's function.

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


def combine(*auxspcs):
    ''' Combine a set of :class:`AuxiliarySpace` objects. attr:`chempot`
        is inherited from the first element.
    '''

    nphys = [auxspc.nphys for auxspc in auxspcs]
    if not all([x == nphys[0] for x in nphys]):
        raise ValueError('Size of physical space must be the same to '
                         'combine AuxiliarySpace objects.')
    nphys = nphys[0]

    naux = sum([auxspc.naux for auxspc in auxspcs])
    dtype = np.result_type(*[auxspc.coupling for auxspc in auxspcs])

    energy = np.zeros((naux,))
    coupling = np.zeros((nphys, naux), dtype=dtype)

    p1 = 0
    for auxspc in auxspcs:
        p0, p1 = p1, p1 + auxspc.naux
        energy[p0:p1] = auxspc.energy
        coupling[:,p0:p1] = auxspc.coupling

    auxspc = auxspcs[0].__class__(energy, coupling, chempot=auxspcs[0].chempot)

    return auxspc


def davidson(auxspc, phys, chempot=None, nroots=1, which='SM', tol=1e-14, maxiter=None, ntrial=None):
    ''' Diagonalise the result of :func:`AuxiliarySpace.get_array` using
        the sparse :func:`AuxiliarySpace.dot` method, with the Davidson
        algorithm.

        This algorithm may perform poorly for IPs or EAs if they are
        not extremal eigenvalues, which they are not in standard AGF2.

    Args:
        auxspc : AuxiliarySpace or subclass
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
             `LM` : Largest (in magnitude) eigenvalues.
             `SM` : Smallest (in magnitude) eigenvalues.
             `LA` : Largest (algebraic) eigenvalues.
             `SA` : Smallest (algebraic) eigenvalues.
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

    _check_phys_shape(auxspc, phys)
    dim = auxspc.nphys + auxspc.naux

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

    matvec = lambda x: auxspc.dot(phys, np.asarray(x))
    diag = np.concatenate([np.diag(phys), auxspc.energy])
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

    conv, w, v = lib.davidson1(matvec, guess, diag, tol=tol, nroots=nroots,
                               max_space=ntrial, max_cycle=maxiter, pick=pick)

    return conv, w, v


def _band_lanczos(se_occ, n=0, max_memory=None):
    ''' Perform the banded Lanczos algorithm for compression of a
        self-energy according to consistency in its separate
        particle and hole moments.
    '''

    nblk = n+1
    nphys, naux = se_occ.coupling.shape
    bandwidth = nblk * nphys

    q = np.zeros((bandwidth, naux))
    t = np.zeros((bandwidth, bandwidth))
    r = np.zeros((naux))

    # cholesky qr factorisation of v.T
    coupling = se_occ.coupling
    x = np.dot(coupling, coupling.T)

    try:
        v_tri = np.linalg.cholesky(x).T
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(x)
        w[w < 1e-20] = 1e-20
        x_posdef = np.dot(np.dot(v, np.diag(w)), v.T)
        v_tri = np.linalg.cholesky(x_posdef).T

    q[:nphys] = np.dot(np.linalg.inv(v_tri).T, coupling)

    for i in range(bandwidth):
        r[:] = se_occ.energy * q[i]

        start = max(i-nphys, 0)
        if start != i:
            r -= np.dot(t[i,start:i], q[start:i])

        for j in range(i, min(i+nphys, bandwidth)):
            t[i,j] = t[j,i] = np.dot(r, q[j])
            # r := -t[i,j] * q[j] + r
            scipy.linalg.blas.daxpy(q[j], r, a=-t[i,j])

        if (i+nphys) < bandwidth:
            len_r = np.linalg.norm(r)
            t[i,i+nphys] = t[i+nphys,i] = len_r
            q[i+nphys] = r / (len_r + 1./LARGE_DENOM)

    return v_tri, t

def _compress_part_via_se(se_occ, n=0):
    ''' Compress the auxiliaries of the occupied or virtual part of
        the self-energy according to consistency in its moments.
    '''

    if se_occ.nphys > se_occ.naux:
        # breaks this version of the algorithm and is also pointless
        e = se_occ.energy.copy()
        v = se_occ.coupling.copy()
    else:
        v_tri, t = _band_lanczos(se_occ, n=n)
        e, v = np.linalg.eigh(t)
        v = np.dot(v_tri.T, v[:se_occ.nphys])

    return e, v

def _compress_via_se(se, n=0):
    ''' Compress the auxiliaries of the seperate occupied and
        virtual parts of the self-energy according to consistency
        in its moments.
    '''

    if se.naux == 0:
        return se.energy, se.coupling

    se_occ = se.get_occupied()
    se_vir = se.get_virtual()

    e = []
    v = []

    if se_occ.naux > 0:
        e_occ, v_occ = _compress_part_via_se(se_occ, n=n)
        e.append(e_occ)
        v.append(v_occ)

    if se_vir.naux > 0:
        e_vir, v_vir = _compress_part_via_se(se_vir, n=n)
        e.append(e_vir)
        v.append(v_vir)

    e = np.concatenate(e, axis=0)
    v = np.concatenate(v, axis=-1)

    return e, v

def compress_via_se(se, n=0):
    ''' Compress the auxiliaries of the seperate occupied and
        virtual parts of the self-energy according to consistency
        in its moments.

    Args:
        se : SelfEnergy
            Auxiliaries of the self-energy

    Kwargs:
        n : int
            Truncation parameter, conserves the seperate particle
            and hole moments to order 2*n+1.

    Returns:
        :class:`SelfEnergy` with reduced auxiliary dimension

    Ref:
        [1] H. Muther, T. Taigel and T.T.S. Kuo, Nucl. Phys., 482,
            1988, pp. 601-616.
        [2] D. Van Neck,  K. Piers and M. Waroquier, J. Chem. Phys.,
            115, 2001, pp. 15-25.
        [3] H. Muther and L.D. Skouras, Nucl. Phys., 55, 1993,
            pp. 541-562.
        [4] Y. Dewulf, D. Van Neck, L. Van Daele and M. Waroquier,
            Phys. Lett. B, 396, 1997, pp. 7-14.
    '''

    e, v = _compress_via_se(se, n=n)
    se_red = SelfEnergy(e, v, chempot=se.chempot)

    return se_red


def _build_projector(se, phys, n=0, tol=1e-12):
    ''' Builds the vectors which project the auxiliary space into a
        compress one with consistency in the seperate particle and
        hole moments up to order 2n+1.
    '''

    _check_phys_shape(se, phys)

    nphys, naux = se.coupling.shape
    w, v = se.eig(phys)

    def _part(w, v, s):
        en = w[s][None] ** np.arange(n+1)[:,None]
        v = v[:,s]
        p = np.einsum('xi,pi,ni->xpn', v[nphys:], v[:nphys], en)
        return p.reshape(naux, nphys*(n+1))

    p = np.hstack((_part(w, v, w < se.chempot),
                   _part(w, v, w >= se.chempot)))

    norm = np.linalg.norm(p, axis=0, keepdims=True)
    norm[np.absolute(norm) == 0] = 1./LARGE_DENOM

    p /= norm
    w, p = np.linalg.eigh(np.dot(p, p.T))
    p = p[:, w > tol]
    nvec = p.shape[1]

    p = np.block([[np.eye(nphys), np.zeros((nphys, nvec))],
                  [np.zeros((naux, nphys)), p]])

    return p

def _compress_via_gf(se, phys, n=0, tol=1e-12):
    ''' Compress the auxiliaries of the seperate occupied and
        virtual parts of the self-energy according to consistency
        in the moments of the Green's function
    '''

    nphys = se.nphys

    p = _build_projector(se, phys, n=n, tol=tol)
    h_tilde = np.dot(p.T, se.dot(phys, p))
    p = None

    e, v = np.linalg.eigh(h_tilde[nphys:,nphys:])
    v = np.dot(h_tilde[:nphys,nphys:], v)

    return e, v

def compress_via_gf(se, phys, n=0, tol=1e-12):
    ''' Compress the auxiliaries of the seperate occupied and
        virtual parts of the self-energy according to consistency
        in the moments of the Green's function

    Args:
        se : SelfEnergy
            Auxiliaries of the self-energy
        phys : 2D array
            Physical space (1p + 1h), typically the Fock matrix

    Kwargs:
        n : int
            Truncation parameter, conserves the seperate particle
            and hole moments to order 2*n+1.
        tol : float
            Linear dependecy tolerance. Default value is 1e-12

    Returns:
        :class:`SelfEnergy` with reduced auxiliary dimension
    '''

    e, v = _compress_via_gf(se, phys, n=n, tol=tol)
    se_red = SelfEnergy(e, v, chempot=se.chempot)

    return se_red


def _check_phys_shape(auxspc, phys):
    if np.shape(phys) != (auxspc.nphys, auxspc.nphys):
        raise ValueError('Size of physical space must be the same as '
                         'leading dimension of couplings.')
