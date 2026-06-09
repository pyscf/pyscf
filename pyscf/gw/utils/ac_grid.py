#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
#

"""
Grids and analytical continuation functions for GW and RPA.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union
from scipy.optimize import least_squares
import warnings

from pyscf.lib import chkfile, logger


# analytical continuation interface
class AC_Method(ABC):
    """
    Base class for AC methods

    Attributes
    ----------
    shape : tuple
        Shape of the data array that was passed to ac_fit, excluding the frequency axis.
    """

    _data_keys : set
    _method : str

    def __init__(self, *args, **options):
        if args:
            warnings.warn(f"{len(args)} unused positional arguments passed to the AC_method constructor.")
        if options:
            warnings.warn(f"Unused keyword arguments passed to the AC_method constructor: {options}")
        self.shape = ()

    @abstractmethod
    def ac_fit(self, data: np.ndarray, omega: np.ndarray, axis: int = -1):
        """The kernel of the AC method. This function should be implemented in the derived class.

        Parameters
        ----------
        data : np.ndarray
            Data to be fit, e.g. self energy
        omega : np.ndarray[np.double]
            1D imaginary frequency grid
        axis : int, optional
            Indicates which axis of the data array corresponds to the frequency axis, by default -1.
            Example: data.shape is (nmo, nmo, nw), call ac_fit(data, omega, axis=-1)
                     data.shape is (nw, nmo, nmo), call ac_fit(data, omega, axis=0)
        """
        raise NotImplementedError

    @abstractmethod
    def ac_eval(self, freqs: np.ndarray, axis: int = -1):
        """After you call ac_fit, you can call this function to evaluate the AC at
           arbitrary complex frequencies.

        Parameters
        ----------
        freqs : np.ndarray[np.complex128] | complex
            1D array of complex frequencies, or a single complex frequency
        axis : int, optional
            Indicates which axis of the output array should correspond to the frequency axis, by default -1.
            Example: if you want (nmo, nmo, nfreq), call ac_eval(freqs, axis=-1)
                     if you want (nfreq, nmo, nmo), call ac_eval(data, freqs, axis=0)
            If freqs is a scalar, the output shape will be the same as the input data shape.

        Returns
        -------
        np.ndarray
            Interpolated/approximated values at the new complex frequencies
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, multiidx: Union[Tuple[int], slice]):
        """
        Get a slice of the AC method. This function should be implemented in the derived class.
        Defining this method lets you do things like acobj[p,q] or acobj[p,:].

        Parameters
        ----------
        multiidx : Tuple[int] | slice
            Index or slice object for the AC method.

        Returns
        -------
        object
            New instance of the AC method with the selected slice.
        """
        raise NotImplementedError

    @abstractmethod
    def diagonal(self, axis1=0, axis2=1):
        """Create a new instance of the AC method with only the diagonal elements
           of the self.coeff tensor. Convenient for getting diagonal of self-energy
           after calculating the full self-energy matrix.

           This behaves more or less the same as np.diagonal.

        Parameters
        ----------
        axis1 : int, optional
            First axis, by default 0
        axis2 : int, optional
            Second axis, by default 1

        Returns
        -------
        object
            New instance of the AC method with only the diagonal elements.
        """
        raise NotImplementedError

    def save(self, chkfilename: str, dataname: str = 'ac'):
        """Save the AC object and coefficients to an HDF5 file.

        Parameters
        ----------

        """
        data_dic = { key : getattr(self, key) for key in self._data_keys }
        data_dic['method'] = self._method
        chkfile.dump(chkfilename, dataname, data_dic)


def load_ac(chkfilename: str, dataname: str = 'ac') -> AC_Method:
    """Load an AC object from an HDF5 file.

    Parameters
    ----------
    chkfilename : str
        Path to the HDF5 file
    dataname : str, optional
        Name of the dataset in the HDF5 file, by default 'ac'

    Returns
    -------
    AC_Method
        The loaded AC object
    """
    data_dic = chkfile.load(chkfilename, dataname)
    method = data_dic.pop('method')
    ac_class = {'twopole': TwoPoleAC, 'pade': PadeAC}[method.decode()]

    # Instantiate the AC object
    acobj = ac_class.__new__(ac_class)
    # Update the AC object with the data from the HDF5 file
    acobj.__dict__.update(data_dic)
    acobj.shape = acobj.coeff.shape[1:]
    return acobj


# grids
def _get_scaled_legendre_roots(nw, x0=0.5):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [0, inf)
    Ref: www.cond-mat.de/events/correl19/manuscripts/ren.pdf

    Returns:
        freqs : 1D array
        wts : 1D array
    """
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    freqs_new = x0 * (1.0 + freqs) / (1.0 - freqs)
    wts = wts * 2.0 * x0 / (1.0 - freqs) ** 2
    return freqs_new, wts


def _get_clenshaw_curtis_roots(nw):
    """
    Clenshaw-Curtis qaudrature on [0,inf)
    Ref: J. Chem. Phys. 132, 234114 (2010)
    Returns:
        freqs : 1D array
        wts : 1D array
    """
    freqs = np.zeros(nw)
    wts = np.zeros(nw)
    a = 0.2
    for w in range(nw):
        t = (w + 1.0) / nw * np.pi * 0.5
        freqs[w] = a / np.tan(t)
        if w != nw - 1:
            wts[w] = a * np.pi * 0.5 / nw / (np.sin(t) ** 2)
        else:
            wts[w] = a * np.pi * 0.25 / nw / (np.sin(t) ** 2)
    return freqs[::-1], wts[::-1]


# Pade fitting
def _get_ac_idx(nw, npts=18, step_ratio=2.0 / 3.0, idx_start=1):
    """Get an array of indices, with stepsize decreasing.

    Parameters
    ----------
    nw : int
        number of frequency points
    npts : int, optional
        final number of selected points, by default 18
    step_ratio : float, optional
        final stepsize / initial stepsize, by default 2.0/3.0
    idx_start : int, optional
        first index of final array, by default 1

    Returns
    -------
    np.ndarray
        an array for indexing frequency and omega.
    """
    if nw <= npts:
        raise ValueError('nw (%s) should be larger than npts (%s)' % (nw, npts))
    steps = np.linspace(1.0, step_ratio, npts)
    steps /= np.sum(steps)
    steps = np.cumsum(steps * nw)
    steps += idx_start - steps[0]
    steps = np.round(steps).astype(int)
    return steps


def pade_thiele_ndarray(freqs, zn, coeff):
    """NDarray-friendly analytic continuation using Pade-Thiele method.

    Parameters
    ----------
    freqs : np.ndarray, shape (nfreqs,), complex
        Points in the complex plane at which to evaluate the analytic continuation.
    zn : np.ndarray, shape (ncoeff,), complex
        interpolation points
    coeff : np.ndarray, shape (ncoeff, M1, M2, ...,), complex
        Pade-Thiele coefficients

    Returns
    -------
    np.ndarray, shape (nfreqs, M1, M2, ...,), complex
        Pade-Thiele analytic continuation evaluated at `freqs`
    """
    ncoeff = len(coeff)

    if freqs.ndim != 1 or zn.ndim != 1:
        raise ValueError('freqs and zn must be 1D arrays')
    if ncoeff != len(zn):
        raise ValueError('coeff and zn must have the same length')
    freqs_broadcasted = np.expand_dims(freqs, tuple(range(1, coeff.ndim)))

    X = coeff[-1] * (freqs_broadcasted - zn[-2])
    # X has shape (nfreqs, M1, M2, ...)

    for i in range(ncoeff - 1):
        idx = ncoeff - i - 1
        X = coeff[idx] * (freqs_broadcasted - zn[idx - 1]) / (1.0 + X)
    X = coeff[0] / (1.0 + X)
    return X


def thiele_ndarray(fn, zn):
    """Iterative Thiele algorithm to compute coefficients of Pade approximant

    Parameters
    ----------
    fn : np.ndarray, shape (nw, N1, N2, ...,), complex
        Function values at the points zn
    zn : np.ndarray, shape(nw,), complex
        Points in the complex plane used to compute fn

    Returns
    -------
    np.ndarray, shape(nw, N1, N2, ...), complex
        Coefficients of Pade approximant
    """
    nw = len(zn)
    # No need to allocate coeffs since g = coeffs at the end.
    g = fn.copy()
    # g has shape (nw, N1, N2, ...)

    zn_broadcasted = np.expand_dims(zn, tuple(range(1, g.ndim)))
    # zn_broadcasted has shape (nw, 1, 1, ..., 1)

    for i in range(1, nw):
        # At this stage, coeffs[i-1] is already computed.
        # coeffs[i-1] = g[i-1]

        g[i:] = (g[i - 1] - g[i:]) / ((zn_broadcasted[i:] - zn_broadcasted[i - 1]) * g[i:])
    return g


class PadeAC(AC_Method):
    """
    Analytic continuation to real axis using a Pade approximation
    from Thiele's reciprocal difference method
    Reference: J. Low Temp. Phys. 29, 179 (1977)
    """

    _data_keys = {'npts', 'step_ratio', 'coeff', 'omega', 'idx'}
    _method = 'pade'

    def __init__(self, *args, npts=18, step_ratio=2.0 / 3.0, **options):
        """Constructor for PadeAC class

        Parameters
        ----------
        npts : int, optional
            Number of frequency points to use for AC, by default 18
        step_ratio : float, optional
            Frequency grid step ratio, by default 2.0/3.0
        """
        super().__init__(*args, **options)
        self.step_ratio = step_ratio
        if npts % 2 != 0:
            warnings.warn(f'Pade AC: npts should be even, but {npts} was given. Using {npts-1} instead.')
            npts -= 1
            assert npts > 0
        self.npts = npts
        self.coeff = None  # Pade fitting coefficient
        self.omega = None  # input frequency grids
        self.idx = None  # idx of frequency grids used for fitting

    @property
    def omega_fit(self):
        """Return the frequency grid used for fitting."""
        if self.omega is None or self.idx is None:
            return None
        return self.omega[self.idx]

    def ac_fit(self, data, omega, axis=-1):
        """Compute Pade-Thiele AC coefficients for the given data and omega.

        Parameters
        ----------
        data : np.ndarray
            Data to be fit, e.g. self energy
        omega : np.ndarray[np.double]
            1D imaginary frequency grid
        axis : int, optional
            Indicates which axis of the data array corresponds to the frequency axis, by default -1.
            Example: data.shape is (nmo, nmo, nw), call ac_fit(data, omega, axis=-1)
                     data.shape is (nw, nmo, nmo), call ac_fit(data, omega, axis=0)
        """
        self.omega = np.asarray(omega).copy()
        nw = self.omega.size
        data = np.asarray(data)

        assert omega.ndim == 1
        assert data.shape[axis] == nw

        if self.idx is None:
            self.idx = _get_ac_idx(nw, npts=self.npts, step_ratio=self.step_ratio)

        sub_omega = self.omega[self.idx]
        sub_data = np.moveaxis(data, axis, 0)[self.idx]
        self.coeff = thiele_ndarray(sub_data, sub_omega)
        self.shape = self.coeff.shape[1:]

    def ac_eval(self, freqs, axis=-1):
        """Evaluate Pade AC at arbitrary complex frequencies.

        Parameters
        ----------
        freqs : np.ndarray[np.complex128]
            1D array of complex frequencies
        axis : int, optional
            Indicates which axis of the output array should correspond to the frequency axis, by default -1.
            Example: if you want (nmo, nmo, nfreq), call ac_eval(freqs, axis=-1)
                     if you want (nfreq, nmo, nmo), call ac_eval(data, freqs, axis=0)

        Returns
        -------
        np.ndarray
            Pade-Thiele AC evaluated at `freqs`
        """
        if self.coeff is None or self.omega is None:
            raise ValueError('Pade coefficients not set. Call ac_fit first.')
        is_scalar = np.isscalar(freqs)
        freqs = np.asarray(freqs)

        # Handle scalar freqs
        if np.ndim(freqs) == 0:
            freqs = np.array([freqs])

        assert freqs.ndim == 1
        if freqs.dtype != np.complex128:
            freqs = freqs.astype(np.complex128)
        X = pade_thiele_ndarray(freqs, self.omega[self.idx], self.coeff)
        if not is_scalar:
            return np.moveaxis(X, 0, axis)
        else:
            return X[0]

    def __getitem__(self, multi_idx):
        assert self.coeff is not None
        new_obj = self.__class__(npts=self.npts, step_ratio=self.step_ratio)
        new_obj.coeff = self.coeff[(slice(None, None, None), *np.index_exp[multi_idx])]
        new_obj.omega = self.omega
        new_obj.idx = self.idx
        new_obj.shape = new_obj.coeff.shape[1:]

        return new_obj

    def diagonal(self, axis1=0, axis2=1):
        assert self.coeff is not None
        assert len(self.shape) >= 2
        new_obj = self.__class__(npts=self.npts, step_ratio=self.step_ratio)
        new_obj.coeff = np.diagonal(self.coeff, axis1=axis1 + 1, axis2=axis2 + 1)
        new_obj.omega = self.omega
        new_obj.idx = self.idx
        new_obj.shape = new_obj.coeff.shape[1:]
        return new_obj


def AC_pade_thiele_diag(sigma, omega, npts=18, step_ratio=2.0 / 3.0):
    """Pade fitting for diagonal elements for a matrix.

    Parameters
    ----------
    sigma : complex ndarray
        matrix to fit, (norbs, nomega)
    omega : complex array
        frequency of the matrix sigma (nomega)
    npts : int, optional
        number of selected points, by default 18
    step_ratio : _type_, optional
        step ratio to select points, by default 2.0/3.0

    Returns
    -------
    acobj.coeff : complex ndarray
        fitting coefficient
    acobj.omega[acobj.idx] : complex ndarray
        selected frequency points for fitting
    """
    acobj = PadeAC(npts=npts, step_ratio=step_ratio)
    acobj.ac_fit(sigma, omega)
    return acobj.coeff, acobj.omega[acobj.idx]


def AC_pade_thiele_full(sigma, omega, npts=18, step_ratio=2.0 / 3.0):
    """Pade fitting for full matrix.

    Parameters
    ----------
    sigma : complex ndarray
        matrix to fit, (norbs, nomega)
    omega : complex array
        frequency of the matrix sigma (nomega)
    npts : int, optional
        number of selected points, by default 18
    step_ratio : _type_, optional
        step ratio to select points, by default 2.0/3.0

    Returns
    -------
    acobj.coeff : complex ndarray
        fitting coefficient
    acobj.omega[acobj.idx] : complex ndarray
        selected frequency points for fitting
    """
    acobj = PadeAC(npts=npts, step_ratio=step_ratio)
    acobj.ac_fit(sigma, omega)
    return acobj.coeff, acobj.omega[acobj.idx]


# two-pole fitting
class TwoPoleAC(AC_Method):
    """Two-pole analytic continuation method."""

    _data_keys = {'coeff', 'omega', 'orbs', 'nocc'}
    _method = 'twopole'

    def __init__(self, orbs, nocc, **options):
        """Constructor for TwoPoleAC.

        Parameters
        ----------
        orbs : list[int]
            indices of active orbitals
        nocc : int
            number of occupied orbitals
        """
        super().__init__(**options)
        self.coeff = None
        self.omega = None
        self.orbs = orbs
        self.nocc = nocc

    def ac_fit(self, data, omega, axis=-1):
        """Compute two-pole AC coefficients for the given data and omega.

        Parameters
        ----------
        data : np.ndarray
            Data to be fit, e.g. self energy
        omega : np.ndarray[np.double]
            1D imaginary frequency grid
        axis : int, optional
            Indicates which axis of the data array corresponds to the frequency axis, by default -1.
            Example: data.shape is (nmo, nmo, nw), call ac_fit(data, omega, axis=-1)
                     data.shape is (nw, nmo, nmo), call ac_fit(data, omega, axis=0)
        """
        self.omega = np.asarray(omega).copy()
        nw = self.omega.size
        data = np.asarray(data)

        assert omega.ndim == 1
        assert data.shape[axis] == nw

        # Move the frequency axis to the last axis
        data_transpose = np.moveaxis(data, axis, -1)

        # Shape of the data array, excluding the frequency axis
        self.shape = data_transpose.shape[:-1]

        coeff = np.zeros((10, *self.shape))

        for idx in np.ndindex(self.shape):
            # randomly generated initial guess
            p = self.orbs[idx[0]] # orbital index
            if p < self.nocc:
                x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5])
            else:
                x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5])
            # TODO: analytic gradient
            xopt = least_squares(
                two_pole_fit,
                x0,
                jac='3-point',
                method='trf',
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=1000,
                verbose=0,
                args=(omega, data_transpose[idx]),
            )
            if not xopt.success:
                log = logger.Logger()
                log.warn('2P-Fit Orb %d not converged, cost function %e' % (p, xopt.cost))
            coeff[(slice(None, None, None), *idx)] = xopt.x.copy()
        self.coeff = coeff

    def ac_eval(self, freqs, axis=-1):
        """Evaluate two-pole AC at arbitrary complex frequencies.

        Parameters
        ----------
        freqs : np.ndarray[np.complex128]
            1D array of complex frequencies
        axis : int, optional
            Indicates which axis of the output array should correspond to the frequency axis, by default -1.
            Example: if you want (nmo, nmo, nfreq), call ac_eval(freqs, axis=-1)
                     if you want (nfreq, nmo, nmo), call ac_eval(data, freqs, axis=0)

        Returns
        -------
        np.ndarray
            Pade-Thiele AC evaluated at `freqs`
        """
        if self.coeff is None or self.shape is None:
            raise ValueError('two-pole coefficients not set. Call ac_fit first.')
        is_scalar = np.isscalar(freqs)
        freqs = np.asarray(freqs)

        if self.coeff is None:
            raise ValueError("Pade coefficients not set. Call ac_fit first.")
        freqs = np.asarray(freqs)

        # Handle scalar freqs
        if np.ndim(freqs) == 0:
            freqs = np.array([freqs])

        assert freqs.ndim == 1

        if freqs.dtype != np.complex128:
            freqs = freqs.astype(np.complex128)
        cf = self.coeff[:5] + 1j * self.coeff[5:]

        freqs_broadcasted = np.expand_dims(freqs, axis=tuple(range(1, len(self.shape) + 1)))
        acvals = cf[0] + cf[1] / (freqs_broadcasted + cf[3]) + cf[2] / (freqs_broadcasted + cf[4])
        if not is_scalar:
            return np.moveaxis(acvals, 0, axis)
        else:
            return acvals[0]

    def __getitem__(self, multi_idx):
        assert self.coeff is not None
        new_obj = self.__class__(self.orbs, self.nocc)
        new_obj.coeff = self.coeff[(slice(None, None, None), *np.index_exp[multi_idx])]
        new_obj.omega = self.omega
        new_obj.shape = new_obj.coeff.shape[1:]
        return new_obj

    def diagonal(self, axis1=0, axis2=1):
        if self.coeff is None:
            raise ValueError("Two pole coefficients not set. Call ac_fit first.")
        if len(self.shape) < 2:
            raise ValueError("Two pole coefficients are not >=2D. Cannot get diagonal.")

        new_obj = self.__class__(self.orbs, self.nocc)
        new_obj.coeff = np.diagonal(self.coeff, axis1=axis1+1, axis2=axis2+1)
        new_obj.omega = self.omega
        new_obj.shape = new_obj.coeff.shape[1:]
        return new_obj


def two_pole_fit(coeff, omega, sigma):
    cf = coeff[:5] + 1j * coeff[5:]
    f = cf[0] + cf[1] / (omega + cf[3]) + cf[2] / (omega + cf[4]) - sigma
    f[0] = f[0] / 0.01
    return np.array([f.real, f.imag]).reshape(-1)


def two_pole(freqs, coeff):
    cf = coeff[:5] + 1j * coeff[5:]
    return cf[0] + cf[1] / (freqs + cf[3]) + cf[2] / (freqs + cf[4])
