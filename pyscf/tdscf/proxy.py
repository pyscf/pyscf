#  Author: Artem Pulkin
"""
This and other `proxy` modules implement the time-dependent mean-field procedure using the existing pyscf
implementations as a black box. The main purpose of these modules is to overcome the existing limitations in pyscf
(i.e. real-only orbitals, davidson diagonalizer, incomplete Bloch space, etc). The primary performance drawback is that,
unlike the original pyscf routines with an implicit construction of the eigenvalue problem, these modules construct TD
matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TD roots. Several variants of proxy-TD are available:

 * (this module) `pyscf.tdscf.proxy`: the molecular implementation;
 * `pyscf.pbc.tdscf.proxy`: PBC (periodic boundary condition) Gamma-point-only implementation;
 * `pyscf.pbc.tdscf.kproxy_supercell`: PBC implementation constructing supercells. Works with an arbitrary number of
   k-points but has an overhead due to ignoring the momentum conservation law. In addition, works only with
   time reversal invariant (TRI) models: i.e. the k-point grid has to be aligned and contain at least one TRI momentum.
 * `pyscf.pbc.tdscf.kproxy`: same as the above but respect the momentum conservation and, thus, diagonlizes smaller
   matrices (the performance gain is the total number of k-points in the model).
"""

# Convention for these modules:
# * PhysERI is the proxying class constructing time-dependent matrices
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDProxy provides a container

from pyscf.tdscf.common_slow import TDProxyMatrixBlocks, MolecularMFMixin, TDBase, format_mask
from pyscf.tdscf import rhf_slow, TDDFT, TDHF
from pyscf.lib import logger

import numpy


def molecular_response_ov(vind, space_ov, nocc, nmo, double, log_dest):
    """
    Retrieves a raw response matrix.
    Args:
        vind (Callable): a pyscf matvec routine;
        space_ov (ndarray): the active `ov` space mask: either the same mask for both rows and columns (1D array) or
        separate `ov` masks for rows and columns (2D array);
        nocc (int): the number of occupied orbitals (frozen and active);
        nmo (int): the total number of orbitals;
        double (bool): set to True if `vind` returns the double-sized (i.e. full) matrix;
        log_dest (object): pyscf logging;

    Returns:
        The TD matrix.

    Note:
        The runtime scales with the size of the column mask `space_ov[1]` but not the row mask `space_ov[0]`.
    """
    space_ov = numpy.array(space_ov)

    nocc_full, nvirt_full = nocc, nmo - nocc
    size_full = nocc_full * nvirt_full

    if space_ov.shape not in ((size_full,), (2, size_full)):
        raise ValueError("The 'space_ov' argument should be a 1D array with dimension {size_full:d} or a 2D array with"
                         " dimensions 2x{size_full:d}, found: {actual}".format(
                             size_full=size_full,
                             actual=space_ov.shape,))
    ov1, ov2 = space_ov

    size = sum(ov2)

    probe = numpy.zeros((size, 2 * size_full if double else size_full))
    probe[numpy.arange(probe.shape[0]), numpy.argwhere(ov2)[:, 0]] = 1
    logger.debug1(log_dest, "Requesting response against {} matrix (column space: {})".format(
        "x".join(str(i) for i in probe.shape), format_mask(ov2),
    ))
    result = vind(probe).T
    logger.debug1(log_dest, "  output: {}".format(result.shape))

    if double:
        result = result[numpy.tile(ov1, 2)]
        half = sum(ov1)
        result_a = result[:half]
        result_b = result[half:]
        return result_a, -result_b.conj()
    else:
        return result[ov1]


def orb2ov(space, nocc):
    """
    Converts orbital active space specification into ov-pairs space spec.
    Args:
        space (ndarray): the obital space;
        nocc (int): the number of occupied orbitals;

    Returns:
        The ov space specification.
    """
    space = numpy.array(space)
    o = space[..., :nocc]
    v = space[..., nocc:]
    return (o[..., numpy.newaxis] * v[..., numpy.newaxis, :]).reshape(space.shape[:-1] + (-1,))


def molecular_response(vind, space, nocc, nmo, double, log_dest):
    """
    Retrieves a raw response matrix.
    Args:
        vind (Callable): a pyscf matvec routine;
        space (ndarray): the active orbital space mask: either the same mask for both rows and columns (1D array) or
        separate orbital masks for rows and columns (2D array);
        nocc (int): the number of occupied orbitals (frozen and active);
        nmo (int): the total number of orbitals;
        double (bool): set to True if `vind` returns the double-sized (i.e. full) matrix;
        log_dest (object): pyscf logging;

    Returns:
        The TD matrix.
    """
    space = numpy.array(space)

    if space.shape == (nmo,):
        space = numpy.repeat(space[numpy.newaxis, :], 2, axis=0)
    elif space.shape != (2, nmo):
        raise ValueError(
            "The 'space' argument should be a 1D array with dimension {size_full:d} or a 2D array with"
            " dimensions 2x{size_full:d}, found: {actual}".format(
                size_full=nocc,
                actual=space.shape,
            ))
    return molecular_response_ov(vind, orb2ov(space, nocc), nocc, nmo, double, log_dest)


def mk_make_canonic(m, o, v, return_ov=False, space_ov=None):
    """
    Makes the output of pyscf TDDFT matrix (MK form) to be canonic.
    Args:
        m (ndarray): the TDDFT matrix;
        o (ndarray): occupied orbital energies;
        v (ndarray): virtual orbital energies;
        return_ov (bool): if True, returns the K-matrix as well;
        space_ov (ndarray): an optional ov space;

    Returns:
        The rotated matrix as well as an optional K-matrix.
    """
    e_ov = (v[numpy.newaxis, :] - o[:, numpy.newaxis]).reshape(-1)
    if space_ov:
        e_ov = e_ov[space_ov]
    e_ov_sqr = e_ov ** .5
    result = m * (e_ov_sqr[numpy.newaxis, :] / e_ov_sqr[:, numpy.newaxis])
    if return_ov:
        return result, numpy.diag(e_ov)
    else:
        return result


class PhysERI(MolecularMFMixin, TDProxyMatrixBlocks):
    proxy_choices = {
        "hf": TDHF,
        "dft": TDDFT,
    }

    def __init__(self, model, proxy, frozen=None):
        """
        A proxy class for calculating TD matrix blocks (molecular version).

        Args:
            model: the base model;
            proxy: a pyscf proxy with TD response function, one of 'hf', 'dft';
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        # Note: the "proxy" argument does not matter at all since pyscf returns same objects for both HF and DFT
        # mean-field solutions. The argument is left for consistency and explicitness
        if proxy not in self.proxy_choices:
            raise ValueError("Unknown proxy: {}".format(repr(proxy)))
        TDProxyMatrixBlocks.__init__(self, self.proxy_choices[proxy](model))
        MolecularMFMixin.__init__(self, model, frozen=frozen)

    def proxy_is_double(self):
        """
        Determines if double-sized matrices are proxied.
        Returns:
            True if double-sized matrices are proxied.
        """
        nocc_full = self.nocc_full
        nmo_full = self.nmo_full
        size_full = nocc_full * (nmo_full - nocc_full)
        size_hdiag = len(self.proxy_diag)

        if size_full == size_hdiag:
            return False

        elif 2 * size_full == size_hdiag:
            return True

        else:
            raise RuntimeError("Do not recognize the size of TD diagonal: {:d}. The size of ov-space is {:d}".format(
                size_hdiag, size_full
            ))

    def proxy_response(self):
        """
        A raw response matrix.
        Returns:
            A raw response matrix.
        """
        return molecular_response(
            self.proxy_vind,
            self.space,
            self.nocc_full,
            self.nmo_full,
            self.proxy_is_double(),
            self.model,
        )

    def tdhf_primary_form(self, *args, **kwargs):
        """
        A primary form of TD matrixes.

        Returns:
            Output type: "full", "ab", or "mk" and the corresponding matrix(es).
        """
        if not self.proxy_is_double():
            # The MK case
            e_occ, e_virt = self.mo_energy[:self.nocc], self.mo_energy[self.nocc:]
            return ("mk",) + mk_make_canonic(self.proxy_response(), e_occ, e_virt, return_ov=True)

        else:
            # Full case
            return ("ab",) + self.proxy_response()


vector_to_amplitudes = rhf_slow.vector_to_amplitudes


class TDProxy(TDBase):
    proxy_eri = PhysERI
    v2a = staticmethod(vector_to_amplitudes)

    def __init__(self, mf, proxy, frozen=None):
        """
        Performs TD calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf: the base restricted mean-field model;
            proxy: a pyscf proxy with TD response function, one of 'hf', 'dft';
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        super(TDProxy, self).__init__(mf, frozen=frozen)
        self.__proxy__ = proxy

    def ao2mo(self):
        """
        Prepares ERI.

        Returns:
            A suitable ERI.
        """
        return self.proxy_eri(self._scf, self.__proxy__, frozen=self.frozen)
