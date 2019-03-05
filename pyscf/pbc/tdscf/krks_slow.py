#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent Kohn-Sham procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TDKS roots in a reliable fashion without any issues related to the
Davidson procedure. Several variants of TDKS are available:

 * `pyscf.tdscf.rks_slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rks_slow`: PBC (periodic boundary condition) implementation for RKS objects of `pyscf.pbc.scf`
   modules;
 * `pyscf.pbc.tdscf.krks_slow_supercell`: PBC implementation for KRKS objects of `pyscf.pbc.scf` modules.
   Works with an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
 * `pyscf.pbc.tdscf.krks_slow_gamma`: A Gamma-point calculation resembling the original `pyscf.pbc.tdscf.krks`
   module. Despite its name, it accepts KRKS objects with an arbitrary number of k-points but finds only few TDKS roots
   corresponding to collective oscillations without momentum transfer;
 * (this moodule)`pyscf.pbc.tdscf.krks_slow`: PBC implementation for KRKS objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are proxy classes for computing the full TDDFT matrix
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDRKS provides a container

from pyscf.pbc.tdscf import krks_slow_supercell, krhf_slow
from pyscf.lib import logger

import numpy


def orb2ov(space, nocc, nmo):
    """
    Converts orbital active space specification into ov-pairs space spec.
    Args:
        space (ndarray): the obital space;
        nocc (Iterable): the numbers of occupied orbitals per k-point;
        nmo (Iterable): the total numbers of orbitals per k-point;

    Returns:
        The ov space specification.
    """
    m = krks_slow_supercell.ko_mask(nocc, nmo)
    o = space[m]
    v = space[~m]
    return (o[:, numpy.newaxis] * v[numpy.newaxis, :]).reshape(-1)


def ov2orb(space, nocc, nmo):
    """
    Converts ov-pairs active space specification into orbital space spec.
    Args:
        space (ndarray): the ov space;
        nocc (int): the total number of occupied orbitals;
        nmo (int): the total number of orbitals;

    Returns:
        The orbital space specification.
    """
    s = space.reshape(nocc, nmo - nocc)
    s_o = numpy.any(s, axis=1)
    s_v = numpy.any(s, axis=0)
    return numpy.concatenate((s_o, s_v))


def supercell_response_ov(vind, space_ov, nocc, nmo, double, rot_bloch, log_dest):
    """
    Retrieves a raw response matrix.
    Args:
        vind (Callable): a pyscf matvec routine;
        space_ov (ndarray): the active `ov` space in the Bloch space;
        nocc (ndarray): the numbers of occupied orbitals (frozen and active) per k-point;
        nmo (ndarray): the total number of orbitals per k-point;
        double (bool): set to True if `vind` returns the double-sized (i.e. full) matrix;
        rot_bloch (ndarray): a matrix specifying the rotation from real orbitals returned from pyscf to Bloch
        functions;
        log_dest (object): pyscf logging;

    Returns:
        The TD matrix.
    """
    if not double:
        raise NotImplementedError("Not implemented for MK-type matrixes")

    # Full space dims
    nocc_full = sum(nocc)
    nmo_full = sum(nmo)
    size_ov_full = nocc_full * (nmo_full - nocc_full)

    space_ov = numpy.array(space_ov)

    if space_ov.shape == (size_ov_full,):
        space_ov = numpy.repeat(space_ov[numpy.newaxis, :], 2, axis=0)
    elif space_ov.shape != (2, size_ov_full):
        raise ValueError("The 'space' argument should a 1D array with dimension {:d} or a 2D array with dimensions {},"
                         " found: {}".format(size_ov_full, (2, size_ov_full), space_ov.shape))

    space_orb = tuple(ov2orb(i, nocc_full, nmo_full) for i in space_ov)
    space_full = tuple(j[orb2ov(i, nocc, nmo)] for i, j in zip(space_orb, space_ov))

    result_orb = krks_slow_supercell.supercell_response(
        vind,
        space_orb,
        nocc, nmo, double,
        rot_bloch,
        log_dest
    )
    logger.debug1(log_dest, "Reshaping into a matrix and slicing ov ...")

    result = []
    for m in result_orb:
        d1, d2, d3, d4 = m.shape
        result.append(m.reshape(d1 * d2, d3 * d4)[space_full[0], :][:, space_full[1]])

    return tuple(result)


def k2ov(nocc, nmo, k):
    """
    Converts k-index pairs into ov mask.
    Args:
        nocc (Iterable): occupation numbers per k-point;
        nmo (Iterable): numbers of orbitals per k-point;
        k (Iterable): a list of second k-indexes;

    Returns:
        An ov-mask.
    """
    o = numpy.concatenate(tuple(
        (i, ) * n
        for i, n in enumerate(nocc)
    ))
    v = numpy.concatenate(tuple(
        (i, ) * (m - n)
        for i, (n, m) in enumerate(zip(nocc, nmo))
    ))
    ov_o, ov_v = numpy.meshgrid(o, v, indexing="ij")
    ov_o, ov_v = ov_o.reshape(-1), ov_v.reshape(-1)

    result = numpy.zeros(ov_o.shape, dtype=bool)
    for k1, k2 in enumerate(k):
        result[numpy.logical_and(ov_o == k1, ov_v == k2)] = True
    return result


class PhysERI(krks_slow_supercell.PhysERI):
    def __init__(self, model, x, mf_constructor, frozen=None, proxy=None):
        """
        A proxy class for calculating the TDKS matrix blocks (k-point version).

        Args:
            model (KRKS): the base model with a regular k-point grid which includes the Gamma-point;
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        super(PhysERI, self).__init__(model, x, mf_constructor, frozen=frozen, proxy=proxy)

    def proxy_response_ov(self, k1, k2):
        """
        A raw response submatrix corresponding to specific k-points.
        Args:
            k1 (Iterable): a list of second k-indexes (row dimension);
            k2 (Iterable): a list of second k-indexes (column dimension);

        Returns:
            A raw response matrix.
        """
        space_ov = orb2ov(numpy.concatenate(self.space), self.nocc_full, self.nmo_full)
        return supercell_response_ov(
            self.proxy_vind,
            (
                numpy.logical_and(space_ov, k2ov(self.nocc, self.nmo, k1)),
                numpy.logical_and(space_ov, k2ov(self.nocc, self.nmo, k2)),
            ),
            self.nocc_full,
            self.nmo_full,
            self.proxy_is_double(),
            self.model_super.supercell_inv_rotation,
            self.model,
        )

    # This is needed for krhf_slow.get_block_k_ix
    get_k_ix = krhf_slow.PhysERI.get_k_ix.im_func

    def tdhf_primary_form(self, k):
        """
        A primary form of TDKS matrixes (full).
        Args:
            k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer
            vector or a single integer with the second index assuming the first index being zero;

        Returns:
            Output type: "full", and the corresponding matrix.
        """
        # 1. Convert k into pairs of rows and column k
        k1, k2, _, _ = krhf_slow.get_block_k_ix(self, k)
        # 2. Retrieve the 4 matrix blocks
        a, _ = self.proxy_response_ov(k1, k1)
        _, b = self.proxy_response_ov(k1, k2)
        _, b_star = self.proxy_response_ov(k2, k1)
        a_star, _ = self.proxy_response_ov(k2, k2)
        # 3. Merge them into the full matrix
        return "full", numpy.block([[a, b], [-b_star.conj(), -a_star.conj()]])


vector_to_amplitudes = krhf_slow.vector_to_amplitudes


class TDRKS(krks_slow_supercell.TDRKS):
    v2a = staticmethod(vector_to_amplitudes)
    proxy_eri = PhysERI

    def __init__(self, mf, x, mf_constructor, frozen=None, proxy=None):
        """
        Performs TDKS calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RKS): the base restricted DFT model;
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        super(TDRKS, self).__init__(mf, x, mf_constructor, frozen=frozen, proxy=proxy)
        self.e = {}
        self.xy = {}

    def kernel(self, k=None):
        """
        Calculates eigenstates and eigenvalues of the TDHF problem.
        Args:
            k (tuple, int): momentum transfer: either an index specifying the momentum transfer or a list of such
            indexes;

        Returns:
            Positive eigenvalues and eigenvectors.
        """
        if k is None:
            k = numpy.arange(len(self._scf.kpts))

        if isinstance(k, int):
            k = [k]

        for kk in k:
            self.e[kk], self.xy[kk] = self.__kernel__(k=kk)
        return self.e, self.xy
