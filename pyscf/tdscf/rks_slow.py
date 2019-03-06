#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent Kohn-Sham procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TDKS roots in a reliable fashion without any issues related to the
Davidson procedure. Several variants of TDKS are available:

 * (this module) `pyscf.tdscf.rks_slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rks_slow`: PBC (periodic boundary condition) implementation for RKS objects of `pyscf.pbc.scf`
 modules;
 * `pyscf.pbc.tdscf.krks_slow_supercell`: PBC implementation for KRKS objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
 * `pyscf.pbc.tdscf.krks_slow_gamma`: A Gamma-point calculation resembling the original `pyscf.pbc.tdscf.krks`
   module. Despite its name, it accepts KRKS objects with an arbitrary number of k-points but finds only few TDKS roots
   corresponding to collective oscillations without momentum transfer;
 * `pyscf.pbc.tdscf.krks_slow`: PBC implementation for KRKS objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are proxy classes for computing the full TDDFT matrix
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDRKS provides a container

from pyscf.tdscf.common_slow import TDProxyMatrixBlocks, MolecularMFMixin, TDBase, format_mask
from pyscf.tdscf import rhf_slow, TDDFT
from pyscf.lib import logger

import numpy


def orb2ov(space, nocc):
    """
    Converts orbital active space specification into ov-pairs space spec.
    Args:
        space (ndarray): the obital space;
        nocc (int): the number of occupied orbitals;

    Returns:
        The ov space specification.
    """
    o = space[:nocc]
    v = space[nocc:]
    return (o[:, numpy.newaxis] * v[numpy.newaxis, :]).reshape(-1)


def molecular_response(vind, space, nocc, nmo, double, log_dest):
    """
    Retrieves a raw response matrix.
    Args:
        vind (Callable): a pyscf matvec routine;
        space (ndarray): the active space: either for both rows and columns (1D array) or for rows and columns separately (2D array);
        nocc (int): the number of occupied orbitals (frozen and active);
        nmo (int): the total number of orbitals;
        double (bool): set to True if `vind` returns the double-sized (i.e. full) matrix;
        log_dest (object): pyscf logging;

    Returns:
        The TD matrix.
    """
    space = numpy.array(space)

    nocc_full, nvirt_full = nocc, nmo - nocc
    size_full = nocc_full * nvirt_full

    if space.shape == (nmo,):
        ov1 = ov2 = orb2ov(space, nocc)
    elif space.shape == (2, nmo):
        ov1 = orb2ov(space[0], nocc)
        ov2 = orb2ov(space[1], nocc)
    else:
        raise ValueError("The 'space' argument should be either a plain array or a 2D array with first dimension = "
                         "2, found shape: {}".format(space.shape))

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

    def __init__(self, model, frozen=None, proxy=None):
        """
        A proxy class for calculating the TDKS matrix blocks (molecular version).

        Args:
            model (RKS): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        TDProxyMatrixBlocks.__init__(self, proxy if proxy is not None else TDDFT(model))
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


class TDRKS(TDBase):
    proxy_eri = PhysERI
    v2a = staticmethod(vector_to_amplitudes)

    def __init__(self, mf, frozen=None, proxy=None):
        """
        Performs TDKS calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RKS): the base restricted DFT model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        super(TDRKS, self).__init__(mf, frozen=frozen)
        self.fast = True
        self.__proxy__ = proxy

    def ao2mo(self):
        """
        Prepares ERI.

        Returns:
            A suitable ERI.
        """
        return self.proxy_eri(self._scf, frozen=self.frozen, proxy=self.__proxy__)
