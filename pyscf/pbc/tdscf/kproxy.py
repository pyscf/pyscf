#  Author: Artem Pulkin
"""
This and other `proxy` modules implement the time-dependent mean-field procedure using the existing pyscf
implementations as a black box. The main purpose of these modules is to overcome the existing limitations in pyscf
(i.e. real-only orbitals, davidson diagonalizer, incomplete Bloch space, etc). The primary performance drawback is that,
unlike the original pyscf routines with an implicit construction of the eigenvalue problem, these modules construct TD
matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TD roots. Several variants of proxy-TD are available:

 * `pyscf.tdscf.proxy`: the molecular implementation;
 * `pyscf.pbc.tdscf.proxy`: PBC (periodic boundary condition) Gamma-point-only implementation;
 * `pyscf.pbc.tdscf.kproxy_supercell`: PBC implementation constructing supercells. Works with an arbitrary number of
   k-points but has an overhead due to ignoring the momentum conservation law. In addition, works only with
   time reversal invariant (TRI) models: i.e. the k-point grid has to be aligned and contain at least one TRI momentum.
 * (this module) `pyscf.pbc.tdscf.kproxy`: same as the above but respect the momentum conservation and, thus, diagonlizes smaller
   matrices (the performance gain is the total number of k-points in the model).
"""

# Convention for these modules:
# * PhysERI is the proxying class constructing time-dependent matrices
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDProxy provides a container

from functools import reduce
from pyscf.pbc.tdscf import kproxy_supercell, krhf_slow

import numpy


def kov2ov(nocc, nmo, k):
    """
    Converts k point pairs into ov mask.
    Args:
        nocc (Iterable): occupation numbers per k-point;
        nmo (Iterable): numbers of orbitals per k-point;
        k (ndarray): k-point pairs;

    Returns:
        An ov-mask. Basis order: [k_o, o, k_v, v].
    """
    nocc = numpy.asanyarray(nocc)
    nmo = numpy.asanyarray(nmo)

    nvirt = nmo - nocc

    mask = numpy.zeros((sum(nocc), sum(nvirt)), dtype=bool)

    o_e = numpy.cumsum(nocc)
    o_s = o_e - o_e[0]

    v_e = numpy.cumsum(nvirt)
    v_s = v_e - v_e[0]

    for k1, k2 in enumerate(k):
        mask[o_s[k1]:o_e[k1], v_s[k2]:v_e[k2]] = True

    return mask.reshape(-1)


class PhysERI(kproxy_supercell.PhysERI):
    def __init__(self, model, proxy, x, mf_constructor, frozen=None, **kwargs):
        """
        A proxy class for calculating the TD matrix blocks (k-point version).

        Args:
            model: the base model with a time reversal-invariant k-point grid;
            proxy: a pyscf proxy with TD response function, one of 'hf', 'dft';
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            **kwargs: arguments to `k2s` function constructing supercells;
        """
        super(PhysERI, self).__init__(model, proxy, x, mf_constructor, frozen=frozen, **kwargs)

    def get_ov_space_mask(self):
        """
        Prepares the space mask in the ov form.
        Returns:
            The mask in the ov form.
        """
        return kproxy_supercell.orb2ov(numpy.concatenate(self.space), self.nocc_full, self.nmo_full)

    def kov2ov(self, k):
        """
        Converts k-ov mask into ov mask.
        Args:
            k (ndarray): k-point pairs;

        Returns:
            An ov-mask. Basis order: [k_o, o, k_v, v].
        """
        mask = self.get_ov_space_mask()
        return numpy.logical_and(mask, kov2ov(self.nocc_full, self.nmo_full, k))

    def proxy_response_ov_batch(self, k_row, k_col):
        """
        A raw response submatrix corresponding to specific k-points.
        Args:
            k_row (ndarray): sets of k-point pairs (row index);
            k_col (ndarray): sets of k-point pairs (column index);

        Returns:
            A raw response matrix.
        """
        masks_row = tuple(self.kov2ov(i) for i in k_row)
        masks_col = tuple(self.kov2ov(i) for i in k_col)

        full_mask_row = reduce(numpy.logical_or, masks_row)
        full_mask_col = reduce(numpy.logical_or, masks_col)

        big = kproxy_supercell.supercell_response_ov(
            self.proxy_vind,
            (full_mask_row, full_mask_col),
            self.nocc_full,
            self.nmo_full,
            self.proxy_is_double(),
            self.model_super.supercell_inv_rotation,
            self.model,
        )

        result = []

        for m_row, m_col in zip(masks_row, masks_col):
            m_row_red = m_row[full_mask_row]
            m_col_red = m_col[full_mask_col]
            result.append(tuple(i[m_row_red][:, m_col_red] for i in big))

        return tuple(result)

    # This is needed for krhf_slow.get_block_k_ix
    get_k_ix = krhf_slow.PhysERI.get_k_ix.im_func

    def tdhf_primary_form(self, k):
        """
        A primary form of TD matrices (full).
        Args:
            k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer
            vector or a single integer with the second index assuming the first index being zero;

        Returns:
            Output type: "full", and the corresponding matrix.
        """
        r1, r2, c1, c2 = krhf_slow.get_block_k_ix(self, k)
        (a, _), (_, b), (_, b_star), (a_star, _) = self.proxy_response_ov_batch((r1, r1, r2, r2), (c1, c2, c1, c2))
        return "full", numpy.block([[a, b], [-b_star.conj(), -a_star.conj()]])


vector_to_amplitudes = krhf_slow.vector_to_amplitudes


class TDProxy(kproxy_supercell.TDProxy):
    v2a = staticmethod(vector_to_amplitudes)
    proxy_eri = PhysERI

    def __init__(self, mf, proxy, x, mf_constructor, frozen=None, **kwargs):
        """
        Performs TD calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf: the base model with a time-reversal invariant k-point grid;
            proxy: a pyscf proxy with TD response function, one of 'hf', 'dft';
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            **kwargs: arguments to `k2s` function constructing supercells;
        """
        super(TDProxy, self).__init__(mf, proxy, x, mf_constructor, frozen=frozen, **kwargs)
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
