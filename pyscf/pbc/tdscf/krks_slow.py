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

    def get_ov_space_mask(self):
        """
        Prepares the space mask in the ov form.
        Returns:
            The mask in the ov form.
        """
        return krks_slow_supercell.orb2ov(numpy.concatenate(self.space), self.nocc_full, self.nmo_full)

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

        big = krks_slow_supercell.supercell_response_ov(
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
        A primary form of TDKS matrixes (full).
        Args:
            k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer
            vector or a single integer with the second index assuming the first index being zero;

        Returns:
            Output type: "full", and the corresponding matrix.
        """
        # 1. Convert k into pairs of rows and column k
        r1, r2, c1, c2 = krhf_slow.get_block_k_ix(self, k)

        (a, _), (_, b), (_, b_star), (a_star, _) = self.proxy_response_ov_batch((r1, r1, r2, r2), (c1, c2, c1, c2))

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


if __name__ == "__main__":
    from pyscf.pbc.gto import Cell
    from pyscf.pbc.scf import KRKS

    cell = Cell()
    # Lift some degeneracies
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.67   1.68   1.69
    '''
    cell.basis = {'C': [[0, (0.8, 1.0)],
                        [1, (1.0, 1.0)]]}
    # cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 7
    cell.build()

    gs = [2, 1, 1]
    k = cell.make_kpts(gs)

    model_krks = KRKS(cell, k)
    model_krks.kernel()

    model = TDRKS(model_krks, gs, KRKS)
    model.kernel(0)
    print model.eri.proxy_vind.text_stats()
