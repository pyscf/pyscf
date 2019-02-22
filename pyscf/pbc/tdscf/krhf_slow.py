#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent Hartree-Fock procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly via an AO-MO transformation, i.e. with a O(N^5) complexity scaling. As a result, regular
`numpy.linalg.eig` can be used to retrieve TDHF roots in a reliable fashion without any issues related to the Davidson
procedure. Several variants of TDHF are available:

 * `pyscf.tdscf.rhf_slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rhf_slow`: PBC (periodic boundary condition) implementation for RHF objects of `pyscf.pbc.scf`
   modules;
 * `pyscf.pbc.tdscf.krhf_slow_supercell`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
 * `pyscf.pbc.tdscf.krhf_slow_gamma`: A Gamma-point calculation resembling the original `pyscf.pbc.tdscf.krhf`
   module. Despite its name, it accepts KRHF objects with an arbitrary number of k-points but finds only few TDHF roots
   corresponding to collective oscillations without momentum transfer;
 * (this module) `pyscf.pbc.tdscf.krhf_slow`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

from pyscf.pbc.tdscf import krhf_slow_supercell as td
from pyscf.tdscf import rhf_slow
from pyscf.tdscf.common_slow import mknj2i

import numpy


# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are 2-electron integral routines computed directly (for debug purposes), with a 4-fold
#   symmetry and with an 8-fold symmetry
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDRHF provides a container


class PhysERI(td.PhysERI):
    primary_driver = "full"

    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing a full transformation of integrals to Bloch functions. No symmetries are
        employed in this class. The ERIs are returned in blocks of k-points.

        Args:
            model (KRHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals for all
            k-points or multiple lists of frozen orbitals for each k-point;
        """
        super(PhysERI, self).__init__(model, frozen=frozen)

    def get_k_ix(self, item, like):
        """
        Retrieves block indexes: row and column.
        Args:
            item (str): a string of 'mknj' letters;
            like (tuple): a 2-tuple with sample pair of k-points;

        Returns:
            Row and column indexes of a sub-block with conserving momentum.
        """
        item_i = numpy.argsort(mknj2i(item))
        item_code = ''.join("++--"[i] for i in item_i)
        if item_code[0] == item_code[1]:
            kc = self.kconserv  # ++-- --++
        elif item_code[0] == item_code[2]:
            kc = self.kconserv.swapaxes(1, 2)  # +-+- -+-+
        elif item_code[1] == item_code[2]:
            kc = self.kconserv.transpose(2, 0, 1)  # +--+ -++-
        else:
            raise RuntimeError("Unknown case: {}".format(item_code))

        y = kc[like]
        x = kc[0, y[0]]

        return x, y

    def tdhf_diag(self, block):
        """
        Retrieves the merged diagonal block only with specific pairs of k-indexes (k, block[k]).
        Args:
            block (Iterable): a k-point pair `k2 = pair[k1]` for each k1;

        Returns:
            The diagonal block.
        """
        return super(PhysERI, self).tdhf_diag(pairs=enumerate(block))

    def eri_mknj(self, item, pair_row, pair_column):
        """
        Retrieves the merged ERI block using 'mknj' notation with pairs of k-indexes (k1, k1, k2, k2).
        Args:
            item (str): a 4-character string of 'mknj' letters;
            pair_row (Iterable): a k-point pair `k2 = pair_row[k1]` for each k1 (row indexes in the final matrix);
            pair_column (Iterable): a k-point pair `k4 = pair_row[k3]` for each k3 (column indexes in the final matrix);

        Returns:
            The corresponding block of ERI (phys notation).
        """
        return super(PhysERI, self).eri_mknj(
            item,
            pairs_row=enumerate(pair_row),
            pairs_column=enumerate(pair_column),
        )

    def tdhf_primary_form(self, k):
        """
        A primary form of TDHF matrixes (full).
        Args:
            k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer
            vector or a single integer with the second index assuming the first index being zero;

        Returns:
            Output type: "full", and the corresponding matrix.
        """
        r1, r2, c1, c2 = get_block_k_ix(self, k)
        d1 = self.tdhf_diag(r1)
        d2 = self.tdhf_diag(r2)
        a = d1 + 2 * self["knmj", r1, c1] - self["knjm", r1, c1]
        b = 2 * self["kjmn", r1, c2] - self["kjnm", r1, c2]
        a_ = d2 + 2 * self["mjkn", r2, c2] - self["mjnk", r2, c2]
        b_ = 2 * self["mnkj", r2, c1] - self["mnjk", r2, c1]
        return "full", numpy.block([[a, b], [-b_, -a_]])


class PhysERI4(PhysERI):

    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. A 4-fold
        symmetry of complex-valued functions is employed in this class. The ERIs are returned in blocks of k-points.

        Args:
            model (KRHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals for all
            k-points or multiple lists of frozen orbitals for each k-point;
        """
        td.PhysERI4.__init__.im_func(self, model, frozen=frozen)

    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    def __calc_block__(self, item, k):
        if self.kconserv[k[:3]] == k[3]:
            return td.PhysERI4.__calc_block__.im_func(self, item, k)
        else:
            raise ValueError("K is not conserved: {}, expected {}".format(
                repr(k),
                k[:3] + (self.kconserv[k[:3]],),
            ))


class PhysERI8(PhysERI4):
    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. An 8-fold
        symmetry of real-valued functions is employed in this class. The ERIs are returned in blocks of k-points.

        Args:
            model (KRHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals for all
            k-points or multiple lists of frozen orbitals for each k-point;
        """
        super(PhysERI8, self).__init__(model, frozen=frozen)

    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), False),
        ((3, 2, 1, 0), False),

        ((2, 1, 0, 3), False),
        ((3, 0, 1, 2), False),
        ((0, 3, 2, 1), False),
        ((1, 2, 3, 0), False),
    ]


def get_block_k_ix(eri, k):
    """
    Retrieves k indexes of the block with a specific momentum transfer.
    Args:
        eri (TDDFTMatrixBlocks): ERI of the problem;
        k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer
        vector or a single integer with the second index assuming the first index being zero;

    Returns:
        4 arrays: r1, r2, c1, c2 specifying k-indexes of the ERI matrix block.

        +-----------------+-------------+-------------+-----+-----------------+-------------+-------------+-----+-----------------+
        |                 | k34=0,c1[0] | k34=1,c1[1] | ... | k34=nk-1,c1[-1] | k34=0,c2[0] | k34=1,c2[1] | ... | k34=nk-1,c2[-1] |
        +-----------------+-------------+-------------+-----+-----------------+-------------+-------------+-----+-----------------+
        |   k12=0,r1[0]   |                                                   |                                                   |
        +-----------------+                                                   |                                                   |
        |   k12=1,r1[1]   |                                                   |                                                   |
        +-----------------+                  Block r1, c1                     |                  Block r1, c2                     |
        |       ...       |                                                   |                                                   |
        +-----------------+                                                   |                                                   |
        | k12=nk-1,r1[-1] |                                                   |                                                   |
        +-----------------+---------------------------------------------------+---------------------------------------------------+
        |   k12=0,r2[0]   |                                                   |                                                   |
        +-----------------+                                                   |                                                   |
        |   k12=1,r2[1]   |                                                   |                                                   |
        +-----------------+                  Block r2, c1                     |                  Block r2, c2                     |
        |       ...       |                                                   |                                                   |
        +-----------------+                                                   |                                                   |
        | k12=nk-1,r2[-1] |                                                   |                                                   |
        +-----------------+---------------------------------------------------+---------------------------------------------------+
    """
    # All checks here are for debugging purposes
    if isinstance(k, int):
        k = (0, k)
    r1, c1 = eri.get_k_ix("knmj", k)
    assert r1[k[0]] == k[1]
    # knmj and kjmn share row indexes
    _, c2 = eri.get_k_ix("kjmn", (0, r1[0]))
    assert abs(r1 - _).max() == 0
    # knmj and mnkj share column indexes
    _, r2 = eri.get_k_ix("mnkj", (0, c1[0]))
    assert abs(c1 - _).max() == 0
    _r, _c = eri.get_k_ix("mjkn", (0, r2[0]))
    assert abs(r2 - _r).max() == 0
    assert abs(c2 - _c).max() == 0
    _c, _r = eri.get_k_ix("mjkn", (0, c2[0]))
    assert abs(r2 - _r).max() == 0
    assert abs(c2 - _c).max() == 0

    assert abs(r1 - c1).max() == 0
    assert abs(r2 - c2).max() == 0
    assert abs(r1[r2] - numpy.arange(len(r1))).max() == 0
    # The output is, basically, r1, argsort(r1), r1, argsort(r1)
    return r1, r2, c1, c2


def vector_to_amplitudes(vectors, nocc, nmo):
    """
    Transforms (reshapes) and normalizes vectors into amplitudes.
    Args:
        vectors (numpy.ndarray): raw eigenvectors to transform;
        nocc (tuple): numbers of occupied orbitals;
        nmo (int): the total number of orbitals per k-point;

    Returns:
        Amplitudes with the following shape: (# of roots, 2 (x or y), # of kpts, # of occupied orbitals,
        # of virtual orbitals).
    """
    if not all(i == nocc[0] for i in nocc):
        raise NotImplementedError("Varying occupation numbers are not implemented yet")
    nk = len(nocc)
    nocc = nocc[0]
    if not all(i == nmo[0] for i in nmo):
        raise NotImplementedError("Varying AO spaces are not implemented yet")
    nmo = nmo[0]
    vectors = numpy.asanyarray(vectors)
    vectors = vectors.reshape(2, nk, nocc, nmo-nocc, vectors.shape[1])
    norm = (abs(vectors) ** 2).sum(axis=(1, 2, 3))
    norm = 2 * (norm[0] - norm[1])
    vectors /= norm ** .5
    return vectors.transpose(4, 0, 1, 2, 3)


class TDRHF(rhf_slow.TDRHF):
    eri4 = PhysERI4
    eri8 = PhysERI8
    v2a = staticmethod(vector_to_amplitudes)

    def __init__(self, mf, frozen=None):
        """
        Performs TDHF calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RHF): the base restricted Hartree-Fock model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals for all
            k-points or multiple lists of frozen orbitals for each k-point;
        """
        super(TDRHF, self).__init__(mf, frozen=frozen)
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
