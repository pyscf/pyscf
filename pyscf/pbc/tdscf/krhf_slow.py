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
from pyscf.lib import logger

import numpy
import scipy


# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are 2-electron integral routines computed directly (for debug purposes), with a 4-fold
#   symmetry and with an 8-fold symmetry
# * build_matrix builds the full TDHF matrix
# * eig performs diagonalization and selects roots
# * vector_to_amplitudes reshapes and normalizes the solution
# * kernel assembles everything
# * TDRHF provides a container


k_nocc = td.k_nocc


class PhysERI(td.PhysERI):

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a full transformation of integrals to Bloch functions. No symmetries are
        employed in this class. The ERIs are returned in blocks of k-points.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI, self).__init__(model)

    def get_k_ix(self, item, like):
        """
        Retrieves block indexes: row and column.
        Args:
            item (str): a string of 'mknj' letters;
            like (tuple): a 2-tuple with sample pair of k-points;

        Returns:
            Row and column indexes of a sub-block with conserving momentum.
        """
        item_i = numpy.argsort(self.__mknj2i__(item))
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
            block (Iterable): a list of pairs `block[k]` for each k;

        Returns:
            The diagonal block.
        """
        return super(PhysERI, self).tdhf_diag(pairs=((i, block[i]) for i in range(len(self.model.kpts))))

    def eri_mknj(self, item, block_x, block_y):
        """
        Retrieves the merged ERI block using 'mknj' notation with pairs of k-indexes (k1, k1, k2, k2).
        Args:
            item (str): a 4-character string of 'mknj' letters;
            block_x (Iterable): a list of pairs `block_x[k]` for each k (row indexes);
            block_y (Iterable): a list of pairs `block_y[k]` for each k (column indexes);

        Returns:
            The corresponding block of ERI (phys notation).
        """
        result = []
        for k1, k2 in enumerate(block_x):
            result.append([])
            for k3, k4 in enumerate(block_y):
                result[-1].append(self.eri_mknj_k(item, (k1, k2, k3, k4)))
        r = numpy.block(result)
        return r / len(self.model.kpts)


class PhysERI4(PhysERI):

    def __init__(self, model):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. A 4-fold
        symmetry of complex-valued functions is employed in this class. The ERIs are returned in blocks of k-points.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI4, self).__init__(model)

    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    __calc_block__ = td.PhysERI4.__calc_block__.im_func


class PhysERI8(PhysERI4):
    def __init__(self, model):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. An 8-fold
        symmetry of real-valued functions is employed in this class. The ERIs are returned in blocks of k-points.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI8, self).__init__(model)

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
    return r1, r2, c1, c2


def build_matrix(eri, k):
    """
    Full matrix of the TDRHF problem.
    Args:
        eri (TDDFTMatrixBlocks): ERI of the problem;
        k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer
        vector or a single integer with the second index assuming the first index being zero;

    Returns:
        The matrix.
    """
    r1, r2, c1, c2 = get_block_k_ix(eri, k)

    d1 = eri.tdhf_diag(r1)
    d2 = eri.tdhf_diag(r2)

    m11 = d1 + 2 * eri["knmj", r1, c1] - eri["knjm", r1, c1]
    m12 = 2 * eri["kjmn", r1, c2] - eri["kjnm", r1, c2]
    m21 = 2 * eri["mnkj", r2, c1] - eri["mnjk", r2, c1]
    m22 = d2 + 2 * eri["mjkn", r2, c2] - eri["mjnk", r2, c2]

    m = numpy.array([[m11, m12], [-m21, -m22]])

    return m.transpose((0, 2, 1, 3)).reshape(
        (m.shape[0] * m.shape[2], m.shape[1] * m.shape[3])
    )


eig = td.eig


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


def kernel(model, k, driver=None, nroots=None, return_eri=False):
    """
    Calculates eigenstates and eigenvalues of the TDHF problem.
    Args:
        model (RHF, PhysERI): the HF model or ERI;
        k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer;
        driver (str): one of the drivers;
        nroots (int): the number of roots to calculate;
        return_eri (bool): will also return ERI if True;

    Returns:
        Positive eigenvalues and eigenvectors.
    """
    if isinstance(model, PhysERI):
        eri = model
    else:
        if numpy.iscomplexobj(model.mo_coeff):
            logger.debug1(model, "4-fold symmetry used (complex orbitals)")
            eri = PhysERI4(model)
        else:
            logger.debug1(model, "8-fold symmetry used (real orbitals)")
            eri = PhysERI8(model)
    vals, vecs = eig(build_matrix(eri, k), driver=driver, nroots=nroots)
    vecs = vector_to_amplitudes(vecs, eri.nocc, eri.nmo)
    if return_eri:
        return vals, vecs, eri
    else:
        return vals, vecs


class TDRHF(object):
    def __init__(self, mf):
        """
        Performs TDHF calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RHF): the base restricted Hartree-Fock model;
        """
        self._scf = mf
        self.driver = None
        self.nroots = None
        self.eri = None
        self.xy = {}
        self.e = {}

    def kernel(self, k):
        """
        Calculates eigenstates and eigenvalues of the TDHF problem.
        Args:
            k (tuple, int): momentum transfer: either a pair of k-point indexes specifying the momentum transfer;

        Returns:
            Positive eigenvalues and eigenvectors.
        """
        self.e[k], self.xy[k], self.eri = kernel(
            self._scf if self.eri is None else self.eri,
            k,
            driver=self.driver,
            nroots=self.nroots,
            return_eri=True,
        )
        return self.e[k], self.xy[k]
