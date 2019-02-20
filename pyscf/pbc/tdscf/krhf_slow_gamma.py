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
   an arbitrary number of k-points but has a overhead due to an effective construction of a supercell;
 * (this module) `pyscf.pbc.tdscf.krhf_slow_gamma`: A Gamma-point calculation resembling the original `pyscf.pbc.tdscf.krhf`
   module. Despite its name, it accepts KRHF objects with an arbitrary number of k-points but finds only few TDHF roots
   corresponding to collective oscillations without momentum transfer;
 * `pyscf.pbc.tdscf.krhf_slow`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

from pyscf.pbc.tdscf import krhf_slow_supercell as td
from pyscf.tdscf import rhf_slow

import numpy


# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are 2-electron integral routines computed directly (for debug purposes), with a 4-fold
#   symmetry and with an 8-fold symmetry
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDRHF provides a container


class PhysERI(td.PhysERI):

    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing a full transformation of integrals to Bloch functions. No symmetries are
        employed in this class. Only a subset of transformed ERI is returned, corresponding to oscillations without a
        momentum transfer. The k-points of the returned ERIs come in two pairs :math:`(k_1 k_1 | k_2 k_2)`. As a result,
        one of the diagonal blocks of the full TDHF matrix can be reconstructed. For other blocks, look into `PhysERI`
        class of this momdule.

        Args:
            model (KRHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals for all
            k-points or multiple lists of frozen orbitals for each k-point;
        """
        super(PhysERI, self).__init__(model, frozen=frozen)

    def tdhf_diag(self):
        """
        Retrieves the merged diagonal block with equal pairs of k-indexes (k, k) only.

        Returns:
            The diagonal block.
        """
        return super(PhysERI, self).tdhf_diag(pairs=((i, i) for i in range(len(self.model.kpts))))

    def __calc_block__(self, item, k):
        if k in self.__full_eri_k__:
            return super(PhysERI, self).__calc_block__(item, k)
        else:
            raise RuntimeError("The block k = {:d} + {:d} - {:d} - {:d} does not conserve momentum".format(*k))

    def eri_mknj(self, item):
        """
        Retrieves the merged ERI block using 'mknj' notation with pairs of k-indexes (k1, k1, k2, k2).
        Args:
            item (str): a 4-character string of 'mknj' letters;

        Returns:
            The corresponding block of ERI (phys notation).
        """
        return super(PhysERI, self).eri_mknj(
            item,
            pairs_row=((i, i) for i in range(len(self.model.kpts))),
            pairs_column=((i, i) for i in range(len(self.model.kpts))),
        )


class PhysERI4(PhysERI):

    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. A 4-fold
        symmetry of complex-valued functions is employed in this class. Only a subset of transformed ERI is returned,
        corresponding to oscillations without a momentum transfer. The k-points of the returned ERIs come in two pairs
        :math:`(k_1 k_1 | k_2 k_2)`. As a result, one of the diagonal blocks of the full TDHF matrix can be
        reconstructed. For other blocks, look into `PhysERI` class of this momdule.

        Args:
            model (KRHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals for all
            k-points or multiple lists of frozen orbitals for each k-point;
        """
        super(PhysERI4, self).__init__(model, frozen=frozen)

    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    def __calc_block__(self, item, k):
        if k in self.__full_eri_k__:
            return td.PhysERI4.__calc_block__.im_func(self, item, k)
        else:
            raise RuntimeError("The block k = {:d} + {:d} - {:d} - {:d} does not conserve momentum".format(*k))


class PhysERI8(PhysERI4):
    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. An 8-fold
        symmetry of real-valued functions is employed in this class. Only a subset of transformed ERI is returned,
        corresponding to oscillations without a momentum transfer. The k-points of the returned ERIs come in two pairs
        :math:`(k_1 k_1 | k_2 k_2)`. As a result, one of the diagonal blocks of the full TDHF matrix can be
        reconstructed. For other blocks, look into `PhysERI` class of this momdule.

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


def vector_to_amplitudes(vectors, nocc, nmo):
    """
    Transforms (reshapes) and normalizes vectors into amplitudes.
    Args:
        vectors (numpy.ndarray): raw eigenvectors to transform;
        nocc (tuple): numbers of occupied orbitals;
        nmo (int): the total number of orbitals per k-point;

    Returns:
        Amplitudes with the following shape: (# of roots, 2 (x or y), # of kpts, # of kpts, # of occupied orbitals,
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
    # Compared to krhf_slow_supercell, only one k-point index is present here. The second index corresponds to
    # momentum transfer and is integrated out
    vectors = vectors.reshape(2, nk, nocc, nmo-nocc, vectors.shape[1])
    norm = (abs(vectors) ** 2).sum(axis=(1, 2, 3))
    norm = 2 * (norm[0] - norm[1])
    vectors /= norm ** .5
    return vectors.transpose(4, 0, 1, 2, 3)


class TDRHF(rhf_slow.TDRHF):
    eri4 = PhysERI4
    eri8 = PhysERI8
    v2a = staticmethod(vector_to_amplitudes)
