"""
This and other `_slow` modules implement the time-dependent Hartree-Fock procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly via an AO-MO transformation, i.e. with a O(N^5) complexity scaling. As a result, regular
`numpy.linalg.eig` can be used to retrieve TDHF roots in a reliable fashion without any issues related to the Davidson
procedure. Several variants of TDHF are available:

 * `pyscf.tdscf.rhf.slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rhf_slow`: PBC (periodic boundary condition) implementation for RHF objects of `pyscf.pbc.scf`
   modules;
 * `pyscf.pbc.tdscf.krhf_slow_supercell`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
 * (this module) `pyscf.pbc.tdscf.krhf_slow`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

from pyscf.pbc.tdscf.krhf_slow_supercell import PhysERI as PhysERI_S, PhysERI4 as PhysERI4_S, build_matrix, eig, k_nocc
from pyscf.lib import logger

import numpy
import scipy


class PhysERIGamma(PhysERI_S):

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a full transformation of integrals to Bloch functions. No symmetries are
        employed in this class. Only a subset of transformed ERI is returned, corresponding to oscillations without a
        momentum transfer. The k-points of the returned ERIs come in two pairs :math:`(k_1 k_1 | k_2 k_2)`. As a result,
        one of the diagonal blocks of the full TDHF matrix can be reconstructed. For other blocks, look into `PhysERI`
        class of this momdule.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERIGamma, self).__init__(model)

    def assemble_diag_block(self):
        result = []
        for k in range(len(self.model.kpts)):
            b = self.get_diag_block(k, k)
            o1, v1, o2, v2 = b.shape
            b = b.reshape(o1 * v1, o2 * v2)
            result.append(b)
        return scipy.linalg.block_diag(*result)

    def assemble_block(self, item):
        result = []
        nkpts = len(self.model.kpts)
        for k1 in range(nkpts):
            result.append([])
            for k2 in range(nkpts):
                x = self.get_block_mknj_notation(item, (k1, k1, k2, k2))
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
                result[-1].append(x)

        r = numpy.block(result)
        return r / len(self.model.kpts)


class PhysERI4Gamma(PhysERIGamma):

    def __init__(self, model):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. A 4-fold
        symmetry of complex-valued functions is employed in this class. Only a subset of transformed ERI is returned,
        corresponding to oscillations without a momentum transfer. The k-points of the returned ERIs come in two pairs
        :math:`(k_1 k_1 | k_2 k_2)`. As a result, one of the diagonal blocks of the full TDHF matrix can be
        reconstructed. For other blocks, look into `PhysERI` class of this momdule.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI4Gamma, self).__init__(model)

    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    __calc_block__ = PhysERI4_S.__calc_block__.im_func


class PhysERI8Gamma(PhysERI4Gamma):
    def __init__(self, model):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. An 8-fold
        symmetry of real-valued functions is employed in this class. Only a subset of transformed ERI is returned,
        corresponding to oscillations without a momentum transfer. The k-points of the returned ERIs come in two pairs
        :math:`(k_1 k_1 | k_2 k_2)`. As a result, one of the diagonal blocks of the full TDHF matrix can be
        reconstructed. For other blocks, look into `PhysERI` class of this momdule.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI8Gamma, self).__init__(model)

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
        raise NotImplementedError("Non-equal occupation numbers are not implemented yet")
    nk = len(nocc)
    nocc = nocc[0]
    vectors = numpy.asanyarray(vectors)
    # Compared to krhf_slow_supercell, only one k-point index is present here. The second index corresponds to
    # momentum transfer and is integrated out
    vectors = vectors.reshape(2, nk, nocc, nmo-nocc, vectors.shape[1])
    norm = (abs(vectors) ** 2).sum(axis=(1, 2, 3))
    norm = 2 * (norm[0] - norm[1])
    vectors /= norm ** .5
    return vectors.transpose(4, 0, 1, 2, 3)


def kernel_gamma(model, driver=None, nroots=None):
    """
    Calculates eigenstates and eigenvalues of the TDHF problem without momentum transfer.
    Args:
        model (RHF): the HF model;
        driver (str): one of the drivers;
        nroots (int): the number of roots ot calculate (ignored for `driver` == 'eig');

    Returns:
        Positive eigenvalues and eigenvectors.
    """
    if numpy.iscomplexobj(model.mo_coeff):
        logger.debug1(model, "4-fold symmetry used (complex orbitals)")
        eri = PhysERI4Gamma(model)
    else:
        logger.debug1(model, "8-fold symmetry used (real orbitals)")
        eri = PhysERI8Gamma(model)
    vals, vecs = eig(build_matrix(eri), driver=driver, nroots=nroots)
    return vals, vector_to_amplitudes(vecs, eri.nocc, model.mo_coeff[0].shape[0])


class PhysERI(PhysERI_S):

    def __init__(self, model, k):
        """
        The TDHF ERI implementation performing a full transformation of integrals to Bloch functions. No symmetries are
        employed in this class. Only a subset of transformed ERI is returned, corresponding to oscillations with a
        specific value of momentum transfer. The k-points of the returned ERIs come in two pairs
        :math:`(k_1 k_1 - dk | k_2 k_2 + dk)`. As a result, all diagonal blocks of the full TDHF matrix with different
        values of :math:`dk` can be reconstructed.

        Args:
            model (KRHF): the base model;
            k (int): the momentum transfer index corresponding to all momnetum pairs `i, j` satisfying
            `kconserv[k, 0, i] == j`.
        """
        super(PhysERI, self).__init__(model)
        self.k = k
        # Note that kconserv is in phys notation
        self.kconserv_k0 = self.kconserv[k, :, 0]

    def assemble_diag_block(self):
        result = []
        for k in range(len(self.model.kpts)):
            k2 = self.kconserv_k0[k]
            b = self.get_diag_block(k, k2)
            o1, v1, o2, v2 = b.shape
            b = b.reshape(o1 * v1, o2 * v2)
            result.append(b)
        return scipy.linalg.block_diag(*result)

    def __get_adjusted_k__(self, item, k1, k2):
        k3 = self.kconserv_k0[k1]
        k4 = self.kconserv[k1, k2, k3]
        # For item == mknj return k1, k3, k2, k4
        # item_i = self.__mknj2i__(item)
        # result = tuple((k1, k3, k2, k4)[i] for i in numpy.argsort(item_i))
        result = (k1, k3, k2, k4)
        return result

    def assemble_block(self, item):
        result = []
        nkpts = len(self.model.kpts)
        for k1 in range(nkpts):
            result.append([])
            for k2 in range(nkpts):
                x = self.get_block_mknj_notation(item, self.__get_adjusted_k__(item, k1, k2))
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
                result[-1].append(x)
        r = numpy.block(result)
        return r / len(self.model.kpts)


class PhysERI4(PhysERI):

    def __init__(self, model, k):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. A 4-fold
        symmetry of complex-valued functions is employed in this class. Only a subset of transformed ERI is returned,
        corresponding to oscillations with a specific value of momentum transfer. The k-points of the returned ERIs come
        in two pairs :math:`(k_1 k_1 - dk | k_2 k_2 + dk)`. As a result, all diagonal blocks of the full TDHF matrix
        with different values of :math:`dk` can be reconstructed.

        Args:
            model (KRHF): the base model;
            k (int): the momentum transfer index corresponding to all momnetum pairs `i, j` satisfying
            `kconserv[k, 0, i] == j`.
        """
        super(PhysERI4, self).__init__(model, k)

    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    __calc_block__ = PhysERI4Gamma.__calc_block__.im_func


class PhysERI8(PhysERI4):
    def __init__(self, model, k):
        """
        The TDHF ERI implementation performing partial transformations of integrals to Bloch functions. An 8-fold
        symmetry of real-valued functions is employed in this class. Only a subset of transformed ERI is returned,
        corresponding to oscillations with a specific value of momentum transfer. The k-points of the returned ERIs come
        in two pairs :math:`(k_1 k_1 - dk | k_2 k_2 + dk)`. As a result, all diagonal blocks of the full TDHF matrix
        with different values of :math:`dk` can be reconstructed.

        Args:
            model (KRHF): the base model;
            k (int): the momentum transfer index corresponding to all momnetum pairs `i, j` satisfying
            `kconserv[k, 0, i] == j`.
        """
        super(PhysERI8, self).__init__(model, k)

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


def kernel(model, k, driver=None, nroots=None):
    """
    Calculates eigenstates and eigenvalues of the TDHF problem without momentum transfer.
    Args:
        model (RHF): the HF model;
        k (int): the momentum transfer index corresponding to all momnetum pairs `i, j` satisfying
        `kconserv[k, 0, i] == j`.
        driver (str): one of the drivers;
        nroots (int): the number of roots ot calculate (ignored for `driver` == 'eig');

    Returns:
        Positive eigenvalues and eigenvectors.
    """
    if numpy.iscomplexobj(model.mo_coeff):
        logger.debug1(model, "4-fold symmetry used (complex orbitals)")
        eri = PhysERI4(model, k)
    else:
        logger.debug1(model, "8-fold symmetry used (real orbitals)")
        eri = PhysERI8(model, k)
    return eig(build_matrix(eri), driver=driver, nroots=nroots)
