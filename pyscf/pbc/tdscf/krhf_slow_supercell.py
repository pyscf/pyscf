"""
This and other `_slow` modules implement the time-dependent Hartree-Fock procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly via an AO-MO transformation, i.e. with a O(N^5) complexity scaling. As a result, regular
`numpy.linalg.eig` can be used to retrieve TDHF roots in a reliable fashion without any issues related to the Davidson
procedure. Several variants of TDHF are available:

 * `pyscf.tdscf.rhf.slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rhf_slow`: PBC (periodic boundary condition) implementation for RHF objects of `pyscf.pbc.scf`
   modules;
 * (this module) `pyscf.pbc.tdscf.krhf_slow_supercell`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules.
   Works with an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
 * `pyscf.pbc.tdscf.krhf_slow_gamma`: A Gamma-point calculation resembling the original `pyscf.pbc.tdscf.krhf`
   module. Despite its name, it accepts KRHF objects with an arbitrary number of k-points but finds only few TDHF roots
   corresponding to collective oscillations without momentum transfer;
 * `pyscf.pbc.tdscf.krhf_slow`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

from pyscf.pbc.tools import get_kconserv
from pyscf.tdscf import rhf_slow as td
from pyscf.lib import logger

from pyscf.pbc.lib.kpts_helper import loop_kkk

import numpy
import scipy


# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are 2-electron integral routines computed directly (for debug purposes), with a 4-fold
#   symmetry and with an 8-fold symmetry
# * build_matrix builds the full TDHF matrix
# * eig performs diagonalization and selects roots
# * vector_to_amplitudes reshapes and normalizes the solution
# * kernel assembles everything


def k_nocc(model):
    """
    Retrieves occupation numbers.
    Args:
        model (RHF): the model;

    Returns:
        Occupation numbers of the model.
    """
    return tuple(int(i.sum()) // 2 for i in model.mo_occ)


class PhysERI(td.TDDFTMatrixBlocks):

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a full transformation of integrals to Bloch functions. No symmetries are
        employed in this class.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI, self).__init__()
        self.model = model
        # Phys representation
        self.kconserv = get_kconserv(self.model.cell, self.model.kpts).swapaxes(1, 2)
        self.__full_eri_k__ = {}
        for k in loop_kkk(len(model.kpts)):
            k = k + (self.kconserv[k],)
            self.__full_eri_k__[k] = self.ao2mo_k(tuple(self.model.mo_coeff[j] for j in k), k)

    @property
    def nocc(self):
        """Occupation numbers."""
        return k_nocc(self.model)

    def ao2mo_k(self, coeff, k):
        """
        Phys ERI in MO basis.
        Args:
            coeff (Iterable): MO orbitals;
            k (Iterable): the 4 k-points MOs correspond to;

        Returns:
            ERI in MO basis.
        """
        coeff = (coeff[0], coeff[2], coeff[1], coeff[3])
        k = (k[0], k[2], k[1], k[3])
        result = self.model.with_df.ao2mo(coeff, tuple(self.model.kpts[i] for i in k), compact=False)
        return result.reshape(
            tuple(i.shape[1] for i in coeff)
        ).swapaxes(1, 2)

    def __get_mo_energies__(self, k1, k2):
        return self.model.mo_energy[k1][:self.nocc[k1]], self.model.mo_energy[k2][self.nocc[k2]:]

    def assemble_diag_block(self):
        result = []
        for k1 in range(len(self.model.kpts)):
            for k2 in range(len(self.model.kpts)):
                b = self.get_diag_block(k1, k2)
                o1, v1, o2, v2 = b.shape
                b = b.reshape(o1 * v1, o2 * v2)
                result.append(b)
        return scipy.linalg.block_diag(*result)

    def __calc_block__(self, item, k):
        if k in self.__full_eri_k__:
            slc = tuple(slice(self.nocc[_k]) if i == 'o' else slice(self.nocc[_k], None) for i, _k in zip(item, k))
            return self.__full_eri_k__[k][slc]
        else:
            return numpy.zeros(tuple(
                self.nocc[_k] if i == 'o' else self.model.mo_coeff[_k].shape[-1] - self.nocc[_k]
                for i, _k in zip(item, k)
            ))

    def __permute_args__(self, args, order):
        k = args[0]
        return tuple(k[i] for i in order)

    def get_block_mknj_notation(self, item, k):
        """
        Retrieves ERI block using 'mknj' notation.
        Args:
            item (str): a 4-character string of 'mknj' letters;
            k (Iterable): k indexes;

        Returns:
            The corresponding block of ERI (phys notation).
        """
        if len(item) != 4 or not isinstance(item, str) or set(item) != set('mknj'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        item_i = self.__mknj2i__(item)
        k = tuple(k[i] for i in item_i)
        return super(PhysERI, self).get_block_mknj_notation(item, k)

    def assemble_block(self, item):
        result = []
        nkpts = len(self.model.kpts)
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                result.append([])
                for k3 in range(nkpts):
                    for k4 in range(nkpts):
                        x = self.get_block_mknj_notation(item, (k1, k2, k3, k4))
                        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
                        result[-1].append(x)

        r = numpy.block(result)
        return r / len(self.model.kpts)


class PhysERI4(PhysERI):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a partial transformation of integrals to Bloch functions. A 4-fold
        symmetry of complex-valued wavefunctions is employed in this class.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI, self).__init__()
        self.model = model
        self.kconserv = get_kconserv(self.model.cell, self.model.kpts).swapaxes(1, 2)

    def __calc_block__(self, item, k):
        if self.kconserv[k[:3]] == k[3]:
            return self.ao2mo_k(tuple(
                self.model.mo_coeff[_k][:, :self.nocc[_k]] if i == "o" else self.model.mo_coeff[_k][:, self.nocc[_k]:]
                for i, _k in zip(item, k)
            ), k)
        else:
            return numpy.zeros(tuple(
                self.nocc[_k] if i == 'o' else self.model.mo_coeff[_k].shape[-1] - self.nocc[_k]
                for i, _k in zip(item, k)
            ))


class PhysERI8(PhysERI4):
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

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a partial transformation of integrals to Bloch functions. An 8-fold
        symmetry of real-valued wavefunctions is employed in this class.

        Args:
            model (KRHF): the base model;
        """
        super(PhysERI8, self).__init__(model)


build_matrix = td.build_matrix
eig = td.eig


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
    vectors = vectors.reshape(2, nk, nk, nocc, nmo-nocc, vectors.shape[1])
    norm = (abs(vectors) ** 2).sum(axis=(1, 2, 3, 4))
    norm = 2 * (norm[0] - norm[1])
    vectors /= norm ** .5
    return vectors.transpose(5, 0, 1, 2, 3, 4)


def kernel(model, driver=None, nroots=None):
    """
    Calculates eigenstates and eigenvalues of the TDHF problem.
    Args:
        model (RHF): the HF model;
        driver (str): one of the drivers;
        nroots (int): the number of roots ot calculate (ignored for `driver` == 'eig');

    Returns:
        Positive eigenvalues and eigenvectors.
    """
    if numpy.iscomplexobj(model.mo_coeff):
        logger.debug1(model, "4-fold symmetry used (complex orbitals)")
        eri = PhysERI4(model)
    else:
        logger.debug1(model, "8-fold symmetry used (real orbitals)")
        eri = PhysERI8(model)
    vals, vecs = eig(build_matrix(eri), driver=driver, nroots=nroots)
    return vals, vector_to_amplitudes(vecs, eri.nocc, model.mo_coeff[0].shape[0])
