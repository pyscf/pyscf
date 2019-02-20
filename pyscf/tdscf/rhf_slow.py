#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent Hartree-Fock procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly via an AO-MO transformation, i.e. with a O(N^5) complexity scaling. As a result, regular
`numpy.linalg.eig` can be used to retrieve TDHF roots in a reliable fashion without any issues related to the Davidson
procedure. Several variants of TDHF are available:

 * (this module) `pyscf.tdscf.rhf_slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rhf_slow`: PBC (periodic boundary condition) implementation for RHF objects of `pyscf.pbc.scf`
   modules;
 * `pyscf.pbc.tdscf.krhf_slow_supercell`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
 * `pyscf.pbc.tdscf.krhf_slow_gamma`: A Gamma-point calculation resembling the original `pyscf.pbc.tdscf.krhf`
   module. Despite its name, it accepts KRHF objects with an arbitrary number of k-points but finds only few TDHF roots
   corresponding to collective oscillations without momentum transfer;
 * `pyscf.pbc.tdscf.krhf_slow`: PBC implementation for KRHF objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf.common_slow import TDERIMatrixBlocks, MolecularMFMixin, TDBase

import numpy

# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are 2-electron integral routines computed directly (for debug purposes), with a 4-fold
#   symmetry and with an 8-fold symmetry
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDRHF provides a container


class PhysERI(MolecularMFMixin, TDERIMatrixBlocks):
    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing a full AO-MO transformation of integrals. No symmetries are employed in
        this class.

        Args:
            model (RHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        TDERIMatrixBlocks.__init__(self)
        MolecularMFMixin.__init__(self, model, frozen=frozen)
        self.__full_eri__ = self.ao2mo((self.mo_coeff,) * 4)

    def ao2mo(self, coeff):
        """
        Phys ERI in MO basis.
        Args:
            coeff (Iterable): MO orbitals;

        Returns:
            ERI in MO basis.
        """
        coeff = (coeff[0], coeff[2], coeff[1], coeff[3])

        if "with_df" in dir(self.model):
            if "kpt" in dir(self.model):
                result = self.model.with_df.ao2mo(coeff, (self.model.kpt,) * 4, compact=False)
            else:
                result = self.model.with_df.ao2mo(coeff, compact=False)
        else:
            result = ao2mo.general(self.model.mol, coeff, compact=False)

        return result.reshape(
            tuple(i.shape[1] for i in coeff)
        ).swapaxes(1, 2)

    def __get_mo_energies__(self):
        return self.mo_energy[:self.nocc], self.mo_energy[self.nocc:]

    def __calc_block__(self, item):
        slc = tuple(slice(self.nocc) if i == 'o' else slice(self.nocc, None) for i in item)
        return self.__full_eri__[slc]


class PhysERI4(PhysERI):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing a partial AO-MO transformation of integrals of a molecular system. A
        4-fold symmetry of complex-valued orbitals is used.

        Args:
            model (RHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        TDERIMatrixBlocks.__init__(self)
        MolecularMFMixin.__init__(self, model, frozen=frozen)

    def __calc_block__(self, item):
        o = self.mo_coeff[:, :self.nocc]
        v = self.mo_coeff[:, self.nocc:]
        logger.info(self.model, "Computing {} ...".format(''.join(item)))
        return self.ao2mo(tuple(o if i == "o" else v for i in item))


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

    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing a partial AO-MO transformation of integrals of a molecular system. An
        8-fold symmetry of real-valued orbitals is used.

        Args:
            model (RHF): the base model;
        """
        super(PhysERI8, self).__init__(model, frozen=frozen)


def vector_to_amplitudes(vectors, nocc, nmo):
    """
    Transforms (reshapes) and normalizes vectors into amplitudes.
    Args:
        vectors (numpy.ndarray): raw eigenvectors to transform;
        nocc (int): number of occupied orbitals;
        nmo (int): the total number of orbitals;

    Returns:
        Amplitudes with the following shape: (# of roots, 2 (x or y), # of occupied orbitals, # of virtual orbitals).
    """
    vectors = numpy.asanyarray(vectors)
    vectors = vectors.reshape(2, nocc, nmo-nocc, vectors.shape[1])
    norm = (abs(vectors) ** 2).sum(axis=(1, 2))
    norm = 2 * (norm[0] - norm[1])
    vectors /= norm ** .5
    return vectors.transpose(3, 0, 1, 2)


class TDRHF(TDBase):
    eri1 = PhysERI
    eri4 = PhysERI4
    eri8 = PhysERI8
    v2a = staticmethod(vector_to_amplitudes)

    def ao2mo(self):
        """
        Picks ERI: either 4-fold or 8-fold symmetric.

        Returns:
            A suitable ERI.
        """
        if numpy.iscomplexobj(self._scf.mo_coeff):
            if self.eri4 is not None:
                logger.debug1(self._scf, "4-fold symmetry used (complex orbitals)")
                return self.eri4(self._scf, frozen=self.frozen)
            elif self.eri1 is not None:
                logger.debug1(self._scf, "fallback: no symmetry used (complex orbitals)")
                return self.eri1(self._scf, frozen=self.frozen)
            else:
                raise RuntimeError("Failed to pick ERI for complex MOs: both eri1 and eri4 are None")
        else:
            if self.eri8 is not None:
                logger.debug1(self._scf, "8-fold symmetry used (real orbitals)")
                return self.eri8(self._scf, frozen=self.frozen)
            elif self.eri1 is not None:
                logger.debug1(self._scf, "fallback: no symmetry used (real orbitals)")
                return self.eri1(self._scf, frozen=self.frozen)
            else:
                raise RuntimeError("Failed to pick ERI for real MOs: both eri1 and eri8 are None")
