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

import numpy


# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are 2-electron integral routines computed directly (for debug purposes), with a 4-fold
#   symmetry and with an 8-fold symmetry
# * build_matrix builds the full TDHF matrix
# * eig performs diagonalization and selects roots
# * vector_to_amplitudes reshapes and normalizes the solution
# * kernel assembles everything
# * TDRHF provides a container


class TDDFTMatrixBlocks(object):
    symmetries = [
        ((0, 1, 2, 3), False),
    ]

    def __init__(self):
        """
        This a prototype class for transformed ERIs used in all TDHF calculations. It handles integral blocks and
        the diagonal part found in Eq. 7.5 of RevModPhys.36.844. Integral blocks are obtained via __getitem__.
        """
        self.__eri__ = {}

    def __get_mo_energies__(self, *args, **kwargs):
        raise NotImplementedError

    def get_diag_block(self, *args):
        """
        Retrieves the diagonal block.
        Args:
            *args: args passed to `__get_mo_energies__`;

        Returns:
            The diagonal block.
        """
        e_occ, e_virt = self.__get_mo_energies__(*args)
        diag = (- e_occ[:, numpy.newaxis] + e_virt[numpy.newaxis, :]).reshape(-1)
        return numpy.diag(diag).reshape((len(e_occ), len(e_virt), len(e_occ), len(e_virt)))

    def assemble_diag_block(self):
        """
        Assembles the diagonal part of the TDDFT blocks.
        Returns:
            The diagonal block.
        """
        raise NotImplementedError

    def __calc_block__(self, item, *args):
        raise NotImplementedError

    def eri_ov(self, item, *args):
        """
        Retrieves ERI block using 'ov' notation.
        Args:
            item (str): a 4-character string of 'o' and 'v' letters;
            *args: other args passed to `__calc_block__`;

        Returns:
            The corresponding block of ERI (phys notation).
        """
        if len(item) != 4 or not isinstance(item, str) or not set(item).issubset('ov'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        args = (item, ) + args
        if args in self.__eri__:
            return self.__eri__[args]

        result = self.__calc_block__(*args)

        for permutation, conjugation in self.symmetries:
            permuted_args = tuple(
                tuple(arg[_i] for _i in permutation)
                for arg in args
            )
            if conjugation:
                self.__eri__[permuted_args] = result.transpose(*permutation).conj()
            else:
                self.__eri__[permuted_args] = result.transpose(*permutation)
        return result

    def __mknj2i__(self, item):
        notation = "mknj"
        notation = dict(zip(notation, range(len(notation))))
        return tuple(notation[i] for i in item)

    def eri_mknj(self, item, *args):
        """
        Retrieves ERI block using 'mknj' notation.
        Args:
            item (str): a 4-character string of 'mknj' letters;
            *args: other arguments passed to `get_block_ov_notation`;

        Returns:
            The corresponding block of ERI (phys notation).
        """
        if len(item) != 4 or not isinstance(item, str) or set(item) != set('mknj'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        item = self.__mknj2i__(item)
        n_ov = ''.join('o' if i % 2 == 0 else 'v' for i in item)
        return self.eri_ov(n_ov, *args).transpose(*numpy.argsort(item))

    def assemble_block(self, item, *args):
        """
        Assembles the TDDFT block.
        Args:
            item (str): a 4-character string of 'mknj' letters;

        Returns:
            The TDDFT matrix block.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, str):
            spec, args = item, tuple()
        else:
            spec, args = item[0], item[1:]
        if set(spec) == set("mknj"):
            return self.assemble_block(spec, *args)
        elif set(spec).issubset("ov"):
            return self.eri_ov(spec, *args)
        else:
            raise ValueError("Unknown item: {}".format(repr(item)))


class PhysERI(TDDFTMatrixBlocks):

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a full AO-MO transformation of integrals. No symmetries are employed in
        this class.

        Args:
            model (RHF): the base model;
        """
        super(PhysERI, self).__init__()
        self.model = model
        self.__full_eri__ = self.ao2mo((self.model.mo_coeff,) * 4)

    @property
    def nocc(self):
        return int(self.model.mo_occ.sum()) // 2

    @property
    def nmo(self):
        return self.model.mo_coeff.shape[0]

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
        return self.model.mo_energy[:self.nocc], self.model.mo_energy[self.nocc:]

    def assemble_diag_block(self):
        result = self.get_diag_block()
        o1, v1, o2, v2 = result.shape
        return result.reshape(o1 * v1, o2 * v2)

    def __calc_block__(self, item):
        slc = tuple(slice(self.nocc) if i == 'o' else slice(self.nocc, None) for i in item)
        return self.__full_eri__[slc]

    def __permute_args__(self, args, order):
        assert len(args) == 0
        return tuple()

    def assemble_block(self, item):
        result = self.eri_mknj(item)
        o1, v1, o2, v2 = result.shape
        return result.reshape(o1 * v1, o2 * v2)


class PhysERI4(PhysERI):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a partial AO-MO transformation of integrals of a molecular system. A
        4-fold symmetry of complex-valued orbitals is used.

        Args:
            model (RHF): the base model;
        """
        super(PhysERI, self).__init__()
        self.model = model

    def __calc_block__(self, item):
        o = self.model.mo_coeff[:, :self.nocc]
        v = self.model.mo_coeff[:, self.nocc:]
        logger.info(self.model, "Calculating {} ...".format(item))
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

    def __init__(self, model):
        """
        The TDHF ERI implementation performing a partial AO-MO transformation of integrals of a molecular system. An
        8-fold symmetry of real-valued orbitals is used.

        Args:
            model (RHF): the base model;
        """
        super(PhysERI8, self).__init__(model)


def build_matrix(eri):
    """
    Full matrix of the TDRHF problem.
    Args:
        eri (TDDFTMatrixBlocks): ERI of the problem;

    Returns:
        The matrix.
    """
    d = eri.assemble_diag_block()
    m = numpy.array([
        [d + 2 * eri["knmj"] - eri["knjm"],   2 * eri["kjmn"] - eri["kjnm"]],
        [  - 2 * eri["mnkj"] + eri["mnjk"], - 2 * eri["mjkn"] + eri["mjnk"] - d],
    ])

    return m.transpose(0, 2, 1, 3).reshape(
        (m.shape[0] * m.shape[2], m.shape[1] * m.shape[3])
    )


def eig(m, driver=None, nroots=None):
    """
    Diagonalizes TDHF matrix.
    Args:
        m (numpy.ndarray): the matrix to diagonalize;
        driver (str): one of the drivers;
        nroots (int): the number of roots ot calculate (ignored for `driver` == 'eig');

    Returns:

    """
    if driver is None:
        driver = 'eig'
    if driver == 'eig':
        vals, vecs = numpy.linalg.eig(m)
        order = numpy.argsort(vals)
        vals = vals[order][len(vals) // 2:][:nroots]
        vecs = vecs[:, order][:, vecs.shape[1] // 2:][:, :nroots]
    else:
        raise ValueError("Unknown driver: {}".format(driver))
    return vals, vecs


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


def kernel(model, driver=None, nroots=None, return_eri=False):
    """
    Calculates eigenstates and eigenvalues of the TDHF problem.
    Args:
        model (RHF, PhysERI): the HF model or ERI;
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
    vals, vecs = eig(build_matrix(eri), driver=driver, nroots=nroots)
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
        self.xy = None
        self.e = None

    def kernel(self):
        """
        Calculates eigenstates and eigenvalues of the TDHF problem.

        Returns:
            Positive eigenvalues and eigenvectors.
        """
        self.e, self.xy, self.eri = kernel(
            self._scf if self.eri is None else self.eri,
            driver=self.driver,
            nroots=self.nroots,
            return_eri=True,
        )
        return self.e, self.xy
