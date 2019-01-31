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
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDRHF provides a container


class TDDFTMatrixBlocks(object):
    symmetries = [
        ((0, 1, 2, 3), False),
    ]

    def __init__(self):
        """
        This a prototype class for transformed ERIs used in all TDHF calculations. It handles integral blocks and
        the diagonal part found in Eq. 7.5 of RevModPhys.36.844. Important routines are:
         * tdhf_diag - builds a diagonal block;
         * eri_ov - ERI in ov notation (4-tensor);
         * eri_mknj - ERI in mknj notation (matrix with paired dimensions);
        """
        self.__eri__ = {}

    def __get_mo_energies__(self, *args, **kwargs):
        """This routine collects occupied and virtual MO energies."""
        raise NotImplementedError

    def tdhf_diag(self, *args):
        """
        Retrieves the diagonal block.
        Args:
            *args: args passed to `__get_mo_energies__`;

        Returns:
            The diagonal block.
        """
        e_occ, e_virt = self.__get_mo_energies__(*args)
        diag = (- e_occ[:, numpy.newaxis] + e_virt[numpy.newaxis, :]).reshape(-1)
        return numpy.diag(diag).reshape((len(e_occ) * len(e_virt), len(e_occ) * len(e_virt)))

    def __calc_block__(self, item, *args):
        raise NotImplementedError

    def eri_ov(self, item, *args):
        """
        Retrieves ERI block using 'ov' notation.
        Args:
            item (str): a 4-character string of 'o' and 'v' letters;
            *args: other args passed to `__calc_block__`;

        Returns:
            The corresponding block of ERI (4-tensor, phys notation).
        """
        if len(item) != 4 or not isinstance(item, str) or not set(item).issubset('ov'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        args = (tuple(item), ) + args
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
            The corresponding block of ERI (matrix with paired dimensions).
        """
        if len(item) != 4 or not isinstance(item, str) or set(item) != set('mknj'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        item = self.__mknj2i__(item)
        n_ov = ''.join('o' if i % 2 == 0 else 'v' for i in item)
        args = tuple(
            tuple(arg[i] for i in item)
            for arg in args
        )
        result = self.eri_ov(n_ov, *args).transpose(*numpy.argsort(item))
        i, j, k, l = result.shape
        result = result.reshape((i * j, k * l))
        return result

    def __getitem__(self, item):
        if isinstance(item, str):
            spec, args = item, tuple()
        else:
            spec, args = item[0], item[1:]
        if set(spec) == set("mknj"):
            return self.eri_mknj(spec, *args)
        elif set(spec).issubset("ov"):
            return self.eri_ov(spec, *args)
        else:
            raise ValueError("Unknown item: {}".format(repr(item)))

    def tdhf_matrix(self):
        """
        Full matrix of the TDRHF problem.
        Returns:
            The matrix.
        """
        d = self.tdhf_diag()
        m = numpy.array([
            [d + 2 * self["knmj"] - self["knjm"], 2 * self["kjmn"] - self["kjnm"]],
            [- 2 * self["mnkj"] + self["mnjk"], - 2 * self["mjkn"] + self["mjnk"] - d],
        ])

        return m.transpose(0, 2, 1, 3).reshape(
            (m.shape[0] * m.shape[2], m.shape[1] * m.shape[3])
        )


def format_frozen(frozen, nmo):
    """
    Formats the argument into a mask array of bools where False values correspond to frozen orbitals.
    Args:
        frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        nmo (int): the total number of molecular orbitals;

    Returns:
        The mask array.
    """
    space = numpy.ones(nmo, dtype=bool)
    if frozen is None:
        pass
    elif isinstance(frozen, int):
        space[:frozen] = False
    elif isinstance(frozen, (tuple, list, numpy.ndarray)):
        space[frozen] = False
    else:
        raise ValueError("Cannot recognize the 'frozen' argument: expected None, int or Iterable")
    return space


class PhysERI(TDDFTMatrixBlocks):

    def __init__(self, model, frozen=None):
        """
        The TDHF ERI implementation performing a full AO-MO transformation of integrals. No symmetries are employed in
        this class.

        Args:
            model (RHF): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        super(PhysERI, self).__init__()
        self.model = model
        self.space = format_frozen(frozen, len(model.mo_energy))
        self.__full_eri__ = self.ao2mo((self.mo_coeff,) * 4)

    @property
    def mo_coeff(self):
        return self.model.mo_coeff[:, self.space]

    @property
    def mo_energy(self):
        return self.model.mo_energy[self.space]

    @property
    def nocc(self):
        return int(self.model.mo_occ[self.space].sum() // 2)

    @property
    def nmo(self):
        return self.space.sum()

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
        TDDFTMatrixBlocks.__init__(self)
        self.model = model
        self.space = format_frozen(frozen, len(model.mo_energy))

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


def kernel(eri, driver=None, nroots=None, **kwargs):
    """
    Calculates eigenstates and eigenvalues of the TDHF problem.
    Args:
        eri (TDDFTMatrixBlocks): ERI;
        driver (str): one of the drivers;
        nroots (int): the number of roots to calculate;
        **kwargs: arguments to `eri.tdhf_matrix`;

    Returns:
        Positive eigenvalues and eigenvectors.
    """
    if not isinstance(eri, TDDFTMatrixBlocks):
        raise ValueError("The argument must be ERI object")
    logger.debug1(eri.model, "Preparing TDHF matrix ...")
    m = eri.tdhf_matrix(**kwargs)
    logger.debug1(eri.model, "Diagonalizing a {} matrix with {} ...".format(
        'x'.join(map(str, m.shape)),
        "'{}'".format(driver) if driver is not None else "a default method",
    ))
    vals, vecs = eig(m, driver=driver, nroots=nroots)
    return vals, vecs


class TDRHF(object):
    eri1 = None
    eri4 = PhysERI4
    eri8 = PhysERI8
    v2a = staticmethod(vector_to_amplitudes)

    def __init__(self, mf, frozen=None):
        """
        Performs TDHF calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RHF): the base restricted Hartree-Fock model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        self._scf = mf
        self.driver = None
        self.nroots = None
        self.eri = None
        self.xy = None
        self.e = None
        self.frozen = frozen

    def __kernel__(self, **kwargs):
        """Silent implementation of kernel."""
        if self.eri is None:
            self.eri = self.ao2mo()

        e, xy = kernel(
            self.eri,
            driver=self.driver,
            nroots=self.nroots,
            **kwargs
        )
        xy = self.vector_to_amplitudes(xy)
        return e, xy

    def kernel(self):
        """
        Calculates eigenstates and eigenvalues of the TDHF problem.

        Returns:
            Positive eigenvalues and eigenvectors.
        """
        self.e, self.xy = self.__kernel__()
        return self.e, self.xy

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

    def vector_to_amplitudes(self, vectors):
        """
        Transforms (reshapes) and normalizes vectors into amplitudes.
        Args:
            vectors (numpy.ndarray): raw eigenvectors to transform;

        Returns:
            Amplitudes with the following shape: (# of roots, 2 (x or y), # of occupied orbitals, # of virtual orbitals).
        """
        return self.v2a(vectors, self.eri.nocc, self.eri.nmo)
