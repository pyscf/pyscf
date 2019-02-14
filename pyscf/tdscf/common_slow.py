#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly. As a result, regular `numpy.linalg.eig` can be used to retrieve TDHF roots in a reliable
fashion without any issues related to the Davidson procedure.

This is a helper module defining basic interfaces.
"""

from pyscf.lib import logger

import numpy
from scipy.linalg import solve


def msize(m):
    """
    Checks whether the matrix is square and returns its size.
    Args:
        m (numpy.ndarray): the matrix to measure;

    Returns:
        An integer with the size.
    """
    s = m.shape[0]
    if m.shape != (s, s):
        raise ValueError("Do not recognize the shape (must be a square matrix): {}".format(m.shape))
    return s


def full2ab(full):
    """
    Transforms a full TD matrix into A and B parts.
    Args:
        full (numpy.ndarray): the full TD matrix;

    Returns:
        A and B submatrices.
    """
    s = msize(full)
    if s % 2 != 0:
        raise ValueError("Not an even matrix size: {:d}".format(s))

    s2 = s // 2
    return full[:s2, :s2], full[:s2, s2:]


def ab2full(a, b):
    """
    Transforms A and B TD matrices into a full matrix.
    Args:
        a (numpy.ndarray): TD A-matrix;
        b (numpy.ndarray): TD B-matrix;

    Returns:
        The full TD matrix.
    """
    sa = msize(a)
    sb = msize(b)
    if sa != sb:
        raise ValueError("Input matrix dimensions do not match: {:d} vs {:d}".format(sa, sb))
    return numpy.block([[a, b], [-b.conj(), -a.conj()]])


def ab2mkk(a, b):
    """
    Transforms A and B TD matrices into MK and K matrices.
    Args:
        a (numpy.ndarray): TD A-matrix;
        b (numpy.ndarray): TD B-matrix;

    Returns:
        MK and K submatrices.
    """
    if numpy.iscomplexobj(a) or numpy.iscomplexobj(b):
        raise ValueError("A- and/or B-matrixes are complex-valued: no transform is possible")
    tdhf_k, tdhf_m = a - b, a + b
    tdhf_mk = tdhf_m.dot(tdhf_k)
    return tdhf_mk, tdhf_k


def mkk2ab(mk, k):
    """
    Transforms MK and M TD matrices into A and B matrices.
    Args:
        mk (numpy.ndarray): TD MK-matrix;
        k (numpy.ndarray): TD K-matrix;

    Returns:
        A and B submatrices.
    """
    if numpy.iscomplexobj(mk) or numpy.iscomplexobj(k):
        raise ValueError("MK- and/or K-matrixes are complex-valued: no transform is possible")
    m = solve(k.T, mk.T).T
    a = 0.5 * (m + k)
    b = 0.5 * (m - k)
    return a, b


def full2mkk(full):
    """
    Transforms a full TD matrix into MK and K parts.
    Args:
        full (numpy.ndarray): the full TD matrix;

    Returns:
        MK and K submatrices.
    """
    return ab2mkk(*full2ab(full))


def mkk2full(mk, k):
    """
    Transforms MK and M TD matrices into a full TD matrix.
    Args:
        mk (numpy.ndarray): TD MK-matrix;
        k (numpy.ndarray): TD K-matrix;

    Returns:
        The full TD matrix.
    """
    return ab2full(*mkk2ab(mk, k))


class TDMatrixBlocks(object):
    # When implementing, the primary driver has to be chosen. It can be ither of 'full', 'ab' or 'mk'.
    primary_driver = "full"

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

    def tdhf_ab_form(self, *args, **kwargs):
        """
        The A-B form of the TD problem.

        Returns:
            A and B TD matrices.
        """
        if self.primary_driver == "ab":
            raise NotImplementedError("The primary driver is set to 'ab' but `tdhf_ab_form` is not implemented")
        elif self.primary_driver == "full":
            return full2ab(self.tdhf_full_form(*args, **kwargs))
        elif self.primary_driver == "mk":
            return mkk2ab(*self.tdhf_mk_form(*args, **kwargs))
        else:
            raise RuntimeError("Unknown primary driver set: '{}'".format(self.primary_driver))

    def tdhf_full_form(self, *args, **kwargs):
        """
        The full form of the TD problem.

        Returns:
            The full TD matrix.
        """
        if self.primary_driver == "ab":
            return ab2full(*self.tdhf_ab_form(*args, **kwargs))
        elif self.primary_driver == "full":
            raise NotImplementedError("The primary driver is set to 'full' but `tdhf_full_form` is not implemented")
        elif self.primary_driver == "mk":
            return mkk2full(*self.tdhf_mk_form(*args, **kwargs))
        else:
            raise RuntimeError("Unknown primary driver set: '{}'".format(self.primary_driver))

    def tdhf_mk_form(self, *args, **kwargs):
        """
        The MK form of the TD problem.

        Returns:
            MK and K TD matrixes.
        """
        if self.primary_driver == "ab":
            return ab2mkk(*self.tdhf_ab_form(*args, **kwargs))
        elif self.primary_driver == "full":
            return full2mkk(self.tdhf_full_form(*args, **kwargs))
        elif self.primary_driver == "mk":
            raise NotImplementedError("The primary driver is set to 'mk' but `tdhf_mk_form` is not implemented")
        else:
            raise RuntimeError("Unknown primary driver set: '{}'".format(self.primary_driver))


class TDERIMatrixBlocks(TDMatrixBlocks):
    symmetries = [
        ((0, 1, 2, 3), False),
    ]
    primary_driver = "ab"

    def __init__(self):
        """
        This a prototype class for transformed ERIs used in all TDHF calculations. It handles integral blocks and
        the diagonal part found in Eq. 7.5 of RevModPhys.36.844.
        """
        # Caching
        self.__eri__ = {}

    def __get_mo_energies__(self, *args, **kwargs):
        """This routine collects occupied and virtual MO energies."""
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

    def tdhf_ab_form(self, *args, **kwargs):
        """
        The A-B form of the TD problem.

        Returns:
            A and B TD matrices.
        """
        d = self.tdhf_diag()
        return d + 2 * self["knmj"] - self["knjm"], 2 * self["kjmn"] - self["kjnm"]


def eig(m, driver=None, nroots=None, half=True):
    """
    Eigenvalue problem solver.
    Args:
        m (numpy.ndarray): the matrix to diagonalize;
        driver (str): one of the drivers;
        nroots (int): the number of roots ot calculate (ignored for `driver` == 'eig');
        half (bool): if True, implies spectrum symmetry and takes only a half of eigenvalues;

    Returns:

    """
    if driver is None:
        driver = 'eig'
    if driver == 'eig':
        vals, vecs = numpy.linalg.eig(m)
        order = numpy.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        if half:
            vals, vecs = vals[len(vals) // 2:], vecs[:, vecs.shape[1] // 2:]
            vecs = vecs[:, ]
        vals, vecs = vals[:nroots], vecs[:, :nroots]
    else:
        raise ValueError("Unknown driver: {}".format(driver))
    return vals, vecs


def kernel(eri, driver=None, fast=True, nroots=None, **kwargs):
    """
    Calculates eigenstates and eigenvalues of the TDHF problem.
    Args:
        eri (TDDFTMatrixBlocks): ERI;
        driver (str): one of the eigenvalue problem drivers;
        fast (bool): whether to run diagonalization on smaller matrixes;
        nroots (int): the number of roots to calculate;
        **kwargs: arguments to `eri.tdhf_matrix`;

    Returns:
        Positive eigenvalues and eigenvectors.
    """
    if not isinstance(eri, TDMatrixBlocks):
        raise ValueError("The argument must be ERI object")

    if fast:

        logger.debug1(eri.model, "Preparing TDHF matrix (fast) ...")
        tdhf_mk, tdhf_k = eri.tdhf_mk_form(**kwargs)
        logger.debug1(eri.model, "Diagonalizing a {} matrix with {} ...".format(
            'x'.join(map(str, tdhf_mk.shape)),
            "'{}'".format(driver) if driver is not None else "a default method",
        ))
        vals, vecs_x = eig(tdhf_mk, driver=driver, nroots=nroots, half=False)

        vals = vals ** .5
        vecs_y = (1. / vals)[numpy.newaxis, :] * tdhf_k.dot(vecs_x)
        vecs_u, vecs_v = vecs_y + vecs_x, vecs_y - vecs_x
        return vals, numpy.concatenate((vecs_u, vecs_v), axis=0)

    else:

        logger.debug1(eri.model, "Preparing TDHF matrix ...")
        m = eri.tdhf_full_form(**kwargs)
        logger.debug1(eri.model, "Diagonalizing a {} matrix with {} ...".format(
            'x'.join(map(str, m.shape)),
            "'{}'".format(driver) if driver is not None else "a default method",
        ))
        return eig(m, driver=driver, nroots=nroots)


class TDBase(object):
    v2a = None

    def __init__(self, mf, frozen=None):
        """
        Performs TD calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf: the mean-field model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        self._scf = mf
        self.driver = None
        self.nroots = None
        self.eri = None
        self.xy = None
        self.e = None
        self.frozen = frozen
        self.fast = not numpy.iscomplexobj(mf.mo_coeff)

    def __kernel__(self, **kwargs):
        """Silent implementation of kernel which does not change attributes."""
        if self.eri is None:
            self.eri = self.ao2mo()

        e, xy = kernel(
            self.eri,
            driver=self.driver,
            nroots=self.nroots,
            fast=self.fast,
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
        raise NotImplementedError

    def vector_to_amplitudes(self, vectors):
        """
        Transforms (reshapes) and normalizes vectors into amplitudes.
        Args:
            vectors (numpy.ndarray): raw eigenvectors to transform;

        Returns:
            Amplitudes with the following shape: (# of roots, 2 (x or y), # of occupied orbitals, # of virtual orbitals).
        """
        return self.v2a(vectors, self.eri.nocc, self.eri.nmo)
