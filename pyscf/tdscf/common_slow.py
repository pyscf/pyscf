#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly. As a result, regular `numpy.linalg.eig` can be used to retrieve TDHF roots in a reliable
fashion without any issues related to the Davidson procedure.

This is a helper module defining basic interfaces.
"""

import sys
from pyscf.lib import logger

from pyscf.pbc.tools import get_kconserv

import numpy
from scipy.linalg import solve

from itertools import count, groupby

if sys.version_info >= (3,):
    unicode = str

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


def full2ab(full, tolerance=1e-12):
    """
    Transforms a full TD matrix into A and B parts.
    Args:
        full (numpy.ndarray): the full TD matrix;
        tolerance (float): a tolerance for checking whether the full matrix is in the ABBA-form;

    Returns:
        A and B submatrices.
    """
    s = msize(full)
    if s % 2 != 0:
        raise ValueError("Not an even matrix size: {:d}".format(s))

    s2 = s // 2
    a, b = full[:s2, :s2], full[:s2, s2:]
    b_, a_ = full[s2:, :s2].conj(), full[s2:, s2:].conj()
    delta = max(abs(a + a_).max(), abs(b + b_).max())
    if delta > tolerance:
        raise ValueError("The full matrix is not in the ABBA-form, delta: {:.3e}".format(delta))

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


def ab2mkk(a, b, tolerance=1e-12):
    """
    Transforms A and B TD matrices into MK and K matrices.
    Args:
        a (numpy.ndarray): TD A-matrix;
        b (numpy.ndarray): TD B-matrix;
        tolerance (float): a tolerance for checking whether the input matrices are real;

    Returns:
        MK and K submatrices.
    """
    if max(abs(a.imag).max(), abs(b.imag).max()) > tolerance:
        raise ValueError("A- and/or B-matrixes are complex-valued: no transform is possible")
    a, b = a.real, b.real
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
    def tdhf_primary_form(self, *args, **kwargs):
        """
        A primary form of TDHF matrixes.

        Returns:
            Output type: "full", "ab", or "mk" and the corresponding matrix(es).
        """
        raise NotImplementedError

    @staticmethod
    def __check_primary_form__(m):
        if not isinstance(m, tuple):
            raise ValueError("The value returned by `tdhf_primary_form` is not a tuple")
        if len(m) < 1:
            raise ValueError("Empty tuple returned by `tdhf_primary_form`")
        if not isinstance(m[0], (str, unicode)):
            raise ValueError("The first item returned by `tdhf_primary_form` must be a string")
        forms = dict(ab=3, mk=3, full=2)
        if m[0] in forms:
            if len(m) != forms[m[0]]:
                raise ValueError("The {} form returned by `tdhf_primary_form` must contain {:d} values".format(
                    m[0].upper(), forms[m[0]],
                ))
        else:
            raise ValueError("Unknown form specification returned by `tdhf_primary_form`: {}".format(m[0]))

    def tdhf_ab_form(self, *args, **kwargs):
        """
        The A-B form of the TD problem.

        Returns:
            A and B TD matrices.
        """
        m = self.tdhf_primary_form(*args, **kwargs)
        self.__check_primary_form__(m)
        if m[0] == "ab":
            return m[1:]
        elif m[0] == "full":
            return full2ab(m[1])
        elif m[0] == "mk":
            return mkk2ab(*m[1:])

    def tdhf_full_form(self, *args, **kwargs):
        """
        The full form of the TD problem.

        Returns:
            The full TD matrix.
        """
        m = self.tdhf_primary_form(*args, **kwargs)
        self.__check_primary_form__(m)
        if m[0] == "ab":
            return ab2full(*m[1:])
        elif m[0] == "full":
            return m[1]
        elif m[0] == "mk":
            return mkk2full(*m[1:])

    def tdhf_mk_form(self, *args, **kwargs):
        """
        The MK form of the TD problem.

        Returns:
            MK and K TD matrixes.
        """
        m = self.tdhf_primary_form(*args, **kwargs)
        self.__check_primary_form__(m)
        if m[0] == "ab":
            return ab2mkk(*m[1:])
        elif m[0] == "full":
            return full2mkk(m[1])
        elif m[0] == "mk":
            return m[1:]


def mknj2i(item):
    """
    Transforms "mknj" notation into tensor index order for the ERI.
    Args:
        item (str): an arbitrary transpose of "mknj" letters;

    Returns:
        4 indexes.
    """
    notation = "mknj"
    notation = dict(zip(notation, range(len(notation))))
    return tuple(notation[i] for i in item)


class TDERIMatrixBlocks(TDMatrixBlocks):
    symmetries = [
        ((0, 1, 2, 3), False),
    ]

    def __init__(self):
        """
        This a prototype class for TD calculations based on ERI (TD-HF). It handles integral blocks and
        the diagonal part, see Eq. 7.5 of RevModPhys.36.844.
        """
        # Caching
        self.__eri__ = {}

    def __get_mo_energies__(self, *args, **kwargs):
        """This routine collects occupied and virtual MO energies."""
        raise NotImplementedError

    def __calc_block__(self, item, *args):
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

        item = mknj2i(item)
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

    def tdhf_primary_form(self, *args, **kwargs):
        """
        A primary form of TDHF matrixes (AB).

        Returns:
            Output type: "ab", and the corresponding matrixes.
        """
        d = self.tdhf_diag(*args, **kwargs)
        a = d + 2 * self["knmj"] - self["knjm"]
        b = 2 * self["kjmn"] - self["kjnm"]
        return "ab", a, b


class TDProxyMatrixBlocks(TDMatrixBlocks):
    def __init__(self, model):
        """
        This a prototype class for TD calculations based on proxying pyscf classes such as TDDFT. It is a work-around
        class. It accepts a `pyscf.tdscf.*` class and uses its matvec to construct a full-sized TD matrix.
        Args:
            model: a pyscf base model to extract TD matrix from;
        """
        super(TDProxyMatrixBlocks, self).__init__()
        self.proxy_model = model
        self.proxy_vind, self.proxy_diag = self.proxy_model.gen_vind(self.proxy_model._scf)
        self.proxy_vind = VindTracker(self.proxy_vind)

    def tdhf_primary_form(self, *args, **kwargs):
        raise NotImplementedError


def format_frozen_mol(frozen, nmo):
    """
    Formats the argument into a mask array of bools where False values correspond to frozen molecular orbitals.
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


class MolecularMFMixin(object):
    def __init__(self, model, frozen=None):
        """
        A mixin to support custom slices of mean-field attributes: `mo_coeff`, `mo_energy`, ...

        Molecular version. Also supports single k-point inputs.

        Args:
            model: the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        self.__is_k__ = False
        if "kpts" in dir(model):
            self.__is_k__ = True
            if len(model.kpts) != 1:
                raise ValueError("Only a single k-point supported, found: model.kpts = {}".format(model.kpts))
        self.model = model
        self.space = format_frozen_mol(frozen, len(self.squeeze(model.mo_energy)))

    def squeeze(self, x):
        """Squeezes quantities in the case of a PBC model."""
        return x[0] if self.__is_k__ else x

    @property
    def mo_coeff(self):
        """MO coefficients."""
        return self.squeeze(self.model.mo_coeff)[:, self.space]

    @property
    def mo_energy(self):
        """MO energies."""
        return self.squeeze(self.model.mo_energy)[self.space]

    @property
    def mo_occ(self):
        """MO occupation numbers."""
        return self.squeeze(self.model.mo_occ)[self.space]

    @property
    def nocc(self):
        """The number of occupied orbitals."""
        return int(self.squeeze(self.model.mo_occ)[self.space].sum() // 2)

    @property
    def nmo(self):
        """The total number of molecular orbitals."""
        return self.space.sum()

    @property
    def mo_coeff_full(self):
        """MO coefficients."""
        return self.squeeze(self.model.mo_coeff)

    @property
    def nocc_full(self):
        """The true (including frozen degrees of freedom) number of occupied orbitals."""
        return int(self.squeeze(self.model.mo_occ).sum() // 2)

    @property
    def nmo_full(self):
        """The true (including frozen degrees of freedom) total number of molecular orbitals."""
        return len(self.space)


def format_frozen_k(frozen, nmo, nk):
    """
    Formats the argument into a mask array of bools where False values correspond to frozen orbitals for each k-point.
    Args:
        frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals for all k-points or
        multiple lists of frozen orbitals for each k-point;
        nmo (int): the total number of molecular orbitals;
        nk (int): the total number of k-points;

    Returns:
        The mask array.
    """
    space = numpy.ones((nk, nmo), dtype=bool)
    if frozen is None:
        pass
    elif isinstance(frozen, int):
        space[:, :frozen] = False
    elif isinstance(frozen, (tuple, list, numpy.ndarray)):
        if len(frozen) > 0:
            if isinstance(frozen[0], int):
                space[:, frozen] = False
            else:
                for i in range(nk):
                    space[i, frozen[i]] = False
    else:
        raise ValueError("Cannot recognize the 'frozen' argument: expected None, int or Iterable")
    return space


def k_nocc(model):
    """
    Retrieves occupation numbers.
    Args:
        model (RHF): the model;

    Returns:
        Numbers of occupied orbitals in the model.
    """
    return tuple(int(i.sum() // 2) for i in model.mo_occ)


def k_nmo(model):
    """
    Retrieves number of AOs per k-point.
    Args:
        model (RHF): the model;

    Returns:
        Numbers of AOs in the model.
    """
    return tuple(i.shape[1] for i in model.mo_coeff)


class PeriodicMFMixin(object):
    def __init__(self, model, frozen=None):
        """
        A mixin to support custom slices of mean-field attributes: `mo_coeff`, `mo_energy`, ...

        PBC version.

        Args:
            model: the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        self.model = model
        self.space = format_frozen_k(frozen, len(model.mo_energy[0]), len(model.kpts))
        self.kconserv = get_kconserv(self.model.cell, self.model.kpts).swapaxes(1, 2)

    @property
    def mo_coeff(self):
        """MO coefficients."""
        return tuple(i[:, j] for i, j in zip(self.model.mo_coeff, self.space))

    @property
    def mo_energy(self):
        """MO energies."""
        return tuple(i[j] for i, j in zip(self.model.mo_energy, self.space))

    @property
    def mo_occ(self):
        """MO occupation numbers."""
        return tuple(i[j] for i, j in zip(self.model.mo_occ, self.space))

    @property
    def nocc(self):
        """The number of occupied orbitals."""
        return k_nocc(self)

    @property
    def nmo(self):
        """The total number of molecular orbitals."""
        return k_nmo(self)

    @property
    def mo_coeff_full(self):
        """MO coefficients."""
        return self.model.mo_coeff

    @property
    def nocc_full(self):
        """The true (including frozen degrees of freedom) number of occupied orbitals."""
        return k_nocc(self.model)

    @property
    def nmo_full(self):
        """The true (including frozen degrees of freedom) total number of molecular orbitals."""
        return k_nmo(self.model)


class VindTracker(object):
    def __init__(self, vind):
        """
        Tracks calls to `vind` (a matrix-vector multiplication density response routine).
        Args:
            vind (Callable): a matvec product routine;
        """
        self.vind = vind
        self.args = self.results = self.errors = None
        self.reset()

    def reset(self):
        """
        Resets statistics.
        """
        self.args = []
        self.results = []
        self.errors = []

    def __call__(self, v):
        if not isinstance(v, numpy.ndarray):
            raise ValueError("The input is not an array")
        self.args.append(v.shape)
        try:
            r = self.vind(v)
        except Exception as e:
            self.results.append(None)
            self.errors.append(e)
            raise
        r = numpy.array(r)
        self.results.append(r.shape)
        self.errors.append(None)
        return r

    def __iter__(self):
        for i, o, e in zip(self.args, self.results, self.errors):
            yield i, o, e

    @property
    def ncalls(self):
        return len(self.args)

    @property
    def msize(self):
        for i in self.results:
            if i is not None:
                return i[1]
        return None

    @property
    def elements_total(self):
        return self.msize ** 2

    @property
    def elements_calc(self):
        return sum(map(lambda i: i[0] * i[1] if i is not None else 0, self.results))

    @property
    def ratio(self):
        return 1.0 * self.elements_calc / self.elements_total

    def text_stats(self):
        return "--------------------\nVind call statistics\n--------------------\n" \
            "  calls: {total_calls:d}\n" \
            "  elements total: {total_elems:d} ({size})\n" \
            "  elements calculated: {total_calc:d}\n" \
            "  ratio: {ratio:.3f}".format(
                total_calls=self.ncalls,
                total_elems=self.elements_total,
                size="x".join((str(self.msize),) * 2),
                total_calc=self.elements_calc,
                ratio=self.ratio,
            )


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
        self.fast = not numpy.iscomplexobj(numpy.asanyarray(mf.mo_coeff))

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


def format_mask(x):
    """
    Formats a mask into a readable string.
    Args:
        x (ndarray): an array with the mask;

    Returns:
        A readable string with the mask.
    """
    x = numpy.asanyarray(x)
    if len(x) == 0:
        return "(empty)"
    if x.dtype == bool:
        x = numpy.argwhere(x)[:, 0]
    grps = tuple(list(g) for _, g in groupby(x, lambda n, c=count(): n-next(c)))
    return ",".join("{:d}-{:d}".format(i[0], i[-1]) if len(i) > 1 else "{:d}".format(i[0]) for i in grps)
