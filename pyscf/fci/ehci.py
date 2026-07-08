#!/usr/bin/env python
# Copyright 2026
#
# Prototype hierarchy configuration interaction (hCI / ehCI) solver for PySCF.
#
# Place this file in:
#     pyscf/fci/ehci.py
#
# It implements the hierarchy parameter introduced by Kossoski, Damour,
# and Loos:
#
#     h = (e + s/2) / 2 = 0.5 * e + 0.25 * s
#
# where e is the excitation degree relative to a reference determinant and
# s is the seniority number, i.e. the number of singly occupied spatial
# orbitals.
#
# This implementation supports arbitrary determinant-pair spaces. It does
# NOT variationally include the full alpha-string x beta-string direct product.
# Instead, it stores the hCI wave function in a compressed determinant-pair
# basis and uses PySCF selected_ci contractions internally by unpacking to a
# rectangular alpha x beta array, applying H, and projecting back.
#
# This is meant as a correctness-first implementation. It reuses PySCF's
# selected_ci Hamiltonian contraction, so it is much easier to validate but
# not asymptotically optimal for very sparse determinant-pair spaces.

import numpy as np

from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import selected_ci


# -----------------------------------------------------------------------------
# Basic bit-string utilities
# -----------------------------------------------------------------------------

def _popcount(x):
    """Return the number of set bits in integer x."""
    return int(x).bit_count()


def _as_int_tuple(pair):
    """Convert a determinant pair to plain Python ints."""
    return int(pair[0]), int(pair[1])


def make_reference_strings(norb, nelec, ref_strs=None, spin=None):
    r"""
    Return reference alpha and beta strings.

    By default, this is the closed-/open-shell Aufbau determinant in the
    current orbital ordering:

        alpha orbitals: 0, 1, ..., neleca-1
        beta  orbitals: 0, 1, ..., nelecb-1

    Parameters
    ----------
    norb : int
        Number of spatial orbitals.
    nelec : int or tuple[int, int]
        Number of electrons. If int, PySCF's spin convention is used.
    ref_strs : tuple[int, int] or None
        Optional user-supplied (alpha_ref, beta_ref).
    spin : int or None
        PySCF spin = neleca - nelecb, used only when nelec is an int.

    Returns
    -------
    alpha_ref, beta_ref : int, int
    """
    neleca, nelecb = direct_spin1._unpack_nelec(nelec, spin)

    if ref_strs is None:
        alpha_ref = (1 << neleca) - 1
        beta_ref = (1 << nelecb) - 1
        return alpha_ref, beta_ref

    if len(ref_strs) != 2:
        raise ValueError("ref_strs must be a tuple/list: (alpha_ref, beta_ref)")

    alpha_ref, beta_ref = int(ref_strs[0]), int(ref_strs[1])

    if alpha_ref < 0 or beta_ref < 0:
        raise ValueError("Reference strings must be non-negative integers")
    if alpha_ref >= (1 << norb) or beta_ref >= (1 << norb):
        raise ValueError("Reference string contains orbital index >= norb")
    if _popcount(alpha_ref) != neleca:
        raise ValueError(
            f"alpha_ref has {_popcount(alpha_ref)} electrons but neleca={neleca}"
        )
    if _popcount(beta_ref) != nelecb:
        raise ValueError(
            f"beta_ref has {_popcount(beta_ref)} electrons but nelecb={nelecb}"
        )

    return alpha_ref, beta_ref


def seniority(alpha_str, beta_str):
    r"""
    Seniority number of determinant (alpha_str, beta_str).

    The seniority number is the number of spatial orbitals that are singly
    occupied. In bit form this is simply:

        seniority = popcount(alpha XOR beta)

    because alpha XOR beta is 1 exactly on orbitals occupied by only one spin.
    """
    return _popcount(int(alpha_str) ^ int(beta_str))


def excitation_degree(alpha_str, beta_str, alpha_ref, beta_ref):
    r"""
    Excitation degree e relative to a reference determinant.

    For fixed electron number, e is the number of occupied spin orbitals in the
    reference that are not occupied in the current determinant. Equivalently,
    it is the number of holes in the reference.
    """
    alpha_str = int(alpha_str)
    beta_str = int(beta_str)
    alpha_ref = int(alpha_ref)
    beta_ref = int(beta_ref)

    holes_a = alpha_ref & (~alpha_str)
    holes_b = beta_ref & (~beta_str)
    return _popcount(holes_a) + _popcount(holes_b)


def hierarchy_value(alpha_str, beta_str, alpha_ref, beta_ref):
    r"""
    Return h = 0.5 e + 0.25 s for a determinant.
    """
    e = excitation_degree(alpha_str, beta_str, alpha_ref, beta_ref)
    s = seniority(alpha_str, beta_str)
    return 0.5 * e + 0.25 * s


def determinant_info(alpha_str, beta_str, alpha_ref, beta_ref):
    """Return (excitation_degree, seniority, hierarchy_value)."""
    e = excitation_degree(alpha_str, beta_str, alpha_ref, beta_ref)
    s = seniority(alpha_str, beta_str)
    h = 0.5 * e + 0.25 * s
    return e, s, h


def hierarchy_allows(alpha_str, beta_str, alpha_ref, beta_ref, h, tol=1e-12):
    """Return True if determinant belongs to hCI(h)."""
    return hierarchy_value(alpha_str, beta_str, alpha_ref, beta_ref) <= float(h) + tol


# -----------------------------------------------------------------------------
# Determinant generation and hCI selection
# -----------------------------------------------------------------------------

def gen_fci_strings(norb, nelec, spin=None):
    """
    Generate all alpha and beta strings for the full CI space.

    Returns
    -------
    alpha_strs, beta_strs : np.ndarray, np.ndarray
    """
    neleca, nelecb = direct_spin1._unpack_nelec(nelec, spin)
    alpha_strs = cistring.make_strings(range(norb), neleca)
    beta_strs = cistring.make_strings(range(norb), nelecb)
    return (np.asarray(alpha_strs, dtype=np.int64),
            np.asarray(beta_strs, dtype=np.int64))


def allowed_sectors(norb, nelec, h, ref_strs=None, spin=None):
    """
    Count allowed (excitation degree, seniority) sectors for hCI(h).

    This is a diagnostic helper. It generates determinant pairs and counts how
    many determinants belong to each (e, s) block.

    Returns
    -------
    sectors : dict
        Dictionary mapping (e, s) -> count.
    """
    alpha_ref, beta_ref = make_reference_strings(norb, nelec, ref_strs, spin)
    alpha_strs, beta_strs = gen_fci_strings(norb, nelec, spin)

    sectors = {}
    for a in alpha_strs:
        for b in beta_strs:
            e, s, hv = determinant_info(a, b, alpha_ref, beta_ref)
            if hv <= float(h) + 1e-12:
                sectors[(e, s)] = sectors.get((e, s), 0) + 1
    return sectors


def print_allowed_sectors(norb, nelec, h, ref_strs=None, spin=None):
    """Pretty-print allowed hCI sectors."""
    sectors = allowed_sectors(norb, nelec, h, ref_strs=ref_strs, spin=spin)
    print(f"Allowed hCI sectors for h = {h}:")
    print("    e    s    h(e,s)    ndet")
    print("  ---- ---- --------- -------")
    for (e, s), n in sorted(sectors.items()):
        hv = 0.5 * e + 0.25 * s
        print(f"  {e:4d} {s:4d} {hv:9.3f} {n:7d}")
    print(f"Total determinants = {sum(sectors.values())}")


def gen_ehci_pairs(norb, nelec, h, ref_strs=None, spin=None, return_info=False):
    r"""
    Generate determinant pairs belonging to hCI(h).

    A determinant pair (alpha, beta) is included if

        0.5 * e + 0.25 * s <= h

    where e is the excitation degree relative to ref_strs and s is the
    seniority number.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals.
    nelec : int or tuple[int, int]
        Electron count.
    h : float
        Hierarchy parameter.
    ref_strs : tuple[int, int] or None
        Reference determinant. If None, use Aufbau/HF reference.
    spin : int or None
        PySCF spin = neleca - nelecb.
    return_info : bool
        If True, also return per-determinant information array with columns
        (e, s, h_value).

    Returns
    -------
    pairs : np.ndarray, shape (ndet, 2), dtype int64
        Selected determinant pairs.
    info : np.ndarray, optional, shape (ndet, 3)
        Columns are e, s, h_value.
    """
    h = float(h)
    if h < 0:
        raise ValueError("h must be non-negative")

    alpha_ref, beta_ref = make_reference_strings(norb, nelec, ref_strs, spin)
    alpha_strs, beta_strs = gen_fci_strings(norb, nelec, spin)

    pairs = []
    infos = []

    for a in alpha_strs:
        ia = int(a)
        for b in beta_strs:
            ib = int(b)
            e, s, hv = determinant_info(ia, ib, alpha_ref, beta_ref)
            if hv <= h + 1e-12:
                pairs.append((ia, ib))
                infos.append((e, s, hv))

    if not pairs:
        raise RuntimeError(f"No determinants selected for h={h}")

    pairs = np.asarray(pairs, dtype=np.int64)
    infos = np.asarray(infos, dtype=float)

    # Sort by alpha string, then beta string for deterministic behavior.
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    infos = infos[order]

    if return_info:
        return pairs, infos
    return pairs


# -----------------------------------------------------------------------------
# Pair-space representation
# -----------------------------------------------------------------------------

class PairSpace(object):
    """
    Representation of an arbitrary selected determinant-pair space.

    PySCF selected_ci works naturally with a rectangular alpha x beta array.
    hCI, however, generally selects specific determinant pairs. PairSpace stores
    the compressed selected-pair list and the maps needed to unpack to / pack
    from the rectangular selected_ci representation.
    """

    def __init__(self, pairs):
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (ndet, 2)")
        if len(pairs) == 0:
            raise ValueError("pairs cannot be empty")

        # Remove duplicates and sort.
        pairs = np.asarray(sorted(set(map(tuple, pairs.tolist()))), dtype=np.int64)

        alpha_strs = np.asarray(sorted({int(a) for a, b in pairs}), dtype=np.int64)
        beta_strs = np.asarray(sorted({int(b) for a, b in pairs}), dtype=np.int64)

        a_addr = {int(a): i for i, a in enumerate(alpha_strs)}
        b_addr = {int(b): i for i, b in enumerate(beta_strs)}

        allowed = np.empty((len(pairs), 2), dtype=np.int64)
        for k, (a, b) in enumerate(pairs):
            allowed[k, 0] = a_addr[int(a)]
            allowed[k, 1] = b_addr[int(b)]

        mask = np.zeros((len(alpha_strs), len(beta_strs)), dtype=bool)
        mask[allowed[:, 0], allowed[:, 1]] = True

        self.pairs = pairs
        self.alpha_strs = alpha_strs
        self.beta_strs = beta_strs
        self.ci_strs = (alpha_strs, beta_strs)
        self.allowed = allowed
        self.mask = mask
        self.ndet = len(pairs)
        self.na = len(alpha_strs)
        self.nb = len(beta_strs)

    def unpack(self, c):
        """
        Convert compressed selected-pair coefficients to a rectangular array.
        """
        c = np.asarray(c)
        if c.size != self.ndet:
            raise ValueError(f"Compressed vector has size {c.size}; expected {self.ndet}")
        rect = np.zeros((self.na, self.nb), dtype=c.dtype)
        rect[self.allowed[:, 0], self.allowed[:, 1]] = c.reshape(-1)
        return rect

    def pack(self, rect):
        """
        Project a rectangular alpha x beta array back to the selected-pair basis.
        """
        rect = np.asarray(rect).reshape(self.na, self.nb)
        return np.asarray(rect[self.allowed[:, 0], self.allowed[:, 1]])

    def pair_index(self, alpha_str, beta_str):
        """Return compressed index for pair, or None if absent."""
        target = (int(alpha_str), int(beta_str))
        for k, pair in enumerate(self.pairs):
            if (int(pair[0]), int(pair[1])) == target:
                return k
        return None

    def summary(self):
        return {
            "ndet": self.ndet,
            "nalpha_strings": self.na,
            "nbeta_strings": self.nb,
            "rectangular_size": self.na * self.nb,
            "sparsity_fraction": self.ndet / float(self.na * self.nb),
        }


class EHCIVector(np.ndarray):
    """Compressed hCI coefficient vector carrying PairSpace metadata."""

    def __array_finalize__(self, obj):
        self._pair_space = getattr(obj, "_pair_space", None)
        self._strs = getattr(obj, "_strs", None)
        self._pairs = getattr(obj, "_pairs", None)
        self._hci_info = getattr(obj, "_hci_info", None)

    def __array_wrap__(self, out, context=None, return_scalar=False):
        if out.shape == self.shape:
            return out
        elif out.shape == ():
            return out[()]
        else:
            return out.view(np.ndarray)


def _as_ehci_vector(c, pair_space, hci_info=None):
    """Tag a 1D ndarray as an EHCIVector."""
    c = np.asarray(c).reshape(pair_space.ndet)
    out = c.view(EHCIVector)
    out._pair_space = pair_space
    out._strs = pair_space.ci_strs
    out._pairs = pair_space.pairs
    out._hci_info = hci_info
    return out


def _compressed_to_rect_sci(civec, pair_space=None):
    """
    Convert compressed hCI vector to PySCF selected_ci.SCIvector.
    """
    if pair_space is None:
        pair_space = getattr(civec, "_pair_space", None)
    if pair_space is None:
        raise ValueError("pair_space was not provided and not found on civec")
    rect = pair_space.unpack(np.asarray(civec).reshape(-1))
    return selected_ci._as_SCIvector(rect, pair_space.ci_strs)


def to_selected_ci(civec, pair_space=None):
    """
    Public helper: convert compressed hCI vector to selected_ci.SCIvector.
    """
    return _compressed_to_rect_sci(civec, pair_space)


def to_fci(civec, norb, nelec, pair_space=None, spin=None):
    """
    Convert compressed hCI vector to full PySCF FCI rectangular array.

    Returns an array of shape:

        (num_alpha_fci_strings, num_beta_fci_strings)
    """
    neleca, nelecb = direct_spin1._unpack_nelec(nelec, spin)
    if pair_space is None:
        pair_space = getattr(civec, "_pair_space", None)
    if pair_space is None:
        raise ValueError("pair_space was not provided and not found on civec")

    na_fci = cistring.num_strings(norb, neleca)
    nb_fci = cistring.num_strings(norb, nelecb)
    fcivec = np.zeros((na_fci, nb_fci), dtype=np.asarray(civec).dtype)

    c = np.asarray(civec).reshape(pair_space.ndet)
    for coeff, (a, b) in zip(c, pair_space.pairs):
        ia = cistring.str2addr(norb, neleca, int(a))
        ib = cistring.str2addr(norb, nelecb, int(b))
        fcivec[ia, ib] = coeff

    return fcivec


def from_fci(fcivec, pairs, norb, nelec, spin=None):
    """
    Extract compressed hCI vector from a full FCI vector and selected pairs.
    """
    neleca, nelecb = direct_spin1._unpack_nelec(nelec, spin)
    pair_space = PairSpace(pairs)
    fcivec = np.asarray(fcivec).reshape(cistring.num_strings(norb, neleca),
                                       cistring.num_strings(norb, nelecb))
    c = np.zeros(pair_space.ndet, dtype=fcivec.dtype)
    for k, (a, b) in enumerate(pair_space.pairs):
        ia = cistring.str2addr(norb, neleca, int(a))
        ib = cistring.str2addr(norb, nelecb, int(b))
        c[k] = fcivec[ia, ib]
    return _as_ehci_vector(c, pair_space)


# -----------------------------------------------------------------------------
# Davidson kernel in compressed determinant-pair space
# -----------------------------------------------------------------------------

def _make_init_guess(hdiag, nroots, ref_index=None):
    """
    Build simple determinant initial guesses in compressed pair space.
    """
    hdiag = np.asarray(hdiag)
    ndet = hdiag.size
    if ndet < nroots:
        raise RuntimeError(f"hCI space has only {ndet} determinants; nroots={nroots}")

    if ref_index is not None and 0 <= ref_index < ndet:
        order = [int(ref_index)]
        for x in np.argsort(hdiag):
            ix = int(x)
            if ix != int(ref_index):
                order.append(ix)
            if len(order) >= nroots:
                break
    else:
        order = [int(x) for x in np.argsort(hdiag)[:nroots]]

    guesses = []
    for ix in order[:nroots]:
        g = np.zeros(ndet)
        g[ix] = 1.0
        guesses.append(g)
    return guesses


def _prepare_ci0(ci0, pair_space, hdiag, nroots, ref_index=None):
    """Normalize user-provided initial guess to list of 1D vectors."""
    if ci0 is None:
        return _make_init_guess(hdiag, nroots, ref_index=ref_index)

    if isinstance(ci0, (list, tuple)):
        out = []
        for x in ci0:
            arr = np.asarray(x)
            if arr.size != pair_space.ndet:
                raise ValueError(
                    f"ci0 root has size {arr.size}; expected {pair_space.ndet}"
                )
            out.append(arr.reshape(pair_space.ndet))
        return out

    arr = np.asarray(ci0)

    # Compressed hCI vector.
    if arr.size == pair_space.ndet:
        return [arr.reshape(pair_space.ndet)]

    # Rectangular selected_ci vector over the internal alpha/beta strings.
    if arr.size == pair_space.na * pair_space.nb:
        rect = arr.reshape(pair_space.na, pair_space.nb)
        return [pair_space.pack(rect)]

    raise ValueError(
        f"Cannot interpret ci0 with size {arr.size}; expected either "
        f"{pair_space.ndet} compressed elements or "
        f"{pair_space.na * pair_space.nb} rectangular elements."
    )


def kernel_fixed_pair_space(myci, h1e, eri, norb, nelec, pairs, ci0=None,
                            tol=None, lindep=None, max_cycle=None,
                            max_space=None, nroots=None,
                            max_memory=None, verbose=None, ecore=0,
                            hci_info=None, ref_strs=None, **kwargs):
    r"""
    Diagonalize the CI Hamiltonian in an arbitrary determinant-pair space.

    The selected-pair Hamiltonian action is implemented as

        c_pair -> c_rect -> H c_rect -> sigma_pair

    which is mathematically P H P in the selected determinant-pair space.

    Parameters
    ----------
    myci : EHCISolver
        Solver object.
    h1e : ndarray
        One-electron integrals in MO basis.
    eri : ndarray
        Two-electron integrals in MO basis, PySCF ao2mo format accepted.
    norb : int
        Number of spatial orbitals.
    nelec : int or tuple
        Electron count.
    pairs : ndarray, shape (ndet, 2)
        Selected determinant pairs.
    ci0 : ndarray or list or None
        Optional initial guess in compressed or rectangular representation.
    ecore : float
        Core/nuclear energy shift added to returned eigenvalue.

    Returns
    -------
    e, c : float/ndarray, EHCIVector/list[EHCIVector]
        Energy and compressed hCI vector(s).
    """
    log = logger.new_logger(myci, verbose)

    if tol is None:
        tol = myci.conv_tol
    if lindep is None:
        lindep = myci.lindep
    if max_cycle is None:
        max_cycle = myci.max_cycle
    if max_space is None:
        max_space = myci.max_space
    if max_memory is None:
        max_memory = myci.max_memory
    if nroots is None:
        nroots = myci.nroots

    if myci.verbose >= logger.WARN:
        myci.check_sanity()

    nelec = direct_spin1._unpack_nelec(nelec, myci.spin)
    pair_space = PairSpace(pairs)
    ci_strs = pair_space.ci_strs

    # Save on solver for later RDM, spin, large_ci, etc.
    myci.pair_space = pair_space
    myci._strs = ci_strs
    myci.pairs = pair_space.pairs

    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    h2e = ao2mo.restore(1, h2e, norb)

    link_index = selected_ci._all_linkstr_index(ci_strs, norb, nelec)

    hdiag_rect = selected_ci.make_hdiag(h1e, eri, ci_strs, norb, nelec)
    hdiag_rect = np.asarray(hdiag_rect).reshape(pair_space.na, pair_space.nb)
    hdiag = pair_space.pack(hdiag_rect)

    ref_index = None
    if ref_strs is not None:
        ref_index = pair_space.pair_index(ref_strs[0], ref_strs[1])

    ci0 = _prepare_ci0(ci0, pair_space, hdiag, nroots, ref_index=ref_index)

    log.info("hCI compressed determinant space: ndet = %d", pair_space.ndet)
    log.info("hCI internal rectangular space: nalpha = %d, nbeta = %d, size = %d",
             pair_space.na, pair_space.nb, pair_space.na * pair_space.nb)
    log.info("hCI sparsity fraction inside rectangular selected_ci space = %.6f",
             pair_space.ndet / float(pair_space.na * pair_space.nb))

    cpu0 = [logger.process_clock(), logger.perf_counter()]

    def hop(c):
        c = np.asarray(c).reshape(pair_space.ndet)
        rect = pair_space.unpack(c)
        sci_vec = selected_ci._as_SCIvector(rect, ci_strs)
        hc_rect = selected_ci.contract_2e(h2e, sci_vec, norb, nelec, link_index)
        cpu0[:] = log.timer_debug1("contract_2e", *cpu0)
        return pair_space.pack(hc_rect).reshape(-1)

    level_shift = getattr(myci, "level_shift", 1e-4)

    def precond(x, e, *args):
        return x / (hdiag - e + level_shift)

    e, c = myci.eig(hop, ci0, precond,
                    tol=tol, lindep=lindep, max_cycle=max_cycle,
                    max_space=max_space, nroots=nroots,
                    max_memory=max_memory, verbose=log, **kwargs)

    if nroots > 1:
        out = [_as_ehci_vector(np.asarray(ci).reshape(pair_space.ndet),
                               pair_space, hci_info=hci_info)
               for ci in c]
        return e + ecore, out
    else:
        out = _as_ehci_vector(np.asarray(c).reshape(pair_space.ndet),
                              pair_space, hci_info=hci_info)
        return e + ecore, out


# -----------------------------------------------------------------------------
# Solver class
# -----------------------------------------------------------------------------

class EHCISolver(selected_ci.SelectedCI):
    r"""
    Hierarchy CI solver.

    Main user-facing option:

        h

    A determinant is included when

        0.5 * excitation_degree + 0.25 * seniority <= h

    Examples
    --------
    >>> cisolver = EHCISolver(mol)
    >>> cisolver.h = 2.0
    >>> e, c = cisolver.kernel(h1, eri, norb, nelec)
    """

    _keys = selected_ci.SelectedCI._keys.union({
        "h", "ref_strs", "pairs", "pair_space",
    })

    def __init__(self, mol=None, h=1.0, ref_strs=None):
        selected_ci.SelectedCI.__init__(self, mol)
        self.h = h
        self.ref_strs = ref_strs
        self.pairs = None
        self.pair_space = None

    def dump_flags(self, verbose=None):
        selected_ci.SelectedCI.dump_flags(self, verbose)
        logger.info(self, "hCI hierarchy parameter h = %s", self.h)
        logger.info(self, "hCI rule: 0.5*e + 0.25*s <= h")
        if self.ref_strs is None:
            logger.info(self, "hCI reference: default Aufbau/HF determinant")
        else:
            logger.info(self, "hCI reference strings: alpha=%s beta=%s",
                        bin(int(self.ref_strs[0])), bin(int(self.ref_strs[1])))

    def gen_pairs(self, norb, nelec, h=None, ref_strs=None, return_info=False):
        """Generate hCI determinant pairs for this solver."""
        if h is None:
            h = self.h
        if ref_strs is None:
            ref_strs = self.ref_strs
        return gen_ehci_pairs(norb, nelec, h, ref_strs=ref_strs,
                              spin=self.spin, return_info=return_info)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, h=None, ref_strs=None,
               pairs=None, **kwargs):
        """
        Run hCI calculation.

        Parameters
        ----------
        h : float or None
            Hierarchy parameter. If None, use self.h.
        ref_strs : tuple[int, int] or None
            Reference determinant. If None, use self.ref_strs; if that is None,
            use default Aufbau/HF reference.
        pairs : ndarray or None
            Optional precomputed determinant pairs. If provided, h is not used
            to generate the space.
        """
        if h is None:
            h = self.h
        else:
            self.h = h

        if ref_strs is None:
            ref_strs = self.ref_strs
        else:
            self.ref_strs = ref_strs

        ref = make_reference_strings(norb, nelec, ref_strs, spin=self.spin)

        hci_info = None
        if pairs is None:
            pairs, hci_info = gen_ehci_pairs(norb, nelec, h, ref_strs=ref,
                                             spin=self.spin, return_info=True)
        else:
            pairs = np.asarray(pairs, dtype=np.int64)

        self.pairs = pairs

        return kernel_fixed_pair_space(self, h1e, eri, norb, nelec, pairs,
                                       ci0=ci0, hci_info=hci_info,
                                       ref_strs=ref, **kwargs)

    def contract_2e(self, eri, civec, norb, nelec, link_index=None, **kwargs):
        """
        Contract H with a compressed hCI vector.

        This is mostly for compatibility/testing. The main kernel defines its
        own hop function for speed and clarity.
        """
        pair_space = getattr(civec, "_pair_space", None)
        if pair_space is None:
            pair_space = self.pair_space
        if pair_space is None:
            return selected_ci.SelectedCI.contract_2e(
                self, eri, civec, norb, nelec, link_index=link_index, **kwargs
            )

        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        if link_index is None:
            link_index = selected_ci._all_linkstr_index(pair_space.ci_strs, norb, nelec)
        rect = _compressed_to_rect_sci(civec, pair_space)
        sig_rect = selected_ci.contract_2e(eri, rect, norb, nelec, link_index)
        sig = pair_space.pack(sig_rect)
        return _as_ehci_vector(sig, pair_space)

    def large_ci(self, civec, norb, nelec, tol=0.1, return_strs=True):
        """
        List large hCI coefficients in compressed determinant-pair basis.
        """
        pair_space = getattr(civec, "_pair_space", None)
        if pair_space is None:
            pair_space = self.pair_space
        if pair_space is None:
            raise ValueError("No pair_space available")

        c = np.asarray(civec).reshape(pair_space.ndet)
        idx = np.where(np.abs(c) > tol)[0]
        out = []
        for k in idx:
            coeff = c[k]
            a, b = pair_space.pairs[k]
            if return_strs:
                out.append((coeff, bin(int(a)), bin(int(b))))
            else:
                occa = cistring._strs2occslst(np.asarray([a], dtype=np.int64), norb)[0]
                occb = cistring._strs2occslst(np.asarray([b], dtype=np.int64), norb)[0]
                out.append((coeff, occa, occb))
        return out

    def spin_square(self, civec, norb, nelec):
        """Compute <S^2> using selected_ci after rectangular expansion."""
        pair_space = getattr(civec, "_pair_space", None)
        if pair_space is None:
            pair_space = self.pair_space
        rect = _compressed_to_rect_sci(civec, pair_space)
        return selected_ci.spin_square(rect, norb,
                                      direct_spin1._unpack_nelec(nelec, self.spin))

    def make_rdm1s(self, civec, norb, nelec, link_index=None):
        pair_space = getattr(civec, "_pair_space", None)
        if pair_space is None:
            pair_space = self.pair_space
        rect = _compressed_to_rect_sci(civec, pair_space)
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        return selected_ci.make_rdm1s(rect, norb, nelec, link_index=link_index)

    def make_rdm1(self, civec, norb, nelec, link_index=None):
        dm1a, dm1b = self.make_rdm1s(civec, norb, nelec, link_index=link_index)
        return dm1a + dm1b

    def make_rdm2s(self, civec, norb, nelec, link_index=None, **kwargs):
        pair_space = getattr(civec, "_pair_space", None)
        if pair_space is None:
            pair_space = self.pair_space
        rect = _compressed_to_rect_sci(civec, pair_space)
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        return selected_ci.make_rdm2s(rect, norb, nelec, link_index=link_index, **kwargs)

    def make_rdm2(self, civec, norb, nelec, link_index=None, **kwargs):
        dm2aa, dm2ab, dm2bb = self.make_rdm2s(civec, norb, nelec,
                                             link_index=link_index, **kwargs)
        dm2 = dm2aa.copy()
        dm2 += dm2bb
        dm2 += dm2ab
        dm2 += dm2ab.transpose(2, 3, 0, 1)
        return dm2

    def make_rdm12s(self, civec, norb, nelec, link_index=None, **kwargs):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        neleca, nelecb = nelec
        dm2aa, dm2ab, dm2bb = self.make_rdm2s(civec, norb, nelec,
                                             link_index=link_index, **kwargs)
        if neleca > 1 and nelecb > 1:
            dm1a = np.einsum("iikl->kl", dm2aa) / (neleca - 1)
            dm1b = np.einsum("iikl->kl", dm2bb) / (nelecb - 1)
        else:
            dm1a, dm1b = self.make_rdm1s(civec, norb, nelec, link_index=link_index)
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

    def make_rdm12(self, civec, norb, nelec, link_index=None, **kwargs):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        nelec_tot = sum(nelec)
        dm2 = self.make_rdm2(civec, norb, nelec,
                             link_index=link_index, **kwargs)
        if nelec_tot > 1:
            dm1 = np.einsum("iikl->kl", dm2) / (nelec_tot - 1)
        else:
            dm1 = self.make_rdm1(civec, norb, nelec, link_index=link_index)
        return dm1, dm2

    def to_fci(self, civec, norb, nelec):
        return to_fci(civec, norb, nelec, spin=self.spin)

    def to_selected_ci(self, civec):
        pair_space = getattr(civec, "_pair_space", None)
        if pair_space is None:
            pair_space = self.pair_space
        return to_selected_ci(civec, pair_space=pair_space)

    def print_space(self, norb, nelec):
        """Print allowed (e, s) sectors for this solver's h."""
        print_allowed_sectors(norb, nelec, self.h, ref_strs=self.ref_strs,
                              spin=self.spin)


# Common aliases
EHCI = EHCISolver
HCI = EHCISolver
SCI = EHCISolver


# -----------------------------------------------------------------------------
# Convenience functional interface
# -----------------------------------------------------------------------------

def kernel(h1e, eri, norb, nelec, h=1.0, ci0=None, ref_strs=None,
           level_shift=1e-3, tol=1e-10, lindep=1e-14,
           max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, ecore=0,
           verbose=None, **kwargs):
    """
    Functional interface, similar to pyscf.fci.selected_ci.kernel.

    Example
    -------
    >>> e, c = ehci.kernel(h1, eri, norb, nelec, h=2.0)
    """
    myci = EHCISolver(h=h, ref_strs=ref_strs)
    myci.level_shift = level_shift
    myci.conv_tol = tol
    myci.lindep = lindep
    myci.max_cycle = max_cycle
    myci.max_space = max_space
    myci.nroots = nroots
    myci.davidson_only = davidson_only
    myci.pspace_size = pspace_size
    if verbose is not None:
        myci.verbose = verbose

    return myci.kernel(h1e, eri, norb, nelec, ci0=ci0, ecore=ecore, **kwargs)


if __name__ == "__main__":
    # Small self-test: H2/STO-3G.
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()

    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)

    for hval in [0.0, 1.0, 2.0]:
        solver = EHCISolver(mol, h=hval)
        e_elec, c = solver.kernel(h1, eri, norb, nelec)
        print(f"hCI({hval}) total E = {e_elec + mol.energy_nuc(): .12f}, "
              f"ndet = {solver.pair_space.ndet}")
