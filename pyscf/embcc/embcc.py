import logging
import os.path
import functools

import numpy as np
import scipy
import scipy.linalg
from mpi4py import MPI

import pyscf
import pyscf.lo
import pyscf.cc
import pyscf.ci
import pyscf.mcscf
import pyscf.fci

import pyscf.pbc
import pyscf.pbc.cc

from .orbitals import Orbitals

__all__ = [
        "EmbCC",
        ]

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

# optimize is False by default, despite NumPy's documentation, as of version 1.17
einsum = functools.partial(np.einsum, optimize=True)

def reorder_columns(a, *args):
    """Reorder columns of matrix a. The new order must be specified by a list of tuples,
    where each tuple represents a block of columns, with the first tuple index being the
    first column index and the second tuple index the number of columns in the respective
    block.
    """
    starts, sizes = zip(*args)
    n = len(starts)

    #slices = []
    #for i in range(len(starts)):
    #    start = starts[i]
    #    size = sizes[i]
    #    if start is None and size is None:
    #        s = np.s_[:]
    #    elif start is None:
    #        s = np.s_[:size]
    #    elif size is None:
    #        s = np.s_[start:]
    #    else:
    #        s = np.s_[start:start+size]
    #    slices.append(s)

    #slices
    starts = [s if s is not None else 0 for s in starts]
    ends = [starts[i]+sizes[i] if sizes[i] is not None else None for i in range(n)]
    slices = [np.s_[starts[i]:ends[i]] for i in range(n)]

    b = np.hstack([a[:,s] for s in slices])
    assert b.shape == a.shape
    return b


class Cluster:

    def __init__(self, base, name, indices, solver="CCSD", bath_type=None, tol_bath=1e-3, tol_dmet_bath=1e-8,
            **kwargs):
        """
        Parameters
        ----------
        name :
            Name of cluster.
        indices:
            Atomic orbital indices of cluster.
        """

        self.base = base
        self.name = name
        self.indices = indices
        # Optional
        self.solver = solver
        self.bath_type = bath_type
        self.tol_bath = tol_bath
        self.tol_dmet_bath = tol_dmet_bath

        self.use_ref_orbitals_dmet = kwargs.get("use_ref_orbitals_dmet", True)
        self.use_ref_orbitals_bath = kwargs.get("use_ref_orbitals_bath", True)

        self.set_default_attributes()

    def reset(self, keep_ref_orbitals=True):
        """Reset cluster object. By default it stores the previous orbitals, so they can be used
        as reference orbitals for a new calculation of different geometry."""
        ref_orbitals = self.orbitals
        self.set_default_attributes()
        if keep_ref_orbitals:
            self.ref_orbitals = ref_orbitals
        log.debug("Resetting cluster %s. New vars:\n%s", self.name, vars(self))

    def set_default_attributes(self):
        """Set default attributes of cluster object."""
        # Orbital objects
        self.orbitals = None
        self.ref_orbitals = None
        # Orbitals sizes
        self.nbath0 = 0
        self.nbath = 0
        self.nfrozen = 0
        # Calculation results
        self.converged = True
        self.e_corr = 0.0
        self.e_corr_full = 0.0
        self.e_corr_alt = 0.0

    def __len__(self):
        """The number of local ("imurity") orbitals of the cluster."""
        return len(self.indices)

    def loop_clusters(self, exclude_self=False):
        """Loop over all clusters."""
        for cluster in self.base.clusters:
            if (exclude_self and cluster == self):
                continue
            yield cluster

    @property
    def mf(self):
        """The underlying mean-field object is taken from self.base.
        This is used throughout the construction of orbital spaces and as the reference for
        the correlated solver.

        Accessed attributes and methods are:
        mf.get_ovlp()
        mf.get_hcore()
        mf.get_fock()
        mf.make_rdm1()
        mf.mo_energy
        mf.mo_coeff
        mf.mo_occ
        mf.e_tot
        """
        return self.base.mf

    @property
    def mol(self):
        """The molecule or cell object is taken from self.base.mol.
        It should be the same as self.base.mf.mol by default."""
        return self.base.mol

    @property
    def not_indices(self):
        """Indices which are NOT in the cluster, i.e. complement to self.indices."""
        return np.asarray([i for i in np.arange(self.mol.nao_nr()) if i not in self.indices])

    def make_projector(self):
        """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
        S1 = self.mf.get_ovlp()
        nao = self.mol.nao_nr()
        S2 = S1[np.ix_(self.indices, self.indices)]
        S21 = S1[self.indices]
        #s2_inv = np.linalg.inv(s2)
        #p_21 = np.dot(s2_inv, s21)
        # Better: solve with Cholesky decomposition
        # Solve: S2 * p_21 = S21 for p_21
        p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
        p_12 = np.eye(nao)[:,self.indices]
        p = np.dot(p_12, p_21)
        return p

    def make_projector_s121(self):
        """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
        S1 = self.mf.get_ovlp()
        nao = self.mol.nao_nr()
        S2 = S1[np.ix_(self.indices, self.indices)]
        S21 = S1[self.indices]
        #s2_inv = np.linalg.inv(s2)
        #p_21 = np.dot(s2_inv, s21)
        # Better: solve with Cholesky decomposition
        # Solve: S2 * p_21 = S21 for p_21
        p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
        #p_12 = np.eye(nao)[:,self.indices]
        p = np.dot(S21.T, p_21)
        return p

    def make_local_orbitals(self):
        """Make local orbitals by orthonormalizing local AOs."""
        S = self.mf.get_ovlp()
        norb = S.shape[-1]
        S121 = self.make_projector_s121()
        assert np.allclose(S121, S121.T)
        e, C = scipy.linalg.eigh(S121, b=S)
        rev = np.s_[::-1]
        e = e[rev]
        C = C[:,rev]
        nloc = len(e[e>1e-9])
        assert nloc == len(self), "Error finding local orbitals: %s" % e
        assert np.allclose(np.linalg.multi_dot((C.T, S, C)), np.eye(C.shape[-1]))

        return C

    def project_ref_orbitals(self, C, C_ref, space):
        """Project reference orbitals into available space in new gemetry.

        The projected orbitals will be ordered according to their eigenvalues within the space.

        Parameters
        ----------
        C : ndarray
            Orbital coefficients.
        C_ref : ndarray
            Orbital coefficients of reference orbitals.
        space : slice
            Space of current calculation to use for projection.
        """
        assert (C_ref.shape[-1] > 0)
        C = C.copy()
        S = self.mf.get_ovlp()
        # Diagonalize reference orbitals among themselves (due to change in overlap matrix)
        C_ref = pyscf.lo.vec_lowdin(C_ref, S)
        # Diagonalize projector in space
        CSC = np.linalg.multi_dot((C_ref.T, S, C[:,space]))
        P = np.dot(CSC.T, CSC)
        e, r = np.linalg.eigh(P)
        rev = np.s_[::-1]
        e = e[rev]
        r = r[:,rev]
        C[:,space] = np.dot(C[:,space], r)
        return C, e

    def make_dmet_bath_orbitals(self, C, tol=None):
        """If C_ref is specified, complete DMET orbital space using active projection of reference orbitals."""
        if tol is None:
            tol = self.tol_dmet_bath
        C = C.copy()
        env = np.s_[len(self):]
        S = self.mf.get_ovlp()
        D = np.linalg.multi_dot((C[:,env].T, S, self.mf.make_rdm1(), S, C[:,env])) / 2
        e, v = np.linalg.eigh(D)
        reverse = np.s_[::-1]
        e = e[reverse]
        v = v[:,reverse]
        mask_bath = np.fmin(abs(e), abs(e-1)) >= tol

        sort = np.argsort(np.invert(mask_bath), kind="mergesort")
        e = e[sort]
        v = v[:,sort]

        nbath0 = sum(mask_bath)
        nenvocc = sum(e[nbath0:] > 0.5)

        log.debug("Found %d DMET bath orbitals. Eigenvalues:\n%s\nFollowing eigenvalues:\n%s", nbath0, e[:nbath0], e[nbath0:nbath0+3])
        assert nbath0 <= len(self)

        C[:,env] = np.dot(C[:,env], v)

        # Complete DMET orbital space using reference
        if self.use_ref_orbitals_dmet and self.ref_orbitals is not None:
            C_ref = self.ref_orbitals.get_coeff("dmet-bath")
            nref = C_ref.shape[-1]
            nmissing = nref - nbath0
            if nmissing == 0:
                log.debug("Found %d DMET bath orbitals, reference: %d.", nbath0, nref)
            elif nmissing > 0:
                reftol = 0.8
                # --- Occupied
                ncl = len(self) + nbath0
                C, eig = self.project_ref_orbitals(C, C_ref, space=np.s_[ncl:ncl+nenvocc])
                naddocc = sum(eig >= reftol)
                log.debug("Eigenvalues of projected occupied reference: %s, following: %s", eig[:naddocc], eig[naddocc:naddocc+3])
                nbath0 += naddocc
                # --- Virtual
                ncl = len(self) + nbath0
                nenvocc -= naddocc
                # Diagonalize projector in remaining virtual space
                C, eig = self.project_ref_orbitals(C, C_ref, space=np.s_[ncl+nenvocc:])
                naddvir = sum(eig >= reftol)
                log.debug("Eigenvalues of projected virtual reference: %s, following: %s", eig[:naddvir], eig[naddvir:naddvir+3])
                # Reorder of virtual necessary
                offset = len(self)+nbath0
                C = reorder_columns(C,
                        (None, offset),
                        (offset+nenvocc, naddvir),
                        (offset, nenvocc),
                        (offset+nenvocc+naddvir, None),)
                nbath0 += naddvir
                if nbath0 != nref:
                    log.critical("Number of DMET bath orbitals=%d not equal to reference=%d", nbath0, nref)
            else:
                log.critical("More DMET bath orbitals found than in reference=%d", nref)


        return C, nbath0, nenvocc

    def make_power_bath_orbitals(self, C, kind, non_local, power=1, tol=None, normalize=False):
        if tol is None:
            tol = self.tol_bath

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        loc = np.s_[:len(self)]

        b = np.einsum("xi,i,ai->xa", csc[non_local], e**power, csc[loc], optimize=True)

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        assert np.all(e > -1e-13)
        rev = np.s_[::-1]
        e = e[rev]
        v = v[:,rev]

        nbath = sum(e >= tol)
        #log.debug("Eigenvalues of kind=%s, power=%d bath:\n%r", kind, power, e)
        log.debug("Eigenvalues of kind=%s, power=%d bath, tolerance=%e", kind, power, tol)
        log.debug("%d eigenvalues above threshold:\n%r", nbath, e[:nbath])
        log.debug("%d eigenvalues below threshold:\n%r", len(e)-nbath, e[nbath:])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def make_uncontracted_dmet_orbitals(self, C, kind, non_local, tol=None, normalize=False):
    #def make_power_bath_orbitals(self, C, kind, non_local, power=1, tol=None, normalize=True):
        if tol is None:
            tol = self.tol_bath

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        loc = np.s_[:len(self)]

        b = np.einsum("xi,ai->xia", csc[non_local], csc[loc], optimize=True)
        b = b.reshape(b.shape[0], b.shape[1]*b.shape[2])

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        assert np.all(e > -1e-13)
        rev = np.s_[::-1]
        e = e[rev]
        v = v[:,rev]

        nbath = sum(e >= tol)
        #log.debug("Eigenvalues of kind=%s, power=%d bath:\n%r", kind, power, e)
        log.debug("Eigenvalues of uncontracted DMET bath of kind=%s, tolerance=%e", kind, tol)
        log.debug("%d eigenvalues above threshold:\n%r", nbath, e[:nbath])
        log.debug("%d eigenvalues below threshold:\n%r", len(e)-nbath, e[nbath:])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def make_matsubara_bath_orbitals(self, C, kind, non_local, npoints=1000, beta=100.0, tol=None, normalize=False):

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        loc = np.s_[:len(self)]

        # Matsubara points
        wn = (2*np.arange(npoints)+1)*np.pi/beta
        kernel = wn[np.newaxis,:] / np.add.outer(self.mf.mo_energy[mask]**2, wn**2)

        b = np.einsum("xi,iw,ai->xaw", csc[non_local], kernel, csc[loc], optimize=True)
        b = b.reshape(b.shape[0], b.shape[1]*b.shape[2])

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        assert np.all(e > -1e-13)
        rev = np.s_[::-1]
        e = e[rev]
        v = v[:,rev]

        nbath = sum(e >= tol)
        #log.debug("Eigenvalues of kind=%s, power=%d bath:\n%r", kind, power, e)
        log.debug("Eigenvalues of kind=%s Matsubara bath, tolerance=%e", kind, tol)
        log.debug("%d eigenvalues above threshold:\n%r", nbath, e[:nbath])
        log.debug("%d eigenvalues below threshold:\n%r", len(e)-nbath, e[nbath:])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def run_solver(self, solver=None, max_power=0, pertT=False, diagonalize_fock=True, cc_verbose=4,
            ref_orbitals=None):

        solver = solver or self.solver

        if solver is None:
            self.e_corr = 0.0
            self.e_corr_alt = 0.0
            return 1

        ref_orbitals = ref_orbitals or self.ref_orbitals

        C = self.make_local_orbitals()
        C, nbath0, nenvocc = self.make_dmet_bath_orbitals(C)
        nbath = nbath0

        ncl = len(self)+nbath0
        orbitals = Orbitals(C)
        orbitals.define_space("local", np.s_[:len(self)])
        orbitals.define_space("dmet-bath", np.s_[len(self):ncl])
        orbitals.define_space("occ-env", np.s_[ncl:ncl+nenvocc])
        orbitals.define_space("vir-env", np.s_[ncl+nenvocc:])

        # Use previous orbitals
        if ref_orbitals and self.use_ref_orbitals_bath:
            # Occupied
            nbathocc = ref_orbitals.get_size("occ-bath")
            if nbathocc == 0:
                log.debug("No reference occupied bath orbitals.")
            else:
                C, eig = self.project_ref_orbitals(C, ref_orbitals.get_coeff("occ-bath"),
                        orbitals.get_indices("occ-env"))
                log.debug("Eigenvalues of %d projected occupied bath orbitals:\n%s",
                        nbathocc, eig[:nbathocc])
                log.debug("Next 3 eigenvalues: %s", eig[nbathocc:nbathocc+3])
            # Virtual
            nbathvir = ref_orbitals.get_size("vir-bath")
            if nbathvir == 0:
                log.debug("No reference virtual bath orbitals.")
            else:
                C, eig = self.project_ref_orbitals(C, ref_orbitals.get_coeff("vir-bath"),
                        orbitals.get_indices("vir-env"))
                log.debug("Eigenvalues of %d projected virtual bath orbitals:\n%s",
                        nbathvir, eig[:nbathvir])
                log.debug("Next 3 eigenvalues: %s", eig[nbathvir:nbathvir+3])

        # Add additional power bath orbitals
        else:
            nbathocc = 0
            nbathvir = 0
            # Power orbitals
            for power in range(1, max_power+1):
                occ_space = np.s_[len(self)+nbath0+nbathocc:len(self)+nbath0+nenvocc]
                C, nbo = self.make_power_bath_orbitals(C, "occ", occ_space, power=power)
                vir_space = np.s_[len(self)+nbath0+nenvocc+nbathvir:]
                C, nbv = self.make_power_bath_orbitals(C, "vir", vir_space, power=power)
                nbathocc += nbo
                nbathvir += nbv
            # Uncontracted DMET
            if self.bath_type == "uncontracted":
                occ_space = np.s_[len(self)+nbath0+nbathocc:len(self)+nbath0+nenvocc]
                C, nbo = self.make_uncontracted_dmet_orbitals(C, "occ", occ_space, tol=self.tol_bath)
                vir_space = np.s_[len(self)+nbath0+nenvocc+nbathvir:]
                C, nbv = self.make_uncontracted_dmet_orbitals(C, "vir", vir_space, tol=self.tol_bath)
                nbathocc += nbo
                nbathvir += nbv
            # Matsubara
            elif self.bath_type == "matsubara":
                occ_space = np.s_[len(self)+nbath0+nbathocc:len(self)+nbath0+nenvocc]
                C, nbo = self.make_matsubara_bath_orbitals(C, "occ", occ_space, tol=self.tol_bath)
                vir_space = np.s_[len(self)+nbath0+nenvocc+nbathvir:]
                C, nbv = self.make_matsubara_bath_orbitals(C, "vir", vir_space, tol=self.tol_bath)
                nbathocc += nbo
                nbathvir += nbv

        # The virtuals require reordering:
        ncl = len(self)+nbath0
        nvir0 = ncl+nenvocc                 # Start index for virtuals
        C = np.hstack((
            C[:,:ncl+nbathocc],               # impurity + DMET bath + occupied bath
            C[:,nvir0:nvir0+nbathvir],        # virtual bath
            C[:,ncl+nbathocc:nvir0],          # occupied frozen
            C[:,nvir0+nbathvir:],             # virtual frozen
            ))
        nbath += nbathocc
        nbath += nbathvir

        # At this point save reference orbitals for other calculations
        orbitals.C = C
        orbitals.define_space("occ-bath", np.s_[ncl:ncl+nbathocc])
        orbitals.define_space("vir-bath", np.s_[ncl+nbathocc:ncl+nbathocc+nbathvir])
        orbitals.delete_space("occ-env")
        orbitals.define_space("occ-env", np.s_[ncl+nbathocc+nbathvir:ncl+nbathocc+nbathvir+nenvocc])
        orbitals.delete_space("vir-env")
        orbitals.define_space("vir-env", np.s_[ncl+nbathocc+nbathvir+nenvocc:])
        self.orbitals = orbitals

        # Diagonalize cluster DM (Necessary for CCSD)
        S = self.mf.get_ovlp()
        SDS_hf = np.linalg.multi_dot((S, self.mf.make_rdm1(), S))
        ncl = len(self) + nbath0
        cl = np.s_[:ncl]
        D = np.linalg.multi_dot((C[:,cl].T, SDS_hf, C[:,cl])) / 2
        e, v = np.linalg.eigh(D)
        assert np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=1e-6, rtol=0)
        reverse = np.s_[::-1]
        e = e[reverse]
        v = v[:,reverse]
        C_cc = C.copy()
        C_cc[:,cl] = np.dot(C_cc[:,cl], v)
        nocc_cl = sum(e > 0.5)
        log.debug("Occupied/virtual states in local+DMET space: %d/%d", nocc_cl, ncl-nocc_cl)

        # Sort occupancy
        occ = np.einsum("ai,ab,bi->i", C_cc, SDS_hf, C_cc, optimize=True)
        assert np.allclose(np.fmin(abs(occ), abs(occ-2)), 0, atol=1e-6, rtol=0), "Error in occupancy: %s" % occ
        occ = np.asarray([2 if occ > 1 else 0 for occ in occ])
        sort = np.argsort(-occ, kind="mergesort") # mergesort is stable (keeps relative order)
        rank = np.argsort(sort)
        C_cc = C_cc[:,sort]
        occ = occ[sort]
        nocc = sum(occ > 0)
        nactive = len(self) + nbath
        frozen = rank[nactive:]
        active = rank[:nactive]
        nocc_active = sum(occ[active] > 0)

        log.debug("Occupancy of local + DMET bath orbitals:\n%s", occ[rank[:len(self)+nbath0]])
        log.debug("Occupancy of other bath orbitals:\n%s", occ[rank[len(self)+nbath0:len(self)+nbath]])
        log.debug("Occupancy of frozen orbitals:\n%s", occ[frozen])

        self.nbath0 = nbath0
        self.nbath = nbath
        self.nfrozen = len(frozen)

        # Nothing to correlate for a single orbital
        if len(self) == 1 and nbath == 0:
            self.e_corr = 0.0
            self.e_corr_full = 0.0
            self.e_corr_alt = 0.0
            self.converged = True
            return 1

        # Accelerates convergence
        if diagonalize_fock:
            F = np.linalg.multi_dot((C_cc.T, self.mf.get_fock(), C_cc))
            # Occupied active
            o = np.nonzero(occ > 0)[0]
            o = np.asarray([i for i in o if i in active])
            if len(o) > 0:
                e, r = np.linalg.eigh(F[np.ix_(o, o)])
                C_cc[:,o] = np.dot(C_cc[:,o], r)
            # Virtual active
            v = np.nonzero(occ == 0)[0]
            v = np.asarray([i for i in v if i in active])
            if len(v) > 0:
                e, r = np.linalg.eigh(F[np.ix_(v, v)])
                C_cc[:,v] = np.dot(C_cc[:,v], r)

        pbc = hasattr(self.mol, "a")
        if pbc:
            log.debug("\"A matrix\" found. Switching to pbc code.")

        # Do not calculate correlation energy
        if solver == "CCSD":
            if pbc:
                ccsd = pyscf.pbc.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            else:
                ccsd = pyscf.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            ccsd.max_cycle = 100
            ccsd.verbose = cc_verbose
            log.debug("Running CCSD...")
            ccsd.kernel()
            log.debug("CCSD done. converged: %r", ccsd.converged)
            C1 = ccsd.t1
            C2 = ccsd.t2 + einsum('ia,jb->ijab', ccsd.t1, ccsd.t1)

            self.converged = ccsd.converged
            self.e_corr_full = ccsd.e_corr

        elif solver == "CISD":
            cisd = pyscf.ci.CISD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            cisd.max_cycle = 100
            cisd.verbose = cc_verbose
            log.debug("Running CISD...")
            cisd.kernel()
            log.debug("CISD done. converged: %r", cisd.converged)
            C0, C1, C2 = cisd.cisdvec_to_amplitudes(cisd.ci)
            # Intermediate normalization
            renorm = 1/C0
            C1 *= renorm
            C2 *= renorm

            self.converged = cisd.converged
            self.e_corr_full = cisd.e_corr

        elif solver == "FCI":
            casci = pyscf.mcscf.CASCI(self.mol, nactive, 2*nocc_active)
            casci.canonicalization = False
            C_cas = pyscf.mcscf.addons.sort_mo(casci, mo_coeff=C_cc, caslst=active, base=0)
            log.debug("Running FCI...")
            e_tot, e_cas, wf, mo_coeff, mo_energy = casci.kernel(mo_coeff=C_cas)
            log.debug("FCI done. converged: %r", casci.converged)
            assert np.allclose(mo_coeff, C_cas)
            cisdvec = pyscf.ci.cisd.from_fcivec(wf, nactive, 2*nocc_active)
            C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, nactive, nocc_active)
            # Intermediate normalization
            renorm = 1/C0
            C1 *= renorm
            C2 *= renorm

            self.converged = casci.converged
            self.e_corr_full = e_tot - self.mf.e_tot

        else:
            raise ValueError("Unknown solver: %s" % solver)

        log.debug("Calculating local energy...")

        if solver == "CCSD":
            #self.e_ccsd, self.e_pt = self.get_local_energy_old(ccsd, pertT=pertT)
            #self.e_ccsd_v, _ = self.get_local_energy_old(ccsd, projector="vir", pertT=pertT)

            #self.e_ccsd_z = self.get_local_energy_most_indices(ccsd)

            self.e_ccsd = self.get_local_energy(ccsd, C1, C2)
            self.e_ccsd_v = self.get_local_energy(ccsd, C1, C2, "virtual")


            self.e_corr = self.e_ccsd

            # TESTING
            #self.get_local_energy_parts(ccsd, C1, C2)

            # TEMP
            self.e_corr_alt = self.get_local_energy_most_indices(ccsd, C1, C2)


            # CCSD energy of whole cluster
            #self.e_cl_ccsd = ccsd.e_corr

        elif solver == "CISD":
            self.e_cisd = self.get_local_energy(cisd, C1, C2)
            self.e_corr = self.e_cisd
        elif solver == "FCI":
            # Fake CISD
            cisd = pyscf.ci.CISD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            self.e_fci = self.get_local_energy(cisd, C1, C2)
            self.e_corr = self.e_fci

            #self.e_corr_alt = self.get_local_energy_most_indices(cisd, C1, C2)

        log.debug("Calculating local energy done.")

        return int(self.converged)

    def get_local_energy_parts(self, cc, C1, C2):

        a = cc.get_frozen_mask()
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]
        CTS = np.dot(C.T, S)

        # Project one index of T amplitudes
        l= self.indices
        r = self.not_indices
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0

        eris = cc.ao2mo()

        def get_projectors(aos):
            Po = np.dot(CTS[o][:,aos], C[aos][:,o])
            Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
            return Po, Pv

        Lo, Lv = get_projectors(l)
        Ro, Rv = get_projectors(r)

        # Nomenclature:
        # old occupied: i,j
        # old virtual: a,b
        # new occupied: p,q
        # new virtual: s,t
        T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
        T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
        T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
        T1 = T1_ll + (T1_lr + T1_rl)/2

        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * T1)
        if not np.isclose(e1, 0):
            log.warning("Warning: large E1 component: %.8e" % e1)

        #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
        def project_T2(P1, P2, P3, P4):
            T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
            return T2p


        def epart(P1, P2, P3, P4):
            T2_part = project_T2(P1, P2, P3, P4)
            e_part = (2*einsum('ijab,iabj', T2_part, eris.ovvo)
                  - einsum('ijab,jabi', T2_part, eris.ovvo))
            return e_part

        energies = []
        # 4
        energies.append(epart(Lo, Lo, Lv, Lv))
        # 3
        energies.append(2*epart(Lo, Lo, Lv, Rv))
        energies.append(2*epart(Lo, Ro, Lv, Lv))
        assert np.isclose(epart(Lo, Lo, Rv, Lv), epart(Lo, Lo, Lv, Rv))
        assert np.isclose(epart(Ro, Lo, Lv, Lv), epart(Lo, Ro, Lv, Lv))

        energies.append(  epart(Lo, Lo, Rv, Rv))
        energies.append(2*epart(Lo, Ro, Lv, Rv))
        energies.append(2*epart(Lo, Ro, Rv, Lv))
        energies.append(  epart(Ro, Ro, Lv, Lv))

        energies.append(2*epart(Lo, Ro, Rv, Rv))
        energies.append(2*epart(Ro, Ro, Lv, Rv))
        assert np.isclose(epart(Ro, Lo, Rv, Rv), epart(Lo, Ro, Rv, Rv))
        assert np.isclose(epart(Ro, Ro, Rv, Lv), epart(Ro, Ro, Lv, Rv))

        energies.append(  epart(Ro, Ro, Rv, Rv))

        #e4 = e_aaaa
        #e3 = e_aaab + e_aaba + e_abaa + e_baaa
        #e2 = 0.5*(e_aabb + e_abab + e_abba + e_bbaa)

        with open("energy-parts.txt", "a") as f:
            f.write((10*"  %16.8e" + "\n") % tuple(energies))

    def get_local_energy_most_indices_2C(self, cc, C1, C2):

        a = cc.get_frozen_mask()
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]
        CTS = np.dot(C.T, S)

        # Project one index of T amplitudes
        l= self.indices
        r = self.not_indices
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0

        eris = cc.ao2mo()

        def get_projectors(aos):
            Po = np.dot(CTS[o][:,aos], C[aos][:,o])
            Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
            return Po, Pv

        Lo, Lv = get_projectors(l)
        Ro, Rv = get_projectors(r)

        # Nomenclature:
        # old occupied: i,j
        # old virtual: a,b
        # new occupied: p,q
        # new virtual: s,t
        T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
        T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
        T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
        T1 = T1_ll + (T1_lr + T1_rl)/2

        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * T1)
        if not np.isclose(e1, 0):
            log.warning("Warning: large E1 component: %.8e" % e1)

        #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
        def project_T2(P1, P2, P3, P4):
            T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
            return T2p

        f3 = 1.0
        f2 = 0.5
        # 4
        T2 = 1*project_T2(Lo, Lo, Lv, Lv)
        # 3
        T2 += f3*(2*project_T2(Lo, Lo, Lv, Rv)      # factor 2 for LLRL
                + 2*project_T2(Ro, Lo, Lv, Lv))     # factor 2 for RLLL
        # 2
        T2 += f2*(  project_T2(Lo, Lo, Rv, Rv)
                + 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
                + 2*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
                +   project_T2(Ro, Ro, Lv, Lv))

        e2 = (2*einsum('ijab,iabj', T2, eris.ovvo)
               -einsum('ijab,jabi', T2, eris.ovvo))

        e_loc = e1 + e2

        return e_loc

    def get_local_energy_most_indices(self, cc, C1, C2):

        a = cc.get_frozen_mask()
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]
        CTS = np.dot(C.T, S)

        # Project one index of T amplitudes
        l= self.indices
        r = self.not_indices
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0

        eris = cc.ao2mo()

        def get_projectors(aos):
            Po = np.dot(CTS[o][:,aos], C[aos][:,o])
            Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
            return Po, Pv

        Lo, Lv = get_projectors(l)
        Ro, Rv = get_projectors(r)

        # Nomenclature:
        # old occupied: i,j
        # old virtual: a,b
        # new occupied: p,q
        # new virtual: s,t
        T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
        T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
        T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
        T1 = T1_ll + (T1_lr + T1_rl)/2

        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * T1)
        if not np.isclose(e1, 0):
            log.warning("Warning: large E1 component: %.8e" % e1)

        #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
        def project_T2(P1, P2, P3, P4):
            T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
            return T2p

        def project_C2(P1=None, P2=None, P3=None, P4=None):
            pC2 = C2
            if P1 is not None:
                pC2 = einsum("xi,ijab->xjab", P1, pC2)
            if P2 is not None:
                pC2 = einsum("xj,ijab->ixab", P2, pC2)
            if P3 is not None:
                pC2 = einsum("xa,ijab->ijxb", P3, pC2)
            if P4 is not None:
                pC2 = einsum("xb,ijab->ijax", P4, pC2)
            return pC2

        assert np.allclose(project_T2(Lo, Lo, Lv, Rv) + project_T2(Lo, Lo, Lv, Lv), project_C2(Lo, Lo, Lv))

        t0 = MPI.Wtime()
        T2_4 = project_T2(Lo, Lo, Lv, Lv)
        e2_4 = (2*einsum('ijab,iabj', T2_4, eris.ovvo)
                 -einsum('ijab,jabi', T2_4, eris.ovvo))

        T2_3 = (2*project_T2(Lo, Lo, Lv, Rv)
               +2*project_T2(Ro, Lo, Lv, Lv))
        e2_3 = (2*einsum('ijab,iabj', T2_3, eris.ovvo)
                 -einsum('ijab,jabi', T2_3, eris.ovvo))

        e2_1 = 0.0
        e2_211 = 0.0
        e2_22 = 0.0

        # Loop over other fragments
        for x, cx in enumerate(self.base.clusters):
            if cx == self:
                continue
            # These should be democratic between L and X (factor=0.5)
            Xo, Xv = get_projectors(cx.indices)
            T2_2l2x = 0.5*(project_T2(Lo, Lo, Xv, Xv)
                         + project_T2(Lo, Xo, Lv, Xv)
                         + project_T2(Lo, Xo, Xv, Lv)
                         + project_T2(Xo, Lo, Lv, Xv)
                         + project_T2(Xo, Lo, Xv, Lv)
                         + project_T2(Xo, Xo, Lv, Lv))
            e2_22 += 2*einsum('ijab,iabj', T2_2l2x, eris.ovvo)
            e2_22 -=   einsum('ijab,jabi', T2_2l2x, eris.ovvo)

            for y, cy in enumerate(self.base.clusters):
                if (cy == self) or (cy == cx):
                    continue
                # These should contribute to L
                # (we can neglect the interchange x <-> y, since both x and y are unrestricted (except x != y)
                Yo, Yv = get_projectors(cy.indices)
                T2_2lxy = (project_T2(Lo, Lo, Xv, Yv)
                         + project_T2(Lo, Xo, Lv, Yv)
                         + project_T2(Lo, Xo, Yv, Lv)
                         + project_T2(Xo, Lo, Lv, Yv)
                         + project_T2(Xo, Lo, Yv, Lv)
                         + project_T2(Xo, Yo, Lv, Lv))
                e2_211 += 2*einsum('ijab,iabj', T2_2lxy, eris.ovvo)
                e2_211 -=   einsum('ijab,jabi', T2_2lxy, eris.ovvo)

                for z, cz in enumerate(self.base.clusters):
                    if (cz == self) or (cz == cx) or (cz == cy):
                        continue
                    # We can neglect interchange between x, y, z (see above)
                    Zo, Zv = get_projectors(cz.indices)
                    T2_lxyz = 0.25*(project_T2(Lo, Xo, Yv, Zv)
                                  + project_T2(Xo, Lo, Yv, Zv)
                                  + project_T2(Xo, Yo, Lv, Zv)
                                  + project_T2(Xo, Yo, Zv, Lv))
                    e2_1 += 2*einsum('ijab,iabj', T2_lxyz, eris.ovvo)
                    e2_1 -=   einsum('ijab,jabi', T2_lxyz, eris.ovvo)
        time_old = MPI.Wtime() - t0



        e2 = e2_4 + e2_3 + e2_211 + e2_22 + e2_1
        e_loc = e1 + e2

        ### ABCD
        ##T2 = 0.25*(2*project_C2(P1=Lo)
        ##         + 2*project_C2(P3=Lv))
        ### Correct AABC
        ##T2 += 0.75*(project_C2(P1=Lo, P2=Lo)
        ##        + 2*project_C2(P1=Lo, P3=Lv)
        ##        + 2*project_C2(P1=Lo, P4=Lv)
        ##        +   project_C2(P3=Lv, P4=Lv))
        ### Correct AABB
        ##for x, cx in enumerate(self.base.clusters):
        ##    Xo, Xv = get_projectors(cx.indices)
        ##    T2 += -0.5*(project_C2(P1=Lo, P2=Lo, P3=Xv, P4=Xv)
        ##            + 2*project_C2(P1=Lo, P2=Xo, P3=Lv, P4=Xv)
        ##            + 2*project_C2(P1=Lo, P2=Xo, P3=Xv, P4=Lv)
        ##              + project_C2(P1=Xo, P2=Xo, P3=Lv, P4=Lv))
        ### Correct AAAB (already correct?)
        ###T2 += 0.75*(2*project_C2(P1=Lo, P2=Lo, P3=Lv)
        ###           +2*project_C2(P2=Lo, P3=Lo, P4=Lv))
        ### Correct AAAA
        ##T2 += 1.0*project_C2(P1=Lo, P2=Lo, P3=Lv, P4=Lv)


        # In the following:
        # L = Local AO
        # A,B,C = non-local AO, which cannot be equal, i.e. A != B != C
        # X = Variable for arbitrary non-local, i.e. A, B, or C
        # R = All non-local (union of A, B, C, ...)
        # P(...) means all possible permutations of ...
        # Factors of 2 are due to (abcd == badc) symmetry

        t0 = MPI.Wtime()

        # QUADRUPLE L
        # ===========
        T2 = project_C2(P1=Lo, P2=Lo, P3=Lv, P4=Lv)

        # TRIPEL L
        # ========
        T2 += 2*project_C2(P1=Lo, P2=Lo, P3=Lv, P4=Rv)
        T2 += 2*project_C2(P1=Lo, P2=Ro, P3=Lv, P4=Lv)

        # DOUBLE L
        # ========
        # P(LLRR) [This wrongly includes: P(LLAA)]
        T2 +=   project_C2(P1=Lo, P2=Lo, P3=Rv, P4=Rv)
        T2 += 2*project_C2(P1=Lo, P2=Ro, P3=Lv, P4=Rv)
        T2 += 2*project_C2(P1=Lo, P2=Ro, P3=Rv, P4=Lv)
        T2 +=   project_C2(P1=Ro, P2=Ro, P3=Lv, P4=Lv)

        # SINGLE L
        # ========
        # P(LRRR) [This wrongly includes: P(LAAR)]
        T2 += 0.25*2*project_C2(P1=Lo, P2=Ro, P3=Rv, P4=Rv)
        T2 += 0.25*2*project_C2(P1=Ro, P2=Ro, P3=Lv, P4=Rv)

        #for x, cx in enumerate(self.base.clusters):
        #    # Skip X=L
        #    if cx == self:
        #        continue
        #for x in self.loop_clusters(exclude_self=True):
        #    Xo, Xv = get_projectors(x.indices)

        # CORRECTIONS
        # ===========
        for x in self.loop_clusters(exclude_self=True):
            Xo, Xv = get_projectors(x.indices)

            # DOUBLE CORRECTION
            # -----------------
            # Correct for wrong inclusion of P(LLAA)
            # The case P(LLAA) was included with prefactor of 1 instead of 1/2
            # We thus need to only correct by "-1/2"
            T2 -= 0.5*  project_C2(P1=Lo, P2=Lo, P3=Xv, P4=Xv)
            T2 -= 0.5*2*project_C2(P1=Lo, P2=Xo, P3=Lv, P4=Xv)
            T2 -= 0.5*2*project_C2(P1=Lo, P2=Xo, P3=Xv, P4=Lv)
            T2 -= 0.5*  project_C2(P1=Xo, P2=Xo, P3=Lv, P4=Lv)

            # SINGLE CORRECTION
            # -----------------
            # Correct for wrong inclusion of P(LAAR)
            # This corrects the case P(LAAB) but overcorrects P(LAAA)!
            T2 -= 0.25*2*project_C2(Lo, Xo, Xv, Rv)
            T2 -= 0.25*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
            T2 -= 0.25*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection
            T2 -= 0.25*2*project_C2(Xo, Xo, Lv, Rv)
            T2 -= 0.25*2*project_C2(Xo, Ro, Lv, Xv) # overcorrection
            T2 -= 0.25*2*project_C2(Ro, Xo, Lv, Xv) # overcorrection

            # Correct overcorrection
            T2 += 0.25*2*2*project_C2(Lo, Xo, Xv, Xv)
            T2 += 0.25*2*2*project_C2(Xo, Xo, Lv, Xv)

        e2_new = (2*einsum('ijab,iabj', T2, eris.ovvo)
                   -einsum('ijab,jabi', T2, eris.ovvo))
        time_new = MPI.Wtime() - t0

        log.debug("Alt E: %.10e vs %.10e", e2, e2_new)
        log.debug("Times: %.3f s vs %.3f s", time_old, time_new)
        assert np.isclose(e2, e2_new)

        return e_loc

    def get_local_energy(self, cc, C1, C2, project="occupied"):

        a = cc.get_frozen_mask()
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]

        # Project one index of amplitudes
        l = self.indices
        r = self.not_indices
        if project == "occupied":
            P = np.linalg.multi_dot((C[:,o].T, S[:,l], C[l][:,o]))
            #S_121 = self.make_projector_s121()
            #P = np.linalg.multi_dot((C_cc[:,o].T, S_121, C_cc[:,o]))
            C1 = einsum("xi,ia->xa", P, C1)
            C2 = einsum("xi,ijab->xjab", P, C2)
        elif project == "virtual":
            P = np.linalg.multi_dot((C[:,v].T, S[:,l], C[l][:,v]))
            #S_121 = self.make_projector_s121()
            #P = np.linalg.multi_dot((C_cc[:,o].T, S_121, C_cc[:,o]))
            C1 = einsum("xa,ia->ia", P, C1)
            C2 = einsum("xa,ijab->ijxb", P, C2)

        eris = cc.ao2mo()
        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * C1)
        if abs(e1) > 1e-7:
            log.warning("Warning: large E1 component: %.8e" % e1)

        e2 = 2*einsum('ijab,iabj', C2, eris.ovvo)
        e2 -=  einsum('ijab,jabi', C2, eris.ovvo)

        e_loc = e1 + e2

        return e_loc

# ===== #

class EmbCC:

    def __init__(self, mf, solver="CCSD", bath_type=None, tol_bath=1e-3, tol_dmet_bath=1e-8,
            use_ref_orbitals_dmet=True, use_ref_orbitals_bath=True):
        self.mf = mf

        if solver not in (None, "CISD", "CCSD", "FCI"):
            raise ValueError("Unknown solver: %s" % solver)
        if bath_type not in (None, "matsubara", "uncontracted"):
            raise ValueError()
        self.solver = solver
        self.bath_type = bath_type
        self.tol_bath = tol_bath
        self.tol_dmet_bath = tol_dmet_bath
        self.use_ref_orbitals_dmet = use_ref_orbitals_dmet
        self.use_ref_orbitals_bath = use_ref_orbitals_bath

        self.clusters = []

    @property
    def mol(self):
        return self.mf.mol

    def make_cluster(self, name, ao_indices, **kwargs):
        kwargs["solver"] = kwargs.get("solver", self.solver)
        kwargs["bath_type"] = kwargs.get("bath_type", self.bath_type)
        kwargs["tol_bath"] = kwargs.get("tol_bath", self.tol_bath)
        kwargs["tol_dmet_bath"] = kwargs.get("tol_dmet_bath", self.tol_dmet_bath)
        kwargs["use_ref_orbitals_dmet"] = kwargs.get("use_ref_orbitals_dmet", self.use_ref_orbitals_dmet)
        kwargs["use_ref_orbitals_bath"] = kwargs.get("use_ref_orbitals_bath", self.use_ref_orbitals_bath)

        cluster = Cluster(self, name, ao_indices, **kwargs)
        return cluster

    def make_atom_clusters(self, **kwargs):
        """Divide atomic orbitals into clusters according to their base atom."""

        # base atom for each AO
        base_atoms = np.asarray([ao[0] for ao in self.mol.ao_labels(None)])

        self.clear_clusters()
        ncluster = self.mol.natm
        for atomid in range(ncluster):
            ao_indices = np.nonzero(base_atoms == atomid)[0]
            name = self.mol.atom_symbol(atomid)
            c = self.make_cluster(name, ao_indices, **kwargs)
            self.clusters.append(c)
        return self.clusters

    def make_ao_clusters(self, **kwargs):
        """Divide atomic orbitals into clusters."""

        self.clear_clusters()
        for aoid in range(self.mol.nao_nr()):
            name = self.mol.ao_labels()[aoid]
            c = self.make_cluster(name, [aoid], **kwargs)
            self.clusters.append(c)
        return self.clusters

    def make_rest_cluster(self, name="rest", **kwargs):
        """Combine all AOs which are not part of a cluster, into a rest cluster."""

        ao_indices = list(range(self.mol.nao_nr()))
        for c in self.clusters:
            ao_indices = [i for i in ao_indices if i not in c.indices]
        if ao_indices:
            c = self.make_cluster(name, ao_indices, **kwargs)
            self.clusters.append(c)
            return c
        else:
            return None

    def make_custom_cluster(self, ao_symbols, name=None, **kwargs):
        """Make custom clusters in terms of AOs.

        Parameters
        ----------
        ao_symbols : iterable
            List of atomic orbital symbols for cluster.
        """
        if isinstance(ao_symbols, str):
            ao_symbols = [ao_symbols]

        if name is None:
            name = ",".join(ao_symbols)

        ao_indices = []
        for ao_idx, ao_label in enumerate(self.mol.ao_labels()):
            for ao_symbol in ao_symbols:
                if ao_symbol in ao_label:
                    log.debug("AO symbol %s found in %s", ao_symbol, ao_label)
                    ao_indices.append(ao_idx)
                    break
        c = self.make_cluster(name, ao_indices, **kwargs)
        self.clusters.append(c)
        return c

    def make_custom_atom_cluster(self, atoms, name=None):
        """Make custom clusters in terms of atoms..

        Parameters
        ----------
        atoms : iterable
            List of atom symbols for cluster.
        """

        if name is None:
            name = ",".join(atoms)
        # base atom for each AO
        ao2atomlbl = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
        ao_indices = np.nonzero(np.isin(ao2atomlbl, atoms))[0]
        c = self.make_cluster(name, ao_indices)
        self.clusters.append(c)
        return c

    def merge_clusters(self, clusters, name=None, **kwargs):
        """Attributes solver, bath_type, tol_bath, and tol_dmet_bath will be taken from first cluster,
        unless specified in **kwargs.
        name will be auto generated, unless specified.

        Parameters
        ----------
        clusters : iterable
            List of clusters to merge.
        """
        clusters_out = []
        merged = []
        for c in self.clusters:
            if c.name.strip() in clusters:
                merged.append(c)
            else:
                clusters_out.append(c)

        if len(merged) < 2:
            raise ValueError("Not enough clusters (%d) found to merge." % len(merged))

        if name is None:
            name = "+".join([c.name for c in merged])
        ao_indices = np.hstack([c.indices for c in merged])
        kwargs["solver"] = kwargs.get("solver", merged[0].solver)
        kwargs["bath_type"] = kwargs.get("bath_type", merged[0].bath_type)
        kwargs["tol_bath"] = kwargs.get("tol_bath", merged[0].tol_bath)
        kwargs["tol_dmet_bath"] = kwargs.get("tol_dmet_bath", merged[0].tol_dmet_bath)
        #merged_cluster = Cluster(merged_name, self.mf, merged_indices,
        #        tol_dmet_bath=tol_dmet_bath, tol_bath=tol_bath)
        c = self.make_cluster(name, ao_indices, **kwargs)
        clusters_out.append(c)
        self.clusters = clusters_out
        return c

    def check_no_overlap(self, clusters=None):
        "Check that no clusters are overlapping."
        if clusters is None:
            clusters = self.clusters
        for c in clusters:
            for c2 in clusters:
                if c == c2:
                    continue
                if np.any(np.isin(c.indices, c2.indices)):
                    log.error("Cluster %s and cluster %s are overlapping.", c.name, c2.name)
                    return False
        return True

    def clear_clusters(self):
        """Clear all previously defined clusters."""
        self.clusters = []

    def print_clusters(self, clusters=None, file=None):
        """Print clusters to logging or file."""
        if clusters is None:
            clusters = self.clusters

        ao_labels = self.mol.ao_labels(None)

        end = "\n" if file else ""
        headfmt = "Cluster %3d: %s with %3d local orbitals:" + end
        linefmt = "%4d %5s %3s %10s" + end

        if file is None:
            for cidx, c in enumerate(clusters):
                log.info(headfmt, cidx, c.name, len(c))
                for ao in c.indices:
                    log.info(linefmt, *ao_labels[ao])
        else:
            with open(file, "w") as f:
                for cidx, c in enumerate(clusters):
                    f.write(headfmt % (cidx, c.name, len(c)))
                    for ao in c.indices:
                        f.write(linefmt % ao_labels[ao])

    def get_cluster(self, name):
        for c in self.clusters:
            if c.name == name:
                return c
        else:
            raise ValueError()

    def run(self, clusters=None, max_power=0, pertT=False):
        if pertT:
            raise NotImplementedError("Perturbative triplet correction currently not implemented.")
        if clusters is None:
            clusters = self.clusters
        if not clusters:
            raise ValueError("No clusters defined for EmbCC calculation.")

        assert self.check_no_overlap()

        MPI_comm.Barrier()
        t_start = MPI.Wtime()

        for idx, c in enumerate(clusters):
            if MPI_rank != (idx % MPI_size):
                continue

            log.debug("Running cluster %s on rank %d", c.name, MPI_rank)
            c.run_solver(max_power=max_power, pertT=pertT)
            log.debug("Cluster %s on rank %d is done.", c.name, MPI_rank)

        all_conv = self.collect_results()
        if MPI_rank == 0:
            self.print_cluster_results()

        MPI_comm.Barrier()
        wtime = MPI.Wtime() - t_start
        log.info("Total wall time for EmbCCSD: %.0f min %.2g s", *divmod(wtime, 60.0))

        return all_conv

    def collect_results(self):
        log.debug("Communicating results.")
        clusters = self.clusters

        # Communicate
        def mpi_reduce(attribute, op=MPI.SUM, root=0):
            res = MPI_comm.reduce(np.asarray([getattr(c, attribute) for c in clusters]), op=op, root=root)
            return res

        converged = mpi_reduce("converged", op=MPI.PROD)
        nbath0 = mpi_reduce("nbath0")
        nbath = mpi_reduce("nbath")
        nfrozen = mpi_reduce("nfrozen")
        #e_cl_ccsd = mpi_reduce("e_cl_ccsd")

        e_corr = mpi_reduce("e_corr")
        e_corr_full = mpi_reduce("e_corr_full")
        e_corr_alt = mpi_reduce("e_corr_alt")

        #e_ccsd = mpi_reduce("e_ccsd")
        #e_pt = mpi_reduce("e_pt")

        #e_ccsd_v = mpi_reduce("e_ccsd_v")
        #e_ccsd_w = mpi_reduce("e_ccsd_w")
        #e_ccsd_z = mpi_reduce("e_ccsd_z")

        #converged = MPI_comm.reduce(np.asarray([c.converged for c in clusters]), op=MPI.PROD, root=0)
        #nbath0 = MPI_comm.reduce(np.asarray([c.nbath0 for c in clusters]), op=MPI.SUM, root=0)
        #nbath = MPI_comm.reduce(np.asarray([c.nbath for c in clusters]), op=MPI.SUM, root=0)
        #nfrozen = MPI_comm.reduce(np.asarray([c.nfrozen for c in clusters]), op=MPI.SUM, root=0)
        #e_cl_ccsd = MPI_comm.reduce(np.asarray([c.e_cl_ccsd for c in clusters]), op=MPI.SUM, root=0)
        #e_ccsd = MPI_comm.reduce(np.asarray([c.e_ccsd for c in clusters]), op=MPI.SUM, root=0)
        #e_pt = MPI_comm.reduce(np.asarray([c.e_pt for c in clusters]), op=MPI.SUM, root=0)

        #e_ccsd_v = MPI_comm.reduce(np.asarray([c.e_ccsd_v for c in clusters]), op=MPI.SUM, root=0)
        #e_ccsd_w = MPI_comm.reduce(np.asarray([c.e_ccsd_w for c in clusters]), op=MPI.SUM, root=0)
        #e_ccsd_z = MPI_comm.reduce(np.asarray([c.e_ccsd_z for c in clusters]), op=MPI.SUM, root=0)

        if MPI_rank == 0:
            for cidx, c in enumerate(self.clusters):
                c.converged = converged[cidx]
                c.nbath0 = nbath0[cidx]
                c.nbath = nbath[cidx]
                c.nfrozen = nfrozen[cidx]
                #c.e_cl_ccsd = e_cl_ccsd[cidx]
                #c.e_ccsd = e_ccsd[cidx]
                c.e_corr = e_corr[cidx]
                #c.e_pt = e_pt[cidx]

                #c.e_ccsd_v = e_ccsd_v[cidx]
                #c.e_ccsd_w = e_ccsd_w[cidx]
                #c.e_ccsd_z = e_ccsd_z[cidx]

                c.e_corr_full = e_corr_full[cidx]
                c.e_corr_alt = e_corr_alt[cidx]

            #self.e_ccsd = sum(e_ccsd)
            self.e_corr = sum(e_corr)
            self.e_corr_alt = sum(e_corr_alt)
            #self.e_pt = sum(e_pt)

            #self.e_corr = self.e_ccsd + self.e_pt
            self.e_tot = self.mf.e_tot + self.e_corr
            self.e_tot_alt = self.mf.e_tot + self.e_corr_alt

            #self.e_ccsd_v = sum(e_ccsd_v)
            #self.e_ccsd_w = sum(e_ccsd_w)
            #self.e_ccsd_z = sum(e_ccsd_z)

        #if MPI_rank == 0:
        #    log.info("Local correlation energy contributions")
        #    log.info("--------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd, self.e_pt)


        #    log.info("Local correlation energy contributions (virtual)")
        #    log.info("------------------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd_v, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd_v, 0.0)


        #    log.info("Local correlation energy contributions (symmetrized)")
        #    log.info("------------------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd_w, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd_w, 0.0)


        #    log.info("Local correlation energy contributions (symmetrized, virtual)")
        #    log.info("------------------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd_z, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd_z, 0.0)


        #    log.info("Full cluster CCSD energies")
        #    log.info("--------------------------")
        #    for c in clusters:
        #        log.info("%10s: CCSD=%+16.8g Eh", c.name, c.e_cl_ccsd)

        #    log.debug("Communicating results done.")

        return np.all(converged)

    def print_cluster_results(self):
        log.info("Energy contributions per cluster")
        log.info("------------------------0-------")
        # Name solver nactive (local, dmet bath, add bath) nfrozen E_corr_full E_corr
        linefmt = "%10s  %6s  %3d (%3d,%3d,%3d)  %3d: Full=%16.8g Eh Local=%16.8g Eh"
        totalfmt = "Total=%16.8g Eh"
        for c in self.clusters:
            log.info(linefmt, c.name, c.solver, len(c)+c.nbath, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_corr_full, c.e_corr)
        log.info(totalfmt, self.e_corr)

    def reset(self, mf=None, **kwargs):
        if mf:
            self.mf = mf
        for c in self.clusters:
            c.reset(**kwargs)
