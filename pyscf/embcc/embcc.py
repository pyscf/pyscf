import logging
import os.path
import functools

import numpy as np
import scipy
import scipy.linalg
from mpi4py import MPI

import pyscf
import pyscf.ao2mo
import pyscf.lo
import pyscf.cc
import pyscf.ci
import pyscf.mcscf
import pyscf.fci
import pyscf.mp

import pyscf.pbc
import pyscf.pbc.cc

from .orbitals import Orbitals
from .util import eigassign, eigreorder_logging

__all__ = [
        "EmbCC",
        ]

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

# optimize is False by default, despite NumPy's documentation, as of version 1.17
einsum = functools.partial(np.einsum, optimize=True)

#class Threshold:
#
#    def __init__(self, minnum=None, maxnum=None, minval=None, maxval=None):
#        self.minnum = minnum
#        self.maxnum = maxnum
#        self.minval = minval
#        self.maxval = maxval
#
#    def get_number(self, a):
#        """Number is in [minnum, maxnum]
#        if minval
#
#
#        if self.number is not None:
#            return self.number
#        else:
#            return

def reorder_columns(a, *args):
    """Reorder columns of matrix a. The new order must be specified by a list of tuples,
    where each tuple represents a block of columns, with the first tuple index being the
    first column index and the second tuple index the number of columns in the respective
    block.
    """
    starts, sizes = zip(*args)
    n = len(starts)

    #slices
    starts = [s if s is not None else 0 for s in starts]
    ends = [starts[i]+sizes[i] if sizes[i] is not None else None for i in range(n)]
    slices = [np.s_[starts[i]:ends[i]] for i in range(n)]

    b = np.hstack([a[:,s] for s in slices])
    assert b.shape == a.shape
    return b

def get_time_string(seconds):
    m, s = divmod(seconds, 60)
    if seconds >= 3600:
        tstr = "%.0f h, %.0f min, %.0f s" % (divmod(m, 60) + (s,))
    elif seconds >= 60:
        tstr = "%.0f min %.1f s" % (m, s)
    else:
        tstr = "%.2f s" % s
    return tstr


# This is not faster than PySCF...
#def transform_eri(mf, C1, C2, C3, C4):
#    mol = mf.mol
#    norb = mol.nao_nr()
#
#    t0 = MPI.Wtime()
#    #eri = mol.intor("int2e", aosym="s8")
#    eri = mol.intor("int2e", aosym="1")
#    time_eri = MPI.Wtime() - t0
#
#    #t0 = MPI.Wtime()
#    #eri = pyscf.ao2mo.restore(1, eri, norb, 1)
#    #time_restore = MPI.Wtime() - t0
#
#    t0 = MPI.Wtime()
#    eri = einsum("abcd,ai,bj,ck,dl->ijkl", eri, C1, C2, C3, C4)
#    time_trafo = MPI.Wtime() - t0
#
#    #log.debug("AO timings: eri=%s, restore=%s, transform=%s", *[get_time_string(t) for t in (time_eri, time_restore, time_trafo)])
#    log.debug("AO timings: eri=%s, transform=%s", *[get_time_string(t) for t in (time_eri, time_trafo)])
#    return eri

class Cluster:

    def __init__(self, base, name, indices, coeff=None, solver="CCSD", bath_type="power", tol_bath=1e-3, tol_dmet_bath=1e-8,
            **kwargs):
        """
        Parameters
        ----------
        name :
            Name of cluster.
        indices:
            Atomic orbital indices of cluster. [ local_orbital_type == "ao" ]
            Intrinsic atomic orbital indices of cluster. [ local_orbital_type == "iao" ]
        """

        self.base = base
        self.name = name
        self.indices = indices
        self.coeff = coeff

        # Optional
        self.solver = solver
        self.bath_type = bath_type


        self.local_orbital_type = kwargs.get("local_orbital_type", "ao")
        if self.local_orbital_type == "iao":
            if self.coeff is None:
                raise ValueError()

        #self.bath_target_size = (None, None)    # (Occupied, Virtual)
        self.bath_target_size = kwargs.get("bath_target_size", [None, None])    # (Occupied, Virtual)
        self.tol_bath = tol_bath
        self.tol_dmet_bath = tol_dmet_bath

        # Virtual natural orbitals
        #self.tol_vno = kwargs.get("tol_vno", 1e-3)
        #self.vno_ratio = kwargs.get("vno_ratio", None)
        #target_no = int(self.vno_ratio*len(self)+0.5) if self.vno_ratio is not None else None
        #self.n_vno = kwargs.get("n_vno", None) or target_no

        self.delta_mp2_correction = kwargs.get("delta_mp2_correction", True)

        self.use_ref_orbitals_dmet = kwargs.get("use_ref_orbitals_dmet", True)
        self.use_ref_orbitals_bath = kwargs.get("use_ref_orbitals_bath", True)
        #self.use_ref_orbitals_bath = False

        self.symmetry_factor = kwargs.get("symmetry_factor", 1.0)


        # Restart solver from previous solution [True/False]
        #self.restart_solver = kwargs.get("restart_solver", True)
        self.restart_solver = kwargs.get("restart_solver", False)
        # Parameters needed for restart (C0, C1, C2 for CISD; T1, T2 for CCSD) are saved here
        self.restart_params = kwargs.get("restart_params", {})

        self.set_default_attributes()


    def reset(self, keep_ref_orbitals=True):
        """Reset cluster object. By default it stores the previous orbitals, so they can be used
        as reference orbitals for a new calculation of different geometry."""
        ref_orbitals = self.orbitals
        self.set_default_attributes()
        if keep_ref_orbitals:
            self.ref_orbitals = ref_orbitals
        #log.debug("Resetting cluster %s. New vars:\n%s", self.name, vars(self))

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
        self.e_corr_dmp2 = 0.0

        self.e_corr_v = 0.0
        self.e_corr_v_dmp2 = 0.0

        self.e_corr_var = 0.0
        self.e_corr_var2 = 0.0
        self.e_corr_var3 = 0.0


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

    def make_projector_s121(self, indices=None):
        """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
        if indices is None:
            indices = self.indices
        S1 = self.mf.get_ovlp()
        nao = self.mol.nao_nr()
        S2 = S1[np.ix_(indices, indices)]
        S21 = S1[indices]
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
        nloc = len(e[e>1e-5])
        assert nloc == len(self), "Error finding local orbitals: %s" % e
        assert np.allclose(np.linalg.multi_dot((C.T, S, C)), np.eye(C.shape[-1]))

        return C


    def project_ref_orbitals(self, C, C_ref, space):
        """Project reference orbitals into available space in new geometry.

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

    def make_dmet_bath_orbitals(self, C, ref_orbitals=None, tol=None):
        """If C_ref is specified, complete DMET orbital space using active projection of reference orbitals."""
        if ref_orbitals is None:
            ref_orbitals=self.ref_orbitals
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
        if self.use_ref_orbitals_dmet and ref_orbitals is not None:
            C_ref = ref_orbitals.get_coeff("dmet-bath")
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

    def make_power_bath_orbitals(self, orbitals, maxpower=5, nbath=None, tol=1e-8):

        C = orbitals.C.copy()
        nbath0 = orbitals.get_size("dmet-bath")
        nenvocc = orbitals.get_size("occ-env")
        nenvvir = orbitals.get_size("vir-env")

        # Occupied
        log.info("Calculating occupied power bath orbitals.")
        nbathocc = 0
        for power in range(1, maxpower+1):
            if nbathocc >= nenvocc:
                break

            occ_space = np.s_[len(self)+nbath0+nbathocc:len(self)+nbath0+nenvocc]
            C, nbo = self.make_power_bath_orbitals_power(C, "occ", occ_space, powers=[power], tol=tol)
            log.info("Power=%d, coupled orbitals=%3d", power, nbo)

            if nbath is not None and nbathocc+nbo >= nbath[0]:
                nbathocc = nbath[0]
                break
            else:
                nbathocc += nbo

        # Virtual
        log.info("Calculating virtual power bath orbitals.")
        nbathvir = 0
        for power in range(1, maxpower+1):
            if nbathvir >= nenvvir:
                break

            vir_space = np.s_[len(self)+nbath0+nenvocc+nbathvir:]
            C, nbv = self.make_power_bath_orbitals_power(C, "vir", vir_space, powers=[power], tol=tol)
            log.info("Power=%d, coupled orbitals=%3d", power, nbv)

            if nbath is not None and nbathvir+nbv >= nbath[1]:
                nbathvir = nbath[1]
                break
            else:
                nbathvir += nbv

        return C, nbathocc, nbathvir

    def make_power_bath_orbitals_power(self, C, kind, non_local, powers=(1,), nbath=None, tol=1e-8,
            normalize=False, eref=None):
        #if tol is None:
        #    tol = self.tol_bath
        assert nbath is not None or tol is not None

        #if eref < max(abs(self.mf.mo_energy)):
        #    log.critical("Reference energy of power orbitals=%.5g smaller than absolute largest HF eigenvalue=%.5g",
        #            eref, max(abs(self.mf.mo_energy)))

        if eref is None:
            #eref = max(abs(self.mf.mo_energy))
            eref = 1.0

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        nloc = len(self)
        loc = np.s_[:nloc]

        b = []
        for power in powers:
            bp = np.einsum("xi,i,ai->xa", csc[non_local], (e/eref)**power, csc[loc], optimize=True)
            b.append(bp)
        b = np.hstack(b)

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        # eigenvalues should never be negative. If they are, it's due to numerical error due to large exponent in power bath orbitals.
        # we demand that our tolerance is at least on order of magnitude larger than the numerical error.
        #assert np.all(e > -1e-10)
        assert np.all(e > -tol/10)
        e, v = e[::-1], v[:,::-1]

        with open("power-%d-%s-%s.txt" % (power, self.name, kind), "ab") as f:
            np.savetxt(f, e[loc][np.newaxis])

        # REORDER COUPLED ACCORDING TO REFERENCE
        #if True:
        if False:
            # Here we reorder the eigenvalues
            CV = np.dot(C[:,non_local], v[:,loc])
            reffile = "power-%d-%s-ref.npz" % (power, kind)
            if os.path.isfile(reffile):
                ref = np.load(reffile)
                e_ref, CV_ref = ref["e"], ref["CV"]

                #if N_ref is not None and R_ref is not None:
                log.debug("Reordering eigenvalues according to reference.")
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v/e", return_cost=True)
                reorder, cost = eigassign(e_ref, CV_ref, e[loc], CV, b=S, cost_matrix="e^2/v", return_cost=True)
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v*e", return_cost=True)
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v*sqrt(e)", return_cost=True)
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="evv", return_cost=True)
                eigreorder_logging(e[loc], reorder, log.debug)
                log.debug("eigassign cost function value=%g", cost)
                reorder_full = np.hstack((reorder, np.arange(nloc, len(e))))
                log.debug("Reorder: %s", reorder)
                log.debug("Full reorder: %s", reorder_full)
                e = e[reorder_full]
                v = v[:,reorder_full]
                CV = CV[:,reorder]

            with open("power-%d-%s-%s-ordered.txt" % (power, self.name, kind), "ab") as f:
                np.savetxt(f, e[loc][np.newaxis])

            np.savez(reffile, e=e[loc], CV=CV)

        # nbath takes preference
        if nbath is not None:
            if tol is not None:
                log.warning("Warning: tolerance is %.g, but nbath=%d is used.", tol, nbath)
            nbath = min(nbath, len(e))
            log.debug("Eigenvalues of kind=%s, power=%d bath", kind, power)
        else:
            nbath = sum(e >= tol)
            log.debug("Eigenvalues of kind=%s, power=%d bath, tolerance=%e", kind, power, tol)
        log.debug("%d included eigenvalues:\n%r", nbath, e[:nbath])
        log.debug("%d excluded eigenvalues (first 3):\n%r", len(e)-nbath, e[nbath:nbath+3])

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

    def make_matsubara_bath_orbitals(self, C, kind, non_local, npoints=1000, beta=100.0,
            nbath=None, tol=None, normalize=False):

        assert nbath is not None or tol is not None

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

        # nbath takes preference
        if nbath is not None:
            if tol is not None:
                log.warning("Warning: tolerance is %.g, but nbath=%d is used.", tol, nbath)
            nbath = min(nbath, len(e))
            log.debug("Eigenvalues of kind=%s bath", kind)
        else:
            nbath = sum(e >= tol)
            log.debug("Eigenvalues of kind=%s bath, tolerance=%e", kind, tol)
        log.debug("%d included eigenvalues:\n%r", nbath, e[:nbath])
        log.debug("%d excluded eigenvalues (first 3):\n%r", len(e)-nbath, e[nbath:nbath+3])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def make_cubegen_file(self, C, orbitals, filename, **kwargs):
        from pyscf.tools import cubegen

        orbital_labels = np.asarray(self.mol.ao_labels(None))[orbitals]
        orbital_labels = ["-".join(x) for x in orbital_labels]

        for idx, orb in enumerate(orbitals):
            filename_orb = "%s-%s" % (filename, orbital_labels[idx])
            cubegen.orbital(self.mol, filename_orb, C[:,orb], **kwargs)

    def analyze_orbitals(self, orbitals=None, sort=True):
        if orbitals is None:
            orbitals = self.orbitals

        active_spaces = ["local", "dmet-bath", "occ-bath", "vir-bath"]
        frozen_spaces = ["occ-env", "vir-env"]
        spaces = [active_spaces, frozen_spaces, *active_spaces, *frozen_spaces]
        chis = np.zeros((self.mol.nao_nr(), len(spaces)))

        # Calculate chi
        for ao in range(self.mol.nao_nr()):
            S = self.mf.get_ovlp()
            S121 = 1/S[ao,ao] * np.outer(S[:,ao], S[:,ao])
            for ispace, space in enumerate(spaces):
                C = orbitals.get_coeff(space)
                SC = np.dot(S121, C)
                chi = np.sum(SC[ao]**2)
                chis[ao,ispace] = chi

        ao_labels = np.asarray(self.mol.ao_labels(None))
        if sort:
            sort = np.argsort(-np.around(chis[:,0], 3), kind="mergesort")
            ao_labels = ao_labels[sort]
            chis2 = chis[sort]
        else:
            chis2 = chis

        # Output
        log.info("Orbitals of cluster %s", self.name)
        log.info("===================="+len(self.name)*"=")
        log.info(("%18s" + " " + len(spaces)*"  %9s"), "Atomic orbital", "Active", "Frozen", "Local", "DMET bath", "Occ. bath", "Vir. bath", "Occ. env.", "Vir. env")
        log.info((18*"-" + " " + len(spaces)*("  "+(9*"-"))))
        for ao in range(self.mol.nao_nr()):
            line = "[%3s %3s %2s %-5s]:" % tuple(ao_labels[ao])
            for ispace, space in enumerate(spaces):
                line += "  %9.3g" % chis2[ao,ispace]
            log.info(line)

        # Active chis
        return chis[:,0]

    def make_mp2_no(self, orbitals, kind, nno=None, tol=None, symmetry_factor=None, N_ref=None, R_ref=None):
        """Select virtual space from MP2 natural orbitals (NOs) according to occupation number."""
        assert nno is not None or tol is not None
        assert kind in ("occ", "vir")

        if symmetry_factor is None:
            symmetry_factor = self.symmetry_factor

        if kind == "vir":
            nvir = orbitals.get_size(("vir-env"))
            if nvir == 0:
                return orbitals, 0, 0.0
        elif kind == "occ":
            nocc = orbitals.get_size(("occ-env"))
            if nocc == 0:
                return orbitals, 0, 0.0

        # Seperate cluster space into occupied and virtual
        C_cl = orbitals.get_coeff(("local", "dmet-bath"))
        S = self.mf.get_ovlp()
        D_cl = np.linalg.multi_dot((C_cl.T, S, self.mf.make_rdm1(), S, C_cl)) / 2
        e, r = np.linalg.eigh(D_cl)
        e, r = e[::-1], r[:,::-1]
        assert np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=1e-6, rtol=0)
        nclocc = sum(e > 0.5)
        nclvir = sum(e < 0.5)
        log.debug("Number of occupied/virtual cluster orbitals: %d/%d", nclocc, nclvir)
        C_cl_o = np.dot(C_cl, r)[:,:nclocc]
        C_cl_v = np.dot(C_cl, r)[:,nclocc:]

        F = self.mf.get_fock()

        def make_MP2(Co, Cv, make_dm=True, local_E=True, canon_occ=True, canon_vir=True):
            Fo = np.linalg.multi_dot((Co.T, F, Co))
            Fv = np.linalg.multi_dot((Cv.T, F, Cv))
            # Canonicalization [optional]
            t0 = MPI.Wtime()
            if canon_occ:
                Eo, Ro = np.linalg.eigh(Fo)
                Co = np.dot(Co, Ro)
            else:
                Eo = np.diag(Fo)
            if canon_vir:
                Ev, Rv = np.linalg.eigh(Fv)
                Cv = np.dot(Cv, Rv)
            else:
                Ev = np.diag(Fv)
            time_canon = MPI.Wtime() - t0
            log.debug("Time for canonicalization: %s", get_time_string(time_canon))
            no = Co.shape[-1]
            nv = Cv.shape[-1]
            # Make T2
            t0 = MPI.Wtime()
            eri = pyscf.ao2mo.general(self.mol, (Co, Cv, Co, Cv)).reshape(no,nv,no,nv)
            time_ao2mo = MPI.Wtime() - t0
            log.debug("Time for ao2mo: %s", get_time_string(time_ao2mo))

            #t1 = MPI.Wtime() - t0
            #t0 = MPI.Wtime()
            #eri2 = pyscf.ao2mo.general(self.mol, (Cv, Co, Cv, Co)).reshape(nv,no,nv,no)
            #t2 = MPI.Wtime() - t0
            #t0 = MPI.Wtime()
            #eri3 = pyscf.ao2mo.general(self.mol, (Co, Cv, Cv, Co)).reshape(no,nv,nv,no)
            #t3 = MPI.Wtime() - t0
            #t0 = MPI.Wtime()
            #eri4 = pyscf.ao2mo.general(self.mol, (Cv, Co, Co, Cv)).reshape(nv,no,no,nv)
            #t4 = MPI.Wtime() - t0

            #with open("ao2mo-times.txt", "a") as f:
            #    f.write("%3d  %3d  %.6g  %.6g  %.6g  %.6g\n" % (no, nv, t1, t2, t3, t4))

            #assert np.allclose(eri, eri2.transpose(1, 0, 3, 2))
            #assert np.allclose(eri, eri3.transpose(0, 1, 3, 2))
            #assert np.allclose(eri, eri4.transpose(1, 0, 2, 3))
            #1/0

            Eov = (Eo[:,np.newaxis] - Ev[np.newaxis,:])
            t0 = MPI.Wtime()
            T2 = np.zeros((no,no,nv,nv))
            for i in range(no):
                d = (Eov[i][np.newaxis,:,np.newaxis] + Eov[:,np.newaxis,:])
                T2[i] += eri[i].transpose(1,0,2) / d
            time_t2 = MPI.Wtime() - t0
            log.debug("Time for T2: %s", get_time_string(time_t2))
            # MP2 energy [with optional local projector P]

            # Alternative: PySCF MP2
            C = np.hstack((Co, Cv))
            E = np.hstack((Eo, Ev))
            occ = np.asarray(no*[2] + nv*[0])
            t0 = MPI.Wtime()
            mp2 = pyscf.mp.MP2(self.mf, mo_coeff=C, mo_occ=occ)
            eris = mp2.ao2mo()
            assert np.allclose(eri, eris.ovov.reshape((no,nv,no,nv)))
            e_mp2, T2b = mp2.kernel(mo_energy=E, eris=eris)
            time_mp2 = MPI.Wtime() - t0
            log.debug("Time for MP2: %s", get_time_string(time_mp2))

            log.debug(T2.shape)
            log.debug(T2b.shape)
            log.debug(np.linalg.norm(T2-T2b))
            log.debug(T2[0,0,0,0])
            log.debug(T2b[0,0,0,0])
            assert np.allclose(T2, T2b)


            if local_E:
                #l = self.indices
                #P = np.linalg.multi_dot((Co.T, S[:,l], Co[l]))

                # CHECK 1
                P = self.get_local_energy_projector(Co)
                #assert np.allclose(P, P2)

                pT2 = einsum("xi,ijab->xjab", P, T2)
            else:
                pT2 = T2
            t0 = MPI.Wtime()
            e_mp2 = 2*einsum('ijab,iajb', pT2, eri)
            e_mp2 -=  einsum('ijab,jaib', pT2, eri)
            time_emp2 = MPI.Wtime() - t0
            log.debug("Time for energy: %s", get_time_string(time_emp2))
            # MP2 density matrix [optional]
            if make_dm:

                t0 = MPI.Wtime()
                #Doo2 = 2*(2*einsum("kiab,kjab->ij", T2, T2)
                #          - einsum("kiab,kjba->ij", T2, T2))
                Doo = 2*(2*einsum("ikab,jkab->ij", T2, T2)
                         - einsum("ikab,jkba->ij", T2, T2))
                #assert np.allclose(Doo, Doo2)

                Dvv = 2*(2*einsum("ijac,ijbc->ab", T2, T2)
                         - einsum("ijac,ijcb->ab", T2, T2))

                time_dm = MPI.Wtime() - t0
                log.debug("Time for DM: %s", get_time_string(time_dm))

                # Rotate back to input coeffients (undo canonicalization)
                if canon_occ:
                    Doo = np.linalg.multi_dot((Ro, Doo, Ro.T))
                if canon_vir:
                    Dvv = np.linalg.multi_dot((Rv, Dvv, Rv.T))
                return e_mp2, Doo, Dvv

            return e_mp2

        # All virtuals
        if kind == "vir":
            Co = C_cl_o
            Cv = np.hstack((C_cl_v, orbitals.get_coeff(("vir-env"))))
            e_mp2_full, _, D = make_MP2(Co, Cv)
            N, R = np.linalg.eigh(D[nclvir:,nclvir:])
        elif kind == "occ":
            Co = np.hstack((C_cl_o, orbitals.get_coeff(("occ-env"))))
            Cv = C_cl_v
            e_mp2_full, D, _ = make_MP2(Co, Cv)
            N, R = np.linalg.eigh(D[nclocc:,nclocc:])

        N, R = N[::-1], R[:,::-1]
        #log.debug("Occupation numbers:\n%s", N)

        # --- TESTING ---
        # Save occupation values
        with open("mp2-no-%s-%s.txt" % (self.name, kind), "ab") as f:
            np.savetxt(f, N[np.newaxis])

        # Here we reorder the eigenvalues
        #if True:
        if False:
            if kind == "vir":
                CR = np.dot(Cv[:,nclvir:], R)
            elif kind == "occ":
                CR = np.dot(Co[:,nclocc:], R)
            reffile = "mp2-no-%s-ref.npz" % kind

            if os.path.isfile(reffile):
                ref = np.load(reffile)
                N_ref, CR_ref = ref["N"], ref["CR"]

                #if N_ref is not None and R_ref is not None:
                log.debug("Reordering eigenvalues according to reference.")
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v/e", return_cost=True)
                reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="e^2/v", return_cost=True)
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v*e", return_cost=True)
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v*sqrt(e)", return_cost=True)
                #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="evv", return_cost=True)
                eigreorder_logging(N, reorder, log.debug)
                log.debug("eigassign cost function value=%g", cost)
                N = N[reorder]
                R = R[:,reorder]
                CR = CR[:,reorder]

            with open("mp2-no-%s-%s-ordered.txt" % (self.name, kind), "ab") as f:
                np.savetxt(f, N[np.newaxis])

            np.savez(reffile, N=N, CR=CR)

        if nno is None:
            nno = sum(N >= tol)
        else:
            nno = min(nno, len(N))

        protect_degeneracies = False
        #protect_degeneracies = True
        # Avoid splitting within degenerate subspace
        if protect_degeneracies and nno > 0:
            #dgen_tol = 1e-10
            N0 = N[nno-1]
            while nno < len(N):
                #if abs(N[nno] - N0) <= dgen_tol:
                if np.isclose(N[nno], N0, atol=1e-9, rtol=1e-6):
                    log.debug("Degenerate MP2 NO found: %.6e vs %.6e - adding to bath space.", N[nno], N0)
                    nno += 1
                else:
                    break

        log.debug("Using %d out of %d MP2 natural %s orbitals", nno, len(N), kind)

        log.debug("Difference in occupation:\n%s", N[:nno])
        log.debug("Following 3:\n%s", N[nno:nno+3])

        # Delta MP2 correction
        # ====================

        if kind == "vir":
            Cno = Cv.copy()
            Cno[:,nclvir:] = np.dot(Cv[:,nclvir:], R)
            nclno = nclvir + nno
            Cno = Cno[:,:nclno]
            e_mp2_no = make_MP2(Co, Cno, make_dm=False)

        elif kind == "occ":
            Cno = Co.copy()
            Cno[:,nclocc:] = np.dot(Co[:,nclocc:], R)
            nclno = nclocc + nno
            Cno = Cno[:,:nclno]
            e_mp2_no = make_MP2(Cno, Cv, make_dm=False)

        e_mp2_full *= symmetry_factor
        e_mp2_no *= symmetry_factor

        e_delta_mp2 = e_mp2_full - e_mp2_no
        log.debug("Delta MP2 correction: full=%.8e, active=%.8e, correction=%+.8e", e_mp2_full, e_mp2_no, e_delta_mp2)

        orbitals_out = orbitals.copy()

        if kind == "vir":
            orbitals_out.transform(R, "vir-env")
            indices = orbitals_out.get_indices("vir-env")
            orbitals_out.delete_space("vir-env")
            orbitals_out.define_space("vir-bath", np.s_[indices[0]:indices[0]+nno])
            orbitals_out.define_space("vir-env", np.s_[indices[0]+nno:])
        elif kind == "occ":
            orbitals_out.transform(R, "occ-env")
            indices = orbitals_out.get_indices("occ-env")
            orbitals_out.delete_space("occ-env")
            orbitals_out.define_space("occ-bath", np.s_[indices[0]:indices[0]+nno])
            orbitals_out.define_space("occ-env", np.s_[indices[0]+nno:])

        return orbitals_out, nno, e_delta_mp2

    def make_cluster(self, ref_orbitals=None):

        # Orbitals from a reference calaculation (e.g. different geometry)
        # Used for recovery of orbitals via active transformation
        ref_orbitals = ref_orbitals or self.ref_orbitals

        C = self.make_local_orbitals()
        C, nbath0, nenvocc = self.make_dmet_bath_orbitals(C, ref_orbitals=ref_orbitals)
        nbath = nbath0

        # Diagonalize cluster DM to get fully occupied/virtual orbitals
        S = self.mf.get_ovlp()
        ncl = len(self) + nbath0
        cl = np.s_[:ncl]
        SC = np.dot(S, C[:,cl])
        D = np.linalg.multi_dot((SC.T, self.mf.make_rdm1(), SC)) / 2
        e, v = np.linalg.eigh(D)
        assert np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=1e-6, rtol=0)
        rev = np.s_[::-1]
        e, v = e[rev], v[:,rev]
        C[:,cl] = np.dot(C[:,cl], v)
        nocc_cl = sum(e > 0.5)
        nvir_cl = sum(e <= 0.5)
        log.info("Occupied/virtual states in local+DMET space: %d/%d", nocc_cl, nvir_cl)

        orbitals = Orbitals(C)
        #orbitals.define_space("local", np.s_[:len(self)])
        #orbitals.define_space("dmet-bath", np.s_[len(self):ncl])
        orbitals.define_space("occ-imp", np.s_[:nocc_cl])
        orbitals.define_space("vir-imp", np.s_[nocc_cl:ncl])
        orbitals.define_space("occ-env", np.s_[ncl:ncl+nenvocc])
        orbitals.define_space("vir-env", np.s_[ncl+nenvocc:])

        return Orbitals


    def run_solver(self, solver=None, max_power=0, pertT=False, diagonalize_fock=True, cc_verbose=4,
            ref_orbitals=None, analyze_orbitals=False):

        solver = solver or self.solver

        # If solver is None, do not correlate cluster ("HF solver")
        if solver is None:
            self.e_corr = 0.0
            self.e_corr_v = 0.0
            self.e_corr_var = 0.0
            self.e_corr_var2 = 0.0
            self.e_corr_var3 = 0.0
            return 1

        # Orbitals from a reference calaculation (e.g. different geometry)
        # Used for recovery of orbitals via active transformation
        ref_orbitals = ref_orbitals or self.ref_orbitals

        #self.make_cubegen_file(np.eye(self.mol.nao_nr()), orbitals=list(range(len(self))), filename="AO")

        if self.coeff is None:
            C = self.make_local_orbitals()
        else:
            # Reorder local orbitals to the front
            C = np.hstack((self.coeff[:,self.indices], self.coeff[:,self.not_indices]))

        #self.make_cubegen_file(C, orbitals=list(range(len(self))), filename="ortho-AO")
        #1/0

        self.C_iao = C[:,:len(self)].copy()

        C, nbath0, nenvocc = self.make_dmet_bath_orbitals(C)
        nbath = nbath0

        ncl = len(self)+nbath0
        orbitals = Orbitals(C)
        orbitals.define_space("local", np.s_[:len(self)])
        orbitals.define_space("dmet-bath", np.s_[len(self):ncl])
        orbitals.define_space("occ-env", np.s_[ncl:ncl+nenvocc])
        orbitals.define_space("vir-env", np.s_[ncl+nenvocc:])


        # Reuse reference orbitals
        # ========================
        log.debug("ref_orbitals: %r", self.ref_orbitals)
        log.debug("use_ref_orbitals_bath: %r", self.use_ref_orbitals_bath)
        if ref_orbitals and self.use_ref_orbitals_bath:
            log.debug("Using reference bath orbitals.")
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

            e_delta_mp2 = 0.0

        # Make new bath orbitals
        else:
            # MP2 natural orbitals
            # ====================
            # Currently only MP2 NOs OR additional bath orbitals are supported!
            #if self.tol_vno or self.n_vno:
            if self.bath_type == "mp2-no":
                log.debug("Making MP2 virtual natural orbitals.")
                t0 = MPI.Wtime()
                orbitals2, nvno, e_delta_mp2_v = self.make_mp2_no(
                        orbitals, kind="vir", nno=self.bath_target_size[1], tol=self.tol_bath)
                log.debug("Wall time for MP2 VNO: %s", get_time_string(MPI.Wtime()-t0))
                C = orbitals2.C
                nbathvir = nvno

                log.debug("Making MP2 occupied natural orbitals.")
                t0 = MPI.Wtime()
                orbitals3, nono, e_delta_mp2_o = self.make_mp2_no(
                        orbitals2, kind="occ", nno=self.bath_target_size[0], tol=self.tol_bath)
                log.debug("Wall time for MP2 ONO: %s", get_time_string(MPI.Wtime()-t0))
                C = orbitals3.C
                nbathocc = nono
                #e_delta_mp2_o = 0.0
                #nbathocc = 0

                e_delta_mp2 = e_delta_mp2_v + e_delta_mp2_o
                log.debug("Total delta MP2 correction=%.8g", e_delta_mp2)

            else:
                e_delta_mp2 = 0.0
                # Use previous orbitals
                #if ref_orbitals and self.use_ref_orbitals_bath:
                #    # Occupied
                #    nbathocc = ref_orbitals.get_size("occ-bath")
                #    if nbathocc == 0:
                #        log.debug("No reference occupied bath orbitals.")
                #    else:
                #        C, eig = self.project_ref_orbitals(C, ref_orbitals.get_coeff("occ-bath"),
                #                orbitals.get_indices("occ-env"))
                #        log.debug("Eigenvalues of %d projected occupied bath orbitals:\n%s",
                #                nbathocc, eig[:nbathocc])
                #        log.debug("Next 3 eigenvalues: %s", eig[nbathocc:nbathocc+3])
                #    # Virtual
                #    nbathvir = ref_orbitals.get_size("vir-bath")
                #    if nbathvir == 0:
                #        log.debug("No reference virtual bath orbitals.")
                #    else:
                #        C, eig = self.project_ref_orbitals(C, ref_orbitals.get_coeff("vir-bath"),
                #                orbitals.get_indices("vir-env"))
                #        log.debug("Eigenvalues of %d projected virtual bath orbitals:\n%s",
                #                nbathvir, eig[:nbathvir])
                #        log.debug("Next 3 eigenvalues: %s", eig[nbathvir:nbathvir+3])

                ## Add additional power bath orbitals
                #else:
                nbathocc = 0
                nbathvir = 0
                # Power orbitals
                if self.bath_type == "power":
                    if self.bath_target_size[0] is not None:
                        C, nbathocc, nbathvir =  self.make_power_bath_orbitals(
                                orbitals, maxpower=10, nbath=self.bath_target_size)
                        log.info("Found %3d/%3d occupied/virtual bath orbitals, with a target of %3d/%3d",
                                nbathocc, nbathvir, *self.bath_target_size)

                    else:
                        for power in range(1, max_power+1):
                            occ_space = np.s_[len(self)+nbath0+nbathocc:len(self)+nbath0+nenvocc]
                            #C, nbo = self.make_power_bath_orbitals(C, "occ", occ_space, power=power)
                            C, nbo = self.make_power_bath_orbitals_power(C, "occ", occ_space, powers=[power],
                                    nbath=self.bath_target_size[0], tol=self.tol_bath)
                            vir_space = np.s_[len(self)+nbath0+nenvocc+nbathvir:]
                            #C, nbv = self.make_power_bath_orbitals(C, "vir", vir_space, power=power)
                            C, nbv = self.make_power_bath_orbitals_power(C, "vir", vir_space, powers=[power],
                                    nbath=self.bath_target_size[1], tol=self.tol_bath)
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
                    C, nbo = self.make_matsubara_bath_orbitals(C, "occ", occ_space, nbath=self.bath_target_size[0], tol=self.tol_bath)
                    vir_space = np.s_[len(self)+nbath0+nenvocc+nbathvir:]
                    C, nbv = self.make_matsubara_bath_orbitals(C, "vir", vir_space, nbath=self.bath_target_size[1], tol=self.tol_bath)
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

        # TEST
        if analyze_orbitals:
            chi = self.analyze_orbitals(orbitals)

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
            self.e_corr_v = 0.0
            self.e_corr_var = 0.0
            self.e_corr_var2 = 0.0
            self.e_corr_var3 = 0.0
            self.converged = True
            return 1

        # Accelerates convergence + Canonicalization necessary for MP2
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

        t0 = MPI.Wtime()

        if solver == "MP2":
            mp2 = pyscf.mp.MP2(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            eris = mp2.ao2mo()
            e_corr_full, t2 = mp2.kernel(eris=eris)
            self.e_corr_full = self.symmetry_factor*e_corr_full
            C1, C2 = None, t2

        elif solver == "CCSD":
            if pbc:
                ccsd = pyscf.pbc.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            else:
                ccsd = pyscf.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            # We want to reuse the integral for local energy
            t0 = MPI.Wtime()
            eris = ccsd.ao2mo()
            log.debug("Time for ao2mo: %s", get_time_string(MPI.Wtime()-t0))
            ccsd.max_cycle = 100
            ccsd.verbose = cc_verbose
            if self.restart_solver:
                log.debug("Running CCSD starting with parameters for: %r...", self.restart_params.keys())
                ccsd.kernel(eris=eris, **self.restart_params)
            else:
                log.debug("Running CCSD...")
                ccsd.kernel(eris=eris)
            log.debug("CCSD done. converged: %r", ccsd.converged)
            if self.restart_solver:
                #self.restart_params = {"t1" : ccsd.t1, "t2" : ccsd.t2}
                self.restart_params["t1"] = ccsd.t1
                self.restart_params["t2"] = ccsd.t2
            C1 = ccsd.t1
            C2 = ccsd.t2 + einsum('ia,jb->ijab', ccsd.t1, ccsd.t1)

            # TESTING
            if self.benchmark is not None:
                a = ccsd.get_frozen_mask()
                C = ccsd.mo_coeff[:,a]
                o = ccsd.mo_occ[a] > 0
                v = ccsd.mo_occ[a] == 0
                Co = C[:,o]
                Cv = C[:,v]

                T2 = einsum("pi,qj,ijab,sa,tb->pqst", Co, Co, ccsd.t2, Cv, Cv)
                ref = self.benchmark

                a2 = ref.get_frozen_mask()
                Cb = ref.mo_coeff[:,a2]
                o2 = ref.mo_occ[a2] > 0
                v2 = ref.mo_occ[a2] == 0
                Co2 = Cb[:,o2]
                Cv2 = Cb[:,v2]

                T2_ref = einsum("pi,qj,ijab,sa,tb->pqst", Co2, Co2, ref.t2, Cv2, Cv2)

                #norm_tot = np.linalg.norm(T2 - T2_ref)
                #l = np.s_[:len(self)]
                #norm_tot = np.linalg.norm(T2[l,l,l,l] - T2_ref[l,l,l,l])

                delta = T2 - T2_ref
                # Relative error
                #delta = delta / (abs(T2_ref)+1e-14)
                delta = np.linalg.norm(T2 - T2_ref, axis=(2,3))
                delta = delta / np.linalg.norm(T2_ref, axis=(2,3))

                #omega = einsum("a,b,c,d->abcd", chi, chi, chi, chi)
                #omega = einsum("a,b,c,d->abcd", chi, chi, np.ones_like(chi), np.ones_like(chi))
                omega = einsum("a,b->ab", chi, chi)

                np.savetxt("omega-vs-T-cluster-%s" % self.name, np.vstack((omega.flatten(), delta.flatten())).T)

            self.converged = ccsd.converged
            e_corr_full = ccsd.e_corr
            self.e_corr_full = self.symmetry_factor*e_corr_full

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
        log.debug("Wall time for solver: %s", get_time_string(MPI.Wtime()-t0))

        log.debug("Calculating local energy...")
        t0 = MPI.Wtime()

        if solver == "MP2":
            self.e_mp2 = self.get_local_energy(mp2, C1, C2, eris=eris)
            self.e_corr = self.e_mp2
            self.e_corr_v = self.e_mp2 + e_delta

        elif solver == "CCSD":
            #self.e_ccsd, self.e_pt = self.get_local_energy_old(ccsd, pertT=pertT)
            #self.e_ccsd_v, _ = self.get_local_energy_old(ccsd, projector="vir", pertT=pertT)

            #self.e_ccsd_z = self.get_local_energy_most_indices(ccsd)

            self.e_ccsd = self.get_local_energy(ccsd, C1, C2, eris=eris)
            #self.e_ccsd = self.get_local_energy(ccsd, C1, C2, eris=eris)

            # TEST:
            #self.e_corr_var = self.get_local_energy(ccsd, C1, C2, project_var="left")
            #self.e_corr_var2 = self.get_local_energy(ccsd, C1, C2, project_var="center")

            self.e_ccsd_v = self.get_local_energy(ccsd, C1, C2, "virtual", eris=eris)
            #self.e_ccsd_v = self.get_local_energy_most_indices_2C(ccsd, C1, C2, eris=eris)

            self.e_corr = self.e_ccsd
            self.e_corr_dmp2 = self.e_ccsd + e_delta_mp2

            #self.e_corr = self.e_ccsd + e_delta

            self.e_corr_v = self.e_ccsd_v
            self.e_corr_v_dmp2 = self.e_ccsd_v + e_delta_mp2

            # TESTING
            #self.get_local_energy_parts(ccsd, C1, C2)

            # TEMP
            #self.e_corr_var = self.get_local_energy_most_indices(ccsd, C1, C2)
            #self.e_corr_var2 = self.get_local_energy_most_indices(ccsd, C1, C2, variant=2)
            #self.e_corr_var3 = self.get_local_energy_most_indices(ccsd, C1, C2, variant=3)


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
        log.debug("Wall time for local energy: %s", get_time_string(MPI.Wtime()-t0))

        return int(self.converged)

    def get_local_energy_projector(self, C, kind="right"):
        """Projector for local energy expression."""
        #log.debug("Making local energy projector for orbital type %s", self.local_orbital_type)
        S = self.mf.get_ovlp()
        if self.local_orbital_type == "ao":
            l = self.indices
            # This is the "natural way" to truncate in AO basis
            if kind == "right":
                P = np.linalg.multi_dot((C.T, S[:,l], C[l]))
            # These two methods - while exact in the full bath limit - might require some thought...
            # See also: CCSD in AO basis paper of Scuseria et al.
            elif kind == "left":
                P = np.linalg.multi_dot((C[l].T, S[l], C))
            elif kind == "center":
                s = scipy.linalg.fractional_matrix_power(S, 0.5)
                assert np.isclose(np.linalg.norm(s.imag), 0)
                s = s.real
                assert np.allclose(np.dot(s, s), S)
                P = np.linalg.multi_dot((C.T, s[:,l], s[l], C))
            else:
                raise ValueError("Unknown kind=%s" % kind)

        elif self.local_orbital_type == "iao":
            #n = C.shape[-1]
            #assert n <= len(self)
            #P = np.zeros((n, n))
            #P[:len(self),:len(self)] = np.eye(len(self))

            C_iao = self.C_iao
            CSC = np.linalg.multi_dot((C.T, S, C_iao))
            P = np.dot(CSC, CSC.T)

        else:
            raise ValueError()

        return P


    #def get_local_energy_parts(self, cc, C1, C2):

    #    a = cc.get_frozen_mask()
    #    # Projector to local, occupied region
    #    S = self.mf.get_ovlp()
    #    C = cc.mo_coeff[:,a]
    #    CTS = np.dot(C.T, S)

    #    # Project one index of T amplitudes
    #    l= self.indices
    #    r = self.not_indices
    #    o = cc.mo_occ[a] > 0
    #    v = cc.mo_occ[a] == 0

    #    eris = cc.ao2mo()

    #    def get_projectors(aos):
    #        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
    #        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
    #        return Po, Pv

    #    Lo, Lv = get_projectors(l)
    #    Ro, Rv = get_projectors(r)

    #    # Nomenclature:
    #    # old occupied: i,j
    #    # old virtual: a,b
    #    # new occupied: p,q
    #    # new virtual: s,t
    #    T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
    #    T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
    #    T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
    #    T1 = T1_ll + (T1_lr + T1_rl)/2

    #    F = eris.fock[o][:,v]
    #    e1 = 2*np.sum(F * T1)
    #    if not np.isclose(e1, 0):
    #        log.warning("Warning: large E1 component: %.8e" % e1)

    #    #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
    #    def project_T2(P1, P2, P3, P4):
    #        T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
    #        return T2p


    #    def epart(P1, P2, P3, P4):
    #        T2_part = project_T2(P1, P2, P3, P4)
    #        e_part = (2*einsum('ijab,iabj', T2_part, eris.ovvo)
    #              - einsum('ijab,jabi', T2_part, eris.ovvo))
    #        return e_part

    #    energies = []
    #    # 4
    #    energies.append(epart(Lo, Lo, Lv, Lv))
    #    # 3
    #    energies.append(2*epart(Lo, Lo, Lv, Rv))
    #    energies.append(2*epart(Lo, Ro, Lv, Lv))
    #    assert np.isclose(epart(Lo, Lo, Rv, Lv), epart(Lo, Lo, Lv, Rv))
    #    assert np.isclose(epart(Ro, Lo, Lv, Lv), epart(Lo, Ro, Lv, Lv))

    #    energies.append(  epart(Lo, Lo, Rv, Rv))
    #    energies.append(2*epart(Lo, Ro, Lv, Rv))
    #    energies.append(2*epart(Lo, Ro, Rv, Lv))
    #    energies.append(  epart(Ro, Ro, Lv, Lv))

    #    energies.append(2*epart(Lo, Ro, Rv, Rv))
    #    energies.append(2*epart(Ro, Ro, Lv, Rv))
    #    assert np.isclose(epart(Ro, Lo, Rv, Rv), epart(Lo, Ro, Rv, Rv))
    #    assert np.isclose(epart(Ro, Ro, Rv, Lv), epart(Ro, Ro, Lv, Rv))

    #    energies.append(  epart(Ro, Ro, Rv, Rv))

    #    #e4 = e_aaaa
    #    #e3 = e_aaab + e_aaba + e_abaa + e_baaa
    #    #e2 = 0.5*(e_aabb + e_abab + e_abba + e_bbaa)

    #    with open("energy-parts.txt", "a") as f:
    #        f.write((10*"  %16.8e" + "\n") % tuple(energies))

    def get_local_energy_most_indices_2C(self, cc, C1, C2, eris=None, symmetry_factor=None):

        if symmetry_factor is None:
            symmetry_factor = self.symmetry_factor

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

        if eris is None:
            log.warning("Warning: recomputing AO->MO integral transformation")
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
        ## 2
        #T2 += f2*(  project_T2(Lo, Lo, Rv, Rv)
        #        + 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
        #        + 2*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
        #        +   project_T2(Ro, Ro, Lv, Lv))

        # 2
        T2 +=   project_T2(Lo, Lo, Rv, Rv)
        T2 += 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
        #T2 += 1*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
        #T2 +=   project_T2(Ro, Ro, Lv, Lv)

        e2 = (2*einsum('ijab,iabj', T2, eris.ovvo)
               -einsum('ijab,jabi', T2, eris.ovvo))

        e_loc = symmetry_factor * (e1 + e2)

        return e_loc

    def get_local_energy_most_indices(self, cc, C1, C2, variant=1):

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

        # ONE-ELECTRON
        # ============
        pC1 = einsum("pi,ia,sa->ps", Lo, C1, Lv)
        pC1 += 0.5*einsum("pi,ia,sa->ps", Lo, C1, Rv)
        pC1 += 0.5*einsum("pi,ia,sa->ps", Ro, C1, Lv)

        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * pC1)
        if not np.isclose(e1, 0):
            log.warning("Warning: large E1 component: %.8e" % e1)

        # TWO-ELECTRON
        # ============

        def project_C2_P1(P1):
            pC2 = einsum("pi,ijab->pjab", P1, C2)
            return pC2

        def project_C2(P1, P2, P3, P4):
            pC2 = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
            return pC2

        if variant == 1:

            # QUADRUPLE L
            # ===========
            pC2 = project_C2(Lo, Lo, Lv, Lv)

            # TRIPEL L
            # ========
            pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
            pC2 += 2*project_C2(Lo, Ro, Lv, Lv)

            # DOUBLE L
            # ========
            # P(LLRR) [This wrongly includes: P(LLAA) - correction below]
            pC2 +=   project_C2(Lo, Lo, Rv, Rv)
            pC2 += 2*project_C2(Lo, Ro, Lv, Rv)
            pC2 += 2*project_C2(Lo, Ro, Rv, Lv)
            pC2 +=   project_C2(Ro, Ro, Lv, Lv)

            # SINGLE L
            # ========
            # P(LRRR) [This wrongly includes: P(LAAR) - correction below]
            four_idx_from_occ = False

            if not four_idx_from_occ:
                pC2 += 0.25*2*project_C2(Lo, Ro, Rv, Rv)
                pC2 += 0.25*2*project_C2(Ro, Ro, Lv, Rv)
            else:
                pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)

            # CORRECTIONS
            # ===========
            for x in self.loop_clusters(exclude_self=True):
                Xo, Xv = get_projectors(x.indices)

                # DOUBLE CORRECTION
                # -----------------
                # Correct for wrong inclusion of P(LLAA)
                # The case P(LLAA) was included with prefactor of 1 instead of 1/2
                # We thus need to only correct by "-1/2"
                pC2 -= 0.5*  project_C2(Lo, Lo, Xv, Xv)
                pC2 -= 0.5*2*project_C2(Lo, Xo, Lv, Xv)
                pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Lv)
                pC2 -= 0.5*  project_C2(Xo, Xo, Lv, Lv)

                # SINGLE CORRECTION
                # -----------------
                # Correct for wrong inclusion of P(LAAR)
                # This corrects the case P(LAAB) but overcorrects P(LAAA)!
                if not four_idx_from_occ:
                    pC2 -= 0.25*2*project_C2(Lo, Xo, Xv, Rv)
                    pC2 -= 0.25*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
                    pC2 -= 0.25*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection
                    pC2 -= 0.25*2*project_C2(Xo, Xo, Lv, Rv)
                    pC2 -= 0.25*2*project_C2(Xo, Ro, Lv, Xv) # overcorrection
                    pC2 -= 0.25*2*project_C2(Ro, Xo, Lv, Xv) # overcorrection

                    # Correct overcorrection
                    pC2 += 0.25*2*2*project_C2(Lo, Xo, Xv, Xv)
                    pC2 += 0.25*2*2*project_C2(Xo, Xo, Lv, Xv)

                else:
                    pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)
                    pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
                    pC2 -= 0.5*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection

                    # Correct overcorrection
                    pC2 += 0.5*2*2*project_C2(Lo, Xo, Xv, Xv)

            e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
                   -einsum('ijab,jabi', pC2, eris.ovvo))

        elif variant == 2:
            # QUADRUPLE L
            # ===========
            pC2 = project_C2(Lo, Lo, Lv, Lv)

            # TRIPEL L
            # ========
            pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
            pC2 += 2*project_C2(Lo, Ro, Lv, Lv)

            # DOUBLE L
            # ========
            pC2 +=   project_C2(Lo, Lo, Rv, Rv)
            pC2 +=   2*project_C2(Lo, Ro, Lv, Rv)
            pC2 +=   2*project_C2(Lo, Ro, Rv, Lv)
            for x in self.loop_clusters(exclude_self=True):
                Xo, Xv = get_projectors(x.indices)
                pC2 -= project_C2(Lo, Xo, Lv, Xv)
                pC2 -= project_C2(Lo, Xo, Xv, Lv)

            # SINGLE L
            # ========

            # This wrongly includes LXXX
            pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)
            for x in self.loop_clusters(exclude_self=True):
                Xo, Xv = get_projectors(x.indices)

                pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv)
                pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)

                pC2 += 0.5*2*project_C2(Lo, Xo, Xv, Xv)

            e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
                   -einsum('ijab,jabi', pC2, eris.ovvo))

        elif variant == 3:
            # QUADRUPLE + TRIPLE L
            # ====================
            pC2 = project_C2_P1(Lo)
            pC2 += project_C2(Ro, Lo, Lv, Lv)
            for x in self.loop_clusters(exclude_self=True):
                Xo, Xv = get_projectors(x.indices)
                pC2 -= project_C2(Lo, Xo, Xv, Xv)

            e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
                   -einsum('ijab,jabi', pC2, eris.ovvo))


        e_loc = e1 + e2

        return e_loc

    def get_local_energy(self, cc, C1, C2, project="occupied", project_kind="right", eris=None,
            symmetry_factor=None):

        if symmetry_factor is None:
            symmetry_factor = self.symmetry_factor

        a = cc.get_frozen_mask()
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]

        # Project one index of amplitudes
        if project == "occupied":
            P = self.get_local_energy_projector(C[:,o], kind=project_kind)
            if C1 is not None:
                C1 = einsum("xi,ia->xa", P, C1)
            C2 = einsum("xi,ijab->xjab", P, C2)
        elif project == "virtual":
            P = self.get_local_energy_projector(C[:,v], kind=project_kind)
            if C1 is not None:
                C1 = einsum("xa,ia->ia", P, C1)
            C2 = einsum("xa,ijab->ijxb", P, C2)

        if eris is None:
            log.warning("Warning: recomputing AO->MO integral transformation")
            eris = cc.ao2mo()

        if C1 is not None:
            F = eris.fock[o][:,v]
            e1 = 2*np.sum(F * C1)
            if abs(e1) > 1e-6:
                log.warning("Warning: large E1 component of energy: %.8e" % e1)
        # MP2
        else:
            e1 = 0

        # CC
        if hasattr(eris, "ovvo"):
            eris_ovvo = eris.ovvo
        # MP2
        else:
            no = C2.shape[0]
            nv = C2.shape[2]
            eris_ovvo = eris.ovov.reshape(no,nv,no,nv).transpose(0, 1, 3, 2)

        e2 = 2*einsum('ijab,iabj', C2, eris_ovvo)
        e2 -=  einsum('ijab,jabi', C2, eris_ovvo)

        e_loc = symmetry_factor * (e1 + e2)

        return e_loc

# ===== #

class EmbCC:

    default_options = [
            "solver",
            "bath_type",
            "tol_bath",
            "bath_target_size",
            "tol_dmet_bath",
            #"tol_vno",
            #"vno_ratio",
            "use_ref_orbitals_dmet",
            "use_ref_orbitals_bath"
            ]

    def __init__(self, mf, solver="CCSD", bath_type="power", bath_target_size=(None, None), tol_bath=1e-3,
            tol_dmet_bath=1e-8,
            #tol_vno=1e-3, vno_ratio=None,
            use_ref_orbitals_dmet=True,
            use_ref_orbitals_bath=True,
            #use_ref_orbitals_bath=False,
            benchmark=None):
        self.mf = mf

        # Options
        if solver not in (None, "MP2", "CISD", "CCSD", "FCI"):
            raise ValueError("Unknown solver: %s" % solver)
        if bath_type not in (None, "power", "matsubara", "uncontracted", "mp2-no"):
            raise ValueError()
        self.solver = solver
        self.bath_type = bath_type
        self.tol_bath = tol_bath
        self.bath_target_size = bath_target_size
        self.tol_dmet_bath = tol_dmet_bath
        #self.tol_vno = tol_vno
        #self.vno_ratio = vno_ratio
        self.use_ref_orbitals_dmet = use_ref_orbitals_dmet
        self.use_ref_orbitals_bath = use_ref_orbitals_bath

        # For testing
        self.benchmark = benchmark

        self.clusters = []

    @property
    def mol(self):
        return self.mf.mol

    def get_cluster_attributes(self, attr):
        attrs = {}
        for cluster in self.clusters:
            attrs[cluster.name] = getattr(cluster, attr)
        return attrs

    def set_cluster_attributes(self, attr, values):
        log.debug("Setting attribute %s of all clusters", attr)
        for cluster in self.clusters:
            setattr(cluster, attr, values[cluster.name])


    def get_orbitals(self):
        return self.get_cluster_attributes("orbitals")

        #orbitals = {}
        #for cluster in self.clusters:
        #    orbitals[cluster.name] = cluster.orbitals
        #return orbitals

    def set_reference_orbitals(self, ref_orbitals):
        return self.set_cluster_attributes("ref_orbitals", ref_orbitals)

        #for cluster in self.clusters:
        #    cluster.ref_orbitals = ref_orbitals[cluster.name]

    def make_cluster(self, name, ao_indices, **kwargs):
        for opt in self.default_options:
            kwargs[opt] = kwargs.get(opt, getattr(self, opt))

        #kwargs["solver"] = kwargs.get("solver", self.solver)
        #kwargs["bath_type"] = kwargs.get("bath_type", self.bath_type)
        #kwargs["tol_bath"] = kwargs.get("tol_bath", self.tol_bath)
        #kwargs["tol_dmet_bath"] = kwargs.get("tol_dmet_bath", self.tol_dmet_bath)
        #kwargs["tol_vno"] = kwargs.get("tol_vno", self.tol_vno)
        #kwargs["vno_ratio"] = kwargs.get("vno_ratio", self.vno_ratio)
        #kwargs["use_ref_orbitals_dmet"] = kwargs.get("use_ref_orbitals_dmet", self.use_ref_orbitals_dmet)
        #kwargs["use_ref_orbitals_bath"] = kwargs.get("use_ref_orbitals_bath", self.use_ref_orbitals_bath)

        kwargs["symmetry_factor"] = kwargs.get("symmetry_factor", 1.0)

        cluster = Cluster(self, name, ao_indices, **kwargs)
        # For testing
        cluster.benchmark = self.benchmark
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

    def get_iao_coeff(self, minao="minao"):
        C_occ = self.mf.mo_coeff[:,self.mf.mo_occ>0]
        C_iao = pyscf.lo.iao.iao(self.mol, C_occ, minao=minao)
        niao = C_iao.shape[-1]
        log.debug("Total number of IAOs=%d", niao)

        # Orthogonalize IAO
        S = self.mf.get_ovlp()
        C_iao = pyscf.lo.vec_lowdin(C_iao, S)

        # Add remaining virtual space
        # Transform to MO basis
        C_iao_mo = np.linalg.multi_dot((self.mf.mo_coeff.T, S, C_iao))
        # Get eigenvectors of projector into complement
        P_iao = np.dot(C_iao_mo, C_iao_mo.T)
        norb = self.mf.mo_coeff.shape[-1]
        P_env = np.eye(norb) - P_iao
        e, R = np.linalg.eigh(P_env)
        #log.debug("Eigenvalues of projector into environment:\n%s", e)
        assert np.all(np.logical_or(abs(e) < 1e-10, abs(e)-1 < 1e-10))
        mask = (e > 1e-10)
        assert (np.sum(mask) + niao == norb)
        C_env = R[:,mask]

        C_mo = np.hstack((C_iao_mo, C_env))
        # Rotate back to AO
        C = np.dot(self.mf.mo_coeff, C_mo)
        assert np.allclose(C.T.dot(S).dot(C) - np.eye(norb), 0)

        # Get base atoms of IAOs
        refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
        iao_atoms = [x[0] for x in refmol.ao_labels(None)]
        #log.debug("Base atoms of IAOs: %r", iao_atoms)

        return C, iao_atoms


    def make_iao_atom_clusters(self, minao="minao", **kwargs):
        """Divide intrinsic atomic orbitals into clusters according to their base atom."""

        C, iao_atoms = self.get_iao_coeff()

        self.clear_clusters()
        ncluster = self.mol.natm
        for atomid in range(ncluster):
            iao_indices = np.nonzero(np.isin(iao_atoms, atomid))[0]
            name = self.mol.atom_symbol(atomid)
            c = self.make_cluster(name, iao_indices, coeff=C, local_orbital_type="iao", **kwargs)
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


    def make_custom_atom_cluster(self, atoms, name=None, **kwargs):
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

        for atom in atoms:
            if atom not in ao2atomlbl:
                raise ValueError("Atom %s not in molecule." % atom)

        ao_indices = np.nonzero(np.isin(ao2atomlbl, atoms))[0]
        c = self.make_cluster(name, ao_indices, **kwargs)
        self.clusters.append(c)
        return c

    def make_custom_iao_atom_cluster(self, atoms, name=None, **kwargs):
        """Make custom clusters in terms of atoms..

        Parameters
        ----------
        atoms : iterable
            List of atom symbols for cluster.
        """

        if name is None:
            name = ",".join(atoms)

        C, iao_atoms = self.get_iao_coeff()

        atom_symbols = [self.mol.atom_symbol(atomid) for atomid in iao_atoms]
        log.debug("Atom symbols: %r", atom_symbols)

        for atom in atoms:
            if atom not in atom_symbols:
                raise ValueError("Atom %s not in molecule." % atom)

        iao_indices = np.nonzero(np.isin(atom_symbols, atoms))[0]
        log.debug("IAO indices: %r", iao_indices)
        cluster = self.make_cluster(name, iao_indices, coeff=C, local_orbital_type="iao", **kwargs)

        self.clusters.append(cluster)
        return cluster



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

        # Get options from first cluster
        for opt in self.default_options:
            kwargs[opt] = kwargs.get(opt, getattr(merged[0], opt))
        #kwargs["solver"] = kwargs.get("solver", merged[0].solver)
        #kwargs["bath_type"] = kwargs.get("bath_type", merged[0].bath_type)
        #kwargs["tol_bath"] = kwargs.get("tol_bath", merged[0].tol_bath)
        #kwargs["tol_dmet_bath"] = kwargs.get("tol_dmet_bath", merged[0].tol_dmet_bath)
        #merged_cluster = Cluster(merged_name, self.mf, merged_indices,
        #        tol_dmet_bath=tol_dmet_bath, tol_bath=tol_bath)
        assert np.all([(m.symmetry_factor == merged[0].symmetry_factor) for m in merged])

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

        if clusters[0].local_orbital_type == "iao":
            raise NotImplementedError()

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
        log.info("Total wall time for EmbCCSD: %s", get_time_string(MPI.Wtime()-t_start))

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
        e_corr_v = mpi_reduce("e_corr_v")

        e_corr_dmp2 = mpi_reduce("e_corr_dmp2")
        e_corr_v_dmp2 = mpi_reduce("e_corr_v_dmp2")


        e_corr_var = mpi_reduce("e_corr_var")
        e_corr_var2 = mpi_reduce("e_corr_var2")
        e_corr_var3 = mpi_reduce("e_corr_var3")

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
                c.e_corr_v = e_corr_v[cidx]

                c.e_corr_dmp2 = e_corr_dmp2[cidx]
                c.e_corr_v_dmp2 = e_corr_v_dmp2[cidx]


                c.e_corr_var = e_corr_var[cidx]
                c.e_corr_var2 = e_corr_var2[cidx]
                c.e_corr_var3 = e_corr_var3[cidx]

            #self.e_ccsd = sum(e_ccsd)
            self.e_corr = sum(e_corr)
            self.e_corr_v = sum(e_corr_v)

            self.e_corr_dmp2 = sum(e_corr_dmp2)
            self.e_corr_v_dmp2 = sum(e_corr_v_dmp2)

            self.e_corr_var = sum(e_corr_var)
            self.e_corr_var2 = sum(e_corr_var2)
            self.e_corr_var3 = sum(e_corr_var3)
            #self.e_pt = sum(e_pt)

            #self.e_corr = self.e_ccsd + self.e_pt
            self.e_tot = self.mf.e_tot + self.e_corr
            self.e_tot_v = self.mf.e_tot + self.e_corr_v

            self.e_tot_dmp2 = self.mf.e_tot + self.e_corr_dmp2
            self.e_tot_v_dmp2 = self.mf.e_tot + self.e_corr_v_dmp2

            self.e_tot_var = self.mf.e_tot + self.e_corr_var
            self.e_tot_var2 = self.mf.e_tot + self.e_corr_var2
            self.e_tot_var3 = self.mf.e_tot + self.e_corr_var3

        return np.all(converged)

    def print_cluster_results(self):
        log.info("Energy contributions per cluster")
        log.info("--------------------------------")
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
