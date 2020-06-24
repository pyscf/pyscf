import logging

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
import pyscf.mp

import pyscf.pbc
import pyscf.pbc.cc
import pyscf.pbc.mp
import pyscf.pbc.tools

#from .orbitals import Orbitals
from .util import *
from .bath import *
from .energy import *

__all__ = [
        "Cluster",
        ]

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

class Cluster:

    def __init__(self, base, name, indices, C_local, C_env, coeff=None, solver="CCSD",
            bath_type="power", bath_size=None, bath_tol=1e-4,
            **kwargs):
        """
        Parameters
        ----------
        base : EmbCC
            Base EmbCC object.
        name :
            Name of cluster.
        indices:
            Atomic orbital indices of cluster. [ local_orbital_type == "ao" ]
            Intrinsic atomic orbital indices of cluster. [ local_orbital_type == "iao" ]
        """


        self.base = base
        log.debug("Making cluster with local orbital type %s", self.local_orbital_type)
        self.name = name
        self.indices = indices

        # NEW: local and environment orbitals
        self.C_local = C_local
        self.C_env = C_env

        assert self.nlocal == len(self.indices)

        #self.nlocal = len(self.indices)

        # Optional
        assert solver in ("MP2", "CISD", "CCSD", "FCI")
        self.solver = solver

        if not hasattr(bath_tol, "__getitem__"):
            bath_tol = (bath_tol, bath_tol)
        if not hasattr(bath_size, "__getitem__"):
            bath_size = (bath_size, bath_size)

        self.bath_type = bath_type
        self.bath_target_size = bath_size
        self.bath_tol = bath_tol

        self.mp2_correction = kwargs.get("mp2_correction", True)

        self.use_ref_orbitals_dmet = kwargs.get("use_ref_orbitals_dmet", True)
        self.use_ref_orbitals_bath = kwargs.get("use_ref_orbitals_bath", True)

        self.symmetry_factor = kwargs.get("symmetry_factor", 1.0)

        # Restart solver from previous solution [True/False]
        #self.restart_solver = kwargs.get("restart_solver", True)
        self.restart_solver = kwargs.get("restart_solver", False)
        # Parameters needed for restart (C0, C1, C2 for CISD; T1, T2 for CCSD) are saved here
        self.restart_params = kwargs.get("restart_params", {})

        # Maximum number of iterations for consistency of amplitudes between clusters
        self.maxiter = kwargs.get("maxiter", 1)

        self.set_default_attributes()

    def set_default_attributes(self):
        """Set default attributes of cluster object."""

        # Orbitals [set when running solver]
        self.C_bath = None      # DMET bath orbitals
        self.C_occclst = None   # Occupied cluster orbitals
        self.C_virclst = None   # Virtual cluster orbitals
        self.C_occbath = None   # Occupied bath orbitals
        self.C_virbath = None   # Virtual bath orbitals
        #self.nfrozen = None
        self.e_delta_mp2 = 0.0

        # These are used by the solver
        self.mo_coeff = None
        self.mo_occ = None
        self.frozen = None
        self.eris = None

        #self.ref_orbitals = None

        # Reference orbitals should be saved with keys
        # dmet-bath, occ-bath, vir-bath
        self.ref_orbitals = {}

        # Orbitals sizes
        #self.nbath0 = 0
        #self.nbath = 0
        #self.nfrozen = 0
        # Calculation results

        self.iteration = 0

        self.converged = False
        self.e_corr = 0.0
        self.e_corr_full = 0.0
        self.e_corr_dmp2 = 0.0


    @property
    def nlocal(self):
        """Number of local (fragment) orbitals."""
        return self.C_local.shape[-1]

    @property
    def ndmetbath(self):
        """Number of DMET bath orbitals."""
        return self.C_bath.shape[-1]

    @property
    def noccbath(self):
        """Number of occupied bath orbitals."""
        return self.C_occbath.shape[-1]

    @property
    def nvirbath(self):
        """Number of virtual bath orbitals."""
        return self.C_virbath.shape[-1]

    @property
    def nfrozen(self):
        """Number of frozen environment orbitals."""
        return len(self.frozen)

    def get_orbitals(self):
        """Get dictionary with orbital coefficients."""
        orbitals = {
                "local" : self.C_local,
                "dmet-bath" : self.C_bath,
                "occ-bath" : self.C_occbath,
                "vir-bath" : self.C_virbath,
                }
        return orbitals

    def reset(self, keep_ref_orbitals=True):
        """Reset cluster object. By default it stores the previous orbitals, so they can be used
        as reference orbitals for a new calculation of different geometry."""
        ref_orbitals = self.get_orbitals()
        self.set_default_attributes()
        if keep_ref_orbitals:
            self.ref_orbitals = ref_orbitals

    #def __len__(self):
    #    """The number of local ("imurity") orbitals of the cluster."""
    #    return len(self.indices)

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
    def has_pbc(self):
        return isinstance(self.mol, pyscf.pbc.gto.Cell)

    @property
    def local_orbital_type(self):
        return self.base.local_orbital_type

    @property
    def not_indices(self):
        """Indices which are NOT in the cluster, i.e. complement to self.indices."""
        return np.asarray([i for i in np.arange(self.mol.nao_nr()) if i not in self.indices])

    #def make_projector(self):
    #    """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
    #    S1 = self.mf.get_ovlp()
    #    nao = self.mol.nao_nr()
    #    S2 = S1[np.ix_(self.indices, self.indices)]
    #    S21 = S1[self.indices]
    #    #s2_inv = np.linalg.inv(s2)
    #    #p_21 = np.dot(s2_inv, s21)
    #    # Better: solve with Cholesky decomposition
    #    # Solve: S2 * p_21 = S21 for p_21
    #    p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
    #    p_12 = np.eye(nao)[:,self.indices]
    #    p = np.dot(p_12, p_21)
    #    return p

    #def make_projector_s121(self, indices=None):
    #    """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
    #    if indices is None:
    #        indices = self.indices
    #    S1 = self.mf.get_ovlp()
    #    nao = self.mol.nao_nr()
    #    S2 = S1[np.ix_(indices, indices)]
    #    S21 = S1[indices]
    #    #s2_inv = np.linalg.inv(s2)
    #    #p_21 = np.dot(s2_inv, s21)
    #    # Better: solve with Cholesky decomposition
    #    # Solve: S2 * p_21 = S21 for p_21
    #    p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
    #    #p_12 = np.eye(nao)[:,self.indices]
    #    p = np.dot(S21.T, p_21)
    #    return p

    # Methods from bath.py
    project_ref_orbitals = project_ref_orbitals
    make_dmet_bath = make_dmet_bath
    make_bath = make_bath
    make_mf_bath = make_mf_bath
    make_mp2_bath = make_mp2_bath
    get_mp2_correction = get_mp2_correction
    transform_mp2_eris = transform_mp2_eris 
    run_mp2 = run_mp2

    # Methods from energy.py
    get_local_energy = get_local_energy

    def analyze_orbitals(self, orbitals=None, sort=True):
        if self.local_orbital_type == "iao":
            raise NotImplementedError()


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


    def diagonalize_cluster_dm(self, C_bath):
        """Diagonalize cluster DM to get fully occupied/virtual orbitals

        Parameters
        ----------
        C_bath : ndarray
            DMET bath orbitals.

        Returns
        -------
        C_occclst : ndarray
            Occupied cluster orbitals.
        C_virclst : ndarray
            Virtual cluster orbitals.
        """
        S = self.mf.get_ovlp()
        C_clst = np.hstack((self.C_local, C_bath))
        D_clst = np.linalg.multi_dot((C_clst.T, S, self.mf.make_rdm1(), S, C_clst)) / 2
        e, R = np.linalg.eigh(D_clst)
        if not np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=1e-6, rtol=0):
            raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not all close to 0 or 1:\n%s", e)
        e, R = e[::-1], R[:,::-1]
        C_clst = np.dot(C_clst, R)
        nocc_clst = sum(e > 0.5)
        nvir_clst = sum(e < 0.5)
        log.info("DMET cluster orbitals: occupied=%3d, virtual=%3d", nocc_clst, nvir_clst)

        C_occclst, C_virclst = np.hsplit(C_clst, [nocc_clst])

        return C_occclst, C_virclst

    def canonicalize(self, *C):
        """Diagonalize Fock matrix within subspace.

        Parameters
        ----------
        *C : ndarrays
            Orbital coefficients.

        Returns
        -------
        C : ndarray
            Canonicalized orbital coefficients.
        """
        C = np.hstack(C)
        F = np.linalg.multi_dot((C.T, self.mf.get_fock(), C))
        E, R = np.linalg.eigh(F)
        C = np.dot(C, R)
        return C

    def get_occup(self, C):
        """Get mean-field occupation of orbitals."""
        S = self.mf.get_ovlp()
        D = np.linalg.multi_dot((C.T, S, self.mf.make_rdm1(), S, C))
        occ = np.diag(D)
        return occ

    def get_local_projector(self, C, kind="right", inverse=False):
        """Projector for local energy expression."""
        #log.debug("Making local energy projector for orbital type %s", self.local_orbital_type)
        S = self.mf.get_ovlp()

        # Project onto space of local atomic orbitals.
        if self.local_orbital_type == "AO":
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

        # Project onto space of local (fragment) orbitals.
        elif self.local_orbital_type == "IAO":
            CSC = np.linalg.multi_dot((C.T, S, self.C_local))
            P = np.dot(CSC, CSC.T)

        if inverse:
            log.debug("Inverting projector")
            P = (np.eye(P.shape[-1]) - P)

        return P

    #def project_amplitudes(self, C, T1, T2, indices_T1=None, indices_T2=None, symmetrize_T2=False, **kwargs):
    def project_amplitudes(self, P, T1, T2, indices_T1=None, indices_T2=None, symmetrize_T2=False):
        """Project full amplitudes to local space.

        Parameters
        ----------
        C : ndarray
            Molecular orbital coefficients of T amplitudes.
        T1 : ndarray
            C1/T1 amplitudes.
        T2 : ndarray
            C2/T2 amplitudes.

        Returns
        -------
        pT1 : ndarray
            Projected C1/T1 amplitudes
        pT2 : ndarray
            Projected C2/T2 amplitudes
        """
        if indices_T1 is None:
            indices_T1 = [0]
        if indices_T2 is None:
            indices_T2 = [0]

        #P = self.get_local_projector(C, **kwargs)
        #log.debug("Project amplitudes shapes: P=%r, T1=%r, T2=%r", P.shape, T1.shape, T2.shape)

        # T1 amplitudes
        assert indices_T1 == [0]
        if T1 is not None:
            pT1 = einsum("xi,ia->xa", P, T1)
        else:
            pT1 = None

        # T2 amplitudes
        if indices_T2 == [0]:
            pT2 = einsum("xi,ijab->xjab", P, T2)
        elif indices_T2 == [1]:
            pT2 = einsum("xj,ijab->ixab", P, T2)
        elif indices_T2 == [0, 1]:
            pT2 = einsum("xi,yj,ijab->xyab", P, P, T2)

        #log.debug("Projected T2 symmetry error = %.3g", np.linalg.norm(pT2 - pT2.transpose(1,0,3,2)))

        if symmetrize_T2:
            raise NotImplementedError()

        return pT1, pT2

    def transform_amplitudes(self, Ro, Rv, T1, T2):
        if T1 is not None:
            T1 = einsum("xi,ya,ia->xy", Ro, Rv, T1)
        else:
            T1 = None
        T2 = einsum("xi,yj,za,wb,ijab->xyzw", Ro, Ro, Rv, Rv, T2)
        return T1, T2

    def run_solver(self, solver=None, mo_coeff=None, mo_occ=None, frozen=None, eris=None):
        solver = solver or self.solver
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        if frozen is None: frozen = self.frozen
        if eris is None: eris = self.eris

        self.iteration += 1
        if self.iteration > 1:
            log.debug("Iteration %d", self.iteration)

        log.debug("Running solver %s for cluster %s on MPI process %d", solver, self.name, MPI_rank)

        if self.has_pbc:
            log.debug("Cell object found -> using pbc code.")

        if solver == "MP2":
            if self.has_pbc:
                mp2 = pyscf.pbc.mp.MP2(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
            else:
                mp2 = pyscf.mp.MP2(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
            if eris is None:
                t0 = MPI.Wtime()
                eris = mp2.ao2mo()
                log.debug("Time for integral transformation: %s", get_time_string(MPI.Wtime()-t0))
            e_corr_full, t2 = mp2.kernel(eris=eris)
            converged = True
            e_corr_full *= self.symmetry_factor
            C1, C2 = None, t2

            e_corr = self.get_local_energy(mp2, C1, C2, eris=eris)

        elif solver == "CCSD":
            if self.has_pbc:
                cc = pyscf.pbc.cc.CCSD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
            else:
                cc = pyscf.cc.CCSD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)

            # We want to reuse the integral for local energy
            if eris is None:
                t0 = MPI.Wtime()
                eris = cc.ao2mo()
                log.debug("Time for integral transformation: %s", get_time_string(MPI.Wtime()-t0))
            cc.max_cycle = 100

            # Taylored CC in iterations > 1
            if self.base.tccT1 is not None:
                log.debug("Adding tailorfunc for tailored CC.")

                tcc_mix_factor = 1

                # Transform to cluster basis
                act = cc.get_frozen_mask()
                Co = mo_coeff[:,act][:,mo_occ[act]>0]
                Cv = mo_coeff[:,act][:,mo_occ[act]==0]
                Cmfo = self.mf.mo_coeff[:,self.mf.mo_occ>0]
                Cmfv = self.mf.mo_coeff[:,self.mf.mo_occ==0]
                S = self.mf.get_ovlp()
                Ro = np.linalg.multi_dot((Co.T, S, Cmfo))
                Rv = np.linalg.multi_dot((Cv.T, S, Cmfv))
                ttcT1, ttcT2 = self.transform_amplitudes(Ro, Rv, self.base.tccT1, self.base.tccT2)

                # Get occupied bath projector
                Pbath = self.get_local_projector(Co, inverse=True)
                #Pbath2 = self.get_local_projector(Co)
                #log.debug("%r", Pbath)
                #log.debug("%r", Pbath2)
                #1/0
                #CSC = np.linalg.multi_dot((Co.T, S, self.C_env))
                #Pbath2 = np.dot(CSC, CSC.T)
                #assert np.allclose(Pbath, Pbath2)

                #CSC = np.linalg.multi_dot((Co.T, S, np.hstack((self.C_occclst, self.C_occbath))))
                CSC = np.linalg.multi_dot((Co.T, S, np.hstack((self.C_bath, self.C_occbath))))
                Pbath2 = np.dot(CSC, CSC.T)
                assert np.allclose(Pbath, Pbath2)

                #log.debug("DIFF %g", np.linalg.norm(Pbath - Pbath2))
                #log.debug("DIFF %g", np.linalg.norm(Pbath + Pbath2 - np.eye(Pbath.shape[-1])))

                def tailorfunc(T1, T2):
                    # Difference of bath to local amplitudes
                    dT1 = ttcT1 - T1
                    dT2 = ttcT2 - T2

                    log.debug("Norm of dT1=%.3g, dT2=%.3g", np.linalg.norm(dT1), np.linalg.norm(dT2))
                    # Project difference amplitudes to bath-bath block in occupied indices
                    #pT1, pT2 = self.project_amplitudes(Co, dT1, dT2, indices_T2=[0, 1])
                    pT1, pT2 = self.project_amplitudes(Pbath, dT1, dT2, indices_T2=[0, 1])
                    _, pT2_0 = self.project_amplitudes(Pbath, None, dT2, indices_T2=[0])
                    _, pT2_1 = self.project_amplitudes(Pbath, None, dT2, indices_T2=[1])
                    pT2 += (pT2_0 + pT2_1)/2

                    log.debug("Norm of pT1=%.3g, pT2=%.3g", np.linalg.norm(pT1), np.linalg.norm(pT2))
                    # Add projected difference amplitudes
                    T1 += tcc_mix_factor*pT1
                    T2 += tcc_mix_factor*pT2
                    return T1, T2

                cc.tailorfunc = tailorfunc


            if self.restart_solver:
                log.debug("Running CCSD starting with parameters for: %r...", self.restart_params.keys())
                cc.kernel(eris=eris, **self.restart_params)
            else:
                log.debug("Running CCSD...")
                cc.kernel(eris=eris)
            log.debug("CCSD done. converged: %r", cc.converged)
            if self.restart_solver:
                self.restart_params["t1"] = cc.t1
                self.restart_params["t2"] = cc.t2

            converged = cc.converged
            e_corr_full = self.symmetry_factor*cc.e_corr
            C1 = cc.t1
            C2 = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)

            e_corr = self.get_local_energy(cc, C1, C2, eris=eris)

            # TESTING: Get global amplitudes:
            #if False:
            #if True:
            if self.maxiter > 1:
                if self.base.T1 is None:
                    No = sum(self.mf.mo_occ > 0)
                    Nv = len(self.mf.mo_occ) - No
                    self.base.T1 = np.zeros((No, Nv))
                    self.base.T2 = np.zeros((No, No, Nv, Nv))

                act = cc.get_frozen_mask()
                occ = cc.mo_occ[act] > 0
                vir = cc.mo_occ[act] == 0
                # Projector to local, occupied region
                S = self.mf.get_ovlp()
                Co = cc.mo_coeff[:,act][:,occ]
                Cv = cc.mo_coeff[:,act][:,vir]

                P = self.get_local_projector(Co)
                #pT1, pT2 = self.project_amplitudes(Co, cc.t1, cc.t2)
                pT1, pT2 = self.project_amplitudes(P, cc.t1, cc.t2)

                # Transform to HF MO basis
                Cmfo = self.mf.mo_coeff[:,self.mf.mo_occ>0]
                Cmfv = self.mf.mo_coeff[:,self.mf.mo_occ==0]
                Ro = np.linalg.multi_dot((Cmfo.T, S, Co))
                Rv = np.linalg.multi_dot((Cmfv.T, S, Cv))
                pT1, pT2 = self.transform_amplitudes(Ro, Rv, pT1, pT2)

                # Restore symmetry?
                pT2sym = (pT2 + pT2.transpose(1,0,3,2))/2.0
                log.debug("T2 symmetry error = %.3g", np.linalg.norm(pT2 - pT2.transpose(1,0,3,2)))
                log.debug("T2 symmetry error = %.3g", np.linalg.norm(pT2sym - pT2sym.transpose(1,0,3,2)))

                # ??
                pT2 = pT2sym

                # Alternative to?
                #pT1a, pT2a00 = self.project_amplitudes(Co, cc.t1, cc.t2, indices_T2=[0,1])
                #_, pT2a01 = self.project_amplitudes(Co, None, cc.t2, indices_T2=[0])
                #_, pT2a10 = self.project_amplitudes(Co, None, cc.t2, indices_T2=[1])
                ##pT2a = pT2a00 + (pT2a01+pT2a10)/2
                #pT2a = (pT2a01+pT2a10)/2 #- pT2a00
                #pT1a, pT2a = self.transform_amplitudes(Ro, Rv, pT1a, pT2a)
                #log.debug("T2 symmetry error = %.3g", np.linalg.norm(pT2a - pT2a.transpose(1,0,3,2)))
                #assert np.allclose(pT1a, pT1)
                #assert np.allclose(pT2a, pT2sym)
                #1/0

                self.base.T1 += pT1
                self.base.T2 += pT2

        elif solver == "CISD":
            # Currently not maintained
            raise NotImplementedError()

            cc = pyscf.ci.CISD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
            cisd.max_cycle = 100
            log.debug("Running CISD...")
            cc.kernel()
            log.debug("CISD done. converged: %r", cc.converged)
            converged = cc.converged
            e_corr_full = cc.e_corr
            # Intermediate normalization
            C0, C1, C2 = cc.cisdvec_to_amplitudes(cc.ci)
            renorm = 1/C0
            C1 *= renorm
            C2 *= renorm

            e_corr = self.get_local_energy(cc, C1, C2, eris=eris)

        elif solver == "FCI":
            # Currently not maintained
            raise NotImplementedError()

            nactive = mo_coeff.shape[-1] - len(frozen)
            #nocc_active

            casci = pyscf.mcscf.CASCI(self.mol, nactive, 2*nocc_active)
            casci.canonicalization = False
            mo_coeff_cas = pyscf.mcscf.addons.sort_mo(casci, mo_coeff=mo_coeff, caslst=active, base=0)
            log.debug("Running FCI...")
            e_tot, e_cas, wf, mo_coeff, mo_energy = casci.kernel(mo_coeff=mo_coeff_cas)
            log.debug("FCI done. converged: %r", casci.converged)
            assert np.allclose(mo_coeff, mo_coeff_cas)
            cisdvec = pyscf.ci.cisd.from_fcivec(wf, nactive, 2*nocc_active)
            C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, nactive, nocc_active)
            # Intermediate normalization
            renorm = 1/C0
            C1 *= renorm
            C2 *= renorm

            self.converged = casci.converged
            self.e_corr_full = e_tot - self.mf.e_tot

            # Create fake CISD object
            cisd = pyscf.ci.CISD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
            self.e_fci = self.get_local_energy(cisd, C1, C2)
            self.e_corr = self.e_fci

        else:
            raise ValueError("Unknown solver: %s" % solver)

        # Store integrals for iterations
        if self.maxiter > 1:
            self.eris = eris

        log.debug("Full cluster correlation energy=%.8g", e_corr_full)

        self.converged = converged
        if self.e_corr != 0.0:
            log.debug("dE=%.8g", (e_corr-self.e_corr))
        self.e_corr = e_corr
        self.e_corr_dmp2 = e_corr + self.e_delta_mp2

        return converged, e_corr

    def run(self, solver=None, ref_orbitals=None):
        """Construct bath orbitals and run solver.

        Paramters
        ---------
        solver : str
            Method ["MP2", "CISD", "CCSD", "FCI"]
        ref_orbitals : dict
            Dictionary with reference orbitals.

        Returns
        -------
        converged : bool
        """

        solver = solver or self.solver
        # Orbitals from a reference calaculation (e.g. different geometry)
        # Used for recovery of orbitals via active transformation
        ref_orbitals = ref_orbitals or self.ref_orbitals

        # === Make DMET bath orbital and diagonalize DM in cluster space
        C_bath, C_occenv, C_virenv = self.make_dmet_bath(C_ref=ref_orbitals.get("dmet-bath", None))
        C_occclst, C_virclst = self.diagonalize_cluster_dm(C_bath)

        self.C_bath = C_bath
        self.C_occclst = C_occclst
        self.C_virclst = C_virclst

        # === Additional bath orbitals
        C_occbath, C_occenv, e_delta_occ = self.make_bath(C_occenv, self.bath_type, "occ",
                ref_orbitals.get("occ-bath", None), nbath=self.bath_target_size[0], tol=self.bath_tol[0])
        C_virbath, C_virenv, e_delta_vir = self.make_bath(C_virenv, self.bath_type, "vir",
                ref_orbitals.get("vir-bath", None), nbath=self.bath_target_size[1], tol=self.bath_tol[1])
        self.C_occbath = C_occbath
        self.C_virbath = C_virbath
        self.e_delta_mp2 = e_delta_occ + e_delta_vir
        log.debug("MP2 correction = %.8g", self.e_delta_mp2)

        # === Canonicalize orbitals
        C_occact = self.canonicalize(C_occclst, C_occbath)
        C_viract = self.canonicalize(C_virclst, C_virbath)

        # Combine, important to keep occupied orbitals first
        # Put frozen orbitals to the front and back
        Co = np.hstack((C_occenv, C_occact))
        Cv = np.hstack((C_viract, C_virenv))
        mo_coeff = np.hstack((Co, Cv))
        No = Co.shape[-1]
        Nv = Cv.shape[-1]
        # Check occupations
        assert np.allclose(self.get_occup(Co), 2)
        assert np.allclose(self.get_occup(Cv), 0)
        mo_occ = np.asarray(No*[2] + Nv*[0])
        frozen_occ = list(range(C_occenv.shape[-1]))
        frozen_vir = list(range(Co.shape[-1]+C_viract.shape[-1], mo_coeff.shape[-1]))
        frozen = frozen_occ + frozen_vir
        # Run solver
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.frozen = frozen
        log.debug("Frozen orbitals=%d", self.nfrozen)

        t0 = MPI.Wtime()
        converged, e_corr = self.run_solver(solver, mo_coeff, mo_occ, frozen)
        log.debug("Wall time for solver: %s", get_time_string(MPI.Wtime()-t0))

        #self.converged = converged
        #self.e_corr = e_corr
        #self.e_corr_dmp2 = e_corr + e_delta_mp2

        return converged, e_corr
