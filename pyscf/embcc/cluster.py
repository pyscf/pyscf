# Standard libaries
import os
import os.path
import logging
from collections import OrderedDict
import functools
from datetime import datetime
from timeit import default_timer as timer

# External libaries
import numpy as np
import scipy
import scipy.linalg

# Internal libaries
import pyscf
import pyscf.pbc
from pyscf.pbc.tools import cubegen

# Local modules
from .solver import ClusterSolver
from .util import *
from .dmet_bath import make_dmet_bath, project_ref_orbitals
#from .bath import *
from .mp2_bath import make_mp2_bno
from .energy import *
from . import ccsd_t
from . import helper
from . import psubspace

__all__ = [
        "Cluster",
        ]

log = logging.getLogger(__name__)

class Cluster:

    def __init__(self, base, cluster_id, name,
            #indices, C_local, C_env,
            c_frag, c_env,
            ao_indices=None,
            solver="CCSD",
            # Bath
            bath_type="mp2-natorb", bath_size=None, bath_tol=1e-4, bath_energy_tol=None,
            bno_threshold=None, bno_threshold_factor=1,
            **kwargs):
        """
        Parameters
        ----------
        base : EmbCC
            Base EmbCC object.
        cluster_id : int
            Unique ID of cluster.
        name :
            Name of cluster.
        indices:
            Atomic orbital indices of cluster. [ local_orbital_type == "ao" ]
            Intrinsic atomic orbital indices of cluster. [ local_orbital_type == "iao" ]
        """
        self.base = base
        self.id = cluster_id
        self.name = name
        #self.indices = indices
        self.ao_indices = ao_indices
        msg = "CREATING CLUSTER %d: %s" % (self.id, self.name)
        log.info(msg)
        log.info(len(msg)*"*")
        log.info("  * Local orbital type= %s", self.local_orbital_type)     # depends on self.base

        # NEW: local and environment orbitals
        if c_frag.shape[-1] == 0:
            raise ValueError("No local orbitals in cluster %d:%s" % (self.id, self.name))

        self.c_frag = c_frag
        self.c_env = c_env
        log.info("  * Size= %d", self.size) # depends on self.c_frag

        # Determine number of mean-field electrons in fragment space
        sc = np.dot(self.base.ovlp, self.c_frag)
        dm = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc))
        self.nelec_mf_frag = np.trace(dm)
        log.info("  * Mean-field electrons= %.5f", self.nelec_mf_frag)

        # Options
        if solver not in self.base.VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver
        log.info("  * Solver= %s", self.solver)

        #if not hasattr(bath_size, "__getitem__"):
        if not has_length(bath_size, 2):
            bath_size = (bath_size, bath_size)
        #if not hasattr(bath_tol, "__getitem__"):
        if not has_length(bath_tol, 2):
            bath_tol = (bath_tol, bath_tol)
        #if not hasattr(bath_energy_tol, "__getitem__"):
        if not has_length(bath_energy_tol, 2):
            bath_energy_tol = (bath_energy_tol, bath_energy_tol)

        #assert bath_type in (None, "full", "power", "matsubara", "mp2-natorb")
        if bath_type not in self.base.VALID_BATH_TYPES:
            raise ValueError("Unknown bath type: %s" % bath_type)

        self.bath_type = bath_type
        self.bath_target_size = bath_size
        self.bath_tol = bath_tol
        self.bath_energy_tol = bath_energy_tol

        assert len(self.bath_target_size) == 2
        assert len(self.bath_tol) == 2
        assert len(self.bath_energy_tol) == 2

        # Bath natural orbital (BNO) threshold
        if bno_threshold is None:
            bno_threshold = self.base.bno_threshold
        assert len(bno_threshold) == len(self.base.bno_threshold)
        # Sort so that most expensive calculation comes first (allows projecting ERIs)
        self.bno_threshold = (bno_threshold_factor*np.asarray(bno_threshold)).sort()

        # Other options from kwargs:
        self.solver_options = kwargs.get("solver_options", {})

        self.mp2_correction = kwargs.get("mp2_correction", True)
        # Make MP2 correction tuple for (occupied, virtual) bath
        if not hasattr(self.mp2_correction, "__getitem__"):
            self.mp2_correction = (self.mp2_correction, self.mp2_correction)

        # Bath parameters

        self.coupled_bath = kwargs.get("coupled_bath", False)

        # Add additional orbitals to 0-cluster
        # Add first-order power orbitals
        self.power1_occ_bath_tol = kwargs.get("power1_occ_bath_tol", False)
        self.power1_vir_bath_tol = kwargs.get("power1_vir_bath_tol", False)
        # Add local orbitals:
        self.local_occ_bath_tol = kwargs.get("local_occ_bath_tol", False)
        self.local_vir_bath_tol = kwargs.get("local_vir_bath_tol", False)

        # Other
        self.symmetry_factor = kwargs.get("symmetry_factor", 1.0)
        log.info("  * Symmetry factor= %f", self.symmetry_factor)

        # Restart solver from previous solution [True/False]
        #self.restart_solver = kwargs.get("restart_solver", True)
        self.restart_solver = kwargs.get("restart_solver", False)
        # Parameters needed for restart (C0, C1, C2 for CISD; T1, T2 for CCSD) are saved here
        self.restart_params = kwargs.get("restart_params", {})

        # By default use PBC code if self.mol has attribute "cell"
        #self.use_pbc = kwargs.get("use_pbc", False)
        self.use_pbc = kwargs.get("use_pbc", self.has_pbc)


        self.nelectron_target = kwargs.get("nelectron_target", None)

        self.use_energy_tol_as_delta_mp2 = kwargs.get("use_energy_tol_as_delta_mp2", False)

        #self.make_rdm1 = kwargs.get("make_rdm1", False)     # Calculate RDM1 in cluster?

        opts = Options()
        opts.dmet_bath_tol = self.base.opts.get("dmet_bath_tol", 1e-4)
        opts.bath_tol_per_elec = self.base.opts.get("bath_tol_per_elec", False)
        opts.prim_mp2_bath_tol_occ = self.base.opts.get("prim_mp2_bath_tol_occ", False)
        opts.prim_mp2_bath_tol_vir = self.base.opts.get("prim_mp2_bath_tol_vir", False)
        opts.make_rdm1 = self.base.opts.get("make_rdm1", False)
        opts.eom_ccsd = self.base.opts.get("eom_ccsd", False)
        opts.plot_orbitals = self.base.opts.get("plot_orbitals", False)
        self.opts = opts

        #self.project_type = kwargs.get("project_type", "first-occ")

        if self.opts.plot_orbitals:
            os.makedirs(self.base.opts.plot_orbitals_dir, exist_ok=True)
            name = "%s.cube" % os.path.join(self.base.opts.plot_orbitals_dir, self.name)
            self.cubefile = cubegen.CubeFile(self.mol, filename=name, **self.base.opts.plot_orbitals_kwargs)
            self.cubefile.add_orbital(self.c_frag.copy())

        self.ccsd_t_max_orbitals = kwargs.get("ccsd_t_max_orbitals", 200)

        self.set_default_attributes()

    def set_default_attributes(self):
        """Set default attributes of cluster object."""

        # Orbitals [set when running solver]
        ####self.C_bath = None      # DMET bath orbitals
        ####self.C_occclst = None   # Occupied cluster orbitals
        ####self.C_virclst = None   # Virtual cluster orbitals
        ####self.C_occbath = None   # Occupied bath orbitals
        ####self.C_virbath = None   # Virtual bath orbitals

        self.nactive = 0
        self.nfrozen = 0

        ###self.active = []
        ###self.active_occ = []
        ###self.active_vir = []
        ###self.frozen = []
        ###self.frozen_occ = []
        ###self.frozen_vir = []

        # These are used by the solver
        ##self.mo_coeff = None
        ##self.mo_occ = None
        ##self.eris = None

        # Local amplitudes [C1, C2] can be stored here and
        # used by other fragments
        ##self.amplitudes = {}

        # Orbitals sizes
        #self.nbath0 = 0
        #self.nbath = 0
        #self.nfrozen = 0
        # Calculation results

        # Intermediate values
        self.c_no_occ = self.c_no_vir = None
        self.n_no_occ = self.n_no_vir = None

        # Save correlation energies for different BNO thresholds
        #self.e_corrs = {}
        self.e_corrs = len(self.bno_threshold)*[None]

        self.iteration = 0

        # Output values
        self.converged = False
        #self.e_corr = 0.0
        self.e_delta_mp2 = 0.0
        self.e_pert_t = 0.0
        self.e_corr_dmp2 = 0.0

        # For testing:
        # Full cluster correlation energy
        #self.e_corr_full = 0.0
        #self.e_corr_v = 0.0
        #self.e_corr_d = 0.0
        #self.e_dmet = 0.0

        # For EMO-CCSD
        self.eom_ip_energy = None
        self.eom_ea_energy = None

    #@property
    #def nlocal(self):
    #    """Number of local (fragment) orbitals."""
    #    return self.C_local.shape[-1]

    def trimmed_name(self, length=10):
        if len(self.name) <= length:
            return self.name
        return self.name[:(length-3)] + "..."

    @property
    def size(self):
        """Number of local (fragment) orbitals."""
        return self.c_frag.shape[-1]

    ##@property
    ##def ndmetbath(self):
    ##    """Number of DMET bath orbitals."""
    ##    return self.C_bath.shape[-1]

    ##@property
    ##def noccbath(self):
    ##    """Number of occupied bath orbitals."""
    ##    return self.C_occbath.shape[-1]

    ##@property
    ##def nvirbath(self):
    ##    """Number of virtual bath orbitals."""
    ##    return self.C_virbath.shape[-1]

    ##@property
    ##def nactive(self):
    ##    """Number of active orbitals."""
    ##    return len(self.active)

    ##@property
    ##def nfrozen(self):
    ##    """Number of frozen environment orbitals."""
    ##    return len(self.frozen)

    @property
    def e_corr(self):
        """Best guess for correlation energy, using the lowest BNO threshold."""
        #key = min(list(self.e_corrs.keys()))
        #return self.e_corrs[key]
        idx = np.argmin(self.bno_threshold)
        return self.e_corrs[idx]

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
    def energy_factor(self):
        return self.symmetry_factor

    # Register frunctions of bath.py as methods
    make_dmet_bath = make_dmet_bath
    # OLD
    #project_ref_orbitals = project_ref_orbitals
    #make_bath = make_bath
    #make_local_bath = make_local_bath
    #make_mf_bath = make_mf_bath
    #make_mp2_bath = make_mp2_bath
    #get_mp2_correction = get_mp2_correction
    #transform_mp2_eris = transform_mp2_eris
    #run_mp2 = run_mp2
    #run_mp2_general = run_mp2_general

    # Register frunctions of energy.py as methods
    get_local_amplitudes = get_local_amplitudes
    get_local_amplitudes_general = get_local_amplitudes_general
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
            S = self.base.ovlp
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
            Bath orbitals.

        Returns
        -------
        C_occclst : ndarray
            Occupied cluster orbitals.
        C_virclst : ndarray
            Virtual cluster orbitals.
        """
        C_clst = np.hstack((self.c_frag, C_bath))
        sc = np.dot(self.base.ovlp, C_clst)
        D_clst = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc)) / 2
        e, R = np.linalg.eigh(D_clst)
        tol = self.opts.dmet_bath_tol
        if not np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=tol, rtol=0):
            raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not all close to 0 or 1:\n%s", e)
        e, R = e[::-1], R[:,::-1]
        C_clst = np.dot(C_clst, R)
        nocc_clst = sum(e > 0.5)
        nvir_clst = sum(e < 0.5)
        log.info("Number of cluster orbitals: occ=%3d, vir=%3d", nocc_clst, nvir_clst)

        C_occclst, C_virclst = np.hsplit(C_clst, [nocc_clst])

        return C_occclst, C_virclst

    def canonicalize(self, *mo_coeff, eigenvalues=False):
        """Diagonalize Fock matrix within subspace.

        Parameters
        ----------
        *mo_coeff : ndarrays
            Orbital coefficients.
        eigenvalues : ndarray
            Return MO energies of canonicalized orbitals.

        Returns
        -------
        mo_canon : ndarray
            Canonicalized orbital coefficients.
        rot : ndarray
            Rotation matrix: np.dot(mo_coeff, rot) = mo_canon.
        """
        mo_coeff = np.hstack(mo_coeff)
        fock = np.linalg.multi_dot((mo_coeff.T, self.base.get_fock(), mo_coeff))
        mo_energy, rot = np.linalg.eigh(fock)
        mo_canon = np.dot(mo_coeff, rot)
        if eigenvalues:
            return mo_canon, rot, mo_energy
        return mo_canon, rot

    def get_occup(self, mo_coeff):
        """Get mean-field occupation numbers (diagonal of 1-RDM) of orbitals.

        Parameters
        ----------
        mo_coeff : ndarray, shape(N, M)
            Orbital coefficients.

        Returns
        -------
        occ : ndarray, shape(M)
            Occupation numbers of orbitals.
        """
        sc = np.dot(self.base.ovlp, mo_coeff)
        dm = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc))
        occ = np.diag(dm)
        return occ

    def get_local_projector(self, C, kind="right", inverse=False):
        """Projector for one index of amplitudes local energy expression.

        Parameters
        ----------
        C : ndarray, shape(N, M)
            Occupied or virtual orbital coefficients.
        kind : str ["right", "left", "center"], optional
            Only for AO local orbitals.
        inverse : bool, optional
            If true, return the environment projector 1-P instead.

        Return
        ------
        P : ndarray, shape(M, M)
            Projection matrix.
        """
        S = self.base.ovlp

        # Project onto space of local (fragment) orbitals.
        #if self.local_orbital_type in ("IAO", "LAO"):
        if self.local_orbital_type in ("IAO", "LAO", "PMO"):
            CSC = np.linalg.multi_dot((C.T, S, self.c_frag))
            P = np.dot(CSC, CSC.T)

        # Project onto space of local atomic orbitals.
        #elif self.local_orbital_type in ("AO", "NonOrth-IAO", "PMO"):
        elif self.local_orbital_type in ("AO", "NonOrth-IAO"):
            #l = self.indices
            l = self.ao_indices
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

        if inverse:
            P = (np.eye(P.shape[-1]) - P)

            # DEBUG
            #CSC = np.linalg.multi_dot((C.T, S, self.C_env))
            #P2 = np.dot(CSC, CSC.T)
            #assert np.allclose(P, P2)


        return P

    def project_amplitudes(self, P, T1, T2, indices_T1=None, indices_T2=None, symmetrize_T2=False):
        """Project full amplitudes to local space.

        Parameters
        ----------
        P : ndarray
            Projector.
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

        if symmetrize_T2:
            log.debug("Projected T2 symmetry error = %.3g", np.linalg.norm(pT2 - pT2.transpose(1,0,3,2)))
            pT2 = (pT2 + pT2.transpose(1,0,3,2))/2

        return pT1, pT2

    def transform_amplitudes(self, Ro, Rv, T1, T2):
        if T1 is not None:
            T1 = einsum("xi,ya,ia->xy", Ro, Rv, T1)
        else:
            T1 = None
        T2 = einsum("xi,yj,za,wb,ijab->xyzw", Ro, Ro, Rv, Rv, T2)
        return T1, T2

    def additional_bath_for_cluster(self, c_bath, c_occenv, c_virenv):
        """Add additional bath orbitals to cluster (fragment+DMET bath)."""
        if self.power1_occ_bath_tol is not False:
            c_add, c_occenv, _ = make_mf_bath(self, c_occenv, "occ", bathtype="power",
                    tol=self.power1_occ_bath_tol)
            log.info("Adding %d first-order occupied power bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_add, c_bath))
        if self.power1_vir_bath_tol is not False:
            c_add, c_virenv, _ = make_mf_bath(self, c_virenv, "vir", bathtype="power",
                    tol=self.power1_vir_bath_tol)
            log.info("Adding %d first-order virtual power bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_bath, c_add))
        # Local orbitals:
        if self.local_occ_bath_tol is not False:
            c_add, c_occenv = make_local_bath(self, c_occenv, tol=self.local_occ_bath_tol)
            log.info("Adding %d local occupied bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_add, c_bath))
        if self.local_vir_bath_tol is not False:
            c_add, c_virenv = make_local_bath(self, c_virenv, tol=self.local_vir_bath_tol)
            log.info("Adding %d local virtual bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_bath, c_add))
        return c_bath, c_occenv, c_virenv


    def kernel(self, solver=None, bno_threshold=None):
        """Construct bath orbitals and run solver.

        Parameters
        ----------
        solver : str
            Method ["MP2", "CISD", "CCSD", "CCSD(T)", "FCI"]
        bno_threshold : list
            List of bath natural orbital (BNO) thresholds.
        """

        solver = solver or self.solver
        if bno_threshold is None:
            bno_threshold = self.bno_threshold

        t0_bath = t0 = timer()
        log.info("MAKING DMET BATH")
        log.info("****************")
        log.changeIndentLevel(1)
        c_dmet, c_env_occ, c_env_vir = self.make_dmet_bath(tol=self.opts.dmet_bath_tol)
        log.debug("Time for DMET bath: %s", get_time_string(timer()-t0))
        # Plotting of DMET bath orbitals
        if self.opts.plot_orbitals:
            self.cubefile.add_orbital(c_dmet.copy(), dset_idx=1001)
        log.changeIndentLevel(-1)

        # Add additional orbitals to cluster [optional]
        c_dmet, c_env_occ, c_env_vir = self.additional_bath_for_cluster(c_dmet, c_env_occ, c_env_vir)

        # Diagonalize cluster DM to separate cluster occupied and virtual
        self.c_cluster_occ, self.c_cluster_vir = self.diagonalize_cluster_dm(c_dmet)

        # Add cluster orbitals to plot
        #if self.opts.plot_orbitals:
        #    self.cubefile.add_orbital(C_occclst.copy(), dset_idx=2001)
        #    self.cubefile.add_orbital(C_virclst.copy(), dset_idx=3001)

        # Primary MP2 bath orbitals
        # TODO NOT MAINTAINED
        #if True:
        #    if self.opts.prim_mp2_bath_tol_occ:
        #        log.info("Adding primary occupied MP2 bath orbitals")
        #        C_add_o, C_rest_o, *_ = self.make_mp2_bath(C_occclst, C_virclst, "occ",
        #                c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
        #                mp2_correction=False)
        #    if self.opts.prim_mp2_bath_tol_vir:
        #        log.info("Adding primary virtual MP2 bath orbitals")
        #        C_add_v, C_rest_v, *_ = self.make_mp2_bath(C_occclst, C_virclst, "vir",
        #                c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
        #                mp2_correction=False)
        #    # Combine
        #    if self.opts.prim_mp2_bath_tol_occ:
        #        C_bath = np.hstack((C_add_o, C_bath))
        #        C_occenv = C_rest_o
        #    if self.opts.prim_mp2_bath_tol_vir:
        #        C_bath = np.hstack((C_bath, C_add_v))
        #        C_virenv = C_rest_v

        #    # Re-diagonalize cluster DM to separate cluster occupied and virtual
        #    C_occclst, C_virclst = self.diagonalize_cluster_dm(C_bath)

        #self.C_bath = C_bath

        # === Additional bath orbitals

        log.info("MAKING OCCUPIED BATH")
        log.info("********************")
        t0 = timer()
        log.changeIndentLevel(1)
        #C_occbath, C_occenv2, e_delta_occ, _ = self.make_bath(
        #        C_occenv, self.bath_type, "occ",
        #        # New for MP2 bath:
        #        C_occenv=C_occenv, C_virenv=C_virenv,
        #        nbath=self.bath_target_size[0], tol=self.bath_tol[0])

        self.c_no_occ, self.n_no_occ = make_mp2_bno(
                #self, "occ", np.hstack((self.c_cluster_occ, c_env_occ)), self.c_cluster_vir,
                self, "occ", self.c_cluster_occ, self.c_cluster_vir, c_env_occ, c_env_vir)

        log.debug("Time for occupied %r bath: %s", self.bath_type, get_time_string(timer()-t0))
        log.changeIndentLevel(-1)

        log.info("MAKING VIRTUAL BATH")
        log.info("*******************")
        t0 = timer()
        log.changeIndentLevel(1)
        #C_virbath, C_virenv2, e_delta_vir, _ = self.make_bath(
        #        C_virenv, self.bath_type, "vir",
        #        # New for MP2 bath:
        #        C_occenv=C_occenv, C_virenv=C_virenv,
        #        nbath=self.bath_target_size[1], tol=self.bath_tol[1])

        self.c_no_vir, self.n_no_vir = make_mp2_bno(
                #self, "vir", self.c_cluster_occ, np.hstack((self.c_cluster_vir, c_env_vir)),
                self, "vir", self.c_cluster_occ, self.c_cluster_vir, c_env_occ, c_env_vir)
                #c_occ_frozen=c_env_occ)

        log.debug("Time for virtual %r bath: %s", self.bath_type, get_time_string(timer()-t0))
        log.changeIndentLevel(-1)

        # TODO PLOTS

        # Plot orbitals
        if self.opts.plot_orbitals:
            # Save state of cubefile, in case a replot of the same data is required later:
            self.cubefile.save_state("%s.pkl" % self.cubefile.filename)
            # Write cube-file:
            self.cubefile.write()

        #C_occenv, C_virenv = C_occenv2, C_virenv2
        #self.C_occbath = C_occbath
        #self.C_virbath = C_virbath
        #self.C_occenv = C_occenv
        #self.C_virenv = C_virenv
        log.debug("Wall time for bath: %s", get_time_string(timer()-t0_bath))

        eris = None
        for icalc, bno_thr in enumerate(bno_threshold):
            log.info("RUN %2d - BNO THRESHOLD= %.1e", icalc, bno_thr)
            log.info("*******************************")
            log.changeIndentLevel(1)
            e_corr, eris = self.run_bno_threshold(solver, bno_thr, eris=eris)
            log.info("BNO threshold= %.1e :  E(corr)= %+16.8f Ha", bno_thr, e_corr)
            #self.e_corrs[bno_thr] = e_corr
            self.e_corrs[icalc] = e_corr
            log.changeIndentLevel(-1)


    def run_bno_threshold(self, solver, bno_thr, eris=None):
        #self.e_delta_mp2 = e_delta_occ + e_delta_vir
        #log.debug("MP2 correction = %.8g", self.e_delta_mp2)

        assert (self.c_no_occ is not None)
        assert (self.c_no_vir is not None)

        log.info("Occupied MP2 BNOs:")
        c_nbo_occ, c_frozen_occ = self.apply_bno_threshold(self.c_no_occ, self.n_no_occ, bno_thr)
        log.info("Virtual MP2 BNOs:")
        c_nbo_vir, c_frozen_vir = self.apply_bno_threshold(self.c_no_vir, self.n_no_vir, bno_thr)

        # Canonicalize orbitals
        c_active_occ = self.canonicalize(self.c_cluster_occ, c_nbo_occ)[0]
        c_active_vir = self.canonicalize(self.c_cluster_vir, c_nbo_vir)[0]

        # NEEDED???
        #self.C_occact = C_occact
        #self.C_viract = C_viract

        # Combine, important to keep occupied orbitals first!
        # Put frozen (occenv, virenv) orbitals to the front and back
        # and active orbitals (occact, viract) in the middle
        c_occ = np.hstack((c_frozen_occ, c_active_occ))
        c_vir = np.hstack((c_active_vir, c_frozen_vir))
        nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
        mo_coeff = np.hstack((c_occ, c_vir))

        # Check occupations
        n_occ = self.get_occup(c_occ)
        if not np.allclose(n_occ, 2, atol=2*self.opts.dmet_bath_tol):
            raise RuntimeError("Incorrect occupation of occupied orbitals:\n%r" % n_occ)
        n_vir = self.get_occup(c_vir)
        if not np.allclose(n_vir, 0, atol=2*self.opts.dmet_bath_tol):
            raise RuntimeError("Incorrect occupation of virtual orbitals:\n%r" % n_vir)
        mo_occ = np.asarray(nocc*[2] + nvir*[0])

        # Run solver
        #self.mo_coeff = mo_coeff
        #self.mo_occ = mo_occ
        #self.active = active
        #self.active_occ = active_occ
        #self.active_vir = active_vir
        #self.frozen = frozen
        #self.frozen_occ = frozen_occ
        #self.frozen_vir = frozen_vir

        nocc_frozen = c_frozen_occ.shape[-1]
        nvir_frozen = c_frozen_vir.shape[-1]
        nfrozen = nocc_frozen + nvir_frozen
        nactive = c_active_occ.shape[-1] + c_active_vir.shape[-1]

        log.info("ORBITALS")
        log.info("********")
        log.info("  * Occupied: active= %4d  frozen= %4d  total= %4d", c_active_occ.shape[-1], nocc_frozen, c_occ.shape[-1])
        log.info("  * Virtual:  active= %4d  frozen= %4d  total= %4d", c_active_vir.shape[-1], nvir_frozen, c_vir.shape[-1])
        log.info("  * Total:    active= %4d  frozen= %4d  total= %4d", nactive, nfrozen, mo_coeff.shape[-1])

        log.info("RUNNING %s SOLVER", solver)
        log.info((len(solver)+15)*"*")

        # If superspace ERIs were calculated before, they can be transformed and used again
        if eris is not None:
            log.debug("Projecting previous ERIs onto subspace")
            t0 = timer()
            eris = psubspace.project_eris(eris, c_active_occ, c_active_vir, ovlp=self.base.ovlp)
            log.debug("Time to project ERIs: %s", get_time_string(timer()-t0))

        # Create solver object
        t0 = timer()
        csolver = ClusterSolver(self, solver, mo_coeff, mo_occ, nocc_frozen=nocc_frozen, nvir_frozen=nvir_frozen, eris=eris)
        csolver.kernel()
        self.converged = csolver.converged
        self.e_corr_full = csolver.e_corr
        eris = csolver._eris
        log.debug("Time for %s solver: %s", csolver.solver, get_time_string(timer()-t0))

        pc1, pc2 = self.get_local_amplitudes(csolver._solver, csolver.c1, csolver.c2)
        e_corr = self.get_local_energy(csolver._solver, pc1, pc2, eris=csolver._eris)
        # Population analysis
        if self.opts.make_rdm1 and csolver.dm1 is not None:
            try:
                self.pop_analysis(csolver.dm1)
            except Exception as e:
                log.error("ERROR in population analysis: %s", e)
        # EOM analysis
        if self.opts.eom_ccsd in (True, "IP"):
            self.eom_ip_energy, _ = self.eom_analysis(csolver, "IP")
        if self.opts.eom_ccsd in (True, "EA"):
            self.eom_ea_energy, _ = self.eom_analysis(csolver, "EA")

        return e_corr, eris

    def apply_bno_threshold(self, c_no, n_no, bno_thr):
        """Split natural orbitals (NO) into bath and rest."""
        n_bno = sum(n_no >= bno_thr)
        n_rest = len(n_no)-n_bno
        n_in, n_cut = np.split(n_no, [n_bno])
        # Logging
        fmt = "  %4s: N= %4d  max= %8.3g  min= %8.3g  sum= %8.3g ( %6.2f %%)"
        if n_bno > 0:
            log.info(fmt, "Bath", n_bno, max(n_in), min(n_in), np.sum(n_in), 100*np.sum(n_in)/np.sum(n_no))
        else:
            log.info(fmt[:13], "Bath", 0)
        if n_rest > 0:
            log.info(fmt, "Rest", n_rest, max(n_cut), min(n_cut), np.sum(n_cut), 100*np.sum(n_cut)/np.sum(n_no))
        else:
            log.info(fmt[:13], "Rest", 0)

        c_bno, c_rest = np.hsplit(c_no, [n_bno])
        return c_bno, c_rest


    def pop_analysis(self, dm1, filename=None, mode="a", sig_tol=0.01):
        """Perform population analsis for the given density-matrix and compare to the MF."""
        if filename is None:
            filename = "%s-%s.txt" % (self.base.opts.popfile, self.name)

        sc = np.dot(self.base.ovlp, self.base.lo)
        dm1 = np.linalg.multi_dot((sc.T, dm1, sc))
        pop, chg = self.mf.mulliken_pop(dm=dm1, s=np.eye(dm1.shape[-1]))
        pop_mf = self.base.pop_mf
        chg_mf = self.base.pop_mf_chg

        tstamp = datetime.now()

        log.info("[%s] Writing cluster population analysis to file \"%s\"", tstamp, filename)
        with open(filename, mode) as f:
            f.write("[%s] Population analysis\n" % tstamp)
            f.write("*%s*********************\n" % (26*"*"))

            # per orbital
            for i, s in enumerate(self.mf.mol.ao_labels()):
                dmf = (pop[i]-pop_mf[i])
                sig = (" !" if abs(dmf)>=sig_tol else "")
                f.write("  orb= %4d %-16s occ= %10.5f dHF= %+10.5f%s\n" %
                        (i, s, pop[i], dmf, sig))
            # Charge per atom
            f.write("[%s] Atomic charges\n" % tstamp)
            f.write("*%s****************\n" % (26*"*"))
            for ia in range(self.mf.mol.natm):
                symb = self.mf.mol.atom_symbol(ia)
                dmf = (chg[ia]-chg_mf[ia])
                sig = (" !" if abs(dmf)>=sig_tol else "")
                f.write("  atom= %3d %-3s charge= %10.5f dHF= %+10.5f%s\n" %
                        (ia, symb, chg[ia], dmf, sig))

        return pop, chg

    def eom_analysis(self, csolver, kind, filename=None, mode="a", sort_weight=True, r1_min=1e-2):
        kind = kind.upper()
        assert kind in ("IP", "EA")

        if filename is None:
            filename = "%s-%s.txt" % (self.base.opts.eomfile, self.name)

        sc = np.dot(self.base.ovlp, self.base.lo)
        if kind == "IP":
            e, c = csolver.ip_energy, csolver.ip_coeff
        else:
            e, c = csolver.ea_energy, csolver.ea_coeff
        nroots = len(e)
        eris = csolver._eris
        cc = csolver._solver

        log.info("EOM-CCSD %s energies= %r", kind, e[:5].tolist())
        tstamp = datetime.now()
        log.info("[%s] Writing detailed cluster %s-EOM analysis to file \"%s\"", tstamp, kind, filename)

        with open(filename, mode) as f:
            f.write("[%s] %s-EOM analysis\n" % (tstamp, kind))
            f.write("*%s*****************\n" % (26*"*"))

            for root in range(nroots):
                r1 = c[root][:cc.nocc]
                qp = np.linalg.norm(r1)**2
                f.write("  %s-EOM-CCSD root= %2d , energy= %+16.8g , QP-weight= %10.5g\n" %
                        (kind, root, e[root], qp))
                if qp < 0.0 or qp > 1.0:
                    log.error("Error: QP-weight not between 0 and 1!")
                r1lo = einsum("i,ai,al->l", r1, eris.mo_coeff[:,:cc.nocc], sc)

                if sort_weight:
                    order = np.argsort(-r1lo**2)
                    for ao, lab in enumerate(np.asarray(self.mf.mol.ao_labels())[order]):
                        wgt = r1lo[order][ao]**2
                        if wgt < r1_min*qp:
                            break
                        f.write("  * Weight of %s root %2d on OrthAO %-16s = %10.5f\n" %
                                (kind, root, lab, wgt))
                else:
                    for ao, lab in enumerate(ao_labels):
                        wgt = r1lo[ao]**2
                        if wgt < r1_min*qp:
                            continue
                        f.write("  * Weight of %s root %2d on OrthAO %-16s = %10.5f\n" %
                                (kind, root, lab, wgt))

        return e, c
