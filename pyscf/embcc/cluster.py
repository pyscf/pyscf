# Standard libaries
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

# Local modules
from .solver import ClusterSolver
from .util import *
from .bath import *
from .energy import *
from . import ccsd_t
#from . import pbc_gdf_ao2mo
from . import ao2mo_j3c
#from .embcc import VALID_SOLVERS

__all__ = [
        "Cluster",
        ]

log = logging.getLogger(__name__)

class Cluster:

    def __init__(self, base, cluster_id, name,
            #indices, C_local, C_env,
            C_local, C_env,
            ao_indices=None,
            solver="CCSD",
            # Bath
            bath_type="mp2-natorb", bath_size=None, bath_tol=1e-4, bath_energy_tol=None,
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
        if C_local.shape[-1] == 0:
            raise ValueError("No local orbitals in cluster %d:%s" % (self.id, self.name))

        self.C_local = C_local
        self.C_env = C_env
        log.info("  * Size= %d", self.size)     # depends on self.C_local

        # Determine number of mean-field electrons in fragment space
        s = self.base.get_ovlp()
        dm = np.linalg.multi_dot((self.C_local.T, s, self.mf.make_rdm1(), s, self.C_local))
        self.nelec_mf_frag = np.trace(dm)
        log.info("  * Mean-field electrons= %.5f", self.nelec_mf_frag)

        #assert self.nlocal == len(self.indices)
        #assert (self.size == len(self.indices) or self.local_orbital_type in ("NonOrth-IAO", "PMO"))

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

        # Relative tolerances?
        if kwargs.get("bath_tol_per_electron", True):
            if bath_tol[0] is not None:
                bath_tol = (self.nelec_mf_frag*bath_tol[0], bath_tol[1])
            if bath_tol[1] is not None:
                bath_tol = (bath_tol[0], self.nelec_mf_frag*bath_tol[1])
            if bath_energy_tol[0] is not None:
                bath_energy_tol = (self.nelec_mf_frag*bath_energy_tol[0], bath_energy_tol[1])
            if bath_energy_tol[1] is not None:
                bath_energy_tol = (bath_energy_tol[0], self.nelec_mf_frag*bath_energy_tol[1])

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

        # Other options from kwargs:
        self.solver_options = kwargs.get("solver_options", {})

        self.mp2_correction = kwargs.get("mp2_correction", True)
        # Make MP2 correction tuple for (occupied, virtual) bath
        if not hasattr(self.mp2_correction, "__getitem__"):
            self.mp2_correction = (self.mp2_correction, self.mp2_correction)

        # Bath parameters

        # Currently not in use (always on!)
        self.use_ref_orbitals_dmet = kwargs.get("use_ref_orbitals_dmet", True)
        self.use_ref_orbitals_bath = kwargs.get("use_ref_orbitals_bath", False)

        self.dmet_bath_tol = kwargs.get("dmet_bath_tol", 1e-4)
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

        self.opts = Options()
        self.opts.prim_mp2_bath_tol_occ = self.base.opts.get("prim_mp2_bath_tol_occ", False)
        self.opts.prim_mp2_bath_tol_vir = self.base.opts.get("prim_mp2_bath_tol_vir", False)
        self.opts.make_rdm1 = self.base.opts.get("make_rdm1", False)
        self.opts.eom_ccsd = self.base.opts.get("eom_ccsd", False)

        #self.project_type = kwargs.get("project_type", "first-occ")

        # Orbital coefficents added to this dictionary can be written as an orbital file
        #self.orbitals = {"Fragment" : self.C_local}
        self.orbitals = OrderedDict()
        self.orbitals["Fragment"] = self.C_local

        #
        #self.occbath_eigref = kwargs.get("occbath-eigref", None)
        #self.virbath_eigref = kwargs.get("virbath-eigref", None)
        self.refdata_in = kwargs.get("refdata", {})

        # Maximum number of iterations for consistency of amplitudes between clusters
        self.maxiter = kwargs.get("maxiter", 1)

        self.ccsd_t_max_orbitals = kwargs.get("ccsd_t_max_orbitals", 200)

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

        self.active = []
        self.active_occ = []
        self.active_vir = []
        self.frozen = []
        self.frozen_occ = []
        self.frozen_vir = []

        # These are used by the solver
        self.mo_coeff = None
        self.mo_occ = None
        self.eris = None

        #self.ref_orbitals = None

        # Reference orbitals should be saved with keys
        # dmet-bath, occ-bath, vir-bath
        #self.refdata = {}
        self.refdata_out = {}
        #self.ref_orbitals = {}

        # Local amplitudes [C1, C2] can be stored here and
        # used by other fragments
        self.amplitudes = {}

        # Orbitals sizes
        #self.nbath0 = 0
        #self.nbath = 0
        #self.nfrozen = 0
        # Calculation results

        self.iteration = 0

        # Output values
        self.converged = False
        self.e_corr = 0.0
        self.e_pert_t = 0.0
        self.e_pert_t2 = 0.0
        self.e_corr_dmp2 = 0.0

        # For testing:
        # Full cluster correlation energy
        self.e_corr_full = 0.0
        self.e_corr_v = 0.0
        self.e_corr_d = 0.0

        self.e_dmet = 0.0

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
    def nactive(self):
        """Number of active orbitals."""
        return len(self.active)

    @property
    def nfrozen(self):
        """Number of frozen environment orbitals."""
        return len(self.frozen)

    #def get_orbitals(self):
    #    """Get dictionary with orbital coefficients."""
    #    orbitals = {
    #            "local" : self.C_local,
    #            "dmet-bath" : self.C_bath,
    #            "occ-bath" : self.C_occbath,
    #            "vir-bath" : self.C_virbath,
    #            }
    #    return orbitals

    def get_refdata(self):
        """Get data of reference calculation for smooth PES."""
        refdata = self.refdata_out
        log.debug("Getting refdata: %r", refdata.keys())
        #refdata = {
        #        "dmet-bath" : self.C_bath,
        #        "occbath-eigref" : self.occbath_eigref,
        #        "virbath-eigref" : self.virbath_eigref,
        #        }
        return refdata

    def set_refdata(self, refdata):
        log.debug("Setting refdata: %r", refdata.keys())
        #self.refdata = refdata
        self.refdata_in = refdata
        #self.dmetbath_ref = refdata["dmet-bath"]
        #self.occbath_eigref = refdata["occbath-eigref"]
        #self.virbath_eigref = refdata["virbath-eigref"]

    def reset(self, keep_ref_orbitals=True):
        """Reset cluster object. By default it stores the previous orbitals, so they can be used
        as reference orbitals for a new calculation of different geometry."""
        ref_orbitals = self.get_orbitals()
        self.set_default_attributes()
        if keep_ref_orbitals:
            self.ref_orbitals = ref_orbitals

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

    ##@property
    ##def mo_energy(self):
    ##    return self.base.mo_energy

    ##@property
    ##def mo_coeff(self):
    ##    return self.base.mo_coeff

    @property
    def mol(self):
        """The molecule or cell object is taken from self.base.mol.
        It should be the same as self.base.mf.mol by default."""
        return self.base.mol

    @property
    def has_pbc(self):
        return isinstance(self.mol, pyscf.pbc.gto.Cell)

    #@property
    #def madelung(self):
    #    """Madelung constant for PBC systems."""
    #    from pyscf.pbc import tools
    #    mad = tools.madelung(self.mol, self.mf.kpt)
    #    return mad

    @property
    def local_orbital_type(self):
        return self.base.local_orbital_type

    @property
    def energy_factor(self):
        return self.symmetry_factor

    # Register frunctions of bath.py as methods
    project_ref_orbitals = project_ref_orbitals
    make_dmet_bath = make_dmet_bath
    make_bath = make_bath
    make_local_bath = make_local_bath
    make_mf_bath = make_mf_bath
    make_mp2_bath = make_mp2_bath
    get_mp2_correction = get_mp2_correction
    transform_mp2_eris = transform_mp2_eris
    run_mp2 = run_mp2
    run_mp2_general = run_mp2_general

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
            Bath orbitals.

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
        tol = self.dmet_bath_tol
        if not np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=tol, rtol=0):
            raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not all close to 0 or 1:\n%s", e)
        e, R = e[::-1], R[:,::-1]
        C_clst = np.dot(C_clst, R)
        nocc_clst = sum(e > 0.5)
        nvir_clst = sum(e < 0.5)
        log.info("Number of cluster orbitals: occ=%3d, vir=%3d", nocc_clst, nvir_clst)

        C_occclst, C_virclst = np.hsplit(C_clst, [nocc_clst])

        return C_occclst, C_virclst

    def canonicalize(self, *C, eigenvalues=False):
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
        #F = np.linalg.multi_dot((C.T, self.mf.get_fock(), C))
        #F = np.linalg.multi_dot((C.T, self.base.fock, C))
        F = np.linalg.multi_dot((C.T, self.base.get_fock(), C))
        e, R = np.linalg.eigh(F)
        C = np.dot(C, R)
        if eigenvalues:
            return C, e
        return C

    def get_occup(self, C):
        """Get mean-field occupation numbers (diagonal of 1-RDM) of orbitals.

        Parameters
        ----------
        C : ndarray, shape(N, M)
            Orbital coefficients.

        Returns
        -------
        occ : ndarray, shape(M)
            Occupation numbers of orbitals.
        """
        S = self.mf.get_ovlp()
        D = np.linalg.multi_dot((C.T, S, self.mf.make_rdm1(), S, C))
        occ = np.diag(D)
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
        S = self.mf.get_ovlp()

        # Project onto space of local (fragment) orbitals.
        #if self.local_orbital_type in ("IAO", "LAO"):
        if self.local_orbital_type in ("IAO", "LAO", "PMO"):
            CSC = np.linalg.multi_dot((C.T, S, self.C_local))
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


    def run(self, solver=None,
            #ref_orbitals=None
            refdata=None,
            ):
        """Construct bath orbitals and run solver.

        Paramters
        ---------
        solver : str
            Method ["MP2", "CISD", "CCSD", "CCSD(T)", "FCI"]
        ref_orbitals : dict
            Dictionary with reference orbitals.

        Returns
        -------
        converged : bool
        """

        solver = solver or self.solver
        # Orbitals from a reference calaculation (e.g. different geometry)
        # Used for recovery of orbitals via active transformation
        refdata = refdata or self.refdata_in
        if False:
            ref_orbitals = ref_orbitals or self.ref_orbitals

            # === Make DMET bath orbital and diagonalize DM in cluster space
            if ref_orbitals.get("dmet-bath", None) is not None:
                assert np.allclose(ref_orbitals["dmet-bath"], self.refdata_in["dmet-bath"])

        t0_bath = t0 = timer()
        log.info("MAKING DMET BATH")
        log.info("****************")
        log.changeIndentLevel(1)
        #C_bath, C_occenv, C_virenv = self.make_dmet_bath(C_ref=ref_orbitals.get("dmet-bath", None))
        C_bath, C_occenv, C_virenv = self.make_dmet_bath(C_ref=refdata.get("dmet-bath", None), tol=self.dmet_bath_tol)
        log.debug("Time for DMET bath: %s", get_time_string(timer()-t0))
        log.changeIndentLevel(-1)

        # Add additional orbitals to cluster [optional]
        # First-order power orbitals:
        #if self.power1_occ_bath_tol is not False:
        #    C_add, C_occenv, _ = make_mf_bath(self, C_occenv, "occ", bathtype="power",
        #            tol=self.power1_occ_bath_tol)
        #    log.info("Adding %d first-order occupied power bath orbitals to cluster.", C_add.shape[-1])
        #    C_bath = np.hstack((C_add, C_bath))
        #if self.power1_vir_bath_tol is not False:
        #    C_add, C_virenv, _ = make_mf_bath(self, C_virenv, "vir", bathtype="power",
        #            tol=self.power1_vir_bath_tol)
        #    log.info("Adding %d first-order virtual power bath orbitals to cluster.", C_add.shape[-1])
        #    C_bath = np.hstack((C_bath, C_add))
        ## Local orbitals:
        #if self.local_occ_bath_tol is not False:
        #    C_add, C_occenv = make_local_bath(self, C_occenv, tol=self.local_occ_bath_tol)
        #    log.info("Adding %d local occupied bath orbitals to cluster.", C_add.shape[-1])
        #    C_bath = np.hstack((C_add, C_bath))
        #if self.local_vir_bath_tol is not False:
        #    C_add, C_virenv = make_local_bath(self, C_virenv, tol=self.local_vir_bath_tol)
        #    log.info("Adding %d local virtual bath orbitals to cluster.", C_add.shape[-1])
        #    C_bath = np.hstack((C_bath, C_add))
        C_bath, C_occenv, C_virenv = self.additional_bath_for_cluster(C_bath, C_occenv, C_virenv)

        # Diagonalize cluster DM to separate cluster occupied and virtual
        C_occclst, C_virclst = self.diagonalize_cluster_dm(C_bath)

        # Primary MP2 bath orbitals
        if True:
            if self.opts.prim_mp2_bath_tol_occ:
                log.info("Adding primary occupied MP2 bath orbitals")
                C_add_o, C_rest_o, *_ = self.make_mp2_bath(C_occclst, C_virclst, "occ",
                        c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
                        mp2_correction=False)
            if self.opts.prim_mp2_bath_tol_vir:
                log.info("Adding primary virtual MP2 bath orbitals")
                C_add_v, C_rest_v, *_ = self.make_mp2_bath(C_occclst, C_virclst, "vir",
                        c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
                        mp2_correction=False)
            # Combine
            if self.opts.prim_mp2_bath_tol_occ:
                C_bath = np.hstack((C_add_o, C_bath))
                C_occenv = C_rest_o
            if self.opts.prim_mp2_bath_tol_vir:
                C_bath = np.hstack((C_bath, C_add_v))
                C_virenv = C_rest_v

            # Re-diagonalize cluster DM to separate cluster occupied and virtual
            C_occclst, C_virclst = self.diagonalize_cluster_dm(C_bath)

        self.C_bath = C_bath

        # Canonicalize cluster
        if False:
            C_occclst, e = self.canonicalize(C_occclst, eigenvalues=True)
            log.debug("Occupied cluster Fock eigenvalues: %r", e)
            C_virclst, e = self.canonicalize(C_virclst, eigenvalues=True)
            log.debug("Virtual cluster Fock eigenvalues: %r", e)

        self.C_occclst = C_occclst
        self.C_virclst = C_virclst

        # For orbital plotting
        self.orbitals["DMET-bath"] = C_bath
        self.orbitals["Occ.-Cluster"] = C_occclst
        self.orbitals["Vir.-Cluster"] = C_virclst

        self.refdata_out["dmet-bath"] = C_bath

        # === Additional bath orbitals

        if self.use_ref_orbitals_bath:
            #C_occref = ref_orbitals.get("occ-bath", None)
            #C_virref = ref_orbitals.get("vir-bath", None)
            #if C_occref is not None:
            #    assert np.allclose(C_occref, self.refdata_in["occ-bath"])
            #if C_virref is not None:
            #    assert np.allclose(C_virref, self.refdata_in["vir-bath"])
            C_occref = refdata.get("occ-bath", None)
            C_virref = refdata.get("vir-bath", None)
        else:
            C_occref = None
            C_virref = None

        # Reorder
        #occbath_eigref = self.occbath_eigref
        #virbath_eigref = self.virbath_eigref
        occbath_eigref = refdata.get("occbath-eigref", None)
        virbath_eigref = refdata.get("virbath-eigref", None)
        #virbath_eigref = self.virbath_eigref

        # TEST
        #occbath_eigref = None
        #virbath_eigref = None

        log.info("MAKING OCCUPIED BATH")
        log.info("********************")
        t0 = timer()
        log.changeIndentLevel(1)
        C_occbath, C_occenv2, e_delta_occ, occbath_eigref = self.make_bath(
                C_occenv, self.bath_type, "occ",
                C_ref=C_occref, eigref=occbath_eigref,
                # New for MP2 bath:
                C_occenv=C_occenv, C_virenv=C_virenv,
                nbath=self.bath_target_size[0], tol=self.bath_tol[0], energy_tol=self.bath_energy_tol[0])
        log.debug("Time for occupied %r bath: %s", self.bath_type, get_time_string(timer()-t0))
        log.changeIndentLevel(-1)

        log.info("MAKING VIRTUAL BATH")
        log.info("*******************")
        t0 = timer()
        log.changeIndentLevel(1)
        C_virbath, C_virenv2, e_delta_vir, virbath_eigref = self.make_bath(
                C_virenv, self.bath_type, "vir",
                C_ref=C_virref, eigref=virbath_eigref,
                # New for MP2 bath:
                C_occenv=C_occenv, C_virenv=C_virenv,
                nbath=self.bath_target_size[1], tol=self.bath_tol[1], energy_tol=self.bath_energy_tol[1])
        log.debug("Time for virtual %r bath: %s", self.bath_type, get_time_string(timer()-t0))
        log.changeIndentLevel(-1)

        C_occenv, C_virenv = C_occenv2, C_virenv2
        self.C_occbath = C_occbath
        self.C_virbath = C_virbath
        self.C_occenv = C_occenv
        self.C_virenv = C_virenv

        # For orbital ploting
        self.orbitals["Occ.-bath"] = C_occbath
        self.orbitals["Vir.-bath"] = C_virbath
        self.orbitals["Occ.-env."] = C_occenv
        self.orbitals["Vir.-env."] = C_virenv

        self.refdata_out["occ-bath"] = C_occbath
        self.refdata_out["vir-bath"] = C_virbath

        # For future reorderings
        #self.occbath_eigref = occbath_eigref
        #self.virbath_eigref = virbath_eigref
        self.refdata_out["occbath-eigref"] = occbath_eigref
        self.refdata_out["virbath-eigref"] = virbath_eigref

        self.e_delta_mp2 = e_delta_occ + e_delta_vir
        log.debug("MP2 correction = %.8g", self.e_delta_mp2)

        # FULL MP2 correction [TESTING]
        #Co1 = np.hstack((C_occclst, C_occenv))
        #Cv1 = np.hstack((C_virclst, C_virenv))
        #Co2 = np.hstack((C_occclst, C_occbath))
        #Cv2 = np.hstack((C_virclst, C_virbath))
        #self.e_delta_mp2 = self.get_mp2_correction(Co1, Cv1, Co2, Cv2)
        #log.debug("Full MP2 correction = %.8g", self.e_delta_mp2)

        # === Canonicalize orbitals
        C_occact = self.canonicalize(C_occclst, C_occbath)
        C_viract = self.canonicalize(C_virclst, C_virbath)
        #C_occact = np.hstack((C_occclst, C_occbath))
        #C_viract = np.hstack((C_virclst, C_virbath))
        self.C_occact = C_occact
        self.C_viract = C_viract

        # Combine, important to keep occupied orbitals first
        # Put frozen (occenv, virenv) orbitals to the front and back
        # and active orbitals (occact, viract) in the middle
        Co = np.hstack((C_occenv, C_occact))
        Cv = np.hstack((C_viract, C_virenv))
        mo_coeff = np.hstack((Co, Cv))
        No = Co.shape[-1]
        Nv = Cv.shape[-1]
        # Check occupations
        assert np.allclose(self.get_occup(Co), 2, atol=2*self.dmet_bath_tol), "%r" % self.get_occup(Co)
        assert np.allclose(self.get_occup(Cv), 0, atol=2*self.dmet_bath_tol), "%r" % self.get_occup(Cv)
        mo_occ = np.asarray(No*[2] + Nv*[0])

        frozen_occ = list(range(C_occenv.shape[-1]))
        active_occ = list(range(C_occenv.shape[-1], No))
        active_vir = list(range(No, No+C_viract.shape[-1]))
        frozen_vir = list(range(No+C_viract.shape[-1], No+Nv))
        active = active_occ + active_vir
        frozen = frozen_occ + frozen_vir
        # Run solver
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.active = active
        self.active_occ = active_occ
        self.active_vir = active_vir
        self.frozen = frozen
        self.frozen_occ = frozen_occ
        self.frozen_vir = frozen_vir
        log.debug("Wall time for bath: %s", get_time_string(timer()-t0_bath))

        log.info("ORBITALS")
        log.info("********")
        log.info("  * Occupied: active= %4d  frozen= %4d  total= %4d", len(active_occ), len(frozen_occ), len(active_occ)+len(frozen_occ))
        log.info("  * Virtual:  active= %4d  frozen= %4d  total= %4d", len(active_vir), len(frozen_vir), len(active_vir)+len(frozen_vir))
        log.info("  * Total:    active= %4d  frozen= %4d  total= %4d", self.nactive, self.nfrozen, self.nactive+self.nfrozen)

        log.info("RUNNING %s SOLVER", solver)
        log.info((len(solver)+15)*"*")

        # OLD
        #t0 = MPI.Wtime()
        #log.changeIndentLevel(1)
        #converged, e_corr = self.run_solver(solver, mo_coeff, mo_occ, active=active, frozen=frozen)
        #log.debug("Wall time for %s solver: %s", solver, get_time_string(MPI.Wtime()-t0))
        #log.changeIndentLevel(-1)

        # NEW: Create solver object
        t0 = timer()
        log.changeIndentLevel(1)
        csolver = ClusterSolver(self, solver, mo_coeff, mo_occ, active=active, frozen=frozen)
        csolver.run()
        self.converged = csolver.converged
        self.e_corr_full = csolver.e_corr
        log.debug("Wall time for %s solver: %s", csolver.solver, get_time_string(timer()-t0))

        pc1, pc2 = self.get_local_amplitudes(csolver._solver, csolver.c1, csolver.c2)
        self.e_corr = self.get_local_energy(csolver._solver, pc1, pc2, eris=csolver._eris)

        # Population analysis
        if self.opts.make_rdm1 and csolver.dm1 is not None:
            self.pop_analysis(csolver.dm1)
        # EOM analysis
        if self.opts.eom_ccsd in (True, "IP"):
            self.eom_ip_energy, _ = self.eom_analysis(csolver, "IP")
        if self.opts.eom_ccsd in (True, "EA"):
            self.eom_ea_energy, _ = self.eom_analysis(csolver, "EA")

        log.changeIndentLevel(-1)

        return self.converged, self.e_corr

    def pop_analysis(self, dm1, filename=None, mode="a", sig_tol=0.01):
        """Perform population analsis for the given density-matrix and compare to the MF."""
        if filename is None:
            filename = "%s-%s.txt" % (self.base.opts.popfile, self.name)

        s = self.base.get_ovlp()
        lo = self.base.lo
        dm1 = np.linalg.multi_dot((lo.T, s, dm1, s, lo))
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

        s = self.base.get_ovlp()
        lo = self.base.lo
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
                r1lo = einsum("i,ai,ab,bl->l", r1, eris.mo_coeff[:,:cc.nocc], s, lo)

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

    def create_orbital_file(self, filetype="molden"):
        if filetype not in ("cube", "molden"):
            raise ValueError("Unknown file type: %s" % filetype)
        ext = {"molden" : "molden", "cube" : "cube"}
        filename = "cluster-%d.%s" % (self.id, ext[filetype])
        create_orbital_file(self.mol, filename, self.orbitals, filetype=filetype)

        #if filetype == "molden":
        #    from pyscf.tools import molden

        #    with open(filename, "w") as f:
        #        molden.header(self.mol, f)
        #        labels = []
        #        coeffs = []
        #        for name, C in orbitals.items():
        #            labels += C.shape[-1]*[name]
        #            coeffs.append(C)
        #        coeffs = np.hstack(coeffs)
        #        molden.orbital_coeff(self.mol, f, coeffs, symm=labels)

        #        #for name, C in self.orbitals.items():
        #            #symm = orb_labels.get(name, "?")
        #            #symm = C.shape[-1] * [name]
        #            #molden.orbital_coeff(self.mol, f, C)
        #            #molden.orbital_coeff(self.mol, f, C, symm=symm)
        #elif filetype == "cube":
        #    raise NotImplementedError()
        #    for orbkind, C in self.orbitals.items():
        #        for j in range(C.shape[-1]):
        #            filename = "C%d-%s-%d.cube" % (self.id, orbkind, j)
        #            make_cubegen_file(self.mol, C[:,j], filename)
