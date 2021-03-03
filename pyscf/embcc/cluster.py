# Standard libaries
import logging
from collections import OrderedDict
import functools

# External libaries
import numpy as np
import scipy
import scipy.linalg
from mpi4py import MPI

# Internal libaries
import pyscf
#import pyscf.lo
#import pyscf.cc
#import pyscf.ci
#import pyscf.mcscf
#import pyscf.fci
#import pyscf.mp
import pyscf.pbc
#import pyscf.pbc.cc
#import pyscf.pbc.mp
#import pyscf.pbc.tools

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

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

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
        self.opts.make_rdm1 = self.base.opts.get("make_rdm1", False)
        self.opts.ip_eom = self.base.opts.get("ip_eom", False)
        self.opts.ea_eom = self.base.opts.get("ea_eom", False)

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


    ##def run_solver(self, solver=None, mo_coeff=None, mo_occ=None, active=None, frozen=None, eris=None,
    ##        solver_options=None):
    ##    solver = solver or self.solver
    ##    if mo_coeff is None: mo_coeff = self.mo_coeff
    ##    if mo_occ is None: mo_occ = self.mo_occ
    ##    if active is None: active = self.active
    ##    if frozen is None: frozen = self.frozen
    ##    if eris is None: eris = self.eris
    ##    if solver_options is None: solver_options = self.solver_options

    ##    self.iteration += 1
    ##    if self.iteration > 1:
    ##        log.debug("Iteration %d", self.iteration)

    ##    if len(active) == 1:
    ##        log.debug("Only one orbital in cluster. No correlation energy.")
    ##        solver = None

    ##    if self.has_pbc:
    ##        log.debug("Cell object found.")

    ##    if solver is None:
    ##        log.debug("No solver")
    ##        e_corr_full = 0
    ##        e_corr = 0
    ##        converged = True

    ##    # MP2
    ##    # ===
    ##    elif solver == "MP2":
    ##        if self.use_pbc:
    ##            mp2 = pyscf.pbc.mp.MP2(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
    ##        else:
    ##            mp2 = pyscf.mp.MP2(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
    ##        solverobj = mp2

    ##        if eris is None:
    ##            t0 = MPI.Wtime()
    ##            if self.use_pbc:
    ##                c_act = mo_coeff[:,active]
    ##                fock = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
    ##                eris = mp2.ao2mo(direct_init=True, mo_energy=np.diag(fock), fock=fock)
    ##            elif hasattr(mp2, "with_df"):
    ##                eris = mp2.ao2mo(store_eris=True)
    ##            else:
    ##                eris = mp2.ao2mo()
    ##            log.debug("Time for integral transformation: %s", get_time_string(MPI.Wtime()-t0))
    ##            #log_time("integral transformation", MPI.Wtime()-t0)

    ##        e_corr_full, t2 = mp2.kernel(eris=eris, hf_reference=True)
    ##        converged = True
    ##        e_corr_full *= self.energy_factor
    ##        C1, C2 = None, t2

    ##        pC1, pC2 = self.get_local_amplitudes(mp2, C1, C2, symmetrize=True)
    ##        e_corr = self.get_local_energy(mp2, pC1, pC2, eris=eris)

    ##    # CCSD
    ##    # ====
    ##    elif solver in ("CCSD", "CCSD(T)"):
    ##        if self.use_pbc:
    ##            cc = pyscf.pbc.cc.CCSD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
    ##        else:
    ##            cc = pyscf.cc.CCSD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)

    ##        solverobj = cc
    ##        self.cc = cc

    ##        # We want to reuse the integral for local energy
    ##        if eris is None:
    ##            t0 = MPI.Wtime()
    ##            #eris = cc.ao2mo()
    ##            # [Avoid expensive Fock rebuild in PBC]
    ##            if self.use_pbc:
    ##                c_act = mo_coeff[:,active]
    ##                fock = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
    ##                if hasattr(self.mf.with_df, "_cderi") and isinstance(self.mf.with_df._cderi, np.ndarray):
    ##                    #t0 = MPI.Wtime()
    ##                    #eris = pbc_gdf_ao2mo.ao2mo(cc, fock=fock)
    ##                    #t1 = MPI.Wtime()
    ##                    # TEST NEW
    ##                    eris = ao2mo_j3c.ao2mo_ccsd(cc, fock=fock)
    ##                    #t2 = MPI.Wtime()
    ##                    #log.info("TIME OLD: %f TIME NEW: %f", (t1-t0), (t2-t1))
    ##                    #assert np.allclose(eris.mo_energy, eris2.mo_energy)
    ##                    #assert np.allclose(eris.oooo, eris2.oooo)
    ##                    #assert np.allclose(eris.ovoo, eris2.ovoo)
    ##                    #assert np.allclose(eris.ovov, eris2.ovov)
    ##                    #assert np.allclose(eris.ovvo, eris2.ovvo)
    ##                    #assert np.allclose(eris.oovv, eris2.oovv)
    ##                    #assert np.allclose(eris.ovvv, eris2.ovvv)
    ##                    #assert np.allclose(eris.vvvv, eris2.vvvv)
    ##                else:
    ##                    eris = cc.ao2mo_direct(fock=fock)
    ##            else:
    ##                eris = cc.ao2mo()
    ##            t = (MPI.Wtime()-t0)
    ##            log.debug("Time for integral transformation [s]: %.3f (%s)", t, get_time_string(t))
    ##        cc.max_cycle = 100

    ##        t0 = MPI.Wtime()
    ##        if self.restart_solver:
    ##            log.debug("Running CCSD starting with parameters for: %r...", self.restart_params.keys())
    ##            cc.kernel(eris=eris, **self.restart_params)
    ##        else:
    ##            log.debug("Running CCSD...")
    ##            cc.kernel(eris=eris)
    ##        log.debug("CCSD done. converged: %r", cc.converged)
    ##        t = (MPI.Wtime()-t0)
    ##        log.debug("Time for CCSD [s]: %.3f (%s)", t, get_time_string(t))

    ##        if self.restart_solver:
    ##            self.restart_params["t1"] = cc.t1
    ##            self.restart_params["t2"] = cc.t2

    ##        converged = cc.converged
    ##        e_corr_full = self.energy_factor*cc.e_corr

    ##        log.info("Diagnostic")
    ##        log.info("**********")
    ##        dg_t1 = cc.get_t1_diagnostic()
    ##        dg_d1 = cc.get_d1_diagnostic()
    ##        dg_d2 = cc.get_d2_diagnostic()
    ##        log.info("  (T1<0.02: good / D1<0.02: good, D1<0.05: fair / D2<0.15: good, D2<0.18: fair)")
    ##        log.info("  (good: MP2~CCSD~CCSD(T) / fair: use MP2/CCSD with caution)")
    ##        dg_t1_msg = "good" if dg_t1 <= 0.02 else "inadequate!"
    ##        dg_d1_msg = "good" if dg_d1 <= 0.02 else ("fair" if dg_d1 <= 0.05 else "inadequate!")
    ##        dg_d2_msg = "good" if dg_d2 <= 0.15 else ("fair" if dg_d2 <= 0.18 else "inadequate!")
    ##        fmtstr = "  * %2s=%6g (%s)"
    ##        log.info(fmtstr, "T1", dg_t1, dg_t1_msg)
    ##        log.info(fmtstr, "D1", dg_d1, dg_d1_msg)
    ##        log.info(fmtstr, "D2", dg_d2, dg_d2_msg)
    ##        if dg_t1 > 0.02 or dg_d1 > 0.05 or dg_d2 > 0.18:
    ##            log.warning("  WARNING: some diagnostic(s) indicate CCSD may not be adequate.")

    ##        C1, C2 = amplitudes_T2C(cc.t1, cc.t2)
    ##        pC1, pC2 = self.get_local_amplitudes(cc, C1, C2)
    ##        e_corr = self.get_local_energy(cc, pC1, pC2, eris=eris)

    ##        # Calculate (T) contribution
    ##        if solver == "CCSD(T)":
    ##            # Skip (T) if too expensive
    ##            if len(active) > self.ccsd_t_max_orbitals:
    ##                log.warning("Number of orbitals = %d. Skipping calculation of (T)." % len(active))
    ##            else:
    ##                t0 = MPI.Wtime()
    ##                pT1, pT2 = self.get_local_amplitudes(cc, cc.t1, cc.t2)
    ##                # Symmetrized T2 gives slightly different (T) energy!
    ##                #pT1, pT2 = self.get_local_amplitudes(cc, cc.t1, cc.t2, symmetrize=True)
    ##                self.e_pert_t = self.energy_factor*ccsd_t.kernel_new(cc.t1, cc.t2, pT2, eris)
    ##                t = (MPI.Wtime()-t0)
    ##                log.debug("Time for (T) [s]: %.3f (%s)", t, get_time_string(t))

    ##        # Other energy variants
    ##        if False:
    ##            pC1v, pC2v = self.get_local_amplitudes(cc, C1, C2, variant="first-vir")
    ##            pC1d, pC2d = self.get_local_amplitudes(cc, C1, C2, variant="democratic")
    ##            e_corr_v = self.get_local_energy(cc, pC1v, pC2v, eris=eris)
    ##            e_corr_d = self.get_local_energy(cc, pC1d, pC2d, eris=eris)

    ##        # TESTING: Get global amplitudes:
    ##        #if False:
    ##        #if True:
    ##        if self.maxiter > 1:
    ##            log.debug("Maxiter=%3d, storing amplitudes.", self.maxiter)
    ##            if self.base.T1 is None:
    ##                No = sum(self.base.mo_occ > 0)
    ##                Nv = len(self.base.mo_occ) - No
    ##                self.base.T1 = np.zeros((No, Nv))
    ##                self.base.T2 = np.zeros((No, No, Nv, Nv))

    ##            pT1, pT2 = self.get_local_amplitudes(cc, cc.t1, cc.t2)
    ##            #pT1, pT2 = self.get_local_amplitudes(cc, cc.t1, cc.t2, variant="democratic")

    ##            # Transform to HF MO basis
    ##            act = cc.get_frozen_mask()
    ##            occ = cc.mo_occ[act] > 0
    ##            vir = cc.mo_occ[act] == 0
    ##            S = self.mf.get_ovlp()
    ##            Co = cc.mo_coeff[:,act][:,occ]
    ##            Cv = cc.mo_coeff[:,act][:,vir]
    ##            Cmfo = self.base.mo_coeff[:,self.base.mo_occ>0]
    ##            Cmfv = self.base.mo_coeff[:,self.base.mo_occ==0]
    ##            Ro = np.linalg.multi_dot((Cmfo.T, S, Co))
    ##            Rv = np.linalg.multi_dot((Cmfv.T, S, Cv))
    ##            pT1, pT2 = self.transform_amplitudes(Ro, Rv, pT1, pT2)

    ##            self.base.T1 += pT1
    ##            self.base.T2 += pT2

    ##    elif solver == "CISD":
    ##        # Currently not maintained
    ##        #raise NotImplementedError()
    ##        if self.use_pbc:
    ##            ci = pyscf.pbc.ci.CISD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
    ##        else:
    ##            ci = pyscf.ci.CISD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)
    ##        solverobj = ci

    ##        # We want to reuse the integral for local energy
    ##        if eris is None:
    ##            t0 = MPI.Wtime()
    ##            eris = ci.ao2mo()
    ##            t = (MPI.Wtime() - t0)
    ##            log.debug("Time for integral transformation [s]: %.3f (%s)", t, get_time_string(t))
    ##        ci.max_cycle = 100

    ##        log.debug("Running CISD...")
    ##        if self.nelectron_target is None:
    ##            ci.kernel(eris=eris)
    ##        else:
    ##            # Buggy, doesn't work
    ##            raise NotImplementedError()

    ##            S = self.mf.get_ovlp()
    ##            px = self.get_local_projector(mo_coeff)
    ##            b = np.linalg.multi_dot((S, self.C_local, self.C_local.T, S))

    ##            h1e = self.mf.get_hcore()
    ##            h1e_func = self.mf.get_hcore

    ##            cptmin = 0.0
    ##            cptmax = +3.0
    ##            #cptmin = -0.5
    ##            #cptmax = +0.5

    ##            ntol = 1e-6

    ##            def electron_error(chempot):
    ##                nonlocal e_tot, wf

    ##                ci._scf.get_hcore = lambda *args : h1e - chempot*b
    ##                #ci._scf.get_hcore = lambda *args : h1e
    ##                ci.kernel(eris=eris)
    ##                dm1xx = np.linalg.multi_dot((px.T, ci.make_rdm1(), px))
    ##                nx = np.trace(dm1xx)
    ##                nerr = (nx - self.nelectron_target)
    ##                log.debug("chempot=%16.8g, electrons=%16.8g, error=%16.8g", chempot, nx, nerr)
    ##                assert ci.converged

    ##                if abs(nerr) < ntol:
    ##                    log.debug("Electron error |%e| below tolerance of %e", nerr, ntol)
    ##                    raise StopIteration

    ##                return nerr

    ##            try:
    ##                scipy.optimize.brentq(electron_error, cptmin, cptmax)
    ##            except StopIteration:
    ##                pass

    ##            # Reset hcore Hamiltonian
    ##            ci._scf.get_hcore = h1e_func

    ##        log.debug("CISD done. converged: %r", ci.converged)

    ##        converged = ci.converged
    ##        e_corr_full = self.energy_factor*ci.e_corr
    ##        # Intermediate normalization
    ##        C0, C1, C2 = ci.cisdvec_to_amplitudes(ci.ci)
    ##        # Renormalize
    ##        C1 *= 1/C0
    ##        C2 *= 1/C0

    ##        pC1, pC2 = self.get_local_amplitudes(ci, C1, C2)
    ##        e_corr = self.get_local_energy(ci, pC1, pC2, eris=eris)

    ##    #elif solver == "FCI":
    ##    elif solver in ("FCI-spin0", "FCI-spin1"):

    ##        nocc_active = len(self.active_occ)
    ##        casci = pyscf.mcscf.CASCI(self.mf, self.nactive, 2*nocc_active)
    ##        solverobj = casci
    ##        # Solver options
    ##        casci.verbose = 10
    ##        casci.canonicalization = False
    ##        #casci.fix_spin_(ss=0)
    ##        # TEST SPIN
    ##        if solver == "FCI-spin0":
    ##            casci.fcisolver = pyscf.fci.direct_spin0.FCISolver(self.mol)
    ##        casci.fcisolver.conv_tol = 1e-9
    ##        casci.fcisolver.threads = 1
    ##        casci.fcisolver.max_cycle = 400
    ##        #casci.fcisolver.level_shift = 5e-3

    ##        if solver_options:
    ##            spin = solver_options.pop("fix_spin", None)
    ##            if spin is not None:
    ##                log.debug("Setting fix_spin to %r", spin)
    ##                casci.fix_spin_(ss=spin)

    ##            for key, value in solver_options.items():
    ##                log.debug("Setting solver attribute %s to value %r", key, value)
    ##                setattr(casci.fcisolver, key, value)

    ##        # The sorting of the orbitals above should already have placed the CAS in the correct position

    ##        log.debug("Running FCI...")
    ##        if self.nelectron_target is None:
    ##            e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff)
    ##        # Chemical potential loop
    ##        else:

    ##            S = self.mf.get_ovlp()
    ##            px = self.get_local_projector(mo_coeff)
    ##            b = np.linalg.multi_dot((S, self.C_local, self.C_local.T, S))

    ##            t = np.linalg.multi_dot((S, mo_coeff, px))
    ##            h1e = casci.get_hcore()
    ##            h1e_func = casci.get_hcore

    ##            cptmin = -4
    ##            cptmax = 0
    ##            #cptmin = -0.5
    ##            #cptmax = +0.5

    ##            ntol = 1e-6
    ##            e_tot = None
    ##            wf = None

    ##            def electron_error(chempot):
    ##                nonlocal e_tot, wf

    ##                #casci.get_hcore = lambda *args : h1e - chempot*b
    ##                casci.get_hcore = lambda *args : h1e - chempot*(S-b)

    ##                e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff, ci0=wf)
    ##                #e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff)
    ##                dm1xx = np.linalg.multi_dot((t.T, casci.make_rdm1(), t))
    ##                nx = np.trace(dm1xx)
    ##                nerr = (nx - self.nelectron_target)
    ##                log.debug("chempot=%16.8g, electrons=%16.8g, error=%16.8g", chempot, nx, nerr)
    ##                assert casci.converged

    ##                if abs(nerr) < ntol:
    ##                    log.debug("Electron error |%e| below tolerance of %e", nerr, ntol)
    ##                    raise StopIteration

    ##                return nerr

    ##            try:
    ##                scipy.optimize.brentq(electron_error, cptmin, cptmax)
    ##            except StopIteration:
    ##                pass

    ##            # Reset hcore Hamiltonian
    ##            casci.get_hcore = h1e_func

    ##        #assert np.allclose(mo_coeff_casci, mo_coeff)
    ##        #dma, dmb = casci.make_rdm1s()
    ##        #log.debug("Alpha: %r", np.diag(dma))
    ##        #log.debug("Beta: %r", np.diag(dmb))
    ##        log.debug("FCI done. converged: %r", casci.converged)
    ##        #log.debug("Shape of WF: %r", list(wf.shape))
    ##        cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, 2*nocc_active)
    ##        C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc_active)
    ##        # Intermediate normalization
    ##        log.debug("Weight of reference determinant = %.8e", C0)
    ##        renorm = 1/C0
    ##        C1 *= renorm
    ##        C2 *= renorm

    ##        converged = casci.converged
    ##        e_corr_full = self.energy_factor*(e_tot - self.mf.e_tot)

    ##        # Create fake CISD object
    ##        cisd = pyscf.ci.CISD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)

    ##        if eris is None:
    ##            t0 = MPI.Wtime()
    ##            eris = cisd.ao2mo()
    ##            log.debug("Time for integral transformation: %s", get_time_string(MPI.Wtime()-t0))

    ##        pC1, pC2 = self.get_local_amplitudes(cisd, C1, C2)
    ##        e_corr = self.get_local_energy(cisd, pC1, pC2, eris=eris)

    ##    else:
    ##        raise ValueError("Unknown solver: %s" % solver)

    ##    # Store integrals for iterations
    ##    if self.maxiter > 1:
    ##        self.eris = eris

    ##    if solver == "FCI":
    ##        # Store amplitudes
    ##        #S = self.mf.get_ovlp()
    ##        # Rotation from occupied to local+DMET
    ##        #Ro = np.linalg.multi_dot((np.hstack((self.C_local, self.C_bath)).T, S, self.C_occclst))
    ##        #assert Ro.shape[0] == Ro.shape[1]
    ##        # Rotation from virtual to local+DMET
    ##        #Rv = np.linalg.multi_dot((np.hstack((self.C_local, self.C_bath)).T, S, self.C_virclst))
    ##        #assert Rv.shape[0] == Rv.shape[1]
    ##        #rC1, rC2 = self.transform_amplitudes((Ro, Rv, pC1, pC2))
    ##        #log.debug("Transforming C2 amplitudes: %r -> %r", list(pC2.shape), list(rC2.shape))
    ##        #self.amplitudes["C_occ"] = mo_coeff[:,mo_occ>0]
    ##        #self.amplitudes["C_vir"] = mo_coeff[:,mo_occ==0]
    ##        self.amplitudes["C1"] = C1
    ##        self.amplitudes["C2"] = C2

    ##    log.debug("Full cluster correlation energy = %.10g htr", e_corr_full)
    ##    self.e_corr_full = e_corr_full

    ##    self.converged = converged
    ##    if self.e_corr != 0.0:
    ##        log.debug("dEcorr=%.8g", (e_corr-self.e_corr))
    ##    self.e_corr = e_corr


    ##    #self.e_corr_dmp2 = e_corr + self.e_delta_mp2

    ##    #self.e_corr_v = e_corr_v
    ##    #self.e_corr_d = e_corr_d

    ##    # RDM1
    ##    if False:
    ##    #if True and solver:

    ##        #if solver != "FCI":
    ##        if not solver.startswith("FCI"):
    ##            dm1 = solverobj.make_rdm1()
    ##            dm2 = solverobj.make_rdm2()
    ##        else:
    ##            dm1, dm2 = pyscf.mcscf.addons.make_rdm12(solverobj, ao_repr=False)

    ##        px = self.get_local_projector(mo_coeff)

    ##        # DMET energy
    ##        if True:
    ##            dm1x = np.einsum("ai,ij->aj", px, dm1)
    ##            dm2x = np.einsum("ai,ijkl->ajkl", px, dm2)
    ##            #h1e = np.einsum('pi,pq,qj->ij', mo_coeff, self.mf.get_hcore(), mo_coeff)
    ##            h1e = np.einsum('pi,pq,qj->ij', mo_coeff, self.base.get_hcore(), mo_coeff)
    ##            nmo = mo_coeff.shape[-1]
    ##            eri = pyscf.ao2mo.kernel(self.mol, mo_coeff, compact=False).reshape([nmo]*4)
    ##            self.e_dmet = self.energy_factor*(np.einsum('pq,qp', h1e, dm1x) + np.einsum('pqrs,pqrs', eri, dm2x) / 2)
    ##            log.debug("E(DMET) = %e", self.e_dmet)

    ##            # TEST full energy
    ##            #e_full = self.energy_factor*(np.einsum('pq,qp', h1e, dm1) + np.einsum('pqrs,pqrs', eri, dm2) / 2) + self.mol.energy_nuc()
    ##            #log.debug("E(full) = %16.8g htr, reference = %16.8g htr", e_full, solverobj.e_tot)
    ##            #assert np.isclose(e_full, solverobj.e_tot)


    ##        # Count fragment electrons
    ##        dm1xx = np.einsum("ai,ij,bj->ab", px, dm1, px)
    ##        n = np.trace(dm1)
    ##        nx = np.trace(dm1xx)
    ##        log.info("Number of local/total electrons: %12.8f / %12.8f ", nx, n)

    ##        self.nelectron_corr_x = nx

    ##    return converged, e_corr

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

        t0_bath = t0 = MPI.Wtime()
        log.info("MAKING DMET BATH")
        log.info("****************")
        log.changeIndentLevel(1)
        #C_bath, C_occenv, C_virenv = self.make_dmet_bath(C_ref=ref_orbitals.get("dmet-bath", None))
        C_bath, C_occenv, C_virenv = self.make_dmet_bath(C_ref=refdata.get("dmet-bath", None), tol=self.dmet_bath_tol)
        log.debug("Time for DMET bath: %s", get_time_string(MPI.Wtime()-t0))
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
        t0 = MPI.Wtime()
        log.changeIndentLevel(1)
        C_occbath, C_occenv2, e_delta_occ, occbath_eigref = self.make_bath(
                C_occenv, self.bath_type, "occ",
                C_ref=C_occref, eigref=occbath_eigref,
                # New for MP2 bath:
                C_occenv=C_occenv, C_virenv=C_virenv,
                nbath=self.bath_target_size[0], tol=self.bath_tol[0], energy_tol=self.bath_energy_tol[0])
        log.debug("Time for occupied %r bath: %s", self.bath_type, get_time_string(MPI.Wtime()-t0))
        log.changeIndentLevel(-1)

        log.info("MAKING VIRTUAL BATH")
        log.info("*******************")
        t0 = MPI.Wtime()
        log.changeIndentLevel(1)
        C_virbath, C_virenv2, e_delta_vir, virbath_eigref = self.make_bath(
                C_virenv, self.bath_type, "vir",
                C_ref=C_virref, eigref=virbath_eigref,
                # New for MP2 bath:
                C_occenv=C_occenv, C_virenv=C_virenv,
                nbath=self.bath_target_size[1], tol=self.bath_tol[1], energy_tol=self.bath_energy_tol[1])
        log.debug("Time for virtual %r bath: %s", self.bath_type, get_time_string(MPI.Wtime()-t0))
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
        log.debug("Wall time for bath: %s", get_time_string(MPI.Wtime()-t0_bath))

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
        t0 = MPI.Wtime()
        log.changeIndentLevel(1)
        csolver = ClusterSolver(self, solver, mo_coeff, mo_occ, active=active, frozen=frozen)
        csolver.run()
        self.converged = csolver.converged
        self.e_corr_full = csolver.e_corr
        log.debug("Wall time for %s solver: %s", csolver.solver, get_time_string(MPI.Wtime()-t0))

        pc1, pc2 = self.get_local_amplitudes(csolver._solver, csolver.c1, csolver.c2)
        self.e_corr = self.get_local_energy(csolver._solver, pc1, pc2, eris=csolver._eris)

        # Population analysis
        if self.opts.make_rdm1 and csolver.dm1 is not None:
            self.pop_analysis(csolver.dm1)

        log.changeIndentLevel(-1)

        return self.converged, self.e_corr

    def pop_analysis(self, dm1, sig_tol=0.01):
        """Perform population analsis for the given density-matrix and compare to the MF."""
        s = self.base.get_ovlp()
        lo = self.base.lo
        dm1 = np.linalg.multi_dot((lo.T, s, dm1, s, lo))
        pop, chg = self.mf.mulliken_pop(dm=dm1, s=np.eye(dm1.shape[-1]))
        log.info("Population analysis")
        log.info("*******************")
        pop_mf = self.base.pop_mf
        chg_mf = self.base.pop_mf_chg
        # per orbital
        for i, s in enumerate(self.mf.mol.ao_labels()):
            dmf = (pop[i]-pop_mf[i])
            sig = (" !" if abs(dmf)>=sig_tol else "")
            log.info("  * Population of OrthAO %4d %-16s = %10.5f , delta(MF)= %+10.5f%s", i, s, pop[i], dmf, sig)
        # Charge per atom
        log.info("Atomic charges")
        log.info("**************")
        for ia in range(self.mf.mol.natm):
            symb = self.mf.mol.atom_symbol(ia)
            dmf = (chg[ia]-chg_mf[ia])
            sig = (" !" if abs(dmf)>=sig_tol else "")
            log.info("  * Charge at atom %3d %-3s = %10.5f , delta(MF)= %+10.5f%s", ia, symb, chg[ia], dmf, sig)

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
