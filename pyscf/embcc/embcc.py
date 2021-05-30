import logging
import os.path
import functools
from datetime import datetime
import dataclasses

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.lo
import pyscf.scf
import pyscf.pbc
import pyscf.pbc.tools

from . import util
from .util import *
from .localao import localize_ao
from . import helper
from .qemb import QEmbeddingMethod
from .fragment import Fragment

try:
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    timer = MPI.Wtime
except ImportError:
    MPI = False
    MPI_rank = 0
    MPI_size = 1
    from timeit import default_timer as timer

log = logging.getLogger(__name__)


@dataclasses.dataclass
class EmbCCOptions(Options):
    """Options for EmbCC calculations."""
    # --- Fragment settings
    fragment_type: str = 'IAO'
    localize_fragment: bool = False     # Perform numerical localization on fragment orbitals
    iao_minao : str = 'auto'            # Minimal basis for IAOs
    # --- Bath settings
    dmet_threshold: float = 1e-4
    orbfile: str = None                 # Filename for orbital coefficients
    # If multiple bno thresholds are to be calculated, we can project integrals and amplitudes from a previous larger cluster:
    project_eris: bool = False          # Project ERIs from a pervious larger cluster (corresponding to larger eta), can result in a loss of accuracy especially for large basis sets!
    project_init_guess: bool = True     # Project converted T1,T2 amplitudes from a previous larger cluster
    orthogonal_mo_tol: float = False
    #Orbital file
    plot_orbitals: bool = False
    plot_orbitals_dir: str = "orbitals"
    plot_orbitals_kwargs: dict = dataclasses.field(default_factory=dict)
    # --- Solver settings
    solver_options: dict = dataclasses.field(default_factory=dict)
    make_rdm1: bool = False
    popfile: str = "population"         # Filename for population analysis
    eom_ccsd: bool = False              # Perform EOM-CCSD in each cluster by default
    eomfile: str = "eom-ccsd"           # Filename for EOM-CCSD states

    # --- Other
    energy_partition: str = 'first-occ'
    strict: bool = False                # Stop if cluster not converged


VALID_SOLVERS = [None, "", "MP2", "CISD", "CCSD", "CCSD(T)", 'FCI', "FCI-spin0", "FCI-spin1"]

class EmbCC(QEmbeddingMethod):


    def __init__(self, mf, solver='CCSD', bno_threshold=1e-8, **kwargs):
        """Embedded CCSD calcluation object.

        Parameters
        ----------
        mf : pyscf.scf object
            Converged mean-field object.
        solver : str, optional
            Solver for embedding problem. Default: 'CCSD'.
        bno_threshold : float, optiona
            Bath natural orbital threshold. Default: 1e-8.
        **kwargs :
            See class `EmbCCOptions` for additional options.
        """
        t_start = timer()

        log.info("INITIALIZING EmbCC")
        log.info("******************")
        log.changeIndentLevel(1)

        self.opts = EmbCCOptions(**kwargs)
        log.info("EmbCC parameters:")
        for key, val in self.opts.items():
            log.info('  * %-24s %r', key + ':', val)

        super(EmbCC, self).__init__(mf)

        # --- Check input
        if not mf.converged:
            if self.opts.strict:
                raise RuntimeError("Mean-field calculation not converged.")
            else:
                log.error("Mean-field calculation not converged.")
        if solver not in VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver

        # Minimal basis for IAO
        if self.opts.iao_minao == 'auto':
            self.opts.iao_minao = helper.get_minimal_basis(self.mol.basis)
            log.warning("Minimal basis set %s for IAOs was selected automatically.",  self.opts.iao_minao)
        log.info("Computational basis= %s", self.mol.basis)
        log.info("Minimal basis=       %s", self.opts.iao_minao)

        # Bath natural orbital threshold
        if np.isscalar(bno_threshold):
            bno_threshold = [bno_threshold]
        self.bno_threshold = bno_threshold

        # Orthogonalize insufficiently orthogonal MOs
        # (For example as a result of k2gamma conversion with low cell.precision)
        c = self.mo_coeff.copy()
        assert np.all(c.imag == 0), "max|Im(C)|= %.2e" % abs(c.imag).max()
        ctsc = np.linalg.multi_dot((c.T, self.get_ovlp(), c))
        nonorth = abs(ctsc - np.eye(ctsc.shape[-1])).max()
        log.info("Max. non-orthogonality of input orbitals= %.2e%s", nonorth, " (!!!)" if nonorth > 1e-5 else "")
        if self.opts.orthogonal_mo_tol and nonorth > self.opts.orthogonal_mo_tol:
            t0 = timer()
            log.info("Orthogonalizing orbitals...")
            self.mo_coeff = helper.orthogonalize_mo(c, self.get_ovlp())
            change = abs(np.diag(np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), c)))-1)
            log.info("Max. orbital change= %.2e%s", change.max(), " (!!!)" if change.max() > 1e-4 else "")
            log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))

        # Prepare fragments
        #if self.local_orbital_type in ("IAO", "LAO"):
        if self.opts.fragment_type in ("IAO", "LAO"):
            t0 = timer()
            self.init_fragments()
            log.timing("Time for fragment initialization: %s", time_string(timer()-t0))

        log.timing("Time for EmbCC setup: %s", time_string(timer()-t_start))
        log.changeIndentLevel(-1)

        # Intermediate and output attributes
        self.clusters = []
        self.e_corr = 0.0           # Correlation energy
        self.e_pert_t = 0.0         # CCSD(T) correction
        self.e_delta_mp2 = 0.0      # MP2 correction

    def init_fragments(self):
        if self.opts.fragment_type == "IAO":
            self.C_ao, self.C_env, self.iao_labels = self.make_iao_coeffs(minao=self.opts.iao_minao)
            #log.debug("IAO labels:")
            #for ao in self.iao_labels:
            #    log.debug("%r", ao)
            self.ao_labels = self.iao_labels
        elif self.opts.fragment_type == "LAO":
            self.C_ao, self.lao_labels = self.make_lowdin_ao()
            self.ao_labels = self.lao_labels

        locmethod = self.opts.localize_fragment
        if locmethod:
            log.debug("Localize fragment orbitals with %s method", locmethod)

            #orbs = {self.ao_labels[i] : self.C_ao[:,i:i+1] for i in range(self.C_ao.shape[-1])}
            #orbs = {"A" : self.C_ao}
            #create_orbital_file(self.mol, "%s.molden" % self.local_orbital_type, orbs)
            coeffs = self.C_ao
            names = [("%d-%s-%s-%s" % l).rstrip("-") for l in self.ao_labels]
            #create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment")
            create_orbital_file(self.mol, self.opts.fragment_type, coeffs, names, directory="fragment", filetype="cube")

            t0 = timer()
            if locmethod in ("BF", "ER", "PM"):
                localizer = getattr(pyscf.lo, locmethod)(self.mol)
                localizer.init_guess = None
                #localizer.pop_method = "lowdin"
                C_loc = localizer.kernel(self.C_ao, verbose=4)
            elif locmethod == "LAO":
                #centers = [l[0] for l in self.mol.ao_labels(None)]
                centers = [l[0] for l in self.ao_labels]
                log.debug("Atom centers: %r", centers)
                C_loc = localize_ao(self.mol, self.C_ao, centers)

            #C_loc = locfunc(self.mol).kernel(self.C_ao, verbose=4)
            log.timing("Time for orbital localization: %s", time_string(timer()-t0))
            assert C_loc.shape == self.C_ao.shape
            # Check that all orbitals kept their fundamental character
            chi = np.einsum("ai,ab,bi->i", self.C_ao, self.get_ovlp(), C_loc)
            log.info("Diagonal of AO-Loc(AO) overlap: %r", chi)
            log.info("Smallest value: %.3g" % np.amin(chi))
            #assert np.all(chi > 0.5)
            self.C_ao = C_loc

            #orbs = {"A" : self.C_ao}
            #orbs = {self.ao_labels[i] : self.C_ao[:,i:i+1] for i in range(self.C_ao.shape[-1])}
            #create_orbital_file(self.mol, "%s-local.molden" % self.local_orbital_type, orbs)
            #raise SystemExit()

            coeffs = self.C_ao
            names = [("%d-%s-%s-%s" % l).rstrip("-") for l in self.ao_labels]
            #create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment-localized")
            create_orbital_file(self.mol, self.opts.fragment_type, coeffs, names, directory="fragment-localized", filetype="cube")


    @property
    def nclusters(self):
        """Number of cluster."""
        return len(self.clusters)

    @property
    def ncalc(self):
        """Number of calculations in each cluster."""
        return len(self.bno_threshold)

    @property
    def e_tot(self):
        """Total energy."""
        return self.e_mf + self.e_corr

    # -------------------------------------------------------------------------------------------- #

    def make_ao_projector(self, ao_indices):
        """Create projector into AO subspace.

        Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b

        Parameters
        ----------
        ao_indices : list
            Indices of subspace AOs.

        Returns
        -------
        P : ndarray
            Projector into AO subspace.
        """
        S1 = self.get_ovlp()
        S2 = S1[np.ix_(ao_indices, ao_indices)]
        S21 = S1[ao_indices]
        P21 = scipy.linalg.solve(S2, S21, assume_a="pos")
        P = np.dot(S21.T, P21)
        assert np.allclose(P, P.T)
        return P

    def make_ao_projector_general(self, ao_indices, ao_labels=None, basis2=None):
        """Create projector into AO space.

        Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b

        Parameters
        ----------
        ao_indices : list
            Indices of AOs in space 2.
        ao_labels : list, optional
            Labels for AOs in space 2. If not None, `ao_indices` is ignored.
        basis2 : str, optional
            Basis of space 2. If none, the same basis is used.

        Returns
        -------
        P : ndarray
            Projector into AO space.
        """
        mol1 = self.mol
        s1 = self.get_ovlp()
        if basis2 is not None:
            mol2 = mol1.copy()
            if getattr(mol2, 'rcut', None) is not None:
                mol2.rcut = None
            mol2.build(False, False, basis=basis2)

            if getattr(mol1, 'pbc_intor', None):  # cell object has pbc_intor method
                #from pyscf.pbc import gto as pbcgto
                # At the moment: Gamma point only
                s2 = np.asarray(mol2.pbc_intor('int1e_ovlp', hermi=1, kpts=None))
                s12 = np.asarray(pyscf.pbc.gto.cell.intor_cross('int1e_ovlp', mol1, mol2, kpts=None))
                assert s1.ndim == 2
                #s1, s2, s12 = s1[0], s2[0], s12[0]
            else:
                #s1 = mol1.intor_symmetric('int1e_ovlp')
                s2 = mol2.intor_symmetric('int1e_ovlp')
                s12 = pyscf.gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
        else:
            mol2 = mol1
            #if getattr(mol1, 'pbc_intor', None):  # cell object has pbc_intor method
            #    s1 = np.asarray(mol1.pbc_intor('int1e_ovlp', hermi=1, kpts=None))
            #else:
            #    s1 = mol1.intor_symmetric('int1e_ovlp')
            s2 = s12 = s1

        # Convert AO labels to AO indices
        if ao_labels is not None:
            log.debug("AO labels:\n%r", ao_labels)
            ao_indices = mol2.search_ao_label(ao_labels)
        assert (ao_indices == np.sort(ao_indices)).all()
        log.debug("Basis 2 contains AOs:\n%r", np.asarray(mol2.ao_labels())[ao_indices])

        # Restrict basis function of space 2
        s2 = s2[np.ix_(ao_indices, ao_indices)]
        s12 = s12[:,ao_indices]
        s21 = s12.T.conj()

        p21 = scipy.linalg.solve(s2, s21, assume_a="pos")
        p = np.dot(s21.T, p21)
        assert np.allclose(p, p.T)

        # TESTING
        if basis2 is None:
            p_test = self.make_ao_projector(ao_indices)
            assert np.allclose(p, p_test)

        return p


    def make_local_ao_orbitals(self, ao_indices):
        #S = self.mf.get_ovlp()
        S = self.get_ovlp()
        nao = S.shape[-1]
        P = self.make_ao_projector(ao_indices)
        e, C = scipy.linalg.eigh(P, b=S)
        e, C = e[::-1], C[:,::-1]
        size = len(e[e>1e-5])
        if size != len(ao_indices):
            raise RuntimeError("Error finding local orbitals. Eigenvalues: %s" % e)
        assert np.allclose(np.linalg.multi_dot((C.T, S, C)) - np.eye(nao), 0)
        C_local, C_env = np.hsplit(C, [size])

        return C_local, C_env

    def make_local_iao_orbitals(self, iao_indices):
        C_local = self.C_ao[:,iao_indices]
        #not_indices = np.asarray([i for i in np.arange(len(iao_indices)) if i not in iao_indices])
        not_indices = np.asarray([i for i in np.arange(self.C_ao.shape[-1]) if i not in iao_indices])

        if len(not_indices) > 0:
            C_env = np.hstack((self.C_ao[:,not_indices], self.C_env))
        else:
            C_env = self.C_env

        return C_local, C_env

    def make_local_lao_orbitals(self, lao_indices):
        # TODO: combine with IAO?
        C_local = self.C_ao[:,lao_indices]
        not_indices = np.asarray([i for i in np.arange(self.C_ao.shape[-1]) if i not in lao_indices])
        C_env = self.C_ao[:,not_indices]

        return C_local, C_env

    def make_local_nonorth_iao_orbitals(self, ao_indices, minao="minao"):
        C_occ = self.mo_coeff[:,self.mo_occ>0]
        C_ao = pyscf.lo.iao.iao(self.mol, C_occ, minao=minao)

        ao_labels = np.asarray(self.mol.ao_labels())[ao_indices]
        refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
        iao_labels = refmol.ao_labels()
        assert len(iao_labels) == C_ao.shape[-1]

        loc = np.isin(iao_labels, ao_labels)
        log.debug("Local NonOrth IAOs: %r", (np.asarray(iao_labels)[loc]).tolist())
        nlocal = np.count_nonzero(loc)
        log.debug("Number of local IAOs=%3d", nlocal)

        C_local = C_ao[:,loc]
        # Orthogonalize locally
        #S = self.mf.get_ovlp()
        S = self.get_ovlp()
        C_local = pyscf.lo.vec_lowdin(C_local, S)

        # Add remaining space
        # Transform to MO basis
        C_local_mo = np.linalg.multi_dot((self.mo_coeff.T, S, C_local))
        # Get eigenvectors of projector into complement
        P_local = np.dot(C_local_mo, C_local_mo.T)
        norb = self.mo_coeff.shape[-1]
        P_env = np.eye(norb) - P_local
        e, C = np.linalg.eigh(P_env)
        assert np.all(np.logical_or(abs(e) < 1e-10, abs(e)-1 < 1e-10))
        mask_env = (e > 1e-10)
        assert (np.sum(mask_env) + nlocal == norb)
        # Transform back to AO basis
        C_env = np.dot(self.mo_coeff, C[:,mask_env])

        # Test orthogonality
        C = np.hstack((C_local, C_env))
        assert np.allclose(C.T.dot(S).dot(C) - np.eye(norb), 0)

        return C_local, C_env

    # -------------------------------------------------------------------------------------------- #

    def make_cluster(self, name, C_local, C_env, **kwargs):
        """Create cluster object and add to list.

        Parameters
        ----------
        name : str
            Unique name for cluster.
        C_local : ndarray
            Local (fragment) orbitals of cluster.
        C_env : ndarray
            All environment (non-fragment) orbials.

        Returns
        -------
        cluster : Cluster
            Cluster object
        """
        #assert len(indices) > 0
        # Get new ID
        cluster_id = len(self.clusters) + 1
        # Check that ID is unique
        for cluster in self.clusters:
            if (cluster_id == cluster.id):
                raise RuntimeError("Cluster with ID %d already exists: %s" % (cluster_id, cluster.id_name))

        # Symmetry factor, if symmetry related fragments exist
        # TODO: Determine symmetry automatically
        kwargs["sym_factor"] = kwargs.get("sym_factor", 1.0)
        cluster = Fragment(self, fid=cluster_id, name=name,
                c_frag=C_local, c_env=C_env, **kwargs)
        self.clusters.append(cluster)
        return cluster

    def make_custom_cluster(self, ao_labels, name=None, **kwargs):
        """Each AO label can have multiple space separated strings"""
        if isinstance(ao_labels, str):
            ao_labels = [ao_labels]

        #aos = []
        #for ao in ao_labels:
        #    aos.append(ao.split())

        if name is None:
            #name = ",".join(["-".join(ao) for ao in aos])
            name = ";".join([",".join(ao_label.split()) for ao_label in ao_labels])

        # Orthogonal intrinsic AOs
        if self.opts.fragment_type == "IAO":

            indices = []
            refmol = pyscf.lo.iao.reference_mol(self.mol, minao=self.opts.iao_minao)
            for ao in ao_labels:
                ao_indices = refmol.search_ao_label(ao).tolist()
                log.debug("IAOs for label %s: %r", ao, (np.asarray(refmol.ao_labels())[ao_indices]).tolist())
                if not ao_indices:
                    raise ValueError("No orbitals found for label %s" % ao)
                indices += ao_indices

            #for idx, iao in enumerate(refmol.ao_labels(None)):
            #assert indices == indices2, "%r vs %r" % (indices, indices2)

            C_local, C_env = self.make_local_iao_orbitals(indices)
        else:
            raise NotImplementedError()

        #cluster = self.make_cluster(name, indices, C_local, C_env, **kwargs)
        cluster = self.make_cluster(name, C_local, C_env, **kwargs)
        return cluster

    def make_atom_cluster(self, atoms, name=None, check_atoms=True, **kwargs):
        """
        Parameters
        ---------
        atoms : list of int/str or int/str
            Atom labels of atoms in cluster.
        name : str
            Name of cluster.
        """
        # Atoms may be a single atom index/label
        #if isinstance(atoms, int) or isinstance(atoms, str):
        if not isinstance(atoms, (tuple, list, np.ndarray)):
            atoms = [atoms]

        # Check if atoms are valid labels of molecule
        atom_labels_mol = [self.mol.atom_symbol(atomid) for atomid in range(self.mol.natm)]
        if isinstance(atoms[0], str) and check_atoms:
            for atom in atoms:
                if atom not in atom_labels_mol:
                    raise ValueError("Atom with label %s not in molecule." % atom)

        # Get atom indices/labels
        if isinstance(atoms[0], (int, np.integer)):
            atom_indices = atoms
            atom_labels = [self.mol.atom_symbol(i) for i in atoms]
        else:
            atom_indices = np.nonzero(np.isin(atom_labels_mol, atoms))[0]
            atom_labels = atoms
        assert len(atom_indices) == len(atom_labels)

        # Generate cluster name if not given
        if name is None:
            name = ",".join(atom_labels)

        # Indices refers to AOs or IAOs, respectively

        # Non-orthogonal AOs
        if self.opts.fragment_type == "AO":
            # Base atom for each AO
            ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
            indices = np.nonzero(np.isin(ao_atoms, atoms))[0]
            C_local, C_env = self.make_local_ao_orbitals(indices)

        # Lowdin orthonalized AOs
        elif self.opts.fragment_type == "LAO":
            lao_atoms = [lao[1] for lao in self.lao_labels]
            indices = np.nonzero(np.isin(lao_atoms, atom_labels))[0]
            C_local, C_env = self.make_local_lao_orbitals(indices)

        # Orthogonal intrinsic AOs
        elif self.opts.fragment_type == "IAO":
            iao_atoms = [iao[0] for iao in self.iao_labels]
            indices = np.nonzero(np.isin(iao_atoms, atom_indices))[0]

            C_local, C_env = self.make_local_iao_orbitals(indices)

        # Non-orthogonal intrinsic AOs
        elif self.opts.fragment_type == "NonOrth-IAO":
            ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
            indices = np.nonzero(np.isin(ao_atoms, atom_labels))[0]
            C_local, C_env = self.make_local_nonorth_iao_orbitals(indices, minao=self.opts.iao_minao)

        # Projected molecular orbitals
        # (AVAS paper)
        elif self.opts.fragment_type == "PMO":
            #ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
            #indices = np.nonzero(np.isin(ao_atoms, atoms))[0]

            # Use atom labels as AO labels
            log.debug("Making occupied projector.")
            Po = self.make_ao_projector_general(None, ao_labels=atom_labels, basis2=kwargs.pop("basis_proj_occ", None))
            log.debug("Making virtual projector.")
            Pv = self.make_ao_projector_general(None, ao_labels=atom_labels, basis2=kwargs.pop("basis_proj_vir", None))
            log.debug("Done.")

            o = (self.mo_occ > 0)
            v = (self.mo_occ == 0)
            C = self.mo_coeff
            So = np.linalg.multi_dot((C[:,o].T, Po, C[:,o]))
            Sv = np.linalg.multi_dot((C[:,v].T, Pv, C[:,v]))
            eo, Vo = np.linalg.eigh(So)
            ev, Vv = np.linalg.eigh(Sv)
            rev = np.s_[::-1]
            eo, Vo = eo[rev], Vo[:,rev]
            ev, Vv = ev[rev], Vv[:,rev]
            log.debug("Non-zero occupied eigenvalues:\n%r", eo[eo>1e-10])
            log.debug("Non-zero virtual eigenvalues:\n%r", ev[ev>1e-10])
            #tol = 1e-8
            tol = 0.1
            lo = eo > tol
            lv = ev > tol
            Co = np.dot(C[:,o], Vo)
            Cv = np.dot(C[:,v], Vv)
            C_local = np.hstack((Co[:,lo], Cv[:,lv]))
            C_env = np.hstack((Co[:,~lo], Cv[:,~lv]))
            log.debug("Number of local orbitals: %d", C_local.shape[-1])
            log.debug("Number of environment orbitals: %d", C_env.shape[-1])

        #cluster = self.make_cluster(name, C_local, C_env, indices=indices, **kwargs)
        #indices = None
        #indices = list(range(C_local.shape[-1]))
        #cluster = self.make_cluster(name, indices, C_local, C_env, **kwargs)
        cluster = self.make_cluster(name, C_local, C_env, **kwargs)

        # TEMP
        #ao_indices = get_ao_indices_at_atoms(self.mol, atomids)
        ao_indices = helper.atom_labels_to_ao_indices(self.mol, atom_labels)
        cluster.ao_indices = ao_indices

        return cluster

    def make_all_atom_clusters(self, **kwargs):
        """Make a cluster for each atom in the molecule."""
        for atomid in range(self.mol.natm):
            atom_symbol = self.mol.atom_symbol(atomid)
            self.make_atom_cluster(atom_symbol, **kwargs)


    def get_cluster_attributes(self, attr):
        """Get attribute for each cluster."""
        attrs = []
        for cluster in self.clusters:
            attrs.append(getattr(cluster, attr))
        return attrs


    def set_cluster_attributes(self, attr, values):
        """Set attribute for each cluster."""
        log.debug("Setting attribute %s of all clusters", attr)
        for i, cluster in enumerate(self.clusters):
            setattr(cluster, attr, values[i])


    def make_lowdin_ao(self):
        S = self.get_ovlp()
        C_lao = pyscf.lo.vec_lowdin(np.eye(S.shape[-1]), S)
        lao_labels = self.mol.ao_labels(None)
        #lao_labels = self.mol.ao_labels()
        #lao_labels = self.mol.ao_labels("%d-%s-%s-%s")
        #lao_labels = [s.rstrip("-") for s in lao_labels]

        return C_lao, lao_labels

    def make_lowdin_ao_per_atom(self):
        #S = self.mf.get_ovlp()
        S = self.get_ovlp()
        C_lao = np.zeros_like(S)
        aorange = self.mol.aoslice_by_atom()
        for atomid in range(self.mol.natm):
            aos = aorange[atomid]
            aos = np.s_[aos[2]:aos[3]]
            Sa = S[aos,aos]
            Ca = pyscf.lo.vec_lowdin(np.eye(Sa.shape[-1]), Sa)
            C_lao[aos,aos] = Ca

        C_lao = pyscf.lo.vec_lowdin(C_lao, S)
        lao_labels = self.mol.ao_labels(None)
        return C_lao, lao_labels

    def pop_analysis(self, filename=None, mode="a"):
        if filename is None and self.opts.popfile:
            filename = "%s.txt" % self.opts.popfile
        mo = np.linalg.solve(self.lo, self.mf.mo_coeff)
        dm = self.mf.make_rdm1(mo, self.mf.mo_occ)
        pop, chg = self.mf.mulliken_pop(dm=dm, s=np.eye(dm.shape[-1]))

        if filename:
            tstamp = datetime.now()
            log.info("[%s] Writing mean-field population analysis to file \"%s\"", tstamp, filename)
            with open(filename, mode) as f:
                f.write("[%s] Mean-field population analysis\n" % tstamp)
                f.write("*%s********************************\n" % (26*"*"))
                # per orbital
                for i, s in enumerate(self.mol.ao_labels()):
                    f.write("  * MF population of OrthAO %4d %-16s = %10.5f\n" % (i, s, pop[i]))
                # per atom
                f.write("[%s] Mean-field atomic charges\n" % tstamp)
                f.write("*%s***************************\n" % (26*"*"))
                for ia in range(self.mol.natm):
                    symb = self.mol.atom_symbol(ia)
                    f.write("  * MF charge at atom %3d %-3s = %10.5f\n" % (ia, symb, chg[ia]))

        return pop, chg


    def kernel(self, **kwargs):

        if MPI: MPI_comm.Barrier()
        t_start = timer()

        if not self.clusters:
            raise ValueError("No clusters defined for calculation.")

        if self.opts.orbfile:
            filename = "%s.txt" % self.opts.orbfile
            tstamp = datetime.now()
            nfo = self.C_ao.shape[-1]
            #ao_labels = ["-".join(x) for x in self.mol.ao_labels(None)]
            ao_labels = ["-".join([str(xi) for xi in x]) for x in self.mol.ao_labels(None)]
            iao_labels = ["-".join([str(xi) for xi in x]) for x in self.iao_labels]
            #iao_labels = ["-".join(x) for x in self.iao_labels]
            log.info("[%s] Writing fragment orbitals to file \"%s\"", tstamp, filename)
            with open(filename, "a") as f:
                f.write("[%s] Fragment Orbitals\n" % tstamp)
                f.write("*%s*******************\n" % (26*"*"))
                # Header
                fmtline = "%20s" + nfo*"   %20s" + "\n"
                f.write(fmtline % ("AO", *iao_labels))
                fmtline = "%20s" + nfo*"   %+20.8e" + "\n"
                # Loop over AO
                for i in range(self.C_ao.shape[0]):
                    f.write(fmtline % (ao_labels[i], *self.C_ao[i]))

        # Mean-field population analysis
        self.lo = pyscf.lo.orth_ao(self.mol, "lowdin")
        self.pop_mf, self.pop_mf_chg = self.pop_analysis()

        nelec_frags = sum([x.sym_factor*x.nelectron for x in self.clusters])
        log.info("Total number of mean-field electrons over all fragments= %.8f", nelec_frags)
        if abs(nelec_frags - np.rint(nelec_frags)) > 1e-4:
            log.warning("Number of electrons not integer!")

        for idx, fragment in enumerate(self.clusters):
            if MPI_rank != (idx % MPI_size):
                continue

            mpi_info = (" on MPI process %3d" % MPI_rank) if MPI_size > 1 else ""
            msg = "RUNNING FRAGMENT %3d: %s%s" % (fragment.id, fragment.name, mpi_info.upper())
            log.info(msg)
            log.info(len(msg)*"*")
            log.changeIndentLevel(1)
            fragment.kernel(**kwargs)
            log.info("Fragment %3d: %s%s is done.", fragment.id, fragment.name, mpi_info)
            log.changeIndentLevel(-1)

        #results = self.collect_results("converged", "e_corr", "e_delta_mp2", "e_corr_v", "e_corr_d")
        #attributes = ["converged", "e_corr", "e_pert_t",
        #        #"e_pert_t2",
        #        "e_delta_mp2",
        #        "e_dmet", "e_corr_full", "e_corr_v", "e_corr_d",
        #        "nactive", "nfrozen"]

        attributes = ["converged", "e_corr", "e_delta_mp2", "e_pert_t", "nactive", "nfrozen"]

        results = self.collect_results(*attributes)
        if MPI_rank == 0 and not np.all(results["converged"]):
            log.critical("CRITICAL: The following fragments did not converge:")
            for i, x in enumerate(self.clusters):
                if not results["converged"][i]:
                    log.critical("%3d %s solver= %s", x.id, x.name, x.solver)
            if self.opts.strict:
                raise RuntimeError("Not all fragments converged")

        self.e_corr = sum(results["e_corr"])
        #self.e_pert_t = sum(results["e_pert_t"])
        #self.e_pert_t2 = sum(results["e_pert_t2"])
        self.e_delta_mp2 = sum(results["e_delta_mp2"])

        #self.e_corr_full = sum(results["e_corr_full"])

        if MPI_rank == 0:
            self.print_results(results)

        if MPI: MPI_comm.Barrier()
        log.info("Total wall time:  %s", time_string(timer()-t_start))

        log.info("All done.")

    # Alias for kernel
    run = kernel

    def collect_results(self, *attributes):
        """Use MPI to collect results from all fragments."""

        #log.debug("Collecting attributes %r from all clusters", (attributes,))
        clusters = self.clusters

        if MPI:
            def reduce_fragment(attr, op=MPI.SUM, root=0):
                res = MPI_comm.reduce(np.asarray([getattr(x, attr) for x in clusters]), op=op, root=root)
                return res
        else:
            def reduce_fragment(attr):
                res = np.asarray([getattr(x, attr) for x in clusters])
                return res

        results = {}
        for attr in attributes:
            results[attr] = reduce_fragment(attr)

        return results

    def show_cluster_sizes(self, results, show_largest=True):
        log.info("CLUSTER SIZES")
        log.info("*************")
        fmtstr = "  * %3d %-10s  :  active=%4d  frozen=%4d  ( %5.1f %%)"
        imax = [0]
        for i, x in enumerate(self.clusters):
            nactive = results["nactive"][i]
            nfrozen = results["nfrozen"][i]
            log.info(fmtstr, x.id, x.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.nmo)
            if i == 0:
                continue
            if nactive > results["nactive"][imax[0]]:
                imax = [i]
            elif nactive == results["nactive"][imax[0]]:
                imax.append(i)

        if show_largest and len(self.clusters) > 1:
            log.info("LARGEST CLUSTER")
            log.info("***************")
            for i in imax:
                x = self.clusters[i]
                nactive = results["nactive"][i]
                nfrozen = results["nfrozen"][i]
                log.info(fmtstr, x.id, x.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.nmo)

    def print_results(self, results):
        self.show_cluster_sizes(results)

        log.info("FRAGMENT ENERGIES")
        log.info("*****************")
        log.info("CCSD / CCSD+dMP2 / CCSD+dMP2+(T)")
        fmtstr = "  * %3d %-10s  :  %+16.8f Ha  %+16.8f Ha  %+16.8f Ha"
        for i, x in enumerate(self.clusters):
            e_corr = results["e_corr"][i]
            e_pert_t = results["e_pert_t"][i]
            e_delta_mp2 = results["e_delta_mp2"][i]
            log.info(fmtstr, x.id, x.trimmed_name(10), e_corr, e_corr+e_delta_mp2, e_corr+e_delta_mp2+e_pert_t)

        log.info("  * %-14s  :  %+16.8f Ha  %+16.8f Ha  %+16.8f Ha", "total", self.e_corr, self.e_corr+self.e_delta_mp2, self.e_corr+self.e_delta_mp2+self.e_pert_t)
        log.info("E(corr)= %+16.8f Ha", self.e_corr)
        log.info("E(tot)=  %+16.8f Ha", self.e_tot)

    def get_energies(self):
        """Get total energy."""
        energies = np.zeros(self.ncalc)
        energies[:] = self.e_mf
        for x in self.clusters:
            energies += x.e_corrs
        return energies

    def get_cluster_sizes(self):
        sizes = np.zeros((self.nclusters, self.ncalc), dtype=np.int)
        for ix, x in enumerate(self.clusters):
            sizes[ix] = x.n_active
        return sizes


    def print_clusters(self):
        """Print fragments of calculations."""
        log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for cluster in self.clusters:
            log.info("%3d  %20s  %8s  %4d", cluster.id, cluster.name, cluster.solver, cluster.size)

