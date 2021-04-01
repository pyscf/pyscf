import logging
import os.path
import functools
from datetime import datetime

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.lo
import pyscf.scf

from .util import *
from .cluster import Cluster
from .localao import localize_ao

log = logging.getLogger(__name__)

try:
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    log.debug("mpi4py found. MPI rank/size= %3d / %3d", MPI_rank, MPI_size)
    timer = MPI.Wtime
except (ImportError, ModuleNotFoundError):
    MPI = False
    MPI_rank = 0
    MPI_size = 1
    log.debug("mpi4py not found.")
    from timeit import default_timer as timer

__all__ = ["EmbCC",]

class EmbCC:
    """What should this be named?"""

    VALID_LOCAL_TYPES = ["AO", "IAO", "LAO", "NonOrth-IAO", "PMO"]
    VALID_SOLVERS = [None, "MP2", "CISD", "CCSD", "CCSD(T)", "FCI-spin0", "FCI-spin1"]
    VALID_BATH_TYPES = [
            None,
            "local",
            "power", "matsubara",
            "mp2-natorb",
            #"mp2-natorb-2", "mp2-natorb-3",
            #"mp2-natorb-4",
            "full", "random"]

    # These optionals are automatically transferred to any created cluster object
    default_options = [
            "solver",
            "use_ref_orbitals_dmet",
            "use_ref_orbitals_bath",
            "mp2_correction",
            "maxiter",
            # BATH
            "bath_type",
            "bath_size",
            "bath_tol",
            "bath_tol_per_electron",
            "bath_energy_tol",
            "dmet_bath_tol",
            "power1_occ_bath_tol",
            "power1_vir_bath_tol",
            "local_occ_bath_tol",
            "local_vir_bath_tol",
            ]

    def __init__(self, mf,
            local_type="IAO",       # TODO: rename, fragment_type?
            solver="CCSD",
            bath_type="mp2-natorb",
            bath_size=None,
            bath_tol=None,
            bath_tol_per_electron=False,
            bath_energy_tol=1e-3,
            #minao="minao",
            minao=None,
            use_ref_orbitals_dmet=True,
            use_ref_orbitals_bath=False,
            mp2_correction=True,
            maxiter=1,
            dmet_bath_tol=1e-4,
            energy_part="first-occ",
            # Perform numerical localization on fragment orbitals
            localize_fragment=False,
            # Additional bath orbitals
            power1_occ_bath_tol=False, power1_vir_bath_tol=False, local_occ_bath_tol=False, local_vir_bath_tol=False,
            **kwargs
            ):
        """
        Parameters
        ----------
        mf : pyscf.scf object
            Converged mean-field object.
        minao :
            Minimal basis for intrinsic atomic orbitals (IAO).
        dmet_bath_tol : float, optional
            Tolerance for DMET bath orbitals; orbitals with occupation larger than `dmet_bath_tol`,
            or smaller than 1-`dmet_bath_tol` are included as bath orbitals.


        make_rdm1 : [True, False]
            Calculate RDM1 in cluster.
        eom_ccsd : [True, False, "IP", "EA"]
            Default: False.
        """

        log.info("INITIALIZING EmbCC")
        log.info("******************")
        log.changeIndentLevel(1)

        # --- Check input
        if not mf.converged:
            log.warning("Mean-field calculation not converged.")

        # Local orbital types:
        # AO : Atomic orbitals non-orthogonal wrt to non-fragment AOs
        # IAO : Intrinstric atom-orbitals
        # LAO : Lowdin orthogonalized AOs
        if local_type not in self.VALID_LOCAL_TYPES:
            raise ValueError("Unknown local_type: %s" % local_type)
        if solver not in self.VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        #if bath_type not in (None, "power", "matsubara", "uncontracted", "mp2-natorb"):
        if bath_type not in self.VALID_BATH_TYPES:
            raise ValueError("Unknown bath type: %s" % bath_type)

        # Convert KRHF to Gamma-point RHF
        if isinstance(mf, pyscf.pbc.scf.KRHF) or isinstance(mf.mo_coeff, list):
            log.info("Converting KRHF to gamma-point RHF calculation")
            assert np.allclose(mf.kpts[0], 0)
            mf_g = pyscf.pbc.scf.RHF(mf.cell)
            mf_g.kpts = mf.kpts[0]
            mf_g.mo_energy = mf.mo_energy[0].copy()
            mf_g.mo_occ = mf.mo_occ[0].copy()
            mf_g.mo_coeff = mf.mo_coeff[0].copy()
            mf_g.with_df = mf.with_df
            mf = mf_g

        self.mf = mf
        self.mo_energy = mf.mo_energy.copy()
        self.mo_occ = mf.mo_occ.copy()
        self.mo_coeff = mf.mo_coeff.copy()

        self.max_memory = self.mf.max_memory

        self.local_orbital_type = local_type

        # Minimal basis for IAO
        default_minao = {
                "gth-dzv" : "gth-szv",
                "gth-dzvp" : "gth-szv",
                "gth-tzvp" : "gth-szv",
                "gth-tzv2p" : "gth-szv",
                }
        if minao is None:
            log.warning("No minimal basis for IAOs specified.")
            minao = default_minao.get(self.mol.basis, "minao")
            log.info("Computational basis= %s -> using minimal basis= %s", self.mol.basis, minao)
        self.minao = minao
        log.info("Computational basis= %s", self.mol.basis)
        log.info("Minimal basis=       %s", self.minao)

        self.maxiter = maxiter
        self.energy_part = energy_part

        # New implementation of options
        # TODO: Change other options to here
        self.opts = Options()
        default_opts = {
                # Bath settings
                "prim_mp2_bath_tol_occ" : False,
                "prim_mp2_bath_tol_vir" : False,
                #"orthogonal_mo_tol" : 1e-8,
                "orbfile" : None,         # Filename for orbital coefficients
                "orthogonal_mo_tol" : False,
                # Population analysis
                "make_rdm1" : False,
                "popfile" : "population",       # Filename for population analysis
                # EOM-CCSD
                "eom_ccsd" : False,
                "eomfile" : "eom-ccsd",         # Filename for EOM-CCSD states
                # Other
                "strict" : False,               # Stop if cluster not converged
                }
        for key, val in default_opts.items():
            setattr(self.opts, key, kwargs.pop(key, val))
        if kwargs:
            raise ValueError("Unknown arguments: %r" % kwargs.keys())

        # Options
        self.solver = solver

        self.bath_type = bath_type
        self.bath_size = bath_size
        self.bath_tol = bath_tol
        self.bath_tol_per_electron = bath_tol_per_electron
        self.bath_energy_tol = bath_energy_tol
        self.use_ref_orbitals_dmet = use_ref_orbitals_dmet
        self.use_ref_orbitals_bath = use_ref_orbitals_bath
        self.mp2_correction = mp2_correction
        self.dmet_bath_tol = dmet_bath_tol
        self.localize_fragment = localize_fragment
        # Additional bath orbitals
        self.power1_occ_bath_tol = power1_occ_bath_tol
        self.power1_vir_bath_tol = power1_vir_bath_tol
        self.local_occ_bath_tol = local_occ_bath_tol
        self.local_vir_bath_tol = local_vir_bath_tol

        self.clusters = []

        # Correlation energy
        self.e_corr = 0.0
        # CCSD(T) correction
        self.e_pert_t = 0.0
        # MP2 correction
        self.e_delta_mp2 = 0.0

        # Full cluster correlation energy. Only makes sense for a single cluster, otherwise double counting!
        self.e_corr_full = 0.0

        # [TESTING]
        # Global amplitudes
        self.T1 = None
        self.T2 = None
        # For tailored CC
        self.tccT1 = None
        self.tccT2 = None

        # Fock matrix
        # These two fock matrices are different for loose values of cell.precision
        # Which should be used?
        # (We only need Fock matrix for canonicalization, therefore not too important to be accurate for CCSD.
        # However, what about MP2 & CCSD(T)?)
        recalc_fock = False
        if recalc_fock:
            t0 = timer()
            self._fock = self.mf.get_fock()
            log.debug("Time for Fock matrix: %s", get_time_string(timer()-t0))
        else:
            cs = np.dot(self.mo_coeff.T, self.get_ovlp())
            #log.debug("SHAPES: %r %r %r %r", list(self.mo_coeff.shape), list(self.get_ovlp().shape), list(cs.shape), list(self.mo_energy.shape))
            self._fock = np.einsum("ia,i,ib->ab", cs, self.mo_energy, cs)

        # Orthogonalize insufficiently orthogonal MOs
        # (For example as a result of k2gamma conversion with low cell.precision)
        c = self.mo_coeff.copy()
        assert np.allclose(c.imag, 0)
        ctsc = np.linalg.multi_dot((c.T, self.get_ovlp(), c))
        nonorth = abs(ctsc - np.eye(ctsc.shape[-1])).max()
        log.info("Max. non-orthogonality of input orbitals= %.2e%s", nonorth, " (!!!)" if nonorth > 1e-4 else "")
        if self.opts.orthogonal_mo_tol and nonorth > self.opts.orthogonal_mo_tol:
            log.info("Orthogonalizing orbitals...")
            self.mo_coeff = orthogonalize_mo(c, self.get_ovlp(), tol=1e-6)
            change = abs(np.diag(np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), c)))-1)
            log.info("Max. orbital change= %.2e%s", change.max(), " (!!!)" if change.max() > 1e-2 else "")

        # Prepare fragments
        if self.local_orbital_type in ("IAO", "LAO"):
            self.init_fragments()


        log.changeIndentLevel(-1)

    def init_fragments(self):
        if self.local_orbital_type == "IAO":
            self.C_ao, self.C_env, self.iao_labels = self.make_iao(minao=self.minao)
            #log.debug("IAO labels:")
            #for ao in self.iao_labels:
            #    log.debug("%r", ao)
            self.ao_labels = self.iao_labels
        elif self.local_orbital_type == "LAO":
            self.C_ao, self.lao_labels = self.make_lowdin_ao()
            self.ao_labels = self.lao_labels

        if self.localize_fragment:
            log.debug("Localize fragment orbitals with %s method", self.localize_fragment)

            #orbs = {self.ao_labels[i] : self.C_ao[:,i:i+1] for i in range(self.C_ao.shape[-1])}
            #orbs = {"A" : self.C_ao}
            #create_orbital_file(self.mol, "%s.molden" % self.local_orbital_type, orbs)
            coeffs = self.C_ao
            names = [("%d-%s-%s-%s" % l).rstrip("-") for l in self.ao_labels]
            #create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment")
            create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment", filetype="cube")

            t0 = timer()
            if self.localize_fragment in ("BF", "ER", "PM"):
                #locfunc = getattr(pyscf.lo, self.localize_fragment)
                localizer = getattr(pyscf.lo, self.localize_fragment)(self.mol)
                localizer.init_guess = None
                #localizer.pop_method = "lowdin"
                C_loc = localizer.kernel(self.C_ao, verbose=4)
            elif self.localize_fragment == "LAO":
                #centers = [l[0] for l in self.mol.ao_labels(None)]
                centers = [l[0] for l in self.ao_labels]
                log.debug("Atom centers: %r", centers)
                C_loc = localize_ao(self.mol, self.C_ao, centers)

            #C_loc = locfunc(self.mol).kernel(self.C_ao, verbose=4)
            log.debug("Time for orbital localization: %s", get_time_string(timer()-t0))
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
            create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment-localized", filetype="cube")

    @property
    def mol(self):
        return self.mf.mol

    @property
    def norb(self):
        return self.mol.nao_nr()

    def get_ovlp(self, *args, **kwargs):
        return self.mf.get_ovlp(*args, **kwargs)

    def get_hcore(self, *args, **kwargs):
        return self.mf.get_hcore(*args, **kwargs)

    def get_fock(self):
        return self._fock

    @property
    def nclusters(self):
        """Number of cluster."""
        return len(self.clusters)

    @property
    def e_tot(self):
        """Total energy."""
        return self.mf.e_tot + self.e_corr

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
        #S1 = self.mf.get_ovlp()
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
        if basis2 is not None:
            mol2 = mol1.copy()
            if getattr(mol2, 'rcut', None) is not None:
                mol2.rcut = None
            mol2.build(False, False, basis=basis2)

            if getattr(mol1, 'pbc_intor', None):  # cell object has pbc_intor method
                #from pyscf.pbc import gto as pbcgto
                # At the moment: Gamma point only
                kpts = None
                s1 = np.asarray(mol1.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
                s2 = np.asarray(mol2.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
                s12 = np.asarray(pyscf.pbc.gto.cell.intor_cross('int1e_ovlp', mol1, mol2, kpts=kpts))
                assert s1.ndim == 2
                #s1, s2, s12 = s1[0], s2[0], s12[0]
            else:
                s1 = mol1.intor_symmetric('int1e_ovlp')
                s2 = mol2.intor_symmetric('int1e_ovlp')
                s12 = pyscf.gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
        else:
            mol2 = mol1
            if getattr(mol1, 'pbc_intor', None):  # cell object has pbc_intor method
                s1 = np.asarray(mol1.pbc_intor('int1e_ovlp', hermi=1, kpts=None))
            else:
                s1 = mol1.intor_symmetric('int1e_ovlp')
            s2 = s1.copy()
            s12 = s1.copy()

        # Convert AO labels to AO indices
        if ao_labels is not None:
            log.debug("AO labels:\n%r", ao_labels)
            ao_indices = mol2.search_ao_label(ao_labels)
        assert (ao_indices == np.sort(ao_indices)).all()
        log.debug("Basis 2 contains AOs:\n%r", np.asarray(mol2.ao_labels())[ao_indices])

        #assert np.allclose(s1, mol1.get_ovlp())
        #log.debug("s1.shape: %r", list(s1.shape))
        #log.debug("ovlp.shape: %r", list(self.mf.get_ovlp().shape))
        #log.debug("norm: %e", np.linalg.norm(s1 - self.mf.get_ovlp()))
        #assert np.allclose(s1, self.mf.get_ovlp())
        assert np.allclose(s1, self.get_ovlp())
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

    #def make_cluster(self, name, indices, C_local, C_env, **kwargs):
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
            assert (cluster_id != cluster.id)
        # Make name unique
        for i in range(1, 100):
            name_i = name if i == 1 else "%s-%d" % (name, i)
            if name_i not in [x.name for x in self.clusters]:
                name = name_i
                break
        else:
            raise ValueError("Cluster with name %s already exists." % name)
        # Pass options to cluster object via keyword arguments
        for opt in self.default_options:
            kwargs[opt] = kwargs.get(opt, getattr(self, opt))
        # Symmetry factor, if symmetry related clusters exist in molecule (e.g. hydrogen rings)
        kwargs["symmetry_factor"] = kwargs.get("symmetry_factor", 1.0)
        #cluster = Cluster(self, cluster_id=cluster_id, name=name, C_local=C_local, C_env=C_env, **kwargs)
        cluster = Cluster(self, cluster_id=cluster_id, name=name,
                #indices=indices,
                C_local=C_local, C_env=C_env, **kwargs)
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
        if self.local_orbital_type == "IAO":

            indices = []
            refmol = pyscf.lo.iao.reference_mol(self.mol, minao=self.minao)
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
        if isinstance(atoms, int) or isinstance(atoms, str):
            atoms = [atoms]

        # Check if atoms are valid labels of molecule
        atom_labels_mol = [self.mol.atom_symbol(atomid) for atomid in range(self.mol.natm)]
        if isinstance(atoms[0], str) and check_atoms:
            for atom in atoms:
                if atom not in atom_labels_mol:
                    raise ValueError("Atom with label %s not in molecule." % atom)

        # Get atom indices/labels
        if isinstance(atoms[0], int):
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
        if self.local_orbital_type == "AO":
            # Base atom for each AO
            ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
            indices = np.nonzero(np.isin(ao_atoms, atoms))[0]
            C_local, C_env = self.make_local_ao_orbitals(indices)

        # Lowdin orthonalized AOs
        elif self.local_orbital_type == "LAO":
            lao_atoms = [lao[1] for lao in self.lao_labels]
            indices = np.nonzero(np.isin(lao_atoms, atom_labels))[0]
            C_local, C_env = self.make_local_lao_orbitals(indices)

        # Orthogonal intrinsic AOs
        elif self.local_orbital_type == "IAO":
            # Base atom for each IAO
            #iao_atoms = [iao[1] for iao in self.iao_labels]
            #indices = np.nonzero(np.isin(iao_atoms, atom_labels))[0]
            # NEW: Atoms by index!
            iao_atoms = [iao[0] for iao in self.iao_labels]
            indices = np.nonzero(np.isin(iao_atoms, atom_indices))[0]

            C_local, C_env = self.make_local_iao_orbitals(indices)

        # Non-orthogonal intrinsic AOs
        elif self.local_orbital_type == "NonOrth-IAO":
            ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
            indices = np.nonzero(np.isin(ao_atoms, atom_labels))[0]
            C_local, C_env = self.make_local_nonorth_iao_orbitals(indices, minao=self.minao)

        # Projected molecular orbitals
        # (AVAS paper)
        elif self.local_orbital_type == "PMO":
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
        ao_indices = atom_labels_to_ao_indices(self.mol, atom_labels)
        cluster.ao_indices = ao_indices

        return cluster

    def make_all_atom_clusters(self, **kwargs):
        """Make a cluster for each atom in the molecule."""
        for atomid in range(self.mol.natm):
            atom_symbol = self.mol.atom_symbol(atomid)
            self.make_atom_cluster(atom_symbol, **kwargs)

    def get_cluster_attributes(self, attr):
        """Get attribute for each cluster."""
        attrs = {}
        for cluster in self.clusters:
            attrs[cluster.name] = getattr(cluster, attr)
        return attrs

    def set_cluster_attributes(self, attr, values):
        """Set attribute for each cluster."""
        log.debug("Setting attribute %s of all clusters", attr)
        for cluster in self.clusters:
            setattr(cluster, attr, values[cluster.name])


    def get_nelectron_total(self):
        ne = self.get_cluster_attributes("nelectron_corr_x")
        ne = sum(ne.values())
        return ne

    #def get_orbitals(self):
    #    """Get orbitals of each cluster."""
    #    orbitals = {}
    #    for cluster in self.clusters:
    #        orbitals[cluster.name] = cluster.get_orbitals()
    #    return orbitals

    #def set_reference_orbitals(self, ref_orbitals):
    #    return self.set_cluster_attributes("ref_orbitals", ref_orbitals)

    def get_refdata(self):
        """Get refdata for future calculations."""
        refdata = {}
        for cluster in self.clusters:
            refdata[cluster.name] = cluster.get_refdata()
        return refdata

    def set_refdata(self, refdata):
        """Get refdata from previous calculations."""
        if refdata is None:
            return False
        for cluster in self.clusters:
            cluster.set_refdata(refdata[cluster.name])
        return True

    def make_iao(self, minao="minao"):
        """Make intrinsic atomic orbitals.

        Parameters
        ----------
        minao : str, optional
            Minimal basis set for IAOs.

        Returns
        -------
        C_iao : ndarray
            IAO coefficients.
        C_env : ndarray
            Remaining orbital coefficients.
        iao_atoms : list
            Atom ID for each IAO.
        """
        # Orthogonality of input mo_coeff
        mo_coeff = self.mo_coeff
        norb = mo_coeff.shape[-1]
        S = self.get_ovlp()
        nonorthmax = abs(mo_coeff.T.dot(S).dot(mo_coeff) - np.eye(norb)).max()
        log.debug("Max orthogonality error in canonical basis = %.1e" % nonorthmax)

        C_occ = self.mo_coeff[:,self.mo_occ>0]
        C_iao = pyscf.lo.iao.iao(self.mol, C_occ, minao=minao)
        niao = C_iao.shape[-1]
        log.debug("Total number of IAOs=%3d", niao)

        # Orthogonalize IAO
        #S = self.mf.get_ovlp()
        C_iao = pyscf.lo.vec_lowdin(C_iao, S)

        # Add remaining virtual space
        # Transform to MO basis
        C_iao_mo = np.linalg.multi_dot((self.mo_coeff.T, S, C_iao))
        # Get eigenvectors of projector into complement
        P_iao = np.dot(C_iao_mo, C_iao_mo.T)
        P_env = np.eye(norb) - P_iao
        e, C = np.linalg.eigh(P_env)
        # Tolerance for environment orbitals
        # Warning
        mask = np.logical_and(e>1e-6, e<(1-1e-6))
        if np.any(mask):
            log.warning("IAO states with large eigenvalues of Projector 1-P_IAO:\n%r", e[mask])

        tol = 1e-4
        assert np.all(np.logical_or(abs(e) < tol, abs(e)-1 < tol))
        mask_env = (e > tol)
        #assert (np.sum(mask_env) + niao == norb)
        if not (np.sum(mask_env) + niao == norb):
            log.critical("Eigenvalues of projector:\n%r", e)
            log.critical("Number of eigenvalues above %e = %d", tol, np.sum(mask_env))
            log.critical("Total number of orbitals = %d", norb)
            raise RuntimeError()
        # Transform back to AO basis
        C_env = np.dot(self.mo_coeff, C[:,mask_env])

        # Get base atoms of IAOs
        refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
        iao_labels = refmol.ao_labels(None)
        #iao_labels = refmol.ao_labels()
        assert len(iao_labels) == C_iao.shape[-1]

        # Test orthogonality
        C = np.hstack((C_iao, C_env))
        #assert np.allclose(C.T.dot(S).dot(C) - np.eye(norb), 0, 1e-5)
        ortherr = abs(C.T.dot(S).dot(C) - np.eye(norb)).max()
        log.debug("Max orthogonality error in rotated basis = %.1e" % ortherr)
        assert (ortherr < max(2*nonorthmax, 1e-7))
        assert (ortherr < 1e-4)

        # Check that all electrons are in IAO DM
        dm_iao = np.linalg.multi_dot((C_iao.T, S, self.mf.make_rdm1(), S, C_iao))
        nelec_iao = np.trace(dm_iao)
        log.debug("Total number of electrons in IAOs: %.8f", nelec_iao)
        if abs(nelec_iao - self.mol.nelectron) > 1e-4:
            log.error("ERROR: IAOs do not span entire occupied space.")

        # Print electron distribution
        log.info("MEAN-FIELD OCCUPANCY PER ATOM")
        log.info("*****************************")
        iao_atoms = np.asarray([i[0] for i in iao_labels])
        for a in range(self.mol.natm):
            mask = np.where(iao_atoms == a)[0]
            ne = np.trace(dm_iao[mask][:,mask])
            log.info("  * %3d: %-6s= %.8f", a, self.mol.atom_symbol(a), ne)

        return C_iao, C_env, iao_labels

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

        nelec_frags = sum([x.symmetry_factor*x.nelec_mf_frag for x in self.clusters])
        log.info("Total number of mean-field electrons over all fragments= %.8f", nelec_frags)
        if abs(nelec_frags - np.rint(nelec_frags)) > 1e-4:
            log.warning("WARNING: Number of electrons not integer!")

        for idx, cluster in enumerate(self.clusters):
            if MPI_rank != (idx % MPI_size):
                continue

            mpi_info = (" on MPI process %3d" % MPI_rank) if MPI_size > 1 else ""
            msg = "RUNNING CLUSTER %3d: %s%s" % (cluster.id, cluster.name, mpi_info.upper())
            log.info(msg)
            log.info(len(msg)*"*")
            log.changeIndentLevel(1)
            cluster.run(**kwargs)
            log.info("Cluster %3d: %s%s is done.", cluster.id, cluster.name, mpi_info)
            log.changeIndentLevel(-1)

        #results = self.collect_results("converged", "e_corr", "e_delta_mp2", "e_corr_v", "e_corr_d")
        attributes = ["converged", "e_corr", "e_pert_t",
                #"e_pert_t2",
                "e_delta_mp2",
                "e_dmet", "e_corr_full", "e_corr_v", "e_corr_d",
                "nactive", "nfrozen"]
        results = self.collect_results(*attributes)
        if MPI_rank == 0 and not np.all(results["converged"]):
            log.critical("CRITICAL: The following fragments did not converge:")
            for i, x in enumerate(self.clusters):
                if not results["converged"][i]:
                    log.critical("%3d %s solver= %s", x.id, x.name, x.solver)
            if self.opts.strict:
                raise RuntimeError("Not all cluster converged")

        self.e_corr = sum(results["e_corr"])
        self.e_pert_t = sum(results["e_pert_t"])
        #self.e_pert_t2 = sum(results["e_pert_t2"])
        self.e_delta_mp2 = sum(results["e_delta_mp2"])

        self.e_dmet = sum(results["e_dmet"]) + self.mol.energy_nuc()

        self.e_corr_full = sum(results["e_corr_full"])

        self.e_corr_v = sum(results["e_corr_v"])
        self.e_corr_d = sum(results["e_corr_d"])

        if MPI_rank == 0:
            self.print_results(results)

        if MPI: MPI_comm.Barrier()
        #log.info("Total wall time for EmbCC: %s", get_time_string(MPI.Wtime()-t_start))
        t_tot = (timer() - t_start)
        log.info("Total wall time [s]: %.5g (%s)", t_tot, get_time_string(t_tot))

        if self.maxiter > 1:
            for it in range(2, self.maxiter+1):

                self.tccT1 = self.T1
                self.tccT2 = self.T2
                self.T1 = None
                self.T2 = None

                for idx, cluster in enumerate(self.clusters):
                    if MPI_rank != (idx % MPI_size):
                        continue

                    # Only rerun the solver
                    log.debug("Running cluster %s on MPI process=%d...", cluster.name, MPI_rank)
                    cluster.run_solver()
                    log.debug("Cluster %s on MPI process=%d is done.", cluster.name, MPI_rank)

                #results = self.collect_results("converged", "e_corr", "e_delta_mp2", "e_corr_v", "e_corr_d")
                results = self.collect_results("converged", "e_corr", "e_delta_mp2", "e_corr_full", "e_corr_v", "e_corr_d")
                if MPI_rank == 0 and not np.all(results["converged"]):
                    log.debug("converged = %s", results["converged"])
                    raise RuntimeError("Not all cluster converged")

                self.e_corr = sum(results["e_corr"])
                self.e_delta_mp2 = sum(results["e_delta_mp2"])
                self.e_corr_full = sum(results["e_corr_full"])
                self.e_corr_v = sum(results["e_corr_v"])
                self.e_corr_d = sum(results["e_corr_d"])

                if MPI_rank == 0:
                    self.print_results(results)

                if MPI: MPI_comm.Barrier()
                log.info("Total wall time for EmbCC: %s", get_time_string(timer()-t_start))

        log.info("All done.")

    # Alias for kernel
    run = kernel

    def collect_results(self, *attributes):
        """Use MPI to collect results from all clusters."""

        #log.debug("Collecting attributes %r from all clusters", (attributes,))
        clusters = self.clusters

        if MPI:
            def reduce_cluster(attr, op=MPI.SUM, root=0):
                res = MPI_comm.reduce(np.asarray([getattr(x, attr) for x in clusters]), op=op, root=root)
                return res
        else:
            def reduce_cluster(attr):
                res = np.asarray([getattr(x, attr) for x in clusters])
                return res

        results = {}
        for attr in attributes:
            results[attr] = reduce_cluster(attr)

        return results

    def show_cluster_sizes(self, results, show_largest=True):
        log.info("CLUSTER SIZES")
        log.info("*************")
        fmtstr = "  * %3d %-10s  :  active=%4d  frozen=%4d  ( %5.1f %%)"
        imax = [0]
        for i, x in enumerate(self.clusters):
            nactive = results["nactive"][i]
            nfrozen = results["nfrozen"][i]
            log.info(fmtstr, x.id, x.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.norb)
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
                log.info(fmtstr, x.id, x.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.norb)

    def print_results(self, results):
        self.show_cluster_sizes(results)

        log.info("CLUSTER ENERGIES")
        log.info("****************")
        log.info("CCSD / CCSD+dMP2 / CCSD+dMP2+(T)")
        fmtstr = "  * %3d %-10s  :  %16.8g  %16.8g  %16.8g"
        for i, x in enumerate(self.clusters):
            e_corr = results["e_corr"][i]
            e_pert_t = results["e_pert_t"][i]
            e_delta_mp2 = results["e_delta_mp2"][i]
            log.info(fmtstr, x.id, x.trimmed_name(10), e_corr, e_corr+e_delta_mp2, e_corr+e_delta_mp2+e_pert_t)

        log.info("  * %-14s  :  %16.8g  %16.8g  %16.8g", "total", self.e_corr, self.e_corr+self.e_delta_mp2, self.e_corr+self.e_delta_mp2+self.e_pert_t)
        #log.info("E(DMET) = %16.8g htr", self.e_dmet)

    def reset(self, mf=None, **kwargs):
        if mf:
            self.mf = mf
        for cluster in self.clusters:
            cluster.reset(**kwargs)

    def print_clusters(self):
        """Print fragments of calculations."""
        log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for cluster in self.clusters:
            log.info("%3d  %20s  %8s  %4d", cluster.id, cluster.name, cluster.solver, cluster.size)


    def print_clusters_orbitals(self, file=None, filemode="a"):
        """Print clusters orbitals to log or file.

        Parameters
        ----------
        file : str, optional
            If not None, write output to file.
        """
        # Format strings
        end = "\n" if file else ""
        if self.local_orbital_type == "AO":
            orbital_name = "atomic"
        if self.local_orbital_type == "LAO":
            orbital_name = "Lowdin orthogonalized atomic"
        if self.local_orbital_type == "IAO":
            orbital_name = "intrinsic atomic"

        headfmt = "Cluster %3d: %s with %3d {} orbitals:".format(orbital_name) + end

        linefmt = "%4d %5s %3s %10s" + end

        if self.local_orbital_type == "AO":
            labels = self.mol.ao_labels(None)
        elif self.local_orbital_type == "LAO":
            labels = self.lao_labels
        elif self.local_orbital_type == "IAO":
            labels = self.iao_labels

        if file is None:
            for cluster in self.clusters:
                log.info(headfmt, cluster.id, cluster.name, cluster.size)
                for idx in cluster.indices:
                    log.info(linefmt, *labels[idx])
        else:
            with open(file, filemode) as f:
                for cluster in self.clusters:
                    f.write(headfmt % (cluster.id, cluster.name, cluster.size))
                    for idx in cluster.indices:
                        f.write(linefmt % labels[idx])



