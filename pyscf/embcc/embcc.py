import logging
import os.path
import functools

import numpy as np
import scipy
import scipy.linalg
from mpi4py import MPI

import pyscf
import pyscf.lo

from .util import *
from .cluster import Cluster

__all__ = [
        "EmbCC",
        ]

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

class EmbCC:
    """What should this be named?"""

    default_options = [
            "solver",
            "bath_type",
            "bath_tol",
            "bath_size",
            "use_ref_orbitals_dmet",
            "use_ref_orbitals_bath",
            "mp2_correction",
            "maxiter",
            ]

    def __init__(self, mf,
            local_type="IAO",
            solver="CCSD",
            bath_type="mp2-natorb", bath_size=None, bath_tol=1e-3,
            minao="minao",
            use_ref_orbitals_dmet=True,
            #use_ref_orbitals_bath=True,
            use_ref_orbitals_bath=False,
            mp2_correction=False,
            maxiter=1,
            ):
        """
        Parameters
        ----------
        mf : pyscf.scf object
            Converged mean-field object.
        minao :
            Minimal basis for intrinsic atomic orbitals (IAO).
        """

        # Check input
        if not mf.converged:
            raise ValueError("Mean-field calculation not converged.")
        # Local orbital types:
        # AO : Atomic orbitals non-orthogonal wrt to non-fragment AOs
        # IAO : Intrinstric atom-orbitals
        # LAO : Lowdin orthogonalized AOs

        if local_type not in ("AO", "IAO", "LAO"):
            raise ValueError("Unknown local_type: %s" % local_type)
        if solver not in (None, "MP2", "CISD", "CCSD", "FCI"):
            raise ValueError("Unknown solver: %s" % solver)
        #if bath_type not in (None, "power", "matsubara", "uncontracted", "mp2-natorb"):
        if bath_type not in (None, "power", "matsubara", "uncontracted", "mp2-natorb", "full"):
            raise ValueError("Unknown bath type: %s" % bath_type)

        self.mf = mf
        self.local_orbital_type = local_type
        if self.local_orbital_type == "IAO":
            self.C_iao, self.C_env, self.iao_labels = self.make_iao(minao=minao)
            log.debug("IAO labels:")
            for ao in self.iao_labels:
                log.debug("%r", ao)
        elif self.local_orbital_type == "LAO":
            self.C_lao, self.lao_labels = self.make_lowdin_ao()
            #self.local_orbital_type = "IAO"

        self.maxiter = maxiter
        # Options
        self.solver = solver

        self.bath_type = bath_type
        self.bath_tol = bath_tol
        self.bath_size = bath_size
        self.use_ref_orbitals_dmet = use_ref_orbitals_dmet
        self.use_ref_orbitals_bath = use_ref_orbitals_bath
        self.mp2_correction = mp2_correction

        self.clusters = []

        # Correlation energy and MP2 correction
        self.e_corr = 0.0
        self.e_delta_mp2 = 0.0

        self.e_corr_full = 0.0

        # Global amplitudes
        self.T1 = None
        self.T2 = None
        # For tailored CC
        self.tccT1 = None
        self.tccT2 = None

    @property
    def mol(self):
        return self.mf.mol

    @property
    def nclusters(self):
        """Number of cluster."""
        return len(self.clusters)

    @property
    def e_tot(self):
        return self.mf.e_tot + self.e_corr

    def make_ao_projector(self, ao_indices):
        """Create projetor into AO subspace

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
        S1 = self.mf.get_ovlp()
        S2 = S1[np.ix_(ao_indices, ao_indices)]
        S21 = S1[ao_indices]
        P21 = scipy.linalg.solve(S2, S21, assume_a="pos")
        P = np.dot(S21.T, P21)
        assert np.allclose(P, P.T)
        return P

    def make_local_ao_orbitals(self, ao_indices):
        S = self.mf.get_ovlp()
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
        C_local = self.C_iao[:,iao_indices]
        #not_indices = np.asarray([i for i in np.arange(len(iao_indices)) if i not in iao_indices])
        not_indices = np.asarray([i for i in np.arange(self.C_iao.shape[-1]) if i not in iao_indices])
        C_env = np.hstack((self.C_iao[:,not_indices], self.C_env))

        return C_local, C_env

    def make_local_lao_orbitals(self, lao_indices):
        C_local = self.C_lao[:,lao_indices]
        not_indices = np.asarray([i for i in np.arange(self.C_lao.shape[-1]) if i not in lao_indices])
        C_env = self.C_lao[:,not_indices]

        return C_local, C_env

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
        # Get new ID
        cluster_id = len(self.clusters)
        # Check that ID and name are unique
        for cluster in self.clusters:
            if name == cluster.name:
                raise ValueError("Cluster with name %s already exists." % name)
            assert (cluster_id != cluster.id)
        # Pass options to cluster object via keyword arguments
        for opt in self.default_options:
            kwargs[opt] = kwargs.get(opt, getattr(self, opt))
        # Symmetry factor, if symmetry related clusters exist in molecule (e.g. hydrogen rings)
        kwargs["symmetry_factor"] = kwargs.get("symmetry_factor", 1.0)
        cluster = Cluster(self, cluster_id=cluster_id, name=name, C_local=C_local, C_env=C_env, **kwargs)
        self.clusters.append(cluster)
        return cluster

    def make_atom_cluster(self, atoms, name=None, **kwargs):
        """
        Parameters
        ---------
        atoms : list or str
            Atom labels of atoms in cluster.
        name : str
            Name of cluster.
        """
        # atoms may be a single atom label
        if isinstance(atoms, str):
            atoms = [atoms]
        # Check if atoms are valid labels of molecule
        atom_symbols = [self.mol.atom_symbol(atomid) for atomid in range(self.mol.natm)]
        for atom in atoms:
            if atom not in atom_symbols:
                raise ValueError("Atom %s not in molecule." % atom)
        if name is None:
            name = ",".join(atoms)

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
            indices = np.nonzero(np.isin(lao_atoms, atoms))[0]
            C_local, C_env = self.make_local_lao_orbitals(indices)

        # Orthogonal intrinsic AOs
        elif self.local_orbital_type == "IAO":
            # Base atom for each IAO
            iao_atoms = [iao[1] for iao in self.iao_labels]
            indices = np.nonzero(np.isin(iao_atoms, atoms))[0]
            C_local, C_env = self.make_local_iao_orbitals(indices)

        cluster = self.make_cluster(name, C_local, C_env, indices=indices, **kwargs)
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

    def get_orbitals(self):
        """Get orbitals of each cluster."""
        orbitals = {}
        for cluster in self.clusters:
            orbitals[cluster.name] = cluster.get_orbitals()
        return orbitals

    def set_reference_orbitals(self, ref_orbitals):
        return self.set_cluster_attributes("ref_orbitals", ref_orbitals)


    def get_refdata(self):
        refdata = {}
        for cluster in self.clusters:
            refdata[cluster.name] = cluster.get_refdata()
        return refdata

    def set_refdata(self, refdata):
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
        C_occ = self.mf.mo_coeff[:,self.mf.mo_occ>0]
        C_iao = pyscf.lo.iao.iao(self.mol, C_occ, minao=minao)
        niao = C_iao.shape[-1]
        log.debug("Total number of IAOs=%3d", niao)

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
        e, C = np.linalg.eigh(P_env)
        assert np.all(np.logical_or(abs(e) < 1e-10, abs(e)-1 < 1e-10))
        mask_env = (e > 1e-10)
        assert (np.sum(mask_env) + niao == norb)
        # Transform back to AO basis
        C_env = np.dot(self.mf.mo_coeff, C[:,mask_env])

        # Get base atoms of IAOs
        refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
        iao_labels = refmol.ao_labels(None)
        assert len(iao_labels) == C_iao.shape[-1]

        C = np.hstack((C_iao, C_env))
        assert np.allclose(C.T.dot(S).dot(C) - np.eye(norb), 0)

        return C_iao, C_env, iao_labels

    def make_lowdin_ao(self):
        """We can simply (ab)use the IAO code, since the nomenclature here."""
        S = self.mf.get_ovlp()
        C_lao = pyscf.lo.vec_lowdin(np.eye(S.shape[-1]), S)
        lao_labels = self.mol.ao_labels(None)
        return C_lao, lao_labels


    def run(self, **kwargs):
        if not self.clusters:
            raise ValueError("No clusters defined for EmbCC calculation.")

        MPI_comm.Barrier()
        t_start = MPI.Wtime()

        for idx, cluster in enumerate(self.clusters):
            if MPI_rank != (idx % MPI_size):
                continue

            log.debug("Running cluster %s on MPI process=%d...", cluster.name, MPI_rank)
            cluster.run(**kwargs)
            log.debug("Cluster %s on MPI process=%d is done.", cluster.name, MPI_rank)

        #results = self.collect_results("converged", "e_corr", "e_delta_mp2", "e_corr_v", "e_corr_d")
        results = self.collect_results("converged", "e_corr", "e_delta_mp2", "e_corr_full", "e_corr_v", "e_corr_d")
        if MPI_rank == 0 and not np.all(results["converged"]):
            log.debug("converged = %s", results["converged"])
            raise RuntimeError("Not all cluster converged")

        # TEST energy
        #import pyscf
        #import pyscf.cc
        #cc = pyscf.cc.CCSD(self.mf)
        #eris = cc.ao2mo()
        #energy = cc.energy(t1=self.T1, t2=self.T2, eris=eris)
        #log.info("EmbCC energy=%.8g", energy)

        self.e_corr = sum(results["e_corr"])
        self.e_delta_mp2 = sum(results["e_delta_mp2"])

        self.e_corr_full = sum(results["e_corr_full"])

        self.e_corr_v = sum(results["e_corr_v"])
        self.e_corr_d = sum(results["e_corr_d"])

        if MPI_rank == 0:
            self.print_results(results)

        MPI_comm.Barrier()
        log.info("Total wall time for EmbCC: %s", get_time_string(MPI.Wtime()-t_start))

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

                MPI_comm.Barrier()
                log.info("Total wall time for EmbCC: %s", get_time_string(MPI.Wtime()-t_start))

    def collect_results(self, *attributes):
        log.debug("Collecting attributes %r from all clusters", (attributes,))
        clusters = self.clusters

        def mpi_reduce(attr, op=MPI.SUM, root=0):
            res = MPI_comm.reduce(np.asarray([getattr(c, attr) for c in clusters]), op=op, root=root)
            return res

        results = {}
        for attr in attributes:
            results[attr] = mpi_reduce(attr)

        return results

    def print_results(self, results):
        log.info("Energy contributions per cluster")
        log.info("--------------------------------")
        # Name solver nactive (local, dmet bath, add bath) nfrozen E_corr_full E_corr
        #linefmt = "%10s  %6s  %3d (%3d,%3d,%3d)  %3d: Full=%16.8g Eh Local=%16.8g Eh"
        #totalfmt = "Total=%16.8g Eh"
        #for c in self.clusters:
        #    log.info(linefmt, c.name, c.solver, len(c)+c.nbath, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_corr_full, c.e_corr)
        linefmt  = "%3d %20s [%8s] = %16.8g htr + %16.8g htr = %16.8g"
        totalfmt = "Total = %16.8g htr , %16.8g htr"
        for i, cluster in enumerate(self.clusters):
            e_corr = results["e_corr"][i]
            e_delta_mp2 = results["e_delta_mp2"][i]
            log.info(linefmt, cluster.id, cluster.name, cluster.solver, e_corr, e_delta_mp2, e_corr+e_delta_mp2)
        log.info(totalfmt, self.e_corr, self.e_corr + self.e_delta_mp2)

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
