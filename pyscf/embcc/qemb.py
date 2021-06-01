import logging
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.mp
import pyscf.lo
import pyscf.pbc
import pyscf.pbc.gto

from .util import *
from .kao2gmo import gdf_to_pyscf_eris

log = logging.getLogger(__name__)

class QEmbeddingMethod:

    def __init__(self, mf):

        # k-space unfolding
        if hasattr(mf, "kpts") and mf.kpts is not None:
            t0 = timer()
            self.kdf = mf.with_df
            log.info("Mean-field calculations has %d k-points; unfolding to supercell.", self.ncells)
            log.debug("type(df._cderi)= %r", type(self.kdf._cderi))
            mf = pyscf.pbc.tools.k2gamma.k2gamma(mf)
            log.timing("Time for k->Gamma unfolding of mean-field calculation:  %s", time_string(timer()-t0))
        else:
            self.kdf = None
        self.mf = mf

        # Copy MO attributes, so they can be modified later with no side-effects (updating the mean-field)
        self.mo_energy = self.mf.mo_energy.copy()
        self.mo_occ = self.mf.mo_occ.copy()
        self.mo_coeff = self.mf.mo_coeff.copy()
        self._ovlp = self.mf.get_ovlp()
        # Recalcution of Fock matrix expensive for PBC!
        # => avoid self._fock = self.mf.get_fock()
        # (however, loss of accuracy for large values for cell.precision!)
        cs = np.dot(self.mo_coeff.T, self._ovlp)
        self._fock = np.dot(cs.T*self.mo_energy, cs)

        # Fragments
        self.fragments = []

        # Some output
        if self.mf.converged:
            log.info("E(MF)= %+16.8f Ha", self.e_mf)
        else:
            log.warning("E(MF)= %+16.8f Ha (not converged!)", self.e_mf)
        log.info("n(AO)= %4d  n(MO)= %4d  n(linear dep.)= %4d", self.nao, self.nmo, self.nao-self.nmo)
        idterr = self.mo_coeff.T.dot(self._ovlp).dot(self.mo_coeff) - np.eye(self.nmo)
        log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of MF orbitals: L2= %.2e  Linf= %.2e", np.linalg.norm(idterr), abs(idterr).max())


    @property
    def mol(self):
        """Mole or Cell object."""
        return self.mf.mol

    @property
    def has_pbc(self):
        """If the system has periodic boundary conditions."""
        #return isinstance(self.mol, pyscf.pbc.gto.Cell)
        return hasattr(self.mol, 'a') and self.mol.a is not None

    @property
    def kpts(self):
        """k-points for periodic calculations."""
        if self.kdf is None:
            return None
        return self.kdf.kpts

    @property
    def kcell(self):
        """Unit cell for periodic calculations."""
        if self.kdf is None:
            return None
        return self.kdf.cell

    @property
    def nao(self):
        """Number of atomic orbitals."""
        return self.mol.nao_nr()

    @property
    def nmo(self):
        """Number of molecular orbitals."""
        return len(self.mo_energy)

    @property
    def ncells(self):
        """Number of primitive cells within supercell."""
        if self.kpts is None:
            return 1
        return len(self.kpts)

    @property
    def nfrag(self):
        """Number of fragments."""
        return len(self.fragments)

    def get_ovlp(self):
        """AO overlap matrix."""
        return self._ovlp

    def get_fock(self):
        """Fock matrix in AO basis."""
        return self._fock

    def get_eris(self, cm):
        """Get ERIS for post-HF methods.

        For unfolded PBC calculations, this folds the MO back into k-space
        and contracts with the k-space three-center integrals..

        Arguments
        ---------
        cm: pyscf.mp.mp2.MP2, pyscf.cc.ccsd.CCSD, or pyscf.cc.rccsd.RCCSD
            Correlated method, must have mo_coeff set.

        Returns
        -------
        eris: pyscf.mp.mp2._ChemistsERIs, pyscf.cc.ccsd._ChemistsERIs, or pyscf.cc.rccsd._ChemistsERIs
            ERIs which can be used for the respective correlated method.
        """
        # Molecules or supercell:
        if self.kdf is None:
            log.debugv('ao2mo method: %r', cm.ao2mo)
            if isinstance(cm, pyscf.mp.dfmp2.DFMP2):
                return cm.ao2mo(store_eris=True)
            else:
                return cm.ao2mo()
        # k-point sampled primtive cell:
        eris = gdf_to_pyscf_eris(self.mf, self.kdf, cm, fock=self.get_fock())
        return eris

    @property
    def e_mf(self):
        """Total mean-field energy."""
        return self.mf.e_tot/self.ncells


    # --- Fragments ---
    # =================

    def make_iao_coeffs(self, minao='minao'):
        """Make intrinsic atomic orbitals.

        Parameters
        ----------
        minao : str, optional
            Minimal basis set for IAOs. Default: 'minao'.

        Returns
        -------
        c_iao : (N, M) array
            IAO coefficients.
        c_rest : (N, L) array
            Remaining virtual orbital coefficients.
        iao_labels : list
            Orbital label (atom-id, atom symbol, nl string, m string) for each IAO.
        """
        mo_coeff = self.mo_coeff
        ovlp = self.get_ovlp()

        c_occ = self.mo_coeff[:,self.mo_occ>0]
        c_iao = pyscf.lo.iao.iao(self.mol, c_occ, minao=minao)
        niao = c_iao.shape[-1]
        log.info("Total number of IAOs= %4d", niao)

        # Orthogonalize IAO using symmetric orthogonalization
        c_iao = pyscf.lo.vec_lowdin(c_iao, ovlp)

        # Add remaining virtual space, work in MO space, so that we automatically get the
        # correct linear dependency treatment, if n(MO) != n(AO)
        c_iao_mo = np.linalg.multi_dot((self.mo_coeff.T, ovlp, c_iao))
        # Get eigenvectors of projector into complement
        p_iao = np.dot(c_iao_mo, c_iao_mo.T)
        p_rest = np.eye(self.nmo) - p_iao
        e, c = np.linalg.eigh(p_rest)

        # Corresponding expression in AO basis (but no linear-dependency treatment):
        # p_rest = ovlp - ovlp.dot(c_iao).dot(c_iao.T).dot(ovlp)
        # e, c = scipy.linalg.eigh(p_rest, ovlp)
        # c_rest = c[:,e>0.5]

        # Ideally, all eigenvalues of P_env should be 0 (IAOs) or 1 (non-IAO)
        # Error if > 1e-3
        mask_iao, mask_rest = (e <= 0.5), (e > 0.5)
        e_iao, e_rest = e[mask_iao], e[mask_rest]
        if np.any(abs(e_iao) > 1e-3):
            log.error("CRITICAL: Some IAO eigenvalues of 1-P_IAO are not close to 0:\n%r", e_iao)
        elif np.any(abs(e_iao) > 1e-6):
            log.warning("Some IAO eigenvalues e of 1-P_IAO are not close to 0: n= %d max|e|= %.2e",
                    np.count_nonzero(abs(e_iao) > 1e-6), abs(e_iao).max())
        if np.any(abs(1-e_rest) > 1e-3):
            log.error("CRITICAL: Some non-IAO eigenvalues of 1-P_IAO are not close to 1:\n%r", e_rest)
        elif np.any(abs(1-e_rest) > 1e-6):
            log.warning("Some non-IAO eigenvalues e of 1-P_IAO are not close to 1: n= %d max|1-e|= %.2e",
                    np.count_nonzero(abs(1-e_rest) > 1e-6), abs(1-e_rest).max())

        if not (np.sum(mask_rest) + niao == self.nmo):
            log.critical("Error in construction of remaining virtual orbitals! Eigenvalues of projector 1-P_IAO:\n%r", e)
            log.critical("Number of eigenvalues above 0.5 = %d", np.sum(mask_rest))
            log.critical("Total number of orbitals = %d", self.nmo)
            raise RuntimeError("Incorrect number of remaining virtual orbitals")
        c_rest = np.dot(self.mo_coeff, c[:,mask_rest])        # Transform back to AO basis

        iao_labels = self.get_iao_labels(minao=minao)
        assert (len(iao_labels) == niao)

        # --- Some checks below:

        # Test orthogonality of IAO + rest
        c_all = np.hstack((c_iao, c_rest))
        idterr = c_all.T.dot(ovlp).dot(c_all) - np.eye(self.nmo)
        log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of IAO+vir. orbitals: L2= %.2e  Linf= %.2e", np.linalg.norm(idterr), abs(idterr).max())

        # Check that all electrons are in IAO space
        sc = np.dot(ovlp, c_iao)
        dm_iao = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc))
        nelec_iao = np.trace(dm_iao)
        log.debugv('nelec_iao= %.8f', nelec_iao)
        if abs(nelec_iao - self.mol.nelectron) > 1e-5:
            log.error("IAOs do not contain the correct number of electrons: %.8f", nelec_iao)

        # Occupancy per atom
        n_occ = []
        iao_atoms = np.asarray([i[0] for i in iao_labels])
        log.debugv('iao_atoms= %r', iao_atoms)
        for a in range(self.mol.natm):
            mask = np.where(iao_atoms == a)[0]
            n_occ.append(np.diag(dm_iao[mask][:,mask]))
        log.debugv("n_occ: %r", n_occ)

        # Check lattice symmetry if k-point mf object was used
        tsym = False
        if self.ncells > 1:
            # IAO occupations per cell
            n_occ_cell = np.split(np.hstack(n_occ), self.ncells)
            log.debugv("n_occ_cell: %r", n_occ_cell)
            # Compare all cells to the primitive cell
            log.debugv("list: %r", [np.allclose(n_occ_cell[i], n_occ_cell[0]) for i in range(self.ncells)])
            tsym = np.all([np.allclose(n_occ_cell[i], n_occ_cell[0]) for i in range(self.ncells)])
        log.debugv("tsym: %r", tsym)

        # Print occupations of IAOs
        log.info("IAO MEAN-FIELD OCCUPANCY PER ATOM")
        log.info("*********************************")
        for a in range(self.mol.natm if not tsym else self.kcell.natm):
            mask = np.where(iao_atoms == a)[0]
            log.debugv('mask= %r', mask)
            fmt = "  * %3d: %-8s total= %12.8f" + len(n_occ[a])*"  %s= %10.8f"
            labels = [("_".join((x[2], x[3])) if x[3] else x[2]) for x in np.asarray(iao_labels)[mask]]
            vals = [val for pair in zip(labels, n_occ[a]) for val in pair]
            log.info(fmt, a, self.mol.atom_symbol(a), np.sum(n_occ[a]), *vals)

        return c_iao, c_rest, iao_labels


    def get_iao_labels(self, minao):
        """Get AO labels of IAOs"""
        refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
        iao_labels_refmol = refmol.ao_labels(None)
        log.debugv('iao_labels_refmol: %r', iao_labels_refmol)
        if refmol.natm == self.mol.natm:
            iao_labels = iao_labels_refmol
        # If there are ghost atoms in the system, they will be removed in refmol.
        # For this reason, the atom IDs of mol and refmol will not agree anymore.
        # Here we will correct the atom IDs of refmol to agree with mol
        # (they will no longer be contiguous integers).
        else:
            ref2mol = []
            for refatm in range(refmol.natm):
                ref_coords = refmol.atom_coord(refatm)
                for atm in range(self.mol.natm):
                    coords = self.mol.atom_coord(atm)
                    if np.allclose(coords, ref_coords):
                        log.debugv('reference cell atom %r maps to atom %r', refatm, atm)
                        ref2mol.append(atm)
                        break
                else:
                    raise RuntimeError("No atom found with coordinates %r" % ref_coords)
            iao_labels = []
            for iao in iao_labels_refmol:
                iao_labels.append((ref2mol[iao[0]], iao[1], iao[2], iao[3]))
        log.debugv('iao_labels: %r', iao_labels)
        assert (len(iao_labels_refmol) == len(iao_labels))
        return iao_labels


class QEmbeddingFragment:


    def __init__(self, base, fid, name, c_frag, c_env, sym_factor=1.0,
            # Metadata
            aos=None, atoms=None):
        """
        Parameters
        ----------
        base : QEmbeddingMethod
            Quantum embedding method the fragment is part of.
        fid : int
            Fragment ID.
        name : str
            Name of fragment.
        c_frag : (nAO, nFrag) array
            Fragment orbitals.
        c_env : (nAO, nEnv) array
            Environment (non-fragment) orbitals.
        sym_factor : float, optional
            Symmetry factor (number of symmetry equivalent fragments).
        aos : list or int, optional
        atoms : list or int, optional
        """
        msg = "CREATING FRAGMENT %d: %s" % (fid, name)
        log.info(msg)
        log.info(len(msg)*"*")

        self.base = base
        self.id = fid
        self.name = name
        self.c_frag = c_frag
        self.c_env = c_env
        self.sym_factor = sym_factor
        self.aos = aos
        self.atoms = atoms

        # Some output
        log.info("  * Number of fragment orbitals= %d", self.size)
        log.info("  * Symmetry factor= %f", self.sym_factor)
        log.info("  * Number of mean-field electrons= %.6f", self.nelectron)
        if self.aos:
            log.info("  * Associated AOs= %r", self.aos)
        if self.atoms:
            log.info("  * Associated atoms= %r", self.atoms)

    @property
    def mol(self):
        return self.base.mol

    @property
    def mf(self):
        return self.base.mf

    @property
    def size(self):
        """Number of fragment orbitals."""
        return self.c_frag.shape[-1]

    @property
    def nelectron(self):
        """Number of mean-field electrons."""
        sc = np.dot(self.base.get_ovlp(), self.c_frag)
        ne = np.einsum("ai,ab,bi->", sc, self.mf.make_rdm1(), sc)
        return ne

    def trimmed_name(self, length=10):
        """Fragment name trimmed to a given maximum length."""
        if len(self.name) <= length:
            return self.name
        return self.name[:(length-3)] + "..."

    @property
    def id_name(self):
        """Use this whenever a unique name is needed (for example to open a seperate file for each fragment)."""
        return "%s-%s" % (self.id, self.trimmed_name())

    def get_mo_occupation(self, *mo_coeff):
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
        mo_coeff = np.hstack(mo_coeff)
        sc = np.dot(self.base.get_ovlp(), mo_coeff)
        occ = einsum('ai,ab,bi->i', sc, self.mf.make_rdm1(), sc)
        return occ

    def canonicalize_mo(self, *mo_coeff, eigenvalues=False):
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
        mo_can = np.dot(mo_coeff, rot)
        if eigenvalues:
            return mo_can, rot, mo_energy
        return mo_can, rot


    def diagonalize_cluster_dm(self, c_bath, tol=1e-4):
        """Diagonalize cluster (fragment+bath) DM to get fully occupied and virtual orbitals.

        Parameters
        ----------
        c_bath : ndarray
            Bath orbitals.
        tol : float, optional
            If set, check that all eigenvalues of the cluster DM are close
            to 0 or 1, with the tolerance given by tol. Default= 1e-4.

        Returns
        -------
        c_occclt : ndarray
            Occupied cluster orbitals.
        c_virclt : ndarray
            Virtual cluster orbitals.
        """
        c_clt = np.hstack((self.c_frag, c_bath))
        sc = np.dot(self.base.get_ovlp(), c_clt)
        dm = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc)) / 2
        e, v = np.linalg.eigh(dm)
        if tol and not np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=tol, rtol=0):
            raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not all close to 0 or 1:\n%s", e)
        e, v = e[::-1], v[:,::-1]
        c_clt = np.dot(c_clt, v)
        nocc = sum(e >= 0.5)
        c_occclt, c_virclt = np.hsplit(c_clt, [nocc])
        return c_occclt, c_virclt


