import logging
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.gto
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
        if hasattr(mf, 'kpts') and mf.kpts is not None:
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


    # --- Properties
    # ==============

    @property
    def mol(self):
        """Mole or Cell object."""
        return self.mf.mol

    @property
    def has_lattice_vectors(self):
        """Flag if self.mol has lattice vectors defined."""
        return (hasattr(self.mol, 'a') and self.mol.a is not None)

    @property
    def boundary_cond(self):
        """Type of boundary condition."""
        if not self.has_lattice_vectors:
            return 'open'
        if self.mol.dimension == 1:
            return 'periodic-1D'
        if self.mol.dimension == 2:
            return 'periodic-2D'
        return 'periodic'

    @property
    def kpts(self):
        """k-points for periodic calculations.

        TODO: for *not unfolded* Gamma-point calculations, does this return None?
        Should it return [(0,0,0)]?
        """
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

    @property
    def e_mf(self):
        """Total mean-field energy."""
        return self.mf.e_tot/self.ncells


    # --- Integral methods
    # ====================

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
            log.debugv("ao2mo method: %r", cm.ao2mo)
            if isinstance(cm, pyscf.mp.dfmp2.DFMP2):
                # TODO: This is a hack, requiring modified PySCF - normal DFMP2 does not store 4c (ov|ov) integrals
                return cm.ao2mo(store_eris=True)
            else:
                return cm.ao2mo()
        # k-point sampled primtive cell:
        eris = gdf_to_pyscf_eris(self.mf, self.kdf, cm, fock=self.get_fock())
        return eris


    # --- Methods for fragmentation
    # =============================
    # 1)
    # 2) IAO specific
    # 3) AO specific

    # 1)





    # 2) IAO specific
    # ---------------

    def make_iao_coeffs(self, minao='minao', return_rest=True):
        """Make intrinsic atomic orbitals (IAOs) and remaining virtual orbitals via projection.

        Parameters
        ----------
        minao : str, optional
            Minimal basis set for IAOs. Default: 'minao'.
        return_rest : bool, optional
            Return coefficients of remaining virtual orbitals. Default: `True`.

        Returns
        -------
        c_iao : (nAO, nIAO) array
            IAO coefficients.
        c_rest : (nAO, nRest) array
            Remaining virtual orbital coefficients. `None`, if `make_rest == False`.
        """
        mo_coeff = self.mo_coeff
        ovlp = self.get_ovlp()

        c_occ = self.mo_coeff[:,self.mo_occ>0]
        c_iao = pyscf.lo.iao.iao(self.mol, c_occ, minao=minao)
        niao = c_iao.shape[-1]
        log.info("Total number of IAOs= %4d", niao)

        # Orthogonalize IAO using symmetric orthogonalization
        c_iao = pyscf.lo.vec_lowdin(c_iao, ovlp)

        # Check that all electrons are in IAO space
        sc = np.dot(ovlp, c_iao)
        dm_iao = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc))
        nelec_iao = np.trace(dm_iao)
        log.debugv('nelec_iao= %.8f', nelec_iao)
        if abs(nelec_iao - self.mol.nelectron) > 1e-5:
            log.error("IAOs do not contain the correct number of electrons: %.8f", nelec_iao)

        # Test orthogonality of IAO
        idterr = c_iao.T.dot(ovlp).dot(c_iao) - np.eye(niao)
        log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of IAO: L2= %.2e  Linf= %.2e", np.linalg.norm(idterr), abs(idterr).max())

        if not return_rest:
            return c_iao, None

        # Add remaining virtual space, work in MO space, so that we automatically get the
        # correct linear dependency treatment, if nMO < nAO
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

        # --- Some checks below:

        # Test orthogonality of IAO + rest
        c_all = np.hstack((c_iao, c_rest))
        idterr = c_all.T.dot(ovlp).dot(c_all) - np.eye(self.nmo)
        log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of IAO+vir. orbitals: L2= %.2e  Linf= %.2e", np.linalg.norm(idterr), abs(idterr).max())

        return c_iao, c_rest


    def get_iao_occupancy(self, c_iao, minao='minao', verbose=True):
        """Get electron occupancy of IAOs.

        Parameters
        ----------
        c_iao : (nAO, nIAO) array
            IAO coefficients.
        minao : str, optional
            Minimal basis set, which was used to construct IAOs. Default: 'minao'.
        verbose : bool, optional
            Check lattice symmetry of IAOs and print occupations per atom.
            Default: True.

        Returns
        -------
        occ_iao : (nIAO,) array
            Occupation of IAOs.
        """
        sc = np.dot(self.get_ovlp(), c_iao)
        occ_iao = einsum('ai,ab,bi->i', sc, self.mf.make_rdm1(), sc)
        if not verbose:
            return occ_iao

        iao_labels = self.get_iao_labels(minao)
        if len(iao_labels) != c_iao.shape[-1]:
            raise RuntimeError("Inconsistent number of IAOs. Were different minimal basis sets used?")
        # Occupancy per atom
        occ_atom = []
        iao_atoms = np.asarray([i[0] for i in iao_labels])
        log.debugv('iao_atoms= %r', iao_atoms)
        for a in range(self.mol.natm):
            mask = np.where(iao_atoms == a)[0]
            occ_atom.append(occ_iao[mask])
        log.debugv("occ_atom: %r", occ_atom)

        # Check lattice symmetry if k-point mf object was used
        tsym = False
        if self.ncells > 1:
            # IAO occupations per cell
            occ_cell = np.split(np.hstack(occ_atom), self.ncells)
            log.debugv("occ_cell: %r", occ_cell)
            # Compare all cells to the primitive cell
            log.debugv("list: %r", [np.allclose(occ_cell[i], occ_cell[0]) for i in range(self.ncells)])
            tsym = np.all([np.allclose(occ_cell[i], occ_cell[0]) for i in range(self.ncells)])
        log.debugv("tsym: %r", tsym)

        # Print occupations of IAOs
        log.info("IAO MEAN-FIELD OCCUPANCY PER ATOM")
        log.info("*********************************")
        for a in range(self.mol.natm if not tsym else self.kcell.natm):
            mask = np.where(iao_atoms == a)[0]
            log.debugv('mask= %r', mask)
            fmt = "  * %3d: %-8s total= %12.8f" + len(occ_atom[a])*"  %s= %10.8f"
            labels = [("_".join((x[2], x[3])) if x[3] else x[2]) for x in np.asarray(iao_labels)[mask]]
            vals = [val for pair in zip(labels, occ_atom[a]) for val in pair]
            log.info(fmt, a, self.mol.atom_symbol(a), np.sum(occ_atom[a]), *vals)

        return occ_iao


    def get_iao_labels(self, minao):
        """Get labels of IAOs

        Parameters
        ----------
        minao : str, optional
            Minimal basis set for IAOs. Default: 'minao'.

        Returns
        -------
        iao_labels : list of length nIAO
            Orbital label (atom-id, atom symbol, nl string, m string) for each IAO.
        """
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


    # 3) AO specific
    # --------------

    def get_subset_ao_projector(self, aos):
        """Get projector onto AO subspace in the non-orthogonal AO basis.

        Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b
        This is a special case of the more general `get_ao_projector`, which can also
        handle a different AO basis set.

        Parameters
        ----------
        aos : list of AO indices or AO labels or mask
            List of indices/labels or mask of subspace AOs. If a list of labels is given,
            it is converted to AO indices using the PySCF `search_ao_label` function.

        Returns
        -------
        p : (nAO, nAO) array
            Projector onto AO subspace.
        """
        s1 = self.get_ovlp()
        if aos is None:
            aos = np.s_[:]

        if isinstance(aos, slice):
            s2 = s1[aos,aos]
        elif isinstance(aos[0], str):
            log.debugv("Searching for AO indices of AOs %r", aos)
            aos_idx = self.mol.search_ao_label(aos)
            log.debugv("Found AO indices: %r", aos_idx)
            log.debugv("Corresponding to AO labels: %r", np.asarray(self.mol.ao_labels())[aos_idx])
            if len(aos_idx) == 0:
                raise RuntimeError("No AOs with labels %r found" % aos)
            aos = aos_idx
            s2 = s1[np.ix_(aos, aos)]
        else:
            s2 = s1[np.ix_(aos, aos)]
        s21 = s1[aos]
        p21 = scipy.linalg.solve(s2, s21, assume_a="pos")
        p = np.dot(s21.T, p21)
        assert np.allclose(p, p.T)
        return p


    def get_ao_projector(self, aos, basis=None):
        """Get projector onto AO subspace in the non-orthogonal AO basis.

        Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b

        TODO: This is probably not correct (check Ref. above) if basis is not fully contained
        in the span of self.mol.basis. In this case a <1|2> is missing.

        Parameters
        ----------
        aos : list of AO indices or AO labels or mask
            List of indices/labels or mask of subspace AOs. If a list of labels is given,
            it is converted to AO indices using the PySCF `search_ao_label` function.
        basis : str, optional
            Basis set for AO subspace. If `None`, the same basis set as that of `self.mol`
            is used. Default: `None`.

        Returns
        -------
        p : (nAO, nAO) array
            Projector onto AO subspace.
        """

        mol1 = self.mol
        s1 = self.get_ovlp()

        # AOs are given in same basis as mol1
        if basis is None:
            mol2 = mol1
            s2 = s21 = s1
        # AOs of a different basis
        else:
            mol2 = mol1.copy()
            # What was this for? - commented for now
            #if getattr(mol2, 'rcut', None) is not None:
            #    mol2.rcut = None
            mol2.build(False, False, basis=basis2)

            if self.boundary_cond == 'open':
                s2 = mol2.intor_symmetric('int1e_ovlp')
                s12 = pyscf.gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
            else:
                s2 = np.asarray(mol2.pbc_intor('int1e_ovlp', hermi=1, kpts=None))
                s21 = np.asarray(pyscf.pbc.gto.cell.intor_cross('int1e_ovlp', mol1, mol2, kpts=None))
        assert s1.ndim == 2
        assert s2.ndim == 2
        assert s21.ndim == 2

        # All AOs
        if aos is None:
            aos = np.s_[:]

        if isinstance(aos, slice):
            s2 = s1[aos,aos]
        elif isinstance(aos[0], str):
            log.debugv("Searching for AO indices of AOs %r", aos)
            aos_idx = mol2.search_ao_label(aos)
            log.debugv("Found AO indices: %r", aos_idx)
            log.debugv("Corresponding to AO labels: %r", np.asarray(mol2.ao_labels())[aos_idx])
            if len(aos_idx) == 0:
                raise RuntimeError("No AOs with labels %r found" % aos)
            aos = aos_idx
            s2 = s1[np.ix_(aos, aos)]
        else:
            s2 = s1[np.ix_(aos, aos)]

        s21 = s21[aos]

        p21 = scipy.linalg.solve(s2, s21, assume_a="pos")
        p = np.dot(s21.T, p21)
        assert np.allclose(p, p.T)
        return p






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

    def canonicalize_mo(self, *mo_coeff, eigvals=False):
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
        if eigvals:
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


