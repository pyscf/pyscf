"""These functions take a cluster instance as first argument ("self")."""

import numpy as np

import pyscf
import pyscf.lo

from .util import *

def project_ref_orbitals(self, C_ref, C):
    """Project reference orbitals into available space in new geometry.

    The projected orbitals will be ordered according to their eigenvalues within the space.

    Parameters
    ----------
    C : ndarray
        Orbital coefficients.
    C_ref : ndarray
        Orbital coefficients of reference orbitals.
    """
    nref = C_ref.shape[-1]
    assert (nref > 0)
    assert (C.shape[-1] > 0)
    self.log.debug("Projecting %d reference orbitals into space of %d orbitals", nref, C.shape[-1])
    S = self.base.get_ovlp()
    # Diagonalize reference orbitals among themselves (due to change in overlap matrix)
    C_ref_orth = pyscf.lo.vec_lowdin(C_ref, S)
    assert (C_ref_orth.shape == C_ref.shape)
    # Diagonalize projector in space
    CSC = np.linalg.multi_dot((C_ref_orth.T, S, C))
    P = np.dot(CSC.T, CSC)
    e, R = np.linalg.eigh(P)
    e, R = e[::-1], R[:,::-1]
    C = np.dot(C, R)
    #self.log.debug("All eigenvalues:\n%r", e)

    return C, e

def make_dmet_bath(self, C_ref=None, nbath=None, tol=1e-4, reftol=0.8):
    """Calculate DMET bath, occupied environment and virtual environment orbitals.

    If C_ref is not None, complete DMET orbital space using active transformation of reference orbitals.

    TODO: reftol should not be necessary - just determine how many DMET bath orbital N are missing
    from C_ref and take the N largest eigenvalues over the combined occupied and virtual
    eigenvalues.

    Parameters
    ----------
    C_ref : ndarray, optional
        Reference DMET bath orbitals from previous calculation.
    tol : float, optional
        Tolerance for DMET orbitals in eigendecomposition of density-matrix.
    reftol : float, optional
        Tolerance for DMET orbitals in projection of reference orbitals.

    Returns
    -------
    C_bath : ndarray
        DMET bath orbitals.
    C_occenv : ndarray
        Occupied environment orbitals.
    C_virenv : ndarray
        Virtual environment orbitals.
    """
    C_env = self.c_env
    # Divide by 2 to get eigenvalues in [0,1]
    sc = np.dot(self.base.get_ovlp(), C_env)
    D_env = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc)) / 2
    eig, R = np.linalg.eigh(D_env)
    eig, R = eig[::-1], R[:,::-1]

    if (eig.min() < -1e-9):
        self.log.warning("Min eigenvalue of env. DM = %.12e", eig.min())
    if ((eig.max()-1) > 1e-9):
        self.log.warning("Max eigenvalue of env. DM = %.12e", eig.max())

    C_env = np.dot(C_env, R)

    if nbath is not None:
        #mask_bath = np.argsort(abs(e-0.5))[:nbath]
        #b0 = min
        # We can assume e is sorted:
        #mask_occenv = np.asarray(
        raise NotImplementedError()
    else:
        mask_bath = np.logical_and(eig >= tol, eig <= 1-tol)
        mask_occenv = eig > 1-tol
        mask_virenv = eig < tol
        nbath = sum(mask_bath)

    noccenv = sum(mask_occenv)
    nvirenv = sum(mask_virenv)
    self.log.info("DMET bath:  n(Bath)= %4d  n(occ-Env)= %4d  n(vir-Env)= %4d", nbath, noccenv, nvirenv)
    assert (nbath + noccenv + nvirenv == C_env.shape[-1])
    C_bath = C_env[:,mask_bath].copy()
    C_occenv = C_env[:,mask_occenv].copy()
    C_virenv = C_env[:,mask_virenv].copy()

    # Orbitals in [print_tol, 1-print_tol] will be printed (even if they don't fall in the DMET tol range)
    print_tol = 1e-10
    # DMET bath orbitals with eigenvalue in [strong_tol, 1-strong_tol] are printed as strongly entangled
    strong_tol = 0.1
    limits = [print_tol, tol, strong_tol, 1-strong_tol, 1-tol, 1-print_tol]
    if np.any(np.logical_and(eig > limits[0], eig <= limits[-1])):
        names = [
                "Unentangled vir. env. orbital",
                "Weakly-entangled vir. bath orbital",
                "Strongly-entangled bath orbital",
                "Weakly-entangled occ. bath orbital",
                "Unentangled occ. env. orbital",
                ]
        self.log.info("Non-(0 or 1) eigenvalues (n) of environment DM:")
        for i, e in enumerate(eig):
            name = None
            for j, llim in enumerate(limits[:-1]):
                ulim = limits[j+1]
                if (llim < e and e <= ulim):
                    name = names[j]
                    break
            if name:
                self.log.info("  * %-34s  n= %12.6g  1-n= %12.6g", name, e, 1-e)

    # Calculate entanglement entropy
    entropy = np.sum(eig * (1-eig))
    entropy_bath = np.sum(eig[mask_bath] * (1-eig[mask_bath]))
    self.log.info("Entanglement entropy: total= %.6e  bath= %.6e  captured=  %.2f %%",
            entropy, entropy_bath, 100.0*entropy_bath/entropy)

    # Complete DMET orbital space using reference orbitals
    if C_ref is not None:
        nref = C_ref.shape[-1]
        self.log.debug("%d reference DMET orbitals given.", nref)
        nmissing = nref - nbath

        # DEBUG
        _, eig = self.project_ref_orbitals(C_ref, C_bath)
        self.log.debug("Eigenvalues of reference orbitals projected into DMET bath:\n%r", eig)

        if nmissing == 0:
            self.log.debug("Number of DMET orbitals equal to reference.")
        elif nmissing > 0:
            # Perform the projection separately for occupied and virtual environment space
            # Otherwise, it is not guaranteed that the additional bath orbitals are
            # fully (or very close to fully) occupied or virtual.
            # --- Occupied
            C_occenv, eig = self.project_ref_orbitals(C_ref, C_occenv)
            mask_occref = eig >= reftol
            mask_occenv = eig < reftol
            self.log.debug("Eigenvalues of projected occupied reference: %s", eig[mask_occref])
            if np.any(mask_occenv):
                self.log.debug("Largest remaining: %s", max(eig[mask_occenv]))
            # --- Virtual
            C_virenv, eig = self.project_ref_orbitals(C_ref, C_virenv)
            mask_virref = eig >= reftol
            mask_virenv = eig < reftol
            self.log.debug("Eigenvalues of projected virtual reference: %s", eig[mask_virref])
            if np.any(mask_virenv):
                self.log.debug("Largest remaining: %s", max(eig[mask_virenv]))
            # -- Update coefficient matrices
            C_bath = np.hstack((C_bath, C_occenv[:,mask_occref], C_virenv[:,mask_virref]))
            C_occenv = C_occenv[:,mask_occenv].copy()
            C_virenv = C_virenv[:,mask_virenv].copy()
            nbath = C_bath.shape[-1]
            self.log.debug("New number of occupied environment orbitals: %d", C_occenv.shape[-1])
            self.log.debug("New number of virtual environment orbitals: %d", C_virenv.shape[-1])
            if nbath != nref:
                err = "Number of DMET bath orbitals=%d not equal to reference=%d" % (nbath, nref)
                self.log.critical(err)
                raise RuntimeError(err)
        else:
            err = "More DMET bath orbitals found than in reference!"
            self.log.critical(err)
            raise RuntimeError(err)

    # There should never be more DMET bath orbitals than fragment orbitals
    assert nbath <= self.c_frag.shape[-1]

    return C_bath, C_occenv, C_virenv
