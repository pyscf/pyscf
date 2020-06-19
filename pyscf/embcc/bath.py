"""These functions take a cluster instance as first argument ("self")."""

import logging
import numpy as np
from .util import *

__all__ = [
        "make_dmet_bath",
        "make_bath",
        "make_mf_bath",
        ]


log = logging.getLogger(__name__)

def make_dmet_bath(self, C_ref=None, tol=1e-8, reftol=0.8):
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
    C_local = self.C_local
    C_env = self.C_env
    S = self.mf.get_ovlp()
    # Divide by 2 to get eigenvalues in [0,1]
    D_env = np.linalg.multi_dot((C_env.T, S, self.mf.make_rdm1(), S, C_env)) / 2
    e, R = np.linalg.eigh(D_env)
    e, R = e[::-1], R[:,::-1]
    assert np.all(e > -1e-12)
    assert np.all(e < 1+1e-12)
    C_env = np.dot(C_env, R)
    mask_bath = np.logical_and(e >= tol, e <= 1-tol)
    mask_occenv = e > 1-tol
    mask_virenv = e < tol
    nbath = sum(mask_bath)
    noccenv = sum(mask_occenv)
    nvirenv = sum(mask_virenv)
    assert (nbath + noccenv + nvirenv == C_env.shape[-1])
    C_bath = C_env[:,mask_bath].copy()
    C_occenv = C_env[:,mask_occenv].copy()
    C_virenv = C_env[:,mask_virenv].copy()

    log.debug("Found %d DMET bath orbitals. Eigenvalues:\n%s", nbath, e[mask_bath])
    log.debug("Found %d occupied orbitals.", noccenv)
    if noccenv > 0:
        log.debug("Smallest eigenvalue: %.3g", min(e[mask_occenv]))
    log.debug("Found %d virtual orbitals.", nvirenv)
    if nvirenv > 0:
        log.debug("Largest eigenvalue: %.3g", max(e[mask_virenv]))

    # Complete DMET orbital space using reference orbitals
    if C_ref is not None:
        nref = C_ref.shape[-1]
        log.debug("%d reference DMET orbitals given.", nref)
        nmissing = nref - nbath
        if nmissing == 0:
            log.debug("Number of DMET orbitals equal to reference.")
        elif nmissing > 0:
            # Perform the projection separately for occupied and virtual environment space
            # Otherwise, it is not guaranteed that the additional bath orbitals are
            # fully (or very close to fully) occupied or virtual.
            # --- Occupied
            C_occenv, eig = self.project_ref_orbitals_new(C_occenv, C_ref)
            mask_occref = eig >= reftol
            mask_occenv = eig < reftol
            log.debug("Eigenvalues of projected occupied reference: %s, Largest remaining: %s",
                    eig[mask_occref], max(eig[mask_occenv]))
            # --- Virtual
            C_virenv, eig = self.project_ref_orbitals_new(C_virenv, C_ref)
            mask_virref = eig >= reftol
            mask_virenv = eig < reftol
            log.debug("Eigenvalues of projected virtual reference: %s, Largest remaining: %s",
                    eig[mask_virref], max(eig[mask_virenv]))
            # -- Update coefficient matrices
            C_bath = np.hstack((C_bath, C_occenv[:,mask_occref], C_virenv[:,mask_virref]))
            C_occenv = C_occenv[:,mask_occenv].copy()
            C_virenv = C_virenv[:,mask_virenv].copy()
            nbath = C_bath.shape[-1]
            if nbath != nref:
                log.critical("Number of DMET bath orbitals=%d not equal to reference=%d", nbath, nref)
        else:
            log.warning("More DMET bath orbitals found than in reference!")

    # There should never be more DMET bath orbitals than fragment orbitals
    assert nbath <= C_local.shape[-1]

    return C_bath, C_occenv, C_virenv

# ================================================================================================ #

def make_bath(self, C_env, bathtype, kind, C_ref=None, nbath=None, tol=None, **kwargs):
    """Make additional bath (beyond DMET bath) orbitals.

    Parameters
    ----------
    C_env : ndarray
        Environment orbitals.
    bathtype : str
        Type of bath orbitals.
    kind : str
        Occupied or virtual.
    C_ref : ndarray, optional
        Reference bath orbitals from previous calculation.

    Returns
    -------
    C_bath : ndarray
        Bath orbitals.
    C_env : ndarray
        Environment orbitals.
    """


    # --- Occupied
    # Project
    if C_ref is not None:
        nref = C_ref.shape[-1]
        C, eig = self.project_ref_orbitals_new(C_ref, C_env)
        log.debug("Eigenvalues of projected bath orbitals:\n%s", eig[:nref])
        log.debug("Largest remaining: %s", eig[nref:nref+3])
        C_bath, C_env = np.hsplit(C, [nref])
    #elif bathtype == "matsubara":
    else:
        log.debug("Making bath orbitals of type %s", bathtype)
        C_bath, C_env = self.make_mf_bath(C_env, kind, bathtype=bathtype, nbath=nbath, tol=tol,
                **kwargs)

    return C_bath, C_env

# ================================================================================================ #

def make_mf_bath(self, C_env, kind, bathtype, nbath=None, tol=None, **kwargs):
    """Make mean-field bath orbitals.

    Parameters
    ----------
    C_env : ndarray
        Environment orbital for bath orbital construction.
    kind : str
        Occcupied or virtual bath orbitals.

    Returns
    -------
    C_bath : ndarray
        Matsubara bath orbitals.
    C_env : ndarray
        Remaining environment orbitals.
    """

    if kind == "occ":
        mask = self.mf.mo_occ > 0
    elif kind == "vir":
        mask = self.mf.mo_occ == 0

    S = self.mf.get_ovlp()
    CSC_loc = np.linalg.multi_dot((self.C_local.T, S, self.mf.mo_coeff[:,mask]))
    CSC_env = np.linalg.multi_dot((C_env.T, S, self.mf.mo_coeff[:,mask]))
    e = self.mf.mo_energy[mask]

    # Matsubara points
    if bathtype == "power":
        power = kwargs.get("power", 1)
        kernel = e**power
        B = einsum("xi,i,ai->xa", CSC_env, kernel, CSC_loc)
    elif bathtype == "matsubara":
        npoints = kwargs.get("npoints", 1000)
        beta = kwargs.get("beta", 100.0)
        wn = (2*np.arange(npoints)+1)*np.pi/beta
        kernel = wn[np.newaxis,:] / np.add.outer(e**2, wn**2)
        B = einsum("xi,iw,ai->xaw", CSC_env, kernel, CSC_loc)
        B = B.reshape(B.shape[0], B.shape[1]*B.shape[2])
    else:
        raise ValueError("Unknown bathtype: %s" % bathtype)

    P = np.dot(B, B.T)
    e, R = np.linalg.eigh(P)
    assert np.all(e > -1e-13)
    e, R = e[::-1], R[:,::-1]

    # nbath takes preference
    if nbath is not None:
        if tol is not None:
            log.warning("Warning: tolerance is %.g, but nbath=%d is used.", tol, nbath)
        nbath = min(nbath, len(e))
    elif tol is not None:
        nbath = sum(e >= tol)
    else:
        raise ValueError()
    log.debug("Eigenvalues of projector into %s bath space", kind)
    log.debug("%d included eigenvalues:\n%r", nbath, e[:nbath])
    log.debug("%d excluded eigenvalues (first 3):\n%r", (len(e)-nbath), e[nbath:nbath+3])
    log.debug("%d excluded eigenvalues (all):\n%r", (len(e)-nbath), e[nbath:])

    C = np.dot(C_env, R)
    C_bath, C_env = np.hsplit(C, [nbath])

    return C_bath, C_env
