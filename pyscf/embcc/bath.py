"""These functions take a cluster instance as first argument ("self")."""

import logging
import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.mp
import pyscf.pbc
import pyscf.pbc.mp

from .util import *

__all__ = [
        "project_ref_orbitals",
        "make_dmet_bath",
        "make_bath",
        "make_mf_bath",
        "transform_mp2_eris",
        "run_mp2",
        "get_mp2_correction",
        "make_mp2_bath",
        ]

log = logging.getLogger(__name__)

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
    assert (C_ref.shape[-1] > 0)
    S = self.mf.get_ovlp()
    # Diagonalize reference orbitals among themselves (due to change in overlap matrix)
    C_ref = pyscf.lo.vec_lowdin(C_ref, S)
    # Diagonalize projector in space
    CSC = np.linalg.multi_dot((C_ref.T, S, C))
    P = np.dot(CSC.T, CSC)
    e, R = np.linalg.eigh(P)
    e, R = e[::-1], R[:,::-1]
    C = np.dot(C, R)

    return C, e

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
            C_occenv, eig = self.project_ref_orbitals(C_ref, C_occenv)
            mask_occref = eig >= reftol
            mask_occenv = eig < reftol
            log.debug("Eigenvalues of projected occupied reference: %s, Largest remaining: %s",
                    eig[mask_occref], max(eig[mask_occenv]))
            # --- Virtual
            C_virenv, eig = self.project_ref_orbitals(C_ref, C_virenv)
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

    log.debug("Making bath with nbath=%r and tol=%r", nbath, tol)
    e_delta_mp2 = None
    # Project
    if C_ref is not None:
        nref = C_ref.shape[-1]
        if nref > 0:
            C_env, eig = self.project_ref_orbitals(C_ref, C_env)
            log.debug("Eigenvalues of projected bath orbitals:\n%s", eig[:nref])
            log.debug("Largest remaining: %s", eig[nref:nref+3])
        C_bath, C_env = np.hsplit(C_env, [nref])

    elif bathtype == "mp2-natorb":
        log.debug("Making bath orbitals of type %s", bathtype)
        C_occclst = self.C_occclst
        C_virclst = self.C_virclst
        C_bath, C_env, e_delta_mp2 = self.make_mp2_bath(C_occclst, C_virclst, C_env, kind, nbath=nbath, tol=tol,
                **kwargs)
    elif bathtype is not None:
        log.debug("Making bath orbitals of type %s", bathtype)
        C_bath, C_env = self.make_mf_bath(C_env, kind, bathtype=bathtype, nbath=nbath, tol=tol,
                **kwargs)
    else:
        log.debug("No bath to make.")
        C_bath, C_env = np.hsplit(C_env, [0])

    if self.mp2_correction and C_env.shape[-1] > 0:
        if e_delta_mp2 is None:
            if kind == "occ":
                Co_act = np.hstack((self.C_occclst, C_bath))
                Co_all = np.hstack((Co_act, C_env))
                Cv = self.C_virclst
                e_delta_mp2 = self.get_mp2_correction(Co_all, Cv, Co_act, Cv)
            elif kind == "vir":
                Cv_act = np.hstack((self.C_virclst, C_bath))
                Cv_all = np.hstack((Cv_act, C_env))
                Co = self.C_occclst
                e_delta_mp2 = self.get_mp2_correction(Co, Cv_all, Co, Cv_act)
    else:
        e_delta_mp2 = 0.0

    return C_bath, C_env, e_delta_mp2

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

        if isinstance(nbath, (float, np.floating)):
            assert nbath >= 0.0
            assert nbath <= 1.0
            nbath_int = int(nbath*len(e) + 0.5)
            log.info("nbath = %.1f %% -> nbath = %d", nbath*100, nbath_int)
            nbath = nbath_int

        if tol is not None:
            log.warning("Warning: tolerance is %.g, but nbath=%r is used.", tol, nbath)
        nbath = min(nbath, len(e))
    elif tol is not None:
        nbath = sum(e >= tol)
    else:
        raise ValueError("Neither nbath nor tol specified.")
    log.debug("Eigenvalues of projector into %s bath space", kind)
    log.debug("%d included eigenvalues:\n%r", nbath, e[:nbath])
    log.debug("%d excluded eigenvalues (first 3):\n%r", (len(e)-nbath), e[nbath:nbath+3])
    log.debug("%d excluded eigenvalues (all):\n%r", (len(e)-nbath), e[nbath:])

    C = np.dot(C_env, R)
    C_bath, C_env = np.hsplit(C, [nbath])

    return C_bath, C_env

def run_mp2(self, Co, Cv, make_dm=False, canon_occ=True, canon_vir=True, eris=None):
    """Select virtual space from MP2 natural orbitals (NOs) according to occupation number."""

    F = self.mf.get_fock()
    Fo = np.linalg.multi_dot((Co.T, F, Co))
    Fv = np.linalg.multi_dot((Cv.T, F, Cv))
    # Canonicalization [optional]
    if canon_occ:
        eo, Ro = np.linalg.eigh(Fo)
        Co = np.dot(Co, Ro)
    else:
        eo = np.diag(Fo)
    if canon_vir:
        ev, Rv = np.linalg.eigh(Fv)
        Cv = np.dot(Cv, Rv)
    else:
        ev = np.diag(Fv)
    C = np.hstack((Co, Cv))
    eigs = np.hstack((eo, ev))
    no = Co.shape[-1]
    nv = Cv.shape[-1]
    # Use PySCF MP2 for T2 amplitudes
    occ = np.asarray(no*[2] + nv*[0])
    if self.has_pbc:
        mp2 = pyscf.pbc.mp.MP2(self.mf, mo_coeff=C, mo_occ=occ)
    else:
        mp2 = pyscf.mp.MP2(self.mf, mo_coeff=C, mo_occ=occ)
    # Integral transformation
    t0 = MPI.Wtime()
    if eris is None:
        eris = mp2.ao2mo()
        time_ao2mo = MPI.Wtime() - t0
        log.debug("Time for AO->MO: %s", get_time_string(time_ao2mo))
    # Reuse perviously obtained integral transformation into N^2 sized quantity (rather than N^4)
    else:
        eris = self.transform_mp2_eris(eris, Co, Cv)
        #eris2 = mp2.ao2mo()
        #log.debug("Eris difference=%.3e", np.linalg.norm(eris.ovov - eris2.ovov))
        #assert np.allclose(eris.ovov, eris2.ovov)
        time_mo2mo = MPI.Wtime() - t0
        log.debug("Time for MO->MO: %s", get_time_string(time_mo2mo))

    eris.nocc = mp2.nocc

    # T2 amplitudes
    e_mp2_full, T2 = mp2.kernel(mo_energy=eigs, eris=eris)
    e_mp2_full *= self.symmetry_factor
    log.debug("Full MP2 energy = %12.8g htr", e_mp2_full)

    # Calculate local energy
    # Project first occupied index onto local space
    P = self.get_local_projector(Co)
    pT2 = einsum("xi,ijab->xjab", P, T2)
    e_mp2 = self.symmetry_factor * mp2.energy(pT2, eris)

    # MP2 density matrix [optional]
    if make_dm:
        #Doo = 2*(2*einsum("ikab,jkab->ij", T2, T2)
        #         - einsum("ikab,jkba->ij", T2, T2))
        #Dvv = 2*(2*einsum("ijac,ijbc->ab", T2, T2)
        #         - einsum("ijac,ijcb->ab", T2, T2))
        Doo, Dvv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
        Doo, Dvv = -2*Doo, 2*Dvv

        # Rotate back to input coeffients (undo canonicalization)
        if canon_occ:
            Doo = np.linalg.multi_dot((Ro, Doo, Ro.T))
        if canon_vir:
            Dvv = np.linalg.multi_dot((Rv, Dvv, Rv.T))

        return e_mp2, eris, Doo, Dvv
    else:
        Doo = Dvv = None

    return e_mp2, eris, Doo, Dvv

def transform_mp2_eris(self, eris, Co, Cv):
    """Transform OVOV kind ERIS."""

    S = self.mf.get_ovlp()
    Co0, Cv0 = np.hsplit(eris.mo_coeff, [eris.nocc])
    no0 = Co0.shape[-1]
    nv0 = Cv0.shape[-1]
    no = Co.shape[-1]
    nv = Cv.shape[-1]

    transform_occ = (no != no0 or not np.allclose(Co, Co0))
    transform_vir = (nv != nv0 or not np.allclose(Cv, Cv0))
    if transform_occ:
        Ro = np.linalg.multi_dot((Co.T, S, Co0))
    if transform_vir:
        Rv = np.linalg.multi_dot((Cv.T, S, Cv0))

    govov = eris.ovov.reshape((no0, nv0, no0, nv0))
    if transform_occ and transform_vir:
        govov = einsum("xi,ya,zj,wb,iajb->xyzw", Ro, Rv, Ro, Rv, govov)
    elif transform_occ:
        govov = einsum("xi,zj,iajb->xazb", Ro, Ro, govov)
    elif transform_vir:
        govov = einsum("ya,wb,iajb->iyjw", Rv, Rv, govov)
    eris.ovov = govov.reshape((no*nv, no*nv))
    eris.mo_coeff = None
    eris.mo_energy = None
    return eris

def get_mp2_correction(self, Co1, Cv1, Co2, Cv2):
    """Calculate delta MP2 correction."""
    e_mp2_all, eris = self.run_mp2(Co1, Cv1)[:2]
    e_mp2_act = self.run_mp2(Co2, Cv2, eris=eris)[0]
    e_delta_mp2 = e_mp2_all - e_mp2_act
    log.debug("MP2 correction: all=%.4g, active=%.4g, correction=%+.4g",
            e_mp2_all, e_mp2_act, e_delta_mp2)
    return e_delta_mp2

def make_mp2_bath(self, C_occclst, C_virclst, C_env, kind, nbath=None, tol=None, **kwargs):
    """Select virtual space from MP2 natural orbitals (NOs) according to occupation number."""
    assert nbath is not None or tol is not None
    assert kind in ("occ", "vir")

    if kind == "occ":
        Co = np.hstack((C_occclst, C_env))
        Cv = C_virclst
    elif kind == "vir":
        Co = C_occclst
        Cv = np.hstack((C_virclst, C_env))

    e_mp2_all, eris, Do, Dv = self.run_mp2(Co, Cv, make_dm=True, **kwargs)

    env = np.s_[-C_env.shape[-1]:]
    if kind == "occ":
        D = Do[env,env]
    elif kind == "vir":
        D = Dv[env,env]
    N, R = np.linalg.eigh(D)
    N, R = N[::-1], R[:,::-1]
    C_env = np.dot(C_env, R)

    ##### Here we reorder the eigenvalues
    #####if True:
    ####if False:
    ####    if kind == "vir":
    ####        CR = np.dot(Cv[:,nclvir:], R)
    ####    elif kind == "occ":
    ####        CR = np.dot(Co[:,nclocc:], R)
    ####    reffile = "mp2-no-%s-ref.npz" % kind

    ####    if os.path.isfile(reffile):
    ####        ref = np.load(reffile)
    ####        N_ref, CR_ref = ref["N"], ref["CR"]

    ####        #if N_ref is not None and R_ref is not None:
    ####        log.debug("Reordering eigenvalues according to reference.")
    ####        #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v/e", return_cost=True)
    ####        reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="e^2/v", return_cost=True)
    ####        #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v*e", return_cost=True)
    ####        #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="v*sqrt(e)", return_cost=True)
    ####        #reorder, cost = eigassign(N_ref, CR_ref, N, CR, b=S, cost_matrix="evv", return_cost=True)
    ####        eigreorder_logging(N, reorder, log.debug)
    ####        log.debug("eigassign cost function value=%g", cost)
    ####        N = N[reorder]
    ####        R = R[:,reorder]
    ####        CR = CR[:,reorder]

    ####    with open("mp2-no-%s-%s-ordered.txt" % (self.name, kind), "ab") as f:
    ####        np.savetxt(f, N[np.newaxis])

    ####    np.savez(reffile, N=N, CR=CR)

    if nbath is None:
        nbath = sum(N >= tol)
    else:
        if isinstance(nbath, (float, np.floating)):
            assert nbath >= 0.0
            assert nbath <= 1.0
            nbath_int = int(nbath*len(N) + 0.5)
            log.info("nbath = %.1f %% -> nbath = %d", nbath*100, nbath_int)
            nbath = nbath_int

        nbath = min(nbath, len(N))

    ###protect_degeneracies = False
    ####protect_degeneracies = True
    #### Avoid splitting within degenerate subspace
    ###if protect_degeneracies and nno > 0:
    ###    #dgen_tol = 1e-10
    ###    N0 = N[nno-1]
    ###    while nno < len(N):
    ###        #if abs(N[nno] - N0) <= dgen_tol:
    ###        if np.isclose(N[nno], N0, atol=1e-9, rtol=1e-6):
    ###            log.debug("Degenerate MP2 NO found: %.6e vs %.6e - adding to bath space.", N[nno], N0)
    ###            nno += 1
    ###        else:
    ###            break

    log.debug("MP2 natural %s bath orbitals: active=%d, frozen=%d", kind, nbath, (len(N)-nbath))
    log.debug("Active occupation:\n%s", N[:nbath])
    log.debug("Following 3:\n%s", N[nbath:nbath+3])
    C_bath, C_env = np.hsplit(C_env, [nbath])

    # Delta MP2 correction
    # ====================

    if self.mp2_correction and C_env.shape[-1] > 0:
        S = self.mf.get_ovlp()
        if kind == "occ":
            Co_act = np.hstack((C_occclst, C_bath))
            Cv_act = Cv
        elif kind == "vir":
            Co_act = Co
            Cv_act = np.hstack((C_virclst, C_bath))

        e_mp2_act = self.run_mp2(Co_act, Cv_act, eris=eris, **kwargs)[0]
        e_delta_mp2 = e_mp2_all - e_mp2_act
        log.debug("MP2 correction (%s): all=%.4g, active=%.4g, correction=%+.4g", kind, e_mp2_all, e_mp2_act, e_delta_mp2)
    else:
        e_delta_mp2 = 0.0

    return C_bath, C_env, e_delta_mp2
