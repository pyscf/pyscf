"""These functions take a cluster instance as first argument ("self")."""

import logging
import numpy as np
from timeit import default_timer as timer

import pyscf
import pyscf.mp
import pyscf.pbc
import pyscf.pbc.mp
import pyscf.ao2mo

from .util import *
from .tests import *
#from . import pbc_gdf_ao2mo
from . import ao2mo_j3c

__all__ = [
        "project_ref_orbitals",
        "make_dmet_bath",
        "make_bath",
        "make_local_bath",
        "make_mf_bath",
        "transform_mp2_eris",
        "run_mp2",
        "run_mp2_general",
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
    nref = C_ref.shape[-1]
    assert (nref > 0)
    assert (C.shape[-1] > 0)
    log.debug("Projecting %d reference orbitals into space of %d orbitals", nref, C.shape[-1])
    S = self.mf.get_ovlp()
    # Diagonalize reference orbitals among themselves (due to change in overlap matrix)
    C_ref_orth = pyscf.lo.vec_lowdin(C_ref, S)
    assert (C_ref_orth.shape == C_ref.shape)
    # Diagonalize projector in space
    CSC = np.linalg.multi_dot((C_ref_orth.T, S, C))
    P = np.dot(CSC.T, CSC)
    e, R = np.linalg.eigh(P)
    e, R = e[::-1], R[:,::-1]
    C = np.dot(C, R)
    #log.debug("All eigenvalues:\n%r", e)

    return C, e

def make_dmet_bath(self, C_ref=None, nbath=None, tol=1e-8, reftol=0.8):
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
    eig, R = np.linalg.eigh(D_env)
    eig, R = eig[::-1], R[:,::-1]

    if (eig.min() < -1e-9):
        log.warning("Min eigenvalue of env. DM = %.12e", eig.min())
    if ((eig.max()-1) > 1e-9):
        log.warning("Max eigenvalue of env. DM = %.12e", eig.max())

    # Entanglement spectrum
    if False:
        np.savetxt("dmet-spectrum.txt", eig)
        raise SystemExit()

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
    log.info("Number of DMET bath orbitals:        %d", nbath)
    log.info("Number of occ. environment orbitals: %d", noccenv)
    log.info("Number of vir. environment orbitals: %d", nvirenv)
    assert (nbath + noccenv + nvirenv == C_env.shape[-1])
    C_bath = C_env[:,mask_bath].copy()
    C_occenv = C_env[:,mask_occenv].copy()
    C_virenv = C_env[:,mask_virenv].copy()

    # FANCY NEW PRINTING
    print_tol = 1e-10
    strong_tol = 0.1
    #limits = [1-print_tol, 1-tol, 1-strong_tol, strong_tol, tol, print_tol]
    limits = [print_tol, tol, strong_tol, 1-strong_tol, 1-tol, 1-print_tol]
    if np.any(np.logical_and(eig > limits[0], eig <= limits[-1])):
        names = [
                "Unentangled vir. env. orbital",
                "Weakly-entangled vir. bath orbital",
                "Strongly-entangled bath orbital",
                "Weakly-entangled occ. bath orbital",
                "Unentangled occ. env. orbital",
                ]
        log.info("Non-trivial (0 or 1) eigenvalues (e) of non-fragment DM:")
        for i, e in enumerate(eig):
            name = None
            for j, llim in enumerate(limits[:-1]):
                ulim = limits[j+1]
                if (llim < e and e <= ulim):
                    name = names[j]
                    break
            if name:
                log.info("  * %-34s :  e=%12.6g  1-e=%12.6g", name, e, 1-e)

    #mask_ttol = np.logical_and(eig >= ttol, eig <= 1-ttol)
    #mask_occ = np.logical_and(eig >= 0.5, eig <= 1-ttol)
    #log.info("  1-eigenvalues of orbitals with occupation >= 0.5:")
    #for e in eig[mask_occ]:
    #    if (1-e) < tol:
    #        s = "environment"
    #    elif (1-e) < 0.1:
    #        s = "weakly entangled"
    #    else:
    #        s = "strongly entangled"
    #    log.info("    %16.10g  ->  %s occupied orbital", 1-e, s)
    #    log.info("    %s occupied orbital  ->  %16.10g", s, 1-e)
    #mask_vir = np.logical_and(eig >= ttol, eig < 0.5)
    #log.info("  eigenvalues of orbitals with occupation < 0.5:")
    #for e in eig[mask_vir]:
    #    if e < tol:
    #        s = "environment"
    #    elif e < 0.1:
    #        s = "weakly entangled"
    #    else:
    #        s = "strongly entangled"
    #    log.info("    %16.10g  ->  %s virtual orbital", e, s)

    # Calculate entanglement entropy
    entropy = np.sum(eig * (1-eig))
    entropy_bath = np.sum(eig[mask_bath] * (1-eig[mask_bath]))
    log.info("Full entanglement entropy: %.6e", entropy)
    log.info("Bath entanglement entropy: %.6e ( %.2f %%)", entropy_bath, 100*entropy_bath/entropy)

    # TEST
    #p = np.linalg.multi_dot((C_local.T, S, self.mf.make_rdm1(), S, C_local)) / 2
    #l, u = np.linalg.eigh(p)
    #log.info("Eigenvalues of occupied fragment space: %r", l)
    #entropy = np.sum(l * (1-l))
    #log.info("Entanglement entropy fragment=%.6e", entropy)

    # Complete DMET orbital space using reference orbitals
    if C_ref is not None:
        nref = C_ref.shape[-1]
        log.debug("%d reference DMET orbitals given.", nref)
        nmissing = nref - nbath

        # DEBUG
        _, eig = self.project_ref_orbitals(C_ref, C_bath)
        log.debug("Eigenvalues of reference orbitals projected into DMET bath:\n%r", eig)

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
            log.debug("Eigenvalues of projected occupied reference: %s", eig[mask_occref])
            if np.any(mask_occenv):
                log.debug("Largest remaining: %s", max(eig[mask_occenv]))
            # --- Virtual
            C_virenv, eig = self.project_ref_orbitals(C_ref, C_virenv)
            mask_virref = eig >= reftol
            mask_virenv = eig < reftol
            log.debug("Eigenvalues of projected virtual reference: %s", eig[mask_virref])
            if np.any(mask_virenv):
                log.debug("Largest remaining: %s", max(eig[mask_virenv]))
            # -- Update coefficient matrices
            C_bath = np.hstack((C_bath, C_occenv[:,mask_occref], C_virenv[:,mask_virref]))
            C_occenv = C_occenv[:,mask_occenv].copy()
            C_virenv = C_virenv[:,mask_virenv].copy()
            nbath = C_bath.shape[-1]
            log.debug("New number of occupied environment orbitals: %d", C_occenv.shape[-1])
            log.debug("New number of virtual environment orbitals: %d", C_virenv.shape[-1])
            if nbath != nref:
                err = "Number of DMET bath orbitals=%d not equal to reference=%d" % (nbath, nref)
                log.critical(err)
                raise RuntimeError(err)
        else:
            err = "More DMET bath orbitals found than in reference!"
            log.critical(err)
            raise RuntimeError(err)

    # There should never be more DMET bath orbitals than fragment orbitals
    assert nbath <= C_local.shape[-1]

    return C_bath, C_occenv, C_virenv

# ================================================================================================ #

def make_bath(self, C_env, bathtype, kind, C_ref=None, eigref=None,
        nbath=None, tol=None, energy_tol=None,
        # For new MP2 bath:
        C_occenv=None, C_virenv=None,
        **kwargs):
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
    eigref : tuple, optional
        Eigenpair reference data. If given, the bath orbitals are sorted correspondingly.

    Returns
    -------
    C_bath : ndarray
        Bath orbitals.
    C_env : ndarray
        Environment orbitals.
    """
    log.debug("type=%r, nbath=%r, tol=%r, energy_tol=%r, C_ref given=%r",
            bathtype, nbath, tol, energy_tol, C_ref is not None)

    eigref_out = None
    e_delta_mp2 = None
    # No bath (None or False)
    if not bathtype:
        C_bath, C_env = np.hsplit(C_env, [0])
        assert C_bath.shape[-1] == 0
    # Full environment as bath
    elif bathtype == "full":
        C_bath, C_env = np.hsplit(C_env, [C_env.shape[-1]])
        assert C_env.shape[-1] == 0
    # Project reference orbitals
    elif C_ref is not None:
        nref = C_ref.shape[-1]
        if nref > 0:
            C_env, eig = self.project_ref_orbitals(C_ref, C_env)
            log.debug("Eigenvalues of projected bath orbitals:\n%s", eig[:nref])
            log.debug("Largest remaining: %s", eig[nref:nref+3])
        C_bath, C_env = np.hsplit(C_env, [nref])
    # Make new bath orbitals
    else:
        #if bathtype.startswith("mp2"):
        if bathtype == "local":
            C_bath, C_env = self.make_local_bath(C_env, nbath=nbath, tol=tol)

        elif bathtype == "mp2-natorb":
            C_bath, C_env, e_delta_mp2, eigref_out = self.make_mp2_bath(
                    self.C_occclst, self.C_virclst,
                    kind,
                    c_occenv=C_occenv, c_virenv=C_virenv,
                    eigref=eigref,
                    nbath=nbath, tol=tol, energy_tol=energy_tol,
                    **kwargs)
        # MF type bath
        else:
            C_bath, C_env, eigref_out = self.make_mf_bath(C_env, kind, bathtype=bathtype, eigref=eigref, nbath=nbath, tol=tol,
                    **kwargs)

    # MP2 correction [only if there are environment (non bath) states, else = 0.0]
    #if self.mp2_correction and C_env.shape[-1] > 0:
    kind2n = {"occ" : 0, "vir" : 1}
    if self.mp2_correction[kind2n[kind]] and C_env.shape[-1] > 0:
        if e_delta_mp2 is None:
            if kind == "occ":
                log.debug("Calculating occupied MP2 correction.")
                Co_act = np.hstack((self.C_occclst, C_bath))
                Co_all = np.hstack((Co_act, C_env))
                Cv = self.C_virclst
                e_delta_mp2 = self.get_mp2_correction(Co_all, Cv, Co_act, Cv)
            elif kind == "vir":
                log.debug("Calculating virtual MP2 correction.")
                Cv_act = np.hstack((self.C_virclst, C_bath))
                Cv_all = np.hstack((Cv_act, C_env))
                Co = self.C_occclst
                e_delta_mp2 = self.get_mp2_correction(Co, Cv_all, Co, Cv_act)

    else:
        log.debug("No MP2 correction.")
        e_delta_mp2 = 0.0

    return C_bath, C_env, e_delta_mp2, eigref_out

# ================================================================================================ #

def make_mf_bath(self, C_env, kind, bathtype, eigref=None, nbath=None, tol=None, **kwargs):
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
        mask = self.base.mo_occ > 0
    elif kind == "vir":
        mask = self.base.mo_occ == 0

    S = self.mf.get_ovlp()
    CSC_loc = np.linalg.multi_dot((self.C_local.T, S, self.base.mo_coeff[:,mask]))
    CSC_env = np.linalg.multi_dot((C_env.T, S, self.base.mo_coeff[:,mask]))
    e = self.base.mo_energy[mask]

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
    # For testing
    elif bathtype == "random":
        B = 2*np.random.rand(CSC_env.shape[0], 500)-1
    else:
        raise ValueError("Unknown bathtype: %s" % bathtype)

    P = np.dot(B, B.T)
    e, R = np.linalg.eigh(P)
    assert np.all(e > -1e-13)
    e, R = e[::-1], R[:,::-1]
    C_env = np.dot(C_env, R)

    # Reorder here
    if False:
        with open("bath-%s-%s.txt" % (self.name, kind), "ab") as f:
            np.savetxt(f, e[np.newaxis])

    # Here we reorder the eigenvalues
    if True:
        if eigref is not None:
            log.debug("eigref given: performing reordering of eigenpairs.")
            # Get reordering array
            reorder, cost = eigassign(eigref[0], eigref[1], e, C_env, b=S, cost_matrix="er/v")
            log.debug("Optimized linear assignment cost=%.3e", cost)

            e = e[reorder]
            C_env = C_env[:,reorder]

        if False:
            with open("bath-%s-%s-ordered.txt" % (self.name, kind), "ab") as f:
                np.savetxt(f, e[np.newaxis])

    # Reference for future calculation
    eigref_out = (e.copy(), C_env.copy())

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

    C_bath, C_env = np.hsplit(C_env, [nbath])

    return C_bath, C_env, eigref_out

# ================================================================================================ #

#def run_mp2(self, Co, Cv, make_dm=False, canon_occ=True, canon_vir=True, eris=None):
##def run_mp2(self, Co, Cv, make_dm=False, canon_occ=False, canon_vir=True, eris=None):
##def run_mp2(self, Co, Cv, make_dm=False, canon_occ=False, canon_vir=False, eris=None):
#    """Select virtual space from MP2 natural orbitals (NOs) according to occupation number."""
#
#    #F = self.mf.get_fock()
#    #F = self.base.fock
#    F = self.base.get_fock()
#    Fo = np.linalg.multi_dot((Co.T, F, Co))
#    Fv = np.linalg.multi_dot((Cv.T, F, Cv))
#    # Canonicalization [optional]
#    if canon_occ:
#        eo, Ro = np.linalg.eigh(Fo)
#        Co = np.dot(Co, Ro)
#    else:
#        eo = np.diag(Fo)
#    if canon_vir:
#        ev, Rv = np.linalg.eigh(Fv)
#        Cv = np.dot(Cv, Rv)
#    else:
#        ev = np.diag(Fv)
#    C = np.hstack((Co, Cv))
#    eigs = np.hstack((eo, ev))
#    no = Co.shape[-1]
#    nv = Cv.shape[-1]
#    # Use PySCF MP2 for T2 amplitudes
#    occ = np.asarray(no*[2] + nv*[0])
#    if self.has_pbc:
#        mp2 = pyscf.pbc.mp.MP2(self.mf, mo_coeff=C, mo_occ=occ)
#    else:
#        mp2 = pyscf.mp.MP2(self.mf, mo_coeff=C, mo_occ=occ)
#    # Integral transformation
#    t0 = MPI.Wtime()
#    if eris is None:
#        eris = mp2.ao2mo()
#        # Does not work for DF at the moment...
#        assert (eris.ovov is not None)
#        time_ao2mo = MPI.Wtime() - t0
#        log.debug("Time for AO->MO: %s", get_time_string(time_ao2mo))
#    # Reuse perviously obtained integral transformation into N^2 sized quantity (rather than N^4)
#    else:
#        eris = self.transform_mp2_eris(eris, Co, Cv)
#        #eris2 = mp2.ao2mo()
#        #log.debug("Eris difference=%.3e", np.linalg.norm(eris.ovov - eris2.ovov))
#        #assert np.allclose(eris.ovov, eris2.ovov)
#        time_mo2mo = MPI.Wtime() - t0
#        log.debug("Time for MO->MO: %s", get_time_string(time_mo2mo))
#
#    eris.nocc = mp2.nocc
#
#    # T2 amplitudes
#    #e_mp2_full, T2 = mp2.kernel(mo_energy=eigs, eris=eris)
#    e_mp2_full, T2 = mp2.kernel(mo_energy=eigs, eris=eris, hf_reference="legacy")
#    e_mp2_full *= self.symmetry_factor
#    log.debug("Full MP2 energy = %12.8g htr", e_mp2_full)
#
#    # Calculate local energy
#    # Project first occupied index onto local space
#    _, pT2 = self.get_local_amplitudes(mp2, None, T2, symmetrize=False)
#    #_, pT2 = self.get_local_amplitudes(mp2, None, T2, variant="democratic")
#    e_mp2 = self.symmetry_factor * mp2.energy(pT2, eris)
#
#    # MP2 density matrix [optional]
#    if make_dm:
#        #Doo = 2*(2*einsum("ikab,jkab->ij", T2, T2)
#        #         - einsum("ikab,jkba->ij", T2, T2))
#        #Dvv = 2*(2*einsum("ijac,ijbc->ab", T2, T2)
#        #         - einsum("ijac,ijcb->ab", T2, T2))
#        Doo, Dvv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
#        Doo, Dvv = -2*Doo, 2*Dvv
#
#        # Rotate back to input coeffients (undo canonicalization)
#        if canon_occ:
#            Doo = np.linalg.multi_dot((Ro, Doo, Ro.T))
#        if canon_vir:
#            Dvv = np.linalg.multi_dot((Rv, Dvv, Rv.T))
#    else:
#        Doo = Dvv = None
#
#    # Rotate T2 back
#    T2 = einsum("ijab,ki,lj,ca,db->klcd", T2, Ro, Ro, Rv, Rv)
#
#    #return e_mp2, eris, Doo, Dvv
#    return e_mp2, eris, Doo, Dvv, T2


def run_mp2(self, c_occ, c_vir, c_occenv=None, c_virenv=None, canonicalize=True, eris=None, local_dm=False):
    """Select virtual space from MP2 natural orbitals (NOs) according to occupation number.

    Parameters
    ----------
    c_occ : ndarray
        Active occupied orbitals.
    c_vir : ndarray
        Active virtual orbitals.
    c_occenv : ndarray, optional
        Frozen occupied orbitals.
    c_virenv : ndarray, optional
        Frozen virtual orbitals.
    canonicalize : bool, tuple(2), optional
        Canonicalize occupied/virtual active orbitals.
    eris: TODO

    Returns
    -------
    TODO
    """

    # Canonicalization [optional]
    if canonicalize in (True, False):
        canonicalize = 2*[canonicalize]
    f = self.base.get_fock()
    fo = np.linalg.multi_dot((c_occ.T, f, c_occ))
    fv = np.linalg.multi_dot((c_vir.T, f, c_vir))
    if canonicalize[0]:
        eo, ro = np.linalg.eigh(fo)
        c_occ = np.dot(c_occ, ro)
    else:
        eo = np.diag(fo)
    if canonicalize[1]:
        ev, rv = np.linalg.eigh(fv)
        c_vir = np.dot(c_vir, rv)
    else:
        ev = np.diag(fv)

    # Setup MP2 object
    nao = c_occ.shape[0]
    if c_occenv is None:
        c_occenv = np.zeros((nao, 0))
    if c_virenv is None:
        c_virenv = np.zeros((nao, 0))
    c_act = np.hstack((c_occ, c_vir))
    c_all = np.hstack((c_occenv, c_act, c_virenv))
    norb = c_all.shape[-1]
    noccenv = c_occenv.shape[-1]
    nvirenv = c_virenv.shape[-1]
    frozen = list(range(noccenv)) + list(range(norb-nvirenv, norb))
    #log.debug("Active orbitals: %d occupied and %d virtual", c_occ.shape[-1], c_vir.shape[-1])
    #log.debug("Freezing %d occupied and %d virtual orbitals with indices:\n%r", noccenv, nvirenv, frozen)
    if self.use_pbc:
        cls = pyscf.pbc.mp.MP2
    else:
        cls = pyscf.mp.MP2
    mp2 = cls(self.mf, mo_coeff=c_all, frozen=frozen)

    # Integral transformation
    t0 = timer()
    if eris is None:
        # For PBC [direct_init to avoid expensive Fock rebuild]
        if self.use_pbc:
            fock = np.linalg.multi_dot((c_act.T, f, c_act))
            # TRY NEW
            if hasattr(self.mf.with_df, "_cderi") and isinstance(self.mf.with_df._cderi, np.ndarray):
                #eris = pbc_gdf_ao2mo.ao2mo(mp2, fock=fock, mp2=True)
                # TEST NEW
                #mo_energy = np.hstack((eo, ev))
                #eris = ao2mo_j3c.ao2mo_mp2(mp2, mo_energy=mo_energy)
                eris = ao2mo_j3c.ao2mo_mp2(mp2, fock=fock)
                #assert np.allclose(eris.mo_energy, eris2.mo_energy)
                #assert np.allclose(eris.ovov, eris2.ovov)

            else:
                mo_energy = np.hstack((eo, ev))
                eris = mp2.ao2mo(direct_init=True, mo_energy=mo_energy, fock=fock)
        # For molecular calculations with DF
        elif hasattr(mp2, "with_df"):
            eris = mp2.ao2mo(store_eris=True)
        else:
            eris = mp2.ao2mo()
        # PySCF forgets to set this...
        if eris.nocc is None:
            eris.nocc = c_occ.shape[-1]
        #log.debug("Made eris with no=%d nv=%d n=%d", c_occ.shape[-1], c_vir.shape[-1], eris.ovov[:].size)
    # Reuse perviously obtained integral transformation into N^2 sized quantity (rather than N^4)
    else:
        log.debug("Transforming previous eris.")
        #raise NotImplementedError()
        eris = self.transform_mp2_eris(eris, c_occ, c_vir)
        #eris2 = mp2.ao2mo()
        #eris2 = mp2.ao2mo(direct_init=True, mo_energy=np.hstack((eo, ev)))
        #log.debug("Eris difference=%.3e", np.linalg.norm(eris.ovov - eris2.ovov))
        #assert np.allclose(eris.ovov, eris2.ovov)
    t = (timer() - t0)
    if t > 1:
        log.debug("Time for integral transformation [s]: %.3f (%s)", t, get_time_string(t))
    assert (eris.ovov is not None)

    t0 = timer()
    e_mp2_full, t2 = mp2.kernel(eris=eris, hf_reference=True)
    log.info("Min(|T2|)= %.2e Max(|T2|)= %.2e", abs(t2).min(), abs(t2).max())
    t = (timer() - t0)
    if t > 10:
        log.debug("Time for MP2 kernel [s]: %.3f (%s)", t, get_time_string(t))
    e_mp2_full *= self.symmetry_factor
    log.debug("Full MP2 energy=  %12.8g htr", e_mp2_full)

    # Exact MP2 amplitudes
    #if True and hasattr(self.base, "_t2_exact"):
    if False:
        c_mf_occ = self.base.mo_coeff[:,self.base.mo_occ>0]
        c_mf_vir = self.base.mo_coeff[:,self.base.mo_occ==0]
        s = self.base.get_ovlp()
        csco = np.linalg.multi_dot((c_mf_occ.T, s, c_occ))
        cscv = np.linalg.multi_dot((c_mf_vir.T, s, c_vir))
        t2_exact = einsum("ijab,ik,jl,ac,bd->klcd", self.base._t2_exact, csco, csco, cscv, cscv)
        assert t2_exact.shape == t2.shape
        log.info("Difference T2: %g", np.linalg.norm(t2 - t2_exact))
        t2 = t2_exact

    no, nv = t2.shape[0], t2.shape[2]

    # Bild T2 manually [testing]
    #t2_man = np.zeros_like(t2)
    #eris_ovov = np.asarray(eris.ovov).reshape(no,nv,no,nv)
    #for i in range(no):
    #    gi = eris_ovov[i].transpose(1, 0, 2)
    #    ei = fo[i] + np.subtract.outer(fo, np.add.outer(fv, fv))
    #    assert (gi.shape == ei.shape)
    #    t2i = - gi**2 / ei
    #    t2_man[i] = t2i
    ##assert np.allclose(t2, t2_man)
    ##1/0

    # Calculate local energy
    # Project first occupied index onto local space
    _, t2loc = self.get_local_amplitudes(mp2, None, t2, symmetrize=True)
    log.info("Min(|pT2|)= %.2e Max(|pT2|)= %.2e", abs(t2loc).min(), abs(t2loc).max())

    e_mp2 = self.symmetry_factor * mp2.energy(t2loc, eris)
    log.debug("Local MP2 energy= %12.8g htr", e_mp2)

    # MP2 density matrix
    if local_dm is True:
        log.debug("Constructing DM from local T2 amplitudes.")
        t2l, t2r = t2loc, t2loc
    elif local_dm == "semi":
        log.debug("Constructing DM from semi-local T2 amplitudes.")
        t2l, t2r = t2loc, t2
    elif local_dm is False:
        log.debug("Constructing DM from full T2 amplitudes.")
        t2l, t2r = t2, t2
        # This is equivalent to:
        # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
        # do, dv = -2*do, 2*dv
    else:
        raise ValueError("Unknown value for local_dm: %r" % local_dm)
    dmo = 2*(2*einsum("ik...,jk...->ij", t2l, t2r)
             - einsum("ik...,kj...->ij", t2l, t2r))
    dmv = 2*(2*einsum("...ac,...bc->ab", t2l, t2r)
             - einsum("...ac,...cb->ab", t2l, t2r))
    if local_dm == "semi":
        dmo = (dmo + dmo.T)/2
        dmv = (dmv + dmv.T)/2

    # Rotate back to input coeffients (undo canonicalization)
    if canonicalize[0]:
        t2 = einsum("ij...,xi,yj->xy...", t2, ro, ro)
        dmo = np.linalg.multi_dot((ro, dmo, ro.T))
        #g = eris.ovov[:].reshape((no, nv, no, nv))
        #g = einsum("iajb,xi,yj->xayb", g, ro, ro)
        #eris.ovov = g.reshape((no*nv, no*nv))
        #eris.mo_coeff = None
    if canonicalize[1]:
        t2 = einsum("...ab,xa,yb->...xy", t2, rv, rv)
        dmv = np.linalg.multi_dot((rv, dmv, rv.T))
        #g = eris.ovov[:].reshape((no, nv, no, nv))
        #g = einsum("iajb,xa,yb->ixjy", g, rv, rv)
        #eris.ovov = g.reshape((no*nv, no*nv))
        #eris.mo_coeff = None

    assert np.allclose(dmo, dmo.T)
    assert np.allclose(dmv, dmv.T)

    return t2, eris, dmo, dmv, e_mp2

def run_mp2_general(self, c_occ, c_vir, c_occ2=None, c_vir2=None, eris=None, canonicalize=True):
    """Run MP2 calculations for general orbitals..

    if c_occ1 == c_occ2 and c_vir1 == c_vir2, the PySCF kernel can be used
    """

    if canonicalize in (True, False):
        canonicalize = 4*[canonicalize]

    if c_occ2 is None:
        c_occ2 = c_occ
    if c_vir2 is None:
        c_vir2 = c_vir

    equal_c_occ = (c_occ is c_occ2)
    equal_c_vir = (c_vir is c_vir2)
    normal_mp2 = (equal_c_occ and equal_c_vir)

    # Different fock matrix for PBC with exxdiv?
    coeffs_in = [c_occ, c_occ2, c_vir, c_vir2]
    fock = self.base.get_fock()

    eigs = []
    rots = []
    coeffs = []
    for i in range(4):
        f = np.linalg.multi_dot((coeffs_in[i].T, fock, coeffs_in[i]))
        if canonicalize[i]:
            if (i == 1 and equal_c_occ) or (i == 3 and equal_c_vir):
                e, r = eigs[-1], rots[-1]
            else:
                e, r = np.linalg.eigh(f)
            eigs.append(e)
            rots.append(r)
            coeffs.append(np.dot(coeffs_in[i], r))
        else:
            eigs.append(np.diag(f))
            rots.append(None)
            coeffs.append(coeffs_in[i])

    sizes = [c.shape[-1] for c in coeffs]

    # Reordering [this function is its own inverse]
    reorder = lambda x: (x[0], x[2], x[1], x[3])

    # Integral transformation
    if eris is None:
        # Does not work for DF at the moment...
        #assert (eris.ovov is not None)
        # TEST
        t0 = timer()
        if getattr(self.mf, "with_df", False):
            g_ovov = self.mf.with_df.ao2mo(reorder(coeffs))
        elif self.mf._eri is not None:
            g_ovov = pyscf.ao2mo.general(self.mf._eri, reorder(coeffs))
        else:
            # Out core...
            # Temporary:
            g_ovov = pyscf.ao2mo.general(self.mol, reorder(coeffs))

        g_ovov = g_ovov.reshape(reorder(sizes))
        time_ao2mo = timer() - t0
        log.debug("Time for AO->MO transformation of ERIs: %s", get_time_string(time_ao2mo))
    # Reuse perviously obtained integral transformation into N^2 sized quantity (rather than N^4)
    else:
        raise NotImplementedError()
        #eris = self.transform_mp2_eris(eris, Co, Cv)
        ##eris2 = mp2.ao2mo()
        ##log.debug("Eris difference=%.3e", np.linalg.norm(eris.ovov - eris2.ovov))
        ##assert np.allclose(eris.ovov, eris2.ovov)
        #time_mo2mo = MPI.Wtime() - t0
        #log.debug("Time for MO->MO: %s", get_time_string(time_mo2mo))

    # Make T2 amplitudes and energy
    e2_full = 0.0
    t2 = np.zeros(sizes)
    for i in range(sizes[0]):
        g_iovv = g_ovov[i].transpose(1, 0, 2)
        e_iovv = eigs[0][i] + np.subtract.outer(eigs[1], np.add.outer(eigs[2], eigs[3]))
        assert (g_iovv.shape == e_iovv.shape)
        t2[i] = g_iovv / e_iovv
        #e2_full += (2*einsum("jab,jab", t2[i], g_iovv)
        #            - einsum("jab,jba", t2[i], g_iovv))


    e2_full *= self.symmetry_factor
    log.debug("Full MP2 energy = %12.8g htr", e2_full)

    # Calculate local energy
    # Project first occupied index onto local space
    #_, pT2 = self.get_local_amplitudes(mp2, None, T2, symmetrize=False)
    #_, pT2 = self.get_local_amplitudes(mp2, None, T2, variant="democratic")
    #e_mp2 = self.symmetry_factor * mp2.energy(pT2, eris)

    ploc = self.get_local_projector(coeffs[0])
    t2loc = einsum("ix,i...->x...", ploc, t2)

    # MP2 density matrix
    # In order to get a symmetric DM in the end we calculate it twice ASSUMING the normal symmetry of T2
    # (In reality it does not have this symmetry)
    # This is different to symmetrizing the DM at the end (i.e. dm2 is not dm.T).
    # Idealy, we would symmetrize T2 itself, but this is not possible here, since the dimensions to not match...
    # Is what is done here still equivalent?
    #if kind == "occupied":
    #    dm = 2*(2*einsum("ikab,jkab->ij", t2, t2)
    #            - einsum("ikab,kjab->ij", t2, t2))
    #    dm2 = 2*(2*einsum("kiab,kjab->ij", t2, t2)
    #             - einsum("kiab,jkab->ij", t2, t2))
    #    dm = (dm + dm2)/2

    #    r = rotations[0]
    #elif kind == "virtual":
    #    dm = 2*(2*einsum("ijac,ijbc->ab", t2, t2)
    #            - einsum("ijac,ijcb->ab", t2, t2))
    #    dm2 = 2*(2*einsum("ijca,ijcb->ab", t2, t2)
    #             - einsum("ijca,ijbc->ab", t2, t2))
    #    dm = (dm + dm2)/2

    #    r = rotations[3]
    #assert dm.shape[0] == dm.shape[1]

    # Here we ignore the exchange like contributions...
    #dm_occ = 4*einsum("ikab,jkab->ij", t2, t2)
    #dm_vir = 4*einsum("ijac,ijbc->ab", t2, t2)
    dm_occ = 4*einsum("ikab,jkab->ij", t2loc, t2loc)
    dm_vir = 4*einsum("ijac,ijbc->ab", t2loc, t2loc)
    assert np.allclose(dm_occ, dm_occ.T)
    assert np.allclose(dm_vir, dm_vir.T)

    # Undo canonicalization
    if rots[0] is not None:
        dm_occ = np.linalg.multi_dot((rots[0], dm_occ, rots[0].T))
    if rots[2] is not None:
        dm_vir = np.linalg.multi_dot((rots[2], dm_vir, rots[2].T))

    ## Determine symmetry
    #dm_sym = (dm + dm.T)
    #dm_anti = (dm - dm.T)
    #norm_sym = np.linalg.norm(dm_sym)
    #norm_anti = np.linalg.norm(dm_anti)
    #sym = (norm_sym - norm_anti) / (norm_sym + norm_anti)
    #log.info("Symmetry of DM = %.4g", sym)
    ## Ignore antisymmetric part:
    #dm = dm_sym
    #assert np.allclose(dm, dm.T)

    #return e_mp2, eris, Doo, Dvv
    return t2, dm_occ, dm_vir

# ================================================================================================ #

def transform_mp2_eris(self, eris, Co, Cv):
    """Transform eris of kind (ov|ov) (occupied-virtual-occupied-virtual)"""
    assert (eris is not None)
    assert (eris.ovov is not None)

    S = self.mf.get_ovlp()
    Co0, Cv0 = np.hsplit(eris.mo_coeff, [eris.nocc])
    #log.debug("eris.nocc = %d", eris.nocc)
    no0 = Co0.shape[-1]
    nv0 = Cv0.shape[-1]
    no = Co.shape[-1]
    nv = Cv.shape[-1]

    transform_occ = (no != no0 or not np.allclose(Co, Co0))
    if transform_occ:
        Ro = np.linalg.multi_dot((Co.T, S, Co0))
    else:
        Ro = np.eye(no)
    transform_vir = (nv != nv0 or not np.allclose(Cv, Cv0))
    if transform_vir:
        Rv = np.linalg.multi_dot((Cv.T, S, Cv0))
    else:
        Rv = np.eye(nv)
    R = np.block([
        [Ro, np.zeros((no, nv0))],
        [np.zeros((nv, no0)), Rv]])

    #govov = eris.ovov.reshape((no0, nv0, no0, nv0))
    # eris.ovov may be hfd5 dataset on disk -> allocate in memory with [:]
    govov = eris.ovov[:].reshape((no0, nv0, no0, nv0))
    if transform_occ and transform_vir:
        govov = einsum("xi,ya,zj,wb,iajb->xyzw", Ro, Rv, Ro, Rv, govov)
    elif transform_occ:
        govov = einsum("xi,zj,iajb->xazb", Ro, Ro, govov)
    elif transform_vir:
        govov = einsum("ya,wb,iajb->iyjw", Rv, Rv, govov)
    eris.ovov = govov.reshape((no*nv, no*nv))
    eris.mo_coeff = np.hstack((Co, Cv))
    eris.fock = np.linalg.multi_dot((R, eris.fock, R.T))
    eris.mo_energy = np.diag(eris.fock)
    return eris

# ================================================================================================ #

def get_mp2_correction(self, Co1, Cv1, Co2, Cv2):
    """Calculate delta MP2 correction."""
    e_mp2_all, eris = self.run_mp2(Co1, Cv1)[:2]
    e_mp2_act = self.run_mp2(Co2, Cv2, eris=eris)[0]
    e_delta_mp2 = e_mp2_all - e_mp2_act
    log.debug("MP2 correction: all=%.4g, active=%.4g, correction=%+.4g",
            e_mp2_all, e_mp2_act, e_delta_mp2)
    return e_delta_mp2

# ================================================================================================ #

def make_mp2_bath(self,
        c_occclst, c_virclst,
        #C_env,
        kind,
        # For new MP2 bath:
        #c_occenv=None, c_virenv=None,
        c_occenv, c_virenv,
        eigref=None,
        nbath=None, tol=None, energy_tol=None):
    """Select occupied or virtual bath space from MP2 natural orbitals.

    The natural orbitals are calculated only including the local virtual (occupied)
    cluster orbitals when calculating occupied (virtual) bath orbitals, i.e. they do not correspond
    to the full system MP2 natural orbitals and are different for every cluster.

    Set nbath to a very high number or tol to -1 to get all bath orbitals.

    Parameters
    ----------
    C_occclst : ndarray
        Occupied cluster (fragment + DMET bath) orbitals.
    C_virclst : ndarray
        Virtual cluster (fragment + DMET bath) orbitals.
    C_env : ndarray
        Environment orbitals. These need to be off purely occupied character if kind=="occ"
        and of purely virtual character if kind=="vir".
    kind : str ["occ", "vir"]
        Calculate occupied or virtual bath orbitals.
    eigref : tuple(eigenvalues, eigenvectors), optional
        Reference eigenvalues and eigenvectors from previous calculation for consistent sorting
        (see also: eigref_out).
    nbath : int, optional
        Target number of bath orbitals. If given, tol is ignored.
        The actual number of bath orbitals might be lower, if not enough
        environment orbitals are available.
    tol : float, optional
        Occupation change tolerance for bath orbitals. Ignored if nbath is not None.
        Should be chosen between 0 and 1.

    Returns
    -------
    C_bath : ndarray
        MP2 natural orbitals with the largest occupation change.
    C_env : ndarray
        Remaining MP2 natural orbitals.
    e_delta_mp2 : float
        MP2 correction energy (0 if self.mp2_correction == False)
    eigref_out : tuple(eigenvalues, eigenvectors)
        Reference eigenvalues and eigenvectors for future calculation for consistent sorting
        (see also: eigref).
    """
    if nbath is None and tol is None and energy_tol is None:
        raise ValueError("nbath, tol, and energy_tol are None.")
    if kind not in ("occ", "vir"):
        raise ValueError("Unknown kind: %s", kind)
    kindname = {"occ": "occupied", "vir" : "virtual"}
    kind = kindname[kind]

    # All occupied and virtual orbitals
    c_occall = np.hstack((c_occclst, c_occenv))
    c_virall = np.hstack((c_virclst, c_virenv))

    # All occupied orbitals, only cluster virtual orbitals
    if kind == "occupied":
        c_occ = c_occall
        c_vir = c_virclst
        c_env = c_occenv
        t2, eris, dm, _, e_mp2_all = run_mp2(
                self, c_occ, c_vir, c_occenv=None, c_virenv=c_virenv,
                local_dm=False)
        ncluster = c_occclst.shape[-1]
        nenv = c_occ.shape[-1] - ncluster

    # All virtual orbitals, only cluster occupied orbitals
    elif kind == "virtual":
        c_occ = c_occclst
        c_vir = c_virall
        c_env = c_virenv
        t2, eris, _, dm, e_mp2_all = run_mp2(
                self, c_occ, c_vir, c_occenv=c_occenv, c_virenv=None)
        ncluster = c_virclst.shape[-1]
        nenv = c_vir.shape[-1] - ncluster

    # Diagonalize environment-environment block of MP2 DM correction
    # and rotate into natural orbital basis, with the orbitals sorted
    # with decreasing (absolute) occupation change
    # [Note that dm_occ is minus the change of the occupied DM]

    env = np.s_[ncluster:]
    dm = dm[env,env]
    dm_occ, dm_rot = np.linalg.eigh(dm)
    assert (len(dm_occ) == nenv)
    if np.any(dm_occ < -1e-12):
        raise RuntimeError("Negative occupation values detected: %r" % dm_occ[dm_occ < -1e-12])
    dm_occ, dm_rot = dm_occ[::-1], dm_rot[:,::-1]
    c_rot = np.dot(c_env, dm_rot)

    if False:
        with open("MP2-natorb-%s-%s.txt" % (self.name, kind), "ab") as f:
            np.savetxt(f, dm_occ[np.newaxis])

    # Reorder the eigenvalues and vectors according to the reference ordering, specified in eigref
    if False:
        if eigref is not None:
            log.debug("eigref given: performing reordering of eigenpairs.")
            # Get reordering array
            S = self.base.get_ovlp()
            reorder, cost = eigassign(eigref[0], eigref[1], N, C_no, b=S, cost_matrix="er/v")
            log.debug("Optimized linear assignment cost=%.3e", cost)
            N = N[reorder]
            C_no = C_no[:,reorder]
        if False:
            with open("MP2-natorb-%s-%s-ordered.txt" % (self.name, kind), "ab") as f:
                np.savetxt(f, N[np.newaxis])

    # Return ordered eigenpairs as a reference for future calculations
    eigref_out = (dm_occ.copy(), c_rot.copy())

    # --- Determine number of bath orbitals
    # The number of bath orbitals can be determined in 3 ways, in decreasing priority:
    #
    # 1) Via nbath. If nbath is an integer, this number will be used directly.
    # If nbath is a float between 0 and 1, it denotes the relative number of bath orbitals
    # 2) Via an occupation tolerance: tol
    # 3) Via an energy tolerance: energy_tol
    #
    e_delta_mp2 = None
    if nbath is not None:
        #if isinstance(nbath, (float, np.floating)):
        #    assert nbath >= 0.0
        #    assert nbath <= 1.0
        #    nbath_int = int(nbath*len(dm_occ) + 0.5)
        #    log.info("nbath = %.1f %% -> nbath = %d", nbath*100, nbath_int)
        #    nbath = nbath_int

        # Changed behavior!!!
        #if isinstance(nbath, (float, np.floating)):
        #    assert ((nbath >= 0.0) and (nbath <= 1.0))
        #    dm_occ_tot = np.sum(dm_occ)
        #    # Add bath orbitals
        #    for n in range(len(dm_occ)+1):
        #        dm_occ_n = np.sum(dm_occ[:n])
        #        if (dm_occ_n / dm_occ_tot) >= nbath:
        #            break
        #    nbath = n
        #    log.info("(De)occupation of environment space: all %d orbitals= %.5f  %d bath orbitals= %.5f ( %.3f%% )",
        #            len(dm_occ), dm_occ_tot, nbath, dm_occ_n, 100.0*dm_occ_n/dm_occ_tot)

        nbath = min(nbath, len(dm_occ))
    # Determine number of bath orbitals based on occupation tolerance
    elif tol is not None:
        if tol >= 0.0:
            nbath = sum(dm_occ >= tol)
        # Changed behavior!!!
        else:
            tol = 1+tol
            assert ((tol >= 0.0) and (tol <= 1.0))
            dm_occ_tot = np.sum(dm_occ)
            # Add bath orbitals
            for n in range(len(dm_occ)+1):
                dm_occ_n = np.sum(dm_occ[:n])
                if (dm_occ_n / dm_occ_tot) >= tol:
                    break
            nbath = n
            log.info("(De)occupation of environment space: all %d orbitals= %.5f  %d bath orbitals= %.5f ( %.3f%% )",
                    len(dm_occ), dm_occ_tot, nbath, dm_occ_n, 100.0*dm_occ_n/dm_occ_tot)
            assert (nbath <= len(dm_occ))

    # Determine number of bath orbitals based on energy tolerance
    elif energy_tol is not None:
        _, t2_loc = self.get_local_amplitudes_general(None, t2, c_occ, c_vir)
        # Rotate T2 and ERI into NO basis
        no, nv = t2_loc.shape[1:3]
        g = np.asarray(eris.ovov).reshape(no,nv,no,nv)
        rot = np.block(
                [[np.eye(ncluster), np.zeros((ncluster, nenv))],
                [np.zeros((nenv, ncluster)), dm_rot]])
        if kind == "occupied":
            t2_loc = einsum("ij...,ix,jy->xy...", t2_loc, rot, rot)
            g = einsum("iajb,ix,jy->xayb", g, rot, rot)
            em = (2*einsum("ijab,iajb->ij", t2_loc, g)
                  - einsum("ijab,ibja->ij", t2_loc, g))
        elif kind == "virtual":
            t2_loc = einsum("...ab,ax,by->...xy", t2_loc, rot, rot)
            g = einsum("iajb,ax,by->ixjy", g, rot, rot)
            em = (2*einsum("ijab,iajb->ab", t2_loc, g)
                  - einsum("ijab,ibja->ab", t2_loc, g))

        e_full = np.sum(em)
        log.debug("Full energy=%g", e_full)
        nbath, err = 0, e_full      # in case nenv == 0
        for nbath in range(nenv+1):
            act = np.s_[:ncluster+nbath]
            e_act = np.sum(em[act,act])
            err = (e_full - e_act)
            #log.debug("%3d bath orbitals: %8.6g , error: %8.6g", nbath, e_act, err)

            if abs(err) < energy_tol:
                log.debug("%3d bath orbitals: %8.6g , error: %8.6g", nbath, e_act, err)
                log.debug("Energy tolerance %.1e achieved with %d bath orbitals.", energy_tol, nbath)
                break

        e_delta_mp2 = err

    # Check for degenerate subspaces, split by nbath
    while (nbath > 0) and (nbath < nenv) and np.isclose(dm_occ[nbath-1], dm_occ[nbath], rtol=1e-5, atol=1e-11):
        log.warning("Bath space is splitting a degenerate subspace of occupation numbers: %.8e and %.8e",
                    dm_occ[nbath-1], dm_occ[nbath])
        nbath += 1
        log.warning("Adding one additional bath orbital.")

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

    # Split MP2 natural orbitals into bath and environment
    c_bath, c_env = np.hsplit(c_rot, [nbath])

    # Output some information
    log.info("%s MP2 natural orbitals:" % kind.title())
    if nbath > 0:
        log.info("  * %3d bath orbitals:        largest=%6.3g, smallest=%6.3g, mean=%6.3g.",
            nbath, max(dm_occ[:nbath]), min(dm_occ[:nbath]), np.mean(dm_occ[:nbath]))
    if nbath < nenv:
        log.info("  * %3d environment orbitals: largest=%6.3g, smallest=%6.3g, mean=%6.3g.",
                nenv-nbath, max(dm_occ[nbath:]), min(dm_occ[nbath:]), np.mean(dm_occ[nbath:]))
    #log.debug("All:\n%r", dm_occ)

    # Delta MP2 correction
    kind2int = {"occupied" : 0, "virtual" : 1}
    #if self.mp2_correction and C_env.shape[-1] > 0:
    # No correction if all natural orbitals are already included (0 environment orbitals)
    if self.mp2_correction[kind2int[kind]] and c_env.shape[-1] > 0:
        if e_delta_mp2 is None:
            if kind == "occupied":
                co_act = np.hstack((c_occclst, c_bath))
                #*_, e_mp2_act = self.run_mp2(co_act, c_vir, c_occenv=c_env, c_virenv=c_virenv)
                *_, e_mp2_act = self.run_mp2(co_act, c_vir, c_occenv=c_env, c_virenv=c_virenv, eris=eris)
            elif kind == "virtual":
                cv_act = np.hstack((c_virclst, c_bath))
                #*_, e_mp2_act = self.run_mp2(c_occ, cv_act, c_occenv=c_occenv, c_virenv=c_env)
                *_, e_mp2_act = self.run_mp2(c_occ, cv_act, c_occenv=c_occenv, c_virenv=c_env, eris=eris)

            e_delta_mp2 = e_mp2_all - e_mp2_act
            log.debug("MP2 correction (%s): all=%.4g, active=%.4g, correction=%+.4g", kind, e_mp2_all, e_mp2_act, e_delta_mp2)
    else:
        e_delta_mp2 = 0.0

    return c_bath, c_env, e_delta_mp2, eigref_out

# ====

def make_local_bath(self, c_env, ao_indices=None, nbath=None, tol=1e-9):

    if ao_indices is None:
        ao_indices = self.ao_indices
    assert ao_indices is not None
    p = self.base.make_ao_projector(ao_indices)
    s = np.linalg.multi_dot((c_env.T, p, c_env))
    e, v = np.linalg.eigh(s)
    e, v = e[::-1], v[:,::-1]
    log.debug("Eigenvalues of local orbitals:\n%r", e)
    assert np.all(e > -1e-8), ("Negative eigenvalues found: %r" % e[e <= -1e-8])
    if nbath is not None:
        nbath = min(nbath, len(e))
    elif tol is not None:
        nbath = sum(e >= tol)

    c_env = np.dot(c_env, v)
    c_bath, c_env = np.hsplit(c_env, [nbath])

    if nbath > 0:
        log.debug("%3d bath orbitals: largest=%6.3g, smallest=%6.3g.",
                nbath, max(e[:nbath]), min(e[:nbath]))
    else:
        log.debug("No bath orbitals.")
    if nbath < len(e):
        log.debug("%3d environment orbitals: largest=%6.3g, smallest=%6.3g.",
                (len(e)-nbath), max(e[nbath:]), min(e[nbath:]))
    else:
        log.debug("No environment orbitals.")

    return c_bath, c_env
