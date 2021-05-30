"""Local energy for EmbCC calculations.

These require a projection of the some indices of the C1 and C2
amplitudes.
"""
import logging

import numpy as np
import scipy
import scipy.linalg

from .util import *

__all__ = [
        "get_local_amplitudes",
        "get_local_amplitudes_general",
        "get_local_energy",
        ]

log = logging.getLogger(__name__)


def get_local_amplitudes(self, cc, C1, C2, **kwargs):
    """Wrapper for get_local_amplitudes, where the mo coefficients are extracted from a MP2 or CC object."""

    act = cc.get_frozen_mask()
    occ = cc.mo_occ[act] > 0
    vir = cc.mo_occ[act] == 0
    c = cc.mo_coeff[:,act]
    c_occ = c[:,occ]
    c_vir = c[:,vir]

    return get_local_amplitudes_general(self, C1, C2, c_occ, c_vir, **kwargs)


def get_local_amplitudes_general(self, C1, C2, c_occ, c_vir, part=None, symmetrize=False, inverse=False):
    """Get local contribution of amplitudes."""

    # By default inherit from base object
    if part is None:
        part = self.base.opts.energy_partition
    #log.debug("Amplitude partitioning = %s", part)
    if part not in ("first-occ", "first-vir", "democratic"):
        raise ValueError("Unknown partitioning of amplitudes: %s", part)

    # Projectors into local occupied and virtual space
    if part in ("first-occ", "democratic"):
        Lo = self.get_local_projector(c_occ)
    if part in ("first-vir", "democratic"):
        Lv = self.get_local_projector(c_vir)
    # Projectors into non-local occupied and virtual space
    if part == "democratic":
        Ro = self.get_local_projector(c_occ, inverse=True)
        Rv = self.get_local_projector(c_vir, inverse=True)

    if C1 is not None:
        if part == "first-occ":
            pC1 = einsum("xi,ia->xa", Lo, C1)
        elif part == "first-vir":
            pC1 = einsum("ia,xa->ix", C1, Lv)
        elif part == "democratic":
            pC1 = einsum("xi,ia,ya->xy", Lo, C1, Lv)
            pC1 += einsum("xi,ia,ya->xy", Lo, C1, Rv) / 2.0
            pC1 += einsum("xi,ia,ya->xy", Ro, C1, Lv) / 2.0
    else:
        pC1 = None

    if part == "first-occ":
        pC2 = einsum("xi,ijab->xjab", Lo, C2)
    elif part == "first-vir":
        pC2 = einsum("ijab,xa->ijxb", C2, Lv)
    elif part == "democratic":

        def project_C2(P1, P2, P3, P4):
            pC2 = einsum("xi,yj,ijab,za,wb->xyzw", P1, P2, C2, P3, P4)
            return pC2

        # Factors of 2 due to ij,ab <-> ji,ba symmetry
        # Denominators 1/N due to element being shared between N clusters

        # Quadruple L
        # ===========
        pC2 = project_C2(Lo, Lo, Lv, Lv)

        # Triple L
        # ========
        # P(LLLR)
        pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
        pC2 += 2*project_C2(Lo, Ro, Lv, Lv)

        # Double L
        # ========
        # P(LLRR) [This wrongly includes: 1x P(LLAA), instead of 0.5x - correction below]
        pC2 +=   project_C2(Lo, Lo, Rv, Rv)
        pC2 += 2*project_C2(Lo, Ro, Lv, Rv)
        pC2 += 2*project_C2(Lo, Ro, Rv, Lv)
        pC2 +=   project_C2(Ro, Ro, Lv, Lv)

        # Single L
        # ========
        # P(LRRR) [This wrongly includes: P(LAAR) (where R could be A) - correction below]
        pC2 += 2*project_C2(Lo, Ro, Rv, Rv) / 4.0
        pC2 += 2*project_C2(Ro, Ro, Lv, Rv) / 4.0

        # Corrections
        # ===========
        # Loop over all other clusters x
        for x in self.loop_clusters(exclude_self=True):

            Xo = x.get_local_projector(Co)
            Xv = x.get_local_projector(Cv)

            # Double correction
            # -----------------
            # Correct for wrong inclusion of P(LLAA)
            # The case P(LLAA) was included with prefactor of 1 instead of 1/2
            # We thus need to only correct by "-1/2"
            pC2 -=   project_C2(Lo, Lo, Xv, Xv) / 2.0
            pC2 -= 2*project_C2(Lo, Xo, Lv, Xv) / 2.0
            pC2 -= 2*project_C2(Lo, Xo, Xv, Lv) / 2.0
            pC2 -=   project_C2(Xo, Xo, Lv, Lv) / 2.0

            # Single correction
            # -----------------
            # Correct for wrong inclusion of P(LAAR)
            # This corrects the case P(LAAB) but overcorrects P(LAAA)!
            pC2 -= 2*project_C2(Lo, Xo, Xv, Rv) / 4.0
            pC2 -= 2*project_C2(Lo, Xo, Rv, Xv) / 4.0 # If R == X this is the same as above -> overcorrection
            pC2 -= 2*project_C2(Lo, Ro, Xv, Xv) / 4.0 # overcorrection
            pC2 -= 2*project_C2(Xo, Xo, Lv, Rv) / 4.0
            pC2 -= 2*project_C2(Xo, Ro, Lv, Xv) / 4.0 # overcorrection
            pC2 -= 2*project_C2(Ro, Xo, Lv, Xv) / 4.0 # overcorrection

            # Correct overcorrection
            # The additional factor of 2 comes from how often the term was wrongly included above
            pC2 += 2*2*project_C2(Lo, Xo, Xv, Xv) / 4.0
            pC2 += 2*2*project_C2(Xo, Xo, Lv, Xv) / 4.0

    # Note that the energy should be invariant to symmetrization
    if symmetrize:
        pC2 = (pC2 + pC2.transpose(1,0,3,2)) / 2

    if inverse:
        if pC1 is not None:
            pC1 = C1 - pC1
        pC2 = C2 - pC2

    return pC1, pC2


def get_local_energy(self, cc, pC1, pC2, eris):
    """
    Parameters
    ----------
    cc : pyscf[.pbc].cc.CCSD or pyscf[.pbc].mp.MP2
        PySCF coupled cluster or MP2 object. This function accesses:
            cc.get_frozen_mask()
            cc.mo_occ
    pC1 : ndarray
        Locally projected C1 amplitudes.
    pC2 : ndarray
        Locally projected C2 amplitudes.
    eris :
        PySCF eris object as returned by cc.ao2mo()

    Returns
    -------
    e_loc : float
        Local energy contribution.
    """

    # MP2
    if pC1 is None:
        e1 = 0
    # CC
    else:
        act = cc.get_frozen_mask()
        occ = cc.mo_occ[act] > 0
        vir = cc.mo_occ[act] == 0
        F = eris.fock[occ][:,vir]
        e1 = 2*np.sum(F * pC1)

    if hasattr(eris, "ovvo"):
        eris_ovvo = eris.ovvo
    # MP2 only has eris.ovov - are these the same integrals?
    else:
        no, nv = pC2.shape[1:3]
        eris_ovvo = eris.ovov[:].reshape(no,nv,no,nv).transpose(0, 1, 3, 2).conj()
    e2 = 2*einsum('ijab,iabj', pC2, eris_ovvo)
    e2 -=  einsum('ijab,jabi', pC2, eris_ovvo)

    log.info("Energy components: E1=%16.8g, E2=%16.8g", e1, e2)
    if e1 > 1e-4 and 10*e1 > e2:
        log.warning("WARNING: Large E1 component!")

    # Symmetry factor if fragment is repeated in molecule, (e.g. in hydrogen rings: only calculate one fragment)
    e_loc = self.sym_factor * (e1 + e2)

    return e_loc


#def get_local_energy_parts(self, cc, C1, C2):

#    a = cc.get_frozen_mask()
#    # Projector to local, occupied region
#    S = self.mf.get_ovlp()
#    C = cc.mo_coeff[:,a]
#    CTS = np.dot(C.T, S)

#    # Project one index of T amplitudes
#    l= self.indices
#    r = self.not_indices
#    o = cc.mo_occ[a] > 0
#    v = cc.mo_occ[a] == 0

#    eris = cc.ao2mo()

#    def get_projectors(aos):
#        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
#        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
#        return Po, Pv

#    Lo, Lv = get_projectors(l)
#    Ro, Rv = get_projectors(r)

#    # Nomenclature:
#    # old occupied: i,j
#    # old virtual: a,b
#    # new occupied: p,q
#    # new virtual: s,t
#    T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
#    T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
#    T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
#    T1 = T1_ll + (T1_lr + T1_rl)/2

#    F = eris.fock[o][:,v]
#    e1 = 2*np.sum(F * T1)
#    if not np.isclose(e1, 0):
#        log.warning("Warning: large E1 component: %.8e" % e1)

#    #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
#    def project_T2(P1, P2, P3, P4):
#        T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
#        return T2p


#    def epart(P1, P2, P3, P4):
#        T2_part = project_T2(P1, P2, P3, P4)
#        e_part = (2*einsum('ijab,iabj', T2_part, eris.ovvo)
#              - einsum('ijab,jabi', T2_part, eris.ovvo))
#        return e_part

#    energies = []
#    # 4
#    energies.append(epart(Lo, Lo, Lv, Lv))
#    # 3
#    energies.append(2*epart(Lo, Lo, Lv, Rv))
#    energies.append(2*epart(Lo, Ro, Lv, Lv))
#    assert np.isclose(epart(Lo, Lo, Rv, Lv), epart(Lo, Lo, Lv, Rv))
#    assert np.isclose(epart(Ro, Lo, Lv, Lv), epart(Lo, Ro, Lv, Lv))

#    energies.append(  epart(Lo, Lo, Rv, Rv))
#    energies.append(2*epart(Lo, Ro, Lv, Rv))
#    energies.append(2*epart(Lo, Ro, Rv, Lv))
#    energies.append(  epart(Ro, Ro, Lv, Lv))

#    energies.append(2*epart(Lo, Ro, Rv, Rv))
#    energies.append(2*epart(Ro, Ro, Lv, Rv))
#    assert np.isclose(epart(Ro, Lo, Rv, Rv), epart(Lo, Ro, Rv, Rv))
#    assert np.isclose(epart(Ro, Ro, Rv, Lv), epart(Ro, Ro, Lv, Rv))

#    energies.append(  epart(Ro, Ro, Rv, Rv))

#    #e4 = e_aaaa
#    #e3 = e_aaab + e_aaba + e_abaa + e_baaa
#    #e2 = 0.5*(e_aabb + e_abab + e_abba + e_bbaa)

#    with open("energy-parts.txt", "a") as f:
#        f.write((10*"  %16.8e" + "\n") % tuple(energies))

##def get_local_energy_most_indices_2C(self, cc, C1, C2, eris=None, symmetry_factor=None):
##
##    if symmetry_factor is None:
##        symmetry_factor = self.symmetry_factor
##
##    a = cc.get_frozen_mask()
##    # Projector to local, occupied region
##    S = self.mf.get_ovlp()
##    C = cc.mo_coeff[:,a]
##    CTS = np.dot(C.T, S)
##
##    # Project one index of T amplitudes
##    l= self.indices
##    r = self.not_indices
##    o = cc.mo_occ[a] > 0
##    v = cc.mo_occ[a] == 0
##
##    if eris is None:
##        log.warning("Warning: recomputing AO->MO integral transformation")
##        eris = cc.ao2mo()
##
##    def get_projectors(aos):
##        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
##        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
##        return Po, Pv
##
##    Lo, Lv = get_projectors(l)
##    Ro, Rv = get_projectors(r)
##
##    # Nomenclature:
##    # old occupied: i,j
##    # old virtual: a,b
##    # new occupied: p,q
##    # new virtual: s,t
##    T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
##    T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
##    T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
##    T1 = T1_ll + (T1_lr + T1_rl)/2
##
##    F = eris.fock[o][:,v]
##    e1 = 2*np.sum(F * T1)
##    if not np.isclose(e1, 0):
##        log.warning("Warning: large E1 component: %.8e" % e1)
##
##    #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
##    def project_T2(P1, P2, P3, P4):
##        T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
##        return T2p
##
##    f3 = 1.0
##    f2 = 0.5
##    # 4
##    T2 = 1*project_T2(Lo, Lo, Lv, Lv)
##    # 3
##    T2 += f3*(2*project_T2(Lo, Lo, Lv, Rv)      # factor 2 for LLRL
##            + 2*project_T2(Ro, Lo, Lv, Lv))     # factor 2 for RLLL
##    ## 2
##    #T2 += f2*(  project_T2(Lo, Lo, Rv, Rv)
##    #        + 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
##    #        + 2*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
##    #        +   project_T2(Ro, Ro, Lv, Lv))
##
##    # 2
##    T2 +=   project_T2(Lo, Lo, Rv, Rv)
##    T2 += 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
##    #T2 += 1*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
##    #T2 +=   project_T2(Ro, Ro, Lv, Lv)
##
##    e2 = (2*einsum('ijab,iabj', T2, eris.ovvo)
##           -einsum('ijab,jabi', T2, eris.ovvo))
##
##    e_loc = symmetry_factor * (e1 + e2)
##
##    return e_loc
##
##def get_local_energy_most_indices(self, cc, C1, C2, variant=1):
##
##    a = cc.get_frozen_mask()
##    # Projector to local, occupied region
##    S = self.mf.get_ovlp()
##    C = cc.mo_coeff[:,a]
##    CTS = np.dot(C.T, S)
##
##    # Project one index of T amplitudes
##    l= self.indices
##    r = self.not_indices
##    o = cc.mo_occ[a] > 0
##    v = cc.mo_occ[a] == 0
##
##    eris = cc.ao2mo()
##
##    def get_projectors(aos):
##        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
##        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
##        return Po, Pv
##
##    Lo, Lv = get_projectors(l)
##    Ro, Rv = get_projectors(r)
##
##    # ONE-ELECTRON
##    # ============
##    pC1 = einsum("pi,ia,sa->ps", Lo, C1, Lv)
##    pC1 += 0.5*einsum("pi,ia,sa->ps", Lo, C1, Rv)
##    pC1 += 0.5*einsum("pi,ia,sa->ps", Ro, C1, Lv)
##
##    F = eris.fock[o][:,v]
##    e1 = 2*np.sum(F * pC1)
##    if not np.isclose(e1, 0):
##        log.warning("Warning: large E1 component: %.8e" % e1)
##
##    # TWO-ELECTRON
##    # ============
##
##    def project_C2_P1(P1):
##        pC2 = einsum("pi,ijab->pjab", P1, C2)
##        return pC2
##
##    def project_C2(P1, P2, P3, P4):
##        pC2 = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
##        return pC2
##
##    if variant == 1:
##
##        # QUADRUPLE L
##        # ===========
##        pC2 = project_C2(Lo, Lo, Lv, Lv)
##
##        # TRIPEL L
##        # ========
##        pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Lv, Lv)
##
##        # DOUBLE L
##        # ========
##        # P(LLRR) [This wrongly includes: P(LLAA) - correction below]
##        pC2 +=   project_C2(Lo, Lo, Rv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Lv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Rv, Lv)
##        pC2 +=   project_C2(Ro, Ro, Lv, Lv)
##
##        # SINGLE L
##        # ========
##        # P(LRRR) [This wrongly includes: P(LAAR) - correction below]
##        four_idx_from_occ = False
##
##        if not four_idx_from_occ:
##            pC2 += 0.25*2*project_C2(Lo, Ro, Rv, Rv)
##            pC2 += 0.25*2*project_C2(Ro, Ro, Lv, Rv)
##        else:
##            pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)
##
##        # CORRECTIONS
##        # ===========
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##
##            # DOUBLE CORRECTION
##            # -----------------
##            # Correct for wrong inclusion of P(LLAA)
##            # The case P(LLAA) was included with prefactor of 1 instead of 1/2
##            # We thus need to only correct by "-1/2"
##            pC2 -= 0.5*  project_C2(Lo, Lo, Xv, Xv)
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Lv, Xv)
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Lv)
##            pC2 -= 0.5*  project_C2(Xo, Xo, Lv, Lv)
##
##            # SINGLE CORRECTION
##            # -----------------
##            # Correct for wrong inclusion of P(LAAR)
##            # This corrects the case P(LAAB) but overcorrects P(LAAA)!
##            if not four_idx_from_occ:
##                pC2 -= 0.25*2*project_C2(Lo, Xo, Xv, Rv)
##                pC2 -= 0.25*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
##                pC2 -= 0.25*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection
##                pC2 -= 0.25*2*project_C2(Xo, Xo, Lv, Rv)
##                pC2 -= 0.25*2*project_C2(Xo, Ro, Lv, Xv) # overcorrection
##                pC2 -= 0.25*2*project_C2(Ro, Xo, Lv, Xv) # overcorrection
##
##                # Correct overcorrection
##                pC2 += 0.25*2*2*project_C2(Lo, Xo, Xv, Xv)
##                pC2 += 0.25*2*2*project_C2(Xo, Xo, Lv, Xv)
##
##            else:
##                pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)
##                pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
##                pC2 -= 0.5*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection
##
##                # Correct overcorrection
##                pC2 += 0.5*2*2*project_C2(Lo, Xo, Xv, Xv)
##
##        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
##               -einsum('ijab,jabi', pC2, eris.ovvo))
##
##    elif variant == 2:
##        # QUADRUPLE L
##        # ===========
##        pC2 = project_C2(Lo, Lo, Lv, Lv)
##
##        # TRIPEL L
##        # ========
##        pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Lv, Lv)
##
##        # DOUBLE L
##        # ========
##        pC2 +=   project_C2(Lo, Lo, Rv, Rv)
##        pC2 +=   2*project_C2(Lo, Ro, Lv, Rv)
##        pC2 +=   2*project_C2(Lo, Ro, Rv, Lv)
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##            pC2 -= project_C2(Lo, Xo, Lv, Xv)
##            pC2 -= project_C2(Lo, Xo, Xv, Lv)
##
##        # SINGLE L
##        # ========
##
##        # This wrongly includes LXXX
##        pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv)
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)
##
##            pC2 += 0.5*2*project_C2(Lo, Xo, Xv, Xv)
##
##        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
##               -einsum('ijab,jabi', pC2, eris.ovvo))
##
##    elif variant == 3:
##        # QUADRUPLE + TRIPLE L
##        # ====================
##        pC2 = project_C2_P1(Lo)
##        pC2 += project_C2(Ro, Lo, Lv, Lv)
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##            pC2 -= project_C2(Lo, Xo, Xv, Xv)
##
##        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
##               -einsum('ijab,jabi', pC2, eris.ovvo))
##
##    e_loc = e1 + e2
##
##    return e_loc
