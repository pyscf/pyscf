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
        "get_local_energy",
        ]

log = logging.getLogger(__name__)

def get_local_energy(self, cc, C1, C2, eris, project="occupied", project_kind="right"):

    act = cc.get_frozen_mask()
    occ = cc.mo_occ[act] > 0
    vir = cc.mo_occ[act] == 0
    # Projector to local, occupied region
    S = self.mf.get_ovlp()
    C = cc.mo_coeff[:,act]

    # Project one index of amplitudes
    if project == "occupied":
        P = self.get_local_projector(C[:,occ], kind=project_kind)
        if C1 is not None:
            pC1 = einsum("xi,ia->xa", P, C1)
        pC2 = einsum("xi,ijab->xjab", P, C2)

        # Test new method
        p2C1, p2C2 = self.project_amplitudes(C[:,occ], C1, C2, kind=project_kind)
        assert np.allclose(pC1, p2C1)
        assert np.allclose(pC2, p2C2)

    elif project == "virtual":
        P = self.get_local_projector(C[:,vir], kind=project_kind)
        if C1 is not None:
            pC1 = einsum("xa,ia->ia", P, C1)
        pC2 = einsum("xa,ijab->ijxb", P, C2)

    # MP2
    if C1 is None:
        e1 = 0
    # CC
    else:
        F = eris.fock[occ][:,vir]
        e1 = 2*np.sum(F * pC1)

    # CC
    if hasattr(eris, "ovvo"):
        eris_ovvo = eris.ovvo
    # MP2
    else:
        no, nv = pC2.shape[1:3]
        eris_ovvo = eris.ovov.reshape(no,nv,no,nv).transpose(0, 1, 3, 2)
    e2 = 2*einsum('ijab,iabj', pC2, eris_ovvo)
    e2 -=  einsum('ijab,jabi', pC2, eris_ovvo)

    log.info("Energy components E1=%.8g, E2=%.8g", e1, e2)
    e_loc = self.symmetry_factor * (e1 + e2)

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

def get_local_energy_most_indices_2C(self, cc, C1, C2, eris=None, symmetry_factor=None):

    if symmetry_factor is None:
        symmetry_factor = self.symmetry_factor

    a = cc.get_frozen_mask()
    # Projector to local, occupied region
    S = self.mf.get_ovlp()
    C = cc.mo_coeff[:,a]
    CTS = np.dot(C.T, S)

    # Project one index of T amplitudes
    l= self.indices
    r = self.not_indices
    o = cc.mo_occ[a] > 0
    v = cc.mo_occ[a] == 0

    if eris is None:
        log.warning("Warning: recomputing AO->MO integral transformation")
        eris = cc.ao2mo()

    def get_projectors(aos):
        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
        return Po, Pv

    Lo, Lv = get_projectors(l)
    Ro, Rv = get_projectors(r)

    # Nomenclature:
    # old occupied: i,j
    # old virtual: a,b
    # new occupied: p,q
    # new virtual: s,t
    T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
    T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
    T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
    T1 = T1_ll + (T1_lr + T1_rl)/2

    F = eris.fock[o][:,v]
    e1 = 2*np.sum(F * T1)
    if not np.isclose(e1, 0):
        log.warning("Warning: large E1 component: %.8e" % e1)

    #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
    def project_T2(P1, P2, P3, P4):
        T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
        return T2p

    f3 = 1.0
    f2 = 0.5
    # 4
    T2 = 1*project_T2(Lo, Lo, Lv, Lv)
    # 3
    T2 += f3*(2*project_T2(Lo, Lo, Lv, Rv)      # factor 2 for LLRL
            + 2*project_T2(Ro, Lo, Lv, Lv))     # factor 2 for RLLL
    ## 2
    #T2 += f2*(  project_T2(Lo, Lo, Rv, Rv)
    #        + 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
    #        + 2*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
    #        +   project_T2(Ro, Ro, Lv, Lv))

    # 2
    T2 +=   project_T2(Lo, Lo, Rv, Rv)
    T2 += 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
    #T2 += 1*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
    #T2 +=   project_T2(Ro, Ro, Lv, Lv)

    e2 = (2*einsum('ijab,iabj', T2, eris.ovvo)
           -einsum('ijab,jabi', T2, eris.ovvo))

    e_loc = symmetry_factor * (e1 + e2)

    return e_loc

def get_local_energy_most_indices(self, cc, C1, C2, variant=1):

    a = cc.get_frozen_mask()
    # Projector to local, occupied region
    S = self.mf.get_ovlp()
    C = cc.mo_coeff[:,a]
    CTS = np.dot(C.T, S)

    # Project one index of T amplitudes
    l= self.indices
    r = self.not_indices
    o = cc.mo_occ[a] > 0
    v = cc.mo_occ[a] == 0

    eris = cc.ao2mo()

    def get_projectors(aos):
        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
        return Po, Pv

    Lo, Lv = get_projectors(l)
    Ro, Rv = get_projectors(r)

    # ONE-ELECTRON
    # ============
    pC1 = einsum("pi,ia,sa->ps", Lo, C1, Lv)
    pC1 += 0.5*einsum("pi,ia,sa->ps", Lo, C1, Rv)
    pC1 += 0.5*einsum("pi,ia,sa->ps", Ro, C1, Lv)

    F = eris.fock[o][:,v]
    e1 = 2*np.sum(F * pC1)
    if not np.isclose(e1, 0):
        log.warning("Warning: large E1 component: %.8e" % e1)

    # TWO-ELECTRON
    # ============

    def project_C2_P1(P1):
        pC2 = einsum("pi,ijab->pjab", P1, C2)
        return pC2

    def project_C2(P1, P2, P3, P4):
        pC2 = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
        return pC2

    if variant == 1:

        # QUADRUPLE L
        # ===========
        pC2 = project_C2(Lo, Lo, Lv, Lv)

        # TRIPEL L
        # ========
        pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
        pC2 += 2*project_C2(Lo, Ro, Lv, Lv)

        # DOUBLE L
        # ========
        # P(LLRR) [This wrongly includes: P(LLAA) - correction below]
        pC2 +=   project_C2(Lo, Lo, Rv, Rv)
        pC2 += 2*project_C2(Lo, Ro, Lv, Rv)
        pC2 += 2*project_C2(Lo, Ro, Rv, Lv)
        pC2 +=   project_C2(Ro, Ro, Lv, Lv)

        # SINGLE L
        # ========
        # P(LRRR) [This wrongly includes: P(LAAR) - correction below]
        four_idx_from_occ = False

        if not four_idx_from_occ:
            pC2 += 0.25*2*project_C2(Lo, Ro, Rv, Rv)
            pC2 += 0.25*2*project_C2(Ro, Ro, Lv, Rv)
        else:
            pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)

        # CORRECTIONS
        # ===========
        for x in self.loop_clusters(exclude_self=True):
            Xo, Xv = get_projectors(x.indices)

            # DOUBLE CORRECTION
            # -----------------
            # Correct for wrong inclusion of P(LLAA)
            # The case P(LLAA) was included with prefactor of 1 instead of 1/2
            # We thus need to only correct by "-1/2"
            pC2 -= 0.5*  project_C2(Lo, Lo, Xv, Xv)
            pC2 -= 0.5*2*project_C2(Lo, Xo, Lv, Xv)
            pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Lv)
            pC2 -= 0.5*  project_C2(Xo, Xo, Lv, Lv)

            # SINGLE CORRECTION
            # -----------------
            # Correct for wrong inclusion of P(LAAR)
            # This corrects the case P(LAAB) but overcorrects P(LAAA)!
            if not four_idx_from_occ:
                pC2 -= 0.25*2*project_C2(Lo, Xo, Xv, Rv)
                pC2 -= 0.25*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
                pC2 -= 0.25*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection
                pC2 -= 0.25*2*project_C2(Xo, Xo, Lv, Rv)
                pC2 -= 0.25*2*project_C2(Xo, Ro, Lv, Xv) # overcorrection
                pC2 -= 0.25*2*project_C2(Ro, Xo, Lv, Xv) # overcorrection

                # Correct overcorrection
                pC2 += 0.25*2*2*project_C2(Lo, Xo, Xv, Xv)
                pC2 += 0.25*2*2*project_C2(Xo, Xo, Lv, Xv)

            else:
                pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)
                pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
                pC2 -= 0.5*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection

                # Correct overcorrection
                pC2 += 0.5*2*2*project_C2(Lo, Xo, Xv, Xv)

        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
               -einsum('ijab,jabi', pC2, eris.ovvo))

    elif variant == 2:
        # QUADRUPLE L
        # ===========
        pC2 = project_C2(Lo, Lo, Lv, Lv)

        # TRIPEL L
        # ========
        pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
        pC2 += 2*project_C2(Lo, Ro, Lv, Lv)

        # DOUBLE L
        # ========
        pC2 +=   project_C2(Lo, Lo, Rv, Rv)
        pC2 +=   2*project_C2(Lo, Ro, Lv, Rv)
        pC2 +=   2*project_C2(Lo, Ro, Rv, Lv)
        for x in self.loop_clusters(exclude_self=True):
            Xo, Xv = get_projectors(x.indices)
            pC2 -= project_C2(Lo, Xo, Lv, Xv)
            pC2 -= project_C2(Lo, Xo, Xv, Lv)

        # SINGLE L
        # ========

        # This wrongly includes LXXX
        pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)
        for x in self.loop_clusters(exclude_self=True):
            Xo, Xv = get_projectors(x.indices)

            pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv)
            pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)

            pC2 += 0.5*2*project_C2(Lo, Xo, Xv, Xv)

        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
               -einsum('ijab,jabi', pC2, eris.ovvo))

    elif variant == 3:
        # QUADRUPLE + TRIPLE L
        # ====================
        pC2 = project_C2_P1(Lo)
        pC2 += project_C2(Ro, Lo, Lv, Lv)
        for x in self.loop_clusters(exclude_self=True):
            Xo, Xv = get_projectors(x.indices)
            pC2 -= project_C2(Lo, Xo, Xv, Xv)

        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
               -einsum('ijab,jabi', pC2, eris.ovvo))


    e_loc = e1 + e2

    return e_loc


