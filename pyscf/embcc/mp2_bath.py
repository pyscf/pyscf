"""These functions take a cluster instance as first argument ("self")."""

import logging
import numpy as np
from timeit import default_timer as timer

import pyscf
import pyscf.mp
import pyscf.pbc
import pyscf.pbc.mp

from .util import *
from .psubspace import transform_mp2_eris

log = logging.getLogger(__name__)

# ================================================================================================ #

def make_mp2_bno(self, kind, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir,
        canonicalize=True, local_dm=False, eris=None):
    """Select virtual space from MP2 natural orbitals (NOs) according to occupation number.

    Parameters
    ----------
    c_occ : ndarray
        Active occupied orbitals.
    c_vir : ndarray
        Active virtual orbitals.
    c_occ_frozen : ndarray, optional
        Frozen occupied orbitals.
    c_vir_frozen : ndarray, optional
        Frozen virtual orbitals.
    canonicalize : bool, tuple(2), optional
        Canonicalize occupied/virtual active orbitals.
    eris: TODO

    Returns
    -------
    c_no
    n_no
    """

    if kind == "occ":
        ncluster = c_cluster_occ.shape[-1]
        c_occ = np.hstack((c_cluster_occ, c_env_occ))
        c_vir = c_cluster_vir
        c_occ_frozen = None
        c_vir_frozen = c_env_vir
        c_env = c_env_occ
    elif kind == "vir":
        ncluster = c_cluster_vir.shape[-1]
        c_occ = c_cluster_occ
        c_vir = np.hstack((c_cluster_vir, c_env_vir))
        c_occ_frozen = c_env_occ
        c_vir_frozen = None
        c_env = c_env_vir

    # Canonicalization [optional]
    if canonicalize in (True, False):
        canonicalize = 2*[canonicalize]
    if canonicalize[0]:
        c_occ, r_occ = self.canonicalize(c_occ)
    if canonicalize[1]:
        c_vir, r_vir = self.canonicalize(c_vir)

    # Setup MP2 object
    nao = c_occ.shape[0]
    assert (c_vir.shape[0] == nao)
    if c_occ_frozen is None:
        c_occ_frozen = np.zeros((nao, 0))
    if c_vir_frozen is None:
        c_vir_frozen = np.zeros((nao, 0))

    c_active = np.hstack((c_occ, c_vir))
    c_all = np.hstack((c_occ_frozen, c_active, c_vir_frozen))
    nmo = c_all.shape[-1]
    nocc_frozen = c_occ_frozen.shape[-1]
    nvir_frozen = c_vir_frozen.shape[-1]
    frozen_indices = list(range(nocc_frozen)) + list(range(nmo-nvir_frozen, nmo))
    if self.use_pbc:
        cls = pyscf.pbc.mp.MP2
    else:
        cls = pyscf.mp.MP2
    mp2 = cls(self.mf, mo_coeff=c_all, frozen=frozen_indices)

    # Integral transformation
    t0 = timer()
    if eris is None:
        eris = self.base.get_eris(mp2)
    # Reuse previously obtained integral transformation into N^2 sized quantity (rather than N^4)
    else:
        log.debug("Transforming previous eris.")
        eris = transform_mp2_eris(eris, c_occ, c_vir, ovlp=self.base.get_ovlp())
    log.timing("Time for integral transformation:  %s", get_time_string(timer()-t0))
    assert (eris.ovov is not None)

    t0 = timer()
    e_mp2_full, t2 = mp2.kernel(eris=eris, hf_reference=True)
    nocc, nvir = t2.shape[0], t2.shape[2]
    assert (c_occ.shape[-1] == nocc)
    assert (c_vir.shape[-1] == nvir)
    log.timing("Time for MP2 kernel:  %s", get_time_string(timer()-t0))

    # Energies
    e_mp2_full *= self.symmetry_factor
    t2loc = self.get_local_amplitudes(mp2, None, t2, symmetrize=True)[1]
    e_mp2 = self.symmetry_factor * mp2.energy(t2loc, eris)
    log.debug("Bath E(MP2):  Cluster= %+16.8g Ha  Fragment= %+16.8g Ha", e_mp2_full, e_mp2)

    # MP2 density matrix
    #dm_occ = dm_vir = None
    if local_dm is False:
        log.debug("Constructing DM from full T2 amplitudes.")
        t2l, t2r = t2, t2
        # This is equivalent to:
        # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
        # do, dv = -2*do, 2*dv
    elif local_dm is True:
        # LOCAL DM IS NOT RECOMMENDED - USE SEMI OR FALSE
        log.warning("Using local_dm = True is not recommended - use 'semi' or False")
        log.debug("Constructing DM from local T2 amplitudes.")
        t2l, t2r = t2loc, t2loc
    elif local_dm == "semi":
        log.debug("Constructing DM from semi-local T2 amplitudes.")
        t2l, t2r = t2loc, t2
    else:
        raise ValueError("Unknown value for local_dm: %r" % local_dm)

    if kind == "occ":
        dm = 2*(2*einsum("ikab,jkab->ij", t2l, t2r)
                - einsum("ikab,kjab->ij", t2l, t2r))
        # This turns out to be equivalent:
        #dm = 2*(2*einsum("kiba,kjba->ij", t2l, t2r)
        #        - einsum("kiba,kjab->ij", t2l, t2r))
    else:
        dm = 2*(2*einsum("ijac,ijbc->ab", t2l, t2r)
                - einsum("ijac,ijcb->ab", t2l, t2r))
    if local_dm == "semi":
        dm = (dm + dm.T)/2
    assert np.allclose(dm, dm.T)

    # Undo canonicalization
    if kind == "occ" and canonicalize[0]:
        dm = np.linalg.multi_dot((r_occ, dm, r_occ.T))
    elif kind == "vir" and canonicalize[1]:
        dm = np.linalg.multi_dot((r_vir, dm, r_vir.T))

    env = np.s_[ncluster:]
    n_no, c_no = np.linalg.eigh(dm[env,env])
    n_no, c_no = n_no[::-1], c_no[:,::-1]
    c_no = np.dot(c_env, c_no)

    return c_no, n_no

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

#def make_mp2_bath(self, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir,
#        kind, mp2_correction=None):
#    """Select occupied or virtual bath space from MP2 natural orbitals.
#
#    The natural orbitals are calculated only including the local virtual (occupied)
#    cluster orbitals when calculating occupied (virtual) bath orbitals, i.e. they do not correspond
#    to the full system MP2 natural orbitals and are different for every cluster.
#
#    Parameters
#    ----------
#    c_cluster_occ : ndarray
#        Occupied cluster (fragment + DMET bath) orbitals.
#    c_cluster_vir : ndarray
#        Virtual cluster (fragment + DMET bath) orbitals.
#    C_env : ndarray
#        Environment orbitals. These need to be off purely occupied character if kind=="occ"
#        and of purely virtual character if kind=="vir".
#    kind : str ["occ", "vir"]
#        Calculate occupied or virtual bath orbitals.
#
#    Returns
#    -------
#    c_no : ndarray
#        MP2 natural orbitals.
#    n_no : ndarray
#        Natural occupation numbers.
#    e_delta_mp2 : float
#        MP2 correction energy (0 if self.mp2_correction == False)
#    """
#    if kind not in ("occ", "vir"):
#        raise ValueError("Unknown kind: %s", kind)
#    if mp2_correction is None: mp2_correction = self.mp2_correction[0 if kind == "occ" else 1]
#    kindname = {"occ": "occupied", "vir" : "virtual"}[kind]
#
#    # All occupied and virtual orbitals
#    c_all_occ = np.hstack((c_cluster_occ, c_env_occ))
#    c_all_vir = np.hstack((c_cluster_vir, c_env_vir))
#
#    # All occupied orbitals, only cluster virtual orbitals
#    if kind == "occ":
#        c_occ = np.hstack((c_cluster_occ, c_env_occ))
#        c_vir = c_cluster_vir
#        ... = self.run_mp2(c_occ=c_occ, c_vir=c_vir, c_vir_frozen=c_env_vir)
#        ncluster = c_occclst.shape[-1]
#        nenv = c_occ.shape[-1] - ncluster
#
#    # All virtual orbitals, only cluster occupied orbitals
#    elif kind == "vir":
#        c_occ = c_cluster_occ
#        c_vir = np.hstack((c_cluster_vir, c_env_vir))
#        ... = self.run_mp(c_occ, c_vir, c_occenv=c_occenv, c_virenv=None)
#        ncluster = c_virclst.shape[-1]
#        nenv = c_vir.shape[-1] - ncluster
#
#    # Diagonalize environment-environment block of MP2 DM correction
#    # and rotate into natural orbital basis, with the orbitals sorted
#    # with decreasing (absolute) occupation change
#    # [Note that dm_occ is minus the change of the occupied DM]
#
#    env = np.s_[ncluster:]
#    dm = dm[env,env]
#    dm_occ, dm_rot = np.linalg.eigh(dm)
#    assert (len(dm_occ) == nenv)
#    if np.any(dm_occ < -1e-12):
#        raise RuntimeError("Negative occupation values detected: %r" % dm_occ[dm_occ < -1e-12])
#    dm_occ, dm_rot = dm_occ[::-1], dm_rot[:,::-1]
#    c_rot = np.dot(c_env, dm_rot)
#
#    with open("mp2-bath-occupation.txt", "ab") as f:
#        #np.savetxt(f, dm_occ[np.newaxis], header="MP2 bath orbital occupation of cluster %s" % self.name)
#        np.savetxt(f, dm_occ, fmt="%.10e", header="%s MP2 bath orbital occupation of cluster %s" % (kindname.title(), self.name))
#
#    if self.opts.plot_orbitals:
#        #bins = np.hstack((-np.inf, np.logspace(-9, -3, 9-3+1), np.inf))
#        bins = np.hstack((1, np.logspace(-3, -9, 9-3+1), -1))
#        for idx, upper in enumerate(bins[:-1]):
#            lower = bins[idx+1]
#            mask = np.logical_and((dm_occ > lower), (dm_occ <= upper))
#            if np.any(mask):
#                coeff = c_rot[:,mask]
#                log.info("Plotting MP2 bath density between %.0e and %.0e containing %d orbitals." % (upper, lower, coeff.shape[-1]))
#                dm = np.dot(coeff, coeff.T)
#                dset_idx = (4001 if kind == "occ" else 5001) + idx
#                self.cubefile.add_density(dm, dset_idx=dset_idx)
#
#    return c_no, n_no
#
