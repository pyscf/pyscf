import logging

import numpy as np

from .util import einsum

log = logging.getLogger(__name__)

def project_eris(eris, c_occ, c_vir, ovlp, check_subspace=True):
    """Project eris to a new set of orbital coefficients.

    This routine is mainly useful if the transformation is either unitary,
    or the new MO coefficients span a subspace of the space spanned by the original MOs.

    Parameters
    ----------
    eris : _ChemistERIs
        PySCF ERIs object
    c_occ : (nao, nocc) array
        Occupied MO coefficients.
    c_vir : (nao, nvir) array
        Virtual MO coefficients.
    ovlp : (nao, nao) array
        AO overlap matrix
    check_subspace : bool, optional
        Check if c_occ and c_vir span a subspace of eris.mo_coeff.
        Return None if Not. Default: True.

    Returns
    -------
    eris : _ChemistERIs or None
        ERIs with transformed integral values, as well as transformed attributes
        `mo_coeff`, `fock`, and `mo_energy`.
    """
    c_occ0, c_vir0 = np.hsplit(eris.mo_coeff, [eris.nocc])
    nocc0, nvir0 = c_occ0.shape[-1], c_vir0.shape[-1]
    nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
    log.debug("Projecting ERIs: N(occ)= %3d -> %3d  N(vir)= %3d -> %3d", nocc0, nocc, nvir0, nvir)

    transform_occ = (nocc != nocc0 or not np.allclose(c_occ, c_occ0))
    if transform_occ:
        r_occ = np.linalg.multi_dot((c_occ.T, ovlp, c_occ0))
    else:
        r_occ = np.eye(nocc)
    transform_vir = (nvir != nvir0 or not np.allclose(c_vir, c_vir0))
    if transform_vir:
        r_vir = np.linalg.multi_dot((c_vir.T, ovlp, c_vir0))
    else:
        r_vir = np.eye(nvir)

    # Do nothing
    if not (transform_occ or transform_vir):
        return eris

    # Check that c_occ and c_vir form a subspace of eris.mo_coeff
    # If not return None
    if check_subspace:
        if nocc0 < nocc:
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
        if nvir0 < nvir:
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
        p_occ = np.dot(r_occ.T, r_occ)
        e, v = np.linalg.eigh(p_occ)
        n = np.count_nonzero(abs(e)>1e-8)
        if n < nocc:
            log.debug("e(occ)= %d\n%r", n, e)
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
        p_vir = np.dot(r_vir.T, r_vir)
        e, v = np.linalg.eigh(p_vir)
        n = np.count_nonzero(abs(e)>1e-8)
        if n < nvir:
            log.debug("e(vir)= %d\n%r", n, e)
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")

    r_all = np.block([
        [r_occ, np.zeros((nocc, nvir0))],
        [np.zeros((nvir, nocc0)), r_vir]])

    transform = lambda g, t0, t1, t2, t3 : einsum("abcd,ia,jb,kc,ld -> ijkl", g, t0, t1, t2, t3)

    for kind in ["oooo", "ovoo", "oovv", "ovov", "ovvo", "ovvv", "vvvv"]:
        g = getattr(eris, kind, None)
        if g is None:
            #log.debug("Not transforming (%2s|%2s)", kind[:2], kind[2:])
            continue
        #log.debug("Transforming (%2s|%2s)", kind[:2], kind[2:])

        shape0 = [(nocc0 if (pos == "o") else nvir0) for pos in kind]
        #log.debug("Shape0= %r", shape0)
        t0123 = [(r_occ if (pos == "o") else r_vir) for pos in kind]
        g = transform(g[:].reshape(shape0), *t0123)
        #log.debug("Shape out= %r", g.shape)
        #shape = [(nocc if (pos == "o") else nvir) for pos in kind]
        #shape = (shape[0]*shape[1], shape[2]*shape[3])
        #g = g.reshape(shape)

        setattr(eris, kind, g)

    eris.mo_coeff = np.hstack((c_occ, c_vir))
    eris.nocc = nocc
    eris.fock = np.linalg.multi_dot((r_all, eris.fock, r_all.T))
    eris.mo_energy = np.diag(eris.fock)
    return eris


# TODO
#def project_amplitudes(t1, t2, c_occ, c_vir, ovlp, check_subspace=True):
#    """Project eris to a new set of orbital coefficients.
#
#    This routine is mainly useful if the transformation is either unitary,
#    or the new MO coefficients span a subspace of the space spanned by the original MOs.
#
#    Parameters
#    ----------
#    t1 : (nocc, nvir)
#        T1 amplitudes.
#    t2 : (nocc, nocc, nvir, nvir)
#        T2 amplitudes.
#    c_occ : (nao, nocc) array
#        Occupied MO coefficients.
#    c_vir : (nao, nvir) array
#        Virtual MO coefficients.
#    ovlp : (nao, nao) array
#        AO overlap matrix
#    check_subspace : bool, optional
#        Check if c_occ and c_vir span a subspace of eris.mo_coeff.
#        Return None if Not. Default: True.
#
#    Returns
#    -------
#    eris : _ChemistERIs or None
#        ERIs with transformed integral values, as well as transformed attributes
#        `mo_coeff`, `fock`, and `mo_energy`.
#    """
#    c_occ0, c_vir0 = np.hsplit(eris.mo_coeff, [eris.nocc])
#    nocc0, nvir0 = c_occ0.shape[-1], c_vir0.shape[-1]
#    nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
#    log.debug("Projecting Amplitudes: N(occ)= %3d -> %3d  N(vir)= %3d -> %3d", nocc0, nocc, nvir0, nvir)
#
#    transform_occ = (nocc != nocc0 or not np.allclose(c_occ, c_occ0))
#    if transform_occ:
#        r_occ = np.linalg.multi_dot((c_occ.T, ovlp, c_occ0))
#    else:
#        r_occ = np.eye(nocc)
#    transform_vir = (nvir != nvir0 or not np.allclose(c_vir, c_vir0))
#    if transform_vir:
#        r_vir = np.linalg.multi_dot((c_vir.T, ovlp, c_vir0))
#    else:
#        r_vir = np.eye(nvir)
#
#    # Check that c_occ and c_vir form a subspace of eris.mo_coeff
#    # If not return None
#    if check_subspace:
#        if nocc0 < nocc:
#            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
#        if nvir0 < nvir:
#            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
#        p_occ = np.dot(r_occ.T, r_occ)
#        e, v = np.linalg.eigh(p_occ)
#        n = np.count_nonzero(abs(e)>1e-8)
#        if n < nocc:
#            log.debug("e(occ)= %d\n%r", n, e)
#            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
#        p_vir = np.dot(r_vir.T, r_vir)
#        e, v = np.linalg.eigh(p_vir)
#        n = np.count_nonzero(abs(e)>1e-8)
#        if n < nvir:
#            log.debug("e(vir)= %d\n%r", n, e)
#            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
#
#    r_all = np.block([
#        [r_occ, np.zeros((nocc, nvir0))],
#        [np.zeros((nvir, nocc0)), r_vir]])
#
#    transform = lambda g, t0, t1, t2, t3 : einsum("abcd,ia,jb,kc,ld -> ijkl", g, t0, t1, t2, t3)
#
#    for kind in ["oooo", "ovoo", "oovv", "ovov", "ovvo", "ovvv", "vvvv"]:
#        g = getattr(eris, kind, None)
#        if g is None:
#            #log.debug("Not transforming (%2s|%2s)", kind[:2], kind[2:])
#            continue
#        #log.debug("Transforming (%2s|%2s)", kind[:2], kind[2:])
#
#        shape0 = [(nocc0 if (pos == "o") else nvir0) for pos in kind]
#        #log.debug("Shape0= %r", shape0)
#        t0123 = [(r_occ if (pos == "o") else r_vir) for pos in kind]
#        g = transform(g[:].reshape(shape0), *t0123)
#        #log.debug("Shape out= %r", g.shape)
#        #shape = [(nocc if (pos == "o") else nvir) for pos in kind]
#        #shape = (shape[0]*shape[1], shape[2]*shape[3])
#        #g = g.reshape(shape)
#
#        setattr(eris, kind, g)
#
#    eris.mo_coeff = np.hstack((c_occ, c_vir))
#    eris.nocc = nocc
#    eris.fock = np.linalg.multi_dot((r_all, eris.fock, r_all.T))
#    eris.mo_energy = np.diag(eris.fock)
#    return eris




def transform_mp2_eris(eris, c_occ, c_vir, ovlp):
    """Transform eris of kind (ov|ov) (occupied-virtual-occupied-virtual)

    OBSOLETE: replaced by transform_eris
    """
    assert (eris is not None)
    assert (eris.ovov is not None)

    c_occ0, c_vir0 = np.hsplit(eris.mo_coeff, [eris.nocc])
    nocc0, nvir0 = c_occ0.shape[-1], c_vir0.shape[-1]
    nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]

    transform_occ = (nocc != nocc0 or not np.allclose(c_occ, c_occ0))
    if transform_occ:
        r_occ = np.linalg.multi_dot((c_occ.T, ovlp, c_occ0))
    else:
        r_occ = np.eye(nocc)
    transform_vir = (nvir != nvir0 or not np.allclose(c_vir, c_vir0))
    if transform_vir:
        r_vir = np.linalg.multi_dot((c_vir.T, ovlp, c_vir0))
    else:
        r_vir = np.eye(nvir)
    r_all = np.block([
        [r_occ, np.zeros((nocc, nvir0))],
        [np.zeros((nvir, nocc0)), r_vir]])

    # eris.ovov may be hfd5 dataset on disk -> allocate in memory with [:]
    govov = eris.ovov[:].reshape(nocc0, nvir0, nocc0, nvir0)
    if transform_occ and transform_vir:
        govov = einsum("iajb,xi,ya,zj,wb->xyzw", govov, r_occ, r_vir, r_occ, r_vir)
    elif transform_occ:
        govov = einsum("iajb,xi,zj->xazb", govov, r_occ, r_occ)
    elif transform_vir:
        govov = einsum("iajb,ya,wb->iyjw", govov, r_vir, r_vir)
    eris.ovov = govov.reshape((nocc*nvir, nocc*nvir))
    eris.mo_coeff = np.hstack((c_occ, c_vir))
    eris.fock = np.linalg.multi_dot((r_all, eris.fock, r_all.T))
    eris.mo_energy = np.diag(eris.fock)
    return eris
