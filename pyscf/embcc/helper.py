import logging

import numpy as np

log = logging.getLogger(__name__)

default_minao = {
        "gth-dzv" : "gth-szv",
        "gth-dzvp" : "gth-szv",
        "gth-tzvp" : "gth-szv",
        "gth-tzv2p" : "gth-szv",
        }

def get_minimal_basis(basis):
    minao = default_minao.get(basis, "minao")
    return minao

def indices_to_bools(indices, n):
    bools = np.zeros(n, dtype=bool)
    bools[np.asarray(indices)] = True
    return bools

def transform_amplitudes(t1, t2, u_occ, u_vir):
    if t1 is not None:
        t1 = einsum("ia,ix,ay->xy", t1, u_occ, u_vir)
    else:
        t1 = None
    if t2 is not None:
        t2 = einsum("ijab,ix,jy,az,bw->xyzw", t2, u_occ, u_occ, u_vir, u_vir)
    else:
        t2 = None
    return t1, t2

def plot_histogram(values, bins=None, maxbarlength=50):
    if bins is None:
        bins = np.hstack([np.inf, np.logspace(-3, -12, 10), -np.inf])
    bins = bins[::-1]
    hist = np.histogram(values, bins)[0]
    bins, hist = bins[::-1], hist[::-1]
    for i, hval in enumerate(hist):
        barlength = int(maxbarlength * hval/hist.max())
        if hval == 0:
            bar = ""
        else:
            barlength = max(barlength, 1)
            bar = ((barlength-1) * "|") + "]" + ("  (%d)" % hval)
        log.info("  %5.0e - %5.0e  |%s", bins[i], bins[i+1], bar)

def atom_labels_to_ao_indices(mol, atom_labels):
    """Convert atom labels to AO indices of mol object."""
    atom_labels_mol = np.asarray([ao[1] for ao in mol.ao_labels(None)])
    ao_indices = np.nonzero(np.isin(atom_labels_mol, atom_labels))[0]
    return ao_indices

def atom_label_to_ids(mol, atom_label):
    """Get all atom IDs corresponding to an atom label."""
    atom_labels = np.asarray([mol.atom_symbol(atomid) for atomid in range(mol.natm)])
    atom_ids = np.where(np.in1d(atom_labels, atom_label))[0]
    return atom_ids

def get_ao_indices_at_atoms(mol, atomids):
    """Return indices of AOs centered at a given atom ID."""
    ao_indices = []
    if not hasattr(atomids, "__len__"):
        atomids = [atomids]
    for atomid in atomids:
        ao_slice = mol.aoslice_by_atom()[atomid]
        ao_indices += list(range(ao_slice[2], ao_slice[3]))
    return ao_indices

def orthogonalize_mo(c, s, tol=1e-6):
    """Orthogonalize MOs, such that C^T S C = I (identity matrix).

    Parameters
    ----------
    c : ndarray
        MO orbital coefficients.
    s : ndarray
        AO overlap matrix.
    tol : float, optional
        Tolerance.

    Returns
    -------
    c_out : ndarray
        Orthogonalized MO coefficients.
    """
    assert np.all(c.imag == 0)
    assert np.allclose(s, s.T)
    l = np.linalg.cholesky(s)
    c2 = np.dot(l.T, c)
    #chi = np.linalg.multi_dot((c.T, s, c))
    chi = np.dot(c2.T, c2)
    chi = (chi + chi.T)/2
    e, v = np.linalg.eigh(chi)
    assert np.all(e > 0)
    r = einsum("ai,i,bi->ab", v, 1/np.sqrt(e), v)
    c_out = np.dot(c, r)
    chi_out = np.linalg.multi_dot((c_out.T, s, c_out))
    # Check orthogonality within tol
    nonorth = abs(chi_out - np.eye(chi_out.shape[-1])).max()
    if tol is not None and nonorth > tol:
        log.error("Orbital non-orthogonality= %.1e", nonorth)

    return c_out



# OLD STUFF
# =========

def eigassign(e1, v1, e2, v2, b=None, cost_matrix="e^2/v"):
    """
    Parameters
    ----------
    b : ndarray
        If set, eigenvalues and eigenvectors belong to a generalized eigenvalue problem of the form Av=Bve.
    cost_matrix : str
        Defines the way to calculate the cost matrix.
    """

    if e1.shape != e2.shape:
        raise ValueError("e1=%r with shape=%r and e2=%r with shape=%r are not compatible." % (e1, e1.shape, e2, e2.shape))
    if v1.shape != v2.shape:
        raise ValueError("v1=%r with shape=%r and v2=%r with shape=%r are not compatible." % (v1, v1.shape, v2, v2.shape))
    if e1.shape[0] != v1.shape[-1]:
        raise ValueError("e1=%r with shape=%r and v1=%r with shape=%r are not compatible." % (e1, e1.shape, v1, v1.shape))
    if e2.shape[0] != v2.shape[-1]:
        raise ValueError("e2=%r with shape=%r and v2=%r with shape=%r are not compatible." % (e2, e2.shape, v2, v2.shape))

    assert np.allclose(e1.imag, 0)
    assert np.allclose(e2.imag, 0)
    assert np.allclose(v1.imag, 0)
    assert np.allclose(v2.imag, 0)

    # Define a cost matrix ("dist") which measures the difference of two eigenpairs (ei,vi), (e'j, v'j)
    # of different eigenvalue problems
    if b is None:
        vmat = np.abs(np.dot(v1.T, v2))
    else:
        vmat = np.abs(np.linalg.multi_dot((v1.T, b, v2)))
    emat = np.abs(np.subtract.outer(e1, e2))

    # relative energy difference
    ematrel = emat / np.fmax(abs(e1), 1e-14)[:,np.newaxis]

    # Original formulation
    if cost_matrix == "(1-v)*e":
        dist = (1-vmat) * emat
    elif cost_matrix == "(1-v)":
        dist = (1-vmat)
    elif cost_matrix == "1/v":
        dist = 1/np.fmax(vmat, 1e-14)
    elif cost_matrix == "v/e":
        dist = -vmat / (emat + 1e-14)
    elif cost_matrix == "e/v":
        dist = emat / np.fmax(vmat, 1e-14)
    elif cost_matrix == "er/v":
        dist = ematrel / np.fmax(vmat, 1e-14)
    elif cost_matrix == "e/v^2":
        dist = emat / np.fmax(vmat, 1e-14)**2
    # This performed best in tests
    elif cost_matrix == "e^2/v":
        dist = emat**2 / np.fmax(vmat, 1e-14)
    elif cost_matrix == "e^2/v**2":
        dist = emat**2 / (vmat + 1e-14)**2
    elif cost_matrix == "e/sqrt(v)":
        dist = emat / np.sqrt(vmat + 1e-14)
    else:
        raise ValueError("Unknown cost_matrix: %s" % cost_matrix)

    row, col = scipy.optimize.linear_sum_assignment(dist)
    # The col indices are the new sorting
    cost = dist[row,col].sum()
    sort = col
    return sort, cost

def eigreorder_logging(e, reorder, log):
    for i, j in enumerate(reorder):
        # No reordering
        if i == j:
            continue
        # Swap between two eigenvalues
        elif reorder[j] == i:
            if i < j:
                log("Reordering eigenvalues %3d <-> %3d : %+6.3g <-> %+6.3g", j, i, e[j], e[i])
        # General reordering
        else:
            log("Reordering eigenvalues %3d --> %3d : %+6.3g --> %+6.3g", j, i, e[j], e[i])


