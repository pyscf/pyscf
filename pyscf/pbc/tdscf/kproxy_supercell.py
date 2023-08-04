#  Author: Artem Pulkin
# flake8: noqa
"""
This and other `proxy` modules implement the time-dependent mean-field procedure using the existing pyscf
implementations as a black box. The main purpose of these modules is to overcome the existing limitations in pyscf
(i.e. real-only orbitals, davidson diagonalizer, incomplete Bloch space, etc). The primary performance drawback is that,
unlike the original pyscf routines with an implicit construction of the eigenvalue problem, these modules construct TD
matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TD roots. Several variants of proxy-TD are available:

 * `pyscf.tdscf.proxy`: the molecular implementation;
 * `pyscf.pbc.tdscf.proxy`: PBC (periodic boundary condition) Gamma-point-only implementation;
 * (this module) `pyscf.pbc.tdscf.kproxy_supercell`: PBC implementation constructing supercells. Works with an arbitrary number of
   k-points but has an overhead due to ignoring the momentum conservation law. In addition, works only with
   time reversal invariant (TRI) models: i.e. the k-point grid has to be aligned and contain at least one TRI momentum.
 * `pyscf.pbc.tdscf.kproxy`: same as the above but respect the momentum conservation and, thus, diagonlizes smaller
   matrices (the performance gain is the total number of k-points in the model).
"""

# Convention for these modules:
# * PhysERI is the proxying class constructing time-dependent matrices
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDProxy provides a container

from functools import reduce
from pyscf.tdscf.common_slow import TDProxyMatrixBlocks, PeriodicMFMixin
from pyscf.tdscf import proxy as mol_proxy
from pyscf.pbc.tdscf import krhf_slow_supercell, KTDDFT, KTDHF
from pyscf.lib import einsum, cartesian_prod, norm, logger
from pyscf.pbc.tools.pbc import super_cell

import numpy
from scipy import sparse

from warnings import warn


def minus_k(model, threshold=None, degeneracy_threshold=None):
    """
    Retrieves an array of indexes of negative k.
    Args:
        model: a mean-field pbc model;
        threshold (float): a threshold for determining the negative;
        degeneracy_threshold (float): a threshold for assuming degeneracy;

    Returns:
        A list of integers with indexes of the corresponding k-points.
    """
    if threshold is None:
        threshold = 1e-8
    if degeneracy_threshold is None:
        degeneracy_threshold = 1e-6
    kpts = model.cell.get_scaled_kpts(model.kpts)
    result = []
    for id_k, k in enumerate(kpts):
        delta = norm(((kpts + k[numpy.newaxis, :]) - .5) % 1 - .5, axis=-1)
        i = numpy.argmin(delta)
        if delta[i] > threshold:
            raise RuntimeError("Could not find a negative k-point for k={} (ID: {:d}, best difference: {:.3e}, "
                               "threshold: {:.3e}). Use the 'threshold' keyword to loosen the threshold or revise"
                               "your model".format(
                repr(k), id_k, delta[i], threshold,
            ))
        delta = abs(model.mo_energy[id_k] - model.mo_energy[i]).max()
        if delta > degeneracy_threshold:
            raise RuntimeError("Non-symmetric band structure (time-reversal) at k={} (ID: {:d}) and k={} (ID: {:d}), "
                               "max difference: {:.3e}, threshold: {:.3e}. This prevents composing real-valued "
                               "orbitals. Use the 'degeneracy_threshold' keyword to loosen the threshold or revise"
                               "your model".format(
                                    repr(k), id_k, repr(kpts[i]), i, delta, degeneracy_threshold,
                                ))
        result.append(i)
    return result


def assert_scf_converged(model, threshold=1e-7):
    """
    Tests if scf is converged.
    Args:
        model: a mean-field model to test;
        threshold (float): threshold for eigenvalue comparison;

    Returns:
        True if model is converged and False otherwise.
    """
    ovlp = model.get_ovlp()
    fock = model.get_fock(dm=model.make_rdm1())
    for k, (m, e, v, o) in enumerate(zip(fock, model.mo_energy, model.mo_coeff, ovlp)):
        delta = norm(numpy.dot(m, v) - e[numpy.newaxis, :] * numpy.dot(o, v), axis=0)
        nabove = (delta > threshold).sum()

        eye = reduce(numpy.dot, (v.conj().T, o, v))
        delta_o = abs(eye - numpy.eye(eye.shape[0])).max()
        if nabove > 0:
            warn("{:d} vectors at k={:d} are not converged, max difference: {:.3e}; orthogonality error: {:.3e} "
                 "warning threshold: {:.3e}".format(
                    nabove, k, max(delta), delta_o, threshold,
                 ))


def sparse_transform(m, *args):
    """
    Performs a sparse transform of a dense tensor.
    Args:
        m (ndarray): a tensor to transform;
        *args: alternating indexes and bases to transform into;

    Returns:
        The transformed tensor.
    """
    result = m
    for i, (index, basis) in enumerate(zip(args[::2], args[1::2])):

        if len(basis.shape) != 2:
            raise ValueError("Transform {:d} is not a matrix: shape = {}".format(
                i, repr(basis.shape)
            ))
        if result.shape[index] != basis.shape[0]:
            raise ValueError("Dimension mismatch of transform {:d}: m.shape[{:d}] = {:d} != basis.shape[0] = {:d}".format(
                i, index, result.shape[index], basis.shape[0],
            ))
        if "getcol" not in dir(basis):
            raise ValueError("No 'getcol' in the transform matrix {:d}: not a CSC sparse matrix?")

        result_shape = result.shape[:index] + (basis.shape[1],) + result.shape[index + 1:]
        new_result = numpy.zeros(result_shape, numpy.common_type(*(
            args[1::2] + (m,)
        )))
        for b2 in range(basis.shape[1]):
            slice_b2 = (slice(None),) * index + (b2,)
            col = basis.getcol(b2)
            for b1 in col.nonzero()[0]:
                slice_b1 = (slice(None),) * index + (b1,)
                new_result[slice_b2] += col[b1, 0] * result[slice_b1]
        result = new_result

    return result


def k2s(model, grid_spec, mf_constructor, threshold=None, degeneracy_threshold=None, imaginary_threshold=None):
    """
    Converts k-point model into a supercell with real orbitals.
    Args:
        model: a mean-field pbc model;
        grid_spec (Iterable): integer dimensions of the k-grid in the mean-field model;
        mf_constructor (Callable): a function constructing the mean-field object;
        threshold (float): a threshold for determining the negative k-point index;
        degeneracy_threshold (float): a threshold for assuming degeneracy when composing real-valued orbitals;
        imaginary_threshold (float): a threshold for asserting real-valued supercell orbitals;

    Returns:
        The same class where the Cell object was replaced by the supercell and all fields were adjusted accordingly.
    """
    # This hack works as follows. Provided TRS Hamiltonian
    #    H(k) = H(-k)*,
    # with same real eigenvalues and eigenfunctions related as
    #    psi(k) = c psi(-k)*,
    # c - arbitrary phase, it is easy to construct real (non-Bloch) eigenvectors of the whole Hamiltonian
    #    real1(|k|) = c* psi(k) + psi(-k) = psi(-k)* + psi(-k)
    # and
    #    real2(|k|) = 1.j * (c* psi(k) - psi(-k)) = 1.j* (psi(-k)* - psi(-k)).
    # The coefficient c is determined as
    #    psi(k) * psi(-k) = c psi(-k)* * psi(-k) = c
    if imaginary_threshold is None:
        imaginary_threshold = 1e-7

    mk = minus_k(model, threshold=threshold, degeneracy_threshold=degeneracy_threshold)

    # Fix phases
    ovlp = model.get_ovlp()
    phases = {}
    for k1, k2 in enumerate(mk):
        if k1 <= k2:
            c1 = model.mo_coeff[k1]
            c2 = model.mo_coeff[k2]
            o = ovlp[k1]
            r = reduce(numpy.dot, (c2.T, o, c1))
            delta = abs(abs(r) - numpy.eye(r.shape[0])).max()
            if delta > imaginary_threshold:
                raise RuntimeError("K-points connected by time reversal {:d} and {:d} are not complex conjugate: "
                                   "the difference {:.3e} is larger than the threshold {:.3e}".format(
                                        k1, k2, delta, imaginary_threshold,
                                    ))
            p = numpy.angle(numpy.diag(r))
            if k1 == k2:
                phases[k1] = numpy.exp(- .5j * p)[numpy.newaxis, :]
            else:
                phases[k1] = numpy.exp(- 1.j * p)[numpy.newaxis, :]

    nk = len(model.kpts)
    t_vecs = cartesian_prod(tuple(numpy.arange(i) for i in grid_spec))
    kpts_frac = model.cell.get_scaled_kpts(model.kpts)

    result = mf_constructor(super_cell(model.cell, grid_spec))
    result_ovlp = result.get_ovlp()[0]

    moe = numpy.concatenate(model.mo_energy)
    moo = numpy.concatenate(model.mo_occ)

    # Complex-valued wf in a supercell
    moc = []
    for mo_coeff, k in zip(model.mo_coeff, kpts_frac):
        psi = (
            mo_coeff[numpy.newaxis, ...] * numpy.exp(2.j * numpy.pi * t_vecs.dot(k))[:, numpy.newaxis, numpy.newaxis]
        ).reshape(-1, mo_coeff.shape[1])
        norms = einsum("ai,ab,bi->i", psi.conj(), result_ovlp, psi) ** .5
        psi /= norms[numpy.newaxis, :]
        moc.append(psi)
    moc = numpy.concatenate(moc, axis=1)

    rotation_matrix = sparse.dok_matrix(moc.shape, dtype=moc.dtype)
    inv_rotation_matrix = sparse.dok_matrix(moc.shape, dtype=moc.dtype)
    nvecs = (0,) + tuple(i.shape[1] for i in model.mo_coeff)
    nvecs = numpy.cumsum(nvecs)
    k_spaces = tuple(numpy.arange(i, j) for i, j in zip(nvecs[:-1], nvecs[1:]))

    for k in range(nk):

        i = k_spaces[k]
        j = k_spaces[mk[k]]

        if k == mk[k]:
            rotation_matrix[i, i] = phases[k]
            inv_rotation_matrix[i, i] = phases[k].conj()

        elif k < mk[k]:
            rotation_matrix[i, i] = .5**.5 * phases[k]
            rotation_matrix[j, i] = .5**.5
            rotation_matrix[i, j] = -1.j * .5**.5 * phases[k]
            rotation_matrix[j, j] = 1.j * .5**.5

            inv_rotation_matrix[i, i] = .5**.5 * phases[k].conj()
            inv_rotation_matrix[j, i] = 1.j * .5**.5 * phases[k].conj()
            inv_rotation_matrix[i, j] = .5**.5
            inv_rotation_matrix[j, j] = -1.j * .5**.5

        else:
            pass

    rotation_matrix = rotation_matrix.tocsc()
    inv_rotation_matrix = inv_rotation_matrix.tocsc()

    moc = sparse_transform(moc, 1, rotation_matrix)
    max_imag = abs(moc.imag).max()
    if max_imag > imaginary_threshold:
        raise RuntimeError("Failed to compose real-valued orbitals: imaginary part is {:.3e}".format(max_imag))
    moc = moc.real

    mok = numpy.concatenate(tuple([i] * len(j) for i, j in enumerate(model.mo_energy)))
    moi = numpy.concatenate(tuple(numpy.arange(len(j)) for j in model.mo_energy))

    order = numpy.argsort(moe)

    moe = moe[order]
    moc = moc[:, order]
    moo = moo[order]
    mok = mok[order]
    moi = moi[order]
    rotation_matrix = rotation_matrix[:, order]
    inv_rotation_matrix = inv_rotation_matrix[order, :]

    result.mo_occ = moo,
    result.mo_energy = moe,
    result.mo_coeff = moc,
    result.supercell_rotation = rotation_matrix
    result.supercell_inv_rotation = inv_rotation_matrix
    result.supercell_orig_k = mok
    result.supercell_orig_i = moi

    assert_scf_converged(result, model.conv_tol ** .5)

    p1 = abs(result.supercell_rotation.dot(result.supercell_inv_rotation) - numpy.eye(rotation_matrix.shape[0])).max()
    p2 = abs(result.supercell_inv_rotation.dot(result.supercell_rotation) - numpy.eye(rotation_matrix.shape[0])).max()
    if p1 > 1e-14 or p2 > 1e-14:
        raise RuntimeError("Rotation matrix error: {:.3e}, {:.3e}".format(p1, p2))

    return result


def ko_mask(nocc, nmo):
    """
    Prepares a mask of an occupied space.
    Args:
        nocc (Iterable): occupation numbers per k-point;
        nmo (Iterable): numbers of orbitals per k-point;

    Returns:
        The mask where `True` denotes occupied orbitals. Basis order: [k, orb=o+v]
    """
    result = numpy.zeros(sum(nmo), dtype=bool)
    offset = 0
    for no, nm in zip(nocc, nmo):
        result[offset:offset+no] = True
        offset += nm
    return result


def split_transform(transform, nocc, nmo, tolerance=1e-14):
    """
    Splits the transform into `oo` and `vv` blocks.
    Args:
        transform (numpy.ndarray): the original transform. The basis order for the transform is [real orb=o+v; k, orb=o+v];
        nocc (Iterable): occupation numbers per k-point;
        nmo (Iterable): the number of orbitals per k-point;
        tolerance (float): tolerance to check zeros at the `ov` block;

    Returns:
        `oo` and `vv` blocks of the transform.
    """
    o_mask = ko_mask(nocc, nmo)
    v_mask = ~o_mask
    ov = transform[:sum(nocc), v_mask]
    vo = transform[sum(nocc):, o_mask]
    if abs(ov).max() > tolerance or abs(vo).max() > tolerance:
        raise ValueError("Occupied and virtual spaces are coupled by the transformation")
    return transform[:sum(nocc), o_mask], transform[sum(nocc):, v_mask]


def supercell_space_required(transform_oo, transform_vv, final_space):
    """
    For a given orbital transformation and a given `ov` mask in the transformed space, calculates a minimal `ov` mask
    in the original space required to achieve this transform.
    Args:
        transform_oo (ndarray): the transformation in the occupied space;
        transform_vv (ndarray): the transformation in the virtual space;
        final_space (ndarray): the final `ov` space. Basis order: [k_o, o, k_v, v];

    Returns:
        The initial active space. Basis order: [k_o, o, k_v, v].
    """
    final_space = numpy.asanyarray(final_space)
    final_space = final_space.reshape(final_space.shape[:-1] + (transform_oo.shape[1], transform_vv.shape[1]))
    result = einsum(
        "ao,bv,...ov->...ab",
        (transform_oo.toarray() != 0).astype(int),
        (transform_vv.toarray() != 0).astype(int),
        final_space.astype(int),
    ) != 0
    return result.reshape(result.shape[:-2] + (-1,))


def get_sparse_ov_transform(oo, vv):
    """
    Retrieves a sparse `ovov` transform out of sparse `oo` and `vv` transforms.
    Args:
        oo (ndarray): the transformation in the occupied space;
        vv (ndarray): the transformation in the virtual space;

    Returns:
        The resulting matrix representing the sparse transform in the `ov` space.
    """
    i, a = oo.shape
    j, b = vv.shape

    # If the input is dense the result is simply
    # return (oo[:, numpy.newaxis, :, numpy.newaxis] * vv[numpy.newaxis, :, numpy.newaxis, :]).reshape(i*j, a*b)

    result_data = numpy.zeros(oo.nnz * vv.nnz, dtype=numpy.common_type(oo, vv))
    result_indices = numpy.zeros(len(result_data), dtype=int)
    result_indptr = numpy.zeros(a * b + 1, dtype=int)

    ptr_counter = 0
    for i_a in range(a):
        oo_col = oo.getcol(i_a)
        assert tuple(oo_col.indptr.tolist()) == (0, len(oo_col.data))
        i_i, oo_col_v = oo_col.indices, oo_col.data
        for i_b in range(b):
            vv_col = vv.getcol(i_b)
            assert tuple(vv_col.indptr.tolist()) == (0, len(vv_col.data))
            i_j, vv_col_v = vv_col.indices, vv_col.data

            data_length = len(i_i) * len(i_j)
            result_indices[ptr_counter:ptr_counter + data_length] = ((i_i * j)[:, numpy.newaxis] + i_j[numpy.newaxis, :]).reshape(-1)
            result_data[ptr_counter:ptr_counter + data_length] = (oo_col_v[:, numpy.newaxis] * vv_col_v[numpy.newaxis, :]).reshape(-1)
            result_indptr[i_a * b + i_b] = ptr_counter

            ptr_counter += data_length

    result_indptr[-1] = ptr_counter
    return sparse.csc_matrix((result_data, result_indices, result_indptr))


def ov2orb(space, nocc, nmo):
    """
    Converts ov-pairs active space specification into orbital space spec.
    Args:
        space (ndarray): the ov space. Basis order: [k_o, o, k_v, v];
        nocc (Iterable): the numbers of occupied orbitals per k-point;
        nmo (Iterable): the total numbers of orbitals per k-point;

    Returns:
        The orbital space specification. Basis order: [k, orb=o+v].
    """
    nocc = numpy.asanyarray(nocc)
    nmo = numpy.asanyarray(nmo)
    nvirt = nmo - nocc

    space = numpy.asanyarray(space)
    space = space.reshape(space.shape[:-1] + (sum(nocc), sum(nvirt)))  # [k_o, o; k_v, v]

    s_o = numpy.any(space, axis=-1)  # [k_o, o]
    s_v = numpy.any(space, axis=-2)  # [k_v, v]

    o_offset = numpy.cumsum(numpy.concatenate(([0], nocc)))
    v_offset = numpy.cumsum(numpy.concatenate(([0], nvirt)))
    result = []
    for o_fr, o_to, v_fr, v_to in zip(o_offset[:-1], o_offset[1:], v_offset[:-1], v_offset[1:]):
        result.append(s_o[..., o_fr:o_to])
        result.append(s_v[..., v_fr:v_to])
    return numpy.concatenate(result, axis=-1)  # [k, orb=o+v]


def supercell_response_ov(vind, space_ov, nocc, nmo, double, rot_bloch, log_dest):
    """
    Retrieves a raw response matrix.
    Args:
        vind (Callable): a pyscf matvec routine;
        space_ov (ndarray): the active `ov` space mask: either the same mask for both rows and columns (1D array) or
        separate `ov` masks for rows and columns (2D array). Basis order: [k_o, o, k_v, v];
        nocc (ndarray): the numbers of occupied orbitals (frozen and active) per k-point;
        nmo (ndarray): the total number of orbitals per k-point;
        double (bool): set to True if `vind` returns the double-sized (i.e. full) matrix;
        rot_bloch (ndarray): a matrix specifying the rotation from real orbitals returned from pyscf to Bloch
        functions;
        log_dest (object): pyscf logging;

    Returns:
        The TD matrix.
    """
    if not double:
        raise NotImplementedError("Not implemented for MK-type matrixes")

    nocc_full = sum(nocc)
    nmo_full = sum(nmo)
    nvirt_full = nmo_full - nocc_full
    size_full = nocc_full * nvirt_full

    space_ov = numpy.array(space_ov)

    if space_ov.shape == (size_full,):
        space_ov = numpy.repeat(space_ov[numpy.newaxis, :], 2, axis=0)

    elif space_ov.shape != (2, size_full):
        raise ValueError(
            "The 'space_ov' argument should be a 1D array with dimension {size_full:d} or a 2D array with"
            " dimensions 2x{size_full:d}, found: {actual}".format(
                size_full=size_full,
                actual=space_ov.shape,
            ))

    oo, vv = split_transform(rot_bloch, nocc, nmo)
    space_real_ov = supercell_space_required(oo, vv, space_ov)

    logger.debug1(log_dest, "Performing a supercell proxy response calculation ...")
    logger.debug1(log_dest, "  Total ov space size: {:d} requested elements: {} real elements to calculate: {}".format(
        size_full,
        "x".join(map(str, space_ov.sum(axis=-1))),
        "x".join(map(str, space_real_ov.sum(axis=-1))),
    ))

    logger.debug1(log_dest, "  collecting the A, B matrices ...")
    response_real_a, response_real_b = mol_proxy.molecular_response_ov(
        vind,
        space_real_ov,
        nocc_full,
        nmo_full,
        double,
        log_dest,
    )
    logger.debug1(log_dest, "  done, shapes: {} and {}".format(
        response_real_a.shape,
        response_real_b.shape,
    ))

    logger.debug1(log_dest, "Transforming into Bloch basis ...")

    ovov_nc = get_sparse_ov_transform(oo, vv.conj())
    ovov_cn = get_sparse_ov_transform(oo.conj(), vv)

    ovov_row = ovov_nc[:, space_ov[0]][space_real_ov[0]]
    ovov_col_a = ovov_cn[:, space_ov[1]][space_real_ov[1]]
    ovov_col_b = ovov_nc[:, space_ov[1]][space_real_ov[1]]

    # Rotate
    logger.debug1(log_dest, "  rotating A ...")
    response_bloch_a = sparse_transform(response_real_a, 0, ovov_row, 1, ovov_col_a)
    logger.debug1(log_dest, "  rotating B ...")
    response_bloch_b = sparse_transform(response_real_b, 0, ovov_row, 1, ovov_col_b)
    logger.debug1(log_dest, "  shapes: {} and {}".format(response_bloch_a.shape, response_bloch_b.shape))

    return response_bloch_a, response_bloch_b


def orb2ov(space, nocc, nmo):
    """
    Converts orbital active space specification into ov-pairs space spec.
    Args:
        space (ndarray): the obital space. Basis order: [k, orb=o+v];
        nocc (Iterable): the numbers of occupied orbitals per k-point;
        nmo (Iterable): the total numbers of orbitals per k-point;

    Returns:
        The ov space specification. Basis order: [k_o, o, k_v, v].
    """
    space = numpy.asanyarray(space)
    m = ko_mask(nocc, nmo)  # [k, orb=o+v]
    o = space[...,  m]  # [k, o]
    v = space[..., ~m]  # [k, v]
    return (o[..., numpy.newaxis] * v[..., numpy.newaxis, :]).reshape(space.shape[:-1] + (-1,))  # [k_o, o, k_v, v]


def supercell_response(vind, space, nocc, nmo, double, rot_bloch, log_dest):
    """
    Retrieves a raw response matrix.
    Args:
        vind (Callable): a pyscf matvec routine;
        space (ndarray): the active space: either for both rows and columns (1D array) or for rows and columns
        separately (2D array). Basis order: [k, orb=o+v];
        nocc (ndarray): the numbers of occupied orbitals (frozen and active) per k-point;
        nmo (ndarray): the total number of orbitals per k-point;
        double (bool): set to True if `vind` returns the double-sized (i.e. full) matrix;
        rot_bloch (ndarray): a matrix specifying the rotation from real orbitals returned from pyscf to Bloch
        functions;
        log_dest (object): pyscf logging;

    Returns:
        The TD matrix.
    """
    if not double:
        raise NotImplementedError("Not implemented for MK-type matrixes")

    # Full space dims
    nmo_full = sum(nmo)
    space = numpy.array(space)

    if space.shape == (nmo_full,):
        space = numpy.repeat(space[numpy.newaxis, :], 2, axis=0)
    elif space.shape != (2, nmo_full):
        raise ValueError("The 'space' argument should a 1D array with dimension {:d} or a 2D array with dimensions {},"
                         " found: {}".format(nmo_full, (2, nmo_full), space.shape))

    return supercell_response_ov(vind, orb2ov(space, nocc, nmo), nocc, nmo, double, rot_bloch, log_dest)


class PhysERI(PeriodicMFMixin, TDProxyMatrixBlocks):
    proxy_choices = {
        "hf": KTDHF,
        "dft": KTDDFT,
    }

    def __init__(self, model, proxy, x, mf_constructor, frozen=None, **kwargs):
        """
        A proxy class for calculating TD matrix blocks (supercell version).

        Args:
            model: the base model with a time reversal-invariant k-point grid;
            proxy: a pyscf proxy with TD response function, one of 'hf', 'dft';
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            **kwargs: arguments to `k2s` function constructing supercells;
        """
        model_super = k2s(model, x, mf_constructor, **kwargs)
        TDProxyMatrixBlocks.__init__(self, self.proxy_choices[proxy](model_super))
        PeriodicMFMixin.__init__(self, model, frozen=frozen)
        self.model_super = model_super

    def proxy_is_double(self):
        """
        Determines if double-sized matrices are proxied.
        Returns:
            True if double-sized matrices are proxied.
        """
        nocc_full = sum(self.nocc_full)
        nmo_full = sum(self.nmo_full)
        size_full = nocc_full * (nmo_full - nocc_full)
        size_hdiag = len(self.proxy_diag)

        if size_full == size_hdiag:
            return False

        elif 2 * size_full == size_hdiag:
            return True

        else:
            raise RuntimeError("Do not recognize the size of TD diagonal: {:d}. The size of ov-space is {:d}".format(
                size_hdiag, size_full
            ))

    def proxy_response(self):
        """
        A raw response matrix.
        Returns:
            A raw response matrix.
        """
        return supercell_response(
            self.proxy_vind,
            numpy.concatenate(self.space),
            self.nocc_full,
            self.nmo_full,
            self.proxy_is_double(),
            self.model_super.supercell_inv_rotation,
            self.model,
        )

    def tdhf_primary_form(self, *args, **kwargs):
        """
        A primary form of TD matrixes.

        Returns:
            Output type: "full", "ab", or "mk" and the corresponding matrix(es).
        """
        a, b = self.proxy_response()

        # Transform into supercell convention: [k_o, o, k_v, v] -> [k_o, k_v, o, v]
        nocc_k = self.nocc[0]
        nk = len(self.nocc)
        nvirt_k = self.nmo[0] - self.nocc[0]
        size = nk * nk * nocc_k * nvirt_k

        a = a.reshape((nk, nocc_k, nk, nvirt_k) * 2).transpose(0, 2, 1, 3, 4, 6, 5, 7)
        b = b.reshape((nk, nocc_k, nk, nvirt_k) * 2).transpose(0, 2, 1, 3, 4, 6, 5, 7)
        return "ab", a.reshape(size, size), b.reshape(size, size)


vector_to_amplitudes = krhf_slow_supercell.vector_to_amplitudes


class TDProxy(mol_proxy.TDProxy):
    v2a = staticmethod(vector_to_amplitudes)
    proxy_eri = PhysERI

    def __init__(self, mf, proxy, x, mf_constructor, frozen=None, **kwargs):
        """
        Performs TD calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf: the base model with a time reversal-invariant k-point grid;
            proxy: a pyscf proxy with TD response function, one of 'hf', 'dft';
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            **kwargs: arguments to `k2s` function constructing supercells;
        """
        super(TDProxy, self).__init__(mf, proxy, frozen=frozen)
        self.fast = False
        self.x = x
        self.mf_constructor = mf_constructor
        self.__k2s_kwargs__ = kwargs

    def ao2mo(self):
        """
        Prepares ERI.

        Returns:
            A suitable ERI.
        """
        return self.proxy_eri(
            self._scf,
            self.__proxy__,
            x=self.x,
            mf_constructor=self.mf_constructor,
            frozen=self.frozen,
            **self.__k2s_kwargs__
        )

