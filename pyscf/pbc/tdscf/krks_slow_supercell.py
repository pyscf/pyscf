#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent Kohn-Sham procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TDKS roots in a reliable fashion without any issues related to the
Davidson procedure. Several variants of TDKS are available:

 * `pyscf.tdscf.rks_slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rks_slow`: PBC (periodic boundary condition) implementation for RKS objects of `pyscf.pbc.scf`
   modules;
 * (this module)`pyscf.pbc.tdscf.krks_slow_supercell`: PBC implementation for KRKS objects of `pyscf.pbc.scf` modules.
   Works with an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
 * `pyscf.pbc.tdscf.krks_slow_gamma`: A Gamma-point calculation resembling the original `pyscf.pbc.tdscf.krks`
   module. Despite its name, it accepts KRKS objects with an arbitrary number of k-points but finds only few TDKS roots
   corresponding to collective oscillations without momentum transfer;
 * `pyscf.pbc.tdscf.krks_slow`: PBC implementation for KRKS objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points and employs k-point conservation (diagonalizes matrix blocks separately).
"""

# Convention for these modules:
# * PhysERI, PhysERI4, PhysERI8 are proxy classes for computing the full TDDFT matrix
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDRKS provides a container

from pyscf.tdscf.common_slow import TDProxyMatrixBlocks, PeriodicMFMixin
from pyscf.tdscf import rks_slow
from pyscf.pbc.tdscf import krhf_slow_supercell, KTDDFT
from pyscf.lib import einsum, cartesian_prod, norm, logger
from pyscf.pbc.tools.pbc import super_cell

import numpy
import scipy
from scipy import sparse

from itertools import product


def minus_k(model, threshold=None, degeneracy_threshold=None):
    """
    Retrieves an array of indexes of negative k.
    Args:
        model (KSCF): an arbitrary mean-field pbc model;
        threshold (float): a threshold for determining the negative;
        degeneracy_threshold (float): a threshold for assuming degeneracy;

    Returns:
        A list of integers with indexes of the corresponding k-points.
    """
    if threshold is None:
        threshold = 1e-8
    if degeneracy_threshold is None:
        degeneracy_threshold = 1e-8
    kpts = model.cell.get_scaled_kpts(model.kpts)
    result = []
    for id_k, k in enumerate(kpts):
        delta = norm(((kpts + k[numpy.newaxis, :]) - .5) % 1 - .5, axis=-1)
        i = numpy.argmin(delta)
        if delta[i] > threshold:
            raise RuntimeError("Could not find a negative k-point for k={} (ID: {:d}, best difference: {:.3e})".format(
                repr(k), id_k, delta[i]
            ))
        if abs(model.mo_energy[id_k] - model.mo_energy[i]).max() > degeneracy_threshold:
            raise RuntimeError("Non-symmetric band structure (time-reversal) at k={} (ID: {:d}) and k={} (ID: {:d}). "
                               "This prevents composing real-valued orbitals".format(
                                    repr(k), id_k, repr(kpts[i]), i,
                                ))
        result.append(i)
    return result


def assert_scf_converged(model, threshold=1e-7):
    """
    Tests if scf is converged.
    Args:
        model (KSCF): a model to test;
        threshold (float): threshold for eigenvalue comparison;

    Returns:
        True if model is converged and False otherwise.
    """
    ovlp = model.get_ovlp()
    fock = model.get_fock(dm=model.make_rdm1())
    # dm = model.make_rdm1()
    # fock = model.get_fock(
    #     model.get_hcore(model.cell),
    #     ovlp,
    #     model.get_veff(model.cell, dm),
    #     dm,
    # )
    for k, (m, e, v, o) in enumerate(zip(fock, model.mo_energy, model.mo_coeff, ovlp)):
        delta = norm(numpy.dot(m, v) - e[numpy.newaxis, :] * numpy.dot(o, v), axis=0)
        nabove = (delta > threshold).sum()

        eye = reduce(numpy.dot, (v.conj().T, o, v))
        delta_o = abs(eye - numpy.eye(eye.shape[0])).max()
        if nabove > 0:
            raise AssertionError("{:d} vectors at k={:d} are not converged, max difference: {:.3e}; orthogonality error: {:.3e}".format(
                nabove, k, max(delta), delta_o,
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
    Converts K to supercell with real orbitals.
    Args:
        model (KSCF): an arbitrary mean-field pbc model;
        grid_spec (Iterable): integer dimensions of the k-grid in KSCF;
        mf_constructor (Callable): a function constructing the mean-field object;
        threshold (float): a threshold for determining the negative k-point index;
        degeneracy_threshold (float): a threshold for assuming degeneracy when composing real-valued orbitals;
        imaginary_threshold (float): a threshold for asserting real-valued supercell orbitals;

    Returns:
        The same class where the Cell object was replaced by the supercell and all fields were adjusted accordingly.
    """
    if imaginary_threshold is None:
        imaginary_threshold = 1e-7

    mk = minus_k(model, threshold=threshold, degeneracy_threshold=degeneracy_threshold)
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
            rotation_matrix[i, i] = 1
            inv_rotation_matrix[i, i] = 1

        elif k < mk[k]:
            rotation_matrix[i, i] = .5**.5
            rotation_matrix[j, i] = .5**.5
            rotation_matrix[i, j] = -1.j * .5**.5
            rotation_matrix[j, j] = 1.j * .5**.5

            inv_rotation_matrix[i, i] = .5**.5
            inv_rotation_matrix[j, i] = 1.j * .5**.5
            inv_rotation_matrix[i, j] = .5**.5
            inv_rotation_matrix[j, j] = -1.j * .5**.5

        else:
            pass

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


def active_space_required(transform, final_space):
    """
    Calulcates the space before transform required for achieving the final active space.
    Args:
        transform (ndarray): the transformation;
        final_space (ndarray): the final active space;

    Returns:
        The initial active space.
    """
    return numpy.any((transform.toarray() != 0)[:, final_space], axis=1)


def supercell_response(vind, space, nocc, double, rot_bloch, mo_occ_bloch, log_dest):
    """
    Retrieves a raw response matrix.
    Args:
        vind (Callable): a pyscf matvec routine;
        space (ndarray): the active space;
        nocc (int): the total number of occupied orbitals (frozen and active);
        double (bool): set to True if `vind` returns the double-sized (i.e. full) matrix;
        rot_bloch (ndarray): a matrix specifying the rotation from real orbitals returned from pyscf to Bloch
        functions;
        mo_occ_bloch (Iterable): occupation numbers of Bloch orbitals;
        log_dest (object): pyscf logging;

    Returns:
        The TD matrix.
    """
    if not double:
        raise NotImplementedError("Not implemented for MK-type matrixes")

    logger.debug1(log_dest, "Performing a supercell proxy response calculation ...")
    # Calculate effective space of the supercell matrix required
    space_bloch = numpy.concatenate(space)
    space_real = active_space_required(rot_bloch, space_bloch)
    logger.debug1(log_dest, "  Bloch total space size: {:d}, supercell space size: {:d}, total space size: {:d}".format(
        sum(space_bloch),
        sum(space_real),
        len(space_bloch),
    ))

    # Retrieve real-valued matrices
    logger.debug1(log_dest, "  collecting the A, B matrices ...")
    response_real_a, response_real_b = rks_slow.molecular_response(vind, space_real, nocc, double)
    logger.debug1(log_dest, "  done, shapes: {} and {}".format(
        response_real_a.shape,
        response_real_b.shape,
    ))

    logger.debug1(log_dest, "Transforming into Bloch basis ...")
    # Reshape into ov
    nmo = sum(space_real)
    nocc = sum(space_real[:nocc])
    nvirt = nmo - nocc
    response_real_a = response_real_a.reshape(nocc, nvirt, nocc, nvirt)
    response_real_b = response_real_b.reshape(nocc, nvirt, nocc, nvirt)
    logger.debug1(log_dest, "  reshape into ov: shapes: {} and {}".format(
        response_real_a.shape,
        response_real_b.shape,
    ))

    # Set rotation
    logger.debug1(log_dest, "  preparing rotation real->Bloch ...")
    o = numpy.concatenate(mo_occ_bloch) != 0
    v = numpy.logical_not(o)

    rot_bloch = rot_bloch[space_real, :][:, space_bloch]
    r_oo = rot_bloch[:nocc, o]
    r_vv = rot_bloch[nocc:, v]
    r_ov = rot_bloch[:nocc, v]
    r_vo = rot_bloch[nocc:, o]
    logger.debug1(log_dest, "  block shapes: oo {} vv {} ov {}".format(
        r_oo.shape,
        r_vv.shape,
        r_ov.shape,
    ))

    if abs(r_ov).max() > 1e-14 or abs(r_vo).max() > 1e-14:
        raise RuntimeError("Occupied and virtual spaces are coupled by the rotation matrix")

    # Rotate
    logger.debug1(log_dest, "  rotating A ...")
    response_bloch_a = sparse_transform(response_real_a, 0, r_oo, 1, r_vv.conj(), 2, r_oo.conj(), 3, r_vv)
    logger.debug1(log_dest, "  rotating B ...")
    response_bloch_b = sparse_transform(response_real_b, 0, r_oo, 1, r_vv.conj(), 2, r_oo, 3, r_vv.conj())
    logger.debug1(log_dest, "  resulting shapes: {} and {}".format(response_bloch_a.shape, response_bloch_b.shape))

    return response_bloch_a, response_bloch_b


class PhysERI(PeriodicMFMixin, TDProxyMatrixBlocks):
    def __init__(self, model, x, mf_constructor, frozen=None, proxy=None):
        """
        A proxy class for calculating the TDKS matrix blocks (supercell version).

        Args:
            model (KRKS): the base model with a regular k-point grid which includes the Gamma-point;
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        model_super = k2s(model, x, mf_constructor)
        TDProxyMatrixBlocks.__init__(self, proxy(model_super) if proxy is not None else KTDDFT(model_super))
        PeriodicMFMixin.__init__(self, model, frozen=frozen)
        self.model_super = model_super

    def tdhf_primary_form(self, *args, **kwargs):
        """
        A primary form of TD matrixes.

        Returns:
            Output type: "full", "ab", or "mk" and the corresponding matrix(es).
        """
        a, b = supercell_response(
            self.proxy_vind,
            self.space,
            sum(self.nocc_full),
            True,
            self.model_super.supercell_inv_rotation,
            self.mo_occ,
            self.model,
        )

        # Transform into supercell convention: [k_o, o, k_v, v] -> [k_o, k_v, o, v]
        nocc_k = self.nocc[0]
        nk = len(self.nocc)
        nvirt_k = self.nmo[0] - self.nocc[0]
        size = nk * nk * nocc_k * nvirt_k

        a = a.reshape((nk, nocc_k, nk, nvirt_k) * 2).transpose(0, 2, 1, 3, 4, 6, 5, 7)
        b = b.reshape((nk, nocc_k, nk, nvirt_k) * 2).transpose(0, 2, 1, 3, 4, 6, 5, 7)
        return "ab", a.reshape(size, size), b.reshape(size, size)


vector_to_amplitudes = krhf_slow_supercell.vector_to_amplitudes


class TDRKS(rks_slow.TDRKS):
    v2a = staticmethod(vector_to_amplitudes)
    proxy_eri = PhysERI

    def __init__(self, mf, x, mf_constructor, frozen=None, proxy=None):
        """
        Performs TDKS calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RKS): the base restricted DFT model;
            x (Iterable): the original k-grid dimensions (numbers of k-points per each axis);
            mf_constructor (Callable): a function constructing the mean-field object;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        super(TDRKS, self).__init__(mf, frozen=frozen, proxy=proxy)
        self.fast = False
        self.x = x
        self.mf_cosntructor = mf_constructor

    def ao2mo(self):
        """
        Prepares ERI.

        Returns:
            A suitable ERI.
        """
        return self.proxy_eri(
            self._scf,
            x=self.x,
            mf_constructor=self.mf_cosntructor,
            frozen=self.frozen,
            proxy=self.__proxy__,
        )
