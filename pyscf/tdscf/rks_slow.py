#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent Kohn-Sham procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TDKS roots in a reliable fashion without any issues related to the
Davidson procedure. Several variants of TDKS are available:

 * (this module) `pyscf.tdscf.rks_slow`: the molecular implementation;
 * `pyscf.pbc.tdscf.rks_slow`: PBC (periodic boundary condition) implementation for RKS objects of `pyscf.pbc.scf`
 modules;
 * `pyscf.pbc.tdscf.krks_slow_supercell`: PBC implementation for KRKS objects of `pyscf.pbc.scf` modules. Works with
   an arbitrary number of k-points but has a overhead due to an effective construction of a supercell.
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

from pyscf.tdscf import rhf_slow as tdhf
import pyscf.tdscf as proxy

import numpy


def rotate_proxy(m, o, v, return_ov=False):
    """
    Rotates the output of pyscf TDDFT matrix to convention.
    Args:
        m (ndarray): the TDDFT matrix to diagonalize;
        o (ndarray): occupied orbital energies;
        v (ndarray): virtual orbital energies;
        return_ov (bool): if True, returns the K-matrix as well;

    Returns:
        The rotated matrix as well as .
    """
    e_ov = (v[numpy.newaxis, :] - o[:, numpy.newaxis]).reshape(-1)
    e_ov_sqr = e_ov ** .5
    result = m * (e_ov_sqr[numpy.newaxis, :] / e_ov_sqr[:, numpy.newaxis])
    if return_ov:
        return result, numpy.diag(e_ov)
    else:
        return result


class PhysERI(tdhf.PhysERI):

    def __init__(self, model, frozen=None):
        """
        A proxy class for calculating the TDKS matrix blocks.

        Args:
            model (RKS): the base model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        tdhf.TDDFTMatrixBlocks.__init__(self)
        self.model = model
        self.proxy_model = proxy.TDDFT(model)
        self.space = tdhf.format_frozen(frozen, len(model.mo_energy))

    @property
    def nocc_full(self):
        return int(self.model.mo_occ.sum() // 2)

    @property
    def nmo_full(self):
        return len(self.model.mo_occ)

    def ao2mo(self, coeff):
        """
        Phys ERI in MO basis.
        Args:
            coeff (Iterable): MO orbitals;

        Returns:
            ERI in MO basis.
        """
        raise RuntimeError("No calls to ao2mo expected in the time-dependent KS workflow")

    def tdhf_a(self, *args, **kwargs):
        """
        The TDHF A-matrix.
        Returns:
            The matrix.
        """
        raise RuntimeError("This is a proxy class: no calls to A and B matrices expected: only 'fast_tdhf_matrix_set'"
                           " can be used")

    def tdhf_b(self, *args, **kwargs):
        """
        The TDHF B-matrix.
        Returns:
            The matrix.
        """
        raise RuntimeError("This is a proxy class: no calls to A and B matrices expected: only 'fast_tdhf_matrix_set'"
                           " can be used")

    def tdhf_matrix(self, *args, **kwargs):
        """
        Full matrix of the TDKS problem. For ProxyERI, only one of `tdhf_matrix`, `fast_tdhf_matrix_set` works,
        depending on whether `proxy_model.gen_vind` returns half- or full-sized matrixes.
        Returns:
            The matrix.
        """
        # Retrieve matrix in the active space specified
        vind, hdiag = self.proxy_model.gen_vind(self.model)

        nmo_full = self.nmo_full
        nocc_full = self.nocc_full
        nvirt_full = nmo_full - nocc_full
        size_full = 2 * nocc_full * nvirt_full

        if len(hdiag) != size_full:
            raise RuntimeError("The underlying TD* matvec routine returns arrays of unexpected size: {:d} vs "
                               "{:d} (expected)".format(len(hdiag), size_full))

        nmo = self.nmo
        nocc = self.nocc
        nvirt = nmo - nocc
        size = 2 * nocc * nvirt

        probe = numpy.zeros((size, size_full))

        o = self.space[:nocc_full]
        v = self.space[nocc_full:]
        ov = numpy.tile((o[:, numpy.newaxis] * v[numpy.newaxis, :]).reshape(-1), 2)

        probe[numpy.arange(size), numpy.argwhere(ov)[:, 0]] = 1
        result = vind(probe).T[ov, :]

        return result

    def fast_tdhf_matrix_set(self):
        """
        A set of real tdks matrixes to perform an optimized diagonlaization. For ProxyERI, only one of `tdhf_matrix`,
        `fast_tdhf_matrix_set` works, depending on whether `proxy_model.gen_vind` returns half- or full-sized matrixes.
        Returns:
            The matrix to diagonalize as well as the matrix to determine the second half of the TD KS solution.
        """
        # Retrieve matrix in the active space specified
        vind, hdiag = self.proxy_model.gen_vind(self.model)

        nmo_full = self.nmo_full
        nocc_full = self.nocc_full
        nvirt_full = nmo_full - nocc_full
        size_full = nocc_full * nvirt_full

        if len(hdiag) != size_full:
            raise RuntimeError("The underlying TD* matvec routine returns arrays of unexpected size: {:d} vs "
                               "{:d} (expected)".format(len(hdiag), size_full))

        nmo = self.nmo
        nocc = self.nocc
        nvirt = nmo - nocc
        size = nocc * nvirt

        probe = numpy.zeros((size, size_full))

        o = self.space[:nocc_full]
        v = self.space[nocc_full:]
        ov = (o[:, numpy.newaxis] * v[numpy.newaxis, :]).reshape(-1)

        probe[numpy.arange(size), numpy.argwhere(ov)[:, 0]] = 1
        result = vind(probe).T[ov, :]

        # The TD matrix is not exactly the output of gen_vind: some rotation is needed
        e_occ = self.mo_energy[:nocc]
        e_virt = self.mo_energy[nocc:]
        return rotate_proxy(result, e_occ, e_virt, return_ov=True)


vector_to_amplitudes = tdhf.vector_to_amplitudes


class TDRKS(tdhf.TDRHF):
    eri1 = PhysERI
    eri4 = eri8 = None

    def __init__(self, mf, frozen=None):
        """
        Performs TDKS calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RKS): the base restricted DFT model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        super(TDRKS, self).__init__(mf, frozen=frozen)
        self.fast = True
