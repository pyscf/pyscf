#  Author: Artem Pulkin
"""
This and other `_slow` modules implement the time-dependent Kohn-Sham procedure. The primary performance drawback is
that, unlike other 'fast' routines with an implicit construction of the eigenvalue problem, these modules construct
TDHF matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TDKS roots in a reliable fashion without any issues related to the
Davidson procedure. Several variants of TDKS are available:

 * `pyscf.tdscf.rks_slow`: the molecular implementation;
 * (this module) `pyscf.pbc.tdscf.rks_slow`: PBC (periodic boundary condition) implementation for RKS objects of
   `pyscf.pbc.scf` modules;
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

from pyscf.tdscf import rks_slow, rhf_slow, common_slow
from pyscf.pbc.tdscf import KTDDFT


class PhysERI(common_slow.GammaMFMixin, rks_slow.PhysERI):

    def __init__(self, model, frozen=None):
        """
        A proxy class for calculating the TDKS matrix blocks (Gamma-point version).

        Args:
            model (KRKS): the base model with a single k-point;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        common_slow.TDProxyMatrixBlocks.__init__(self, KTDDFT(model))
        common_slow.GammaMFMixin.__init__(self, model, frozen=frozen)


vector_to_amplitudes = rhf_slow.vector_to_amplitudes


class TDRKS(rks_slow.TDRKS):
    eri1 = PhysERI

    def __init__(self, mf, frozen=None):
        """
        Performs TDKS calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf (RKS): the base restricted DFT model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
        """
        super(TDRKS, self).__init__(mf, frozen=frozen)
        self.fast = False
