#  Author: Artem Pulkin
"""
This and other `proxy` modules implement the time-dependent mean-field procedure using the existing pyscf
implementations as a black box. The main purpose of these modules is to overcome the existing limitations in pyscf
(i.e. real-only orbitals, davidson diagonalizer, incomplete Bloch space, etc). The primary performance drawback is that,
unlike the original pyscf routines with an implicit construction of the eigenvalue problem, these modules construct TD
matrices explicitly by proxying to pyscf density response routines with a O(N^4) complexity scaling. As a result,
regular `numpy.linalg.eig` can be used to retrieve TD roots. Several variants of proxy-TD are available:

 * `pyscf.tdscf.proxy`: the molecular implementation;
 * (this module) `pyscf.pbc.tdscf.proxy`: PBC (periodic boundary condition) Gamma-point-only implementation;
 * `pyscf.pbc.tdscf.kproxy_supercell`: PBC implementation constructing supercells. Works with an arbitrary number of
   k-points but has an overhead due to ignoring the momentum conservation law. In addition, works only with
   time reversal invariant (TRI) models: i.e. the k-point grid has to be aligned and contain at least one TRI momentum.
 * `pyscf.pbc.tdscf.kproxy`: same as the above but respect the momentum conservation and, thus, diagonlizes smaller
   matrices (the performance gain is the total number of k-points in the model).
"""

# Convention for these modules:
# * PhysERI is the proxying class constructing time-dependent matrices
# * vector_to_amplitudes reshapes and normalizes the solution
# * TDProxy provides a container

from pyscf.tdscf import proxy as mol_proxy, rhf_slow, common_slow
from pyscf.pbc.tdscf import KTDDFT


class PhysERI(common_slow.GammaMFMixin, mol_proxy.PhysERI):

    def __init__(self, model, frozen=None, proxy=None):
        """
        A proxy class for calculating the TD matrix blocks (Gamma-point version).

        Args:
            model: the base model with a single k-point;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        common_slow.TDProxyMatrixBlocks.__init__(self, proxy if proxy is not None else KTDDFT(model))
        common_slow.GammaMFMixin.__init__(self, model, frozen=frozen)


vector_to_amplitudes = rhf_slow.vector_to_amplitudes


class TDProxy(mol_proxy.TDProxy):
    proxy_eri = PhysERI
    v2a = staticmethod(vector_to_amplitudes)

    def __init__(self, mf, frozen=None, proxy=None):
        """
        Performs TD calculation. Roots and eigenvectors are stored in `self.e`, `self.xy`.
        Args:
            mf: the base restricted mean-field model;
            frozen (int, Iterable): the number of frozen valence orbitals or the list of frozen orbitals;
            proxy: a pyscf proxy with TD response function;
        """
        super(TDProxy, self).__init__(mf, frozen=frozen, proxy=proxy)
        self.fast = False
