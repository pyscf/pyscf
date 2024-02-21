"""
This module implements the G0W0 approximation on top of `pyscf.tdscf.rhf_slow` TDHF implementation. Unlike `gw.py`, all
integrals are stored in memory. Several variants of GW are available:

 * `pyscf.gw_slow`: the molecular implementation;
 * `pyscf.pbc.gw.gw_slow`: single-kpoint PBC (periodic boundary condition) implementation;
 * (this module) `pyscf.pbc.gw.kgw_slow_supercell`: a supercell approach to PBC implementation with multiple k-points.
   Runs the molecular code for a model with several k-points for the cost of discarding momentum conservation and using
   dense instead of sparse matrixes;
 * `pyscf.pbc.gw.kgw_slow`: a PBC implementation with multiple k-points;
"""

from pyscf.gw import gw_slow
from pyscf.lib import einsum

import numpy


# Convention for these modules:
# * IMDS contains routines for intermediates
# * kernel finds GW roots
# * GW provides a container


def corrected_moe(eri, k, p):
    """
    Calculates the corrected orbital energy.
    Args:
        eri (PhysERI): a container with electron repulsion integrals;
        k (int): the k-point index;
        p (int): orbital;

    Returns:
        The corrected orbital energy.
    """
    moe = eri.mo_energy[k][p]
    moc = eri.mo_coeff[k][:, p]
    nk = len(eri.mo_energy)
    vk = 0
    for k2 in range(nk):
        vk -= eri.ao2mo_k((
            moc[:, numpy.newaxis],
            eri.mo_coeff_full[k2][:, :eri.nocc_full[k2]],
            eri.mo_coeff_full[k2][:, :eri.nocc_full[k2]],
            moc[:, numpy.newaxis],
        ), (k, k2, k2, k)).squeeze().trace().real
    vk /= nk
    mf = eri.model
    v_mf = mf.get_veff()[k] - mf.get_j()[k]
    v_mf = einsum("i,ij,j", moc.conj(), v_mf, moc).real
    return moe + vk - v_mf


class IMDS(gw_slow.IMDS):
    orb_dims = 2

    def __init__(self, td, eri=None):
        """
        GW intermediates (k-version/supercell).
        Args:
            td: a container with TD solution;
            eri: a container with electron repulsion integrals;
        """
        gw_slow.AbstractIMDS.__init__(self, td, eri=eri)

        self.nk = len(self.td._scf.mo_energy)

        # MF
        self.nocc = sum(self.eri.nocc)
        self.o = numpy.concatenate(tuple(e[:nocc] for e, nocc in zip(self.eri.mo_energy, self.eri.nocc)))
        self.v = numpy.concatenate(tuple(e[nocc:] for e, nocc in zip(self.eri.mo_energy, self.eri.nocc)))

        # TD
        nroots, _, k1, k2, o, v = self.td.xy.shape
        self.td_xy = self.td.xy.transpose(0, 1, 2, 4, 3, 5).reshape(nroots, 2, k1*o, k2*v)
        self.td_e = self.td.e

        self.tdm = self.construct_tdm()

    def eri_ov(self, item):
        result = []
        k = numpy.arange(self.nk)
        for k1 in k:
            result.append([])
            for k2 in k:
                result[-1].append([])
                for k3 in k:
                    result[-1][-1].append([])
                    for k4 in k:
                        result[-1][-1][-1].append(self.eri.eri_ov(item, (k1, k2, k3, k4)))

        r = numpy.block(result)
        return r / len(k)

    __getitem__ = eri_ov

    def __plain_index__(self, p, spec=True):
        k, kp = p
        if kp < self.eri.nocc[k]:
            x = sum(self.eri.nocc[:k]) + kp
            if spec:
                return "o", x
            else:
                return x
        else:
            kp -= self.eri.nocc[k]
            x = sum(self.eri.nmo[:k]) - sum(self.eri.nocc[:k]) + kp
            if spec:
                return "v", x
            else:
                return x + self.nocc

    def get_rhs(self, p, components=False):
        k, kp = p
        # return self.eri.mo_energy[k][kp]
        return corrected_moe(self.eri, k, kp)

    def get_sigma_element(self, omega, p, eta, vir_sgn=1):
        return super().get_sigma_element(omega, self.__plain_index__(p, spec=False), eta, vir_sgn=vir_sgn)

    def initial_guess(self, p):
        k, kp = p
        return self.eri.mo_energy[k][kp]

    @property
    def entire_space(self):
        assert all(i == self.eri.nmo[0] for i in self.eri.nmo)
        return [numpy.arange(self.nk), numpy.arange(self.eri.nmo[0])]


kernel = gw_slow.kernel


class GW(gw_slow.GW):
    base_imds = IMDS
