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
from pyscf.lib import einsum, temporary_env

import numpy


# Convention for these modules:
# * IMDS contains routines for intermediates
# * kernel finds GW roots
# * GW provides a container


class IMDS(gw_slow.IMDS):
    orb_dims = 2

    def __init__(self, tdhf):
        """
        GW intermediates (k-version/supercell).
        Args:
            tdhf (TDRHF): the TDRHF contatainer;
        """
        gw_slow.AbstractIMDS.__init__(self, tdhf)

        self.nk = len(self.mf.kpts)

        # MF
        self.nocc = sum(self.eri.nocc)
        self.o = numpy.concatenate(tuple(e[:nocc] for e, nocc in zip(self.mf.mo_energy, self.eri.nocc)))
        self.v = numpy.concatenate(tuple(e[nocc:] for e, nocc in zip(self.mf.mo_energy, self.eri.nocc)))
        with temporary_env(self.mf, exxdiv=None):
            self.v_mf = self.mf.get_veff() - self.mf.get_j()

        # TD
        nroots, _, k1, k2, o, v = self.tdhf.xy.shape
        self.td_xy = self.tdhf.xy.transpose(0, 1, 2, 4, 3, 5).reshape(nroots, 2, k1*o, k2*v)
        self.td_e = self.tdhf.e

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
        x = sum(self.eri.nocc[:k]) + kp
        if kp < self.eri.nocc[k]:
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
        kind, p_plain = self.__plain_index__(p)
        # 1
        moe = self.mf.mo_energy[k][kp]
        # 2
        if kind == "o":
            vk = - numpy.trace(self["oooo"][p_plain, :, :, p_plain])
        else:
            vk = - numpy.trace(self["ovvo"][:, p_plain, p_plain, :])
        # 3
        v_mf = einsum("i,ij,j", self.mf.mo_coeff[k][:, kp].conj(), self.v_mf[k], self.mf.mo_coeff[k][:, kp])
        if components:
            return moe, vk, -v_mf
        else:
            return moe + vk - v_mf

    def get_sigma_element(self, omega, p, eta, vir_sgn=1):
        return super(IMDS, self).get_sigma_element(omega, self.__plain_index__(p, spec=False), eta, vir_sgn=vir_sgn)

    def initial_guess(self, p):
        k, kp = p
        return self.mf.mo_energy[k][kp]

    @property
    def entire_space(self):
        assert all(i == self.eri.nmo[0] for i in self.eri.nmo)
        return [numpy.arange(self.nk), numpy.arange(self.eri.nmo[0])]


kernel = gw_slow.kernel


class GW(gw_slow.GW):
    base_imds = IMDS

    def __init__(self, tdhf):
        """
        Performs GW calculation. Roots are stored in `self.mo_energy`.
        Args:
            tdhf (TDRHF): the base time-dependent restricted Hartree-Fock model;
        """
        super(GW, self).__init__(tdhf)

    def kernel(self):
        """
        Calculates GW roots.

        Returns:
            GW roots.
        """
        return super(GW, self).kernel()
