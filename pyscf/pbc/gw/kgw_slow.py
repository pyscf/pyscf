"""
This module implements the G0W0 approximation on top of `pyscf.tdscf.rhf_slow` TDHF implementation. Unlike `gw.py`, all
integrals are stored in memory. Several variants of GW are available:

 * `pyscf.gw_slow`: the molecular implementation;
 * `pyscf.pbc.gw.gw_slow`: single-kpoint PBC (periodic boundary condition) implementation;
 * `pyscf.pbc.gw.kgw_slow_supercell`: a supercell approach to PBC implementation with multiple k-points. Runs the
   molecular code for a model with several k-points for the cost of discarding momentum conservation and using dense
   instead of sparse matrixes;
 * (this module) `pyscf.pbc.gw.kgw_slow`: a PBC implementation with multiple k-points;
"""

from pyscf.gw import gw_slow
from pyscf.pbc.gw import kgw_slow_supercell
from pyscf.lib import einsum, direct_sum
from pyscf.pbc.tdscf.krhf_slow import get_block_k_ix

import numpy
from itertools import product


# Convention for these modules:
# * IMDS contains routines for intermediates
# * kernel finds GW roots
# * GW provides a container


class IMDS(kgw_slow_supercell.IMDS):
    def __init__(self, tdhf):
        """
        GW intermediates (k-version).
        Args:
            tdhf (TDRHF): the TDRHF contatainer;
        """
        gw_slow.AbstractIMDS.__init__(self, tdhf)

        self.nk = len(self.mf.kpts)

        # MF
        self.nocc = self.eri.nocc
        self.o = tuple(e[:nocc] for e, nocc in zip(self.mf.mo_energy, self.eri.nocc))
        self.v = tuple(e[nocc:] for e, nocc in zip(self.mf.mo_energy, self.eri.nocc))

        # TD
        self.td_xy = self.tdhf.xy
        self.td_e = self.tdhf.e

        self.tdm = self.construct_tdm()

    def eri_ov(self, item):
        item, k1, k2, k3, k4 = item
        return self.eri.eri_ov(item, (k1, k2, k3, k4)) / len(self.mf.kpts)

    __getitem__ = eri_ov

    def get_rhs(self, p, components=False):
        k, kp = p
        # 1
        moe = self.mf.mo_energy[k][kp]
        # 2
        if kp < self.nocc[k]:
            vk = - sum(
                numpy.trace(self["oooo", k, ki, ki, k][kp, :, :, kp])
                for ki in range(self.nk)
            )
        else:
            vk = - sum(
                numpy.trace(self["ovvo", ki, k, k, ki][:, kp-self.nocc[k], kp-self.nocc[k], :])
                for ki in range(self.nk)
            )
        # 3
        v_mf = self.mf.get_veff() - self.mf.get_j()
        v_mf = einsum("i,ij,j", self.mf.mo_coeff[k][:, kp].conj(), v_mf[k], self.mf.mo_coeff[k][:, kp])
        if components:
            return moe, vk, -v_mf
        else:
            return moe + vk - v_mf

    def construct_vmf(self):
        v_mf = self.mf.get_veff() - self.mf.get_j()
        return list(
            einsum("ia,ij,ja->a", mo.conj(), v, mo)
            for mo, v in zip(self.mf.mo_coeff, v_mf)
        )

    def construct_vk(self):
        result = []
        for k in range(self.nk):
            result.append( - numpy.concatenate([
                sum(einsum('piip->p', self["oooo", k, ki, ki, k]) for ki in range(self.nk)),
                sum(einsum('ippi->p', self["ovvo", ki, k, k, ki]) for ki in range(self.nk)),
            ]))
        return result

    def construct_tdm(self):

        # Indexes of td_x:
        # - k_transfer
        # - k
        # - o: k
        # - v: fw[k]

        # Indexes of td_y:
        # - k_transfer
        # - k
        # - o: k
        # - v: bw[k]

        # Original code:
        # tdm_oo = einsum('vxia,ipaq->vxpq', td_xy, self["oovo"])
        # tdm_ov = einsum('vxia,ipaq->vxpq', td_xy, self["oovv"])
        # tdm_vv = einsum('vxia,ipaq->vxpq', td_xy, self["ovvv"])

        # ERI k:
        # ki, kp, ka=?, kq

        # Now fw[kp] = kq, bw[kq] = kp -> bw[ki] = ka, fw[ka] = ki
        # for x amplitudes, the transfer is bw[0] such that ov -> k, bw[k]
        # for y amplitudes, the transfer is also fw[0] such that ov -> k, bw[k]

        result = []

        for k_transfer in range(self.nk):
            xy_k = self.td_xy[k_transfer]

            fw, bw, _, _ = get_block_k_ix(self.eri, k_transfer)
            result.append([[], []])

            for xy_kx, ix_fw, ix_bw, storage in (
                (xy_k[:, 0], fw, bw, result[k_transfer][0]),  # X
                (xy_k[:, 1], bw, fw, result[k_transfer][1]),  # Y
            ):

                for kp in range(self.nk):

                    tdm_oo = tdm_ov = tdm_vv = 0

                    tdm_vo = 0

                    for ki in range(self.nk):

                        x = xy_kx[:, ki] * 2
                        tdm_oo = tdm_oo + einsum('via,ipaq->vpq', x, self["oovo", ki, kp, ix_fw[ki], ix_bw[kp]])
                        tdm_ov = tdm_ov + einsum('via,ipaq->vpq', x, self["oovv", ki, kp, ix_fw[ki], ix_bw[kp]])
                        tdm_vv = tdm_vv + einsum('via,ipaq->vpq', x, self["ovvv", ki, kp, ix_fw[ki], ix_bw[kp]])
                        tdm_vo = tdm_vo + einsum('via,ipaq->vpq', x, self["ovvo", ki, kp, ix_fw[ki], ix_bw[kp]])

                    tdm = numpy.concatenate(
                        (
                            numpy.concatenate((tdm_oo, tdm_ov), axis=2),
                            numpy.concatenate((tdm_vo, tdm_vv), axis=2)
                        ),
                        axis=1,
                    )

                    storage.append(tdm)
        # The output is the following:
        # for each kp, kq pair two 3-tensors are given
        # The last two indexes in each tensor correspond to kp, kq
        # Given fw, bw, _, _ = get_block_k_ix(self.eri, (kp, kq)),
        # The first index of the two tensors will correspond to tdhf.e[bw[0]], tdhf.e[fw[0]] correspondingly
        return numpy.array(result)

    def get_sigma_element(self, omega, p, eta, vir_sgn=1):
        k, p = p

        # Molecular implementation
        # ------------------------
        # tdm = self.tdm.sum(axis=1)
        # evi = direct_sum('v-i->vi', self.td_e, self.o)
        # eva = direct_sum('v+a->va', self.td_e, self.v)
        # sigma = numpy.sum(tdm[:, :self.nocc, p] ** 2 / (omega + evi - 1j * eta))
        # sigma += numpy.sum(tdm[:, self.nocc:, p] ** 2 / (omega - eva + vir_sgn * 1j * eta))
        # return sigma

        sigma = 0
        for k_transfer, tdm_k in enumerate(self.tdm):
            fw, bw, _, _ = get_block_k_ix(self.eri, k_transfer)

            same = fw == bw
            different = numpy.logical_not(same)

            terms = []

            if same.sum() > 0:
                terms.append((tdm_k[0] + tdm_k[1], fw[same], fw[same]))

            if different.sum() > 0:
                terms.append((tdm_k[0], bw[different], fw[different]))
                terms.append((tdm_k[1], fw[different], bw[different]))

            for tdm_kx, ix_fw, ix_bw in terms:
                k1, k2 = ix_bw[k], k
                evi = direct_sum('v-i->vi', self.td_e[k_transfer], self.o[k1])
                eva = direct_sum('v+a->va', self.td_e[k_transfer], self.v[k1])
                sigma += numpy.sum(tdm_kx[k1][:, :self.nocc[k1], p] ** 2 / (omega + evi - 1j * eta))
                sigma += numpy.sum(tdm_kx[k1][:, self.nocc[k1]:, p] ** 2 / (omega - eva + vir_sgn * 1j * eta))
        return sigma


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
