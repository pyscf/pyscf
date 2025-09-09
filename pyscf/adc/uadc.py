# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Abdelrahman Ahmed <>
#         Samragni Banerjee <samragnibanerjee4@gmail.com>
#         James Serna <jamcar456@gmail.com>
#         Terrence Stahl <terrencestahl1@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>

'''
Unrestricted algebraic diagrammatic construction
'''

import numpy as np
import pyscf.lib as lib
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
from pyscf.adc import uadc_amplitudes
from pyscf import __config__
from pyscf import df
from pyscf import scf


# Excited-state kernel
def kernel(adc, nroots=1, guess=None, eris=None, verbose=None):

    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if isinstance(adc._scf, scf.rohf.ROHF) and (adc.method_type == "ip" or adc.method_type == "ea"):
        logger.warn(
            adc, "EA/IP-ADC with the ROHF reference do not incorporate the occ-vir Fock matrix elements...")

    if eris is None:
        eris = adc.transform_integrals()

    if adc.approx_trans_moments:
        if adc.method in ("adc(2)", "adc(2)-x"):
            logger.warn(
                adc,
                "Approximations for transition moments are requested...\n"
                + adc.method
                + " transition properties will neglect second-order amplitudes...")
        else:
            logger.warn(
                adc,
                "Approximations for transition moments are requested...\n"
                + adc.method
                + " transition properties will neglect third-order amplitudes...")


    imds = adc.get_imds(eris)
    matvec, diag = adc.gen_matvec(imds, eris)

    if guess is None:
        guess = adc.get_init_guess(nroots, diag, ascending = True)
    elif guess=="cis" and adc.method_type == "ee":
        guess = adc.get_init_guess(nroots, diag, ascending = True, type="cis", eris=eris)
    else:
        raise NotImplementedError("Guess type not implemented")

    conv, adc.E, U = lib.linalg_helper.davidson1(
        lambda xs : [matvec(x) for x in xs],
        guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_memory=adc.max_memory,
        max_cycle=adc.max_cycle, max_space=adc.max_space, tol_residual=adc.tol_residual)

    adc.U = np.array(U).T.copy()

    if adc.compute_properties:
        adc.P,adc.X = adc.get_properties(nroots)
    else:
        adc.P = None
        adc.X = None

    nfalse = np.shape(conv)[0] - np.sum(conv)

    if adc.compute_spin_square:
        spin_square, evec_ne = adc.get_spin_square()

    header = ("\n*************************************************************"
              "\n                   ADC calculation summary"
              "\n*************************************************************")
    logger.info(adc, header)

    for n in range(nroots):
        print_string = ('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                        (adc.method, n, adc.E[n], adc.E[n]*27.2114))
        if adc.compute_properties:
            if (adc.method_type == "ee"):
                print_string += ("|  Osc. strength = %10.8f  " % adc.P[n])
                if (adc.compute_spin_square is True):
                    print_string += ("|  S^2 = %10.8f  " % spin_square[n])
            else:
                print_string += ("|  Spec. factor = %10.8f  " % adc.P[n])
        print_string += ("|  conv = %s" % conv[n])
        logger.info(adc, print_string)

    if nfalse >= 1:
        logger.warn(adc, "Davidson iterations for " + str(nfalse) + " root(s) did not converge!!!")

    log.timer('ADC', *cput0)

    return adc.E, adc.U, adc.P, adc.X


def make_ref_rdm1(adc):
    from pyscf.lib import einsum

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]
    t1_2_a = adc.t1[0][0][:]
    t1_2_b = adc.t1[0][1][:]

    ######################
    einsum_type = True
    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    rdm1_a  = np.zeros((nmo_a,nmo_a))
    rdm1_b  = np.zeros((nmo_b,nmo_b))

    ####### ADC(2) SPIN ADAPTED REF OPDM with SQA ################
    ### OCC-OCC ###
    rdm1_a[:nocc_a, :nocc_a]  = einsum('IJ->IJ', np.identity(nocc_a), optimize = einsum_type).copy()

    rdm1_b[:nocc_b, :nocc_b]  = einsum('IJ->IJ', np.identity(nocc_b), optimize = einsum_type).copy()

    rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('Iiab,Jiab->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('Iiab,Jiba->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('Iiab,iJab->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('Iiab,iJba->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('Jiab,iIab->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('Jiab,iIba->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('iIab,iJab->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('iIab,iJba->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] -= einsum('Iiab,Jiab->IJ', t2_1_ab, t2_1_ab, optimize = einsum_type)

    rdm1_b[:nocc_b, :nocc_b] -= einsum('iIab,iJab->IJ', t2_1_ab, t2_1_ab, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('Iiab,Jiab->IJ', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('Iiab,Jiba->IJ', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('Iiab,iJab->IJ', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('Iiab,iJba->IJ', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('Jiab,iIab->IJ', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('Jiab,iIba->IJ', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('iIab,iJab->IJ', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('iIab,iJba->IJ', t2_1_b, t2_1_b, optimize = einsum_type)


    ### OCC-VIR ###
    rdm1_a[:nocc_a, nocc_a:]  = einsum('IA->IA', t1_2_a, optimize = einsum_type).copy()

    rdm1_b[:nocc_b, nocc_b:]  = einsum('IA->IA', t1_2_b, optimize = einsum_type).copy()

    ### VIR-OCC ###
    rdm1_a[nocc_a:, :nocc_a]  = rdm1_a[:nocc_a, nocc_a:].T

    rdm1_b[nocc_b:, :nocc_b]  = rdm1_b[:nocc_b, nocc_b:].T

    ### VIR-VIR ###
    rdm1_a[nocc_a:, nocc_a:]  = 1/16 * einsum('ijAa,ijBa->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijAa,ijaB->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijAa,jiBa->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijAa,jiaB->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijBa,ijaA->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijBa,jiaA->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijaA,ijaB->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijaA,jiaB->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] += einsum('ijAa,ijBa->AB', t2_1_ab, t2_1_ab, optimize = einsum_type)

    rdm1_b[nocc_b:, nocc_b:]  = einsum('ijaA,ijaB->AB', t2_1_ab, t2_1_ab, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijAa,ijBa->AB', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijAa,ijaB->AB', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijAa,jiBa->AB', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijAa,jiaB->AB', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijBa,ijaA->AB', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijBa,jiaA->AB', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijaA,ijaB->AB', t2_1_b, t2_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijaA,jiaB->AB', t2_1_b, t2_1_b, optimize = einsum_type)

    ####### ADC(3) SPIN ADAPTED REF OPDM WITH SQA ################
    if adc.method == "adc(3)":
        t2_2_a = adc.t2[1][0][:]
        t2_2_ab = adc.t2[1][1][:]
        t2_2_b = adc.t2[1][2][:]
        t1_3_a = adc.t1[1][0][:]
        t1_3_b = adc.t1[1][1][:]

        #### OCC-OCC ###
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('Iiab,Jiab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('Iiab,Jiba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('Iiab,iJab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('Iiab,iJba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('Jiab,Iiab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('Jiab,Iiba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('Jiab,iIab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('Jiab,iIba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('iIab,Jiab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('iIab,Jiba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('iIab,iJab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('iIab,iJba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('iJab,Iiab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('iJab,Iiba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/16 * einsum('iJab,iIab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/16 * einsum('iJab,iIba->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= einsum('Iiab,Jiab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= einsum('Jiab,Iiab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)

        rdm1_b[:nocc_b, :nocc_b] -= einsum('iIab,iJab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= einsum('iJab,iIab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('Iiab,Jiab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('Iiab,Jiba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('Iiab,iJab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('Iiab,iJba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('Jiab,Iiab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('Jiab,Iiba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('Jiab,iIab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('Jiab,iIba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('iIab,Jiab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('iIab,Jiba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('iIab,iJab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('iIab,iJba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('iJab,Iiab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('iJab,Iiba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/16 * einsum('iJab,iIab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/16 * einsum('iJab,iIba->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        ##### OCC-VIR ### ####
        rdm1_a[:nocc_a, nocc_a:] += einsum('IA->IA', t1_3_a, optimize = einsum_type).copy()
        rdm1_a[:nocc_a, nocc_a:] += 1/8 * einsum('ia,IiAa->IA', t1_2_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] -= 1/8 * einsum('ia,IiaA->IA', t1_2_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] -= 1/8 * einsum('ia,iIAa->IA', t1_2_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/8 * einsum('ia,iIaA->IA', t1_2_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * einsum('ia,IiAa->IA', t1_2_b, t2_1_ab, optimize = einsum_type)

        rdm1_b[:nocc_b, nocc_b:] += einsum('IA->IA', t1_3_b, optimize = einsum_type).copy()
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * einsum('ia,iIaA->IA', t1_2_a, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/8 * einsum('ia,IiAa->IA', t1_2_b, t2_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] -= 1/8 * einsum('ia,IiaA->IA', t1_2_b, t2_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] -= 1/8 * einsum('ia,iIAa->IA', t1_2_b, t2_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/8 * einsum('ia,iIaA->IA', t1_2_b, t2_1_b, optimize = einsum_type)

        ###### VIR-OCC ###
        rdm1_a[nocc_a:, :nocc_a] = rdm1_a[:nocc_a, nocc_a:].T

        rdm1_b[nocc_b:, :nocc_b] = rdm1_b[:nocc_b, nocc_b:].T

        ##### VIR=VIR ###
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijAa,ijBa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijAa,ijaB->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijAa,jiBa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijAa,jiaB->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijBa,ijAa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijBa,ijaA->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijBa,jiAa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijBa,jiaA->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijaA,ijBa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijaA,ijaB->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijaA,jiBa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijaA,jiaB->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijaB,ijAa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijaB,ijaA->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/16 * einsum('ijaB,jiAa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] -= 1/16 * einsum('ijaB,jiaA->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += einsum('ijAa,ijBa->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += einsum('ijBa,ijAa->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)

        rdm1_b[nocc_b:, nocc_b:] += einsum('ijaA,ijaB->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += einsum('ijaB,ijaA->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijAa,ijBa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijAa,ijaB->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijAa,jiBa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijAa,jiaB->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijBa,ijAa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijBa,ijaA->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijBa,jiAa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijBa,jiaA->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijaA,ijBa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijaA,ijaB->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijaA,jiBa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijaA,jiaB->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijaB,ijAa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijaB,ijaA->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/16 * einsum('ijaB,jiAa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] -= 1/16 * einsum('ijaB,jiaA->AB', t2_1_b, t2_2_b, optimize = einsum_type)

    return (rdm1_a, rdm1_b)

def get_fno_ref(myadc,nroots,ref_state):
    adc2_ref = UADC(myadc._scf,myadc.frozen).set(verbose = 0,method_type = myadc.method_type,with_df = myadc.with_df,if_naf = myadc.if_naf,thresh_naf = myadc.thresh_naf)
    myadc.e2_ref,myadc.v2_ref,myadc.p2_ref,myadc.x2_ref = adc2_ref.kernel(nroots)
    rdm1_gs = adc2_ref.make_ref_rdm1()
    if ref_state is not None:
        rdm1_gs_a = rdm1_gs[0]
        rdm1_gs_b = rdm1_gs[1]
        rdm1_ref_a = adc2_ref.make_rdm1()[0][ref_state - 1]
        rdm1_ref_b = adc2_ref.make_rdm1()[1][ref_state - 1]
        myadc.rdm1_ss = (rdm1_ref_a + rdm1_gs_a, rdm1_ref_b + rdm1_gs_b)
    else:
        myadc.rdm1_ss = rdm1_gs

def make_fno(myadc, rdm1_ss, mf, thresh):
    nocc_a = myadc.nocc_a
    nocc_b = myadc.nocc_b
    nvir_a = myadc.nvir_a
    nvir_b = myadc.nvir_b
    mo_energy_a = myadc.mo_energy_a
    mo_energy_b = myadc.mo_energy_b
    mo_a_coeff = myadc.mo_coeff[0]
    mo_b_coeff = myadc.mo_coeff[1]
    masks = myadc._mo_splitter()
    mask_a = masks[0]
    mask_b = masks[1]
    rdm1_ss_a = rdm1_ss[0]
    rdm1_ss_b = rdm1_ss[1]

    n_a,V_a = np.linalg.eigh(rdm1_ss_a[nocc_a:,nocc_a:])
    idx = np.argsort(n_a)[::-1]
    n_a,V_a = n_a[idx], V_a[:,idx]
    T_a = n_a > thresh
    n_fro_vir_a = np.sum(T_a == 0)
    T_a = np.diag(T_a)
    V_trunc_a = V_a.dot(T_a)
    n_keep_a = V_trunc_a.shape[0]-n_fro_vir_a

    n_b,V_b = np.linalg.eigh(rdm1_ss_b[nocc_b:,nocc_b:])
    idx = np.argsort(n_b)[::-1]
    n_b,V_b = n_b[idx], V_b[:,idx]
    T_b = n_b > thresh
    n_fro_vir_b = np.sum(T_b == 0)
    T_b = np.diag(T_b)
    V_trunc_b = V_b.dot(T_b)
    n_keep_b = V_trunc_b.shape[0]-n_fro_vir_b

    moeoccfrz0_a, moeocc_a, moevir_a, moevirfrz0_a = [mo_energy_a[m] for m in mask_a]
    orboccfrz0_a, orbocc_a, orbvir_a, orbvirfrz0_a = [mo_a_coeff[:,m] for m in mask_a]
    F_can_a =  np.diag(moevir_a)
    F_na_trunc_a = V_trunc_a.T.dot(F_can_a).dot(V_trunc_a)
    _,Z_na_trunc_a = np.linalg.eigh(F_na_trunc_a[:n_keep_a,:n_keep_a])
    U_vir_act_a = orbvir_a.dot(V_trunc_a[:,:n_keep_a]).dot(Z_na_trunc_a)
    U_vir_fro_a = orbvir_a.dot(V_trunc_a[:,n_keep_a:])

    moeoccfrz0_b, moeocc_b, moevir_b, moevirfrz0_b = [mo_energy_b[m] for m in mask_b]
    orboccfrz0_b, orbocc_b, orbvir_b, orbvirfrz0_b = [mo_b_coeff[:,m] for m in mask_b]
    F_can_b =  np.diag(moevir_b)
    F_na_trunc_b = V_trunc_b.T.dot(F_can_b).dot(V_trunc_b)
    _,Z_na_trunc_b = np.linalg.eigh(F_na_trunc_b[:n_keep_b,:n_keep_b])
    U_vir_act_b = orbvir_b.dot(V_trunc_b[:,:n_keep_b]).dot(Z_na_trunc_b)
    U_vir_fro_b = orbvir_b.dot(V_trunc_b[:,n_keep_b:])

    no_comp_a = (orboccfrz0_a,orbocc_a,U_vir_act_a,U_vir_fro_a,orbvirfrz0_a)
    no_coeff_a = np.hstack(no_comp_a)
    nocc_loc_a = np.cumsum([0]+[x.shape[1] for x in no_comp_a]).astype(int)
    no_frozen_a = np.hstack((np.arange(nocc_loc_a[0], nocc_loc_a[1]),
                              np.arange(nocc_loc_a[3], nocc_loc_a[5]))).astype(int)

    no_comp_b = (orboccfrz0_b,orbocc_b,U_vir_act_b,U_vir_fro_b,orbvirfrz0_b)
    no_coeff_b = np.hstack(no_comp_b)
    nocc_loc_b = np.cumsum([0]+[x.shape[1] for x in no_comp_b]).astype(int)
    no_frozen_b = np.hstack((np.arange(nocc_loc_b[0], nocc_loc_b[1]),
                              np.arange(nocc_loc_b[3], nocc_loc_b[5]))).astype(int)

    no_coeff = (no_coeff_a,no_coeff_b)
    no_frozen = (no_frozen_a,no_frozen_b)

    return no_coeff,no_frozen

def get_frozen_mask(myadc):
    moidx = (np.ones(myadc.mo_occ[0].size, dtype=bool),np.ones(myadc.mo_occ[1].size, dtype=bool))
    if myadc.frozen is None:
        pass
    elif isinstance(myadc.frozen, tuple) and len(myadc.frozen) == 2:
        if myadc.frozen[0] is None:
            pass
        elif isinstance(myadc.frozen[0], (int, np.integer)):
            moidx[0][:myadc.frozen[0]] = False
        elif hasattr(myadc.frozen[0], '__len__'):
            moidx[0][list(myadc.frozen[0])] = False
        else:
            raise NotImplementedError
        if myadc.frozen[1] is None:
            pass
        elif isinstance(myadc.frozen[1], (int, np.integer)):
            moidx[1][:myadc.frozen[1]] = False
        elif hasattr(myadc.frozen[1], '__len__'):
            moidx[1][list(myadc.frozen[1])] = False
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return moidx

def _mo_splitter(myadc):
    maskact = myadc.get_frozen_mask()
    maskact_a = maskact[0]
    maskact_b = maskact[1]
    maskocc_a = myadc.mo_occ[0]>1e-6
    maskocc_b = myadc.mo_occ[1]>1e-6
    masks_a = [
        maskocc_a & ~maskact_a,    # frz occ
        maskocc_a &  maskact_a,    # act occ
        ~maskocc_a &  maskact_a,    # act vir
        ~maskocc_a & ~maskact_a,    # frz vir
    ]
    masks_b = [
        maskocc_b & ~maskact_b,    # frz occ
        maskocc_b &  maskact_b,    # act occ
        ~maskocc_b &  maskact_b,    # act vir
        ~maskocc_b & ~maskact_b,    # frz vir
    ]
    return (masks_a, masks_b)

class UADC(lib.StreamObject):
    '''Ground state calculations

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> myadc = adc.UADC(mf).run()

    Saved results

        e_corr : float
            MPn correlation correction
        e_tot : float
            Total energy (HF + correlation)
        t1, t2 :
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    incore_complete = getattr(__config__, 'adc_uadc_UADC_incore_complete', False)

    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff',
        'mol', 'mo_energy_a', 'mo_energy_b', 'incore_complete',
        'scf_energy', 'e_tot', 't1', 't2', 'frozen', 'chkfile',
        'max_space', 'mo_occ', 'max_cycle', 'imds', 'with_df', 'compute_properties',
        'approx_trans_moments', 'evec_print_tol', 'spec_factor_print_tol',
        'E', 'U', 'P', 'X', 'ncvs', 'dip_mom', 'dip_mom_nuc',
        'compute_spin_square', 'f_ov',
        'nocc_a', 'nocc_b', 'nvir_a', 'nvir_b',
        'if_heri_eris', 'if_naf', 'thresh_naf', 'naux'
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        if 'dft' in str(mf.__module__):
            raise NotImplementedError('DFT reference for UADC')

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'adc_uadc_UADC_max_space', 200)
        self.max_cycle = getattr(__config__, 'adc_uadc_UADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_uadc_UADC_conv_tol', 1e-8)
        self.tol_residual = getattr(__config__, 'adc_uadc_UADC_tol_residual', 1e-5)
        self.scf_energy = mf.e_tot

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.f_ov = None

        if isinstance(mf, scf.rohf.ROHF):

            logger.info(mf, "\nROHF reference detected in ADC, semicanonicalizing the orbitals...")

            mo_a = mo_coeff.copy()
            nalpha = mf.mol.nelec[0]
            nbeta = mf.mol.nelec[1]

            h1e = mf.get_hcore()
            dm = mf.make_rdm1()
            vhf = mf.get_veff(mf.mol, dm)

            fock_a = h1e + vhf[0]
            fock_b = h1e + vhf[1]

            if nalpha > nbeta:
                ndocc = nbeta
                nsocc = nalpha - nbeta
            else:
                ndocc = nalpha
                nsocc = nbeta - nalpha

            fock_a = np.dot(mo_a.T,np.dot(fock_a, mo_a))
            fock_b = np.dot(mo_a.T,np.dot(fock_b, mo_a))

            # Semicanonicalize Ca using fock_a, nocc_a -> Ca, mo_energy_a, U_a, f_ov_a
            mo_a_coeff, mo_energy_a, f_ov_a, f_aa = self.semi_canonicalize_orbitals(
                fock_a, ndocc + nsocc, mo_a)

            # Semicanonicalize Cb using fock_b, nocc_b -> Cb, mo_energy_b, U_b, f_ov_b
            mo_b_coeff, mo_energy_b, f_ov_b, f_bb = self.semi_canonicalize_orbitals(fock_b, ndocc, mo_a)

            mo_coeff = [mo_a_coeff, mo_b_coeff]

            f_ov = [f_ov_a, f_ov_b]

            self.f_ov = f_ov
            self.mo_energy_a = mo_energy_a.copy()
            self.mo_energy_b = mo_energy_b.copy()

        else:
            self.mo_energy_a = mf.mo_energy[0]
            self.mo_energy_b = mf.mo_energy[1]

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.imds = lambda:None
        self._nocc = mf.nelec
#NOTE:update start
        self.naux = None
        self.if_naf = False
        self.thresh_naf = 1e-2        
        self.if_heri_eris = False
        self._nmo = None
        if frozen is None:
            self._nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
        elif isinstance(frozen, tuple) and len(frozen) == 2:
            if frozen[0] is None:
                nmo_a = mo_coeff[0].shape[1]
            elif isinstance(frozen[0], (int, np.integer)):
                nmo_a = mo_coeff[0].shape[1]-frozen[0]
            elif hasattr(frozen[0], '__len__'):
                nmo_a = mo_coeff[0].shape[1]-len(frozen[0])
            else:
                raise NotImplementedError
            if frozen[1] is None:
                nmo_b = mo_coeff[1].shape[1]
            elif isinstance(frozen[1], (int, np.integer)):
                nmo_b = mo_coeff[1].shape[1]-frozen[1]
            elif hasattr(frozen[1], '__len__'):
                nmo_b = mo_coeff[1].shape[1]-len(frozen[1])
            else:
                raise NotImplementedError
            self._nmo = (nmo_a, nmo_b)
        else:
            raise NotImplementedError("frozen should be announced as None or a tuple with two elements")

        (mask_a,mask_b) = self.get_frozen_mask()
        if frozen is not None:
            maskocc_a = mf.mo_occ[0]>1e-6
            maskocc_b = mf.mo_occ[1]>1e-6
            occ_a = maskocc_a & mask_a
            occ_b = maskocc_b & mask_b
            self._nocc = (int(occ_a.sum()), int(occ_b.sum()))
            self.mo_coeff = (mo_coeff[0][:,mask_a], mo_coeff[1][:,mask_b])
            if self._nocc[0] == 0 or self._nocc[1] == 0:
                raise ValueError("No occupied alpha or beta orbitals found")

        if self.mo_coeff is self._scf.mo_coeff and self._scf.converged:
            self.mo_energy_a = self.mo_energy_a[mask_a]
            self.mo_energy_b = self.mo_energy_b[mask_b]
        else:
            dm = self._scf.make_rdm1(mo_coeff, self.mo_occ)
            vhf = self._scf.get_veff(self.mol, dm)
            fockao_a, fockao_b = self._scf.get_fock(vhf=vhf, dm=dm)
            fock_a = self.mo_coeff[0].conj().T.dot(fockao_a).dot(self.mo_coeff[0])
            fock_b = self.mo_coeff[1].conj().T.dot(fockao_b).dot(self.mo_coeff[1])
            (self.mo_energy_a,self.mo_energy_b) = (fock_a.diagonal().real,fock_b.diagonal().real)
            self.scf_energy = self._scf.energy_tot(dm=dm, vhf=vhf)
#NOTE:update end
        self._nvir = (self._nmo[0] - self._nocc[0], self._nmo[1] - self._nocc[1])
        self.nocc_a = self._nocc[0]
        self.nocc_b = self._nocc[1]
        self.nvir_a = self._nvir[0]
        self.nvir_b = self._nvir[1]
        self.chkfile = mf.chkfile
        self.method = "adc(2)"
        self.method_type = "ip"
        self.with_df = None
        self.compute_properties = True
        self.approx_trans_moments = False
        self.evec_print_tol = 0.1
        self.spec_factor_print_tol = 0.1
        self.ncvs = None

        self.E = None
        self.U = None
        self.P = None
        self.X = (None,)

        self.compute_spin_square = False

        dip_ints = -self.mol.intor('int1e_r',comp=3)
        dip_mom_a = np.zeros((dip_ints.shape[0], self._nmo[0], self._nmo[0]))
        dip_mom_b = np.zeros((dip_ints.shape[0], self._nmo[1], self._nmo[1]))

        for i in range(dip_ints.shape[0]):
            dip = dip_ints[i,:,:]
            dip_mom_a[i,:,:] = np.dot(self.mo_coeff[0].T, np.dot(dip, self.mo_coeff[0]))
            dip_mom_b[i,:,:] = np.dot(self.mo_coeff[1].T, np.dot(dip, self.mo_coeff[1]))

        self.dip_mom = []
        self.dip_mom.append(dip_mom_a)
        self.dip_mom.append(dip_mom_b)

        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        self.dip_mom_nuc = lib.einsum('i,ix->x', charges, coords)

    compute_amplitudes = uadc_amplitudes.compute_amplitudes
    compute_energy = uadc_amplitudes.compute_energy
    transform_integrals = uadc_ao2mo.transform_integrals_incore
    make_ref_rdm1 = make_ref_rdm1
    get_frozen_mask = get_frozen_mask
    _mo_splitter = _mo_splitter


    def semi_canonicalize_orbitals(self, f, nocc, C):

        # Diagonalize occ-occ block
        evals_oo, evecs_oo = np.linalg.eigh(f[:nocc, :nocc])

        # Diagonalize virt-virt block
        evals_vv, evecs_vv = np.linalg.eigh(f[nocc:, nocc:])

        evals = np.hstack((evals_oo, evals_vv))

        U = np.zeros_like(f)

        U[:nocc, :nocc] = evecs_oo
        U[nocc:, nocc:] = evecs_vv

        C = np.dot(C, U)

        transform_f = np.dot(U.T, np.dot(f, U))
        f_ov = transform_f[:nocc, nocc:].copy()

        return C, evals, f_ov, transform_f

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'tol_residual = %s', self.tol_residual)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def dump_flags_gs(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def kernel_gs(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo_a, nmo_b = self._nmo
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmo_a * (nmo_a+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo_a**4) + nmo_pair**2) * 2 * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):
            if getattr(self, 'with_df', None):
                self.with_df = self.with_df
            else:
                self.with_df = self._scf.with_df

            def df_transform():
                return uadc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (self._scf._eri is None or
              (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
            def outcore_transform():
                return uadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = uadc_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose)
        self._finalize()

        return self.e_corr, self.t1, self.t2

    def kernel(self, nroots=1, guess=None, eris=None):
        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo_a, nmo_b = self._nmo
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmo_a * (nmo_a+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo_a**4) + nmo_pair**2) * 2 * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):
            if getattr(self, 'with_df', None):
                self.with_df = self.with_df
            else:
                self.with_df = self._scf.with_df

            def df_transform():
                return uadc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (self._scf._eri is None or
              (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
            def outcore_transform():
                return uadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = uadc_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose)
        self._finalize()

        self.method_type = self.method_type.lower()
        if (self.method_type == "ea"):
            e_exc, v_exc, spec_fac, X, adc_es = self.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif (self.method_type == "ee"):
            e_exc, v_exc, spec_fac, X, adc_es = self.ee_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):

            if not isinstance(self.ncvs, type(None)) and self.ncvs > 0:
                e_exc, v_exc, spec_fac, X, adc_es = self.ip_cvs_adc(
                    nroots=nroots, guess=guess, eris=eris)
            else:
                e_exc, v_exc, spec_fac, X, adc_es = self.ip_adc(
                    nroots=nroots, guess=guess, eris=eris)
        else:
            raise NotImplementedError(self.method_type)

        self._adc_es = adc_es
        if self.if_heri_eris:
            if self.if_naf:
                return e_exc, v_exc, spec_fac, X, eris, self.naux
            else:
                eris.vvvv = None
                return e_exc, v_exc, spec_fac, X, eris
        else:
            return e_exc, v_exc, spec_fac, X

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'MP%s correlation energy of reference state (a.u.) = %.8f',
                    self.method[4], self.e_corr)
        return self

    def ea_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import uadc_ea
        adc_es = uadc_ea.UADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ee_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import uadc_ee
        adc_es = uadc_ee.UADCEE(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import uadc_ip
        adc_es = uadc_ip.UADCIP(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_cvs_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import uadc_ip_cvs
        adc_es = uadc_ip_cvs.UADCIPCVS(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def density_fit(self, auxbasis=None, with_df=None):
        if with_df is None:
            self.with_df = df.DF(self._scf.mol)
            self.with_df.max_memory = self.max_memory
            self.with_df.stdout = self.stdout
            self.with_df.verbose = self.verbose
            if auxbasis is None:
                self.with_df.auxbasis = self._scf.with_df.auxbasis
            else:
                self.with_df.auxbasis = auxbasis
        else:
            self.with_df = with_df
        return self

    def analyze(self):
        self._adc_es.analyze()

    def compute_dyson_mo(self):
        return self._adc_es.compute_dyson_mo()

    def make_rdm1(self):
        return self._adc_es.make_rdm1()

class UFNOADC3(UADC):
    #J. Chem. Phys. 159, 084113 (2023)
    _keys = UADC._keys | {'delta_e','delta_v','delta_p','delta_x',
                          'e2_ref','v2_ref','p2_ref','x2_ref',
                          'rdm1_ss','correction','frozen_core','ref_state'
                          }

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, correction=True):
        import copy
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.delta_e = None
        self.delta_v = None
        self.delta_p = None
        self.delta_x = None
        self.method = "adc(3)"
        self.correction = correction
        self.e2_ref = None
        self.v2_ref = None
        self.p2_ref = None
        self.x2_ref = None
        self.rdm1_ss = None
        self.ref_state = None
        self.if_naf = True
        self.frozen_core = copy.deepcopy(self.frozen)

    def compute_correction(self, mf, frozen, nroots, eris=None):
        adc2_ssfno = UADC(mf, frozen, self.mo_coeff).set(verbose = 0,method_type = self.method_type,\
                                                         with_df = self.with_df,if_naf = self.if_naf,thresh_naf = self.thresh_naf,naux = self.naux)
        e2_ssfno,v2_ssfno,p2_ssfno,x2_ssfno = adc2_ssfno.kernel(nroots, eris = eris)
        self.delta_e = self.e2_ref - e2_ssfno
        #self.delta_v = self.v2_ref - v2_ssfno
        #self.delta_p = self.p2_ref - p2_ssfno
        #self.delta_x = self.x2_ref - x2_ssfno

    def kernel(self, nroots=1, guess=None, eris=None, thresh = 1e-4, ref_state = None):
        import copy
        self.frozen = copy.deepcopy(self.frozen_core)
        self.ref_state = ref_state
        self.naux = None
        if ref_state is None:
            print("Do fnoadc3 calculation")
            self.if_naf = False
        elif isinstance(ref_state, int) and 0<ref_state<=nroots:
            print(f"Do ss-fno adc3 calculation, the specic state is {ref_state}")
            if self.with_df == None:
                self.if_naf = False
        else:
            raise ValueError("ref_state should be an int type and in (0,nroots]")
        
        get_fno_ref(self, nroots, self.ref_state)
        self.mo_coeff,self.frozen = make_fno(self, self.rdm1_ss, self._scf, thresh)
        adc3_ssfno = UADC(self._scf, self.frozen, self.mo_coeff).set(verbose = self.verbose,method_type = self.method_type,method = "adc(3)",\
                                                         with_df = self.with_df,if_naf = self.if_naf,thresh_naf = self.thresh_naf,\
                                                            if_heri_eris = self.if_heri_eris)
        if not self.correction or self.if_heri_eris is False:
            e_exc, v_exc, spec_fac, x = adc3_ssfno.kernel(nroots, guess, eris)
            if self.correction:
                self.compute_correction(self._scf, self.frozen, nroots)
                e_exc = e_exc + self.delta_e
        elif self.if_naf:
            e_exc, v_exc, spec_fac, x, eris, self.naux = adc3_ssfno.kernel(nroots, guess, eris)
            self.compute_correction(self._scf, self.frozen, nroots, eris)
            e_exc = e_exc + self.delta_e
        else:
            e_exc, v_exc, spec_fac, x, eris = adc3_ssfno.kernel(nroots, guess, eris)
            self.compute_correction(self._scf, self.frozen, nroots, eris)
            e_exc = e_exc + self.delta_e
        v_exc = v_exc# + self.delta_v
        spec_fac = spec_fac# + self.delta_p
        x = x# + self.delta_x
        return e_exc, v_exc, spec_fac, x

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import adc

    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', (0., 0.    , -r/2   )],
        ['N', (0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr -  -0.32201692499346535)

    myadcip = adc.uadc_ip.UADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(2) IP energies")
    print (e[0] - 0.5434389897908212)
    print (e[1] - 0.5434389942222756)
    print (e[2] - 0.6240296265084732)

    print("ADC(2) IP spectroscopic factors")
    print (p[0] - 0.884404855445607)
    print (p[1] - 0.8844048539643351)
    print (p[2] - 0.9096460559671828)

    myadcea = adc.uadc_ea.UADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)
    print("ADC(2) EA energies")
    print (e[0] - 0.09617819143037348)
    print (e[1] - 0.09617819161265123)
    print (e[2] - 0.12583269048810924)

    print("ADC(2) EA spectroscopic factors")
    print (p[0] - 0.991642716974455)
    print (p[1] - 0.9916427170555298)
    print (p[2] - 0.9817184409336244)

    myadc = adc.ADC(mf)
    myadc.method = "adc(3)"
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr - -0.31694173142858517)

    myadcip = adc.uadc_ip.UADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(3) IP energies")
    print (e[0] - 0.5667526838174817)
    print (e[1] - 0.5667526888293601)
    print (e[2] - 0.6099995181296374)

    print("ADC(3) IP spectroscopic factors")
    print (p[0] - 0.9086596203469742)
    print (p[1] - 0.9086596190173993)
    print (p[2] - 0.9214613318791076)

    myadcea = adc.uadc_ea.UADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)

    print("ADC(3) EA energies")
    print (e[0] - 0.09836545519235675)
    print (e[1] - 0.09836545535587536)
    print (e[2] - 0.12957093060942082)

    print("ADC(3) EA spectroscopic factors")
    print (p[0] - 0.9920495578633931)
    print (p[1] - 0.992049557938337)
    print (p[2] - 0.9819274864738444)

    myadc.method = "adc(2)-x"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x IP energies")
    print (e[0] - 0.5405255355249104)
    print (e[1] - 0.5405255399061982)
    print (e[2] - 0.62080267098272)
    print (e[3] - 0.620802670982715)

    myadc.method_type = "ea"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x EA energies")
    print (e[0] - 0.09530653292650725)
    print (e[1] - 0.09530653311305577)
    print (e[2] - 0.1238833077840878)
    print (e[3] - 0.12388330873739162)
