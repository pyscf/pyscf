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
#         Ning-Yuan Chen <cny003@outlook.com>
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
from pyscf.data.nist import HARTREE2EV


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
    elif isinstance(guess, str) and guess == "cis" and adc.method_type == "ee":
        guess = adc.get_init_guess(nroots, diag, ascending = True, type="cis", eris=eris)
    elif isinstance(guess, np.ndarray) or isinstance(guess, list):
        guess = adc.get_init_guess(nroots, diag, ascending = True, type = "read", ini = guess)
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
                        (adc.method, n, adc.E[n], adc.E[n]*HARTREE2EV))
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


def make_ref_rdm1(adc, with_frozen=True, ao_repr=False):
    from pyscf.lib import einsum

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]

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

    if adc.f_ov is None:
        t1_1_a = np.zeros((nocc_a, nvir_a))
        t1_1_b = np.zeros((nocc_b, nvir_b))
    else:
        t1_1_a = adc.t1[2][0][:]
        t1_1_b = adc.t1[2][1][:]

    if adc.t1[0][0] is not None:
        t1_2_a = adc.t1[0][0][:]
        t1_2_b = adc.t1[0][1][:]
    else:
        t1_2_a = np.zeros((nocc_a, nvir_a))
        t1_2_b = np.zeros((nocc_b, nvir_b))

    ####### ADC(2) SPIN ADAPTED REF OPDM with SQA ################
    ### OCC-OCC ###
    rdm1_a[:nocc_a, :nocc_a]  = einsum('IJ->IJ', np.identity(nocc_a), optimize = einsum_type).copy()

    rdm1_b[:nocc_b, :nocc_b]  = einsum('IJ->IJ', np.identity(nocc_b), optimize = einsum_type).copy()

    rdm1_a[:nocc_a, :nocc_a] -= einsum('Ia,Ja->IJ', t1_1_a, t1_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] -= 1/2 * einsum('Iiab,Jiab->IJ', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, :nocc_a] -= einsum('Iiab,Jiab->IJ', t2_1_ab, t2_1_ab, optimize = einsum_type)

    rdm1_b[:nocc_b, :nocc_b] -= einsum('Ia,Ja->IJ', t1_1_b, t1_1_b, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] -= einsum('iIab,iJab->IJ', t2_1_ab, t2_1_ab, optimize = einsum_type)
    rdm1_b[:nocc_b, :nocc_b] -= 1/2 * einsum('Iiab,Jiab->IJ', t2_1_b, t2_1_b, optimize = einsum_type)

    ### OCC-VIR ###
    rdm1_a[:nocc_a, nocc_a:]  = einsum('IA->IA', t1_1_a, optimize = einsum_type).copy()
    rdm1_a[:nocc_a, nocc_a:] += einsum('IA->IA', t1_2_a, optimize = einsum_type)
    rdm1_a[:nocc_a, nocc_a:] += 1/2 * einsum('ia,IiAa->IA', t1_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[:nocc_a, nocc_a:] += 1/2 * einsum('ia,IiAa->IA', t1_1_b, t2_1_ab, optimize = einsum_type)

    rdm1_b[:nocc_b, nocc_b:]  = einsum('IA->IA', t1_1_b, optimize = einsum_type).copy()
    rdm1_b[:nocc_b, nocc_b:] += einsum('IA->IA', t1_2_b, optimize = einsum_type)
    rdm1_b[:nocc_b, nocc_b:] += 1/2 * einsum('ia,iIaA->IA', t1_1_a, t2_1_ab, optimize = einsum_type)
    rdm1_b[:nocc_b, nocc_b:] += 1/2 * einsum('ia,IiAa->IA', t1_1_b, t2_1_b, optimize = einsum_type)

    ### VIR-OCC ###
    rdm1_a[nocc_a:, :nocc_a]  = rdm1_a[:nocc_a, nocc_a:].T

    rdm1_b[nocc_b:, :nocc_b]  = rdm1_b[:nocc_b, nocc_b:].T

    ### VIR-VIR ###
    rdm1_a[nocc_a:, nocc_a:]  = einsum('iA,iB->AB', t1_1_a, t1_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] += 1/2 * einsum('ijAa,ijBa->AB', t2_1_a, t2_1_a, optimize = einsum_type)
    rdm1_a[nocc_a:, nocc_a:] += einsum('ijAa,ijBa->AB', t2_1_ab, t2_1_ab, optimize = einsum_type)

    rdm1_b[nocc_b:, nocc_b:]  = einsum('iA,iB->AB', t1_1_b, t1_1_b, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] += einsum('ijaA,ijaB->AB', t2_1_ab, t2_1_ab, optimize = einsum_type)
    rdm1_b[nocc_b:, nocc_b:] += 1/2 * einsum('ijAa,ijBa->AB', t2_1_b, t2_1_b, optimize = einsum_type)

    ####### ADC(3) SPIN ADAPTED REF OPDM WITH SQA ################
    if adc.method == "adc(3)":
        t2_2_a = adc.t2[1][0][:]
        t2_2_ab = adc.t2[1][1][:]
        t2_2_b = adc.t2[1][2][:]
        if adc.t1[1][0] is not None:
            t1_3_a = adc.t1[1][0][:]
            t1_3_b = adc.t1[1][1][:]
        else:
            t1_3_a = np.zeros((nocc_a, nvir_a))
            t1_3_b = np.zeros((nocc_b, nvir_b))

        #### OCC-OCC ###
        rdm1_a[:nocc_a, :nocc_a] -= einsum('Ia,Ja->IJ', t1_1_a, t1_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= einsum('Ja,Ia->IJ', t1_1_a, t1_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/2 * einsum('Iiab,Jiab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/2 * einsum('Jiab,Iiab->IJ', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= einsum('Iiab,Jiab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= einsum('Jiab,Iiab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/2 * \
            einsum('Ia,ib,Jiab->IJ', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] -= 1/2 * \
            einsum('Ja,ib,Iiab->IJ', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/2 * \
            einsum('Iiab,ia,Jb->IJ', t2_1_a, t1_1_a, t1_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, :nocc_a] += 1/2 * \
            einsum('Jiab,ia,Ib->IJ', t2_1_a, t1_1_a, t1_1_a, optimize = einsum_type)


        rdm1_b[:nocc_b, :nocc_b] -= einsum('Ia,Ja->IJ', t1_1_b, t1_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= einsum('Ja,Ia->IJ', t1_1_b, t1_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= einsum('iIab,iJab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= einsum('iJab,iIab->IJ', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/2 * einsum('Iiab,Jiab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/2 * einsum('Jiab,Iiab->IJ', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/2 * \
            einsum('ia,Ib,iJab->IJ', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] -= 1/2 * \
            einsum('ia,Jb,iIab->IJ', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/2 * \
            einsum('Iiab,ia,Jb->IJ', t2_1_b, t1_1_b, t1_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, :nocc_b] += 1/2 * \
            einsum('Jiab,ia,Ib->IJ', t2_1_b, t1_1_b, t1_1_b, optimize = einsum_type)

        ##### OCC-VIR ### ####
        rdm1_a[:nocc_a, nocc_a:] += einsum('IA->IA', t1_3_a, optimize = einsum_type).copy()
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * einsum('ia,IiAa->IA', t1_2_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * einsum('ia,IiAa->IA', t1_2_b, t2_1_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * einsum('ia,IiAa->IA', t1_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * einsum('ia,IiAa->IA', t1_1_b, t2_2_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] -= 2/3 * einsum('Ia,iA,ia->IA', t1_1_a, t1_1_a, t1_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] -= 1/3 * \
            einsum('Ia,ijab,ijAb->IA', t1_1_a, t2_1_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] -= 1/3 * \
            einsum('iA,ijab,Ijab->IA', t1_1_a, t2_1_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/6 * \
            einsum('ia,ijab,IjAb->IA', t1_1_a, t2_1_a, t2_1_a, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] -= 2/3 * \
            einsum('Ia,ijab,ijAb->IA', t1_1_a, t2_1_ab, t2_1_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] -= 2/3 * \
            einsum('iA,ijab,Ijab->IA', t1_1_a, t2_1_ab, t2_1_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/6 * \
            einsum('ia,ijab,IjAb->IA', t1_1_a, t2_1_ab, t2_1_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/6 * \
            einsum('ia,IjAb,jiba->IA', t1_1_b, t2_1_a, t2_1_ab, optimize = einsum_type)
        rdm1_a[:nocc_a, nocc_a:] += 1/6 * \
            einsum('ia,IjAb,ijab->IA', t1_1_b, t2_1_ab, t2_1_b, optimize = einsum_type)

        rdm1_b[:nocc_b, nocc_b:] += einsum('IA->IA', t1_3_b, optimize = einsum_type).copy()
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * einsum('ia,iIaA->IA', t1_2_a, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * einsum('ia,IiAa->IA', t1_2_b, t2_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * einsum('ia,iIaA->IA', t1_1_a, t2_2_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * einsum('ia,IiAa->IA', t1_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/6 * \
            einsum('ia,ijab,jIbA->IA', t1_1_a, t2_1_a, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/6 * \
            einsum('ia,ijab,IjAb->IA', t1_1_a, t2_1_ab, t2_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] -= 2/3 * einsum('Ia,iA,ia->IA', t1_1_b, t1_1_b, t1_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] -= 2/3 * \
            einsum('Ia,ijba,ijbA->IA', t1_1_b, t2_1_ab, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] -= 2/3 * \
            einsum('iA,jiab,jIab->IA', t1_1_b, t2_1_ab, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/6 * \
            einsum('ia,jiba,jIbA->IA', t1_1_b, t2_1_ab, t2_1_ab, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] -= 1/3 * \
            einsum('Ia,ijab,ijAb->IA', t1_1_b, t2_1_b, t2_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] -= 1/3 * \
            einsum('iA,ijab,Ijab->IA', t1_1_b, t2_1_b, t2_1_b, optimize = einsum_type)
        rdm1_b[:nocc_b, nocc_b:] += 1/6 * \
            einsum('ia,ijab,IjAb->IA', t1_1_b, t2_1_b, t2_1_b, optimize = einsum_type)

        ###### VIR-OCC ###
        rdm1_a[nocc_a:, :nocc_a] = rdm1_a[:nocc_a, nocc_a:].T

        rdm1_b[nocc_b:, :nocc_b] = rdm1_b[:nocc_b, nocc_b:].T

        ##### VIR-VIR ###
        rdm1_a[nocc_a:, nocc_a:] += 1/2 * einsum('ijAa,ijBa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/2 * einsum('ijBa,ijAa->AB', t2_1_a, t2_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += einsum('ijAa,ijBa->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += einsum('ijBa,ijAa->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += einsum('iA,iB->AB', t1_1_a, t1_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += einsum('iB,iA->AB', t1_1_a, t1_2_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/2 * \
            einsum('iA,ja,ijBa->AB', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/2 * \
            einsum('iB,ja,ijAa->AB', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/2 * \
            einsum('ijAa,iB,ja->AB', t2_1_a, t1_1_a, t1_1_a, optimize = einsum_type)
        rdm1_a[nocc_a:, nocc_a:] += 1/2 * \
            einsum('ijBa,iA,ja->AB', t2_1_a, t1_1_a, t1_1_a, optimize = einsum_type)

        rdm1_b[nocc_b:, nocc_b:] += einsum('ijaA,ijaB->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += einsum('ijaB,ijaA->AB', t2_1_ab, t2_2_ab, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/2 * einsum('ijAa,ijBa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/2 * einsum('ijBa,ijAa->AB', t2_1_b, t2_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += einsum('iA,iB->AB', t1_1_b, t1_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += einsum('iB,iA->AB', t1_1_b, t1_2_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/2 * \
            einsum('ia,jA,ijaB->AB', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/2 * \
            einsum('ia,jB,ijaA->AB', t1_1_a, t1_1_b, t2_1_ab, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/2 * \
            einsum('ijAa,iB,ja->AB', t2_1_b, t1_1_b, t1_1_b, optimize = einsum_type)
        rdm1_b[nocc_b:, nocc_b:] += 1/2 * \
            einsum('ijBa,iA,ja->AB', t2_1_b, t1_1_b, t1_1_b, optimize = einsum_type)

    if with_frozen and adc.frozen is not None:
        nmo_a = adc.mo_coeff_hf[0].shape[1]
        nmo_b = adc.mo_coeff_hf[1].shape[1]
        nocc_a = np.count_nonzero(adc.mo_occ[0] > 0)
        nocc_b = np.count_nonzero(adc.mo_occ[1] > 0)
        dm_a = np.zeros((nmo_a,nmo_a))
        dm_b = np.zeros((nmo_b,nmo_b))
        dm_a[np.diag_indices(nocc_a)] = 1
        dm_b[np.diag_indices(nocc_b)] = 1
        moidx = adc.get_frozen_mask()
        moidxa = np.where(moidx[0])[0]
        moidxb = np.where(moidx[1])[0]
        dm_a[moidxa[:,None],moidxa] = rdm1_a
        dm_b[moidxb[:,None],moidxb] = rdm1_b
        rdm1_a = dm_a
        rdm1_b = dm_b
        if ao_repr:
            mo_a, mo_b = adc.mo_coeff_hf
            rdm1_a = lib.einsum('pi,ij,qj->pq', mo_a, rdm1_a, mo_a)
            rdm1_b = lib.einsum('pi,ij,qj->pq', mo_b, rdm1_b, mo_b)

    elif ao_repr:
        mo_a, mo_b = adc.mo_coeff
        rdm1_a = lib.einsum('pi,ij,qj->pq', mo_a, rdm1_a, mo_a)
        rdm1_b = lib.einsum('pi,ij,qj->pq', mo_b, rdm1_b, mo_b)

    return (rdm1_a, rdm1_b)


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
        'mo_coeff_hf', 'mol', 'mo_energy_a', 'mo_energy_b', 'incore_complete',
        'scf_energy', 'e_tot', 't1', 't2', 'frozen', 'chkfile',
        'max_space', 'mo_occ', 'max_cycle', 'imds', 'with_df', 'compute_properties',
        'approx_trans_moments', 'evec_print_tol', 'spec_factor_print_tol',
        'E', 'U', 'P', 'X', 'ncvs', 'dip_mom', 'dip_mom_nuc',
        'compute_spin_square', 'f_ov',
        'nocc_a', 'nocc_b', 'nvir_a', 'nvir_b',
        'if_heri_eris'
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, f_ov=None):

        if 'dft' in str(mf.__module__):
            raise NotImplementedError('DFT reference for UADC')

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

        self.f_ov = f_ov
        self._nmo = None
        self._nocc = mf.nelec
        self.mo_occ = mo_occ

        if isinstance(mf, scf.rohf.ROHF):

            logger.info(mf, "\nROHF reference detected in ADC")

            mo_occa = (mo_occ>1e-8).astype(np.double)
            mo_occb = mo_occ - mo_occa
            self.mo_occ = [mo_occa, mo_occb]
            if_canonical = False
        else:
            self.mo_energy_a = mf.mo_energy[0]
            self.mo_energy_b = mf.mo_energy[1]

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
            if isinstance(mf, scf.rohf.ROHF):

                logger.info(mf, "\nSemicanonicalizing the orbitals...")

                if_canonical = True
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

        elif isinstance(mf, scf.rohf.ROHF) and f_ov is None:
            raise ValueError("f_ov must be provided when mo_coeff is given for ROHF reference")

        self.mo_coeff = mo_coeff
        self.mo_coeff_hf = mo_coeff
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.imds = lambda:None
        self.if_heri_eris = False
        if frozen is None:
            self._nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
        elif hasattr(frozen, '__len__'):
            if len(frozen) != 2:
                raise NotImplementedError("frozen should be announced as None or a array-like object with two elements")
            elif isinstance(frozen, list) or isinstance(frozen, np.ndarray):
                self.frozen = frozen = tuple(frozen)

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

            (mask_a,mask_b) = self.get_frozen_mask()
            maskocc_a = self.mo_occ[0]>1e-6
            maskocc_b = self.mo_occ[1]>1e-6
            occ_a = maskocc_a & mask_a
            occ_b = maskocc_b & mask_b
            self._nocc = (int(occ_a.sum()), int(occ_b.sum()))
            self.mo_coeff = (self.mo_coeff[0][:,mask_a], self.mo_coeff[1][:,mask_b])
            if (self.mo_coeff_hf is self._scf.mo_coeff and self._scf.converged) or \
                    (isinstance(self._scf, scf.rohf.ROHF) and if_canonical):
                self.mo_energy_a = self.mo_energy_a[mask_a]
                self.mo_energy_b = self.mo_energy_b[mask_b]
                if isinstance(self._scf, scf.rohf.ROHF):
                    vir_a = ~maskocc_a & mask_a
                    vir_b = ~maskocc_b & mask_b
                    f_ov_a, f_ov_b = self.f_ov
                    f_ov_a_tmp = f_ov_a[occ_a[:mf.nelec[0]],:]
                    f_ov_a = f_ov_a_tmp[:,vir_a[mf.nelec[0]:]]
                    f_ov_b_tmp = f_ov_b[occ_b[:mf.nelec[1]],:]
                    f_ov_b = f_ov_b_tmp[:,vir_b[mf.nelec[1]:]]
                    self.f_ov = [f_ov_a, f_ov_b]
            else:
                h1e = mf.get_hcore()
                dm = scf.uhf.make_rdm1(mo_coeff, self.mo_occ)
                vhf = scf.uhf.get_veff(mf.mol, dm)
                fock_a = h1e + vhf[0]
                fock_b = h1e + vhf[1]
                fock_a = self.mo_coeff[0].conj().T.dot(fock_a).dot(self.mo_coeff[0])
                fock_b = self.mo_coeff[1].conj().T.dot(fock_b).dot(self.mo_coeff[1])
                (self.mo_energy_a,self.mo_energy_b) = (fock_a.diagonal().real,fock_b.diagonal().real)
                self.scf_energy = self._scf.energy_tot(dm=dm, vhf=vhf)
        else:
            raise NotImplementedError("each element of frozen should be None, an integer or a array-like object")

        self._nvir = (self._nmo[0] - self._nocc[0], self._nmo[1] - self._nocc[1])
        self.nocc_a = self._nocc[0]
        self.nocc_b = self._nocc[1]
        self.nvir_a = self._nvir[0]
        self.nvir_b = self._nvir[1]
        if self.nocc_a == 0 or self.nocc_b == 0:
            raise ValueError("No occupied alpha or beta orbitals found")
        if self.nvir_a == 0 or self.nvir_b == 0:
            raise ValueError("No virtual alpha or beta orbitals found")

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

        nocc_fr_a = self._scf.nelec[0] - self.nocc_a
        nocc_fr_b = self._scf.nelec[1] - self.nocc_b
        nvir_fr_a = self.mo_occ[0].shape[0] - nmo_a - nocc_fr_a
        nvir_fr_b = self.mo_occ[1].shape[0] - nmo_b - nocc_fr_b

        logger.info(self, '******** ADC Orbital Information ********')
        logger.info(self, 'Number of Frozen Occupied Alpha Orbitals: %d', nocc_fr_a)
        logger.info(self, 'Number of Frozen Occupied Beta Orbitals: %d', nocc_fr_b)
        logger.info(self, 'Number of Frozen Virtual Alpha Orbitals: %d', nvir_fr_a)
        logger.info(self, 'Number of Frozen Virtual Beta Orbitals: %d', nvir_fr_b)
        logger.info(self, 'Number of Active Occupied Alpha Orbitals: %d', self.nocc_a)
        logger.info(self, 'Number of Active Occupied Beta Orbitals: %d', self.nocc_b)
        logger.info(self, 'Number of Active Virtual Alpha Orbitals: %d', self.nvir_a)
        logger.info(self, 'Number of Active Virtual Beta Orbitals: %d', self.nvir_b)

        if hasattr(self.frozen, '__len__'):
            if hasattr(self.frozen[0], '__len__'):
                logger.info(self, 'Frozen Orbital List (Alpha): %s', self.frozen[0])
            if hasattr(self.frozen[1], '__len__'):
                logger.info(self, 'Frozen Orbital List (Beta): %s', self.frozen[1])
        logger.info(self, '*****************************************')

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

        nocc_fr_a = self._scf.nelec[0] - self.nocc_a
        nocc_fr_b = self._scf.nelec[1] - self.nocc_b
        nvir_fr_a = self.mo_occ[0].shape[0] - nmo_a - nocc_fr_a
        nvir_fr_b = self.mo_occ[1].shape[0] - nmo_b - nocc_fr_b

        logger.info(self, '******** ADC Orbital Information ********')
        logger.info(self, 'Number of Frozen Occupied Alpha Orbitals: %d', nocc_fr_a)
        logger.info(self, 'Number of Frozen Occupied Beta Orbitals: %d', nocc_fr_b)
        logger.info(self, 'Number of Frozen Virtual Alpha Orbitals: %d', nvir_fr_a)
        logger.info(self, 'Number of Frozen Virtual Beta Orbitals: %d', nvir_fr_b)
        logger.info(self, 'Number of Active Occupied Alpha Orbitals: %d', self.nocc_a)
        logger.info(self, 'Number of Active Occupied Beta Orbitals: %d', self.nocc_b)
        logger.info(self, 'Number of Active Virtual Alpha Orbitals: %d', self.nvir_a)
        logger.info(self, 'Number of Active Virtual Beta Orbitals: %d', self.nvir_b)

        if hasattr(self.frozen, '__len__'):
            if hasattr(self.frozen[0], '__len__'):
                logger.info(self, 'Frozen Orbital List (Alpha): %s', self.frozen[0])
            if hasattr(self.frozen[1], '__len__'):
                logger.info(self, 'Frozen Orbital List (Beta): %s', self.frozen[1])
        logger.info(self, '*****************************************')

        if eris is None:
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

    def make_rdm1(self, with_frozen=True, ao_repr=False):
        list_rdm1_a, list_rdm1_b = self._adc_es._make_rdm1()

        if with_frozen and self.frozen is not None:
            nmo_a = self.mo_coeff_hf[0].shape[1]
            nmo_b = self.mo_coeff_hf[1].shape[1]
            if isinstance(self._scf, scf.rohf.ROHF):
                nocc_a = self._scf.mol.nelec[0]
                nocc_b = self._scf.mol.nelec[1]
            else:
                nocc_a = np.count_nonzero(self.mo_occ[0] > 0)
                nocc_b = np.count_nonzero(self.mo_occ[1] > 0)
            moidx = self.get_frozen_mask()
            moidxa = np.where(moidx[0])[0]
            moidxb = np.where(moidx[1])[0]
            for i in range(self._adc_es.U.shape[1]):
                rdm1_a = list_rdm1_a[i]
                rdm1_b = list_rdm1_b[i]
                dm_a = np.zeros((nmo_a,nmo_a))
                dm_b = np.zeros((nmo_b,nmo_b))
                dm_a[np.diag_indices(nocc_a)] = 1
                dm_b[np.diag_indices(nocc_b)] = 1
                dm_a[moidxa[:,None],moidxa] = rdm1_a
                dm_b[moidxb[:,None],moidxb] = rdm1_b
                rdm1_a = dm_a
                rdm1_b = dm_b
                if ao_repr:
                    mo_a, mo_b = self.mo_coeff_hf
                    rdm1_a = lib.einsum('pi,ij,qj->pq', mo_a, rdm1_a, mo_a)
                    rdm1_b = lib.einsum('pi,ij,qj->pq', mo_b, rdm1_b, mo_b)
                list_rdm1_a[i] = rdm1_a
                list_rdm1_b[i] = rdm1_b

        elif ao_repr:
            mo_a, mo_b = self.mo_coeff
            for i in range(self._adc_es.U.shape[1]):
                rdm1_a = list_rdm1_a[i]
                rdm1_b = list_rdm1_b[i]
                rdm1_a = lib.einsum('pi,ij,qj->pq', mo_a, rdm1_a, mo_a)
                rdm1_b = lib.einsum('pi,ij,qj->pq', mo_b, rdm1_b, mo_b)
                list_rdm1_a[i] = rdm1_a
                list_rdm1_b[i] = rdm1_b

        return (list_rdm1_a, list_rdm1_b)


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
