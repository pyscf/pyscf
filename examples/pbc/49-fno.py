#!/usr/bin/env python
#
# Author: Ning-Yuan Chen <cny003@outlook.com>
#

'''
This example file shows how to use MPn/ADC based FNO/SS-FNO framework to do k-point post-HF calculations.
The first case uses ADC2 to generate SS-FNOs and then do SS-FNO-ADC(3) calculation.
The second case uses these SS-FNOs to do FNO-EOM-CCSD calculation.
The third case uses MP2 to generate FNOs and then do FNO-ADC(3) calculation.
The fourth case demonstrates the multi-threshold FNO feature: several FNO truncation
levels are generated in a single call, with the expensive canonical MP2/ADC(2) step
performed only once.
The sixth case demonstrates the per-k-point FNO mode (mode='per_kpt'), which lets
each k-point keep a different number of active virtuals instead of a common count.
'''

import numpy as np
from pyscf.pbc import gto, scf, adc

def qp_correct(FG, E, P, kptlist):
    if E.shape!=FG.delta_e.shape:
        raise ValueError("The shape of uncorrected E and delta_e should be the same")
    print("start to correct the quasiparticle energy")

    E_p_corrected = []
    mask_fno = FG.p_ssfno>0.5
    mask = P>0.5

    if kptlist is None:
        kptlist = range(FG.nkpts)

    msg = ("\n*************************************************************"
        "\n            FNO quasiparticle energy summary"
        "\n*************************************************************")
    print(msg)

    for k, kshift in enumerate(kptlist):
        delta_e_qp_k = FG.delta_e_qp[k]
        E_p_k = E[k,mask[k]]
        E_p_corrected_k = E_p_k[:min(len(E_p_k), len(delta_e_qp_k))] + delta_e_qp_k[:min(len(E_p_k), len(delta_e_qp_k))]
        sort_indices = np.argsort(E_p_corrected_k)
        E_p_corrected.append(E_p_corrected_k[sort_indices])
        for n in range(E_p_corrected[k].shape[0]):
            print_string = ('%s-FNO k-point %d | qp %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  |  %s-FNO Spec factors = %10.8f  |  target-FNO Spec factors = %10.8f' %
                            (FG.method, kshift, n, E_p_corrected[k][n], E_p_corrected[k][n]*27.2114, FG.method, FG.p_ssfno[k,mask_fno[k]][sort_indices[n]], P[k,mask[k]][sort_indices[n]]))
            print(print_string)

    return E_p_corrected

cell = gto.Cell()
cell.verbose = 0
cell.unit = 'B'

cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.build()

nmp = [2,2,1]
nroots_test = 5

# KRHF
kpts = cell.make_kpts(nmp)
kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
ekrhf = kmf.kernel()

# case1 SS-FNO kADC3

# FNOGenerator
ADCFG = adc.KRADC2FNO(kmf)
ADCFG.method_type = 'ea'
ADCFG.approx_trans_moments = True
ADCFG.verbose = 5
ADCFG.ref_state = [[0],[0]]
ADCFG.kernel(nroots_test,pct_occ=0.90,kptlist=[0])

# kadc3
kadc = adc.KRADC(kmf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ,ADCFG.mo_energy)
kadc.method_type = "ea"
kadc.approx_trans_moments = True
kadc.verbose = 5
kadc.method="adc(3)"
k_e_ea, k_v_ea, k_p_ea, k_x_ea = kadc.kernel(nroots_test,guess=ADCFG.v_ssfno,kptlist=[0])

e_qp=qp_correct(ADCFG,k_e_ea,k_p_ea,kptlist=[0])

# case2 SS-FNO kEOM-CCSD
from pyscf.pbc import cc
from pyscf.pbc.cc.eom_kccsd_rhf import EOMEA

# KRCCSD
mycc = cc.KRCCSD(kmf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ)
mycc.verbose = 5
ekrcc, t1, t2 = mycc.kernel()
# K-EA-EOM-CCSD
myeom = EOMEA(mycc)
eomcc_e,eomcc_v = myeom.kernel(nroots_test,kptlist=[0])

eomcc_e_corrected = ADCFG.correct(eomcc_e)
print("SS-FNO K-EA-EOM-CCSD roots (eV):", eomcc_e_corrected*27.2114)

# case3 FNO-MP3
MPFG = adc.KRADC2FNO(kmf)
MPFG.approx_trans_moments = True
MPFG.verbose = 5
MPFG.kernel_gs(pct_occ=0.1)

kadc_gs  = adc.KRADC(kmf,MPFG.frozen,MPFG.mo_coeff,MPFG.mo_occ,MPFG.mo_energy)
kadc_gs.approx_trans_moments = True
kadc_gs.verbose = 5
kadc_gs.method = "adc(3)"
e_corr,t1,t2 = kadc_gs.kernel_gs()

e_corr_correct = MPFG.correct_corr(e_corr)
print("FNO KMP3 correlation energy (eV):", e_corr_correct*27.2114)

# case4 Multi-threshold FNO-MP3 (ground state)
# When pct_occ (or thresh / nvir_act) is passed as a list, the expensive canonical
# MP2/ADC(2) + 1-RDM step is performed only once, and FNOs are built for every
# threshold.  The output attributes (frozen, mo_coeff, mo_occ, mo_energy,
# delta_e_corr, e_corr_fno) become lists indexed by threshold.
# The resulting energies at two thresholds can be combined by two-point linear
# extrapolation (see below) to reduce the residual FNO truncation error.
MPMFG = adc.KRADC2FNO(kmf)
MPMFG.approx_trans_moments = True
MPMFG.verbose = 5
MPMFG.kernel_gs(pct_occ=[0.5, 0.8, 0.9])

pct_list = [0.5, 0.8, 0.9]
e_corr_list = []
for i in range(len(MPMFG.frozen)):
    kadc_gs = adc.KRADC(kmf, MPMFG.frozen[i], MPMFG.mo_coeff[i],
                         MPMFG.mo_occ[i], MPMFG.mo_energy[i])
    kadc_gs.approx_trans_moments = True
    kadc_gs.verbose = 5
    kadc_gs.method = "adc(3)"
    e_corr, t1, t2 = kadc_gs.kernel_gs()
    e_corr_correct = MPMFG.correct_corr(e_corr, i)
    print('pct %4.2f | n_frozen/kpt = %s | uncorrected E_corr = %.10f eV | corrected E_corr = %.10f eV' %
          (pct_list[i], [len(f) for f in MPMFG.frozen[i]], e_corr*27.2114, e_corr_correct*27.2114))
    e_corr_list.append(e_corr)

# Two-point linear extrapolation (LE) of the FNO correlation energy to zero
# truncation error. For two thresholds a (pct_occ=0.9, index 2) and b
# (pct_occ=0.8, index 1):
#   E_LE(M) = E_a(M) + (E_a(M) - E_b(M))/(E_a(MP2) - E_b(MP2))
#                          * (E_can(MP2) - E_a(MP2))
# where E_i(M) = e_corr_list[i] is the uncorrected FNO correlation energy of
# the target method M (here MP3). With the generator's MP2 additive correction
# delta_e_corr[i] = E_can(MP2) - E_i(MP2), this is evaluated as
#   e_corr_list[a] + (e_corr_list[a] - e_corr_list[b])
#                   /(delta_e_corr[b] - delta_e_corr[a]) * delta_e_corr[a].
e_corr_le = e_corr_list[2] + MPMFG.delta_e_corr[2]*(e_corr_list[2]-e_corr_list[1])\
    /(MPMFG.delta_e_corr[1]-MPMFG.delta_e_corr[2])
print('Linear extrapolated FNO KMP3 correlation energy (eV): %.10f' % (e_corr_le*27.2114))

# case5 Multi-threshold SS-FNO-ADC(3) (excited state)
# The excited-state kernel() also accepts a list of thresholds.
# For multiple thresholds, e_ssfno, v_ssfno, p_ssfno, delta_e, delta_e_qp
# become lists indexed by threshold as well.
ADCMFG = adc.KRADC2FNO(kmf)
ADCMFG.method_type = 'ea'
ADCMFG.approx_trans_moments = True
ADCMFG.verbose = 5
ADCMFG.ref_state = [[0],[0]]
ADCMFG.kernel(nroots_test, pct_occ=[0.80, 0.90], kptlist=[0])

pct_list_es = [0.80, 0.90]
for i in range(len(ADCMFG.frozen)):
    kadc = adc.KRADC(kmf, ADCMFG.frozen[i], ADCMFG.mo_coeff[i],
                       ADCMFG.mo_occ[i])
    kadc.method_type = "ea"
    kadc.approx_trans_moments = True
    kadc.verbose = 5
    kadc.method = "adc(3)"
    k_e_ea, k_v_ea, k_p_ea, _ = kadc.kernel(nroots_test,
                                              guess=ADCMFG.v_ssfno[i],
                                              kptlist=[0])
    k_e_ea_corrected = ADCMFG.correct(k_e_ea, i)
    print('pct %4.2f | n_frozen/kpt = %s | uncorrected root 0 = %.10f eV | corrected root 0 = %.10f eV' %
          (pct_list_es[i], [len(f) for f in ADCMFG.frozen[i]], k_e_ea[0][0]*27.2114, k_e_ea_corrected[0][0]*27.2114))

# case6 Per-k-point FNO (mode='per_kpt')
# By default mode='min' unions the per-k-point truncation masks so every k-point
# keeps the same number of active virtuals.  mode='per_kpt' instead truncates
# each k-point independently, so the frozen count can differ across k-points.
# Here thresh=0.05 keeps 3/2/2/1 active virtuals, illustrating the difference.
PKFG = adc.KRADC2FNO(kmf)
PKFG.mode = 'per_kpt'
PKFG.approx_trans_moments = True
PKFG.verbose = 5
PKFG.kernel_gs(thresh=0.05)

kadc_pk = adc.KRADC(kmf, PKFG.frozen, PKFG.mo_coeff, PKFG.mo_occ, PKFG.mo_energy)
kadc_pk.approx_trans_moments = True
kadc_pk.verbose = 5
kadc_pk.method = 'adc(3)'
e_corr_pk, t1_pk, t2_pk = kadc_pk.kernel_gs()
print('per_kpt FNO-MP3 correlation energy (eV):', (e_corr_pk + PKFG.delta_e_corr)*27.2114)