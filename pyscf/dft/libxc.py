#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          Susi Lehtola <susi.lehtola@gmail.com>

'''
XC functional, the interface to libxc
(http://www.tddft.org/programs/octopus/wiki/index.php/Libxc)
'''

import sys
import warnings
import copy
import ctypes
import math
import numpy
from pyscf import lib

_itrf = lib.load_library('libxc_itrf')
_itrf.LIBXC_is_lda.restype = ctypes.c_int
_itrf.LIBXC_is_gga.restype = ctypes.c_int
_itrf.LIBXC_is_meta_gga.restype = ctypes.c_int
_itrf.LIBXC_is_hybrid.restype = ctypes.c_int
_itrf.LIBXC_max_deriv_order.restype = ctypes.c_int
_itrf.LIBXC_hybrid_coeff.argtypes = [ctypes.c_int]
_itrf.LIBXC_hybrid_coeff.restype = ctypes.c_double
_itrf.LIBXC_nlc_coeff.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
_itrf.LIBXC_rsh_coeff.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double)]

# xc_code from libxc
#cat lib/deps/include/xc_funcs.h  | awk '{printf("'\''%s'\'' %3i",$2,$3); for(i=4;i<NF;i++) {printf(" %s",$i)}; printf("\n")}'  | sed "s|/\*|# |g" | awk '{printf("%-30s : %4i\,",$1,$2); for(i=4;i<NF;i++) {printf(" %s",$i)}; printf("\n")}'

XC = XC_CODES = {
'XC_LDA_X'                     :    1, # Exchange
'XC_LDA_C_WIGNER'              :    2, # Wigner parametrization
'XC_LDA_C_RPA'                 :    3, # Random Phase Approximation
'XC_LDA_C_HL'                  :    4, # Hedin & Lundqvist
'XC_LDA_C_GL'                  :    5, # Gunnarson & Lundqvist
'XC_LDA_C_XALPHA'              :    6, # Slater Xalpha
'XC_LDA_C_VWN'                 :    7, # Vosko, Wilk, & Nusair (5)
'XC_LDA_C_VWN_RPA'             :    8, # Vosko, Wilk, & Nusair (RPA)
'XC_LDA_C_PZ'                  :    9, # Perdew & Zunger
'XC_LDA_C_PZ_MOD'              :   10, # Perdew & Zunger (Modified)
'XC_LDA_C_OB_PZ'               :   11, # Ortiz & Ballone (PZ)
'XC_LDA_C_PW'                  :   12, # Perdew & Wang
'XC_LDA_C_PW_MOD'              :   13, # Perdew & Wang (Modified)
'XC_LDA_C_OB_PW'               :   14, # Ortiz & Ballone (PW)
'XC_LDA_C_2D_AMGB'             :   15, # Attaccalite et al
'XC_LDA_C_2D_PRM'              :   16, # Pittalis, Rasanen & Marques correlation in 2D
'XC_LDA_C_VBH'                 :   17, # von Barth & Hedin
'XC_LDA_C_1D_CSC'              :   18, # Casula, Sorella, and Senatore 1D correlation
'XC_LDA_X_2D'                  :   19, # Exchange in 2D
'XC_LDA_XC_TETER93'            :   20, # Teter 93 parametrization
'XC_LDA_X_1D'                  :   21, # Exchange in 1D
'XC_LDA_C_ML1'                 :   22, # Modified LSD (version 1) of Proynov and Salahub
'XC_LDA_C_ML2'                 :   23, # Modified LSD (version 2) of Proynov and Salahub
'XC_LDA_C_GOMBAS'              :   24, # Gombas parametrization
'XC_LDA_C_PW_RPA'              :   25, # Perdew & Wang fit of the RPA
'XC_LDA_C_1D_LOOS'             :   26, # P-F Loos correlation LDA
'XC_LDA_C_RC04'                :   27, # Ragot-Cortona
'XC_LDA_C_VWN_1'               :   28, # Vosko, Wilk, & Nusair (1)
'XC_LDA_C_VWN_2'               :   29, # Vosko, Wilk, & Nusair (2)
'XC_LDA_C_VWN_3'               :   30, # Vosko, Wilk, & Nusair (3)
'XC_LDA_C_VWN_4'               :   31, # Vosko, Wilk, & Nusair (4)
'XC_LDA_XC_ZLP'                :   43, # Zhao, Levy & Parr, Eq. (20)
'XC_LDA_K_TF'                  :   50, # Thomas-Fermi kinetic energy functional
'XC_LDA_K_LP'                  :   51, # Lee and Parr Gaussian ansatz
'XC_LDA_XC_KSDT'               :  259, # Karasiev et al. parametrization
'XC_GGA_X_GAM'                 :   32, # GAM functional from Minnesota
'XC_GGA_C_GAM'                 :   33, # GAM functional from Minnesota
'XC_GGA_X_HCTH_A'              :   34, # HCTH-A
'XC_GGA_X_EV93'                :   35, # Engel and Vosko
'XC_GGA_X_BGCP'                :   38, # Burke, Cancio, Gould, and Pittalis
'XC_GGA_C_BGCP'                :   39, # Burke, Cancio, Gould, and Pittalis
'XC_GGA_X_LAMBDA_OC2_N'        :   40, # lambda_OC2(N) version of PBE
'XC_GGA_X_B86_R'               :   41, # Revised Becke 86 Xalpha,beta,gamma (with mod. grad. correction)
'XC_GGA_X_LAMBDA_CH_N'         :   44, # lambda_CH(N) version of PBE
'XC_GGA_X_LAMBDA_LO_N'         :   45, # lambda_LO(N) version of PBE
'XC_GGA_X_HJS_B88_V2'          :   46, # HJS screened exchange corrected B88 version
'XC_GGA_C_Q2D'                 :   47, # Chiodo et al
'XC_GGA_X_Q2D'                 :   48, # Chiodo et al
'XC_GGA_X_PBE_MOL'             :   49, # Del Campo, Gazquez, Trickey and Vela (PBE-like)
'XC_GGA_K_TFVW'                :   52, # Thomas-Fermi plus von Weiszaecker correction
'XC_GGA_K_REVAPBEINT'          :   53, # interpolated version of REVAPBE
'XC_GGA_K_APBEINT'             :   54, # interpolated version of APBE
'XC_GGA_K_REVAPBE'             :   55, # revised APBE
'XC_GGA_X_AK13'                :   56, # Armiento & Kuemmel 2013
'XC_GGA_K_MEYER'               :   57, # Meyer, Wang, and Young
'XC_GGA_X_LV_RPW86'            :   58, # Berland and Hyldgaard
'XC_GGA_X_PBE_TCA'             :   59, # PBE revised by Tognetti et al
'XC_GGA_X_PBEINT'              :   60, # PBE for hybrid interfaces
'XC_GGA_C_ZPBEINT'             :   61, # spin-dependent gradient correction to PBEint
'XC_GGA_C_PBEINT'              :   62, # PBE for hybrid interfaces
'XC_GGA_C_ZPBESOL'             :   63, # spin-dependent gradient correction to PBEsol
'XC_GGA_XC_OPBE_D'             :   65, # oPBE_D functional of Goerigk and Grimme
'XC_GGA_XC_OPWLYP_D'           :   66, # oPWLYP-D functional of Goerigk and Grimme
'XC_GGA_XC_OBLYP_D'            :   67, # oBLYP-D functional of Goerigk and Grimme
'XC_GGA_X_VMT84_GE'            :   68, # VMT{8,4} with constraint satisfaction with mu = mu_GE
'XC_GGA_X_VMT84_PBE'           :   69, # VMT{8,4} with constraint satisfaction with mu = mu_PBE
'XC_GGA_X_VMT_GE'              :   70, # Vela, Medel, and Trickey with mu = mu_GE
'XC_GGA_X_VMT_PBE'             :   71, # Vela, Medel, and Trickey with mu = mu_PBE
'XC_GGA_C_N12_SX'              :   79, # N12-SX functional from Minnesota
'XC_GGA_C_N12'                 :   80, # N12 functional from Minnesota
'XC_GGA_X_N12'                 :   82, # N12 functional from Minnesota
'XC_GGA_C_REGTPSS'             :   83, # Regularized TPSS correlation (ex-VPBE)
'XC_GGA_C_OP_XALPHA'           :   84, # one-parameter progressive functional (XALPHA version)
'XC_GGA_C_OP_G96'              :   85, # one-parameter progressive functional (G96 version)
'XC_GGA_C_OP_PBE'              :   86, # one-parameter progressive functional (PBE version)
'XC_GGA_C_OP_B88'              :   87, # one-parameter progressive functional (B88 version)
'XC_GGA_C_FT97'                :   88, # Filatov & Thiel correlation
'XC_GGA_C_SPBE'                :   89, # PBE correlation to be used with the SSB exchange
'XC_GGA_X_SSB_SW'              :   90, # Swarta, Sola and Bickelhaupt correction to PBE
'XC_GGA_X_SSB'                 :   91, # Swarta, Sola and Bickelhaupt
'XC_GGA_X_SSB_D'               :   92, # Swarta, Sola and Bickelhaupt dispersion
'XC_GGA_XC_HCTH_407P'          :   93, # HCTH/407+
'XC_GGA_XC_HCTH_P76'           :   94, # HCTH p=7/6
'XC_GGA_XC_HCTH_P14'           :   95, # HCTH p=1/4
'XC_GGA_XC_B97_GGA1'           :   96, # Becke 97 GGA-1
'XC_GGA_C_HCTH_A'              :   97, # HCTH-A
'XC_GGA_X_BPCCAC'              :   98, # BPCCAC (GRAC for the energy)
'XC_GGA_C_REVTCA'              :   99, # Tognetti, Cortona, Adamo (revised)
'XC_GGA_C_TCA'                 :  100, # Tognetti, Cortona, Adamo
'XC_GGA_X_PBE'                 :  101, # Perdew, Burke & Ernzerhof exchange
'XC_GGA_X_PBE_R'               :  102, # Perdew, Burke & Ernzerhof exchange (revised)
'XC_GGA_X_B86'                 :  103, # Becke 86 Xalpha,beta,gamma
'XC_GGA_X_HERMAN'              :  104, # Herman et al original GGA
'XC_GGA_X_B86_MGC'             :  105, # Becke 86 Xalpha,beta,gamma (with mod. grad. correction)
'XC_GGA_X_B88'                 :  106, # Becke 88
'XC_GGA_X_G96'                 :  107, # Gill 96
'XC_GGA_X_PW86'                :  108, # Perdew & Wang 86
'XC_GGA_X_PW91'                :  109, # Perdew & Wang 91
'XC_GGA_X_OPTX'                :  110, # Handy & Cohen OPTX 01
'XC_GGA_X_DK87_R1'             :  111, # dePristo & Kress 87 (version R1)
'XC_GGA_X_DK87_R2'             :  112, # dePristo & Kress 87 (version R2)
'XC_GGA_X_LG93'                :  113, # Lacks & Gordon 93
'XC_GGA_X_FT97_A'              :  114, # Filatov & Thiel 97 (version A)
'XC_GGA_X_FT97_B'              :  115, # Filatov & Thiel 97 (version B)
'XC_GGA_X_PBE_SOL'             :  116, # Perdew, Burke & Ernzerhof exchange (solids)
'XC_GGA_X_RPBE'                :  117, # Hammer, Hansen & Norskov (PBE-like)
'XC_GGA_X_WC'                  :  118, # Wu & Cohen
'XC_GGA_X_MPW91'               :  119, # Modified form of PW91 by Adamo & Barone
'XC_GGA_X_AM05'                :  120, # Armiento & Mattsson 05 exchange
'XC_GGA_X_PBEA'                :  121, # Madsen (PBE-like)
'XC_GGA_X_MPBE'                :  122, # Adamo & Barone modification to PBE
'XC_GGA_X_XPBE'                :  123, # xPBE reparametrization by Xu & Goddard
'XC_GGA_X_2D_B86_MGC'          :  124, # Becke 86 MGC for 2D systems
'XC_GGA_X_BAYESIAN'            :  125, # Bayesian best fit for the enhancement factor
'XC_GGA_X_PBE_JSJR'            :  126, # JSJR reparametrization by Pedroza, Silva & Capelle
'XC_GGA_X_2D_B88'              :  127, # Becke 88 in 2D
'XC_GGA_X_2D_B86'              :  128, # Becke 86 Xalpha,beta,gamma
'XC_GGA_X_2D_PBE'              :  129, # Perdew, Burke & Ernzerhof exchange in 2D
'XC_GGA_C_PBE'                 :  130, # Perdew, Burke & Ernzerhof correlation
'XC_GGA_C_LYP'                 :  131, # Lee, Yang & Parr
'XC_GGA_C_P86'                 :  132, # Perdew 86
'XC_GGA_C_PBE_SOL'             :  133, # Perdew, Burke & Ernzerhof correlation SOL
'XC_GGA_C_PW91'                :  134, # Perdew & Wang 91
'XC_GGA_C_AM05'                :  135, # Armiento & Mattsson 05 correlation
'XC_GGA_C_XPBE'                :  136, # xPBE reparametrization by Xu & Goddard
'XC_GGA_C_LM'                  :  137, # Langreth and Mehl correlation
'XC_GGA_C_PBE_JRGX'            :  138, # JRGX reparametrization by Pedroza, Silva & Capelle
'XC_GGA_X_OPTB88_VDW'          :  139, # Becke 88 reoptimized to be used with vdW functional of Dion et al
'XC_GGA_X_PBEK1_VDW'           :  140, # PBE reparametrization for vdW
'XC_GGA_X_OPTPBE_VDW'          :  141, # PBE reparametrization for vdW
'XC_GGA_X_RGE2'                :  142, # Regularized PBE
'XC_GGA_C_RGE2'                :  143, # Regularized PBE
'XC_GGA_X_RPW86'               :  144, # refitted Perdew & Wang 86
'XC_GGA_X_KT1'                 :  145, # Keal and Tozer version 1
'XC_GGA_XC_KT2'                :  146, # Keal and Tozer version 2
'XC_GGA_C_WL'                  :  147, # Wilson & Levy
'XC_GGA_C_WI'                  :  148, # Wilson & Ivanov
'XC_GGA_X_MB88'                :  149, # Modified Becke 88 for proton transfer
'XC_GGA_X_SOGGA'               :  150, # Second-order generalized gradient approximation
'XC_GGA_X_SOGGA11'             :  151, # Second-order generalized gradient approximation 2011
'XC_GGA_C_SOGGA11'             :  152, # Second-order generalized gradient approximation 2011
'XC_GGA_C_WI0'                 :  153, # Wilson & Ivanov initial version
'XC_GGA_XC_TH1'                :  154, # Tozer and Handy v. 1
'XC_GGA_XC_TH2'                :  155, # Tozer and Handy v. 2
'XC_GGA_XC_TH3'                :  156, # Tozer and Handy v. 3
'XC_GGA_XC_TH4'                :  157, # Tozer and Handy v. 4
'XC_GGA_X_C09X'                :  158, # C09x to be used with the VdW of Rutgers-Chalmers
'XC_GGA_C_SOGGA11_X'           :  159, # To be used with HYB_GGA_X_SOGGA11_X
'XC_GGA_X_LB'                  :  160, # van Leeuwen & Baerends
'XC_GGA_XC_HCTH_93'            :  161, # HCTH functional fitted to 93 molecules
'XC_GGA_XC_HCTH_120'           :  162, # HCTH functional fitted to 120 molecules
'XC_GGA_XC_HCTH_147'           :  163, # HCTH functional fitted to 147 molecules
'XC_GGA_XC_HCTH_407'           :  164, # HCTH functional fitted to 407 molecules
'XC_GGA_XC_EDF1'               :  165, # Empirical functionals from Adamson, Gill, and Pople
'XC_GGA_XC_XLYP'               :  166, # XLYP functional
'XC_GGA_XC_B97_D'              :  170, # Grimme functional to be used with C6 vdW term
'XC_GGA_XC_PBE1W'              :  173, # Functionals fitted for water
'XC_GGA_XC_MPWLYP1W'           :  174, # Functionals fitted for water
'XC_GGA_XC_PBELYP1W'           :  175, # Functionals fitted for water
'XC_GGA_X_LBM'                 :  182, # van Leeuwen & Baerends modified
'XC_GGA_X_OL2'                 :  183, # Exchange form based on Ou-Yang and Levy v.2
'XC_GGA_X_APBE'                :  184, # mu fixed from the semiclassical neutral atom
'XC_GGA_K_APBE'                :  185, # mu fixed from the semiclassical neutral atom
'XC_GGA_C_APBE'                :  186, # mu fixed from the semiclassical neutral atom
'XC_GGA_K_TW1'                 :  187, # Tran and Wesolowski set 1 (Table II)
'XC_GGA_K_TW2'                 :  188, # Tran and Wesolowski set 2 (Table II)
'XC_GGA_K_TW3'                 :  189, # Tran and Wesolowski set 3 (Table II)
'XC_GGA_K_TW4'                 :  190, # Tran and Wesolowski set 4 (Table II)
'XC_GGA_X_HTBS'                :  191, # Haas, Tran, Blaha, and Schwarz
'XC_GGA_X_AIRY'                :  192, # Constantin et al based on the Airy gas
'XC_GGA_X_LAG'                 :  193, # Local Airy Gas
'XC_GGA_XC_MOHLYP'             :  194, # Functional for organometallic chemistry
'XC_GGA_XC_MOHLYP2'            :  195, # Functional for barrier heights
'XC_GGA_XC_TH_FL'              :  196, # Tozer and Handy v. FL
'XC_GGA_XC_TH_FC'              :  197, # Tozer and Handy v. FC
'XC_GGA_XC_TH_FCFO'            :  198, # Tozer and Handy v. FCFO
'XC_GGA_XC_TH_FCO'             :  199, # Tozer and Handy v. FCO
'XC_GGA_C_OPTC'                :  200, # Optimized correlation functional of Cohen and Handy
'XC_GGA_C_PBELOC'              :  246, # Semilocal dynamical correlation
'XC_GGA_XC_VV10'               :  255, # Vydrov and Van Voorhis
'XC_GGA_C_PBEFE'               :  258, # PBE for formation energies
'XC_GGA_C_OP_PW91'             :  262, # one-parameter progressive functional (PW91 version)
'XC_GGA_X_PBEFE'               :  265, # PBE for formation energies
'XC_GGA_X_CAP'                 :  270, # Correct Asymptotic Potential
'XC_GGA_K_VW'                  :  500, # von Weiszaecker functional
'XC_GGA_K_GE2'                 :  501, # Second-order gradient expansion (l = 1/9)
'XC_GGA_K_GOLDEN'              :  502, # TF-lambda-vW form by Golden (l = 13/45)
'XC_GGA_K_YT65'                :  503, # TF-lambda-vW form by Yonei and Tomishima (l = 1/5)
'XC_GGA_K_BALTIN'              :  504, # TF-lambda-vW form by Baltin (l = 5/9)
'XC_GGA_K_LIEB'                :  505, # TF-lambda-vW form by Lieb (l = 0.185909191)
'XC_GGA_K_ABSP1'               :  506, # gamma-TFvW form by Acharya et al [g = 1 - 1.412/N^(1/3)]
'XC_GGA_K_ABSP2'               :  507, # gamma-TFvW form by Acharya et al [g = 1 - 1.332/N^(1/3)]
'XC_GGA_K_GR'                  :  508, # gamma-TFvW form by Gazquez and Robles
'XC_GGA_K_LUDENA'              :  509, # gamma-TFvW form by Ludena
'XC_GGA_K_GP85'                :  510, # gamma-TFvW form by Ghosh and Parr
'XC_GGA_K_PEARSON'             :  511, # Pearson
'XC_GGA_K_OL1'                 :  512, # Ou-Yang and Levy v.1
'XC_GGA_K_OL2'                 :  513, # Ou-Yang and Levy v.2
'XC_GGA_K_FR_B88'              :  514, # Fuentealba & Reyes (B88 version)
'XC_GGA_K_FR_PW86'             :  515, # Fuentealba & Reyes (PW86 version)
'XC_GGA_K_DK'                  :  516, # DePristo and Kress
'XC_GGA_K_PERDEW'              :  517, # Perdew
'XC_GGA_K_VSK'                 :  518, # Vitos, Skriver, and Kollar
'XC_GGA_K_VJKS'                :  519, # Vitos, Johansson, Kollar, and Skriver
'XC_GGA_K_ERNZERHOF'           :  520, # Ernzerhof
'XC_GGA_K_LC94'                :  521, # Lembarki & Chermette
'XC_GGA_K_LLP'                 :  522, # Lee, Lee & Parr
'XC_GGA_K_THAKKAR'             :  523, # Thakkar 1992
'XC_GGA_X_WPBEH'               :  524, # short-range version of the PBE
'XC_GGA_X_HJS_PBE'             :  525, # HJS screened exchange PBE version
'XC_GGA_X_HJS_PBE_SOL'         :  526, # HJS screened exchange PBE_SOL version
'XC_GGA_X_HJS_B88'             :  527, # HJS screened exchange B88 version
'XC_GGA_X_HJS_B97X'            :  528, # HJS screened exchange B97x version
'XC_GGA_X_ITYH'                :  529, # short-range recipe for exchange GGA functionals
'XC_GGA_X_SFAT'                :  530, # short-range recipe for exchange GGA functionals
'XC_HYB_GGA_X_N12_SX'          :   81, # N12-SX functional from Minnesota
'XC_HYB_GGA_XC_B97_1P'         :  266, # version of B97 by Cohen and Handy
'XC_HYB_GGA_XC_B3PW91'         :  401, # The original (ACM) hybrid of Becke
'XC_HYB_GGA_XC_B3LYP'          :  402, # The (in)famous B3LYP
'XC_HYB_GGA_XC_B3P86'          :  403, # Perdew 86 hybrid similar to B3PW91
'XC_HYB_GGA_XC_O3LYP'          :  404, # hybrid using the optx functional
'XC_HYB_GGA_XC_MPW1K'          :  405, # mixture of mPW91 and PW91 optimized for kinetics
'XC_HYB_GGA_XC_PBEH'           :  406, # aka PBE0 or PBE1PBE
'XC_HYB_GGA_XC_B97'            :  407, # Becke 97
'XC_HYB_GGA_XC_B97_1'          :  408, # Becke 97-1
'XC_HYB_GGA_XC_B97_2'          :  410, # Becke 97-2
'XC_HYB_GGA_XC_X3LYP'          :  411, # hybrid by Xu and Goddard
'XC_HYB_GGA_XC_B1WC'           :  412, # Becke 1-parameter mixture of WC and PBE
'XC_HYB_GGA_XC_B97_K'          :  413, # Boese-Martin for Kinetics
'XC_HYB_GGA_XC_B97_3'          :  414, # Becke 97-3
'XC_HYB_GGA_XC_MPW3PW'         :  415, # mixture with the mPW functional
'XC_HYB_GGA_XC_B1LYP'          :  416, # Becke 1-parameter mixture of B88 and LYP
'XC_HYB_GGA_XC_B1PW91'         :  417, # Becke 1-parameter mixture of B88 and PW91
'XC_HYB_GGA_XC_MPW1PW'         :  418, # Becke 1-parameter mixture of mPW91 and PW91
'XC_HYB_GGA_XC_MPW3LYP'        :  419, # mixture of mPW and LYP
'XC_HYB_GGA_XC_SB98_1A'        :  420, # Schmider-Becke 98 parameterization 1a
'XC_HYB_GGA_XC_SB98_1B'        :  421, # Schmider-Becke 98 parameterization 1b
'XC_HYB_GGA_XC_SB98_1C'        :  422, # Schmider-Becke 98 parameterization 1c
'XC_HYB_GGA_XC_SB98_2A'        :  423, # Schmider-Becke 98 parameterization 2a
'XC_HYB_GGA_XC_SB98_2B'        :  424, # Schmider-Becke 98 parameterization 2b
'XC_HYB_GGA_XC_SB98_2C'        :  425, # Schmider-Becke 98 parameterization 2c
'XC_HYB_GGA_X_SOGGA11_X'       :  426, # Hybrid based on SOGGA11 form
'XC_HYB_GGA_XC_HSE03'          :  427, # the 2003 version of the screened hybrid HSE
'XC_HYB_GGA_XC_HSE06'          :  428, # the 2006 version of the screened hybrid HSE
'XC_HYB_GGA_XC_HJS_PBE'        :  429, # HJS hybrid screened exchange PBE version
'XC_HYB_GGA_XC_HJS_PBE_SOL'    :  430, # HJS hybrid screened exchange PBE_SOL version
'XC_HYB_GGA_XC_HJS_B88'        :  431, # HJS hybrid screened exchange B88 version
'XC_HYB_GGA_XC_HJS_B97X'       :  432, # HJS hybrid screened exchange B97x version
'XC_HYB_GGA_XC_CAM_B3LYP'      :  433, # CAM version of B3LYP
'XC_HYB_GGA_XC_TUNED_CAM_B3LYP':  434, # CAM version of B3LYP tuned for excitations
'XC_HYB_GGA_XC_BHANDH'         :  435, # Becke half-and-half
'XC_HYB_GGA_XC_BHANDHLYP'      :  436, # Becke half-and-half with B88 exchange
'XC_HYB_GGA_XC_MB3LYP_RC04'    :  437, # B3LYP with RC04 LDA
'XC_HYB_GGA_XC_MPWLYP1M'       :  453, # MPW with 1 par. for metals/LYP
'XC_HYB_GGA_XC_REVB3LYP'       :  454, # Revised B3LYP
'XC_HYB_GGA_XC_CAMY_BLYP'      :  455, # BLYP with yukawa screening
'XC_HYB_GGA_XC_PBE0_13'        :  456, # PBE0-1/3
'XC_HYB_GGA_XC_B3LYPS'         :  459, # B3LYP* functional
'XC_HYB_GGA_XC_WB97'           :  463, # Chai and Head-Gordon
'XC_HYB_GGA_XC_WB97X'          :  464, # Chai and Head-Gordon
'XC_HYB_GGA_XC_LRC_WPBEH'      :  465, # Long-range corrected functional by Rorhdanz et al
'XC_HYB_GGA_XC_WB97X_V'        :  466, # Mardirossian and Head-Gordon
'XC_HYB_GGA_XC_LCY_PBE'        :  467, # PBE with yukawa screening
'XC_HYB_GGA_XC_LCY_BLYP'       :  468, # BLYP with yukawa screening
'XC_HYB_GGA_XC_LC_VV10'        :  469, # Vydrov and Van Voorhis
'XC_HYB_GGA_XC_CAMY_B3LYP'     :  470, # B3LYP with Yukawa screening
'XC_HYB_GGA_XC_WB97X_D'        :  471, # Chai and Head-Gordon
'XC_HYB_GGA_XC_HPBEINT'        :  472, # hPBEint
'XC_HYB_GGA_XC_LRC_WPBE'       :  473, # Long-range corrected functional by Rorhdanz et al
'XC_HYB_GGA_XC_B3LYP5'         :  475, # B3LYP with VWN functional 5 instead of RPA
'XC_HYB_GGA_XC_EDF2'           :  476, # Empirical functional from Lin, George and Gill
'XC_HYB_GGA_XC_CAP0'           :  477, # Correct Asymptotic Potential hybrid
'XC_MGGA_C_DLDF'               :   37, # Dispersionless Density Functional
'XC_MGGA_XC_ZLP'               :   42, # Zhao, Levy & Parr, Eq. (21)
'XC_MGGA_XC_OTPSS_D'           :   64, # oTPSS_D functional of Goerigk and Grimme
'XC_MGGA_C_CS'                 :   72, # Colle and Salvetti
'XC_MGGA_C_MN12_SX'            :   73, # Worker for MN12-SX functional
'XC_MGGA_C_MN12_L'             :   74, # MN12-L functional from Minnesota
'XC_MGGA_C_M11_L'              :   75, # M11-L functional from Minnesota
'XC_MGGA_C_M11'                :   76, # Worker for M11 functional
'XC_MGGA_C_M08_SO'             :   77, # Worker for M08-SO functional
'XC_MGGA_C_M08_HX'             :   78, # Worker for M08-HX functional
'XC_MGGA_X_LTA'                :  201, # Local tau approximation of Ernzerhof & Scuseria
'XC_MGGA_X_TPSS'               :  202, # Perdew, Tao, Staroverov & Scuseria exchange
'XC_MGGA_X_M06_L'              :  203, # M06-Local functional of Minnesota
'XC_MGGA_X_GVT4'               :  204, # GVT4 from Van Voorhis and Scuseria
'XC_MGGA_X_TAU_HCTH'           :  205, # tau-HCTH from Boese and Handy
'XC_MGGA_X_BR89'               :  206, # Becke-Roussel 89
'XC_MGGA_X_BJ06'               :  207, # Becke & Johnson correction to Becke-Roussel 89
'XC_MGGA_X_TB09'               :  208, # Tran & Blaha correction to Becke & Johnson
'XC_MGGA_X_RPP09'              :  209, # Rasanen, Pittalis, and Proetto correction to Becke & Johnson
'XC_MGGA_X_2D_PRHG07'          :  210, # Pittalis, Rasanen, Helbig, Gross Exchange Functional
'XC_MGGA_X_2D_PRHG07_PRP10'    :  211, # PRGH07 with PRP10 correction
'XC_MGGA_X_REVTPSS'            :  212, # revised Perdew, Tao, Staroverov & Scuseria exchange
'XC_MGGA_X_PKZB'               :  213, # Perdew, Kurth, Zupan, and Blaha
'XC_MGGA_X_M05'                :  214, # Worker for M05 functional
'XC_MGGA_X_M05_2X'             :  215, # Worker for M05-2X functional
'XC_MGGA_X_M06_HF'             :  216, # Worker for M06-HF functional
'XC_MGGA_X_M06'                :  217, # Worker for M06 functional
'XC_MGGA_X_M06_2X'             :  218, # Worker for M06-2X functional
'XC_MGGA_X_M08_HX'             :  219, # Worker for M08-HX functional
'XC_MGGA_X_M08_SO'             :  220, # Worker for M08-SO functional
'XC_MGGA_X_MS0'                :  221, # MS exchange of Sun, Xiao, and Ruzsinszky
'XC_MGGA_X_MS1'                :  222, # MS1 exchange of Sun, et al
'XC_MGGA_X_MS2'                :  223, # MS2 exchange of Sun, et al
'XC_MGGA_X_M11'                :  225, # Worker for M11 functional
'XC_MGGA_X_M11_L'              :  226, # M11-L functional from Minnesota
'XC_MGGA_X_MN12_L'             :  227, # MN12-L functional from Minnesota
'XC_MGGA_C_CC06'               :  229, # Cancio and Chou 2006
'XC_MGGA_X_MK00'               :  230, # Exchange for accurate virtual orbital energies
'XC_MGGA_C_TPSS'               :  231, # Perdew, Tao, Staroverov & Scuseria correlation
'XC_MGGA_C_VSXC'               :  232, # VSxc from Van Voorhis and Scuseria (correlation part)
'XC_MGGA_C_M06_L'              :  233, # M06-Local functional from Minnesota
'XC_MGGA_C_M06_HF'             :  234, # Worker for M06-HF functional
'XC_MGGA_C_M06'                :  235, # Worker for M06 functional
'XC_MGGA_C_M06_2X'             :  236, # Worker for M06-2X functional
'XC_MGGA_C_M05'                :  237, # Worker for M05 functional
'XC_MGGA_C_M05_2X'             :  238, # Worker for M05-2X functional
'XC_MGGA_C_PKZB'               :  239, # Perdew, Kurth, Zupan, and Blaha
'XC_MGGA_C_BC95'               :  240, # Becke correlation 95
'XC_MGGA_C_REVTPSS'            :  241, # revised TPSS correlation
'XC_MGGA_XC_TPSSLYP1W'         :  242, # Functionals fitted for water
'XC_MGGA_X_MK00B'              :  243, # Exchange for accurate virtual orbital energies (v. B)
'XC_MGGA_X_BLOC'               :  244, # functional with balanced localization
'XC_MGGA_X_MODTPSS'            :  245, # Modified Perdew, Tao, Staroverov & Scuseria exchange
'XC_MGGA_C_TPSSLOC'            :  247, # Semilocal dynamical correlation
'XC_MGGA_X_MBEEF'              :  249, # mBEEF exchange
'XC_MGGA_X_MBEEFVDW'           :  250, # mBEEF-vdW exchange
'XC_MGGA_XC_B97M_V'            :  254, # Mardirossian and Head-Gordon
'XC_MGGA_X_MVS'                :  257, # MVS exchange of Sun, Perdew, and Ruzsinszky
'XC_MGGA_X_MN15_L'             :  260, # MN15-L functional from Minnesota
'XC_MGGA_C_MN15_L'             :  261, # MN15-L functional from Minnesota
'XC_MGGA_X_SCAN'               :  263, # SCAN exchange of Sun, Ruzsinszky, and Perdew
'XC_MGGA_C_SCAN'               :  267, # SCAN correlation
'XC_MGGA_C_MN15'               :  269, # MN15 functional from Minnesota
'XC_HYB_MGGA_X_DLDF'           :   36, # Dispersionless Density Functional
'XC_HYB_MGGA_X_MS2H'           :  224, # MS2 hybrid exchange of Sun, et al
'XC_HYB_MGGA_X_MN12_SX'        :  248, # MN12-SX hybrid functional from Minnesota
'XC_HYB_MGGA_X_SCAN0'          :  264, # SCAN hybrid
'XC_HYB_MGGA_X_MN15'           :  268, # MN15 functional from Minnesota
'XC_HYB_MGGA_XC_M05'           :  438, # M05 functional from Minnesota
'XC_HYB_MGGA_XC_M05_2X'        :  439, # M05-2X functional from Minnesota
'XC_HYB_MGGA_XC_B88B95'        :  440, # Mixture of B88 with BC95 (B1B95)
'XC_HYB_MGGA_XC_B86B95'        :  441, # Mixture of B86 with BC95
'XC_HYB_MGGA_XC_PW86B95'       :  442, # Mixture of PW86 with BC95
'XC_HYB_MGGA_XC_BB1K'          :  443, # Mixture of B88 with BC95 from Zhao and Truhlar
'XC_HYB_MGGA_XC_M06_HF'        :  444, # M06-HF functional from Minnesota
'XC_HYB_MGGA_XC_MPW1B95'       :  445, # Mixture of mPW91 with BC95 from Zhao and Truhlar
'XC_HYB_MGGA_XC_MPWB1K'        :  446, # Mixture of mPW91 with BC95 for kinetics
'XC_HYB_MGGA_XC_X1B95'         :  447, # Mixture of X with BC95
'XC_HYB_MGGA_XC_XB1K'          :  448, # Mixture of X with BC95 for kinetics
'XC_HYB_MGGA_XC_M06'           :  449, # M06 functional from Minnesota
'XC_HYB_MGGA_XC_M06_2X'        :  450, # M06-2X functional from Minnesota
'XC_HYB_MGGA_XC_PW6B95'        :  451, # Mixture of PW91 with BC95 from Zhao and Truhlar
'XC_HYB_MGGA_XC_PWB6K'         :  452, # Mixture of PW91 with BC95 from Zhao and Truhlar for kinetics
'XC_HYB_MGGA_XC_TPSSH'         :  457, # TPSS hybrid
'XC_HYB_MGGA_XC_REVTPSSH'      :  458, # revTPSS hybrid
'XC_HYB_MGGA_XC_M08_HX'        :  460, # M08-HX functional from Minnesota
'XC_HYB_MGGA_XC_M08_SO'        :  461, # M08-SO functional from Minnesota
'XC_HYB_MGGA_XC_M11'           :  462, # M11 functional from Minnesota
'XC_HYB_MGGA_X_MVSH'           :  474, # MVS hybrid
'XC_HYB_MGGA_XC_WB97M_V'       :  531, # Mardirossian and Head-Gordon
#
# alias
#
'LDA'           : 1 ,
'SLATER'        : 1 ,
'VWN3'          : 8,
'VWNRPA'        : 8,
'VWN5'          : 7,
'B88'           : 106,
'BLYP'          : 'B88,LYP',
'BP86'          : 'B88,P86',
'PBE0'          : 406,
'PBE1PBE'       : 406,
'OPTXCORR'      : '0.7344536875999693*SLATER - 0.6984752285760186*OPTX,',
'B3LYP'         : 'B3LYP5',  # VWN5 version
'B3LYP5'        : '.2*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN',
'B3LYPG'        : 402,  # VWN3, used by Gaussian
'B3P86'         : 'B3P865',  # VWN5 version
'B3P865'        : '.2*HF + .08*SLATER + .72*B88, .81*P86 + .19*VWN',
#?'B3P86G'        : 403,  # VWN3, used by Gaussian
'B3P86G'        : '.2*HF + .08*SLATER + .72*B88, .81*P86 + .19*VWN3',
'B3PW91'        : 'B3PW915',
'B3PW915'       : '.2*HF + .08*SLATER + .72*B88, .81*PW91 + .19*VWN',
#'B3PW91G'       : '.2*HF + .08*SLATER + .72*B88, .81*PW91 + .19*VWN3',
'B3PW91G'       : 401,
#'O3LYP5'        : '.1161*HF + .9262*SLATER + .8133*OPTXCORR, .81*LYP + .19*VWN5',
#'O3LYPG'        : '.1161*HF + .9262*SLATER + .8133*OPTXCORR, .81*LYP + .19*VWN3',
'O3LYP'         : 404, # in libxc == '.1161*HF + 0.071006917*SLATER + .8133*OPTX, .81*LYP + .19*VWN5', may be erroreous
'MPW3PW'        : 'MPW3PW5',  # VWN5 version
'MPW3PW5'       : '.2*HF + .08*SLATER + .72*MPW91, .81*PW91 + .19*VWN',
'MPW3PWG'       : 415,  # VWN3, used by Gaussian
'MPW3LYP'       : 'MPW3LYP5',  # VWN5 version
'MPW3LYP5'      : '.218*HF + .073*SLATER + .709*MPW91, .871*LYP + .129*VWN',
'MPW3LYPG'      : 419,  # VWN3, used by Gaussian
'REVB3LYP'      : 'REVB3LYP5',  # VWN5 version
'REVB3LYP5'     : '.2*HF + .13*SLATER + .67*B88, .84*LYP + .16*VWN',
'REVB3LYPG'     : 454,  # VWN3, used by Gaussian
'X3LYP'         : 'X3LYP5',  # VWN5 version
'X3LYP5'        : '.218*HF + .073*SLATER + .542385*B88 + .166615*PW91, .871*LYP + .129*VWN',
'X3LYPG'        : 411,  # VWN3, used by Gaussian
'CAMB3LYP'      : 'XC_HYB_GGA_XC_CAM_B3LYP',
'CAMYBLYP'      : 'XC_HYB_GGA_XC_CAMY_BLYP',
'CAMYB3LYP'     : 'XC_HYB_GGA_XC_CAMY_B3LYP',
'B5050LYP'      : '.5*HF + .08*SLATER + .42*B88, .81*LYP + .19*VWN',
'MPW1LYP'       : '.25*HF + .75*MPW91, LYP',
'MPW1PBE'       : '.25*HF + .75*MPW91, PBE',
'PBE50'         : '.5*HF + .5*PBE, PBE',
'REVPBE0'       : '.25*HF + .75*PBE_R, PBE',
'B1B95'         : 440,
'TPSS0'         : '.25*HF + .75*TPSS, TPSS',
}

def _xc_key_without_underscore(xc_keys):
    new_xc = []
    for key in xc_keys:
        if key[:3] == 'XC_':
            for delimeter in ('_XC_', '_X_', '_C_', '_K_'):
                if delimeter in key:
                    key0, key1 = key.split(delimeter)
                    new_key1 = key1.replace('_', '').replace('-', '')
                    if key1 != new_key1:
                        new_xc.append((key0+delimeter+new_key1, XC_CODES[key]))
                    break
    return new_xc
XC_CODES.update(_xc_key_without_underscore(XC_CODES))
del(_xc_key_without_underscore)

XC_KEYS = set(XC_CODES.keys())

# Some XC functionals have conventional name, like M06-L means M06-L for X
# functional and M06-L for C functional, PBE mean PBE-X plus PBE-C. If the
# conventional name was placed in the XC_CODES, it may lead to recursive
# reference when parsing the xc description.  These names (as exceptions of
# XC_CODES) are listed in XC_ALIAS below and they should be treated as a
# shortcut for XC functional.
XC_ALIAS = {
    # Conventional name : name in XC_CODES
    'BLYP'              : 'B88,LYP',
    'BP86'              : 'B88,P86',
    'PW91'              : 'PW91,PW91',
    'PBE'               : 'PBE,PBE',
    'REVPBE'            : 'PBE_R,PBE',
    'PBESOL'            : 'PBE_SOL,PBE_SOL',
    'PKZB'              : 'PKZB,PKZB',
    'TPSS'              : 'TPSS,TPSS',
    'REVTPSS'           : 'REVTPSS,REVTPSS',
    'SCAN'              : 'SCAN,SCAN',
    'SOGGA'             : 'SOGGA,PBE',
    'BLOC'              : 'BLOC,TPSSLOC',
    'OLYP'              : 'OPTX,LYP',
    'RPBE'              : 'RPBE,PBE',
    'BPBE'              : 'B88,PBE',
    'MPW91'             : 'MPW91,PW91',
    'HFLYP'             : 'HF,LYP',
    'HFPW92'            : 'HF,PW_MOD',
    'SPW92'             : 'SLATER,PW_MOD',
    'SVWN'              : 'SLATER,VWN',
    'MS0'               : 'MS0,REGTPSS',
    'MS1'               : 'MS1,REGTPSS',
    'MS2'               : 'MS2,REGTPSS',
    'MS2H'              : 'MS2H,REGTPSS',
    'MVS'               : 'MVS,REGTPSS',
    'MVSH'              : 'MVSH,REGTPSS',
    'SOGGA11'           : 'SOGGA11,SOGGA11',
    'SOGGA11-X'         : 'SOGGA11_X,SOGGA11_X',
    'KT1'               : 'KT1,VWN',
    'DLDF'              : 'DLDF,DLDF',
    'GAM'               : 'GAM,GAM',
    'M06-L'             : 'M06_L,M06_L',
    'M11-L'             : 'M11_L,M11_L',
    'MN12-L'            : 'MN12_L,MN12_L',
    'MN15-L'            : 'MN15_L,MN15_L',
    'N12'               : 'N12,N12',
    'N12-SX'            : 'N12_SX,N12_SX',
    'MN12-SX'           : 'MN12_SX,MN12_SX',
    'MN15'              : 'MN15,MN15',
    'MBEEF'             : 'MBEEF,PBE_SOL',
    'SCAN0'             : 'SCAN0,SCAN',
    'PBEOP'             : 'PBE,OP_PBE',
    'BOP'               : 'B88,OP_B88',
}
XC_ALIAS.update([(key.replace('-',''), XC_ALIAS[key])
                 for key in XC_ALIAS if '-' in key])

VV10_XC = set(('B97M_V', 'WB97M_V', 'WB97X_V', 'VV10', 'LC_VV10'))

PROBLEMATIC_XC = dict([(XC_CODES[x], x) for x in
                       ('XC_GGA_C_SPBE', 'XC_MGGA_X_TPSS', 'XC_MGGA_X_REVTPSS',
                        'XC_MGGA_C_TPSSLOC', 'XC_HYB_MGGA_XC_TPSSH')])

def xc_type(xc_code):
    if isinstance(xc_code, str):
        if is_nlc(xc_code):
            return 'NLC'
        hyb, fn_facs = parse_xc(xc_code)
    else:
        fn_facs = [(xc_code, 1)]  # mimic fn_facs
    if not fn_facs:
        return 'HF'
    elif all(_itrf.LIBXC_is_lda(ctypes.c_int(xid)) for xid, fac in fn_facs):
        return 'LDA'
    elif any(_itrf.LIBXC_is_meta_gga(ctypes.c_int(xid)) for xid, fac in fn_facs):
        return 'MGGA'
    else:
        # any(_itrf.LIBXC_is_gga(ctypes.c_int(xid)) for xid, fac in fn_facs)
        # include hybrid_xc
        return 'GGA'

def is_lda(xc_code):
    return xc_type(xc_code) == 'LDA'

def is_hybrid_xc(xc_code):
    if isinstance(xc_code, str):
        if xc_code.isdigit():
            return _itrf.LIBXC_is_hybrid(ctypes.c_int(int(xc_code)))
        else:
            return ('HF' in xc_code or hybrid_coeff(xc_code) != 0)
    elif isinstance(xc_code, int):
        return _itrf.LIBXC_is_hybrid(ctypes.c_int(xc_code))
    else:
        return any((is_hybrid_xc(x) for x in xc_code))

def is_meta_gga(xc_code):
    return xc_type(xc_code) == 'MGGA'

def is_gga(xc_code):
    return xc_type(xc_code) == 'GGA'

def is_nlc(xc_code):
    return '__VV10' in xc_code.upper()

def max_deriv_order(xc_code):
    hyb, fn_facs = parse_xc(xc_code)
    if fn_facs:
        return min(_itrf.LIBXC_max_deriv_order(ctypes.c_int(xid)) for xid, fac in fn_facs)
    else:
        return 3

def test_deriv_order(xc_code, deriv, raise_error=False):
    support = deriv <= max_deriv_order(xc_code)
    if not support and raise_error:
        from pyscf.dft import xcfun
        msg = ('libxc library does not support derivative order %d for  %s' %
               (deriv, xc_code))
        try:
            if xcfun.test_deriv_order(xc_code, deriv, raise_error=False):
                msg += ('''
    This functional derivative is supported in the xcfun library.
    The following code can be used to change the libxc library to xcfun library:

        from pyscf.dft import xcfun
        mf._numint.libxc = xcfun
''')
            raise NotImplementedError(msg)
        except KeyError as e:
            sys.stderr.write('\n'+msg+'\n')
            sys.stderr.write('%s not found in xcfun library\n\n' % xc_code)
            raise e
    return support

def hybrid_coeff(xc_code, spin=0):
    '''Support recursively defining hybrid functional
    '''
    hyb, fn_facs = parse_xc(xc_code)
    for xid, fac in fn_facs:
        hyb[0] += fac * _itrf.LIBXC_hybrid_coeff(ctypes.c_int(xid))
    return hyb[0]

def nlc_coeff(xc_code):
    '''Get NLC coefficients
    '''
    hyb, fn_facs = parse_xc(xc_code)
    nlc_pars = [0, 0]
    nlc_tmp = (ctypes.c_double*2)()
    for xid, fac in fn_facs:
        _itrf.LIBXC_nlc_coeff(xid, nlc_tmp)
        nlc_pars[0] += nlc_tmp[0]
        nlc_pars[1] += nlc_tmp[1]
    return nlc_pars

def rsh_coeff(xc_code):
    '''Range-separated parameter and HF exchange components: omega, alpha, beta

    Exc_RSH = c_SR * SR_HFX + c_LR * LR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
            = alpha * HFX + beta * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
            = alpha * LR_HFX + hyb * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec

    SR_HFX = < pi | e^{-omega r_{12}}/r_{12} | iq >
    LR_HFX = < pi | (1-e^{-omega r_{12}})/r_{12} | iq >
    alpha = c_LR
    beta = c_SR - c_LR = hyb - alpha
    '''
    if isinstance(xc_code, str) and ',' in xc_code:
        # Parse only X part for the RSH coefficients.  This is to handle
        # exceptions for C functionals such as M11.
        xc_code = xc_code.split(',')[0] + ','
    hyb, fn_facs = parse_xc(xc_code)

    hyb, alpha, omega = hyb
    beta = hyb - alpha
    rsh_pars = [omega, alpha, beta]
    rsh_tmp = (ctypes.c_double*3)()
    _itrf.LIBXC_rsh_coeff(433, rsh_tmp)
    for xid, fac in fn_facs:
        _itrf.LIBXC_rsh_coeff(xid, rsh_tmp)
        if rsh_pars[0] == 0:
            rsh_pars[0] = rsh_tmp[0]
        elif (rsh_tmp[0] != 0 and rsh_pars[0] != rsh_tmp[0]):
            raise ValueError('Different values of omega found for RSH functionals')
        # libxc-3.0.0 bug https://gitlab.com/libxc/libxc/issues/46
        #if _itrf.LIBXC_is_hybrid(ctypes.c_int(xid)):
        #    rsh_pars[1] += rsh_tmp[1] * fac
        #    rsh_pars[2] += rsh_tmp[2] * fac
        rsh_pars[1] += rsh_tmp[1] * fac
        rsh_pars[2] += rsh_tmp[2] * fac
    return rsh_pars

def parse_xc_name(xc_name='LDA,VWN'):
    '''Convert the XC functional name to libxc library internal ID.
    '''
    fn_facs = parse_xc(xc_name)[1]
    return fn_facs[0][0], fn_facs[1][0]

def parse_xc(description):
    r'''Rules to input functional description:

    * The given functional description must be a one-line string.
    * The functional description is case-insensitive.
    * The functional description string has two parts, separated by ",".  The
      first part describes the exchange functional, the second is the correlation
      functional.

      - If "," was not in string, the entire string is considered as a
        compound XC functional (including both X and C functionals, such as b3lyp).
      - To input only X functional (without C functional), leave the second
        part blank. E.g. description='slater,' means pure LDA functional.
      - To neglect X functional (just apply C functional), leave the first
        part blank. E.g. description=',vwn' means pure VWN functional.
      - If compound XC functional is specified, no matter whehter it is in the
        X part (the string in front of comma) or the C part (the string behind
        comma), both X and C functionals of the compound XC functional will be
        used.

    * The functional name can be placed in arbitrary order.  Two name needs to
      be separated by operators "+" or "-".  Blank spaces are ignored.
      NOTE the parser only reads operators "+" "-" "*".  / is not in support.
    * A functional name can have at most one factor.  If the factor is not
      given, it is set to 1.  Compound functional can be scaled as a unit. For
      example '0.5*b3lyp' is equivalent to
      'HF*0.1 + .04*LDA + .36*B88, .405*LYP + .095*VWN'
    * String "HF" stands for exact exchange (HF K matrix).  Putting "HF" in
      correlation functional part is the same to putting "HF" in exchange
      part.
    * String "RSH" means range-separated operator. Its format is
      RSH(alpha; beta; omega).  Another way to input RSH is to use keywords
      SR_HF and LR_HF: "SR_HF(0.1) * alpha_plus_beta" and "LR_HF(0.1) *
      alpha" where the number in parenthesis is the value of omega.
    * Be careful with the libxc convention on GGA functional, in which the LDA
      contribution has been included.

    Args:
        xc_code : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" was appeared in the string, it stands for the exact exchange.
        rho : ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))

    Kwargs:
        spin : int
            spin polarized if spin > 0
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        ex, vxc, fxc, kxc

        where

        * vxc = (vrho, vsigma, vlapl, vtau) for restricted case

        * vxc for unrestricted case
          | vrho[:,2]   = (u, d)
          | vsigma[:,3] = (uu, ud, dd)
          | vlapl[:,2]  = (u, d)
          | vtau[:,2]   = (u, d)

        * fxc for restricted case:
          (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)

        * fxc for unrestricted case:
          | v2rho2[:,3]     = (u_u, u_d, d_d)
          | v2rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
          | v2sigma2[:,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
          | v2lapl2[:,3]
          | vtau2[:,3]
          | v2rholapl[:,4]
          | v2rhotau[:,4]
          | v2lapltau[:,4]
          | v2sigmalapl[:,6]
          | v2sigmatau[:,6]

        * kxc for restricted case:
          v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
          v3rho2tau, v3rhosigmatau, v3rhotau2, v3sigma2tau, v3sigmatau2, v3tau3

        * kxc for unrestricted case:
          | v3rho3[:,4]       = (u_u_u, u_u_d, u_d_d, d_d_d)
          | v3rho2sigma[:,9]  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
          | v3rhosigma2[:,12] = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
          | v3sigma3[:,10]     = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)
          | v3rho2tau
          | v3rhosigmatau
          | v3rhotau2
          | v3sigma2tau
          | v3sigmatau2
          | v3tau3

        see also libxc_itrf.c
    '''
    hyb = [0, 0, 0]  # hybrid, alpha, omega (== SR_HF, LR_HF, omega)
    if isinstance(description, int):
        return hyb, [(description, 1.)]
    elif not isinstance(description, str): #isinstance(description, (tuple,list)):
        return parse_xc('%s,%s' % tuple(description))

    def assign_omega(omega, hyb_or_sr, lr=0):
        if hyb[2] == omega or omega == 0:
            hyb[0] += hyb_or_sr
            hyb[1] += lr
        elif hyb[2] == 0:
            hyb[0] += hyb_or_sr
            hyb[1] += lr
            hyb[2] = omega
        else:
            raise ValueError('Different values of omega found for RSH functionals')
    fn_facs = []
    def parse_token(token, possible_xc_for):
        if token:
            if token[0] == '-':
                sign = -1
                token = token[1:]
            else:
                sign = 1
            if '*' in token:
                fac, key = token.split('*')
                if fac[0].isalpha():
                    fac, key = key, fac
                fac = sign * float(fac)
            else:
                fac, key = sign, token

            if key[:3] == 'RSH':
# RSH(alpha; beta; omega): Range-separated-hybrid functional
                alpha, beta, omega = [float(x) for x in key[4:-1].split(';')]
                assign_omega(omega, fac*(alpha+beta), fac*alpha)
            elif key == 'HF':
                hyb[0] += fac
                hyb[1] += fac  # also add to LR_HF
            elif 'SR_HF' in key:
                if '(' in key:
                    omega = float(key.split('(')[1].split(')')[0])
                    assign_omega(omega, fac, 0)
                else:  # Assuming this omega the same to the existing omega
                    hyb[0] += fac
            elif 'LR_HF' in key:
                if '(' in key:
                    omega = float(key.split('(')[1].split(')')[0])
                    assign_omega(omega, 0, fac)
                else:
                    hyb[1] += fac  # == alpha
            elif key.isdigit():
                fn_facs.append((int(key), fac))
            else:
                if key in XC_CODES:
                    x_id = XC_CODES[key]
                else:
                    possible_xc = XC_KEYS.intersection(possible_xc_for(key))
                    if possible_xc:
                        if len(possible_xc) > 1:
                            sys.stderr.write('Possible xc_code %s matches %s. '
                                             % (possible_xc, key))
                            for x_id in possible_xc:  # Prefer X functional
                                if '_X_' in x_id:
                                    break
                            else:
                                x_id = possible_xc.pop()
                            sys.stderr.write('XC parser takes %s\n' % x_id)
                            sys.stderr.write('You can add prefix to %s for a '
                                             'specific functional (e.g. X_%s)\n'
                                             % (key, key))
                        else:
                            x_id = possible_xc.pop()
                        x_id = XC_CODES[x_id]
                    else:
                        raise KeyError('Unknown key %s' % key)
                if isinstance(x_id, str):
                    hyb1, fn_facs1 = parse_xc(x_id)
# Recursively scale the composed functional, to support e.g. '0.5*b3lyp'
                    if hyb1[0] != 0 or hyb1[1] != 0:
                        assign_omega(hyb1[2], hyb1[0]*fac, hyb1[1]*fac)
                    fn_facs.extend([(xid, c*fac) for xid, c in fn_facs1])
                elif x_id is None:
                    raise NotImplementedError(key)
                else:
                    fn_facs.append((x_id, fac))
    def possible_x_for(key):
        key1 = key.replace('_', '')
        return set((key, 'XC_'+key,
                    'XC_LDA_X_'+key, 'XC_GGA_X_'+key, 'XC_MGGA_X_'+key,
                    'XC_HYB_GGA_X_'+key, 'XC_HYB_MGGA_X_'+key))
    def possible_xc_for(key):
        return set((key, 'XC_LDA_XC_'+key, 'XC_GGA_XC_'+key, 'XC_MGGA_XC_'+key,
                    'XC_HYB_GGA_XC_'+key, 'XC_HYB_MGGA_XC_'+key))
    def possible_k_for(key):
        return set((key, 'XC_'+key,
                    'XC_LDA_K_'+key, 'XC_GGA_K_'+key,))
    def possible_x_k_for(key):
        return possible_x_for(key).union(possible_k_for(key))
    def possible_c_for(key):
        return set((key, 'XC_'+key,
                    'XC_LDA_C_'+key, 'XC_GGA_C_'+key, 'XC_MGGA_C_'+key))
    def remove_dup(fn_facs):
        fn_ids = []
        facs = []
        n = 0
        for key, val in fn_facs:
            if key in fn_ids:
                facs[fn_ids.index(key)] += val
            else:
                fn_ids.append(key)
                facs.append(val)
                n += 1
        return list(zip(fn_ids, facs))

    description = description.replace(' ','').upper()
    if description in XC_ALIAS:
        description = XC_ALIAS[description]

    if '-' in description:  # To handle e.g. M06-L
        for key in _NAME_WITH_DASH:
            if key in description:
                description = description.replace(key, _NAME_WITH_DASH[key])

    if ',' in description:
        x_code, c_code = description.split(',')
        for token in x_code.replace('-', '+-').split('+'):
            parse_token(token, possible_x_k_for)
        for token in c_code.replace('-', '+-').split('+'):
            parse_token(token, possible_c_for)
    else:
        for token in description.replace('-', '+-').split('+'):
            parse_token(token, possible_xc_for)
    if hyb[2] == 0: # No omega is assigned. LR_HF is 0 for normal Coulomb operator
        hyb[1] = 0
    return hyb, remove_dup(fn_facs)

_NAME_WITH_DASH = {'SR-HF'    : 'SR_HF',
                   'LR-HF'    : 'LR_HF',
                   'OTPSS-D'  : 'OTPSS_D',
                   'B97-1'    : 'B97_1',
                   'B97-2'    : 'B97_2',
                   'B97-3'    : 'B97_3',
                   'B97-K'    : 'B97_K',
                   'B97-D'    : 'B97_D',
                   'HCTH-93'  : 'HCTH_93',
                   'HCTH-120' : 'HCTH_120',
                   'HCTH-147' : 'HCTH_147',
                   'HCTH-407' : 'HCTH_407',
                   'WB97X-D'  : 'WB97X_D',
                   'WB97X-V'  : 'WB97X_V',
                   'WB97M-V'  : 'WB97M_V',
                   'B97M-V'   : 'B97M_V',
                   'M05-2X'   : 'M05_2X',
                   'M06-L'    : 'M06_L',
                   'M06-HF'   : 'M06_HF',
                   'M06-2X'   : 'M06_2X',
                   'M08-HX'   : 'M08_HX',
                   'M08-SO'   : 'M08_SO',
                   'M11-L'    : 'M11_L',
                   'MN12-L'   : 'MN12_L',
                   'MN15-L'   : 'MN15_L',
                   'MN12-SX'  : 'MN12_SX',
                   'N12-SX'   : 'N12_SX',
                   'LRC-WPBE' : 'LRC_WPBE',
                   'LRC-WPBEH': 'LRC_WPBEH',
                   'LC-VV10'  : 'LC_VV10',
                   'CAM-B3LYP': 'CAM_B3LYP'}


def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    r'''Interface to call libxc library to evaluate XC functional, potential
    and functional derivatives.

    * The given functional xc_code must be a one-line string.
    * The functional xc_code is case-insensitive.
    * The functional xc_code string has two parts, separated by ",".  The
      first part describes the exchange functional, the second is the correlation
      functional.

      - If "," not appeared in string, the entire string is considered as X functional.
      - To neglect X functional (just apply C functional), leave blank in the
        first part, eg description=',vwn' for pure VWN functional

    * The functional name can be placed in arbitrary order.  Two name needs to
      be separated by operators "+" or "-".  Blank spaces are ignored.
      NOTE the parser only reads operators "+" "-" "*".  / is not in support.
    * A functional name is associated with one factor.  If the factor is not
      given, it is assumed equaling 1.
    * String "HF" stands for exact exchange (HF K matrix).  It is allowed to
      put in C functional part.
    * Be careful with the libxc convention on GGA functional, in which the LDA
      contribution is included.

    Args:
        xc_code : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" was appeared in the string, it stands for the exact exchange.
        rho : ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))

    Kwargs:
        spin : int
            spin polarized if spin > 0
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        ex, vxc, fxc, kxc

        where

        * vxc = (vrho, vsigma, vlapl, vtau) for restricted case

        * vxc for unrestricted case
          | vrho[:,2]   = (u, d)
          | vsigma[:,3] = (uu, ud, dd)
          | vlapl[:,2]  = (u, d)
          | vtau[:,2]   = (u, d)

        * fxc for restricted case:
          (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)

        * fxc for unrestricted case:
          | v2rho2[:,3]     = (u_u, u_d, d_d)
          | v2rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
          | v2sigma2[:,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
          | v2lapl2[:,3]
          | vtau2[:,3]
          | v2rholapl[:,4]
          | v2rhotau[:,4]
          | v2lapltau[:,4]
          | v2sigmalapl[:,6]
          | v2sigmatau[:,6]

        * kxc for restricted case:
          (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3)

        * kxc for unrestricted case:
          | v3rho3[:,4]       = (u_u_u, u_u_d, u_d_d, d_d_d)
          | v3rho2sigma[:,9]  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
          | v3rhosigma2[:,12] = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
          | v3sigma3[:,10]    = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)

        see also libxc_itrf.c
    '''
    hyb, fn_facs = parse_xc(xc_code)
    return _eval_xc(fn_facs, rho, spin, relativity, deriv, verbose)


SINGULAR_IDS = set((131,  # LYP functions
                    402, 404, 411, 416, 419,   # hybrid LYP functions
                    74 , 75 , 226, 227))       # M11L and MN12L functional
def _eval_xc(fn_facs, rho, spin=0, relativity=0, deriv=1, verbose=None):
    assert(deriv <= 3)
    if spin == 0:
        nspin = 1
        rho_u = rho_d = numpy.asarray(rho, order='C')
    else:
        nspin = 2
        rho_u = numpy.asarray(rho[0], order='C')
        rho_d = numpy.asarray(rho[1], order='C')
    assert(rho_u.dtype == numpy.double)
    assert(rho_d.dtype == numpy.double)

    if rho_u.ndim == 1:
        rho_u = rho_u.reshape(1,-1)
        rho_d = rho_d.reshape(1,-1)
    ngrids = rho_u.shape[1]

    fn_ids = [x[0] for x in fn_facs]
    facs   = [x[1] for x in fn_facs]
    fn_ids_set = set(fn_ids)
    if fn_ids_set.intersection(PROBLEMATIC_XC):
        problem_xc = [PROBLEMATIC_XC[k]
                      for k in fn_ids_set.intersection(PROBLEMATIC_XC)]
        warnings.warn('Libxc functionals %s have large discrepancy to xcfun '
                      'library.\n' % problem_xc)

    if all((is_lda(x) for x in fn_ids)):
        if spin == 0:
            nvar = 1
        else:
            nvar = 2
    elif any((is_meta_gga(x) for x in fn_ids)):
        if spin == 0:
            nvar = 4
        else:
            nvar = 9
    else:  # GGA
        if spin == 0:
            nvar = 2
        else:
            nvar = 5
    outlen = (math.factorial(nvar+deriv) //
              (math.factorial(nvar) * math.factorial(deriv)))
    if SINGULAR_IDS.intersection(fn_ids_set) and deriv > 1:
        non0idx = (rho_u[0] > 1e-10) & (rho_d[0] > 1e-10)
        rho_u = numpy.asarray(rho_u[:,non0idx], order='C')
        rho_d = numpy.asarray(rho_d[:,non0idx], order='C')
        outbuf = numpy.empty((outlen,non0idx.sum()))
    else:
        outbuf = numpy.empty((outlen,ngrids))

    n = len(fn_ids)
    _itrf.LIBXC_eval_xc(ctypes.c_int(n),
                        (ctypes.c_int*n)(*fn_ids), (ctypes.c_double*n)(*facs),
                        ctypes.c_int(nspin),
                        ctypes.c_int(deriv), ctypes.c_int(rho_u.shape[1]),
                        rho_u.ctypes.data_as(ctypes.c_void_p),
                        rho_d.ctypes.data_as(ctypes.c_void_p),
                        outbuf.ctypes.data_as(ctypes.c_void_p))
    if outbuf.shape[1] != ngrids:
        out = numpy.zeros((outlen,ngrids))
        out[:,non0idx] = outbuf
        outbuf = out

    exc = outbuf[0]
    vxc = fxc = kxc = None
    if nvar == 1:  # LDA
        if deriv > 0:
            vxc = (outbuf[1], None, None, None)
        if deriv > 1:
            fxc = (outbuf[2],) + (None,)*9
        if deriv > 2:
            kxc = (outbuf[3], None, None, None)
    elif nvar == 2:
        if spin == 0:  # GGA
            if deriv > 0:
                vxc = (outbuf[1], outbuf[2], None, None)
            if deriv > 1:
                fxc = (outbuf[3], outbuf[4], outbuf[5],) + (None,)*7
            if deriv > 2:
                kxc = outbuf[6:10]
        else:  # LDA
            if deriv > 0:
                vxc = (outbuf[1:3].T, None, None, None)
            if deriv > 1:
                fxc = (outbuf[3:6].T,) + (None,)*9
            if deriv > 2:
                kxc = (outbuf[6:10].T, None, None, None)
    elif nvar == 5:  # GGA
        if deriv > 0:
            vxc = (outbuf[1:3].T, outbuf[3:6].T, None, None)
        if deriv > 1:
            fxc = (outbuf[6:9].T, outbuf[9:15].T, outbuf[15:21].T) + (None,)*7
        if deriv > 2:
            kxc = (outbuf[21:25].T, outbuf[25:34].T, outbuf[34:46].T, outbuf[46:56].T)
    elif nvar == 4:  # MGGA
        if deriv > 0:
            vxc = outbuf[1:5]
        if deriv > 1:
            fxc = outbuf[5:15]
        if deriv > 2:
            kxc = outbuf[15:19]
    elif nvar == 9:  # MGGA
        if deriv > 0:
            vxc = (outbuf[1:3].T, outbuf[3:6].T, outbuf[6:8].T, outbuf[8:10].T)
        if deriv > 1:
            fxc = (outbuf[10:13].T, outbuf[13:19].T, outbuf[19:25].T,
                   outbuf[25:28].T, outbuf[28:31].T, outbuf[31:35].T,
                   outbuf[35:39].T, outbuf[39:43].T, outbuf[43:49].T,
                   outbuf[49:55].T)
    return exc, vxc, fxc, kxc


def define_xc_(ni, description, xctype='LDA', hyb=0, rsh=(0,0,0)):
    '''Define XC functional.  See also :func:`eval_xc` for the rules of input description.

    Args:
        ni : an instance of :class:`NumInt`

        description : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" was appeared in the string, it stands for the exact exchange.

    Kwargs:
        xctype : str
            'LDA' or 'GGA' or 'MGGA'
        hyb : float
            hybrid functional coefficient
        rsh : float
            coefficients for range-separated hybrid functional

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> mf = dft.RKS(mol)
    >>> define_xc_(mf._numint, '.2*HF + .08*LDA + .72*B88, .81*LYP + .19*VWN')
    >>> mf.kernel()
    -76.3783361189611
    >>> define_xc_(mf._numint, 'LDA*.08 + .72*B88 + .2*HF, .81*LYP + .19*VWN')
    >>> mf.kernel()
    -76.3783361189611
    >>> def eval_xc(xc_code, rho, *args, **kwargs):
    ...     exc = 0.01 * rho**2
    ...     vrho = 0.01 * 2 * rho
    ...     vxc = (vrho, None, None, None)
    ...     fxc = None  # 2nd order functional derivative
    ...     kxc = None  # 3rd order functional derivative
    ...     return exc, vxc, fxc, kxc
    >>> define_xc_(mf._numint, eval_xc, xctype='LDA')
    >>> mf.kernel()
    48.8525211046668
    '''
    if isinstance(description, str):
        ni.eval_xc = lambda xc_code, rho, *args, **kwargs: \
                eval_xc(description, rho, *args, **kwargs)
        ni.hybrid_coeff = lambda *args, **kwargs: hybrid_coeff(description)
        ni.rsh_coeff = lambda *args: rsh_coeff(description)
        ni._xc_type = lambda *args: xc_type(description)

    elif callable(description):
        ni.eval_xc = description
        ni.hybrid_coeff = lambda *args, **kwargs: hyb
        ni.rsh_coeff = lambda *args, **kwargs: rsh
        ni._xc_type = lambda *args: xctype
    else:
        raise ValueError('Unknown description %s' % description)
    return ni

def define_xc(ni, description, xctype='LDA', hyb=0, rsh=(0,0,0)):
    return define_xc_(copy.copy(ni), description, xctype, hyb, rsh)
define_xc.__doc__ = define_xc_.__doc__


if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        )#basis = '6311g**',)
    mf = dft.RKS(mol)
    mf._numint.libxc = dft.xcfun
    mf.xc = 'camb3lyp'
    mf.kernel()
    exit()
    mf.xc = 'b88,lyp'
    eref = mf.kernel()

    mf = dft.RKS(mol)
    mf._numint = define_xc(mf._numint, 'BLYP')
    e1 = mf.kernel()
    print(e1 - eref)

    mf = dft.RKS(mol)
    mf._numint = define_xc(mf._numint, 'B3LYP5')
    e1 = mf.kernel()
    print(e1 - -76.4102840115744)
