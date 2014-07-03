#!/usr/bin/env python
# -*- coding: utf-8
#
# File: dft.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Kohn-Sham
'''

__author__ = "Qiming Sun <osirpt.sun@gmail.com>"
__version__ = "$ 0.2 $"

import numpy
from pyscf import gto
from pyscf import lib
import pyscf.lib.logger as log
from pyscf.lib import pyvxc
from pyscf.lib import pycint
import hf


__doc__ = '''Options:
self.chkfile = "/dev/shm/..."
self.fout = "..."
self.diis_space = 6
self.diis_start_cycle = 1
self.damp_factor = 1
self.level_shift_factor = 0
self.scf_threshold = 1e-10
self.max_scf_cycle = 50

self.init_guess(method)         # method = one of "atom", "1e", "chkfile"
self.potential(v, oob)          # v = one of "coulomb", "gaunt"
                                # oob = operator oriented basis level
                                #       1 sp|f> -> |f>
                                #       2 sp|f> -> sr|f>
self.xc_func(func_x,func_c)     # default is LSDA. or HF if not call this function
'''

XC_LDA_X =               1      # Exchange
XC_LDA_C_WIGNER =        2      # Wigner parametrization
XC_LDA_C_RPA =           3      # Random Phase Approximation
XC_LDA_C_HL =            4      # Hedin & Lundqvist
XC_LDA_C_GL =            5      # Gunnarson & Lundqvist
XC_LDA_C_XALPHA =        6      # Slater Xalpha
XC_LDA_C_VWN =           7      # Vosko, Wilk, & Nussair
XC_LDA_C_VWN_RPA =       8      # Vosko, Wilk, & Nussair (RPA)
XC_LDA_C_PZ =            9      # Perdew & Zunger
XC_LDA_C_PZ_MOD =       10      # Perdew & Zunger (Modified)
XC_LDA_C_OB_PZ =        11      # Ortiz & Ballone (PZ)
XC_LDA_C_PW =           12      # Perdew & Wang
XC_LDA_C_PW_MOD =       13      # Perdew & Wang (Modified)
XC_LDA_C_OB_PW =        14      # Ortiz & Ballone (PW)
XC_LDA_C_2D_AMGB =      15      # Attacalite et al
XC_LDA_C_2D_PRM =       16      # Pittalis, Rasanen & Marques correlation in 2D
XC_LDA_C_vBH =          17      # von Barth & Hedin
XC_LDA_C_1D_CSC =       18      # Casula, Sorella, and Senatore 1D correlation
XC_LDA_X_2D =           19      # Exchange in 2D
XC_LDA_XC_TETER93 =     20      # Teter 93 parametrization
XC_LDA_X_1D =           21      # Exchange in 1D
XC_LDA_C_ML1 =          22      # Modified LSD (version 1) of Proynov and Salahub
XC_LDA_C_ML2 =          23      # Modified LSD (version 2) of Proynov and Salahub
XC_LDA_C_GOMBAS =       24      # Gombas parametrization
XC_LDA_C_PW_RPA =       25      # Perdew & Wang fit of the RPA
XC_LDA_K_TF =           50      # Thomas-Fermi kinetic energy functional
XC_LDA_K_LP =           51      # Lee and Parr Gaussian ansatz
XC_GGA_X_PBE =         101      # Perdew, Burke & Ernzerhof exchange
XC_GGA_X_PBE_R =       102      # Perdew, Burke & Ernzerhof exchange (revised)
XC_GGA_X_B86 =         103      # Becke 86 Xalfa,beta,gamma
XC_GGA_X_HERMAN =      104      # Herman et al original GGA
XC_GGA_X_B86_MGC =     105      # Becke 86 Xalfa,beta,gamma (with mod. grad. correction)
XC_GGA_X_B88 =         106      # Becke 88
XC_GGA_X_G96 =         107      # Gill 96
XC_GGA_X_PW86 =        108      # Perdew & Wang 86
XC_GGA_X_PW91 =        109      # Perdew & Wang 91
XC_GGA_X_OPTX =        110      # Handy & Cohen OPTX 01
XC_GGA_X_DK87_R1 =     111      # dePristo & Kress 87 (version R1)
XC_GGA_X_DK87_R2 =     112      # dePristo & Kress 87 (version R2)
XC_GGA_X_LG93 =        113      # Lacks & Gordon 93
XC_GGA_X_FT97_A =      114      # Filatov & Thiel 97 (version A)
XC_GGA_X_FT97_B =      115      # Filatov & Thiel 97 (version B)
XC_GGA_X_PBE_SOL =     116      # Perdew, Burke & Ernzerhof exchange (solids)
XC_GGA_X_RPBE =        117      # Hammer, Hansen & Norskov (PBE-like)
XC_GGA_X_WC =          118      # Wu & Cohen
XC_GGA_X_mPW91 =       119      # Modified form of PW91 by Adamo & Barone
XC_GGA_X_AM05 =        120      # Armiento & Mattsson 05 exchange
XC_GGA_X_PBEA =        121      # Madsen (PBE-like)
XC_GGA_X_MPBE =        122      # Adamo & Barone modification to PBE
XC_GGA_X_XPBE =        123      # xPBE reparametrization by Xu & Goddard
XC_GGA_X_2D_B86_MGC =  124      # Becke 86 MGC for 2D systems
XC_GGA_X_BAYESIAN =    125      # Bayesian best fit for the enhancement factor
XC_GGA_X_PBE_JSJR =    126      # JSJR reparametrization by Pedroza, Silva & Capelle
XC_GGA_X_2D_B88 =      127      # Becke 88 in 2D
XC_GGA_X_2D_B86 =      128      # Becke 86 Xalfa,beta,gamma
XC_GGA_X_2D_PBE =      129      # Perdew, Burke & Ernzerhof exchange in 2D
XC_GGA_C_PBE =         130      # Perdew, Burke & Ernzerhof correlation
XC_GGA_C_LYP =         131      # Lee, Yang & Parr
XC_GGA_C_P86 =         132      # Perdew 86
XC_GGA_C_PBE_SOL =     133      # Perdew, Burke & Ernzerhof correlation SOL
XC_GGA_C_PW91 =        134      # Perdew & Wang 91
XC_GGA_C_AM05 =        135      # Armiento & Mattsson 05 correlation
XC_GGA_C_XPBE =        136      # xPBE reparametrization by Xu & Goddard
XC_GGA_C_LM =          137      # Langreth and Mehl correlation
XC_GGA_C_PBE_JRGX =    138      # JRGX reparametrization by Pedroza, Silva & Capelle
XC_GGA_X_OPTB88_VDW =  139      # Becke 88 reoptimized to be used with vdW functional of Dion et al
XC_GGA_X_PBEK1_VDW =   140      # PBE reparametrization for vdW
XC_GGA_X_OPTPBE_VDW =  141      # PBE reparametrization for vdW
XC_GGA_X_RGE2 =        142      # Regularized PBE
XC_GGA_C_RGE2 =        143      # Regularized PBE
XC_GGA_X_RPW86 =       144      # refitted Perdew & Wang 86
XC_GGA_X_KT1 =         145      # Keal and Tozer version 1
XC_GGA_XC_KT2 =        146      # Keal and Tozer version 2
XC_GGA_C_WL =          147      # Wilson & Levy
XC_GGA_C_WI =          148      # Wilson & Ivanov
XC_GGA_X_MB88 =        149      # Modified Becke 88 for proton transfer
XC_GGA_X_SOGGA =       150      # Second-order generalized gradient approximation
XC_GGA_X_SOGGA11 =     151      # Second-order generalized gradient approximation 2011
XC_GGA_C_SOGGA11 =     152      # Second-order generalized gradient approximation 2011
XC_GGA_C_WI0 =         153      # Wilson & Ivanov initial version
XC_GGA_XC_TH1 =        154      # Tozer and Handy v. 1
XC_GGA_XC_TH2 =        155      # Tozer and Handy v. 2
XC_GGA_XC_TH3 =        156      # Tozer and Handy v. 3
XC_GGA_XC_TH4 =        157      # Tozer and Handy v. 4
XC_GGA_X_C09X =        158      # C09x to be used with the VdW of Rutgers-Chalmers
XC_GGA_C_SOGGA11_X =   159      # To be used with hyb_gga_x_SOGGA11-X
XC_GGA_X_LB =          160      # van Leeuwen & Baerends
XC_GGA_XC_HCTH_93 =    161      # HCTH functional fitted to  93 molecules
XC_GGA_XC_HCTH_120 =   162      # HCTH functional fitted to 120 molecules
XC_GGA_XC_HCTH_147 =   163      # HCTH functional fitted to 147 molecules
XC_GGA_XC_HCTH_407 =   164      # HCTH functional fitted to 147 molecules
XC_GGA_XC_EDF1 =       165      # Empirical functionals from Adamson, Gill, and Pople
XC_GGA_XC_XLYP =       166      # XLYP functional
XC_GGA_XC_B97 =        167      # Becke 97
XC_GGA_XC_B97_1 =      168      # Becke 97-1
XC_GGA_XC_B97_2 =      169      # Becke 97-2
XC_GGA_XC_B97_D =      170      # Grimme functional to be used with C6 vdW term
XC_GGA_XC_B97_K =      171      # Boese-Martin for Kinetics
XC_GGA_XC_B97_3 =      172      # Becke 97-3
XC_GGA_XC_PBE1W =      173      # Functionals fitted for water
XC_GGA_XC_MPWLYP1W =   174      # Functionals fitted for water
XC_GGA_XC_PBELYP1W =   175      # Functionals fitted for water
XC_GGA_XC_SB98_1a =    176      # Schmider-Becke 98 parameterization 1a
XC_GGA_XC_SB98_1b =    177      # Schmider-Becke 98 parameterization 1b
XC_GGA_XC_SB98_1c =    178      # Schmider-Becke 98 parameterization 1c
XC_GGA_XC_SB98_2a =    179      # Schmider-Becke 98 parameterization 2a
XC_GGA_XC_SB98_2b =    180      # Schmider-Becke 98 parameterization 2b
XC_GGA_XC_SB98_2c =    181      # Schmider-Becke 98 parameterization 2c
XC_GGA_X_LBM =         182      # van Leeuwen & Baerends modified
XC_GGA_X_OL2 =         183      # Exchange form based on Ou-Yang and Levy v.2
XC_GGA_X_APBE =        184      # mu fixed from the semiclassical neutral atom
XC_GGA_K_APBE =        185      # mu fixed from the semiclassical neutral atom
XC_GGA_C_APBE =        186      # mu fixed from the semiclassical neutral atom
XC_GGA_K_TW1 =         187      # Tran and Wesolowski set 1 (Table II)
XC_GGA_K_TW2 =         188      # Tran and Wesolowski set 2 (Table II)
XC_GGA_K_TW3 =         189      # Tran and Wesolowski set 3 (Table II)
XC_GGA_K_TW4 =         190      # Tran and Wesolowski set 4 (Table II)
XC_GGA_X_HTBS =        191      # Haas, Tran, Blaha, and Schwarz
XC_GGA_X_AIRY =        192      # Constantin et al based on the Airy gas
XC_GGA_X_LAG =         193      # Local Airy Gas
XC_GGA_XC_MOHLYP =     194      # Functional for organometallic chemistry
XC_GGA_XC_MOHLYP2 =    195      # Functional for barrier heights
XC_GGA_XC_TH_FL =      196      # Tozer and Handy v. FL
XC_GGA_XC_TH_FC =      197      # Tozer and Handy v. FC
XC_GGA_XC_TH_FCFO =    198      # Tozer and Handy v. FCFO
XC_GGA_XC_TH_FCO =     199      # Tozer and Handy v. FCO
XC_GGA_K_VW =          500      # von Weiszaecker functional
XC_GGA_K_GE2 =         501      # Second-order gradient expansion (l = 1/9)
XC_GGA_K_GOLDEN =      502      # TF-lambda-vW form by Golden (l = 13/45)
XC_GGA_K_YT65 =        503      # TF-lambda-vW form by Yonei and Tomishima (l = 1/5)
XC_GGA_K_BALTIN =      504      # TF-lambda-vW form by Baltin (l = 5/9)
XC_GGA_K_LIEB =        505      # TF-lambda-vW form by Lieb (l = 0.185909191)
XC_GGA_K_ABSR1 =       506      # gamma-TFvW form by Acharya et al [g = 1 - 1.412/N^(1/3)]
XC_GGA_K_ABSR2 =       507      # gamma-TFvW form by Acharya et al [g = 1 - 1.332/N^(1/3)]
XC_GGA_K_GR =          508      # gamma-TFvW form by G\'azquez and Robles
XC_GGA_K_LUDENA =      509      # gamma-TFvW form by Lude\~na
XC_GGA_K_GP85 =        510      # gamma-TFvW form by Ghosh and Parr
XC_GGA_K_PEARSON =     511      # Pearson
XC_GGA_K_OL1 =         512      # Ou-Yang and Levy v.1
XC_GGA_K_OL2 =         513      # Ou-Yang and Levy v.2
XC_GGA_K_FR_B88 =      514      # Fuentealba & Reyes (B88 version)
XC_GGA_K_FR_PW86 =     515      # Fuentealba & Reyes (PW86 version)
XC_GGA_K_DK =          516      # DePristo and Kress
XC_GGA_K_PERDEW =      517      # Perdew
XC_GGA_K_VSK =         518      # Vitos, Skriver, and Kollar
XC_GGA_K_VJKS =        519      # Vitos, Johansson, Kollar, and Skriver
XC_GGA_K_ERNZERHOF =   520      # Ernzerhof
XC_GGA_K_LC94 =        521      # Lembarki & Chermette
XC_GGA_K_LLP =         522      # Lee, Lee & Parr
XC_GGA_K_THAKKAR =     523      # Thakkar 1992
XC_HYB_GGA_XC_B3PW91 = 401      # The original hybrid proposed by Becke
XC_HYB_GGA_XC_B3LYP =  402      # The (in)famous B3LYP
XC_HYB_GGA_XC_B3P86 =  403      # Perdew 86 hybrid similar to B3PW91
XC_HYB_GGA_XC_O3LYP =  404      # hybrid using the optx functional
XC_HYB_GGA_XC_mPW1K =  405      # mixture of mPW91 and PW91 optimized for kinetics
XC_HYB_GGA_XC_PBEH =   406      # aka PBE0 or PBE1PBE
XC_HYB_GGA_XC_B97 =    407      # Becke 97
XC_HYB_GGA_XC_B97_1 =  408      # Becke 97-1
XC_HYB_GGA_XC_B97_2 =  410      # Becke 97-2
XC_HYB_GGA_XC_X3LYP =  411      # maybe the best hybrid
XC_HYB_GGA_XC_B1WC =   412      # Becke 1-parameter mixture of WC and PBE
XC_HYB_GGA_XC_B97_K =  413      # Boese-Martin for Kinetics
XC_HYB_GGA_XC_B97_3 =  414      # Becke 97-3
XC_HYB_GGA_XC_mPW3PW = 415      # mixture with the mPW functional
XC_HYB_GGA_XC_B1LYP =  416      # Becke 1-parameter mixture of B88 and LYP
XC_HYB_GGA_XC_B1PW91 = 417      # Becke 1-parameter mixture of B88 and PW91
XC_HYB_GGA_XC_mPW1PW = 418      # Becke 1-parameter mixture of mPW91 and PW91
XC_HYB_GGA_XC_mPW3LYP = 419     # mixture of mPW and LYP
XC_HYB_GGA_XC_SB98_1a = 420     # Schmider-Becke 98 parameterization 1a
XC_HYB_GGA_XC_SB98_1b = 421     # Schmider-Becke 98 parameterization 1b
XC_HYB_GGA_XC_SB98_1c = 422     # Schmider-Becke 98 parameterization 1c
XC_HYB_GGA_XC_SB98_2a = 423     # Schmider-Becke 98 parameterization 2a
XC_HYB_GGA_XC_SB98_2b = 424     # Schmider-Becke 98 parameterization 2b
XC_HYB_GGA_XC_SB98_2c = 425     # Schmider-Becke 98 parameterization 2c
XC_HYB_GGA_X_SOGGA11_X = 426    # Hybrid based on SOGGA11 form
XC_MGGA_X_LTA =        201      # Local tau approximation of Ernzerhof & Scuseria
XC_MGGA_X_TPSS =       202      # Perdew, Tao, Staroverov & Scuseria exchange
XC_MGGA_X_M06L =       203      # Zhao, Truhlar exchange
XC_MGGA_X_GVT4 =       204      # GVT4 from Van Voorhis and Scuseria (exchange part)
XC_MGGA_X_TAU_HCTH =   205      # tau-HCTH from Boese and Handy
XC_MGGA_X_BR89 =       206      # Becke-Roussel 89
XC_MGGA_X_BJ06 =       207      # Becke & Johnson correction to Becke-Roussel 89
XC_MGGA_X_TB09 =       208      # Tran & Blaha correction to Becke & Johnson
XC_MGGA_X_RPP09 =      209      # Rasanen, Pittalis, and Proetto correction to Becke & Johnson
XC_MGGA_X_2D_PRHG07 =  210      # Pittalis, Rasanen, Helbig, Gross Exchange Functional
XC_MGGA_X_2D_PRHG07_PRP10 = 211 # PRGH07 with PRP10 correction
XC_MGGA_C_TPSS =       231      # Perdew, Tao, Staroverov & Scuseria correlation
XC_MGGA_C_VSXC =       232      # VSxc from Van Voorhis and Scuseria (correlation part)

class RKS(hf.SCF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        if mol.nelectron != 1 and mol.nelectron.__mod__(2) is not 0:
            raise ValueError('Invalid electron number %i.' % mol.nelectron)
        hf.SCF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.func_x = XC_LDA_X
        self.func_c = XC_LDA_C_VWN_RPA
        log.debug(self, "initialize pyvxc")
        pyvxc.init_vxc(0, mol._atm, mol._bas, mol._env)

    def __del__(self):
        pyvxc.del_vxc()

    def xc_func(self, x=XC_LDA_X, c=XC_LDA_C_VWN_RPA):
        self.set_xc_func(x, c)

    def set_xc_func(self, x=XC_LDA_X, c=XC_LDA_C_VWN_RPA):
        '''NR Kohn-Sham XC potential'''
        import types
        if isinstance(x, str):
            self.func_x = eval(x)
        elif isinstance(x, int):
            self.func_x = x
        else:
            raise ValueError("X functional error")
        if isinstance(c, str):
            self.func_c = eval(c)
        elif isinstance(c, int):
            self.func_c = c
        else:
            raise ValueError("C functional error")

        def get_eff_potential(mol, dm, dm_last=0, vhf_last=0):
            vj, vxc = self.get_vj_vxc(mol, dm)
            return vj + vxc
        self.get_eff_potential = get_eff_potential
        self.get_eff_potential.__doc__ = \
                'Coulomb + XC functional id (%s,%s)' % (x,c)

    def get_vj_vxc(self, mol, dm):
        self._exc, n, vxc = pyvxc.exc_vxc(self.func_x, self.func_c, dm)
        log.debug(self, "num electrons by numeric integration = %s", n)
        vj, vk = hf.get_vj_vk(pycint.nr_vhf_o3, mol, dm)
        self._ecoul = lib.trace_ab(dm, vj) * .5
        hyb = pyvxc.hybrid_coeff(self.func_x)
        if abs(hyb) > 1e-10:
            vk = vk * hyb * .5
            self._exc -= lib.trace_ab(dm, vk) * .5
            vxc -= vk
        return vj, vxc

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        '''Coulomb + XC functional id (%s,%s) ''' % (self.func_x, self.func_c)
        vj, vxc = self.get_vj_vxc(mol, dm)
        return vj + vxc

    def calc_tot_elec_energy(self, veff, dm, mo_energy, mo_occ):
        sum_mo_energy = numpy.dot(mo_energy, mo_occ)
        coul_dup = lib.trace_ab(dm, veff)
        tot_e = sum_mo_energy - coul_dup + self._ecoul + self._exc
        log.debug(self, "Ecoul = %s  Exc = %s", self._ecoul, self._exc)
        return tot_e, self._ecoul, self._exc



if __name__ == "__main__":
    from pyscf.gto import basis
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = "out_default"

    mol.atom.extend([["He", (0.,0.,0.)], ])
    mol.basis = {
        "He": basis.ccpvdz["He"]}
    mol.grids = { "He": (10, 14),}
    mol.build()

##############
# SCF result
    method = RKS(mol)
    method.init_guess("1e")
    method.xc_func(XC_LDA_X, XC_LDA_C_VWN_RPA)
    energy = method.scf(mol) #-2.85198795793
