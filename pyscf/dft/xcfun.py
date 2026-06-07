#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
XC functional, the interface to xcfun (https://github.com/dftlibs/xcfun)
U. Ekstrom et al, J. Chem. Theory Comput., 6, 1971
'''

import ctypes
from functools import lru_cache
import math
import numpy
from pyscf import lib
from pyscf.dft.xc.utils import remove_dup, format_xc_code
from pyscf.dft import xc_deriv
from pyscf import __config__

_itrf = lib.load_library('libxcfun_itrf')

_itrf.xcfun_splash.restype = ctypes.c_char_p
_itrf.xcfun_version.restype = ctypes.c_char_p
_itrf.XCFUN_eval_xc.restype = ctypes.c_int
_itrf.xcfun_enumerate_parameters.restype = ctypes.c_char_p
_itrf.XCFUN_xc_type.restype = ctypes.c_int
_itrf.xcfun_describe_short.restype = ctypes.c_char_p
_itrf.xcfun_describe_short.argtype = [ctypes.c_char_p]
_itrf.xcfun_describe_long.restype = ctypes.c_char_p
_itrf.xcfun_describe_long.argtype = [ctypes.c_char_p]

__version__ = _itrf.xcfun_version().decode("UTF-8")
__reference__ = _itrf.xcfun_splash().decode("UTF-8")

def print_XC_CODES():
    '''
    Dump the built-in xcfun XC_CODES in a readable format.
    '''
    lda_ids = []
    gga_ids = []
    mgga_ids = []
    xc_codes = {}

    print('XC = XC_CODES = {')
    for i in range(78):
        name = _itrf.xcfun_enumerate_parameters(ctypes.c_int(i))
        sdescr = _itrf.xcfun_describe_short(name)
        #ldescr = _itrf.xcfun_describe_long(ctypes.c_int(i))
        if sdescr is not None:
            name = name.decode('UTF-8')
            key = f"'{name}'"
            sdescr = sdescr.decode('UTF-8')
            print(f'{key:<16s}: {i:2d},  #{sdescr}')
            xc_codes[name] = i

        fntype = _itrf.XCFUN_xc_type(ctypes.c_int(i))
        if fntype == 0:
            lda_ids.append(i)
        elif fntype == 1:
            gga_ids.append(i)
        elif fntype == 2:
            mgga_ids.append(i)

    alias = {
        'SLATER': 'SLATERX',
        'LDA'   : 'SLATERX',
        'VWN'   : 'VWN5C',
        'VWN5'  : 'VWN5C',
        'B88'   : 'BECKEX',
        'LYP'   : 'LYPC',
    }
    for k, v in alias.items():
        key = f"'{k}'"
        print(f'{key:<16s}: {xc_codes[v]:2d},  # {v}')
    print('}')
    print('LDA_IDS = %s' % lda_ids)
    print('GGA_IDS = %s' % gga_ids)
    print('MGGA_IDS = %s' % mgga_ids)

XC = XC_CODES = {
'SLATERX'       :  0,  #Slater LDA exchange
'PW86X'         :  1,  #PW86 exchange
'VWN3C'         :  2,  #VWN3 LDA Correlation functional
'VWN5C'         :  3,  #VWN5 LDA Correlation functional
'PBEC'          :  4,  #PBE correlation functional
'PBEX'          :  5,  #PBE Exchange Functional
'BECKEX'        :  6,  #Becke 88 exchange
'BECKECORRX'    :  7,  #Becke 88 exchange correction
'BECKESRX'      :  8,  #Short range Becke 88 exchange
'BECKECAMX'     :  9,  #CAM Becke 88 exchange
'BRX'           : 10,  #Becke-Roussells exchange with jp dependence
'BRC'           : 11,  #Becke-Roussells correlation with jp dependence
'BRXC'          : 12,  #Becke-Roussells correlation with jp dependence
'LDAERFX'       : 13,  #Short-range spin-dependent LDA exchange functional
'LDAERFC'       : 14,  #Short-range spin-dependent LDA correlation functional
'LDAERFC_JT'    : 15,  #Short-range spin-unpolarized LDA correlation functional
'LYPC'          : 16,  #LYP correlation
'OPTX'          : 17,  #OPTX Handy & Cohen exchange
'OPTXCORR'      : 18,  #OPTX Handy & Cohen exchange -- correction part only
'REVPBEX'       : 19,  #Revised PBE Exchange Functional
'RPBEX'         : 20,  #RPBE Exchange Functional
'SPBEC'         : 21,  #sPBE correlation functional
'VWN_PBEC'      : 22,  #PBE correlation functional using VWN LDA correlation.
'KTX'           : 23,  #KT exchange GGA correction
'TFK'           : 24,  #Thomas-Fermi Kinetic Energy Functional
'TW'            : 25,  #von Weizsacker Kinetic Energy Functional
'PW91X'         : 26,  #Perdew-Wang 1991 GGA Exchange Functional
'PW91K'         : 27,  #PW91 GGA Kinetic Energy Functional
'PW92C'         : 28,  #PW92 LDA correlation
'M05X'          : 29,  #M05 exchange
'M05X2X'        : 30,  #M05-2X exchange
'M06X'          : 31,  #M06 exchange
'M06X2X'        : 32,  #M06-2X exchange
'M06LX'         : 33,  #M06-L exchange
'M06HFX'        : 34,  #M06-HF exchange
'M05X2C'        : 35,  #M05-2X Correlation
'M05C'          : 36,  #M05 Correlation
'M06C'          : 37,  #M06 Correlation
'M06HFC'        : 38,  #M06-HF Correlation
'M06LC'         : 39,  #M06-L Correlation
'M06X2C'        : 40,  #M06-2X Correlation
'TPSSC'         : 41,  #TPSS original correlation functional
'TPSSX'         : 42,  #TPSS original exchange functional
'REVTPSSC'      : 43,  #Revised TPSS correlation functional
'REVTPSSX'      : 44,  #Reviewed TPSS exchange functional
'SCANC'         : 45,  #SCAN correlation functional
'SCANX'         : 46,  #SCAN exchange functional
'RSCANC'        : 47,  #rSCAN correlation functional
'RSCANX'        : 48,  #rSCAN exchange functional
'RPPSCANC'      : 49,  #r++SCAN correlation functional
'RPPSCANX'      : 50,  #r++SCAN exchange functional
'R2SCANC'       : 51,  #r2SCAN correlation functional
'R2SCANX'       : 52,  #r2SCAN exchange functional
'R4SCANC'       : 53,  #r4SCAN correlation functional
'R4SCANX'       : 54,  #r4SCAN exchange functional
'PZ81C'         : 55,  #PZ81 LDA correlation
'P86C'          : 56,  #P86C GGA correlation
'P86CORRC'      : 57,  #P86C GGA correlation
'BTK'           : 58,  #Borgoo-Tozer TS
'VWK'           : 59,  #von Weizsaecker kinetic energy
'B97X'          : 60,  #B97 exchange
'B97C'          : 61,  #B97 correlation
'B97_1X'        : 62,  #B97-1 exchange
'B97_1C'        : 63,  #B97-1 correlation
'B97_2X'        : 64,  #B97-2 exchange
'B97_2C'        : 65,  #B97-2 correlation
'CSC'           : 66,  #Colle-Salvetti correlation functional
'APBEC'         : 67,  #APBE correlation functional.
'APBEX'         : 68,  #APBE Exchange Functional
'ZVPBESOLC'     : 69,  #zvPBEsol correlation Functional
#'BLOCX'         : 70,  #BLOC exchange functional
'PBEINTC'       : 71,  #PBEint correlation Functional
'PBEINTX'       : 72,  #PBEint Exchange Functional
'PBELOCC'       : 73,  #PBEloc correlation functional.
'PBESOLX'       : 74,  #PBEsol Exchange Functional
'TPSSLOCC'      : 75,  #TPSSloc correlation functional
'ZVPBEINTC'     : 76,  #zvPBEint correlation Functional
'PW91C'         : 77,  #PW91 Correlation
#
# alias
#
'SLATER'        : 0,  # SLATERX
'LDA'           : 0,  # SLATERX
'VWN'           : 3,  # VWN5C
'VWN5'          : 3,  # VWN5C
'VWN3'          : 2,  # VWN3C
'SVWN'          : 'SLATERX + VWN5',
'B88'           : 6,  # BECKEX
'LYP'           : 16,
'P86'           : 56,
'M052XX'        : 30,  # M05-2X exchange
'M062XX'        : 32,  # M06-2X exchange
'M052XC'        : 35,  # M05-2X Correlation
'M062XC'        : 40,  # M06-2X Correlation
'BLYP'          : 'B88 + LYP',
'BP86'          : 'B88 + P86',  # Becke-Perdew 1986
'BPW91'         : 'B88 + PW91C',
'BPW92'         : 'B88 + PW92C',
'OLYP'          : '2.4832*SLATER - 1.43169*OPTX + LYP',  # CPL, 341, 319
'KT1X'           : 'SLATERX - 0.006*KTX',  # Keal-Tozer 1, JCP, 119, 3015
'KT2XC'         : '1.07173*SLATER - .006*KTX + 0.576727*VWN5',  # Keal-Tozer 2, JCP, 119, 3015
'KT3XC'         : 'SLATERX*1.092 + KTX*-0.004 + OPTXCORR*-0.925452 + LYPC*0.864409',  # Keal-Tozer 3, JCP, 121, 5654
# == '2.021452*SLATER - .004*KTX - .925452*OPTX + .864409*LYP',
'PBE0'          : '.25*HF + .75*PBEX + PBEC',  # Perdew-Burke-Ernzerhof, JCP, 110, 6158
'PBE1PBE'       : 'PBE0',
'PBEH'          : 'PBE0',
'B3P86'         : 'B3P86G',
'B3P86G'        : '.2*HF + .08*SLATER + .72*B88 + .81*P86C + .19*VWN3C',
'B3P86V5'       : '.2*HF + .08*SLATER + .72*B88 + .81*P86C + .19*VWN5C',
'B3PW91'        : '.2*HF + .08*SLATER + .72*B88 + .81*PW91C + .19*PW92C',
# Note, B3LYP uses VWN3 https://doi.org/10.1016/S0009-2614(97)00207-8.
'B3LYP'         : 'B3LYPG',
'B3LYP5'        : '.2*HF + .08*SLATER + .72*B88 + .81*LYP + .19*VWN5C',
'B3LYPG'        : '.2*HF + .08*SLATER + .72*B88 + .81*LYP + .19*VWN3C', # B3LYP-VWN3 used by Gaussian and libxc
#'O3LYP'         : '.1161*HF + .9262*SLATER + .8133*OPTXCORR + .81*LYP + .19*VWN5C',  # Mol. Phys. 99 607
#'O3LYPG'        : '.1161*HF + .9262*SLATER + .8133*OPTXCORR + .81*LYP + .19*VWN3C',
# libxc implementation as below, see also discussion in https://gitlab.com/libxc/libxc/issues/47
#'O3LYP'         : '.1161*HF + .9262*SLATER + 1.164393477*OPTXCORR + .81*LYP + .19*VWN5C', #1.164393477 = .8133*1.43169
#'O3LYPG'        : '.1161*HF + .9262*SLATER + 1.164393477*OPTXCORR + .81*LYP + .19*VWN3C',
'O3LYP'         : '.1161*HF + 0.071006917*SLATER + .8133*OPTX, .81*LYP + .19*VWN5',  # libxc implementation
'X3LYP'         : 'X3LYPG',
'X3LYPG'        : '.218*HF + .073*SLATER + 0.542385*B88 + .166615*PW91X + .871*LYP + .129*VWN3C',
'X3LYP5'        : '.218*HF + .073*SLATER + 0.542385*B88 + .166615*PW91X + .871*LYP + .129*VWN5C',  # Xu, PNAS, 101, 2673
# Range-separated-hybrid functional: (alpha+beta)*SR_HF(0.33) + alpha*LR_HF(0.33)
# Note default mu of xcfun is 0.4. It can cause discrepancy for CAMB3LYP
'CAMB3LYP'      : '0.19*SR_HF(0.33) + 0.65*LR_HF(0.33) + 0.46*BECKESRX + 0.35*B88 + VWN5C*0.19 + LYPC*0.81',
'CAM_B3LYP'     : 'CAMB3LYP',
'LDAERF'        : 'LDAERFX + LDAERFC',  # Short-range exchange and correlation LDA functional
'B97XC'         : 'B97X + B97C + HF*0.1943',
'B97_1XC'       : 'B97_1X + B97_1C + HF*0.21',
'B97_2XC'       : 'B97_2X + B97_2C + HF*0.21',
'TPSSH'         : '0.1*HF + 0.9*TPSSX + TPSSC',
'TF'            : 'TFK',
}

if getattr(__config__, 'B3LYP_WITH_VWN5', False):
    XC_CODES['B3P86'] = 'B3P86V5'
    XC_CODES['B3LYP'] = 'B3LYP5'

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
    'REVPBE'            : 'REVPBE,PBE',
    'PBESOL'            : 'PBESOL,PBESOL',
#    'PKZB'              : 'PKZB,PKZB',
    'TPSS'              : 'TPSS,TPSS',
    'REVTPSS'           : 'REVTPSS,REVTPSS',
    'SCAN'              : 'SCAN,SCAN',
#    'SOGGA'             : 'SOGGA,PBE',
    #'BLOC'              : 'BLOC,TPSSLOC',
    'OLYP'              : 'OPTX,LYP',
    'RPBE'              : 'RPBE,PBE',
    'BPBE'              : 'B88,PBE',
#    'MPW91'             : 'MPW91,PW91',
    'HFLYP'             : 'HF,LYP',
#    'HFPW92'            : 'HF,PWMOD',
#    'SPW92'             : 'SLATER,PWMOD',
    'SVWN'              : 'SLATER,VWN',
#    'MS0'               : 'MS0,REGTPSS',
#    'MS1'               : 'MS1,REGTPSS',
#    'MS2'               : 'MS2,REGTPSS',
#    'MS2H'              : 'MS2H,REGTPSS',
#    'MVS'               : 'MVS,REGTPSS',
#    'MVSH'              : 'MVSH,REGTPSS',
#    'SOGGA11'           : 'SOGGA11,SOGGA11',
#    'SOGGA11-X'         : 'SOGGA11X,SOGGA11X',
    'KT1'               : 'KT1X,VWN',
#    'DLDF'              : 'DLDF,DLDF',
#    'GAM'               : 'GAM,GAM',
    'M06-L'             : 'M06L,M06L',
#    'M11-L'             : 'M11L,M11L',
#    'MN12-L'            : 'MN12L,MN12L',
#    'MN15-L'            : 'MN15L,MN15L',
#    'N12'               : 'N12,N12',
#    'N12-SX'            : 'N12SX,N12SX',
#    'MN12-SX'           : 'MN12SX,MN12SX',
#    'MN15'              : 'MN15,MN15',
#    'MBEEF'             : 'MBEEF,PBESOL',
#    'SCAN0'             : 'SCAN0,SCAN',
#    'PBEOP'             : 'PBE,OPPBE',
#    'BOP'               : 'B88,OPB88',
    'M05'               : '.28*HF + .72*M05X + M05C',
    'M06'               : '.27*HF +     M06X + M06C',
    #'M05_2X'            : '.56*HF + .44*M05X2X + M06C2X',
    #'M06_2X'            : '.54*HF +     M06X2X + M06C2X',
}
XC_ALIAS.update([(key.replace('-',''), XC_ALIAS[key])
                 for key in XC_ALIAS if '-' in key])

'''
LDA_IDS = set([0, 2, 3, 13, 14, 15, 24, 28, 55])
GGA_IDS = set([1, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26,
               27, 44, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71,
               72, 73, 74, 76, 77])
MGGA_IDS =set([10, 11, 12, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
               42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 66, 70, 75])
'''
HYB_XC = {'PBE0'    , 'PBE1PBE' , 'B3PW91'  , 'B3P86'   , 'B3LYP'   ,
          'B3PW91G' , 'B3P86G'  , 'B3LYPG'  , 'O3LYP'   , 'CAMB3LYP',
          'B97XC'   , 'B97_1XC' , 'B97_2XC' , 'M05XC'   , 'TPSSH'   ,
          'HFLYP'}
RSH_XC = {'CAMB3LYP'}

# The compatibility with the old libxcfun_itrf.so library
try:
    MAX_DERIV_ORDER = ctypes.c_int.in_dll(_itrf, 'XCFUN_max_deriv_order').value
except ValueError:
    MAX_DERIV_ORDER = 3

VV10_XC = {
    'B97M_V'    : (6.0, 0.01),
    'WB97M_V'   : (6.0, 0.01),
    'WB97X_V'   : (6.0, 0.01),
    'VV10'      : (5.9, 0.0093),
    'LC_VV10'   : (6.3, 0.0089),
    'REVSCAN_VV10': (9.8, 0.0093),
    'SCAN_RVV10'  : (15.7, 0.0093),
    'SCAN_VV10'   : (14.0, 0.0093),
    'SCANL_RVV10' : (15.7, 0.0093),
    'SCANL_VV10'  : (14.0, 0.0093),
}
VV10_XC.update([(key.replace('_', ''), val) for key, val in VV10_XC.items()])

@lru_cache(100)
def xc_type(xc_code):
    if xc_code is None:
        return None
    elif isinstance(xc_code, str):
        hyb, fn_facs = parse_xc(xc_code)
    else:
        fn_facs = [(xc_code, 1)]  # mimic fn_facs

    if not fn_facs:
        return 'HF'
    elif all(_itrf.XCFUN_xc_type(ctypes.c_int(xid)) == 0 for xid, val in fn_facs):
        return 'LDA'
    elif any(_itrf.XCFUN_xc_type(ctypes.c_int(xid)) == 2 for xid, val in fn_facs):
        return 'MGGA'
    else:
        # all((xid in GGA_IDS or xid in LDA_IDS for xid, val in fn_fns)):
        # include hybrid_xc and NLC
        return 'GGA'

def is_lda(xc_code):
    return xc_type(xc_code) == 'LDA'

def is_hybrid_xc(xc_code):
    if isinstance(xc_code, str):
        xc_code = xc_code.replace(' ','').upper()
        return ('HF' in xc_code or xc_code in HYB_XC or
                hybrid_coeff(xc_code) != 0)
    elif numpy.issubdtype(type(xc_code), numpy.integer):
        return False
    else:
        return any((is_hybrid_xc(x) for x in xc_code))

def is_meta_gga(xc_code):
    return xc_type(xc_code) == 'MGGA'

def is_gga(xc_code):
    return xc_type(xc_code) == 'GGA'

# Assign a temporary Id to VV10 functionals. parse_xc function needs them to
# parse NLC functionals
XC_CODES.update([(key, 5000+i) for i, key in enumerate(VV10_XC)])
VV10_XC.update([(5000+i, VV10_XC[key]) for i, key in enumerate(VV10_XC)])

def is_nlc(xc_code):
    fn_facs = parse_xc(xc_code)[1]
    return any(xid >= 5000 for xid, c in fn_facs)

def nlc_coeff(xc_code):
    '''Get NLC coefficients
    '''
    xc_code = xc_code.upper()
    if '__VV10' in xc_code:
        raise RuntimeError('Deprecated notation for NLC functional.')

    fn_facs = parse_xc(xc_code)[1]
    nlc_pars = []
    for xid, fac in fn_facs:
        if xid >= 5000:
            nlc_pars.append((VV10_XC[xid], fac))
    return tuple(nlc_pars)

def rsh_coeff(xc_code):
    '''Get Range-separated-hybrid coefficients
    '''
    hyb, fn_facs = parse_xc(xc_code)
    hyb, alpha, omega = hyb
    beta = hyb - alpha
    return omega, alpha, beta

def max_deriv_order(xc_code):
    return MAX_DERIV_ORDER

def test_deriv_order(xc_code, deriv, raise_error=False):
    support = deriv <= max_deriv_order(xc_code)
    if not support and raise_error:
        raise NotImplementedError('xcfun library does not support derivative '
                                  'order %d for  %s' % (deriv, xc_code))
    return support

def hybrid_coeff(xc_code, spin=0):
    hyb, fn_facs = parse_xc(xc_code)
    return hyb[0]

def parse_xc_name(xc_name):
    fn_facs = parse_xc(xc_name)[1]
    return fn_facs[0][0], fn_facs[1][0]

@lru_cache(100)
def parse_xc(description):
    r'''Rules to input functional description:

    * The given functional description must be a one-line string.
    * The functional description is case-insensitive.
    * The functional description string has two parts, separated by ",".  The
      first part describes the exchange functional, the second is the correlation
      functional.


      - If "," not appeared in string, the entire string is treated as the
        name of a compound functional (containing both the exchange and
        the correlation functional) which was declared in the functional
        aliases list. The full list of functional aliases can be obtained by
        calling the function pyscf.dft.xcfun.XC_ALIAS.keys() .

      - To input only X functional (without C functional), leave the second
        part blank. E.g. description='slater,' means a functional with LDA
        contribution only.

      - To neglect the contribution of X functional (just apply C functional),
        leave blank in the first part, e.g. description=',vwn' means a
        functional with VWN only.

      - If compound XC functional is specified, no matter whether it is in the
        X part (the string in front of comma) or the C part (the string behind
        comma), both X and C functionals of the compound XC functional will be
        used.

    * The functional name can be placed in arbitrary order.  Two names need to
      be separated by operators "+" or "-".  Blank spaces are ignored.
      NOTE the parser only reads operators "+" "-" "*".  / is not supported.

    * A functional name can have at most one factor.  If the factor is not
      given, it is set to 1.  Compound functional can be scaled as a unit. For
      example '0.5*b3lyp' is equivalent to
      'HF*0.1 + .04*LDA + .36*B88, .405*LYP + .095*VWN'

    * String "HF" stands for exact exchange (HF K matrix).  "HF" can be put in
      the correlation functional part (after comma). Putting "HF" in the
      correlation part is the same to putting "HF" in the exchange part.

    * String "RSH" means range-separated operator. Its format is
      RSH(omega, alpha, beta).  Another way to input RSH is to use keywords
      SR_HF and LR_HF: "SR_HF(0.1) * alpha_plus_beta" and "LR_HF(0.1) *
      alpha" where the number in parenthesis is the value of omega.

    * Be careful with the convention of GGA functional, in which the LDA
      contribution has been included.
    '''
    hyb = [0, 0, 0]  # hybrid, alpha, omega
    if description is None:
        return tuple(hyb), ()
    elif numpy.issubdtype(type(description), numpy.integer):
        return tuple(hyb), ((description, 1.),)
    elif not isinstance(description, str): #isinstance(description, (tuple,list)):
        return parse_xc('%s,%s' % tuple(description))

    description = description.upper()
    if '-D3' in description or '-D4' in description:
        from pyscf.scf.dispersion import parse_dft
        description, _, _ = parse_dft(description)
        description = description.upper()

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
    def parse_token(token, suffix, search_xc_alias=False):
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
                fac = sign * float(fac.replace('E_', 'E-'))
            else:
                fac, key = sign, token

            if key[:3] == 'RSH':
                # RSH(alpha; beta; omega): Range-separated-hybrid functional
                # See also utils.format_xc_code
                alpha, beta, omega = [float(x) for x in key[4:-1].split(';')]
                assign_omega(omega, fac*(alpha+beta), fac*alpha)
            elif key == 'HF':
                hyb[0] += fac
                hyb[1] += fac  # also add to LR_HF
            elif 'SR_HF' in key or 'SRHF' in key:
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
                if search_xc_alias and key in XC_ALIAS:
                    x_id = XC_ALIAS[key]
                elif key in XC_CODES:
                    x_id = XC_CODES[key]
                elif key+suffix in XC_CODES:
                    x_id = XC_CODES[key+suffix]
                else:
                    raise KeyError('Unknown %s functional  %s' % (suffix, key))
                if isinstance(x_id, str):
                    hyb1, fn_facs1 = parse_xc(x_id)
# Recursively scale the composed functional, to support e.g. '0.5*b3lyp'
                    if hyb1[0] != 0 or hyb1[1] != 0:
                        assign_omega(hyb1[2], hyb1[0]*fac, hyb1[1]*fac)
                    fn_facs.extend([(xid, c*fac) for xid, c in fn_facs1])
                elif x_id is None:
                    raise NotImplementedError('Unknown %s functional  %s' % (suffix, key))
                else:
                    fn_facs.append((x_id, fac))

    description = format_xc_code(description)

    if '-' in description:  # To handle e.g. M06-L
        for key in _NAME_WITH_DASH:
            if key in description:
                description = description.replace(key, _NAME_WITH_DASH[key])

    if ',' in description:
        x_code, c_code = description.split(',')
        for token in x_code.replace('-', '+-').replace(';+', ';').split('+'):
            parse_token(token, 'X')
        for token in c_code.replace('-', '+-').replace(';+', ';').split('+'):
            parse_token(token, 'C')
    else:
        for token in description.replace('-', '+-').replace(';+', ';').split('+'):
            # dftd3 cannot be used in a custom xc description
            assert '-d3' not in token
            parse_token(token, 'XC', search_xc_alias=True)
    if hyb[2] == 0: # No omega is assigned. LR_HF is 0 for normal Coulomb operator
        hyb[1] = 0
    return tuple(hyb), tuple(remove_dup(fn_facs))

_NAME_WITH_DASH = {'SR-HF'  : 'SR_HF',
                   'LR-HF'  : 'LR_HF',
                   'M06-L'  : 'M06L',
                   'M05-2X' : 'M052X',
                   'M06-HF' : 'M06HF',
                   'M06-2X' : 'M062X',
                   'E-'     : 'E_'} # For scientific notation

XC_D0 = 0
XC_D1 = 1
XC_D2 = 2
XC_D3 = 3
XC_D4 = 4

XC_D00 = 0
XC_D10 = 1
XC_D01 = 2
XC_D20 = 3
XC_D11 = 4
XC_D02 = 5
XC_D30 = 6
XC_D21 = 7
XC_D12 = 8
XC_D03 = 9
XC_D40 = 10
XC_D31 = 11
XC_D22 = 12
XC_D13 = 13
XC_D04 = 14

XC_D000 = 0
XC_D100 = 1
XC_D010 = 2
XC_D001 = 3
XC_D200 = 4
XC_D110 = 5
XC_D101 = 6
XC_D020 = 7
XC_D011 = 8
XC_D002 = 9
XC_D300 = 10
XC_D210 = 11
XC_D201 = 12
XC_D120 = 13
XC_D111 = 14
XC_D102 = 15
XC_D030 = 16
XC_D021 = 17
XC_D012 = 18
XC_D003 = 19
XC_D400 = 20
XC_D310 = 21
XC_D301 = 22
XC_D220 = 23
XC_D211 = 24
XC_D202 = 25
XC_D130 = 26
XC_D121 = 27
XC_D112 = 28
XC_D103 = 29
XC_D040 = 30
XC_D031 = 31
XC_D022 = 32
XC_D013 = 33
XC_D004 = 34

XC_D00000 = 0
XC_D10000 = 1
XC_D01000 = 2
XC_D00100 = 3
XC_D00010 = 4
XC_D00001 = 5
XC_D20000 = 6
XC_D11000 = 7
XC_D10100 = 8
XC_D10010 = 9
XC_D10001 = 10
XC_D02000 = 11
XC_D01100 = 12
XC_D01010 = 13
XC_D01001 = 14
XC_D00200 = 15
XC_D00110 = 16
XC_D00101 = 17
XC_D00020 = 18
XC_D00011 = 19
XC_D00002 = 20
XC_D30000 = 21
XC_D21000 = 22
XC_D20100 = 23
XC_D20010 = 24
XC_D20001 = 25
XC_D12000 = 26
XC_D11100 = 27
XC_D11010 = 28
XC_D11001 = 29
XC_D10200 = 30
XC_D10110 = 31
XC_D10101 = 32
XC_D10020 = 33
XC_D10011 = 34
XC_D10002 = 35
XC_D03000 = 36
XC_D02100 = 37
XC_D02010 = 38
XC_D02001 = 39
XC_D01200 = 40
XC_D01110 = 41
XC_D01101 = 42
XC_D01020 = 43
XC_D01011 = 44
XC_D01002 = 45
XC_D00300 = 46
XC_D00210 = 47
XC_D00201 = 48
XC_D00120 = 49
XC_D00111 = 50
XC_D00102 = 51
XC_D00030 = 52
XC_D00021 = 53
XC_D00012 = 54
XC_D00003 = 55
XC_D40000 = 56
XC_D31000 = 57
XC_D30100 = 58
XC_D30010 = 59
XC_D30001 = 60
XC_D22000 = 61
XC_D21100 = 62
XC_D21010 = 63
XC_D21001 = 64
XC_D20200 = 65
XC_D20110 = 66
XC_D20101 = 67
XC_D20020 = 68
XC_D20011 = 69
XC_D20002 = 70
XC_D13000 = 71
XC_D12100 = 72
XC_D12010 = 73
XC_D12001 = 74
XC_D11200 = 75
XC_D11110 = 76
XC_D11101 = 77
XC_D11020 = 78
XC_D11011 = 79
XC_D11002 = 80
XC_D10300 = 81
XC_D10210 = 82
XC_D10201 = 83
XC_D10120 = 84
XC_D10111 = 85
XC_D10102 = 86
XC_D10030 = 87
XC_D10021 = 88
XC_D10012 = 89
XC_D10003 = 90
XC_D04000 = 91
XC_D03100 = 92
XC_D03010 = 93
XC_D03001 = 94
XC_D02200 = 95
XC_D02110 = 96
XC_D02101 = 97
XC_D02020 = 98
XC_D02011 = 99
XC_D02002 = 100
XC_D01300 = 101
XC_D01210 = 102
XC_D01201 = 103
XC_D01120 = 104
XC_D01111 = 105
XC_D01102 = 106
XC_D01030 = 107
XC_D01021 = 108
XC_D01012 = 109
XC_D01003 = 110
XC_D00400 = 111
XC_D00310 = 112
XC_D00301 = 113
XC_D00220 = 114
XC_D00211 = 115
XC_D00202 = 116
XC_D00130 = 117
XC_D00121 = 118
XC_D00112 = 119
XC_D00103 = 120
XC_D00040 = 121
XC_D00031 = 122
XC_D00022 = 123
XC_D00013 = 124
XC_D00004 = 125

XC_D0000000 = 0
XC_D1000000 = 1
XC_D0100000 = 2
XC_D0010000 = 3
XC_D0001000 = 4
XC_D0000100 = 5
XC_D0000010 = 6
XC_D0000001 = 7
XC_D2000000 = 8
XC_D1100000 = 9
XC_D1010000 = 10
XC_D1001000 = 11
XC_D1000100 = 12
XC_D1000010 = 13
XC_D1000001 = 14
XC_D0200000 = 15
XC_D0110000 = 16
XC_D0101000 = 17
XC_D0100100 = 18
XC_D0100010 = 19
XC_D0100001 = 20
XC_D0020000 = 21
XC_D0011000 = 22
XC_D0010100 = 23
XC_D0010010 = 24
XC_D0010001 = 25
XC_D0002000 = 26
XC_D0001100 = 27
XC_D0001010 = 28
XC_D0001001 = 29
XC_D0000200 = 30
XC_D0000110 = 31
XC_D0000101 = 32
XC_D0000020 = 33
XC_D0000011 = 34
XC_D0000002 = 35
XC_D3000000 = 36
XC_D2100000 = 37
XC_D2010000 = 38
XC_D2001000 = 39
XC_D2000100 = 40
XC_D2000010 = 41
XC_D2000001 = 42
XC_D1200000 = 43
XC_D1110000 = 44
XC_D1101000 = 45
XC_D1100100 = 46
XC_D1100010 = 47
XC_D1100001 = 48
XC_D1020000 = 49
XC_D1011000 = 50
XC_D1010100 = 51
XC_D1010010 = 52
XC_D1010001 = 53
XC_D1002000 = 54
XC_D1001100 = 55
XC_D1001010 = 56
XC_D1001001 = 57
XC_D1000200 = 58
XC_D1000110 = 59
XC_D1000101 = 60
XC_D1000020 = 61
XC_D1000011 = 62
XC_D1000002 = 63
XC_D0300000 = 64
XC_D0210000 = 65
XC_D0201000 = 66
XC_D0200100 = 67
XC_D0200010 = 68
XC_D0200001 = 69
XC_D0120000 = 70
XC_D0111000 = 71
XC_D0110100 = 72
XC_D0110010 = 73
XC_D0110001 = 74
XC_D0102000 = 75
XC_D0101100 = 76
XC_D0101010 = 77
XC_D0101001 = 78
XC_D0100200 = 79
XC_D0100110 = 80
XC_D0100101 = 81
XC_D0100020 = 82
XC_D0100011 = 83
XC_D0100002 = 84
XC_D0030000 = 85
XC_D0021000 = 86
XC_D0020100 = 87
XC_D0020010 = 88
XC_D0020001 = 89
XC_D0012000 = 90
XC_D0011100 = 91
XC_D0011010 = 92
XC_D0011001 = 93
XC_D0010200 = 94
XC_D0010110 = 95
XC_D0010101 = 96
XC_D0010020 = 97
XC_D0010011 = 98
XC_D0010002 = 99
XC_D0003000 = 100
XC_D0002100 = 101
XC_D0002010 = 102
XC_D0002001 = 103
XC_D0001200 = 104
XC_D0001110 = 105
XC_D0001101 = 106
XC_D0001020 = 107
XC_D0001011 = 108
XC_D0001002 = 109
XC_D0000300 = 110
XC_D0000210 = 111
XC_D0000201 = 112
XC_D0000120 = 113
XC_D0000111 = 114
XC_D0000102 = 115
XC_D0000030 = 116
XC_D0000021 = 117
XC_D0000012 = 118
XC_D0000003 = 119

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    r'''Interface to call xcfun library to evaluate XC functional, potential
    and functional derivatives. Return derivatives following libxc convention.

    See also :func:`pyscf.dft.libxc.eval_xc`
    '''
    outbuf = eval_xc1(xc_code, rho, spin, deriv, omega)
    exc = outbuf[0]
    vxc = fxc = kxc = None
    xctype = xc_type(xc_code)
    if xctype == 'LDA' and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1]]
        if deriv > 1:
            fxc = [outbuf[2]]
        if deriv > 2:
            kxc = [outbuf[3]]
    elif xctype == 'GGA' and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1], outbuf[2]]
        if deriv > 1:
            fxc = [outbuf[3], outbuf[4], outbuf[5]]
        if deriv > 2:
            kxc = [outbuf[6], outbuf[7], outbuf[8], outbuf[9]]
    elif xctype == 'LDA' and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T]
        if deriv > 1:
            fxc = [outbuf[3:6].T]
        if deriv > 2:
            kxc = [outbuf[6:10].T]
    elif xctype == 'GGA' and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T, outbuf[3:6].T]
        if deriv > 1:
            fxc = [outbuf[[XC_D20000,XC_D11000,XC_D02000]].T,
                   outbuf[[XC_D10100,XC_D10010,XC_D10001,
                           XC_D01100,XC_D01010,XC_D01001]].T,
                   outbuf[[XC_D00200,XC_D00110,XC_D00101,XC_D00020,XC_D00011,XC_D00002]].T]
        if deriv > 2:
            kxc = [outbuf[[XC_D30000,XC_D21000,XC_D12000,XC_D03000]].T,
                   outbuf[[XC_D20100,XC_D20010,XC_D20001,
                           XC_D11100,XC_D11010,XC_D11001,
                           XC_D02100,XC_D02010,XC_D02001]].T,
                   outbuf[[XC_D10200,XC_D10110,XC_D10101,XC_D10020,XC_D10011,XC_D10002,
                           XC_D01200,XC_D01110,XC_D01101,XC_D01020,XC_D01011,XC_D01002]].T,
                   outbuf[[XC_D00300,XC_D00210,XC_D00201,XC_D00120,XC_D00111,
                           XC_D00102,XC_D00030,XC_D00021,XC_D00012,XC_D00003]].T]
# MGGA/MLGGA: Note the MLGGA interface are not implemented. MGGA only needs 3
# input arguments.  To make the interface compatible with libxc, treat MGGA as
# MLGGA
    elif xctype == 'MGGA' and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1], outbuf[2], None, outbuf[3]]
        if deriv > 1:
            fxc = [
                # v2rho2, v2rhosigma, v2sigma2,
                outbuf[XC_D200], outbuf[XC_D110], outbuf[XC_D020],
                # v2lapl2, v2tau2,
                None, outbuf[XC_D002],
                # v2rholapl, v2rhotau,
                None, outbuf[XC_D101],
                # v2lapltau, v2sigmalapl, v2sigmatau,
                None, None, outbuf[XC_D011]]
        if deriv > 2:
            kxc = [
                # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                outbuf[XC_D300], outbuf[XC_D210], outbuf[XC_D120], outbuf[XC_D030],
                # v3rho2lapl, v3rho2tau,
                None, outbuf[XC_D201],
                # v3rhosigmalapl, v3rhosigmatau,
                None, outbuf[XC_D111],
                # v3rholapl2, v3rholapltau, v3rhotau2,
                None, None, outbuf[XC_D102],
                # v3sigma2lapl, v3sigma2tau,
                None, outbuf[XC_D021],
                # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
                None, None, outbuf[XC_D012],
                # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
                None, None, None, outbuf[XC_D003]]
    elif xctype == 'MGGA' and spin == 1:
        if deriv > 0:
            vxc = (outbuf[1:3].T, outbuf[3:6].T, None, outbuf[6:8].T)
        if deriv > 1:
            fxc = [
                # v2rho2, v2rhosigma, v2sigma2,
                outbuf[[XC_D2000000,XC_D1100000,XC_D0200000]].T,
                outbuf[[XC_D1010000,XC_D1001000,XC_D1000100,
                        XC_D0110000,XC_D0101000,XC_D0100100]].T,
                outbuf[[XC_D0020000,XC_D0011000,XC_D0010100,
                        XC_D0002000,XC_D0001100,XC_D0000200]].T,
                # v2lapl2, v2tau2,
                None,
                outbuf[[XC_D0000020,XC_D0000011,XC_D0000002]].T,
                # v2rholapl, v2rhotau,
                None,
                outbuf[[XC_D1000010,XC_D1000001,XC_D0100010,XC_D0100001]].T,
                # v2lapltau, v2sigmalapl, v2sigmatau,
                None, None,
                outbuf[[XC_D0010010,XC_D0010001,XC_D0001010,XC_D0001001,
                        XC_D0000110,XC_D0000101]].T]
        if deriv > 2:
            kxc = [
                # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                outbuf[[XC_D3000000,XC_D2100000,XC_D1200000,XC_D0300000]].T,
                outbuf[[XC_D2010000,XC_D2001000,XC_D2000100,
                        XC_D1110000,XC_D1101000,XC_D1100100,
                        XC_D0210000,XC_D0201000,XC_D0200100]].T,
                outbuf[[XC_D1020000,XC_D1011000,XC_D1010100,XC_D1002000,XC_D1001100,XC_D1000200,
                        XC_D0120000,XC_D0111000,XC_D0110100,XC_D0102000,XC_D0101100,XC_D0100200]].T,
                outbuf[[XC_D0030000,XC_D0021000,XC_D0020100,XC_D0012000,XC_D0011100,
                        XC_D0010200,XC_D0003000,XC_D0002100,XC_D0001200,XC_D0000300]].T,
                # v3rho2lapl, v3rho2tau,
                None,
                outbuf[[XC_D2000010,XC_D2000001,XC_D1100010,XC_D1100001,XC_D0200010,XC_D0200001]].T,
                # v3rhosigmalapl, v3rhosigmatau,
                None,
                outbuf[[XC_D1010010,XC_D1010001,XC_D1001010,XC_D1001001,XC_D1000110,XC_D1000101,
                        XC_D0110010,XC_D0110001,XC_D0101010,XC_D0101001,XC_D0100110,XC_D0100101]].T,
                # v3rholapl2, v3rholapltau, v3rhotau2,
                None, None,
                outbuf[[XC_D1000020,XC_D1000011,XC_D1000002,XC_D0100020,XC_D0100011,XC_D0100002]].T,
                # v3sigma2lapl, v3sigma2tau,
                None,
                outbuf[[XC_D0020010,XC_D0020001,XC_D0011010,XC_D0011001,XC_D0010110,XC_D0010101,
                        XC_D0002010,XC_D0002001,XC_D0001110,XC_D0001101,XC_D0000210,XC_D0000201]].T,
                # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
                None, None,
                outbuf[[XC_D0010020,XC_D0010011,XC_D0010002,
                        XC_D0001020,XC_D0001011,XC_D0001002,
                        XC_D0000120,XC_D0000111,XC_D0000102]].T,
                # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
                None, None, None,
                outbuf[[XC_D0000030,XC_D0000021,XC_D0000012,XC_D0000003]].T]
    return exc, vxc, fxc, kxc

def eval_xc1(xc_code, rho, spin=0, deriv=1, omega=None):
    '''Similar to eval_xc.
    Returns an array with the order of derivatives following xcfun convention.
    '''
    assert deriv <= MAX_DERIV_ORDER
    xctype = xc_type(xc_code)
    assert xctype in ('HF', 'LDA', 'GGA', 'MGGA')

    rho = numpy.asarray(rho, order='C', dtype=numpy.double)
    if xctype == 'MGGA' and rho.shape[-2] == 6:
        rho = numpy.asarray(rho[...,[0,1,2,3,5],:], order='C')

    hyb, fn_facs = parse_xc(xc_code)
    if omega is not None:
        hyb = hyb[:2] + (float(omega),)

    fn_ids = [x[0] for x in fn_facs]
    facs   = [x[1] for x in fn_facs]
    if hyb[2] != 0:
        # Current implementation does not support different omegas for
        # different RSH functionals if there are multiple RSHs
        omega = [hyb[2]] * len(facs)
    else:
        omega = [0] * len(facs)

    nvar, xlen = xc_deriv._XC_NVAR[xctype, spin]
    ngrids = rho.shape[-1]
    rho = rho.reshape(spin+1,nvar,ngrids)
    outlen = lib.comb(xlen+deriv, deriv)
    out = numpy.zeros((ngrids,outlen))
    n = len(fn_ids)
    if n > 0:
        err = _itrf.XCFUN_eval_xc(ctypes.c_int(n),
                                  (ctypes.c_int*n)(*fn_ids),
                                  (ctypes.c_double*n)(*facs),
                                  (ctypes.c_double*n)(*omega),
                                  ctypes.c_int(spin), ctypes.c_int(deriv),
                                  ctypes.c_int(nvar), ctypes.c_int(ngrids),
                                  ctypes.c_int(outlen),
                                  rho.ctypes.data_as(ctypes.c_void_p),
                                  out.ctypes.data_as(ctypes.c_void_p))
        if err != 0:
            raise RuntimeError(f'Failed to eval {xc_code} for deriv={deriv}')
    return lib.transpose(out)

def eval_xc_eff(xc_code, rho, deriv=1, omega=None, spin=None):
    r'''Returns the derivative tensor against the density parameters

    [density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a]

    or spin-polarized density parameters

    [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
     [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].

    It differs from the eval_xc method in the derivatives of non-local part.
    The eval_xc method returns the XC functional derivatives to sigma
    (|\nabla \rho|^2)

    Args:
        rho: 2-dimensional or 3-dimensional array
            Total density or (spin-up, spin-down) densities (and their
            derivatives if GGA or MGGA functionals) on grids

    Kwargs:
        deriv: int
            derivative orders
        omega: float
            define the exponent in the attenuated Coulomb for RSH functional
    '''
    xctype = xc_type(xc_code)
    rho = numpy.asarray(rho, order='C', dtype=numpy.double)
    if xctype == 'MGGA' and rho.shape[-2] == 6:
        rho = numpy.asarray(rho[...,[0,1,2,3,5],:], order='C')

    if spin is None:
        spin_polarized = rho.ndim >= 2 and rho.shape[0] == 2
        if spin_polarized:
            spin = 1
        else:
            spin = 0
    out = eval_xc1(xc_code, rho, spin, deriv, omega)
    return xc_deriv.transform_xc(rho, out, xctype, spin, deriv)

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
        rsh : a list of three floats
            coefficients (omega, alpha, beta) for range-separated hybrid functional.
            omega is the exponent factor in attenuated Coulomb operator e^{-omega r_{12}}/r_{12}
            alpha is the coefficient for long-range part, hybrid coefficient
            can be obtained by alpha + beta

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
    return define_xc_(ni.copy(), description, xctype, hyb, rsh)
define_xc.__doc__ = define_xc_.__doc__
