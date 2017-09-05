#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
XC functional, the interface to xcfun (https://github.com/dftlibs/xcfun)
U. Ekstrom et al, J. Chem. Theory Comput., 6, 1971
'''

import copy
import ctypes
import math
import numpy
from pyscf import lib

_itrf = lib.load_library('libxcfun_itrf')

XC = XC_CODES = {
'SLATERX'       :  0,  # Slater LDA exchange
'VWN5C'         :  1,  # VWN5 LDA Correlation functional
'BECKEX'        :  2,  # Becke 88 exchange
'BECKECORRX'    :  3,  # Becke 88 exchange correction
'BECKESRX'      :  4,  # Short range Becke 88 exchange
'OPTX'          :  5,  # OPTX Handy & Cohen exchange
'LYPC'          :  6,  # LYP correlation
'PBEX'          :  7,  # PBE Exchange Functional
'REVPBEX'       :  8,  # Revised PBE Exchange Functional
'RPBEX'         :  9,  # RPBE Exchange Functional
'PBEC'          : 10,  # PBE correlation functional
'SPBEC'         : 11,  # sPBE correlation functional
'VWN_PBEC'      : 12,  # PBE correlation functional using VWN LDA correlation.
#'RANGESEP_MU'   : 16,  # Error function range separation parameter (1/a0)
'KTX'           : 17,  # KT exchange GGA correction
#'TFK'           : 18,  # Thomas-Fermi Kinetic Energy Functional
'PW91X'         : 19,  # Perdew-Wang 1991 GGA Exchange Functional
#'PW91K'         : 20,  # PW91 GGA Kinetic Energy Functional
'PW92C'         : 21,  # PW92 LDA correlation
'M05X'          : 22,  # M05 exchange
'M05X2X'        : 23,  # M05-2X exchange
'M06X'          : 24,  # M06 exchange
'M06X2X'        : 25,  # M06-2X exchange
'M06LX'         : 26,  # M06-L exchange
'M06HFX'        : 27,  # M06-HF exchange
'BRX'           : 28,  # BR exchange. Becke-Roussels exchange functional.
'M05X2C'        : 29,  # M05-2X Correlation
'M05C'          : 30,  # M05 Correlation
'M06C'          : 31,  # M06 Correlation
'M06LC'         : 32,  # M06-L Correlation
'M06X2C'        : 33,  # M06-2X Correlation
'TPSSC'         : 34,  # TPSS original correlation functional
'TPSSX'         : 35,  # TPSS original exchange functional
'REVTPSSC'      : 36,  # Revised TPSS correlation functional
'REVTPSSX'      : 37,  # Reviewed TPSS exchange functional
#
# alias
#
'SLATER'        :  0,  # SLATERX
'LDA'           :  0,  # SLATERX
'VWN'           :  1,  # VWN5C
'VWN5'          :  1,  # VWN5C
'B88'           :  2,  # BECKEX
'LYP'           :  6,  # LYP correlation
'P86'           : None,
'BLYP'          : 'BECKEX + LYP',
'BP86'          : None,
'BPW91'         : 'BECKEX + PW91C',
'BPW92'         : 'BECKEX + PW92C',
'OLYP'          : '2.4832*SLATER - 1.43169*OPTX + LYP',  # CPL, 341, 319
'KT1'           : '1.006*SLATER - .006*KTX + VWN5',  # JCP, 119, 3015
'KT2'           : '1.07773*SLATER - .006*KTX + 0.576727*VWN5',  # JCP, 119, 3015
'KT3'           : '2.021452*SLATER - .004*KTX - .925452*OPTX + .864409*LYP',  # JCP, 121, 5654
'PBE0'          : '.25*HF + .75*PBEX + PBEC',  # JCP, 110, 6158
'PBE1PBE'       : 'PBE0',
'B3PW91'        : None,
'B3P86'         : None,
# Note, use VWN5 for B3LYP. It is different to the libxc default B3LYP
'B3LYP'         : 'B3LYP5',
'B3LYP5'        : '.2*HF + .08*SLATER + .72*BECKE + .81*LYP + .19*VWN5',
'B3LYPG'        : None, # B3LYP-VWN3 used by Gaussian and libxc
'O3LYP'         : '.1161*HF + .1129*SLATER + .8133*OPTX + .81*LYP + .19*VWN5',  # Mol. Phys. 99 607
'M062X'         : 'M06X2X, M062XC',
'CAMB3LYP'      : None,
}

LDA_IDS = set([0, 1, 13, 14, 15, 16, 18, 21])
GGA_IDS = set([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 19, 20])
MGGA_IDS = set([22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37])
MLGGA_IDS = set([28])
HYB_XC = set(('PBE0'    , 'PBE1PBE' , 'B3PW91'  , 'B3P86'   , 'B3LYP'   ,
              'B3LYPG'  , 'O3LYP'   , 'M062X'   , 'CAMB3LYP',))
MAX_DERIV_ORDER = 3

def is_lda(xc_code):
    if isinstance(xc_code, str):
        if xc_code.isdigit():
            return int(xc_code) in LDA_IDS
        else:
            return all((xid in LDA_IDS for xid, val in parse_xc(xc_code)[1]))
    elif isinstance(xc_code, int):
        return xc_code in LDA_IDS
    else:
        return all((is_lda(x) for x in xc_code))

def is_hybrid_xc(xc_code):
    if isinstance(xc_code, str):
        return ('HF' in xc_code or
                xc_code in HYB_XC or
                abs(parse_xc(xc_code)[0]) > 1e-14)
    elif isinstance(xc_code, int):
        return False
    else:
        return any((is_hybrid_xc(x) for x in xc_code))

def is_meta_gga(xc_code):
    if isinstance(xc_code, str):
        if xc_code.isdigit():
            xc_code = int(xc_code)
            return xc_code in MGGA_IDS or xc_code in MLGGA_IDS
        else:
            return any((xid in MGGA_IDS or xid in MLGGA_IDS
                        for xid, val in parse_xc(xc_code)[1]))
    elif isinstance(xc_code, int):
        return xc_code in MGGA_IDS or xc_code in MLGGA_IDS
    else:
        return any((is_meta_gga(x) for x in xc_code))

def is_gga(xc_code):
    if isinstance(xc_code, str):
        if xc_code.isdigit():
            xc_code = int(xc_code)
            return xc_code in GGA_IDS
        else:
            xc_fns = parse_xc(xc_code)[1]
            return (all((xid in GGA_IDS or xid in LDA_IDS for xid, val in xc_fns)) and
                    not is_lda(xc_code))
    elif isinstance(xc_code, int):
        return xc_code in GGA_IDS
    else:
        return (all((is_gga(x) or is_lda(x) for x in xc_code)) and
                not is_lda(xc_code))

def max_deriv_order(xc_code):
    hyb, fn_facs = parse_xc(xc_code)
    return MAX_DERIV_ORDER

def test_deriv_order(xc_code, deriv, raise_error=False):
    support = deriv <= max_deriv_order(xc_code)
    if not support and raise_error:
        raise NotImplementedError('xcfun library does not support derivative '
                                  'order %d for  %s' % (deriv, xc_code))
    return support

def hybrid_coeff(xc_code, spin=0):
    return parse_xc(xc_code)[0]

def parse_xc_name(xc_name):
    fn_facs = parse_xc(xc_name)[1]
    return fn_facs[0][0], fn_facs[1][0]

def parse_xc(description):
    '''Rules to input functional description:

    * The given functional description must be a one-line string.
    * The functional description is case-insensitive.
    * The functional description string has two parts, separated by ",".  The
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
    * Be careful with the xcfun convention on GGA functional, in which the LDA
      contribution is included.
    '''

    if isinstance(description, int):
        return 0, ((description, 1.))
    elif not isinstance(description, str): #isinstance(description, (tuple,list)):
        return parse_xc('%s,%s' % tuple(description))

    if ',' in description:
        x_code, c_code = description.replace(' ','').replace('_','').upper().split(',')
    else:
        x_code, c_code = description.replace(' ','').replace('_','').upper(), ''

    hyb = [0]
    fn_facs = []
    def parse_token(token, suffix):
        if token:
            if '*' in token:
                fac, key = token.split('*')
                if fac[0].isalpha():
                    fac, key = key, fac
                fac = float(fac)
            else:
                fac, key = 1, token
            if key == 'HF':
                hyb[0] += fac
            elif key.isdigit():
                fn_facs.append((int(key), fac))
            else:
                if key in XC_CODES:
                    x_id = XC_CODES[key]
                elif key+suffix in XC_CODES:
                    x_id = XC_CODES[key+suffix]
                else:
                    raise KeyError('Unknown key %s' % key)
                if isinstance(x_id, str):
                    hyb1, fn_facs1 = parse_xc(x_id)
                    hyb[0] += hyb1
                    fn_facs.extend(fn_facs1)
                elif x_id is None:
                    raise NotImplementedError(key)
                else:
                    fn_facs.append((x_id, fac))
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

    for token in x_code.replace('-', '+-').split('+'):
        parse_token(token, 'X')
    for token in c_code.replace('-', '+-').split('+'):
        parse_token(token, 'C')
    return hyb[0], remove_dup(fn_facs)


def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    r'''Interface to call xcfun library to evaluate XC functional, potential
    and functional derivatives.

    * The given functional xc_code must be a one-line string.
    * The functional xc_code is case-insensitive.
    * The functional xc_code string has two parts, separated by ",".  The
      first part describes the exchange functional, the second is the correlation
      functional.  If "," not appeared in string, entire string is considered as
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
    * Be careful with the xcfun convention on GGA functional, in which the LDA
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
    hyb, fn_facs = parse_xc(xc_code)
    return _eval_xc(fn_facs, rho, spin, relativity, deriv, verbose)

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

def _eval_xc(fn_facs, rho, spin=0, relativity=0, deriv=1, verbose=None):
    assert(deriv < 4)
    if spin == 0:
        rho_u = rho_d = numpy.asarray(rho, order='C')
    else:
        rho_u = numpy.asarray(rho[0], order='C')
        rho_d = numpy.asarray(rho[1], order='C')

    if rho_u.ndim == 2:
        ngrids = rho_u.shape[1]
    else:
        ngrids = len(rho_u)

    fn_ids = [x[0] for x in fn_facs]
    facs   = [x[1] for x in fn_facs]
    if all((is_lda(x) for x in fn_ids)):  # LDA
        if spin == 0:
            nvar = 1
        else:
            nvar = 2
    elif any((is_meta_gga(x) for x in fn_ids)):
        raise RuntimeError('xcfun MGGA interface not correct')
        if spin == 0:
            nvar = 3
        else:
            nvar = 7
    else:  # GGA
        if spin == 0:
            nvar = 2
        else:
            nvar = 5
    outlen = (math.factorial(nvar+deriv) //
              (math.factorial(nvar) * math.factorial(deriv)))
    outbuf = numpy.empty((ngrids,outlen))

    n = len(fn_ids)
    _itrf.XCFUN_eval_xc(ctypes.c_int(n),
                        (ctypes.c_int*n)(*fn_ids), (ctypes.c_double*n)(*facs),
                        ctypes.c_int(spin),
                        ctypes.c_int(deriv), ctypes.c_int(ngrids),
                        rho_u.ctypes.data_as(ctypes.c_void_p),
                        rho_d.ctypes.data_as(ctypes.c_void_p),
                        outbuf.ctypes.data_as(ctypes.c_void_p))

    outbuf = outbuf.T
    exc = outbuf[0]
    vxc = fxc = kxc = None
    if nvar == 1:
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
    elif nvar == 5:
        if deriv > 0:
            vxc = (outbuf[1:3].T, outbuf[3:6].T, None, None)
        if deriv > 1:
            fxc = (outbuf[[XC_D20000,XC_D11000,XC_D02000]].T,
                   outbuf[[XC_D10100,XC_D10010,XC_D10001,
                           XC_D01100,XC_D01010,XC_D01001]].T,
                   outbuf[[XC_D00200,XC_D00110,XC_D00101,XC_D00020,XC_D00011,XC_D00002]].T) + (None,)*7
        if deriv > 2:
            kxc = (outbuf[[XC_D30000,XC_D21000,XC_D12000,XC_D03000]].T,
                   outbuf[[XC_D20100,XC_D20010,XC_D20001,
                           XC_D11100,XC_D11010,XC_D11001,
                           XC_D02100,XC_D02010,XC_D02001]].T,
                   outbuf[[XC_D10200,XC_D10110,XC_D10101,XC_D10020,XC_D10011,XC_D10002,
                           XC_D01200,XC_D01110,XC_D01101,XC_D01020,XC_D01011,XC_D01002]].T,
                   outbuf[[XC_D00300,XC_D00210,XC_D00201,XC_D00120,XC_D00111,
                           XC_D00102,XC_D00030,XC_D00021,XC_D00012,XC_D00003]].T)
# MGGA/MLGGA: Note the MLGGA interface are not implemented. MGGA only needs 3
# input arguments.  To make the interface compatible with libxc, treat MGGA as
# MLGGA
    elif nvar == 3:
        if deriv > 0:
            vxc = (outbuf[1], outbuf[2], numpy.zeros_like(outbuf[1]), outbuf[3])
        if deriv > 1:
            fxc = (outbuf[XC_D200], outbuf[XC_D110], outbuf[XC_D020],
                   None, outbuf[XC_D002], None, outbuf[XC_D101], None, None, outbuf[XC_D011])
        if deriv > 2:
            kxc = (output[XC_D300], output[XC_D210], output[XC_D120], output[XC_D030],
                   output[XC_D201], output[XC_D111], output[XC_D102],
                   output[XC_D021], output[XC_D012], output[XC_D003])
    elif nvar == 7:
        if deriv > 0:
            vxc = (outbuf[1:3].T, outbuf[3:6].T, None, outbuf[6:8].T)
        if deriv > 1:
            fxc = (outbuf[[XC_D2000000,XC_D1100000,XC_D0200000]].T,
                   outbuf[[XC_D1010000,XC_D1001000,XC_D1000100,
                           XC_D0110000,XC_D0101000,XC_D0100100]].T,
                   outbuf[[XC_D0020000,XC_D0011000,XC_D0010100,
                           XC_D0002000,XC_D0001100,XC_D0000200]].T,
                   None,
                   outbuf[[XC_D0000020,XC_D0000011,XC_D0000002]].T,
                   None,
                   outbuf[[XC_D1000010,XC_D1000001,XC_D0100010,XC_D0100001]].T,
                   None, None,
                   outbuf[[XC_D0010010,XC_D0010001,XC_D0001010,XC_D0001001,
                           XC_D0000110,XC_D0000101]].T)
        if deriv > 2:
            kxc = (outbuf[[XC_D3000000,XC_D2100000,XC_D1200000,XC_D0300000]].T,
                   outbuf[[XC_D2010000,XC_D2001000,XC_D2000100,
                           XC_D1110000,XC_D1101000,XC_D1100100,
                           XC_D0210000,XC_D0201000,XC_D0200100]].T,
                   outbuf[[XC_D1020000,XC_D1011000,XC_D1010100,XC_D1002000,XC_D1001100,XC_D1000200,
                           XC_D0120000,XC_D0111000,XC_D0110100,XC_D0102000,XC_D0101100,XC_D0100200]].T,
                   outbuf[[XC_D0030000,XC_D0021000,XC_D0020100,XC_D0012000,XC_D0011100,
                           XC_D0010200,XC_D0003000,XC_D0002100,XC_D0001200,XC_D0000300]].T,
                   output[[XC_D2000010,XC_D2000001,XC_D1100010,XC_D1100001,XC_D0200010,XC_D0200001]].T,
                   output[[XC_D1010010,XC_D1010001,XC_D1001010,XC_D1001001,XC_D1000110,XC_D1000101,
                           XC_D0110010,XC_D0110001,XC_D0101010,XC_D0101001,XC_D0100110,XC_D0100101]].T,
                   output[[XC_D1000020,XC_D1000011,XC_D1000002,XC_D0100020,XC_D0100011,XC_D0100002]].T,
                   output[[XC_D0020010,XC_D0020001,XC_D0011010,XC_D0011001,XC_D0010110,XC_D0010101,
                           XC_D0002010,XC_D0002001,XC_D0001110,XC_D0001101,XC_D0000210,XC_D0000201]].T,
                   output[[XC_D0010020,XC_D0010011,XC_D0010002,
                           XC_D0001020,XC_D0001011,XC_D0001002,
                           XC_D0000120,XC_D0000111,XC_D0000102]].T,
                   output[[XC_D0000030,XC_D0000021,XC_D0000012,XC_D0000003]].T)
    return exc, vxc, fxc, kxc


def define_xc_(ni, description):
    '''Define XC functional.  See also :func:`eval_xc` for the rules of input description.

    Args:
        ni : an instance of :class:`_NumInt`

        description : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" was appeared in the string, it stands for the exact exchange.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> mf = dft.RKS(mol)
    >>> define_xc_(mf._numint, '.2*HF + .08*LDA + .72*B88, .81*LYP + .19*VWN')
    >>> mf.kernel()
    -76.3783361189611
    >>> define_xc_(mf._numint, 'LDA*.08 + .72*B88 + .2*HF, .81*LYP + .19*VWN')
    >>> mf.kernel()
    -76.3783361189611
    '''
    ni.eval_xc = lambda xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None: \
            eval_xc(description, rho, spin, relativity, deriv, verbose)
    ni.hybrid_coeff = lambda *args, **kwargs: hybrid_coeff(description)
    def xc_type(*args):
        if is_lda(description):
            return 'LDA'
        elif is_meta_gga(description):
            raise NotImplementedError('meta-GGA')
        else:
            return 'GGA'
    ni._xc_type = xc_type
    return ni

def define_xc(ni, description):
    return define_xc_(copy.copy(ni), description)
define_xc.__doc__ = define_xc_.__doc__


if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = '6311g',)
    mf = dft.RKS(mol)
    mf._numint.libxc = dft.xcfun
    print(mf.kernel() - -75.8503877483363)

    mf.xc = 'b88,lyp'
    print(mf.kernel() - -76.3969707800463)

    mf.xc = 'b3lyp'
    print(mf.kernel() - -76.3969707800463)

