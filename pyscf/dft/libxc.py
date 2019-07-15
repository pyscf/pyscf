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
_itrf.LIBXC_number_of_functionals.restype = ctypes.c_int
_itrf.LIBXC_functional_numbers.argtypes = (numpy.ctypeslib.ndpointer(dtype=numpy.intc, ndim=1, flags=("W", "C", "A")), )
_itrf.LIBXC_functional_name.argtypes = [ctypes.c_int]
_itrf.LIBXC_functional_name.restype = ctypes.c_char_p
_itrf.LIBXC_hybrid_coeff.argtypes = [ctypes.c_int]
_itrf.LIBXC_hybrid_coeff.restype = ctypes.c_double
_itrf.LIBXC_nlc_coeff.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
_itrf.LIBXC_rsh_coeff.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_double)]

XC_ALIASES = {
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

def available_libxc_functionals():
    # Number of functionals is
    nfunc = _itrf.LIBXC_number_of_functionals()
    # Get functional numbers
    numbers = numpy.zeros(nfunc, dtype=numpy.intc)
    _itrf.LIBXC_functional_numbers(numbers)
    # Returned array
    return {'XC_' + _itrf.LIBXC_functional_name(x).decode("UTF-8").upper() : x for x in numbers}

# xc functionals from libxc
XC_LIBXC = available_libxc_functionals()
print(XC_LIBXC)
XC = XC_CODES = {**XC_ALIASES, **XC_LIBXC}

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
    'OPBE'              : 'OPTX,PBE',
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
    # new in libxc-4.2.3
    'REVSCAN'           : 'XC_MGGA_X_REVSCAN,XC_MGGA_C_REVSCAN',
    'REVSCAN_VV10'      : 'XC_MGGA_X_REVSCAN,XC_MGGA_C_REVSCAN_VV10',
    'SCAN_VV10'         : 'XC_MGGA_X_SCAN,XC_MGGA_C_SCAN_VV10',
    'SCAN_RVV10'        : 'XC_MGGA_X_SCAN,XC_MGGA_C_SCAN_RVV10',
}
XC_ALIAS.update([(key.replace('-',''), XC_ALIAS[key])
                 for key in XC_ALIAS if '-' in key])

VV10_XC = set(('B97M_V', 'WB97M_V', 'WB97X_V', 'VV10', 'LC_VV10',
               'REVSCAN_VV10', 'SCAN_VV10', 'SCAN_RVV10'))

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
    def parse_token(token, ftype, search_xc_alias=False):
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
                if search_xc_alias and key in XC_ALIAS:
                    x_id = XC_ALIAS[key]
                elif key in XC_CODES:
                    x_id = XC_CODES[key]
                else:
                    possible_xc_for = fpossible_dic[ftype]
                    possible_xc = XC_KEYS.intersection(possible_xc_for(key))
                    if possible_xc:
                        if len(possible_xc) > 1:
                            sys.stderr.write('Possible xc_code %s matches %s. '
                                             % (list(possible_xc), key))
                            for x_id in possible_xc:  # Prefer X functional
                                if '_X_' in x_id:
                                    break
                            else:
                                x_id = possible_xc.pop()
                            sys.stderr.write('XC parser takes %s\n' % x_id)
                            sys.stderr.write('You can add prefix to %s for a '
                                             'specific functional (e.g. X_%s, '
                                             'HYB_MGGA_X_%s)\n'
                                             % (key, key, key))
                        else:
                            x_id = possible_xc.pop()
                        x_id = XC_CODES[x_id]
                    else:
                        raise KeyError('Unknown %s functional  %s' % (ftype, key))
                if isinstance(x_id, str):
                    hyb1, fn_facs1 = parse_xc(x_id)
# Recursively scale the composed functional, to support e.g. '0.5*b3lyp'
                    if hyb1[0] != 0 or hyb1[1] != 0:
                        assign_omega(hyb1[2], hyb1[0]*fac, hyb1[1]*fac)
                    fn_facs.extend([(xid, c*fac) for xid, c in fn_facs1])
                elif x_id is None:
                    raise NotImplementedError('%s functional %s' % (ftype, key))
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
    fpossible_dic = {'X': possible_x_for,
                     'C': possible_c_for,
                     'compound XC': possible_xc_for,
                     'K': possible_k_for,
                     'X or K': possible_x_k_for}

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
            parse_token(token, 'X or K')
        for token in c_code.replace('-', '+-').split('+'):
            parse_token(token, 'C')
    else:
        for token in description.replace('-', '+-').split('+'):
            parse_token(token, 'compound XC', search_xc_alias=True)
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
            If "HF" (exact exchange) is appeared in the string, the HF part will
            be skipped.  If an empty string "" is given, the returns exc, vxc,...
            will be vectors of zeros.
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

    n = len(fn_ids)
    if (n == 0 or  # xc_code = '' or xc_code = 'HF', an empty functional
        all((is_lda(x) for x in fn_ids))):
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
        )
    mf = dft.RKS(mol)
    #mf._numint.libxc = dft.xcfun
    mf.xc = 'camb3lyp'
    mf.kernel()
    mf.xc = 'b88,lyp'
    eref = mf.kernel()

    mf = dft.RKS(mol)
    mf._numint = define_xc(mf._numint, 'BLYP')
    e1 = mf.kernel()
    print(e1 - eref)

    mf = dft.RKS(mol)
    mf._numint = define_xc(mf._numint, 'B3LYP5')
    e1 = mf.kernel()
    print(e1 - -75.2753037898599)
