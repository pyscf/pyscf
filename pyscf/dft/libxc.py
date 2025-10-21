#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
#          Dayou Zhang <dayouzhang@outlook.com>
# This file is adapted from `dft/libxc.py` of the PySCF core module
# commit 25eaa9572977b903de24d5c11ad345cecd744728
#
# Implemented functions register_custom_functional_, update_custom_functional_,
# and unregister_custom_functional_. To make this new functionality possible,
# and to improve the memory management of the LibXC data structure, interface
# functions which originally took arguments of type c_int were modified to take
# arguments of type xc_func_type from LibXC. The LibXC data structure are cached
# and managed in the python layer to avoid construction and destruction at every
# API call.

'''
XC functional, the interface to libxc
(https://libxc.gitlab.io)
'''

import sys
import warnings
import ctypes
import math
import numpy
from functools import lru_cache, cached_property
from pyscf import lib
from pyscf.dft.xc.utils import remove_dup, format_xc_code
from pyscf.dft import xc_deriv
from pyscf import __config__

_itrf = lib.load_library('libxc_itrf')
_itrf.LIBXC_is_lda.restype = ctypes.c_int
_itrf.LIBXC_is_gga.restype = ctypes.c_int
_itrf.LIBXC_is_meta_gga.restype = ctypes.c_int
_itrf.LIBXC_needs_laplacian.argtypes = (ctypes.c_int, ctypes.c_void_p)
_itrf.LIBXC_needs_laplacian.restype = ctypes.c_int
_itrf.LIBXC_is_hybrid.restype = ctypes.c_int
_itrf.LIBXC_is_nlc.argtypes = (ctypes.c_int, ctypes.c_void_p)
_itrf.LIBXC_is_nlc.restype = ctypes.c_int
_itrf.LIBXC_is_cam_rsh.argtypes = (ctypes.c_int, ctypes.c_void_p)
_itrf.LIBXC_is_cam_rsh.restype = ctypes.c_int
_itrf.LIBXC_xc_type.argtypes = (ctypes.c_int, ctypes.c_void_p)
_itrf.LIBXC_xc_type.restype = ctypes.c_int
_itrf.LIBXC_max_deriv_order.argtypes = (ctypes.c_int, ctypes.c_void_p)
_itrf.LIBXC_max_deriv_order.restype = ctypes.c_int
_itrf.LIBXC_hybrid_coeff.argtypes = (ctypes.c_void_p, )
_itrf.LIBXC_hybrid_coeff.restype = ctypes.c_double
_itrf.LIBXC_nlc_coeff.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_double))
_itrf.LIBXC_rsh_coeff.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_double))

_itrf.LIBXC_xc_reference.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p))

_itrf.LIBXC_xc_func_init.restype = ctypes.c_void_p
_itrf.LIBXC_xc_func_end.argtypes = (ctypes.c_int, ctypes.c_void_p)
_itrf.LIBXC_xc_func_type_size.restype = ctypes.c_size_t
_itrf.LIBXC_eval_xc.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_void_p)
_itrf.LIBXC_xc_func_set_params.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_double),
                                           ctypes.c_double)

_itrf.xc_version_string.restype = ctypes.c_char_p
_itrf.xc_reference.restype = ctypes.c_char_p
_itrf.xc_reference_doi.restype = ctypes.c_char_p
_itrf.xc_number_of_functionals.restype = ctypes.c_int
_itrf.xc_available_functional_numbers.argtypes = (numpy.ctypeslib.ndpointer(dtype=numpy.intc,
                                                                            ndim=1, flags=("W", "C", "A")), )
_itrf.xc_functional_get_name.argtypes = (ctypes.c_int, )
_itrf.xc_functional_get_name.restype = ctypes.c_char_p
_itrf.xc_functional_get_number.argtypes = (ctypes.c_char_p, )
_itrf.xc_functional_get_number.restype = ctypes.c_int
_itrf.xc_func_get_info.argtypes = (ctypes.c_void_p, )
_itrf.xc_func_get_info.restype = ctypes.c_void_p
_itrf.xc_func_info_get_n_ext_params.argtypes = (ctypes.c_void_p, )
_itrf.xc_func_info_get_n_ext_params.restype = ctypes.c_int
_itrf.xc_func_set_ext_params.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_double))

_XC_FUNC_TYPE_SIZE = _itrf.LIBXC_xc_func_type_size()

def libxc_version():
    '''Returns the version of libxc'''
    return _itrf.xc_version_string().decode("UTF-8")

def libxc_reference():
    '''Returns the reference to libxc'''
    return _itrf.xc_reference().decode("UTF-8")

def libxc_reference_doi():
    '''Returns the reference to libxc'''
    return _itrf.xc_reference_doi().decode("UTF-8")

__version__ = libxc_version()
__reference__ = libxc_reference()
__reference_doi__ = libxc_reference_doi()

def available_libxc_functionals():
    # Number of functionals is
    nfunc = _itrf.xc_number_of_functionals()
    # Get functional numbers
    func_ids = numpy.zeros(nfunc, dtype=numpy.intc)
    _itrf.xc_available_functional_numbers(func_ids)
    # Returned array
    return {_itrf.xc_functional_get_name(x).decode("UTF-8").upper() : x for x in func_ids}

XC = XC_CODES = available_libxc_functionals()
PROBLEMATIC_XC = {}

def print_XC_CODES():
    '''
    Dump the built-in libxc XC_CODES along with their references in a readable format.
    '''
    xc_codes = available_libxc_functionals()
    print('XC = XC_CODES = {')
    for name, func_id in xc_codes.items():
        refs = xc_reference(func_id)
        key = f"'{name}'"
        print(f'{key:<31s}: {func_id:<3d}, # {refs[0]}')
        for r in refs[1:]:
            print(f"                                      # {r}")
    print('}')

def _xc_key_without_underscore(xc_keys):
    new_xc = []
    for key, xc_id in xc_keys.items():
        for delimiter in ('_XC_', '_X_', '_C_', '_K_'):
            if delimiter in key:
                key0, key1 = key.split(delimiter)
                new_key1 = key1.replace('_', '').replace('-', '')
                if key1 != new_key1:
                    new_xc.append((key0+delimiter+new_key1, xc_id))
                break
    return new_xc
XC_CODES.update(_xc_key_without_underscore(XC_CODES))
del (_xc_key_without_underscore)

#
# alias
#
XC_CODES.update({
    'GGA_C_BCGP'    : 'GGA_C_ACGGA',
    'LDA'           : 1 ,
    'SLATER'        : 1 ,
    'VWN3'          : 8,
    'VWNRPA'        : 8,
    'VWN5'          : 7,
    'B88'           : 106,
    'PBE0'          : 406,
    'PBE1PBE'       : 406,
    'OPTXCORR'      : '0.7344536875999693*SLATER - 0.6984752285760186*OPTX,',
    'B3LYP'         : 402,
    'B3LYPG'        : 402,  # used by Gaussian
    'B3LYP5'        : '.2*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN', # VWN5 version
    'B3P86'         : 403,
    'B3P86G'        : 403,  # used by Gaussian
    'B3P86V5'       : '.2*HF + .08*SLATER + .72*B88, .81*P86 + .19*VWN', # VWN5 version
    #'O3LYP5'        : '.1161*HF + .9262*SLATER + .8133*OPTXCORR, .81*LYP + .19*VWN5',
    #'O3LYPG'        : '.1161*HF + .9262*SLATER + .8133*OPTXCORR, .81*LYP + .19*VWNRPA',
    'O3LYP'         : 404, # in libxc == '.1161*HF + 0.071006917*SLATER + .8133*OPTX, .81*LYP + .19*VWN5', may be erroreous
    'MPW3PW'        : 'MPW3PW5',  # VWN5 version
    'MPW3PW5'       : '.2*HF + .08*SLATER + .72*MPW91, .81*PW91 + .19*VWN',
    'MPW3PWG'       : 415,  # used by Gaussian
    'MPW3LYP'       : 'MPW3LYP5',  # VWN5 version
    'MPW3LYP5'      : '.218*HF + .073*SLATER + .709*MPW91, .871*LYP + .129*VWN',
    'MPW3LYPG'      : 419,  # used by Gaussian
    'REVB3LYP'      : 'REVB3LYP5',  # VWN5 version
    'REVB3LYP5'     : '.2*HF + .13*SLATER + .67*B88, .84*LYP + .16*VWN',
    'REVB3LYPG'     : 454,  # used by Gaussian
    'X3LYP'         : 411,
    'X3LYPG'        : 411,  # used by Gaussian
    'X3LYP5'        : '.218*HF + .073*SLATER + .542385*B88 + .166615*PW91, .871*LYP + .129*VWN',
    'CAMB3LYP'      : 'HYB_GGA_XC_CAM_B3LYP',
    'CAMYBLYP'      : 'HYB_GGA_XC_CAMY_BLYP',
    'CAMYB3LYP'     : 'HYB_GGA_XC_CAMY_B3LYP',
    'B5050LYP'      : '.5*HF + .08*SLATER + .42*B88, .81*LYP + .19*VWN',
    'MPW1LYP'       : '.25*HF + .75*MPW91, LYP',
    'MPW1PBE'       : '.25*HF + .75*MPW91, PBE',
    'PBE50'         : '.5*HF + .5*PBE, PBE',
    'REVPBE0'       : '.25*HF + .75*PBE_R, PBE',
    'B1B95'         : 440,
    'TPSS0'         : '.25*HF + .75*TPSS, TPSS',
})  # noqa: E501

if getattr(__config__, 'B3LYP_WITH_VWN5', False):
    XC_CODES['B3P86' ] = 'B3P86V5'
    XC_CODES['B3LYP' ] = 'B3LYP5'
    XC_CODES['X3LYP' ] = 'X3LYP5'

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
    'RSCAN'             : 'RSCAN,RSCAN',
    'R2SCAN'            : 'R2SCAN,R2SCAN',
    'SCANL'             : 'SCANL,SCANL',
    'R2SCANL'           : 'R2SCANL,R2SCANL',
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
    'SOGGA11_X'         : 'SOGGA11_X,SOGGA11_X',
    'KT1'               : 'KT1,VWN',
    'KT2'               : 'GGA_XC_KT2',
    'KT3'               : 'GGA_XC_KT3',
    'DLDF'              : 'DLDF,DLDF',
    'GAM'               : 'GAM,GAM',
    'M06_L'             : 'M06_L,M06_L',
    'M06_SX'            : 'M06_SX,M06_SX',
    'M11_L'             : 'M11_L,M11_L',
    'MN12_L'            : 'MN12_L,MN12_L',
    'MN15_L'            : 'MN15_L,MN15_L',
    'N12'               : 'N12,N12',
    'N12_SX'            : 'N12_SX,N12_SX',
    'MN12_SX'           : 'MN12_SX,MN12_SX',
    'MN15'              : 'MN15,MN15',
    'MBEEF'             : 'MBEEF,PBE_SOL',
    'SCAN0'             : 'SCAN0,SCAN',
    'PBEOP'             : 'PBE,OP_PBE',
    'BOP'               : 'B88,OP_B88',
    'REVSCAN'           : 'MGGA_X_REVSCAN,MGGA_C_REVSCAN',
    'REVSCAN_VV10'      : 'MGGA_X_REVSCAN,MGGA_C_REVSCAN_VV10',
    'SCAN_VV10'         : 'MGGA_X_SCAN,MGGA_C_SCAN_VV10',
    'SCAN_RVV10'        : 'MGGA_X_SCAN,MGGA_C_SCAN_RVV10',
    'M05'               : 'HYB_MGGA_X_M05,MGGA_C_M05',
    'M06'               : 'HYB_MGGA_X_M06,MGGA_C_M06',
    'M05_2X'            : 'HYB_MGGA_X_M05_2X,MGGA_C_M05_2X',
    'M06_2X'            : 'HYB_MGGA_X_M06_2X,MGGA_C_M06_2X',
    'M06_HF'            : 'HYB_MGGA_X_M06_HF,MGGA_C_M06_HF',
    # extra aliases
    'SOGGA11X'          : 'SOGGA11_X',
    'M06L'              : 'M06_L',
    'M11L'              : 'M11_L',
    'MN12L'             : 'MN12_L',
    'MN15L'             : 'MN15_L',
    'N12SX'             : 'N12_SX',
    'MN12SX'            : 'MN12_SX',
    'M052X'             : 'M05_2X',
    'M062X'             : 'M06_2X',
    'M06HF'             : 'M06_HF',
}  # noqa: E122
XC_ALIAS.update([(key.replace('-',''), XC_ALIAS[key])
                 for key in XC_ALIAS if '-' in key])

def xc_reference(xc_code):
    '''Returns the reference to the individual XC functional'''
    xc = _get_xc(xc_code)
    return _xc_reference(xc.nfunc, xc.xc_arr)

def _xc_reference(nfunc, xc_arr):
    refs = []
    c_refs = (8 * nfunc * ctypes.c_char_p)()
    _itrf.LIBXC_xc_reference(nfunc, xc_arr, c_refs)
    for ref in c_refs:
        if ref:
            refs.append(ref.decode("UTF-8"))
    return refs

def xc_type(xc_code):
    '''Returns the type of a functional as a str
    '''
    xc = _get_xc(xc_code)
    return xc.xc_type

def _xc_type(nfunc, xc_arr):
    if nfunc == 0:
        return 'HF'
    else:
        t = _itrf.LIBXC_xc_type(nfunc, xc_arr)
        return ('LDA', 'GGA', 'MGGA', 'UNKNOWN')[t]

def is_lda(xc_code):
    return xc_type(xc_code) == 'LDA'

def is_hybrid_xc(xc_code):
    if xc_code is None:
        return False
    elif isinstance(xc_code, str):
        if 'HF' in xc_code:
            return True
        xc = _get_xc(xc_code)
        return _is_hybrid_xc(xc.xc_objs, xc.hyb, xc.facs, xc_code)
    elif numpy.issubdtype(type(xc_code), numpy.integer):
        xc = _get_xc(xc_code)
        return _is_hybrid_xc(xc.xc_objs, xc.hyb, xc.facs, xc_code)
    else:
        return any((is_hybrid_xc(x) for x in xc_code))

def _is_hybrid_xc(xc_objs, hyb, facs, xc_code):
    if _hybrid_coeff(xc_objs, hyb, facs) != 0:
        return True
    if _rsh_coeff(xc_objs, hyb, facs, xc_code) != (0, 0, 0):
        return True
    return False

def is_meta_gga(xc_code):
    '''Returns True if a functional is a meta GGA
    '''
    return xc_type(xc_code) == 'MGGA'

def is_gga(xc_code):
    '''Returns True if a functional is a GGA
    '''
    return xc_type(xc_code) == 'GGA'

def is_nlc(xc_code):
    # identify nlc by xc_code itself if enable_nlc is None
    if isinstance(xc_code, str) or \
            numpy.issubdtype(type(xc_code), numpy.integer):
        xc = _get_xc(xc_code)
        return xc.is_nlc
    else:
        return any((is_nlc(x) for x in xc_code))

def _is_nlc(nfunc, xc_arr):
    return _itrf.LIBXC_is_nlc(nfunc, xc_arr)

def needs_laplacian(xc_code):
    xc = _get_xc(xc_code)
    return xc.needs_laplacian

def _needs_laplacian(nfunc, xc_arr):
    return _itrf.LIBXC_needs_laplacian(nfunc, xc_arr) != 0

def max_deriv_order(xc_code):
    xc = _get_xc(xc_code)
    return xc.max_deriv_order

def _max_deriv_order(nfunc, xc_arr):
    if nfunc == 0:
        return 3
    else:
        return _itrf.LIBXC_max_deriv_order(nfunc, xc_arr)

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
    xc = _get_xc(xc_code, spin=spin)
    return xc.hybrid_coeff

def _hybrid_coeff(xc_objs, hyb, facs):
    hybs = (fac * _itrf.LIBXC_hybrid_coeff(xc) for fac, xc in zip(facs, xc_objs))
    return hyb[0] + sum(hybs)

def nlc_coeff(xc_code):
    '''Get NLC coefficients
    '''
    xc = _get_xc(xc_code)
    return xc.nlc_coeff

def _nlc_coeff(xc_objs, hyb, facs):
    nlc_pars = []
    nlc_tmp = (ctypes.c_double*2)()
    for fac, xc in zip(facs, xc_objs):
        if _itrf.LIBXC_is_nlc(ctypes.c_int(1), xc):
            _itrf.LIBXC_nlc_coeff(xc, nlc_tmp)
            nlc_pars.append((tuple(nlc_tmp), fac))
    return tuple(nlc_pars)

def rsh_coeff(xc_code):
    '''Range-separated parameter and HF exchange components: omega, alpha, beta

    Exc_RSH = c_LR * LR_HFX + c_SR * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
            = alpha * HFX   + beta * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
            = alpha * LR_HFX + hyb * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec

    SR_HFX = < pi | (1-erf(-omega r_{12}))/r_{12} | iq >
    LR_HFX = < pi | erf(-omega r_{12})/r_{12} | iq >
    alpha = c_LR
    beta = c_SR - c_LR = hyb - alpha
    '''
    if xc_code is None:
        return 0, 0, 0

    xc = _get_xc(xc_code)
    return xc.rsh_coeff

def _rsh_coeff(xc_objs, hyb, facs, xc_code):
    check_omega = True
    if isinstance(xc_code, str) and ',' in xc_code:
        # Parse only X part for the RSH coefficients.  This is to handle
        # exceptions for C functionals such as M11.
        xc_code = format_xc_code(xc_code)
        xc_code = xc_code.split(',')[0] + ','
        if 'SR_HF' in xc_code or 'LR_HF' in xc_code or 'RSH(' in xc_code:
            check_omega = False

    hyb, alpha, omega = hyb
    if omega == 0:
        # SR and LR Coulomb share the same coefficients
        # Note: this change breaks compatibility with pyscf-2.7
        assert hyb == alpha
        beta = 0.
    else:
        beta = hyb - alpha
    rsh_pars = [omega, alpha, beta]
    rsh_tmp = (ctypes.c_double*3)()
    for fac, xc in zip(facs, xc_objs):
        _itrf.LIBXC_rsh_coeff(xc, rsh_tmp)
        if rsh_pars[0] == 0:
            rsh_pars[0] = rsh_tmp[0]
        elif check_omega:
            # Check functional is actually a CAM functional
            if rsh_tmp[0] != 0 and not _itrf.LIBXC_is_cam_rsh(xc):
                raise KeyError('Libxc functional %i employs a range separation '
                               'kernel that is not supported in PySCF' % xc.xc_code)
            # Check omega
            if (rsh_tmp[0] != 0 and rsh_pars[0] != rsh_tmp[0]):
                raise ValueError('Different values of omega found for RSH functionals')
        rsh_pars[1] += rsh_tmp[1] * fac
        rsh_pars[2] += rsh_tmp[2] * fac
    return tuple(rsh_pars)

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
      - If compound XC functional is specified, no matter whether it is in the
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
      RSH(omega, alpha, beta).  Another way to input RSH is to use keywords
      SR_HF and LR_HF: "SR_HF(0.1) * alpha_plus_beta" and "LR_HF(0.1) *
      alpha" where the number in parenthesis is the value of omega.
    * Be careful with the libxc convention on GGA functional, in which the LDA
      contribution has been included.

    Args:
        description : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" was appeared in the string, it stands for the exact exchange.

    Returns:
        decoded XC description, with the data structure
        (hybrid, alpha, omega), ((libxc-Id, fac), (libxc-Id, fac), ...)
    '''  # noqa: E501

    hyb = [0, 0, 0]  # hybrid, alpha, omega (== SR_HF, LR_HF, omega)
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
                                             'specific functional (e.g. GGA_X_%s, '
                                             'HYB_MGGA_X_%s)\n'
                                             % (key, key, key))
                        else:
                            x_id = possible_xc.pop()
                        x_id = XC_CODES[x_id]
                    else:
                        # Some libxc functionals may not be listed in the
                        # XC_CODES table. Query libxc directly
                        x_id = _itrf.xc_functional_get_number(ctypes.c_char_p(key.encode()))
                        if x_id == -1:
                            raise KeyError(f"LibXCFunctional: name '{key}' not found.")
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
        return {key,
                    'LDA_X_'+key, 'GGA_X_'+key, 'MGGA_X_'+key,
                    'HYB_GGA_X_'+key, 'HYB_MGGA_X_'+key}
    def possible_xc_for(key):
        return {key, 'LDA_XC_'+key, 'GGA_XC_'+key, 'MGGA_XC_'+key,
                    'HYB_LDA_XC_'+key, 'HYB_GGA_XC_'+key, 'HYB_MGGA_XC_'+key}
    def possible_k_for(key):
        return {key,
                    'LDA_K_'+key, 'GGA_K_'+key,}
    def possible_x_k_for(key):
        return possible_x_for(key).union(possible_k_for(key))
    def possible_c_for(key):
        return {key,
                    'LDA_C_'+key, 'GGA_C_'+key, 'MGGA_C_'+key}
    fpossible_dic = {'X': possible_x_for,
                     'C': possible_c_for,
                     'compound XC': possible_xc_for,
                     'K': possible_k_for,
                     'X or K': possible_x_k_for}

    description = format_xc_code(description)

    if '-' in description:  # To handle e.g. M06-L
        for key in _NAME_WITH_DASH:
            if key in description:
                description = description.replace(key, _NAME_WITH_DASH[key])

    if ',' in description:
        x_code, c_code = description.split(',')
        for token in x_code.replace('-', '+-').replace(';+', ';').split('+'):
            parse_token(token, 'X or K')
        for token in c_code.replace('-', '+-').replace(';+', ';').split('+'):
            parse_token(token, 'C')
    else:
        for token in description.replace('-', '+-').replace(';+', ';').split('+'):
            # dftd3 cannot be used in a custom xc description
            assert '-d3' not in token
            parse_token(token, 'compound XC', search_xc_alias=True)
    return tuple(hyb), tuple(remove_dup(fn_facs))

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
                   'WB97X-D3' : 'WB97X_D3',
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
                   'CAM-B3LYP': 'CAM_B3LYP',
                   'E-'       : 'E_'} # For scientific notation


def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    r'''Interface to call libxc library to evaluate XC functional, potential
    and functional derivatives.

    * The given functional xc_code must be a one-line string.
    * The functional xc_code is case-insensitive.
    * The functional xc_code string has two parts, separated by ",".  The
      first part describes the exchange functional, the second part sets the
      correlation functional.

      - If "," not appeared in string, the entire string is treated as the
        name of a compound functional (containing both the exchange and
        the correlation functional) which was declared in the functional
        aliases list. The full list of functional aliases can be obtained by
        calling the function pyscf.dft.xcfun.XC_ALIAS.keys() .

        If the string was not found in the aliased functional list, it is
        treated as X functional.

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

    * Be careful with the libxc convention of GGA functional, in which the LDA
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
          (v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)

        * fxc for unrestricted case:
          | v2rho2[:,3]     = (u_u, u_d, d_d)
          | v2rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
          | v2sigma2[:,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
          | v2lapl2[:,3]
          | v2tau2[:,3]     = (u_u, u_d, d_d)
          | v2rholapl[:,4]
          | v2rhotau[:,4]   = (u_u, u_d, d_u, d_d)
          | v2lapltau[:,4]
          | v2sigmalapl[:,6]
          | v2sigmatau[:,6] = (uu_u, uu_d, ud_u, ud_d, dd_u, dd_d)

        * kxc for restricted case:
          (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
           v3rho2lapl, v3rho2tau,
           v3rhosigmalapl, v3rhosigmatau,
           v3rholapl2, v3rholapltau, v3rhotau2,
           v3sigma2lapl, v3sigma2tau,
           v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
           v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)

        * kxc for unrestricted case:
          | v3rho3[:,4]         = (u_u_u, u_u_d, u_d_d, d_d_d)
          | v3rho2sigma[:,9]    = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
          | v3rhosigma2[:,12]   = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
          | v3sigma3[:,10]      = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)
          | v3rho2lapl[:,6]
          | v3rho2tau[:,6]      = (u_u_u, u_u_d, u_d_u, u_d_d, d_d_u, d_d_d)
          | v3rhosigmalapl[:,12]
          | v3rhosigmatau[:,12] = (u_uu_u, u_uu_d, u_ud_u, u_ud_d, u_dd_u, u_dd_d,
                                   d_uu_u, d_uu_d, d_ud_u, d_ud_d, d_dd_u, d_dd_d)
          | v3rholapl2[:,6]
          | v3rholapltau[:,8]
          | v3rhotau2[:,6]      = (u_u_u, u_u_d, u_d_d, d_u_u, d_u_d, d_d_d)
          | v3sigma2lapl[:,12]
          | v3sigma2tau[:,12]   = (uu_uu_u, uu_uu_d, uu_ud_u, uu_ud_d, uu_dd_u, uu_dd_d,
                                   ud_ud_u, ud_ud_d, ud_dd_u, ud_dd_d, dd_dd_u, dd_dd_d)
          | v3sigmalapl2[:,9]
          | v3sigmalapltau[:,12]
          | v3sigmatau2[:,9]    = (uu_u_u, uu_u_d, uu_d_d, ud_u_u, ud_u_d, ud_d_d, dd_u_u, dd_u_d, dd_d_d)
          | v3lapl3[:,4]
          | v3lapl2tau[:,6]
          | v3lapltau2[:,6]
          | v3tau3[:,4]         = (u_u_u, u_u_d, u_d_d, d_d_d)

        see also libxc_itrf.c
    '''  # noqa: E501
    outbuf = _eval_xc(xc_code, rho, spin, deriv, omega)
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
            fxc = [outbuf[6:9].T, outbuf[9:15].T, outbuf[15:21].T]
        if deriv > 2:
            kxc = [outbuf[21:25].T, outbuf[25:34].T, outbuf[34:46].T, outbuf[46:56].T]
    elif xctype == 'MGGA' and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1], outbuf[2], None, outbuf[3]]
        if deriv > 1:
            fxc = [
                # v2rho2, v2rhosigma, v2sigma2,
                outbuf[4], outbuf[5], outbuf[6],
                # v2lapl2, v2tau2,
                None, outbuf[9],
                # v2rholapl, v2rhotau,
                None, outbuf[7],
                # v2lapltau, v2sigmalapl, v2sigmatau,
                None, None, outbuf[8]]
        if deriv > 2:
            # v3lapltau2 might not be strictly 0
            # outbuf[18] = 0
            kxc = [
                # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                outbuf[10], outbuf[11], outbuf[12], outbuf[13],
                # v3rho2lapl, v3rho2tau,
                None, outbuf[14],
                # v3rhosigmalapl, v3rhosigmatau,
                None, outbuf[15],
                # v3rholapl2, v3rholapltau, v3rhotau2,
                None, None, outbuf[16],
                # v3sigma2lapl, v3sigma2tau,
                None, outbuf[17],
                # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
                None, None, outbuf[18],
                # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
                None, None, None, outbuf[19]]
    elif xctype == 'MGGA' and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T, outbuf[3:6].T, None, outbuf[6:8].T]
        if deriv > 1:
            # v2lapltau might not be strictly 0
            # outbuf[39:43] = 0
            fxc = [
                # v2rho2, v2rhosigma, v2sigma2,
                outbuf[8:11].T, outbuf[11:17].T, outbuf[17:23].T,
                # v2lapl2, v2tau2,
                None, outbuf[33:36].T,
                # v2rholapl, v2rhotau,
                None, outbuf[23:27].T,
                # v2lapltau, v2sigmalapl, v2sigmatau,
                None, None, outbuf[27:33].T]
        if deriv > 2:
            # v3lapltau2 might not be strictly 0
            # outbuf[204:216] = 0
            kxc = [
                # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                outbuf[36:40].T, outbuf[40:49].T, outbuf[49:61].T, outbuf[61:71].T,
                # v3rho2lapl, v3rho2tau,
                None, outbuf[71:77].T,
                # v3rhosigmalapl, v3rhosigmatau,
                None, outbuf[77:89].T,
                # v3rholapl2, v3rholapltau, v3rhotau2,
                None, None, outbuf[89:95].T,
                # v3sigma2lapl, v3sigma2tau,
                None, outbuf[95:107].T,
                # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
                None, None, outbuf[107:116].T,
                # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
                None, None, None, outbuf[116:120].T]
    return exc, vxc, fxc, kxc

_GGA_SORT = {
    (1, 2): numpy.array([
        6, 7, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ]),
    (1, 3): numpy.array([
        21, 22, 25, 26, 27, 23, 28, 29, 30, 34, 35, 36, 37, 38, 39, 24, 31, 32,
        33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    ]),
    (1, 4): numpy.array([
        56, 57, 61, 62, 63, 58, 64, 65, 66, 73, 74, 75, 76, 77, 78, 59, 67, 68,
        69, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 60,
        70, 71, 72, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
        122, 123, 124, 125,
    ])
}
_MGGA_SORT = {
    (0, 2): numpy.array([
        0, # v2rho2
        1, # v2rhosigma
        3, # v2rhotau
        2, # v2sigma2
        4, # v2sigmatau
        5, # v2tau2
    ]) + 4,
    (0, 3): numpy.array([
        0, # v3rho3
        1, # v3rho2sigma
        4, # v3rho2tau
        2, # v3rhosigma2
        5, # v3rhosigmatau
        6, # v3rhotau2
        3, # v3sigma3
        7, # v3sigma2tau
        8, # v3sigmatau2
        9, # v3tau3
    ]) + 10,
    (0, 4): numpy.array([
        0,  # v4rho4
        1,  # v4rho3sigma
        5,  # v4rho3tau
        2,  # v4rho2sigma2
        6,  # v4rho2sigmatau
        7,  # v4rho2tau2
        3,  # v4rhosigma3
        8,  # v4rhosigma2tau
        9,  # v4rhosigmatau2
        10, # v4rhotau3
        4,  # v4sigma4
        11, # v4sigma3tau
        12, # v4sigma2tau2
        13, # v4sigmatau3
        14, # v4tau4
    ]) + 20,
    (1, 2): numpy.array([
        8, 9, 11, 12, 13, 23, 24, 10, 14, 15, 16, 25, 26, 17, 18, 19, 27, 28,
        20, 21, 29, 30, 22, 31, 32, 33, 34, 35,
    ]),
    (1, 3): numpy.array([
        36, 37, 40, 41, 42, 71, 72, 38, 43, 44, 45, 73, 74, 49, 50, 51, 77, 78,
        52, 53, 79, 80, 54, 81, 82, 89, 90, 91, 39, 46, 47, 48, 75, 76, 55, 56,
        57, 83, 84, 58, 59, 85, 86, 60, 87, 88, 92, 93, 94, 61, 62, 63, 95, 96,
        64, 65, 97, 98, 66, 99, 100, 107, 108, 109, 67, 68, 101, 102, 69, 103,
        104, 110, 111, 112, 70, 105, 106, 113, 114, 115, 116, 117, 118, 119,
    ]),
    (1, 4): numpy.array([
        120, 121, 125, 126, 127, 190, 191, 122, 128, 129, 130, 192, 193, 137,
        138, 139, 198, 199, 140, 141, 200, 201, 142, 202, 203, 216, 217, 218,
        123, 131, 132, 133, 194, 195, 143, 144, 145, 204, 205, 146, 147, 206,
        207, 148, 208, 209, 219, 220, 221, 155, 156, 157, 225, 226, 158, 159,
        227, 228, 160, 229, 230, 249, 250, 251, 161, 162, 231, 232, 163, 233,
        234, 252, 253, 254, 164, 235, 236, 255, 256, 257, 267, 268, 269, 270,
        124, 134, 135, 136, 196, 197, 149, 150, 151, 210, 211, 152, 153, 212,
        213, 154, 214, 215, 222, 223, 224, 165, 166, 167, 237, 238, 168, 169,
        239, 240, 170, 241, 242, 258, 259, 260, 171, 172, 243, 244, 173, 245,
        246, 261, 262, 263, 174, 247, 248, 264, 265, 266, 271, 272, 273, 274,
        175, 176, 177, 275, 276, 178, 179, 277, 278, 180, 279, 280, 295, 296,
        297, 181, 182, 281, 282, 183, 283, 284, 298, 299, 300, 184, 285, 286,
        301, 302, 303, 313, 314, 315, 316, 185, 186, 287, 288, 187, 289, 290,
        304, 305, 306, 188, 291, 292, 307, 308, 309, 317, 318, 319, 320, 189,
        293, 294, 310, 311, 312, 321, 322, 323, 324, 325, 326, 327, 328, 329,
    ])
}

def eval_xc1(xc_code, rho, spin=0, deriv=1, omega=None):
    '''Similar to eval_xc.
    Returns an array with the order of derivatives following xcfun convention.
    '''
    out = _eval_xc(xc_code, rho, spin, deriv=deriv, omega=omega)
    xctype = xc_type(xc_code)
    idx = _libxc_to_xcfun_indices(xctype, spin, deriv)
    return out[idx]

def _libxc_to_xcfun_indices(xctype, spin=0, deriv=1):
    if deriv <= 1:
        return slice(None)
    elif xctype == 'LDA' or xctype == 'HF':
        return slice(None)
    elif xctype == 'GGA':
        if spin == 0:
            return slice(None)
        else:
            idx = [numpy.arange(6)] # up to deriv=1
            for i in range(2, deriv+1):
                idx.append(_GGA_SORT[(spin, i)])
    else: # MGGA
        if spin == 0:
            idx = [numpy.arange(4)] # up to deriv=1
        else:
            idx = [numpy.arange(8)] # up to deriv=1
        for i in range(2, deriv+1):
            idx.append(_MGGA_SORT[(spin, i)])
    return numpy.hstack(idx)

def _eval_xc(xc_code, rho, spin=0, deriv=1, omega=None):
    xc = _get_xc(xc_code, spin, omega)
    assert deriv <= xc.max_deriv_order
    xctype = xc.xc_type
    assert xctype in ('HF', 'LDA', 'GGA', 'MGGA')

    rho = numpy.asarray(rho, order='C', dtype=numpy.double)
    if xctype == 'MGGA' and rho.shape[-2] == 6:
        rho = numpy.asarray(rho[...,[0,1,2,3,5],:], order='C')

    if xc.needs_laplacian:
        raise NotImplementedError('laplacian in meta-GGA method')

    nvar, xlen = xc_deriv._XC_NVAR[xctype, spin]
    ngrids = rho.shape[-1]
    rho = rho.reshape(spin+1,nvar,ngrids)
    outlen = lib.comb(xlen+deriv, deriv)
    out = numpy.zeros((outlen,ngrids))
    n = xc.nfunc
    if n > 0:
        _itrf.LIBXC_eval_xc(n,
                            xc.xc_arr,
                            (ctypes.c_double*n)(*xc.facs),
                            spin, deriv,
                            nvar, ngrids,
                            outlen,
                            rho.ctypes,
                            out.ctypes)
    return out

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
        spin : int
            spin polarized if spin > 0
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
    ...     vrho = 0.01 * 3 * rho**2
    ...     vxc = (vrho, None, None, None)
    ...     fxc = None  # 2nd order functional derivative
    ...     kxc = None  # 3rd order functional derivative
    ...     return exc, vxc, fxc, kxc
    >>> define_xc_(mf._numint, eval_xc, xctype='LDA')
    >>> mf.kernel()
    48.8525211046668
    '''
    if isinstance(description, str):
        def _eval_xc(xc_code, rho, *args, **kwargs):
            return eval_xc(description, rho, *args, **kwargs)
        ni.eval_xc = _eval_xc
        ni.hybrid_coeff = lambda *args, **kwargs: hybrid_coeff(description)
        ni.rsh_coeff = lambda *args: rsh_coeff(description)
        ni._xc_type = lambda *args: xc_type(description)

    elif callable(description):
        ni.eval_xc = _eval_xc = description
        ni.hybrid_coeff = lambda *args, **kwargs: hyb
        ni.rsh_coeff = lambda *args, **kwargs: rsh
        ni._xc_type = lambda *args: xctype

    else:
        raise ValueError('Unknown description %s' % description)

    def _eval_xc1(xc_code, rho, spin=0, deriv=1, omega=None):
        libxc_out = _eval_xc(xc_code, rho, spin, deriv=deriv, omega=omega)
        nvar, xlen = xc_deriv._XC_NVAR[xctype, spin]
        outlen = lib.comb(xlen+deriv, deriv)
        exc, vxc, fxc, kxc = libxc_out[:4]
        out = [exc]
        if deriv > 0:
            assert vxc is not None
            out.extend([x for x in vxc if x is not None])
        if deriv > 1:
            assert fxc is not None
            if xctype == 'GGA':
                assert len(fxc) == 3, 'fxc for GGA should be arranged as (v2rho2, v2rhosigma, v2sigma2)'
            elif xctype == 'MGGA':
                if len(fxc) == 10:
                    fxc = [fxc[i] for i in [0, 1, 2, 6, 4, 9]]
                else:
                    assert len(fxc) == 6, (
                        'fxc for MGGA should be arranged as\n'
                        '(v2rho2, v2rhosigma, v2sigma2, v2tau2, v2rhotau, v2sigmatau)\nor\n'
                        '(v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, '
                        'v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)')
            assert all(x is not None for x in fxc)
            out.extend(fxc)
        if deriv > 2:
            assert kxc is not None
            out.extend([x for x in kxc if x is not None])
        if spin == 1:
            # Returns of eval_xc are structured as [grid_id,deriv_component]
            # for each term in libxc_out. Change the shape to [deriv_comp, grid_id]
            out = [x.T for x in out]
        out = numpy.vstack(out)[:outlen]
        assert len(out) == outlen
        idx = _libxc_to_xcfun_indices(xctype, spin, deriv)
        return out[idx]
    ni.eval_xc1 = _eval_xc1
    return ni

def define_xc(ni, description, xctype='LDA', hyb=0, rsh=(0,0,0)):
    return define_xc_(ni.copy(), description, xctype, hyb, rsh)
define_xc.__doc__ = define_xc_.__doc__

class XCFunctionalCache:
    '''
    Data structure for caching LibXC C data structures

    Attributes for XCFunctionalCache:
        xc_code : str
            A string to describe the XC functional. See `parse_xc` for details.
        spin : int
            spin polarized if spin > 0
        xc_arr : ctypes array containing all `xc_func_type` struct of each
            functional component
        xc_objs : list of ctypes pointers of the `xc_func_type` struct of each
            functional component. The objects point to the same memory address
            in `xc_arr`
        nfunc : int
            number of LibXC functional components
        hyb : hybrid coefficients as produced by `parse_xc()`
        fn_ids : tuple of int
            LibXC ID of each functional component
        facs : tuple of float
            factor of each functional component

    '''
    def __init__(self, xc_code, spin=0, omega=None, density_threshold=None):
        if isinstance(xc_code, str) and '__VV10' in xc_code:
            raise RuntimeError('Deprecated notation for NLC functional.')
        self.xc_code = xc_code
        self.spin = spin
        self.hyb, fn_facs = parse_xc(xc_code)
        if fn_facs:
            self.fn_ids, self.facs = zip(*fn_facs)
        else:
            self.fn_ids = tuple()
            self.facs = tuple()
        self.nfunc = len(self.facs)
        if omega is not None:
            self.hyb = self.hyb[:2] + (float(omega),)
        if self.hyb[2] != 0:
            # Current implementation does not support different omegas for
            # different RSH functionals if there are multiple RSHs
            omega = [self.hyb[2]] * len(self.facs)
        else:
            omega = [0] * len(self.facs)
        fn_ids_ctypes = (ctypes.c_int * self.nfunc)(*self.fn_ids)
        problem_xc = set(self.fn_ids).intersection(PROBLEMATIC_XC)
        if problem_xc:
            problem_xc = [PROBLEMATIC_XC[k] for k in problem_xc]
            warnings.warn('Libxc functionals %s may have discrepancy to xcfun '
                          'library.\n' % problem_xc)

        self.xc_arr = _itrf.LIBXC_xc_func_init(self.nfunc, fn_ids_ctypes, spin)

        if self.xc_arr is None:
            del self.xc_arr
            raise ValueError(f"{xc_code} is not a valid functional")
        if density_threshold or any(omega):
            self._set_omega_density_threshold_(omega, density_threshold)
        self.xc_objs = [ctypes.cast(self.xc_arr + _XC_FUNC_TYPE_SIZE * i, ctypes.c_void_p)
                        for i in range(self.nfunc)]

    @cached_property
    def xc_type(self):
        return _xc_type(self.nfunc, self.xc_arr)

    @cached_property
    def is_nlc(self):
        return _is_nlc(self.nfunc, self.xc_arr)

    @cached_property
    def needs_laplacian(self):
        return _needs_laplacian(self.nfunc, self.xc_arr)

    @cached_property
    def max_deriv_order(self):
        return _max_deriv_order(self.nfunc, self.xc_arr)

    @cached_property
    def hybrid_coeff(self):
        return _hybrid_coeff(self.xc_objs, self.hyb, self.facs)

    @cached_property
    def nlc_coeff(self):
        return _nlc_coeff(self.xc_objs, self.hyb, self.facs)

    @cached_property
    def rsh_coeff(self):
        return _rsh_coeff(self.xc_objs, self.hyb, self.facs, self.xc_code)

    def customize_(self, ext_params=None, omega=None, hyb=None, facs=None,
                   density_threshold=None, callback=None, clear=True):
        '''
        Customize the functional in place. See `register_custom_functional_` for
        a detailed description of the arguments.

        Additional argument:
            clear: bool
                If True, clear all cached properties. Set to False may have a slight performace
                gain, but one must manually clear cached properties that might change prior to
                any productive calculations, preferably in the callback.
        '''
        obj_by_id = self.obj_by_id()
        if density_threshold or (omega and any(omega)):
            self._set_omega_density_threshold_(omega, density_threshold)
        if hyb is not None:
            self.hyb = hyb
        if facs is not None:
            self.facs = facs
        if ext_params is not None:
            for xid, param in ext_params.items():
                param = numpy.asarray(param, dtype=numpy.double)
                func = obj_by_id[xid]
                info = _itrf.xc_func_get_info(func)
                n = _itrf.xc_func_info_get_n_ext_params(info)
                assert param.size == n, \
                    f"""Unexpected size of external parameters for functional {xid}.
Expected {n} but {param.size} provided."""
                _itrf.xc_func_set_ext_params(func, param.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if callable(callback):
            callback(self, obj_by_id, self.spin)

        if clear:
            # clear all cached properties
            self._clear_cached_property('xc_type')
            self._clear_cached_property('is_nlc')
            self._clear_cached_property('needs_laplacian')
            self._clear_cached_property('max_deriv_order')
            self._clear_cached_property('hybrid_coeff')
            self._clear_cached_property('nlc_coeff')
            self._clear_cached_property('rsh_coeff')

    def _clear_cached_property(self, key):
        try:
            delattr(self, key)
        except AttributeError:
            pass

    def _set_omega_density_threshold_(self, omega, density_threshold):
        if density_threshold is None:
            density_threshold = 0.
        _itrf.LIBXC_xc_func_set_params(self.nfunc, self.xc_arr,
                                       (ctypes.c_double * self.nfunc)(*omega),
                                       density_threshold)

    def __del__(self):
        try:
            _itrf.LIBXC_xc_func_end(self.nfunc, self.xc_arr)
        except AttributeError:
            # skip destructor call when XC failed to initialize
            pass

    def obj_by_id(self):
        '''
        Returns a dict where LibXC functional IDs are used as the key,
        and the corresponding ctypes void pointer of the `xc_func_type`
        structs are used as the value.  This dict is useful for users who
        wish to perform low-level operations on the underlying LibXC data
        structure.
        '''
        return {xid: func for xid, func in zip(self.fn_ids, self.xc_objs)}

_CUSTOM_FUNC_R = {}
_CUSTOM_FUNC_U = {}

@lru_cache(100)
def _get_system_xc(xc_code, spin, omega=None):
    return XCFunctionalCache(xc_code, spin, omega)

def _get_xc(xc_code, spin=0, omega=None):
    if spin > 0:
        if xc_code in _CUSTOM_FUNC_U:
            return _CUSTOM_FUNC_U[xc_code]
    else:
        if xc_code in _CUSTOM_FUNC_R:
            return _CUSTOM_FUNC_R[xc_code]
    return _get_system_xc(xc_code, spin, omega)

def register_custom_functional_(new_xc_code, based_on_xc_code, ext_params=None, omega=None,
                                hyb=None, facs=None, density_threshold=None, callback=None,
                                restricted=True, unrestricted=True):
    '''
    Register a custom functional with name `new_xc_code` based on an existing LibXC
    functional with xc_code `based_on_xc_code`. When this function is called, the
    functional with xc_code `based_on_xc_code` will be constructed first. Then users can
    perform customization to the functional by setting the respective keyword arguments.
    Common customization include setting functional external parameters, adjust mixing
    factors of each functional components, and adjusting hybrid coefficients. Users can
    also use a callback function to perform more advanced modification to the LibXC data
    structure using ctypes or modify the parameters in XCFunctionalCache. Once registered,
    users may use the custom functional by passing `new_xc_code` to any PySCF module that
    uses LibXC. If `new_xc_code` has previously been registered, the older functional will
    be replaced with the new definition. In this case, however, users are encouraged to use
    `update_custom_functional_` instead to avoid reconstructing the LibXC C data structure.

    Args:
    new_xc_code : str
        The string identifier of the customized functional
    based_on_xc_code : str
        The string identifier of the parent functional
    ext_params : dict, with LibXC functional integer ID as key, and an array-like
        object containing the functional parameters as value.
        If not None, set the external parameters of the functional componet from
        the dict using the xc_func_set_ext_params API.
    omega : float
        If not None, set the omega value in self.hyb. If hyb argument is set,
        this argument is ignored.
    hyb : tuple of float
        If not None, set the hybrid coefficients self.hyb
    facs : tuple of float
        If not None, set the factors of each functional component self.facs
    callback : callable
        Callback function to perform more advanced customization.
        The function takes 3 parameters:
            xc: a XCFunctionalCache instance of the functional
            obj_by_id: a dict constructed through xc.obj_by_id, used to access
                underlying LibXC C data structure
            spin: 0 for restricted and 1 for unrestricted
        The callback function will make inplace modifications to xc object or
        the underlying LibXC C data structure. Depending on the `restricted`
        and `unrestricted` arguments, the callback may be called more than once.
    restricted : bool
        If True, create the spin-restricted version of the custom functional
    unrestricted : bool
        If True, create the spin-unrestricted version of the custom functional
    '''
    def build(spin):
        xc = XCFunctionalCache(based_on_xc_code, spin)
        xc.xc_code = new_xc_code
        xc.customize_(ext_params, omega, hyb, facs, density_threshold, callback, False)
        return xc

    try:
        parse_xc(new_xc_code)
        warnings.warn('Attempting to register a custom functional with xc_code '
                      f'"{new_xc_code}", which is a valid built-in xc_code. '
                      'Consider using a different xc_code to avoid confusion.')
    except Exception:
        pass

    if restricted:
        _CUSTOM_FUNC_R[new_xc_code] = build(0)
    if unrestricted:
        _CUSTOM_FUNC_U[new_xc_code] = build(1)

def update_custom_functional_(xc_code, ext_params=None, omega=None,
                              hyb=None, facs=None, density_threshold=None, callback=None,
                              restricted=True, unrestricted=True, clear=True):
    '''
    Update the custom functional with name `xc_code` that was previously registered
    through `register_custom_functional_`. See `register_custom_functional_` for
    a detailed description of the keyword arguments.

    Additional keyword argument:
        clear: bool
            If True, clear all cached properties. Clearing cached properties are
            required if one or more cached properties (such as hybrid_coeff) are
            changed after the update. Set to False allows users to manually handle
            cache; one can clear only the properties that are expected to change
            for a slight performace gain.
    '''
    def update(xc):
        xc.customize_(ext_params, omega, hyb, facs, density_threshold, callback, clear)

    if restricted:
        update(_CUSTOM_FUNC_R[xc_code])
    if unrestricted:
        update(_CUSTOM_FUNC_U[xc_code])

def unregister_custom_functional_(xc_code):
    '''
    Unregister a custom functional with name `xc_code` that was previously registered
    through `register_custom_functional_`.
    '''
    try:
        del _CUSTOM_FUNC_R[xc_code]
    except KeyError:
        pass

    try:
        del _CUSTOM_FUNC_U[xc_code]
    except KeyError:
        pass

