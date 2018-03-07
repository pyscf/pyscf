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

import sys
import subprocess
import tempfile
import numpy
from pyscf import dft, lib
import ctypes

#libdft = lib.load_library('libxc_itrf')
libdft = dft.libxc._itrf
libdft.xc_functional_get_number.restype = ctypes.c_int
libdft.xc_functional_get_name.restype = ctypes.c_char_p
libdft.xc_family_from_id.restype = ctypes.c_int

XC_FAMILY_LDA      =  1
XC_FAMILY_GGA      =  2
XC_FAMILY_MGGA     =  4
XC_FAMILY_LCA      =  8
XC_FAMILY_OEP      = 16
XC_FAMILY_HYB_GGA  = 32
XC_FAMILY_HYB_MGGA = 64

def get_ids(start=1, stop=550, detect_deriv=False):
    lda_ids = []
    gga_ids = []
    mgga_ids = []
    hyb_gga_ids = []
    hyb_mgga_ids = []
    xc_codes = []
    deriv_order = []

    for xc_id in range(start, stop):
        name = libdft.xc_functional_get_name(ctypes.c_int(xc_id))
        if name is None:
            deriv_order.append(-1)
        else:
            family = ctypes.c_int(0)
            number = ctypes.c_int(0)
            fntype = libdft.xc_family_from_id(ctypes.c_int(xc_id), ctypes.byref(family),
                                              ctypes.byref(number))
            known = True
            if (fntype ^ XC_FAMILY_LDA) == 0:
                lda_ids.append(xc_id)
            elif (fntype ^ XC_FAMILY_GGA) == 0:
                gga_ids.append(xc_id)
            elif (fntype ^ XC_FAMILY_MGGA) == 0:
                mgga_ids.append(xc_id)
            elif (fntype ^ XC_FAMILY_HYB_GGA) == 0:
                hyb_gga_ids.append(xc_id)
            elif (fntype ^ XC_FAMILY_HYB_MGGA) == 0:
                hyb_mgga_ids.append(xc_id)
            else:
                known = False
                print('Unknown', name, xc_id, fntype)

            if known:
                xc_codes.append([xc_id, 'XC_'+name.upper()])
                if detect_deriv:
                    deriv_order.append(detect_deriv_order(xc_id))
                    print('%-26s: %3d,  # deriv=%d' %
                          ("'XC_"+name.upper()+"'", xc_id, deriv_order[-1]))
                else:
                    print('%-26s: %3d,' % ("'XC_"+name.upper()+"'", xc_id))
                sys.stdout.flush()
            else:
                deriv_order.append(-1)

    print('LDA_IDS = %s' % lda_ids)
    print('GGA_IDS = %s' % gga_ids)
    print('MGGA_IDS = %s' % mgga_ids)
    print('HYB_GGA_IDS = %s' % hyb_gga_ids)
    print('HYB_MGGA_IDS = %s' % hyb_mgga_ids)
    if detect_deriv:
        print('\nDeriv order')
        print(deriv_order)

tmpcall = '''python -c "from pyscf import dft; dft.libxc.eval_xc(%d, dft.libxc.numpy.zeros((6,2)), deriv=%d)"'''
def detect_deriv_order(xc_id):
    with tempfile.TemporaryFile() as f:
        for i in range(1, 4):
            ret = subprocess.call(tmpcall % (xc_id, i), stderr=f, shell=True)
            #f.seek(0)
            #errmsg = f.read()
            if ret == 1:
                return i-1
        return i

get_ids(start=1, stop=550, detect_deriv=True)
