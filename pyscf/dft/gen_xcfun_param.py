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

from pyscf import dft, lib
import ctypes

#libdft = lib.load_library('libxcfun_itrf')
libdft = dft.xcfun._itrf
libdft.xc_enumerate_parameters.restype = ctypes.c_char_p
libdft.XCFUN_xc_type.restype = ctypes.c_int
libdft.xc_describe_short.restype = ctypes.c_char_p
libdft.xc_describe_short.argtype = [ctypes.c_char_p]
libdft.xc_describe_long.restype = ctypes.c_char_p
libdft.xc_describe_long.argtype = [ctypes.c_char_p]
lda_ids = []
gga_ids = []
mgga_ids = []
mlgga_ids = []
xc_codes = {}
for i in range(68):
    name = libdft.xc_enumerate_parameters(ctypes.c_int(i))
    sdescr = libdft.xc_describe_short(name)
    #ldescr = libdft.xc_describe_long(ctypes.c_int(i))
    if sdescr is not None:
        #print("'%s' : %d,  # %s" % (name, i, sdescr.replace('\n', '. ')))
        print('%-16s: %2d,  # %s' % ("'%s'"%name, i, sdescr.replace('\n', '. ')))
        xc_codes[name] = i

    fntype = libdft.XCFUN_xc_type(ctypes.c_int(i))
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
    print('%-16s: %2d,  # %s' % ("'%s'"%k, xc_codes[v], v))
print('LDA_IDS = %s' % lda_ids)
print('GGA_IDS = %s' % gga_ids)
print('MGGA_IDS = %s' % mgga_ids)

#define XC_D0 0
#define XC_D1 1
#define XC_D2 2
#define XC_D3 3
#define XC_D4 4

#define XC_D00 0
#define XC_D10 1
#define XC_D01 2
#define XC_D20 3
#define XC_D11 4
#define XC_D02 5
#define XC_D30 6
#define XC_D21 7
#define XC_D12 8
#define XC_D03 9
#define XC_D40 10
#define XC_D31 11
#define XC_D22 12
#define XC_D13 13
#define XC_D04 14

#define XC_D00000 0
#define XC_D10000 1
#define XC_D01000 2
#define XC_D00100 3
#define XC_D00010 4
#define XC_D00001 5
#define XC_D20000 6
#define XC_D11000 7
#define XC_D10100 8
#define XC_D10010 9
#define XC_D10001 10
#define XC_D02000 11
#define XC_D01100 12
#define XC_D01010 13
#define XC_D01001 14
#define XC_D00200 15
#define XC_D00110 16
#define XC_D00101 17
#define XC_D00020 18
#define XC_D00011 19
#define XC_D00002 20
#define XC_D30000 21
#define XC_D21000 22
#define XC_D20100 23
#define XC_D20010 24
#define XC_D20001 25
#define XC_D12000 26
#define XC_D11100 27
#define XC_D11010 28
#define XC_D11001 29
#define XC_D10200 30
#define XC_D10110 31
#define XC_D10101 32
#define XC_D10020 33
#define XC_D10011 34
#define XC_D10002 35
#define XC_D03000 36
#define XC_D02100 37
#define XC_D02010 38
#define XC_D02001 39
#define XC_D01200 40
#define XC_D01110 41
#define XC_D01101 42
#define XC_D01020 43
#define XC_D01011 44
#define XC_D01002 45
#define XC_D00300 46
#define XC_D00210 47
#define XC_D00201 48
#define XC_D00120 49
#define XC_D00111 50
#define XC_D00102 51
#define XC_D00030 52
#define XC_D00021 53
#define XC_D00012 54
#define XC_D00003 55
#define XC_D40000 56
#define XC_D31000 57
#define XC_D30100 58
#define XC_D30010 59
#define XC_D30001 60
#define XC_D22000 61
#define XC_D21100 62
#define XC_D21010 63
#define XC_D21001 64
#define XC_D20200 65
#define XC_D20110 66
#define XC_D20101 67
#define XC_D20020 68
#define XC_D20011 69
#define XC_D20002 70
#define XC_D13000 71
#define XC_D12100 72
#define XC_D12010 73
#define XC_D12001 74
#define XC_D11200 75
#define XC_D11110 76
#define XC_D11101 77
#define XC_D11020 78
#define XC_D11011 79
#define XC_D11002 80
#define XC_D10300 81
#define XC_D10210 82
#define XC_D10201 83
#define XC_D10120 84
#define XC_D10111 85
#define XC_D10102 86
#define XC_D10030 87
#define XC_D10021 88
#define XC_D10012 89
#define XC_D10003 90
#define XC_D04000 91
#define XC_D03100 92
#define XC_D03010 93
#define XC_D03001 94
#define XC_D02200 95
#define XC_D02110 96
#define XC_D02101 97
#define XC_D02020 98
#define XC_D02011 99
#define XC_D02002 100
#define XC_D01300 101
#define XC_D01210 102
#define XC_D01201 103
#define XC_D01120 104
#define XC_D01111 105
#define XC_D01102 106
#define XC_D01030 107
#define XC_D01021 108
#define XC_D01012 109
#define XC_D01003 110
#define XC_D00400 111
#define XC_D00310 112
#define XC_D00301 113
#define XC_D00220 114
#define XC_D00211 115
#define XC_D00202 116
#define XC_D00130 117
#define XC_D00121 118
#define XC_D00112 119
#define XC_D00103 120
#define XC_D00040 121
#define XC_D00031 122
#define XC_D00022 123
#define XC_D00013 124
#define XC_D00004 125
