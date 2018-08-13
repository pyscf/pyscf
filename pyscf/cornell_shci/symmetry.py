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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import pyscf.symm

# To Molpro symmetry ID
IRREP_MAP = {'D2h': (1,         # Ag
                     4,         # B1g
                     6,         # B2g
                     7,         # B3g
                     8,         # Au
                     5,         # B1u
                     3,         # B2u
                     2),        # B3u
             'C2v': (1,         # A1
                     4,         # A2
                     2,         # B1
                     3),        # B2
             'C2h': (1,         # Ag
                     4,         # Bg
                     2,         # Au
                     3),        # Bu
             'D2' : (1,         # A
                     4,         # B1
                     3,         # B2
                     2),        # B3
             'Cs' : (1,         # A'
                     2),        # A"
             'C2' : (1,         # A
                     2),        # B
             'Ci' : (1,         # Ag
                     2),        # Au
             'C1' : (1,)}

def d2h_subgroup(gpname):
    if gpname.lower() == 'dooh':
        gpname = 'D2h'
    elif gpname.lower() == 'coov':
        gpname = 'C2v'
    else:
        gpname = pyscf.symm.std_symb(gpname)
    return gpname

def irrep_name2id(gpname, symb):
    irrep_id = pyscf.symm.irrep_name2id(gpname, symb) % 10
    gpname = d2h_subgroup(gpname)
    return IRREP_MAP[gpname][irrep_id]

def convert_orbsym(gpname, orbsym):
    '''Convert orbital symmetry irrep_id to Block internal irrep_id
    '''
    if gpname.lower() == 'dooh':
        orbsym = [IRREP_MAP['D2h'][i % 10] for i in orbsym]
    elif gpname.lower() == 'coov':
        orbsym = [IRREP_MAP['C2h'][i % 10] for i in orbsym]
    else:
        gpname = pyscf.symm.std_symb(gpname)
        orbsym = [IRREP_MAP[gpname][i] for i in orbsym]
    return orbsym

# LZSYM for Dice code
UNDEF = 0
LZSYM_MAP = (
 1    ,  # 0  A1g
 UNDEF,  # 1  A2g
 5    ,  # 2  E1gx
-5    ,  # 3  E1gy
 UNDEF,  # 4  A2u
 2    ,  # 5  A1u
-6    ,  # 6  E1uy
 6    ,  # 7  E1ux
 UNDEF,  # 8
 UNDEF,  # 9
 7    ,  # 10 E2gx
-7    ,  # 11 E2gy
 9    ,  # 12 E3gx
-9    ,  # 13 E3gy
-8    ,  # 14 E2uy
 8    ,  # 15 E2ux
-10   ,  # 16 E3uy
 10   ,  # 17 E3ux
 UNDEF,  # 18
 UNDEF,  # 19
 11   ,  # 20 E4gx
-11   ,  # 21 E4gy
 13   ,  # 22 E5gx
-13   ,  # 23 E5gy
-12   ,  # 24 E4uy
 12   ,  # 25 E4ux
-14   ,  # 26 E5uy
 14   ,  # 27 E5ux
)
def convert_lzsym(gpname, orbsym):
    '''Convert orbital symmetry irrep_id to Block internal irrep_id
    '''
    if gpname.lower() == 'dooh':
        orbsym = [LZSYM_MAP[i] for i in orbsym]
    elif gpname.lower() == 'coov':
        raise NotImplementedError
    else:
        gpname = pyscf.symm.std_symb(gpname)
        orbsym = [IRREP_MAP[gpname][i] for i in orbsym]
    return orbsym
