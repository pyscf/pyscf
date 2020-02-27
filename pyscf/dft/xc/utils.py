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

def format_xc_code(description):
    '''Format the description (removing white space) then convert the
    RSH(omega, alpha, beta) notation to the internal notation RSH(alpha; beta; omega)
    '''
    description = description.replace(' ', '').replace('\n', '').upper()
    if 'RSH' not in description:
        return description

    frags = description.split('RSH')
    out = [frags[0]]
    for frag in frags[1:]:
        rsh_key, rest = frag.split(')')
        if ',' in rsh_key:
            omega, alpha, beta = rsh_key[1:].split(',')
            frag = '(' + ';'.join((alpha, beta, omega)) + ')' + rest
        out.append(frag)

    return 'RSH'.join(out)
