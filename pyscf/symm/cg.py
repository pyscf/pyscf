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

import numpy

def cg_spin(l, jdouble, mjdouble, spin):
    '''Clebsch Gordon coefficient of <l,m,1/2,spin|j,mj>'''
    ll1 = 2 * l + 1
    if jdouble == 2*l+1:
        if spin > 0:
            c = numpy.sqrt(.5*(ll1+mjdouble)/ll1)
        else:
            c = numpy.sqrt(.5*(ll1-mjdouble)/ll1)
    elif jdouble == 2*l-1:
        if spin > 0:
            c =-numpy.sqrt(.5*(ll1-mjdouble)/ll1)
        else:
            c = numpy.sqrt(.5*(ll1+mjdouble)/ll1)
    else:
        c = 0
    return c


if __name__ == '__main__':
    for kappa in list(range(-4,0)) + list(range(1,4)):
        if kappa < 0:
            l = -kappa - 1
            j = l * 2 + 1
        else:
            l = kappa
            j = l * 2 - 1
        print(kappa,l,j)
        for mj in range(-j, j+1, 2):
            print(cg_spin(l, j, mj, 1), cg_spin(l, j, mj, -1))

