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

from fractions import Fraction

import wignernj

def cg_spin(l, jdouble, mjdouble, spin):
    '''Clebsch Gordon coefficient of <l,m,1/2,spin|j,mj>

    The angular momenta j and mj are given as twice their value, so that
    half-integers are represented exactly; spin is +1 for alpha and -1 for
    beta, i.e. twice the value of ms.  m is fixed by mj = m + ms.

    Evaluated exactly with libwignernj (S. Lehtola, Comput. Phys. Commun.
    329, 110342 (2026), doi:10.1016/j.cpc.2026.110342).
    '''
    return wignernj.clebsch_gordan(Fraction(l), Fraction(mjdouble - spin, 2),
                                   Fraction(1, 2), Fraction(spin, 2),
                                   Fraction(jdouble, 2), Fraction(mjdouble, 2))


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
