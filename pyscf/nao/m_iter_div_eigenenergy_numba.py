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

from __future__ import division
import numba as nb
import numpy as np

@nb.jit(nopython=True)
def div_eigenenergy_numba(ksn2e, ksn2f, nfermi, vstart, comega, nm2v_re, nm2v_im, ksn2e_dim):
    """
        multiply the temporary matrix by (fn - fm) (frac{1.0}{w - (Em-En) -1} -
            frac{1.0}{w + (Em - En)})
        using numba
    """

    for n in range(nfermi):
        en = ksn2e[0, 0, n]
        fn = ksn2f[0, 0, n]
        for j in range(n+1, ksn2e_dim, 1):
            em = ksn2e[0, 0, j]
            fm = ksn2f[0, 0, j]
            m = j - vstart
            nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
            nm2v = nm2v * (fn-fm) * ( 1.0 / (comega - (em - en)) - 1.0 /\
                    (comega + (em - en)) )
            nm2v_re[n, m] = nm2v.real
            nm2v_im[n, m] = nm2v.imag

@nb.jit(nopython=True)
def mat_mul_numba(a, b):
    return a*b
