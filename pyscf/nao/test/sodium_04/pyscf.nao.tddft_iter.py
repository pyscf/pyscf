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

from __future__ import print_function, division
import numpy as np
from pyscf.nao import system_vars_c, prod_basis_c, tddft_iter_c

sv = system_vars_c().init_siesta_xml(label='siesta', cd='.', force_gamma=True)
pb = prod_basis_c().init_prod_basis_pp(sv)
td = tddft_iter_c(pb.sv, pb)
omegas = np.linspace(0.0,0.25,150)+1j*td.eps
pxx = -td.comp_polariz_xx(omegas).imag
data = np.array([omegas.real*27.2114, pxx])
print('    td.rf0_ncalls ', td.rf0_ncalls)
print(' td.matvec_ncalls ', td.matvec_ncalls)
np.savetxt('tddft_iter.omega.inter.pxx.txt', data.T, fmt=['%f','%f'])
