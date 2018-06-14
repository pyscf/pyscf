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
from timeit import default_timer as timer
import os
from pyscf.nao import system_vars_c, prod_basis_c, tddft_iter_c
from numpy import allclose


t1 = timer()
sv = system_vars_c().init_siesta_xml(label='siesta', cd='.', force_gamma=True)
t2 = timer(); print('system_vars_c().init_siesta_xml ', t2-t1); t1 = timer()

pbb = prod_basis_c().init_prod_basis_pp_batch(sv)
t2 = timer(); print('prod_basis_c().init_prod_basis_pp_batch(sv) ', t2-t1); t1 = timer()

pba = prod_basis_c().init_prod_basis_pp(sv)
t2 = timer(); print('prod_basis_c().init_prod_basis_pp(sv) ', t2-t1); t1 = timer()

for a,b in zip(pba.bp2info,pbb.bp2info):
  for a1,a2 in zip(a.atoms,b.atoms): assert a1==a2
  for a1,a2 in zip(a.cc2a, b.cc2a): assert a1==a2
  assert allclose(a.vrtx, b.vrtx)
  assert allclose(a.cc, b.cc)

print(abs(pbb.get_da2cc_coo().tocsr()-pba.get_da2cc_coo().tocsr()).sum(), \
      abs(pbb.get_dp_vertex_coo().tocsr()-pba.get_dp_vertex_coo().tocsr()).sum())


""" This is iterative TDDFT with SIESTA starting point """
#td = tddft_iter_c(pb.sv, pb)
#t2 = timer(); print(t2-t1); t1 = timer()


#dn0 = td.apply_rf0(td.moms1[:,0])
#t2 = timer(); print(t2-t1); t1 = timer()

   
