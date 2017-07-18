from __future__ import print_function, division
import os
from pyscf.nao import system_vars_c, prod_basis_c, tddft_iter_c

sv = system_vars_c().init_siesta_xml(label='siesta', cd='.', force_gamma=True)
pb = prod_basis_c().init_pb_pp_libnao_apair(sv)
pb.init_prod_basis_pp()

""" This is iterative TDDFT with SIESTA starting point """
td = tddft_iter_c(pb.sv, pb)
dn0 = td.apply_rf0(td.moms1[:,0])
    
