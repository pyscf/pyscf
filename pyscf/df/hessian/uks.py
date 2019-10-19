#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#
# 
# #
# # Copyright 2019 Tencent America LLC. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # Author: Qiming Sun <osirpt.sun@gmail.com>
# #

'''
Non-relativistic UKS analytical Hessian
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.hessian import uks as uks_hess
from pyscf.df.hessian import uhf as df_uhf_hess


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (time.clock(), time.time())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)

    if mf.nlc != '':
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, beta = mf._numint.rsh_coeff(mf.xc)
    if abs(omega) > 1e-10:
        raise NotImplementedError
    else:
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=mol.spin)
    de2, ej, ek = df_uhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                                atmlst, max_memory, verbose,
                                                abs(hyb) > 1e-10)
    de2 += ej - hyb * ek  # (A,B,dR_A,dR_B)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veffa_diag, veffb_diag = uks_hess._get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    aoslices = mol.aoslice_by_atom()
    vxca, vxcb = uks_hess._get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        veffa = vxca[ia]
        veffb = vxcb[ia]
        de2[i0,i0] += numpy.einsum('xypq,pq->xy', veffa_diag[:,:,p0:p1], dm0a[p0:p1])*2
        de2[i0,i0] += numpy.einsum('xypq,pq->xy', veffb_diag[:,:,p0:p1], dm0b[p0:p1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', veffa[:,:,q0:q1], dm0a[q0:q1])*2
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', veffb[:,:,q0:q1], dm0b[q0:q1])*2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('UKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1aoa, h1aob = uks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    for ia, h1, vj1, vk1a, vk1b in df_uhf_hess._gen_jk(hessobj, mo_coeff, mo_occ, chkfile,
                                                       atmlst, verbose, abs(hyb) > 1e-10):
        f1 = h1 + vj1
        h1aoa[ia] += f1
        h1aob[ia] += f1
        if abs(hyb) > 1e-10:
            h1aoa[ia] -= hyb * vk1a
            h1aob[ia] -= hyb * vk1b

    if chkfile is None:
        return h1aoa, h1aob
    else:
        for ia in atmlst:
            lib.chkfile.save(chkfile, 'scf_f1ao/0/%d'%ia, h1aoa[ia])
            lib.chkfile.save(chkfile, 'scf_f1ao/1/%d'%ia, h1aob[ia])
        return chkfile


class Hessian(uks_hess.Hessian):
    '''Non-relativistic RKS hessian'''
    def __init__(self, mf):
        self.auxbasis_response = 1
        uks_hess.Hessian.__init__(self, mf)

    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    #dft.numint.NumInt.libxc = dft.xcfun
    xc_code = 'b3lyp'

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)],
        ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.spin = 2
    mol.build()
    mf = dft.UKS(mol).density_fit()
    mf.grids.level = 4
    mf.grids.prune = False
    mf.xc = xc_code
    mf.conv_tol = 1e-14
    mf.kernel()
    n3 = mol.natm * 3
    hobj = Hessian(mf)
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)
    print(lib.finger(e2) - 0.0091658598954866277)

