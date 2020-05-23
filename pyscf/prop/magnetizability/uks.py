#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

'''
Non-relativistic magnetizability tensor for UKS
(In testing)

Refs:
[1] R. Cammi, J. Chem. Phys., 109, 3185 (1998)
[2] Todd A. Keith, Chem. Phys., 213, 123 (1996)
'''


import numpy
from pyscf import lib
from pyscf.scf import jk
from pyscf.dft import numint
from pyscf.prop.nmr import uks as uks_nmr
from pyscf.prop.magnetizability import rhf as rhf_mag
from pyscf.prop.magnetizability import uhf as uhf_mag
from pyscf.prop.magnetizability import rks as rks_mag


def dia(magobj, gauge_orig=None):
    mol = magobj.mol
    mf = magobj._scf
    mo_energy = magobj._scf.mo_energy
    mo_coeff = magobj._scf.mo_coeff
    mo_occ = magobj._scf.mo_occ
    orboa = mo_coeff[0][:,mo_occ[0] > 0]
    orbob = mo_coeff[1][:,mo_occ[1] > 0]
    dm0a = lib.tag_array(orboa.dot(orboa.T), mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
    dm0b = lib.tag_array(orbob.dot(orbob.T), mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
    dm0 = dm0a + dm0b
    dme0a = numpy.dot(orboa * mo_energy[0][mo_occ[0] > 0], orboa.T)
    dme0b = numpy.dot(orbob * mo_energy[1][mo_occ[1] > 0], orbob.T)
    dme0 = dme0a + dme0b

    e2 = rhf_mag._get_dia_1e(magobj, gauge_orig, dm0, dme0)

    if gauge_orig is not None:
        return -e2

    # Computing the 2nd order Vxc integrals from GIAO
    grids = mf.grids
    ni = mf._numint
    xc_code = mf.xc
    xctype = ni._xc_type(xc_code)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_code, mol.spin)

    make_rhoa, nset, nao = ni._gen_rho_evaluator(mol, dm0a, hermi=1)
    make_rhob            = ni._gen_rho_evaluator(mol, dm0b, hermi=1)[0]
    ngrids = len(grids.weights)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    BLKSIZE = numint.BLKSIZE
    blksize = min(int(max_memory/12*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)

    vmata = numpy.zeros((3,3,nao,nao))
    vmatb = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize):
            rho = (make_rhoa(0, ao, mask, 'LDA'),
                   make_rhob(0, ao, mask, 'LDA'))
            vxc = ni.eval_xc(xc_code, rho, spin=1, deriv=1)[1]
            vrho = vxc[0]
            r_ao = numpy.einsum('pi,px->pxi', ao, coords)
            aow = numpy.einsum('pxi,p,p->pxi', r_ao, weight, vrho[:,0])
            vmata += lib.einsum('pxi,pyj->xyij', r_ao, aow)
            aow = numpy.einsum('pxi,p,p->pxi', r_ao, weight, vrho[:,1])
            vmatb += lib.einsum('pxi,pyj->xyij', r_ao, aow)
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize):
            rho = (make_rhoa(0, ao, mask, 'GGA'),
                   make_rhob(0, ao, mask, 'GGA'))
            vxc = ni.eval_xc(xc_code, rho, spin=1, deriv=1)[1]
            wva, wvb = numint._uks_gga_wv0(rho, vxc, weight)

            # Computing \nabla (r * AO) = r * \nabla AO + [\nabla,r]_- * AO
            r_ao = numpy.einsum('npi,px->npxi', ao, coords)
            r_ao[1,:,0] += ao[0]
            r_ao[2,:,1] += ao[0]
            r_ao[3,:,2] += ao[0]

            aow = numpy.einsum('npxi,np->pxi', r_ao, wva)
            vmata += lib.einsum('pxi,pyj->xyij', r_ao[0], aow)
            aow = numpy.einsum('npxi,np->pxi', r_ao, wvb)
            vmata += lib.einsum('pxi,pyj->xyij', r_ao[0], aow)
            rho = vxc = vrho = aow = None

        vmata = vmata + vmata.transpose(0,1,3,2)
        vmatb = vmatb + vmatb.transpose(0,1,3,2)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    vmata = rks_mag._add_giao_phase(mol, vmata)
    vmatb = rks_mag._add_giao_phase(mol, vmatb)
    e2 += numpy.einsum('qp,xypq->xy', dm0a, vmata)
    e2 += numpy.einsum('qp,xypq->xy', dm0b, vmatb)
    vmata = vmatb = None

    e2 = e2.ravel()
    # Handle the hybrid functional and the range-separated functional
    if abs(hyb) > 1e-10:
        vs = jk.get_jk(mol, [dm0, dm0a, dm0a, dm0b, dm0b],
                       ['ijkl,ji->s2kl',
                        'ijkl,jk->s1il', 'ijkl,li->s1kj',
                        'ijkl,jk->s1il', 'ijkl,li->s1kj'],
                       'int2e_gg1', 's4', 9, hermi=1)
        e2 += numpy.einsum('xpq,qp->x', vs[0], dm0)
        e2 -= numpy.einsum('xpq,qp->x', vs[1], dm0a) * .5 * hyb
        e2 -= numpy.einsum('xpq,qp->x', vs[2], dm0a) * .5 * hyb
        e2 -= numpy.einsum('xpq,qp->x', vs[3], dm0b) * .5 * hyb
        e2 -= numpy.einsum('xpq,qp->x', vs[4], dm0b) * .5 * hyb
        vk = jk.get_jk(mol, [dm0a, dm0b], ['ijkl,jk->s1il', 'ijkl,jk->s1il'],
                       'int2e_g1g2', 'aa4', 9, hermi=0)
        e2 -= numpy.einsum('xpq,qp->x', vk[0], dm0a) * hyb
        e2 -= numpy.einsum('xpq,qp->x', vk[1], dm0b) * hyb

        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vs = jk.get_jk(mol, [dm0a, dm0a, dm0b, dm0b],
                               ['ijkl,jk->s1il', 'ijkl,li->s1kj',
                                'ijkl,jk->s1il', 'ijkl,li->s1kj'],
                               'int2e_gg1', 's4', 9, hermi=1)
                e2 -= numpy.einsum('xpq,qp->x', vs[0], dm0a) * .5 * (alpha-hyb)
                e2 -= numpy.einsum('xpq,qp->x', vs[1], dm0a) * .5 * (alpha-hyb)
                e2 -= numpy.einsum('xpq,qp->x', vs[2], dm0b) * .5 * (alpha-hyb)
                e2 -= numpy.einsum('xpq,qp->x', vs[3], dm0b) * .5 * (alpha-hyb)
                vk = jk.get_jk(mol, [dm0a, dm0b], ['ijkl,jk->s1il', 'ijkl,jk->s1il'],
                               'int2e_g1g2', 'aa4', 9, hermi=0)
                e2 -= numpy.einsum('xpq,qp->x', vk[0], dm0a) * (alpha-hyb)
                e2 -= numpy.einsum('xpq,qp->x', vk[1], dm0b) * (alpha-hyb)

    else:
        vj = jk.get_jk(mol, dm0, 'ijkl,ji->s2kl',
                       'int2e_gg1', 's4', 9, hermi=1)
        e2 += numpy.einsum('xpq,qp->x', vj, dm0)

    return -e2.reshape(3, 3)


class Magnetizability(uhf_mag.Magnetizability):
    dia = dia
    get_fock = uks_nmr.get_fock
    solve_mo1 = uks_nmr.solve_mo1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['Ne' , (0. , 0. , 0.)], ]
    mol.basis='631g'
    mol.build()

    mf = dft.UKS(mol).run()
    mag = Magnetizability(mf).kernel()
    print(lib.finger(mag) - -0.30375149255154221)

    mf.set(xc = 'b3lyp').run()
    mag = Magnetizability(mf).kernel()
    print(lib.finger(mag) - -0.3022331813238171)

    mol.atom = [
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.  )], ]
    mol.basis = '6-31g'
    mol.build()

    mf = dft.UKS(mol).set(xc='lda,vwn').run()
    mag = Magnetizability(mf).kernel()
    print(lib.finger(mag) - -0.4313210213418015)

    mf = dft.UKS(mol).set(xc='b3lyp').run()
    mag = Magnetizability(mf).kernel()
    print(lib.finger(mag) - -0.42828345739100998)

    mol = gto.M(atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587''',
                basis='ccpvdz', spin=2)
    mf = dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.run()
    mag = Magnetizability(mf).kernel()
    print(lib.finger(mag) - -5.166125828878557)
