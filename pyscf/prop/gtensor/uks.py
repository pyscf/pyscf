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
Non-relativistic unrestricted Kohn-Sham g-tensor
(In testing)

Refs:
    JPC, 101, 3388
    JCP, 115, 11080
    JCP, 119, 10489
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.prop.nmr import uks as uks_nmr
from pyscf.prop.gtensor import uhf as uhf_g
from pyscf.prop.gtensor.uhf import _write, align
from pyscf.data import nist
from pyscf.grad import rks as rks_grad


# Note mo10 is the imaginary part of MO^1
def para(gobj, mo10, mo_coeff, mo_occ, qed_fac=1):
    mol = gobj.mol
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    #qed_fac = (nist.G_ELECTRON - 1)

    orboa = mo_coeff[0][:,mo_occ[0]>0]
    orbob = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)
    dm10a = [reduce(numpy.dot, (mo_coeff[0], x, orboa.T)) for x in mo10[0]]
    dm10b = [reduce(numpy.dot, (mo_coeff[1], x, orbob.T)) for x in mo10[1]]
    dm10a = numpy.asarray([x-x.T for x in dm10a])
    dm10b = numpy.asarray([x-x.T for x in dm10b])

    hso1e = uhf_g.make_h01_soc1e(gobj, mo_coeff, mo_occ, qed_fac)
    gpara1e =-numpy.einsum('xji,yij->xy', dm10a, hso1e)
    gpara1e+= numpy.einsum('xji,yij->xy', dm10b, hso1e)
    gpara1e *= 1./effspin / muB
    _write(gobj, align(gpara1e)[0], 'SOC(1e)/OZ')

    if gobj.para_soc2e:
        gpara2e = gobj.make_para_soc2e((dm0a,dm0b), (dm10a,dm10b), qed_fac)
        _write(gobj, align(gpara2e)[0], 'SOC(2e)/OZ')
    else:
        gpara2e = 0

    gpara = gpara1e + gpara2e
    return gpara


def make_para_soc2e(gobj, dm0, dm10, sso_qed_fac=1):
    if isinstance(gobj.para_soc2e, str):
        with_sso = 'SSO' in gobj.para_soc2e.upper()
        with_soo = 'SOO' in gobj.para_soc2e.upper()
        with_somf = 'SOMF' in gobj.para_soc2e.upper()
        with_amfi = 'AMFI' in gobj.para_soc2e.upper()
        assert(not (with_somf and (with_sso or with_soo)))
    elif gobj.para_soc2e:
        with_sso = with_soo = True
        with_somf = with_amfi = False
    else:
        with_sso = with_soo = with_somf = False
        with_amfi = True

    mol = gobj.mol
    alpha2 = nist.ALPHA ** 2
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    #sso_qed_fac = (nist.G_ELECTRON - 1)

    mf = gobj._scf
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if abs(omega) > 1e-10:
        raise NotImplementedError
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    v1 = get_vxc_soc(ni, mol, mf.grids, mf.xc, dm0,
                     max_memory=max_memory, verbose=gobj.verbose)
    dm10a, dm10b = dm10
    if with_somf:
        ej = numpy.einsum('yil,xli->xy', v1[0]+v1[1], dm10a-dm10b)
    else:
        ej  = numpy.einsum('yil,xli->xy', v1[0], dm10a)
        ej -= numpy.einsum('yil,xli->xy', v1[1], dm10b)
    #ej *= -2  #Veff(-2X) approximation of JCP 122 034107

    gpara2e = 0
    if abs(hyb) > 1e-10:
        if with_amfi:
            vj, vk = uhf_g.get_jk_amfi(mol, dm0)
        else:
            vj, vk = uhf_g.get_jk(mol, dm0)
        if with_sso or with_soo:
            ek  = numpy.einsum('yil,xli->xy', vk[0], dm10a)
            ek -= numpy.einsum('yil,xli->xy', vk[1], dm10b)
            if with_sso:
                ej += numpy.einsum('yij,xji->xy', vj[0]+vj[1], dm10a-dm10b)
                gpara2e -= sso_qed_fac * (ej - hyb * ek)
            if with_soo:
                ej += numpy.einsum('yij,xji->xy', vj[0]-vj[1], dm10a+dm10b)
                gpara2e -= 2 * (ej - hyb * ek)
        else:  # SOMF, see JCP 122, 034107 Eq (19)
            ej += numpy.einsum('yij,xji->xy', vj[0]+vj[1], dm10a-dm10b)
            ek  = numpy.einsum('yil,xli->xy', vk[0]+vk[1], dm10a-dm10b)
            gpara2e -= ej - 1.5 * hyb * ek
    else:
        if with_amfi:
            vj = uhf_g.get_j_amfi(mol, dm0)
        else:
            vj = uhf_g.get_j(mol, dm0)
        if with_sso or with_soo:
            if with_sso:
                ej += numpy.einsum('yij,xji->xy', vj[0]+vj[1], dm10a-dm10b)
                gpara2e -= sso_qed_fac * ej
            if with_soo:
                ej += numpy.einsum('yij,xji->xy', vj[0]-vj[1], dm10a+dm10b)
                gpara2e -= 2 * ej
        else:  # SOMF, see JCP 122, 034107 Eq (19)
            ej += numpy.einsum('yij,xji->xy', vj[0]+vj[1], dm10a-dm10b)
            gpara2e -= ej

    gpara2e *= (alpha2/4) / effspin / muB
    if gobj.verbose >= logger.INFO:
        _write(gobj, align(gpara2e)[0], 'SOC(2e)/OZ')
    return gpara2e


# Treat Vxc as one-particle operator Vnuc
def get_vxc_soc(ni, mol, grids, xc_code, dms, max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi=1)
    ngrids = len(grids.weights)
    BLKSIZE = numint.BLKSIZE
    blksize = min(int(max_memory/12*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        buf = numpy.empty((4,blksize,nao))
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize, buf=buf):
            rho_a = make_rho(0, ao[0], mask, 'LDA')
            rho_b = make_rho(1, ao[0], mask, 'LDA')
            vxc = ni.eval_xc(xc_code, (rho_a, rho_b), 1, deriv=1)[1]
            vrho = vxc[0]
            aow = numpy.einsum('xpi,p->xpi', ao[1:], weight*vrho[:,0])
            _cross3x3_(vmat[0], mol, aow, ao[1:], mask, shls_slice, ao_loc)
            aow = numpy.einsum('xpi,p->xpi', ao[1:], weight*vrho[:,1])
            _cross3x3_(vmat[1], mol, aow, ao[1:], mask, shls_slice, ao_loc)
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        buf = numpy.empty((10,blksize,nao))
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize, buf=buf):
            rho_a = make_rho(0, ao, mask, 'GGA')
            rho_b = make_rho(1, ao, mask, 'GGA')
            vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, deriv=1)[1]
            wva, wvb = numint._uks_gga_wv0((rho_a, rho_b), vxc, weight)

            ip_ao = ao[1:4]
            ipip_ao = ao[4:]
            aow = rks_grad._make_dR_dao_w(ao, wva)
            _cross3x3_(vmat[0], mol, aow, ip_ao, mask, shls_slice, ao_loc)
            aow = rks_grad._make_dR_dao_w(ao, wvb)
            _cross3x3_(vmat[1], mol, aow, ip_ao, mask, shls_slice, ao_loc)
            rho = vxc = vrho = vsigma = wv = aow = None
        vmat = vmat - vmat.transpose(0,1,3,2)

    else:
        raise NotImplementedError('meta-GGA')

    return vmat


def _cross3x3_(out, mol, ao1, ao2, mask, shls_slice, ao_loc):
    out[0] += numint._dot_ao_ao(mol, ao1[1], ao2[2], mask, shls_slice, ao_loc)
    out[0] -= numint._dot_ao_ao(mol, ao1[2], ao2[1], mask, shls_slice, ao_loc)
    out[1] += numint._dot_ao_ao(mol, ao1[2], ao2[0], mask, shls_slice, ao_loc)
    out[1] -= numint._dot_ao_ao(mol, ao1[0], ao2[2], mask, shls_slice, ao_loc)
    out[2] += numint._dot_ao_ao(mol, ao1[0], ao2[1], mask, shls_slice, ao_loc)
    out[2] -= numint._dot_ao_ao(mol, ao1[1], ao2[0], mask, shls_slice, ao_loc)
    return out


class GTensor(uhf_g.GTensor):
    '''dE = B dot gtensor dot s'''
    def __init__(self, scf_method):
        uhf_g.GTensor.__init__(self, scf_method)
        self.dia_soc2e = False
        self.para_soc2e = 'AMFI+SSO+SOO'

    def para(self, mo10=None, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self._scf.mo_coeff
        if mo_occ is None:   mo_occ = self._scf.mo_occ
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(self, mo10, mo_coeff, mo_occ)

    make_para_soc2e = make_para_soc2e
    get_fock = uks_nmr.get_fock


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 1.',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UKS(mol).set(xc='bp86').run()
    gobj = GTensor(mf)
    gobj.gauge_orig = (0,0,0)
    gobj.para_soc2e = False
    gobj.so_eff_charge = True
    print(gobj.align(gobj.kernel())[0])

    mol = gto.M(atom='''
                H 0   0   1
                H 1.2 0   1
                H .1  1.1 0.3
                H .8  .7  .6
                ''',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UKS(mol).set(xc='bp86').run()
    gobj = GTensor(mf)
    #print(gobj.kernel())
    gobj.para_soc2e = 'SSO'
    gobj.dia_soc2e = None
    gobj.so_eff_charge = False
    nao, nmo = mf.mo_coeff[0].shape
    nelec = mol.nelec
    numpy.random.seed(1)
    mo10 =[numpy.random.random((3,nmo,nelec[0])),
           numpy.random.random((3,nmo,nelec[1]))]
    print(lib.finger(para(gobj, mo10, mf.mo_coeff, mf.mo_occ)) - -2.1813250579863279e-05)
    numpy.random.seed(1)
    dm0 = numpy.random.random((2,nao,nao))
    dm0 = dm0 + dm0.transpose(0,2,1)
    dm10 = numpy.random.random((2,3,nao,nao))
    dm10 = dm10 - dm10.transpose(0,1,3,2)
    print(lib.finger(make_para_soc2e(gobj, dm0, dm10)) - 0.0036073897889263721)

