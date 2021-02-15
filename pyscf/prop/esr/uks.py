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
Non-relativistic unrestricted Kohn-Sham electron spin-rotation coupling
(In testing)

Refs:
    J. Phys. Chem. A. 114, 9246, 2010
    Mole. Phys. 9, 6, 585, 1964
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.prop.nmr import uks as uks_nmr
from pyscf.prop.esr import uhf as uhf_esr
from pyscf.grad import rks as rks_grad


# Note mo10 is the imaginary part of MO^1
def para(obj, mo10, mo_coeff, mo_occ, qed_fac=1):
    mol = obj.mol
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    #qed_fac = (nist.G_ELECTRON - 1)

    orboa = mo_coeff[0][:,mo_occ[0]>0]
    orbob = mo_coeff[1][:,mo_occ[1]>0]
    dm10a = [reduce(numpy.dot, (mo_coeff[0], x, orboa.T)) for x in mo10[0]]
    dm10b = [reduce(numpy.dot, (mo_coeff[1], x, orbob.T)) for x in mo10[1]]
    dm10a = numpy.asarray([x-x.T for x in dm10a])
    dm10b = numpy.asarray([x-x.T for x in dm10b])

    hso1e = uhf_esr.make_h01_soc1e(obj, mo_coeff, mo_occ, qed_fac)
    para1e =-numpy.einsum('xji,yij->xy', dm10a, hso1e)
    para1e+= numpy.einsum('xji,yij->xy', dm10b, hso1e)
    para1e *= 1./effspin / muB
    #_write(obj, align(para1e)[0], 'SOC(1e)/OZ')

    if obj.para_soc2e:
        raise NotImplementedError('dia_soc2e = %s' % obj.dia_soc2e)

    para = para1e
    return para


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
            #ipip_ao = ao[4:]
            aow = rks_grad._make_dR_dao_w(ao, wva)
            _cross3x3_(vmat[0], mol, aow, ip_ao, mask, shls_slice, ao_loc)
            aow = rks_grad._make_dR_dao_w(ao, wvb)
            _cross3x3_(vmat[1], mol, aow, ip_ao, mask, shls_slice, ao_loc)
            vxc = vrho = aow = None
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

# Jia, start to work here
class ESR(uhf_esr.ESR):
    '''dE = B dot gtensor dot s'''
    def __init__(self, scf_method):
        uhf_esr.ESR.__init__(self, scf_method)
        self.dia_soc2e = False
        self.para_soc2e = False

    def para(self, mo10=None, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self._scf.mo_coeff
        if mo_occ is None:   mo_occ = self._scf.mo_occ
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(self, mo10, mo_coeff, mo_occ)

    get_fock = uks_nmr.get_fock


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0.1 0; H 0 0 1.',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UKS(mol).set(xc='bp86').run()
    esr_obj = ESR(mf)
    esr_obj.gauge_orig = (0,0,0)
    esr_obj.para_soc2e = False
    esr_obj.so_eff_charge = True
    print(esr_obj.kernel())
