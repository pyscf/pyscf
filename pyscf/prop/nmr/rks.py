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


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.dft import numint
from pyscf.prop.nmr import rhf as rhf_nmr


def get_vxc_giao(ni, mol, grids, xc_code, dms, max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi=1)
    ngrids = len(grids.weights)
    BLKSIZE = numint.BLKSIZE
    blksize = min(int(max_memory/12*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((3,nao,nao))
    if xctype == 'LDA':
        buf = numpy.empty((4,blksize,nao))
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize, buf=buf):
            rho = make_rho(0, ao, mask, 'LDA')
            vxc = ni.eval_xc(xc_code, rho, 0, deriv=1)[1]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao, weight*vrho)
            giao = mol.eval_gto('GTOval_ig', coords, comp=3,
                                non0tab=mask, out=buf[1:])
            vmat[0] += numint._dot_ao_ao(mol, aow, giao[0], mask, shls_slice, ao_loc)
            vmat[1] += numint._dot_ao_ao(mol, aow, giao[1], mask, shls_slice, ao_loc)
            vmat[2] += numint._dot_ao_ao(mol, aow, giao[2], mask, shls_slice, ao_loc)
            rho = vxc = vrho = aow = None
    elif xctype == 'GGA':
        buf = numpy.empty((10,blksize,nao))
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize, buf=buf):
            rho = make_rho(0, ao, mask, 'GGA')
            vxc = ni.eval_xc(xc_code, rho, 0, deriv=1)[1]
            vrho, vsigma = vxc[:2]
            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho
            wv[1:] = rho[1:] * (weight * vsigma * 2)

            aow = numpy.einsum('npi,np->pi', ao[:4], wv)
            giao = mol.eval_gto('GTOval_ig', coords, 3, non0tab=mask, out=buf[4:])
            vmat[0] += numint._dot_ao_ao(mol, aow, giao[0], mask, shls_slice, ao_loc)
            vmat[1] += numint._dot_ao_ao(mol, aow, giao[1], mask, shls_slice, ao_loc)
            vmat[2] += numint._dot_ao_ao(mol, aow, giao[2], mask, shls_slice, ao_loc)

            giao = mol.eval_gto('GTOval_ipig', coords, 9, non0tab=mask, out=buf[1:])
            _gga_sum_(vmat, mol, ao, giao, wv, mask, shls_slice, ao_loc)
            rho = vxc = vrho = vsigma = wv = aow = None
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return vmat - vmat.transpose(0,2,1)

XX, XY, XZ = 0, 1, 2
YX, YY, YZ = 3, 4, 5
ZX, ZY, ZZ = 6, 7, 8
def _gga_sum_(vmat, mol, ao, giao, wv, mask, shls_slice, ao_loc):
    aow = numpy.einsum('pi,p->pi', giao[XX], wv[1])
    aow+= numpy.einsum('pi,p->pi', giao[YX], wv[2])
    aow+= numpy.einsum('pi,p->pi', giao[ZX], wv[3])
    vmat[0] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = numpy.einsum('pi,p->pi', giao[XY], wv[1])
    aow+= numpy.einsum('pi,p->pi', giao[YY], wv[2])
    aow+= numpy.einsum('pi,p->pi', giao[ZY], wv[3])
    vmat[1] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = numpy.einsum('pi,p->pi', giao[XZ], wv[1])
    aow+= numpy.einsum('pi,p->pi', giao[YZ], wv[2])
    aow+= numpy.einsum('pi,p->pi', giao[ZZ], wv[3])
    vmat[2] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)


def get_fock(nmrobj, dm0=None, gauge_orig=None):
    '''First order Fock matrix wrt external magnetic field'''
    if dm0 is None: dm0 = nmrobj._scf.make_rdm1()
    if gauge_orig is None: gauge_orig = nmrobj.gauge_orig

    mol = nmrobj.mol

    if gauge_orig is None:
        log = logger.Logger(nmrobj.stdout, nmrobj.verbose)
        log.debug('First-order GIAO Fock matrix')

        h1 = -.5 * mol.intor('int1e_giao_irjxp', 3)
        h1 -= mol.intor_asymmetric('int1e_ignuc', 3)
        if mol.has_ecp():
            h1 -= mol.intor_asymmetric('ECPscalar_ignuc', 3)
        h1 -= mol.intor('int1e_igkin', 3)

        mf = nmrobj._scf
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.9-mem_now)
        # Attach mo_coeff and mo_occ to improve get_vxc_giao efficiency
        dm0 = lib.tag_array(dm0, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        h1 -= get_vxc_giao(ni, mol, mf.grids, mf.xc, dm0,
                           max_memory=max_memory, verbose=nmrobj.verbose)

        intor = mol._add_suffix('int2e_ig1')
        if abs(hyb) > 1e-10:
            vj, vk = rhf_nmr.get_jk(mol, dm0)
            h1 += vj - .5 * hyb * vk
            if abs(omega) > 1e-10:
                with mol.with_range_coulomb(omega):
                    h1 -= .5*(alpha-hyb) * rhf_nmr.get_jk(mol, dm0)[1]
        else:
            vj = _vhf.direct_mapdm(intor, 'a4ij', 'lk->s1ij',
                                   dm0, 3, mol._atm, mol._bas, mol._env)
            h1 -= vj
    else:
        with mol.with_common_origin(gauge_orig):
            h1 = -.5 * mol.intor('int1e_cg_irxp', 3)
    if nmrobj.chkfile:
        lib.chkfile.dump(nmrobj.chkfile, 'nmr/h1', h1)
    return h1

def solve_mo1(nmrobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              h1=None, s1=None, with_cphf=None):
    if with_cphf is None:
        with_cphf = nmrobj.cphf
    libxc = nmrobj._scf._numint.libxc
    with_cphf = with_cphf and libxc.is_hybrid_xc(nmrobj._scf.xc)
    return rhf_nmr.solve_mo1(nmrobj, mo_energy, mo_coeff, mo_occ,
                             h1, s1, with_cphf)

class NMR(rhf_nmr.NMR):
    get_fock = get_fock
    solve_mo1 = solve_mo1

from pyscf import dft
dft.rks.RKS.NMR = dft.rks_symm.RKS.NMR = lib.class_as_method(NMR)


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

    mf = dft.RKS(mol)
    mf.kernel()
    nmr = mf.NMR()
    msc = nmr.kernel() # _xx,_yy,_zz = 55.131555
    print(lib.finger(msc) -  110.73521186810918)

    mol.atom = [
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.  )], ]
    mol.basis = '6-31g'
    mol.build()

    mf = dft.RKS(mol)
    mf.kernel()
    nmr = NMR(mf)
    msc = nmr.kernel() # _xx,_yy = 368.881201, _zz = 482.413385
    print(lib.finger(msc) - -131.70852986500839)

    mol.basis = 'ccpvdz'
    mol.build(0, 0)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    nmr = NMR(mf)
    msc = nmr.kernel() # _xx,_yy = 387.102778, _zz = 482.207925
    print(lib.finger(msc) - -132.27069885713573)
