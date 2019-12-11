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

'''
spin-free X2C correction for extended systems (experimental feature)
'''

import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.x2c import x2c
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.pbc.df import aft
from pyscf.pbc.df import aft_jk
from pyscf.pbc.df import ft_ao
from pyscf.pbc.scf import ghf
from pyscf import __config__


def sfx2c1e(mf):
    '''Spin-free X2C.
    For the given SCF object, update the hcore constructor.

    Args:
        mf : an SCF object

    Returns:
        An SCF object

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol).sfx2c1e()
    >>> mf.scf()

    >>> mol.symmetry = 1
    >>> mol.build(0, 0)
    >>> mf = scf.UHF(mol).sfx2c1e()
    >>> mf.scf()
    '''
    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            return mf.__class__(mf)
        else:
            return mf

    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__
    class SFX2C1E_SCF(mf_class, x2c._X2C_SCF):
        __doc__ = doc + \
        '''
        Attributes for spin-free X2C:
            with_x2c : X2C object
        '''
        def __init__(self, mf):
            self.__dict__.update(mf.__dict__)
            self.with_x2c = SpinFreeX2C(mf.mol)
            self._keys = self._keys.union(['with_x2c'])

        def get_hcore(self, cell=None, kpts=None, kpt=None):
            if cell is None: cell = self.cell
            if kpts is None:
                if getattr(self, 'kpts', None) is not None:
                    kpts = self.kpts
                else:
                    if kpt is None:
                        kpts = self.kpt
                    else:
                        kpts = kpt
            if self.with_x2c:
                hcore = self.with_x2c.get_hcore(cell, kpts)
                if isinstance(self, ghf.GHF):
                    if kpts.ndim == 1:
                        hcore = scipy.linalg.block_diag(hcore, hcore)
                    else:
                        hcore = [scipy.linalg.block_diag(h, h) for h in hcore]
                return hcore
            else:
                return mf_class.get_hcore(self, cell, kpts)

    return SFX2C1E_SCF(mf)

sfx2c = sfx2c1e

class X2C(x2c.X2C):

    exp_drop = getattr(__config__, 'pbc_x2c_X2C_exp_drop', 0.2)
    approx = getattr(__config__, 'pbc_x2c_X2C_approx', 'atom1e')
    xuncontract = getattr(__config__, 'pbc_x2c_X2C_xuncontract', True)
    basis = getattr(__config__, 'pbc_x2c_X2C_basis', None)

    def __init__(self, cell, kpts=None):
        self.cell = cell
        x2c.X2C.__init__(self, cell)

class SpinFreeX2C(X2C):
    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None:
            kpts_lst = numpy.zeros((1,3))
        else:
            kpts_lst = numpy.reshape(kpts, (-1,3))

        xcell, contr_coeff = self.get_xmol(cell)
        with_df = aft.AFTDF(xcell)
        c = lib.param.LIGHT_SPEED
        assert('1E' in self.approx.upper())
        if 'ATOM' in self.approx.upper():
            atom_slices = xcell.offset_nr_by_atom()
            nao = xcell.nao_nr()
            x = numpy.zeros((nao,nao))
            vloc = numpy.zeros((nao,nao))
            wloc = numpy.zeros((nao,nao))
            for ia in range(xcell.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = xcell.intor('int1e_kin', shls_slice=shls_slice)
                s1 = xcell.intor('int1e_ovlp', shls_slice=shls_slice)
                with xcell.with_rinv_at_nucleus(ia):
                    z = -xcell.atom_charge(ia)
                    v1 = z * xcell.intor('int1e_rinv', shls_slice=shls_slice)
                    w1 = z * xcell.intor('int1e_prinvp', shls_slice=shls_slice)
                vloc[p0:p1,p0:p1] = v1
                wloc[p0:p1,p0:p1] = w1
                x[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
        else:
            raise NotImplementedError

        t = xcell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
        s = xcell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts_lst)
        v = with_df.get_nuc(kpts_lst)
        #w = get_pnucp(with_df, kpts_lst)
        if self.basis is not None:
            s22 = s
            s21 = pbcgto.intor_cross('int1e_ovlp', xcell, cell, kpts=kpts_lst)

        h1_kpts = []
        for k in range(len(kpts_lst)):
# The treatment of pnucp local part has huge effects to hcore
            #h1 = x2c._get_hcore_fw(t[k], vloc, wloc, s[k], x, c) - vloc + v[k]
            #h1 = x2c._get_hcore_fw(t[k], v[k], w[k], s[k], x, c)
            h1 = x2c._get_hcore_fw(t[k], v[k], wloc, s[k], x, c)
            if self.basis is not None:
                c = lib.cho_solve(s22[k], s21[k])
                h1 = reduce(numpy.dot, (c.T, h1, c))
            if self.xuncontract and contr_coeff is not None:
                h1 = reduce(numpy.dot, (contr_coeff.T, h1, contr_coeff))
            h1_kpts.append(h1)

        if kpts is None or numpy.shape(kpts) == (3,):
            h1_kpts = h1_kpts[0]
        return lib.asarray(h1_kpts)

    def get_xmat(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        xcell, contr_coeff = self.get_xmol(cell)
        c = lib.param.LIGHT_SPEED
        assert('1E' in self.approx.upper())
        if 'ATOM' in self.approx.upper():
            atom_slices = xcell.offset_nr_by_atom()
            nao = xcell.nao_nr()
            x = numpy.zeros((nao,nao))
            for ia in range(xcell.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = xcell.intor('int1e_kin', shls_slice=shls_slice)
                s1 = xcell.intor('int1e_ovlp', shls_slice=shls_slice)
                with xcell.with_rinv_at_nucleus(ia):
                    z = -xcell.atom_charge(ia)
                    v1 = z * xcell.intor('int1e_rinv', shls_slice=shls_slice)
                    w1 = z * xcell.intor('int1e_prinvp', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
        else:
            raise NotImplementedError
        return x


# Use Ewald-like technique to compute spVsp.
# spVsp may not be divergent because the numeriator spsp and the denorminator
# in Coulomb kernel 4pi/G^2 are likely cancelled.  Even a real space lattice
# sum can converge to a finite value, it's difficult to accurately converge
# this value, i.e., large number of images in lattice summation is required.
def get_pnucp(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    Gv, Gvbase, kws = cell.get_Gv_weights(mydf.mesh)
    charge = -cell.atom_charges()
    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, mesh=mydf.mesh, Gv=Gv)
    coulG *= kws
    if mydf.eta == 0:
        wj = numpy.zeros((nkpts,nao_pair), dtype=numpy.complex128)
        SI = cell.get_SI(Gv)
        vG = numpy.einsum('i,ix->x', charge, SI) * coulG
        wj = numpy.zeros((nkpts,nao_pair), dtype=numpy.complex128)

    else:
        nuccell = copy.copy(cell)
        half_sph_norm = .5/numpy.sqrt(numpy.pi)
        norm = half_sph_norm/mole.gaussian_int(2, mydf.eta)
        chg_env = [mydf.eta, norm]
        ptr_eta = cell._env.size
        ptr_norm = ptr_eta + 1
        chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
        nuccell._atm = cell._atm
        nuccell._bas = numpy.asarray(chg_bas, dtype=numpy.int32)
        nuccell._env = numpy.hstack((cell._env, chg_env))

        wj = lib.asarray(mydf._int_nuc_vloc(nuccell, kpts_lst, 'int3c2e_pvp1'))
        t1 = log.timer_debug1('pnucp pass1: analytic int', *t1)

        aoaux = ft_ao.ft_ao(nuccell, Gv)
        vG = numpy.einsum('i,xi->x', charge, aoaux) * coulG
        if cell.dimension == 3:
            nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            nucbar *= numpy.pi/cell.vol

            ovlp = cell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
            for k in range(nkpts):
                s = lib.pack_tril(ovlp[k])
                # *2 due to the factor 1/2 in T
                wj[k] -= nucbar*2 * s

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, kpt_allow, kpts_lst,
                                       max_memory=max_memory, aosym='s2',
                                       intor='GTO_ft_pdotp'):
        for k, aoao in enumerate(aoaoks):
            if aft_jk.gamma_point(kpts_lst[k]):
                wj[k] += numpy.einsum('k,kx->x', vG[p0:p1].real, aoao.real)
                wj[k] += numpy.einsum('k,kx->x', vG[p0:p1].imag, aoao.imag)
            else:
                wj[k] += numpy.einsum('k,kx->x', vG[p0:p1].conj(), aoao)
    t1 = log.timer_debug1('contracting pnucp', *t1)

    wj_kpts = []
    for k, kpt in enumerate(kpts_lst):
        if aft_jk.gamma_point(kpt):
            wj_kpts.append(lib.unpack_tril(wj[k].real.copy()))
        else:
            wj_kpts.append(lib.unpack_tril(wj[k]))

    if kpts is None or numpy.shape(kpts) == (3,):
        wj_kpts = wj_kpts[0]
    return numpy.asarray(wj_kpts)


if __name__ == '__main__':
    from pyscf.pbc import scf
    cell = pbcgto.Cell()
    cell.build(unit = 'B',
               a = numpy.eye(3)*4,
               mesh = [11]*3,
               atom = 'H 0 0 0; H 0 0 1.8',
               verbose = 4,
               basis='sto3g')
    lib.param.LIGHT_SPEED = 2
    mf = scf.RHF(cell)
    mf.with_df = aft.AFTDF(cell)
    enr = mf.kernel()
    print('E(NR) = %.12g' % enr)

    mf = sfx2c1e(mf)
    esfx2c = mf.kernel()
    print('E(SFX2C1E) = %.12g' % esfx2c)

    mf = scf.KRHF(cell)
    mf.with_df = aft.AFTDF(cell)
    mf.kpts = cell.make_kpts([2,2,1])
    enr = mf.kernel()
    print('E(k-NR) = %.12g' % enr)

    mf = sfx2c1e(mf)
    esfx2c = mf.kernel()
    print('E(k-SFX2C1E) = %.12g' % esfx2c)

#    cell = pbcgto.M(unit = 'B',
#               a = numpy.eye(3)*4,
#               atom = 'H 0 0 0; H 0 0 1.8',
#               mesh = None,
#               dimension = 2,
#               basis='sto3g')
#    with_df = aft.AFTDF(cell)
#    w0 = get_pnucp(with_df, cell.make_kpts([2,2,1]))
#    with_df = aft.AFTDF(cell)
#    with_df.eta = 0
#    w1 = get_pnucp(with_df, cell.make_kpts([2,2,1]))
#    print(abs(w0-w1).max())
