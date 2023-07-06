#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          Chia-Nan Yeh <yehcanon@gmail.com>
#

'''
One-electron spin-free X2C approximation for extended systems
'''


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
from pyscf.pbc.df import gdf_builder
from pyscf.pbc.scf import ghf
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf import __config__


def sfx2c1e(mf):
    '''Spin-free X2C.
    For the given SCF object, it updates the hcore constructor.

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
            mf.with_x2c = SpinFreeX2CHelper(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, SpinFreeX2CHelper):
            raise NotImplementedError
        else:
            return mf

    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__

    class SFX2C1E_SCF(mf_class, x2c._X2C_SCF):
        __doc__ = doc + '''
        Attributes for spin-free X2C:
            with_x2c : X2C object
        '''
        def __init__(self, mf):
            self.__dict__.update(mf.__dict__)
            self.with_x2c = SpinFreeX2CHelper(mf.mol)
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

        def dump_flags(self, verbose=None):
            mf_class.dump_flags(self, verbose)
            if self.with_x2c:
                self.with_x2c.dump_flags(verbose)
            return self

    with_x2c = SpinFreeX2CHelper(mf.mol)
    return mf.view(SFX2C1E_SCF).add_keys(with_x2c=with_x2c)

sfx2c = sfx2c1e

class PBCX2CHelper(x2c.X2C):

    exp_drop = getattr(__config__, 'pbc_x2c_X2C_exp_drop', 0.2)
    # 1e: X2C1e, atom1e: X2C1e with one-center approximation
    approx = getattr(__config__, 'pbc_x2c_X2C_approx', '1e')
    # By default, uncontracted cell.basis is used to construct the modified Dirac equation.
    xuncontract = getattr(__config__, 'pbc_x2c_X2C_xuncontract', True)
    basis = getattr(__config__, 'pbc_x2c_X2C_basis', None)

    def __init__(self, cell, kpts=None):
        self.cell = cell
        x2c.X2C.__init__(self, cell)

class SpinFreeX2CHelper(PBCX2CHelper):
    '''1-component X2c Foldy-Wouthuysen (FW Hamiltonian  (spin-free part only)
    '''
    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None:
            kpts_lst = numpy.zeros((1,3))
        else:
            kpts_lst = numpy.reshape(kpts, (-1,3))
        # By default, we use uncontracted cell.basis plus additional steep orbital for modified Dirac equation.
        xcell, contr_coeff = self.get_xmol(cell)
        from pyscf.pbc.df import df
        with_df = df.DF(xcell)

        c = lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())
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
            w = get_pnucp(with_df, kpts_lst)

        t = xcell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
        s = xcell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts_lst)
        if cell.pseudo:
            raise NotImplementedError
        else:
            v = lib.asarray(with_df.get_nuc(kpts_lst))
        if self.basis is not None:
            s22 = s
            s21 = pbcgto.intor_cross('int1e_ovlp', xcell, cell, kpts=kpts_lst)

        h1_kpts = []
        for k in range(len(kpts_lst)):
            if 'ATOM' in self.approx.upper():
                # The treatment of pnucp local part has huge effects to hcore
                #h1 = x2c._get_hcore_fw(t[k], vloc, wloc, s[k], x, c) - vloc + v[k]
                #h1 = x2c._get_hcore_fw(t[k], v[k], w[k], s[k], x, c)
                h1 = x2c._get_hcore_fw(t[k], v[k], wloc, s[k], x, c)
            else:
                xk = x2c._x2c1e_xmatrix(t[k], v[k], w[k], s[k], c)
                h1 = x2c._get_hcore_fw(t[k], v[k], w[k], s[k], xk, c)

            if self.basis is not None:
                # If cell = xcell, U = identity matrix
                U = lib.cho_solve(s22[k], s21[k])
                h1 = reduce(numpy.dot, (U.T, h1, U))
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
        assert ('1E' in self.approx.upper())
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

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('approx = %s', self.approx)
        log.info('xuncontract = %d', self.xuncontract)
        if self.basis is not None:
            log.info('basis for X matrix = %s', self.basis)
        return self


# Use Ewald-like technique to compute spVsp without the G=0 contribution.
def get_pnucp(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    eta, mesh, ke_cutoff = gdf_builder._guess_eta(cell, kpts_lst)
    log.debug1('get_pnucp eta = %s mesh = %s', eta, mesh)

    dfbuilder = gdf_builder._CCNucBuilder(cell, kpts_lst)
    dfbuilder.exclude_dd_block = False
    dfbuilder.build()
    fakenuc = aft._fake_nuc(cell, with_pseudo=cell._pseudo)
    wj = dfbuilder._int_nuc_vloc(fakenuc, 'int3c2e_pvp1', aosym='s2')
    t1 = log.timer_debug1('pnucp pass1: analytic int', *t1)

    charge = -cell.atom_charges() # Apply Koseki effective charge?
    if cell.dimension == 3:
        mod_cell = dfbuilder.modchg_cell
        nucbar = (charge / numpy.hstack(mod_cell.bas_exps())).sum()
        nucbar *= numpy.pi/cell.vol

        ovlp = cell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
        for k in range(nkpts):
            s = lib.pack_tril(ovlp[k])
            # *2 due to the factor 1/2 in T
            wj[k] -= nucbar*2 * s

    ft_kern = dfbuilder.supmol_ft.gen_ft_kernel(
        's2', intor='GTO_ft_pdotp', return_complex=False,
        kpts=kpts_lst, verbose=log)

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = Gv.shape[0]
    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
    coulG *= kws
    aoaux = ft_ao.ft_ao(dfbuilder.modchg_cell, Gv)
    vG = numpy.einsum('i,xi->x', charge, aoaux) * coulG
    vGR = vG.real
    vGI = vG.imag
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    Gblksize = max(16, int(max_memory*1e6/16/nao_pair/nkpts))
    Gblksize = min(Gblksize, ngrids, 200000)
    log.debug1('max_memory = %s  Gblksize = %s  ngrids = %s',
               max_memory, Gblksize, ngrids)

    buf = numpy.empty((2, nkpts, Gblksize, nao_pair))
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        # shape of Gpq (nkpts, nGv, nao_pair)
        Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, out=buf)
        for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
            vR  = numpy.einsum('k,kx->x', vGR[p0:p1], GpqR)
            vR += numpy.einsum('k,kx->x', vGI[p0:p1], GpqI)
            wj[k] += vR
            if not is_zero(kpts_lst[k]):
                vI  = numpy.einsum('k,kx->x', vGR[p0:p1], GpqI)
                vI -= numpy.einsum('k,kx->x', vGI[p0:p1], GpqR)
                wj[k] += vI * 1j
    t1 = log.timer_debug1('contracting pnucp', *t1)

    wj_kpts = []
    for k, kpt in enumerate(kpts_lst):
        if is_zero(kpt):
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
