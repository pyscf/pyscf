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
Ref: [1] arXiv:2202.02252 (2022)
The implementation of the spin-orbital version follows the notations in [1].
Additional information can be found in [1] and references therein.
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
from pyscf.pbc.x2c import sfx2c1e
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf import __config__

def x2c1e_gscf(mf):
    '''
    For the given *GHF* object, it generates X2C1E-GSCF object in spin-orbital basis
    and updates the hcore constructor.

    Args:
        mf : an GHF/GKS object

    Return:
        An GHF/GKS object

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.KGHF(mol).x2c1e()
    >>> mf.kernel()
    '''

    # Check if mf is a generalized SCF object
    assert isinstance(mf, ghf.GHF)

    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinOrbitalX2C1EHelper(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, SpinOrbitalX2C1EHelper):
            # An object associated to sfx2c1e.SpinFreeX2CHelper
            raise NotImplementedError
        else:
            return mf

    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__

    class X2C1E_GSCF(mf_class, x2c._X2C_SCF):
        __doc__ = doc + '''
        Attributes for spin-orbital X2C1E for PBC.
            with_x2c : X2C object
        '''
        def __init__(self, mf):
            self.__dict__.update(mf.__dict__)
            self.with_x2c = SpinOrbitalX2C1EHelper(mf.cell)
            self._keys = self._keys.union(['with_x2c'])

        def get_hcore(self, cell=None, kpts=None):
            if cell is None:
                cell = self.cell
            if kpts is None:
                kpts = self.kpts

            if self.with_x2c is not None:
                hcore = self.with_x2c.get_hcore(cell, kpts)
                return hcore
            else:
                return mf_class.get_hcore(self, cell, kpts)

        def dump_flags(self, verbose=None):
            mf_class.dump_flags(self, verbose)
            if self.with_x2c:
                self.with_x2c.dump_flags(verbose)
            return self

    with_x2c = SpinOrbitalX2C1EHelper(mf.mol)
    return mf.view(X2C1E_GSCF).add_keys(with_x2c=with_x2c)

class SpinOrbitalX2C1EHelper(sfx2c1e.PBCX2CHelper):
    def get_hcore(self, cell=None, kpts=None):
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts_lst = numpy.zeros((1,3))
        else:
            kpts_lst = numpy.reshape(kpts, (-1,3))
        # By default, we use uncontracted cell.basis plus additional steep orbital for modified Dirac equation.
        xcell, contr_coeff = self.get_xmol(cell)
        if contr_coeff is not None:
            contr_coeff = _block_diag(contr_coeff)
        from pyscf.pbc.df import df
        with_df = df.DF(xcell)

        c = lib.param.LIGHT_SPEED

        if 'ATOM' in self.approx.upper():
            raise NotImplementedError
        else:
            w_sr = sfx2c1e.get_pnucp(with_df, kpts_lst)
            w_soc = get_pbc_pvxp(with_df, kpts_lst)
            #w_soc = get_pbc_pvxp(xcell, kpts_lst)
            w = []
            for k in range(len(kpts_lst)):
                w_spblk = numpy.vstack([w_soc[k], w_sr[k,None]])
                w_k = _sigma_dot(w_spblk)
                w.append(w_k)
            w = lib.asarray(w)

        t_aa = xcell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
        t    = numpy.array([_block_diag(t_aa[k]) for k in range(len(kpts_lst))])
        s_aa = xcell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts_lst)
        s = numpy.array([_block_diag(s_aa[k]) for k in range(len(kpts_lst))])
        if cell.pseudo:
            raise NotImplementedError
        else:
            v_aa = lib.asarray(with_df.get_nuc(kpts_lst))
            v = numpy.array([_block_diag(v_aa[k]) for k in range(len(kpts_lst))])
        if self.basis is not None:
            s22 = s_aa
            s21 = pbcgto.intor_cross('int1e_ovlp', xcell, cell, kpts=kpts_lst)

        h1_kpts = []
        for k in range(len(kpts_lst)):
            if 'ATOM' in self.approx.upper():
                raise NotImplementedError
            else:
                xk = x2c._x2c1e_xmatrix(t[k], v[k], w[k], s[k], c)
                h1 = x2c._get_hcore_fw(t[k], v[k], w[k], s[k], xk, c)

            if self.basis is not None:
                # If cell = xcell, U = identity matrix
                U = _block_diag(lib.cho_solve(s22[k], s21[k]))
                h1 = reduce(numpy.dot, (U.T, h1, U))
            if self.xuncontract and contr_coeff is not None:
                h1 = reduce(numpy.dot, (contr_coeff.T, h1, contr_coeff))
            h1_kpts.append(h1)

        if kpts is None or numpy.shape(kpts) == (3,):
            h1_kpts = h1_kpts[0]
        return lib.asarray(h1_kpts)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('approx = %s', self.approx)
        log.info('xuncontract = %d', self.xuncontract)
        if self.basis is not None:
            log.info('basis for X matrix = %s', self.basis)
        return self
#
# SOC with 1-center approximation
#
def get_1c_pvxp(cell, kpts=None):
    atom_slices = cell.offset_nr_by_atom()
    nao = cell.nao_nr()
    mat_soc = numpy.zeros((3,nao,nao))
    for ia in range(cell.natm):
        ish0, ish1, p0, p1 = atom_slices[ia]
        shls_slice = (ish0, ish1, ish0, ish1)
        with cell.with_rinv_as_nucleus(ia):
            z = -cell.atom_charge(ia)
            # Apply Koseki effective charge on z?
            w = z * cell.intor('int1e_prinvxp', comp=3, shls_slice=shls_slice)
        mat_soc[:,p0:p1,p0:p1] = w
    return mat_soc

# W_SOC with lattice summation (G != 0)
def get_pbc_pvxp(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(cell.stdout, cell.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()

    dfbuilder = gdf_builder._CCNucBuilder(cell, kpts_lst)
    dfbuilder.exclude_dd_block = False
    dfbuilder.build()
    eta, mesh, ke_cutoff = gdf_builder._guess_eta(cell, kpts_lst)
    log.debug1('get_pnucp eta = %s mesh = %s', eta, mesh)

    fakenuc = aft._fake_nuc(cell, with_pseudo=cell._pseudo)
    soc_mat = dfbuilder._int_nuc_vloc(fakenuc, 'int3c2e_pvxp1_sph',
                                      aosym='s1', comp=3)
    soc_mat = soc_mat.reshape(nkpts,3,nao,nao)
    t1 = log.timer_debug1('pnucp pass1: analytic int', *t1)

    ft_kern = dfbuilder.supmol_ft.gen_ft_kernel(
        's1', intor='GTO_ft_pxp_sph', comp=3, return_complex=False,
        kpts=kpts_lst, verbose=log)

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = Gv.shape[0]
    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
    coulG *= kws
    aoaux = ft_ao.ft_ao(dfbuilder.modchg_cell, Gv)
    charge = -cell.atom_charges() # Apply Koseki effective charge?
    vG = numpy.einsum('i,xi->x', charge, aoaux) * coulG
    vGR = vG.real
    vGI = vG.imag

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    Gblksize = max(16, int(max_memory*1e6/16/3/nao**2/nkpts))
    Gblksize = min(Gblksize, ngrids, 200000)
    log.debug1('max_memory = %s  Gblksize = %s  ngrids = %s',
               max_memory, Gblksize, ngrids)

    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        # shape of Gpq (nkpts, nGv, nao_pair)
        Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow)
        for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
            vR  = numpy.einsum('k,ckpq->cpq', vGR[p0:p1], GpqR)
            vR += numpy.einsum('k,ckpq->cpq', vGI[p0:p1], GpqI)
            soc_mat[k] += vR
            if not is_zero(kpts_lst[k]):
                vI  = numpy.einsum('k,ckpq->cpq', vGR[p0:p1], GpqI)
                vI -= numpy.einsum('k,ckpq->cpq', vGI[p0:p1], GpqR)
                soc_mat[k] += vI * 1j
    t1 = log.timer_debug1('contracting pnucp', *t1)

    soc_mat_kpts = []
    for k, kpt in enumerate(kpts_lst):
        if is_zero(kpt):
            soc_mat_kpts.append(soc_mat[k].real.reshape(3,nao,nao))
        else:
            soc_mat_kpts.append(soc_mat[k].reshape(3,nao,nao))

    if kpts is None or numpy.shape(kpts) == (3,):
        soc_mat_kpts = soc_mat_kpts[0]
    return numpy.asarray(soc_mat_kpts)

def _block_diag(mat):
    '''
    [A 0]
    [0 A]
    '''
    return scipy.linalg.block_diag(mat, mat)


def _sigma_dot(mat):
    '''sigma dot mat'''
    #pauli = 1j * lib.PauliMatrices
    #nao = mat.shape[-1] * 2
    #return lib.einsum('sxy,spq->xpyq', pauli, mat).reshape(nao, nao)
    quaternion = numpy.vstack([1j * lib.PauliMatrices, numpy.eye(2)[None, :, :]])
    nao = mat.shape[-1] * 2
    return lib.einsum('sxy,spq->xpyq', quaternion, mat).reshape(nao, nao)
