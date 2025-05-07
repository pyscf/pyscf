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

import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo

DEBUG = False

libri = lib.load_library('libri')

def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.

    Args:
        mf : an SCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.

        only_dfj : str
            Compute Coulomb integrals only and no approximation for HF
            exchange. Same to RIJONX in ORCA

    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.density_fit(scf.RHF(mol))
    >>> mf.scf()
    -100.005306000435510

    >>> mol.symmetry = 1
    >>> mol.build(0, 0)
    >>> mf = scf.density_fit(scf.UHF(mol))
    >>> mf.scf()
    -100.005306000435510
    '''
    from pyscf import df
    from pyscf.scf import dhf
    from pyscf.df.addons import predefined_auxbasis
    assert (isinstance(mf, scf.hf.SCF))

    if with_df is None:
        mol = mf.mol
        if auxbasis is None and isinstance(mol.basis, str):
            if isinstance(mf, scf.hf.KohnShamDFT):
                xc = mf.xc
            else:
                xc = 'HF'
            if xc == 'LDA,VWN':
                # This is likely the default xc setting of a KS instance.
                # Postpone the auxbasis assignment to with_df.build().
                auxbasis = None
            else:
                auxbasis = predefined_auxbasis(mol, mol.basis, xc)
        if isinstance(mf, dhf.UHF):
            with_df = df.DF4C(mol, auxbasis)
        else:
            with_df = df.DF(mol, auxbasis)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose

    if isinstance(mf, _DFHF):
        mf = mf.copy()
        mf.with_df = with_df
        mf.only_dfj = only_dfj
        return mf

    dfmf = _DFHF(mf, with_df, only_dfj)
    return lib.set_class(dfmf, (_DFHF, mf.__class__))

# 1. A tag to label the derived SCF class
# 2. A hook to register DF specific methods, such as nuc_grad_method.
class _DFHF:
    '''
    Density fitting SCF class

    Attributes for density-fitting SCF:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        with_df : DF object
            Set mf.with_df = None to switch off density fitting mode.

    See also the documents of class for other SCF attributes.
    '''

    __name_mixin__ = 'DF'

    _keys = {'with_df', 'only_dfj'}

    def __init__(self, mf, df=None, only_dfj=None):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.with_df = df
        self.only_dfj = only_dfj
        # Unless DF is used only for J matrix, disable direct_scf for K build.
        # It is more efficient to construct K matrix with MO coefficients than
        # the incremental method in direct_scf.
        self.direct_scf = only_dfj

    def undo_df(self):
        '''Remove the DFHF Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFHF))
        del obj.with_df, obj.only_dfj
        return obj

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        assert (with_j or with_k)
        if dm is None: dm = self.make_rdm1()
        if not self.with_df:
            return super().get_jk(mol, dm, hermi, with_j, with_k, omega)

        vj = vk = None
        with_dfk = with_k and not self.only_dfj
        if with_j or with_dfk:
            if isinstance(self, scf.ghf.GHF):
                def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
                    vj, vk = self.with_df.get_jk(dm.real, hermi, with_j, with_k,
                                                self.direct_scf_tol, omega)
                    if dm.dtype == numpy.complex128:
                        vjI, vkI = self.with_df.get_jk(dm.imag, hermi, with_j, with_k,
                                                    self.direct_scf_tol, omega)
                        if with_j:
                            vj = vj + vjI * 1j
                        if with_k:
                            vk = vk + vkI * 1j
                    return vj, vk
                vj, vk = scf.ghf.get_jk(mol, dm, hermi, with_j, with_dfk,
                                        jkbuild, omega)
            else:
                vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_dfk,
                                            self.direct_scf_tol, omega)
        if with_k and not with_dfk:
            vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        return vj, vk

    # for pyscf 1.0, 1.1 compatibility
    @property
    def _cderi(self):
        naux = self.with_df.get_naoaux()
        return next(self.with_df.loop(blksize=naux))
    @_cderi.setter
    def _cderi(self, x):
        self.with_df._cderi = x

    @property
    def auxbasis(self):
        return getattr(self.with_df, 'auxbasis', None)

    def nuc_grad_method(self):
        from pyscf.df.grad import rhf, rohf, uhf, rks, roks, uks
        if isinstance(self, scf.uhf.UHF):
            if isinstance(self, scf.hf.KohnShamDFT):
                return uks.Gradients(self)
            else:
                return uhf.Gradients(self)
        elif isinstance(self, scf.rohf.ROHF):
            if isinstance(self, scf.hf.KohnShamDFT):
                return roks.Gradients(self)
            else:
                return rohf.Gradients(self)
        elif isinstance(self, scf.rhf.RHF):
            if isinstance(self, scf.hf.KohnShamDFT):
                return rks.Gradients(self)
            else:
                return rhf.Gradients(self)
        else:
            raise NotImplementedError

    Gradients = nuc_grad_method

    def Hessian(self):
        from pyscf.df.hessian import rhf, uhf, rks, uks
        if isinstance(self, (scf.uhf.UHF, scf.rohf.ROHF)):
            if isinstance(self, scf.hf.KohnShamDFT):
                return uks.Hessian(self)
            else:
                return uhf.Hessian(self)
        elif isinstance(self, scf.rohf.ROHF):
            raise NotImplementedError
        elif isinstance(self, scf.rhf.RHF):
            if isinstance(self, scf.hf.KohnShamDFT):
                return rks.Hessian(self)
            else:
                return rhf.Hessian(self)
        else:
            raise NotImplementedError

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError
    NMR = method_not_implemented
    NSR = method_not_implemented
    Polarizability = method_not_implemented
    RotationalGTensor = method_not_implemented
    MP2 = method_not_implemented
    CISD = method_not_implemented
    CCSD = method_not_implemented

    def CASCI(self, ncas, nelecas, auxbasis=None, ncore=None):
        from pyscf import mcscf
        return mcscf.DFCASCI(self, ncas, nelecas, auxbasis, ncore)

    def CASSCF(self, ncas, nelecas, auxbasis=None, ncore=None, frozen=None):
        from pyscf import mcscf
        return mcscf.DFCASSCF(self, ncas, nelecas, auxbasis, ncore, frozen)

    def to_gpu(self):
        obj = self.undo_df().to_gpu().density_fit()
        return lib.to_gpu(self, obj)


def get_jk(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
    assert (with_j or with_k)
    if (not with_k and not dfobj.mol.incore_anyway and
        # 3-center integral tensor is not initialized
        dfobj._cderi is None):
        return get_j(dfobj, dm, hermi, direct_scf_tol), None

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = 0
    vk = numpy.zeros_like(dms)

    if numpy.iscomplexobj(dms):
        if with_j:
            vj = numpy.zeros_like(dms)
        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/nao**2)))
        buf = numpy.empty((blksize,nao,nao))
        buf1 = numpy.empty((nao,blksize,nao))
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            eri1 = lib.unpack_tril(eri1, out=buf)
            if with_j:
                tmp = numpy.einsum('pij,nji->pn', eri1, dms.real)
                vj.real += numpy.einsum('pn,pij->nij', tmp, eri1)
                tmp = numpy.einsum('pij,nji->pn', eri1, dms.imag)
                vj.imag += numpy.einsum('pn,pij->nij', tmp, eri1)
            buf2 = numpy.ndarray((nao,naux,nao), buffer=buf1)
            for k in range(nset):
                buf2[:] = lib.einsum('pij,jk->ipk', eri1, dms[k].real)
                vk[k].real += lib.einsum('ipk,pkj->ij', buf2, eri1)
                buf2[:] = lib.einsum('pij,jk->ipk', eri1, dms[k].imag)
                vk[k].imag += lib.einsum('ipk,pkj->ij', buf2, eri1)
            t1 = log.timer_debug1('jk', *t1)
        if with_j: vj = vj.reshape(dm_shape)
        if with_k: vk = vk.reshape(dm_shape)
        logger.timer(dfobj, 'df vj and vk', *t0)
        return vj, vk

    if with_j:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

    if not with_k:
        for eri1 in dfobj.loop():
            # uses numpy.matmul
            vj += dmtril.dot(eri1.T).dot(eri1)

    elif getattr(dm, 'mo_coeff', None) is not None:
        #TODO: test whether dm.mo_coeff matching dm
        mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
        mo_occ   = numpy.asarray(dm.mo_occ)
        nmo = mo_occ.shape[-1]
        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
        mo_occ   = mo_occ.reshape(-1,nmo)
        if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
            mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
            assert (mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
            mo_occ = numpy.vstack((mo_occa, mo_occb))

        orbo = []
        for k in range(nset):
            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
            orbo.append(numpy.asarray(c, order='F'))

        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
        buf = numpy.empty((blksize*nao,nao))
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            assert (nao_pair == nao*(nao+1)//2)
            if with_j:
                # uses numpy.matmul
                vj += dmtril.dot(eri1.T).dot(eri1)

            for k in range(nset):
                nocc = orbo[k].shape[1]
                if nocc > 0:
                    buf1 = buf[:naux*nocc]
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbo[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    vk[k] += lib.dot(buf1.T, buf1)
            t1 = log.timer_debug1('jk', *t1)
    else:
        #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
        #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
        rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
                 null, ctypes.c_int(0))
        dms = [numpy.asarray(x, order='F') for x in dms]
        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/nao**2)))
        buf = numpy.empty((2,blksize,nao,nao))
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            assert (nao_pair == nao*(nao+1)//2)
            if with_j:
                # uses numpy.matmul
                vj += dmtril.dot(eri1.T).dot(eri1)

            for k in range(nset):
                buf1 = buf[0,:naux]
                fdrv(ftrans, fmmm,
                     buf1.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dms[k].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs)

                buf2 = lib.unpack_tril(eri1, out=buf[1])
                vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))
            t1 = log.timer_debug1('jk', *t1)

    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = vk.reshape(dm_shape)
    logger.timer(dfobj, 'df vj and vk', *t0)
    return vj, vk

def get_j(dfobj, dm, hermi=0, direct_scf_tol=1e-13):
    from pyscf.scf import _vhf
    from pyscf.scf import jk
    from pyscf.df import addons
    t0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = dfobj.mol
    if dfobj._vjopt is None:
        dfobj.auxmol = auxmol = addons.make_auxmol(mol, dfobj.auxbasis)
        opt = _vhf._VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond',
                           dmcondname='CVHFnr_dm_cond',
                           direct_scf_tol=direct_scf_tol)

        # q_cond part 1: the regular int2e (ij|ij) for mol's basis
        opt.init_cvhf_direct(mol, 'int2e', 'CVHFnr_int2e_q_cond')

        # Update q_cond to include the 2e-integrals (auxmol|auxmol)
        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = numpy.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [j2c_diag[i0:i1].max()
                      for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        q_cond = numpy.hstack((opt.q_cond.ravel(), aux_q_cond))
        opt.q_cond = q_cond

        try:
            opt.j2c = j2c = scipy.linalg.cho_factor(j2c, lower=True)
            opt.j2c_type = 'cd'
        except scipy.linalg.LinAlgError:
            opt.j2c = j2c
            opt.j2c_type = 'regular'

        # jk.get_jk function supports 4-index integrals. Use bas_placeholder
        # (l=0, nctr=1, 1 function) to hold the last index.
        bas_placeholder = numpy.array([0, 0, 1, 1, 0, 0, 0, 0],
                                      dtype=numpy.int32)
        fakemol = mol + auxmol
        fakemol._bas = numpy.vstack((fakemol._bas, bas_placeholder))
        opt.fakemol = fakemol
        dfobj._vjopt = opt
        t1 = logger.timer_debug1(dfobj, 'df-vj init_direct_scf', *t1)

    opt = dfobj._vjopt
    fakemol = opt.fakemol
    dm = numpy.asarray(dm, order='C')
    assert dm.dtype == numpy.float64
    dm_shape = dm.shape
    nao = dm_shape[-1]
    dm = dm.reshape(-1,nao,nao)
    n_dm = dm.shape[0]

    # First compute the density in auxiliary basis
    # j3c = fauxe2(mol, auxmol)
    # jaux = numpy.einsum('ijk,ji->k', j3c, dm)
    # rho = numpy.linalg.solve(auxmol.intor('int2c2e'), jaux)
    nbas = mol.nbas
    nbas1 = mol.nbas + dfobj.auxmol.nbas
    shls_slice = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass1_prescreen'):
        jaux = jk.get_jk(fakemol, dm, ['ijkl,ji->kl']*n_dm, 'int3c2e',
                         aosym='s2ij', hermi=0, shls_slice=shls_slice,
                         vhfopt=opt)
    # remove the index corresponding to bas_placeholder
    jaux = numpy.array(jaux)[:,:,0]
    t1 = logger.timer_debug1(dfobj, 'df-vj pass 1', *t1)

    if opt.j2c_type == 'cd':
        rho = scipy.linalg.cho_solve(opt.j2c, jaux.T)
    else:
        rho = scipy.linalg.solve(opt.j2c, jaux.T)
    # transform rho to shape (:,1,naux), to adapt to 3c2e integrals (ij|k)
    rho = rho.T[:,numpy.newaxis,:]
    t1 = logger.timer_debug1(dfobj, 'df-vj solve ', *t1)

    # Next compute the Coulomb matrix
    # j3c = fauxe2(mol, auxmol)
    # vj = numpy.einsum('ijk,k->ij', j3c, rho)
    # temporarily set "_dmcondname=None" to skip the call to set_dm method.
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass2_prescreen',
                           _dmcondname=None):
        # CVHFnr3c2e_vj_pass2_prescreen requires custom dm_cond
        aux_loc = dfobj.auxmol.ao_loc
        dm_cond = [abs(rho[:,:,i0:i1]).max()
                   for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        opt.dm_cond = numpy.array(dm_cond)
        vj = jk.get_jk(fakemol, rho, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                       aosym='s2ij', hermi=1, shls_slice=shls_slice,
                       vhfopt=opt)

    t1 = logger.timer_debug1(dfobj, 'df-vj pass 2', *t1)
    logger.timer(dfobj, 'df-vj', *t0)
    return numpy.asarray(vj).reshape(dm_shape)


def r_get_jk(dfobj, dms, hermi=0, with_j=True, with_k=True):
    '''Relativistic density fitting JK'''
    t0 = (logger.process_clock(), logger.perf_counter())
    mol = dfobj.mol
    c1 = .5 / lib.param.LIGHT_SPEED
    tao = mol.tmap()
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]

    if hermi == 0 and DEBUG:
        # J matrix is symmetrized in this function which is only true for
        # density matrix with time reversal symmetry
        scf.dhf._ensure_time_reversal_symmetry(mol, dms)

    def fjk(dm):
        dm = numpy.asarray(dm, dtype=numpy.complex128)
        fmmm = libri.RIhalfmmm_r_s2_bra_noconj
        fdrv = _ao2mo.libao2mo.AO2MOr_e2_drv
        ftrans = libri.RItranse2_r_s2
        vj = numpy.zeros_like(dm)
        vk = numpy.zeros_like(dm)
        fcopy = libri.RImmm_r_s2_transpose
        rargs = (ctypes.c_int(n2c), (ctypes.c_int*4)(0, n2c, 0, 0),
                 tao.ctypes.data_as(ctypes.c_void_p),
                 ao_loc.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(mol.nbas))
        dmll = numpy.asarray(dm[:n2c,:n2c], order='C')
        dmls = numpy.asarray(dm[:n2c,n2c:], order='C') * c1
        dmsl = numpy.asarray(dm[n2c:,:n2c], order='C') * c1
        dmss = numpy.asarray(dm[n2c:,n2c:], order='C') * c1**2
        for erill, eriss in dfobj.loop():
            naux, nao_pair = erill.shape
            buf = numpy.empty((naux,n2c,n2c), dtype=numpy.complex128)
            buf1 = numpy.empty((naux,n2c,n2c), dtype=numpy.complex128)

            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 erill.ctypes.data_as(ctypes.c_void_p),
                 dmll.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|LL)
            rho = numpy.einsum('kii->k', buf)

            fdrv(ftrans, fcopy,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 erill.ctypes.data_as(ctypes.c_void_p),
                 dmll.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf1 == (P|LL)
            vk[:n2c,:n2c] += numpy.dot(buf1.reshape(-1,n2c).T,
                                       buf.reshape(-1,n2c))

            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eriss.ctypes.data_as(ctypes.c_void_p),
                 dmls.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|LS)
            vk[:n2c,n2c:] += numpy.dot(buf1.reshape(-1,n2c).T,
                                       buf.reshape(-1,n2c)) * c1

            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eriss.ctypes.data_as(ctypes.c_void_p),
                 dmss.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|SS)
            rho += numpy.einsum('kii->k', buf)
            vj[:n2c,:n2c] += lib.unpack_tril(numpy.dot(rho, erill), 1)
            vj[n2c:,n2c:] += lib.unpack_tril(numpy.dot(rho, eriss), 1) * c1**2

            fdrv(ftrans, fcopy,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 eriss.ctypes.data_as(ctypes.c_void_p),
                 dmss.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|SS)
            vk[n2c:,n2c:] += numpy.dot(buf1.reshape(-1,n2c).T,
                                       buf.reshape(-1,n2c)) * c1**2

            if hermi != 1:
                fdrv(ftrans, fmmm,
                     buf.ctypes.data_as(ctypes.c_void_p),
                     erill.ctypes.data_as(ctypes.c_void_p),
                     dmsl.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs) # buf == (P|SL)
                vk[n2c:,:n2c] += numpy.dot(buf1.reshape(-1,n2c).T,
                                           buf.reshape(-1,n2c)) * c1
        if hermi == 1:
            vk[n2c:,:n2c] = vk[:n2c,n2c:].T.conj()
        return vj, vk

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vj, vk = fjk(dms)
    else:
        vjk = [fjk(dm) for dm in dms]
        vj = numpy.array([x[0] for x in vjk])
        vk = numpy.array([x[1] for x in vjk])
    logger.timer(dfobj, 'vj and vk', *t0)
    return vj, vk


if __name__ == '__main__':
    import pyscf.gto
    import pyscf.scf
    mol = pyscf.gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    method = density_fit(pyscf.scf.RHF(mol), 'weigend')
    method.max_memory = 0
    energy = method.scf()
    print(energy, -76.0259362997)

    method = density_fit(pyscf.scf.DHF(mol), 'weigend')
    energy = method.scf()
    print(energy, -76.0807386770) # normal DHF energy is -76.0815679438127

    method = density_fit(pyscf.scf.UKS(mol), 'weigend', only_dfj = True)
    energy = method.scf()
    print(energy, -75.8547753298)

    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
        spin = 1,
        charge = 1,
    )

    method = density_fit(pyscf.scf.UHF(mol), 'weigend')
    energy = method.scf()
    print(energy, -75.6310072359)

    method = density_fit(pyscf.scf.RHF(mol), 'weigend')
    energy = method.scf()
    print(energy, -75.6265157244)
