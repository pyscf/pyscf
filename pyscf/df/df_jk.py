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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import copy
import time
import ctypes
from functools import reduce
import numpy
from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo

libri = lib.load_library('libri')

def density_fit(mf, auxbasis=None, with_df=None):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.

    Args:
        mf : an SCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.

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
    from pyscf.soscf import newton_ah
    assert(isinstance(mf, scf.hf.SCF))

    if isinstance(mf, _DFHF):
        if mf.with_df is None:
            mf = mf.__class__(mf)
        elif mf.with_df.auxbasis != auxbasis:
            if (isinstance(mf, newton_ah._CIAH_SOSCF) and
                isinstance(mf._scf, _DFHF)):
                mf.with_df = copy.copy(mf.with_df)
                mf.with_df.auxbasis = auxbasis
            else:
                raise RuntimeError('DFHF has been initialized. '
                                   'It cannot be initialized twice.')
        return mf

    if with_df is None:
        if isinstance(mf, dhf.UHF):
            with_df = df.DF4C(mf.mol)
        else:
            with_df = df.DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    mf_class = mf.__class__
    class DFHF(mf_class, _DFHF):
        __doc__ = '''
        Density fitting SCF class

        Attributes for density-fitting SCF:
            auxbasis : str or basis dict
                Same format to the input attribute mol.basis.
                The default basis 'weigend+etb' means weigend-coulomb-fit basis
                for light elements and even-tempered basis for heavy elements.
            with_df : DF object
                Set mf.with_df = None to switch off density fitting mode.

        See also the documents of class %s for other SCF attributes.
        ''' % mf_class
        def __init__(self, mf):
            self.__dict__.update(mf.__dict__)
            self._eri = None
            self.auxbasis = auxbasis
            self.direct_scf = False
            self.with_df = with_df
            self._keys = self._keys.union(['auxbasis', 'with_df'])

        def get_jk(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                if mol is None: mol = self.mol
                if dm is None: dm = self.make_rdm1()
                vj, vk = self.with_df.get_jk(dm, hermi)
                return vj, vk
            else:
                return mf_class.get_jk(self, mol, dm, hermi)

        def get_j(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                if mol is None: mol = self.mol
                if dm is None: dm = self.make_rdm1()
                vj = self.with_df.get_jk(dm, hermi, with_k=False)[0]
                return vj
            else:
                return mf_class.get_j(self, mol, dm, hermi)

        def get_k(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                if mol is None: mol = self.mol
                if dm is None: dm = self.make_rdm1()
                vk = self.with_df.get_jk(dm, hermi, with_j=False)[1]
                return vk
            else:
                return mf_class.get_k(self, mol, dm, hermi)

# _cderi accesser for pyscf 1.0, 1.1 compatibility
        @property
        def _cderi(self):
            return self.with_df._cderi
        @_cderi.setter
        def _cderi(self, x):
            self.with_df._cderi = x

    return DFHF(mf)

# A tag to label the derived SCF class
class _DFHF:
    pass


def get_jk(dfobj, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
    t0 = t1 = (time.clock(), time.time())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    assert(with_j or with_k)

    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = [0] * nset

    if not with_k:
        dmtril = []
        idx = numpy.arange(nao)
        for k in range(nset):
            dm = lib.pack_tril(dms[k]+dms[k].T)
            dm[idx*(idx+1)//2+idx] *= .5
            dmtril.append(dm)
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            for k in range(nset):
                rho = numpy.einsum('px,x->p', eri1, dmtril[k])
                vj[k] += numpy.einsum('p,px->x', rho, eri1)

    elif hasattr(dm, 'mo_coeff'):
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
            assert(mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
            mo_occ = numpy.vstack((mo_occa, mo_occb))

        dmtril = []
        orbo = []
        for k in range(nset):
            if with_j:
                dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5

            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
            orbo.append(numpy.asarray(c, order='F'))

        buf = numpy.empty((dfobj.blockdim*nao,nao))
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    rho = numpy.einsum('px,x->p', eri1, dmtril[k])
                    vj[k] += numpy.einsum('p,px->x', rho, eri1)

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
        buf = numpy.empty((2,dfobj.blockdim,nao,nao))
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            for k in range(nset):
                buf1 = buf[0,:naux]
                fdrv(ftrans, fmmm,
                     buf1.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dms[k].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs)
                if with_j:
                    rho = numpy.einsum('kii->k', buf1)
                    vj[k] += numpy.einsum('p,px->x', rho, eri1)

                buf2 = lib.unpack_tril(eri1, out=buf[1])
                vk[k] += lib.dot(buf1.reshape(-1,nao).T,
                                 buf2.reshape(-1,nao))
            t1 = log.timer_debug1('jk', *t1)

    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = numpy.asarray(vk).reshape(dm_shape)
    logger.timer(dfobj, 'vj and vk', *t0)
    return vj, vk


def r_get_jk(dfobj, dms, hermi=1):
    '''Relativistic density fitting JK'''
    t0 = (time.clock(), time.time())
    mol = dfobj.mol
    c1 = .5 / lib.param.LIGHT_SPEED
    tao = mol.tmap()
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]

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
            buf = numpy.empty((naux,n2c,n2c), dtype=numpy.complex)
            buf1 = numpy.empty((naux,n2c,n2c), dtype=numpy.complex)

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
