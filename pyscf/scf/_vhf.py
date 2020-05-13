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

import sys
import ctypes
import _ctypes
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.gto.moleintor import make_cintopt, make_loc, ascint3

libcvhf = lib.load_library('libcvhf')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, name))

class VHFOpt(object):
    def __init__(self, mol, intor,
                 prescreen='CVHFnoscreen', qcondname=None, dmcondname=None):
        '''If function "qcondname" is presented, the qcond (sqrt(integrals))
        and will be initialized in __init__.
        '''
        intor = mol._add_suffix(intor)
        self._this = ctypes.POINTER(_CVHFOpt)()
        #print self._this.contents, expect ValueError: NULL pointer access
        self._intor = intor
        self._cintopt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
        self._dmcondname = dmcondname
        natm = ctypes.c_int(mol.natm)
        nbas = ctypes.c_int(mol.nbas)
        libcvhf.CVHFinit_optimizer(ctypes.byref(self._this),
                                   mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                   mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                   mol._env.ctypes.data_as(ctypes.c_void_p))
        self.prescreen = prescreen
        if qcondname is not None:
            self.init_cvhf_direct(mol, intor, qcondname)

    def init_cvhf_direct(self, mol, intor, qcondname):
        intor = mol._add_suffix(intor)
        if intor == self._intor:
            cintopt = self._cintopt
        else:
            cintopt = lib.c_null_ptr()
        ao_loc = make_loc(mol._bas, intor)
        fsetqcond = getattr(libcvhf, qcondname)
        natm = ctypes.c_int(mol.natm)
        nbas = ctypes.c_int(mol.nbas)
        fsetqcond(self._this, getattr(libcvhf, intor), cintopt,
                  ao_loc.ctypes.data_as(ctypes.c_void_p),
                  mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                  mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                  mol._env.ctypes.data_as(ctypes.c_void_p))

    @property
    def direct_scf_tol(self):
        return self._this.contents.direct_scf_cutoff
    @direct_scf_tol.setter
    def direct_scf_tol(self, v):
        self._this.contents.direct_scf_cutoff = v

    @property
    def prescreen(self):
        return self._this.contents.fprescreen
    @prescreen.setter
    def prescreen(self, v):
        if isinstance(v, int):
            self._this.contents.fprescreen = v
        else:
            self._this.contents.fprescreen = _fpointer(v)

    def set_dm(self, dm, atm, bas, env):
        if self._dmcondname is not None:
            c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
            c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
            c_env = numpy.asarray(env, dtype=numpy.double, order='C')
            natm = ctypes.c_int(c_atm.shape[0])
            nbas = ctypes.c_int(c_bas.shape[0])
            if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
                n_dm = 1
            else:
                n_dm = len(dm)
            dm = numpy.asarray(dm, order='C')
            ao_loc = make_loc(c_bas, self._intor)
            fsetdm = getattr(libcvhf, self._dmcondname)
            fsetdm(self._this,
                   dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm),
                   ao_loc.ctypes.data_as(ctypes.c_void_p),
                   c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                   c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                   c_env.ctypes.data_as(ctypes.c_void_p))

    def __del__(self):
        try:
            libcvhf.CVHFdel_optimizer(ctypes.byref(self._this))
        except AttributeError:
            pass

class _CVHFOpt(ctypes.Structure):
    _fields_ = [('nbas', ctypes.c_int),
                ('_padding', ctypes.c_int),
                ('direct_scf_cutoff', ctypes.c_double),
                ('q_cond', ctypes.c_void_p),
                ('dm_cond', ctypes.c_void_p),
                ('fprescreen', ctypes.c_void_p),
                ('r_vkscreen', ctypes.c_void_p)]

################################################
# for general DM
# hermi = 0 : arbitary
# hermi = 1 : hermitian
# hermi = 2 : anti-hermitian
################################################
def incore(eri, dms, hermi=0, with_j=True, with_k=True):
    assert(eri.dtype == numpy.double)
    eri = numpy.asarray(eri, order='C')
    dms = numpy.asarray(dms, order='C')
    dms_shape = dms.shape
    nao = dms_shape[-1]

    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    vj = vk = None
    if with_j:
        vj = numpy.zeros((n_dm,nao,nao))
    if with_k:
        vk = numpy.zeros((n_dm,nao,nao))

    dmsptr = []
    vjkptr = []
    fjkptr = []

    npair = nao*(nao+1)//2
    if eri.ndim == 2 and npair*npair == eri.size: # 4-fold symmetry eri
        fdrv = getattr(libcvhf, 'CVHFnrs4_incore_drv')
        if with_j:
            # 'ijkl,kl->ij'
            fvj = _fpointer('CVHFics4_kl_s2ij')
            # or
            ## 'ijkl,ij->kl'
            #fvj = _fpointer('CVHFics4_ij_s2kl')
            for i, dm in enumerate(dms):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvj)
        if with_k:
            # 'ijkl,il->jk'
            fvk = _fpointer('CVHFics4_il_s1jk')
            # or
            ## 'ijkl,jk->il'
            #fvk = _fpointer('CVHFics4_jk_s1il')
            for i, dm in enumerate(dms):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvk)

    elif eri.ndim == 1 and npair*(npair+1)//2 == eri.size: # 8-fold symmetry eri
        fdrv = getattr(libcvhf, 'CVHFnrs8_incore_drv')
        if with_j:
            fvj = _fpointer('CVHFics8_tridm_vj')
            tridms = lib.pack_tril(lib.hermi_sum(dms, axes=(0,2,1)))
            idx = numpy.arange(nao)
            tridms[:,idx*(idx+1)//2+idx] *= .5
            for i, tridm in enumerate(tridms):
                dmsptr.append(tridm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvj)
        if with_k:
            if hermi == 1:
                fvk = _fpointer('CVHFics8_jk_s2il')
            else:
                fvk = _fpointer('CVHFics8_jk_s1il')
            for i, dm in enumerate(dms):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvk)
    else:
        raise RuntimeError('Array shape not consistent: DM %s, eri %s'
                           % (dms_shape, eri.shape))

    n_ops = len(dmsptr)
    fdrv(eri.ctypes.data_as(ctypes.c_void_p),
         (ctypes.c_void_p*n_ops)(*dmsptr), (ctypes.c_void_p*n_ops)(*vjkptr),
         ctypes.c_int(n_ops), ctypes.c_int(nao),
         (ctypes.c_void_p*n_ops)(*fjkptr))

    if with_j:
        for i in range(n_dm):
            lib.hermi_triu(vj[i], 1, inplace=True)
        vj = vj.reshape(dms_shape)
    if with_k:
        if hermi != 0:
            for i in range(n_dm):
                lib.hermi_triu(vk[i], hermi, inplace=True)
        vk = vk.reshape(dms_shape)
    return vj, vk

# use int2e_sph as cintor, CVHFnrs8_ij_s2kl, CVHFnrs8_jk_s2il as fjk to call
# direct_mapdm
def direct(dms, atm, bas, env, vhfopt=None, hermi=0, cart=False,
           with_j=True, with_k=True):
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    dms = numpy.asarray(dms, order='C')
    dms_shape = dms.shape
    nao = dms_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    if vhfopt is None:
        if cart:
            intor = 'int2e_cart'
        else:
            intor = 'int2e_sph'
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
        cvhfopt = lib.c_null_ptr()
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor
    cintor = _fpointer(intor)

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    fdot = _fpointer('CVHFdot_nrs8')

    vj = vk = None
    dmsptr = []
    vjkptr = []
    fjk = []

    if with_j:
        fvj = _fpointer('CVHFnrs8_ji_s2kl')
        vj = numpy.empty((n_dm,nao,nao))
        for i, dm in enumerate(dms):
            dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
            vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
            fjk.append(fvj)

    if with_k:
        if hermi == 1:
            fvk = _fpointer('CVHFnrs8_li_s2kj')
        else:
            fvk = _fpointer('CVHFnrs8_li_s1kj')
        vk = numpy.empty((n_dm,nao,nao))
        for i, dm in enumerate(dms):
            dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
            vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
            fjk.append(fvk)

    shls_slice = (ctypes.c_int*8)(*([0, c_bas.shape[0]]*4))
    ao_loc = make_loc(bas, intor)
    n_ops = len(dmsptr)
    comp = 1
    fdrv(cintor, fdot, (ctypes.c_void_p*n_ops)(*fjk),
         (ctypes.c_void_p*n_ops)(*dmsptr), (ctypes.c_void_p*n_ops)(*vjkptr),
         ctypes.c_int(n_ops), ctypes.c_int(comp),
         shls_slice, ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if with_j:
        # vj must be symmetric
        for i in range(n_dm):
            lib.hermi_triu(vj[i], 1, inplace=True)
        vj = vj.reshape(dms_shape)
    if with_k:
        if hermi != 0:
            for i in range(n_dm):
                lib.hermi_triu(vk[i], hermi, inplace=True)
        vk = vk.reshape(dms_shape)
    return vj, vk

# call all fjk for each dm, the return array has len(dms)*len(jkdescript)*ncomp components
# jkdescript: 'ij->s1kl', 'kl->s2ij', ...
def direct_mapdm(intor, aosym, jkdescript,
                 dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                 shls_slice=None):
    assert(aosym in ('s8', 's4', 's2ij', 's2kl', 's1',
                     'aa4', 'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    intor = ascint3(intor)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis,:,:]
    n_dm = len(dms)
    dms = [numpy.asarray(dm, order='C') for dm in dms]
    if isinstance(jkdescript, str):
        jkdescripts = (jkdescript,)
    else:
        jkdescripts = jkdescript
    njk = len(jkdescripts)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cvhfopt = lib.c_null_ptr()
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = getattr(libcvhf, vhfopt._intor)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    dotsym = _INTSYMAP[aosym]
    fdot = _fpointer('CVHFdot_nr'+dotsym)

    if shls_slice is None:
        shls_slice = (0, c_bas.shape[0])*4
    ao_loc = make_loc(bas, intor)

    vjk = []
    descr_sym = [x.split('->') for x in jkdescripts]
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dmsptr = (ctypes.c_void_p*(njk*n_dm))()
    vjkptr = (ctypes.c_void_p*(njk*n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        if dmsym in ('ij', 'kl', 'il', 'kj'):
            sys.stderr.write('not support DM description %s, transpose to %s\n' %
                             (dmsym, dmsym[::-1]))
            dmsym = dmsym[::-1]
        f1 = _fpointer('CVHFnr%s_%s_%s'%(aosym, dmsym, vsym))

        vshape = (n_dm,ncomp) + get_dims(vsym[-2:], shls_slice, ao_loc)
        vjk.append(numpy.empty(vshape))
        for j in range(n_dm):
            if dms[j].shape != get_dims(dmsym, shls_slice, ao_loc):
                raise RuntimeError('dm[%d] shape %s is inconsistent with the '
                                   'shls_slice shape %s' %
                                   (j, dms[j].shape, get_dims(dmsym, shls_slice, ao_loc)))
            dmsptr[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            vjkptr[i*n_dm+j] = vjk[i][j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    shls_slice = (ctypes.c_int*8)(*shls_slice)

    fdrv(cintor, fdot, fjk, dmsptr, vjkptr,
         ctypes.c_int(njk*n_dm), ctypes.c_int(ncomp),
         shls_slice, ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if n_dm * ncomp == 1:
        vjk = [v.reshape(v.shape[2:]) for v in vjk]
    elif n_dm == 1:
        vjk = [v.reshape((ncomp,)+v.shape[2:]) for v in vjk]
    elif ncomp == 1:
        vjk = [v.reshape((n_dm,)+v.shape[2:]) for v in vjk]
    if isinstance(jkdescript, str):
        vjk = vjk[0]
    return vjk

# for density matrices in dms, bind each dm to a jk operator
# jkdescript: 'ij->s1kl', 'kl->s2ij', ...
def direct_bindm(intor, aosym, jkdescript,
                 dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                 shls_slice=None):
    assert(aosym in ('s8', 's4', 's2ij', 's2kl', 's1',
                     'aa4', 'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    intor = ascint3(intor)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis,:,:]
    n_dm = len(dms)
    dms = [numpy.asarray(dm, order='C') for dm in dms]
    if isinstance(jkdescript, str):
        jkdescripts = (jkdescript,)
    else:
        jkdescripts = jkdescript
    njk = len(jkdescripts)
    assert(njk == n_dm)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cvhfopt = lib.c_null_ptr()
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = getattr(libcvhf, vhfopt._intor)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    dotsym = _INTSYMAP[aosym]
    fdot = _fpointer('CVHFdot_nr'+dotsym)

    if shls_slice is None:
        shls_slice = (0, c_bas.shape[0])*4
    ao_loc = make_loc(bas, intor)

    vjk = []
    descr_sym = [x.split('->') for x in jkdescripts]
    fjk = (ctypes.c_void_p*(n_dm))()
    dmsptr = (ctypes.c_void_p*(n_dm))()
    vjkptr = (ctypes.c_void_p*(n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFnr%s_%s_%s'%(aosym, dmsym, vsym))

        if dms[i].shape != get_dims(dmsym, shls_slice, ao_loc):
            raise RuntimeError('dm[%d] shape %s is inconsistent with the '
                               'shls_slice shape %s' %
                               (i, dms[i].shape, get_dims(dmsym, shls_slice, ao_loc)))
        vshape = (ncomp,) + get_dims(vsym[-2:], shls_slice, ao_loc)
        vjk.append(numpy.empty(vshape))
        dmsptr[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        vjkptr[i] = vjk[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    shls_slice = (ctypes.c_int*8)(*shls_slice)

    fdrv(cintor, fdot, fjk, dmsptr, vjkptr,
         ctypes.c_int(n_dm), ctypes.c_int(ncomp),
         shls_slice, ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if ncomp == 1:
        vjk = [v.reshape(v.shape[1:]) for v in vjk]
    if isinstance(jkdescript, str):
        vjk = vjk[0]
    return vjk


# 8-fold permutation symmetry
def int2e_sph(atm, bas, env, cart=False):  # pragma: no cover
    if cart:
        intor = 'int2e_cart'
    else:
        intor = 'int2e_sph'
    return gto.moleintor.getints4c(intor, atm, bas, env, aosym='s8')


################################################################
# relativistic
def rdirect_mapdm(intor, aosym, jkdescript,
                  dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                  shls_slice=None):
    assert(aosym in ('s8', 's4', 's2ij', 's2kl', 's1',
                     'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    intor = ascint3(intor)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis,:,:]
    n_dm = len(dms)
    nao = dms[0].shape[0]
    dms = numpy.asarray(dms, order='C', dtype=numpy.complex128)
    if isinstance(jkdescript, str):
        jkdescript = (jkdescript,)
    njk = len(jkdescript)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cvhfopt = lib.c_null_ptr()
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = getattr(libcvhf, vhfopt._intor)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    dotsym = _INTSYMAP[aosym]
    fdot = _fpointer('CVHFdot_r'+dotsym)

    if shls_slice is None:
        shls_slice = (0, c_bas.shape[0])*4
    else:
        raise NotImplementedError
    ao_loc = make_loc(bas, intor)

    unpackas = _INTUNPACKMAP_R[aosym]
    descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dm1 = (ctypes.c_void_p*(njk*n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFr%s_%s_%s'%(unpackas, dmsym, vsym))
        for j in range(n_dm):
            dm1[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.empty((njk,n_dm*ncomp,nao,nao), dtype=numpy.complex)

    fdrv(cintor, fdot, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(njk*n_dm), ctypes.c_int(ncomp),
         (ctypes.c_int*8)(*shls_slice),
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if n_dm * ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    if njk == 1:
        vjk = vjk.reshape(vjk.shape[1:])
    return vjk

# for density matrices in dms, bind each dm to a jk operator
def rdirect_bindm(intor, aosym, jkdescript,
                  dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                  shls_slice=None):
    assert(aosym in ('s8', 's4', 's2ij', 's2kl', 's1',
                     'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    intor = ascint3(intor)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis,:,:]
    n_dm = len(dms)
    nao = dms[0].shape[0]
    dms = numpy.asarray(dms, order='C', dtype=numpy.complex128)
    if isinstance(jkdescript, str):
        jkdescript = (jkdescript,)
    njk = len(jkdescript)
    assert(njk == n_dm)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cvhfopt = lib.c_null_ptr()
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = getattr(libcvhf, vhfopt._intor)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    dotsym = _INTSYMAP[aosym]
    fdot = _fpointer('CVHFdot_r'+dotsym)

    if shls_slice is None:
        shls_slice = (0, c_bas.shape[0])*4
    else:
        raise NotImplementedError
    ao_loc = make_loc(bas, intor)

    unpackas = _INTUNPACKMAP_R[aosym]
    descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(n_dm))()
    dm1 = (ctypes.c_void_p*(n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFr%s_%s_%s'%(unpackas, dmsym, vsym))
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    vjk = numpy.empty((njk,ncomp,nao,nao), dtype=numpy.complex)

    fdrv(cintor, fdot, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(n_dm), ctypes.c_int(ncomp),
         (ctypes.c_int*8)(*shls_slice),
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    if njk == 1:
        vjk = vjk.reshape(vjk.shape[1:])
    return vjk

# 'a4ij': anti-symm between ij, symm between kl
# 'a4kl': anti-symm between kl, symm between ij
# 'a2ij': anti-symm between ij,
# 'a2kl': anti-symm between kl,
_INTSYMAP= {
    's8'  : 's8'  ,
    's4'  : 's4'  ,
    's2ij': 's2ij',
    's2kl': 's2kl',
    's1'  : 's1'  ,
    'aa4' : 's4'  ,
    'a4ij': 's4'  ,
    'a4kl': 's4'  ,
    'a2ij': 's2ij',
    'a2kl': 's2kl',
}

_INTUNPACKMAP_R = {
    's8'  : 's8'  ,
    's4'  : 's4'  ,
    's2ij': 's2ij',
    's2kl': 's2kl',
    's1'  : 's1'  ,
    'a4ij': 'ah4'  ,
    'a4kl': 'ha4'  ,
    'a2ij': 'ah2ij',
    'a2kl': 'ha2kl',
}

_SHLINDEX = {'i': 0, 'j': 2, 'k': 4, 'l': 6}
def get_dims(descr_sym, shls_slice, ao_loc):
    i = _SHLINDEX[descr_sym[0]]
    j = _SHLINDEX[descr_sym[1]]
    di = ao_loc[shls_slice[i+1]] - ao_loc[shls_slice[i]]
    dj = ao_loc[shls_slice[j+1]] - ao_loc[shls_slice[j]]
    return (di,dj)

