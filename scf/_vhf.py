#!/usr/bin/env python

import os
import ctypes
import _ctypes
import numpy
import pyscf.lib

libcvhf = pyscf.lib.load_library('libcvhf')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, name))

class VHFOpt(object):
    def __init__(self, mol, intor,
                 prescreen='CVHFnoscreen', qcondname=None, dmcondname=None):
        self._this = ctypes.POINTER(_CVHFOpt)()
        #print self._this.contents, expect ValueError: NULL pointer access
        self._intor = _fpointer(intor)
        self._cintopt = ctypes.c_void_p()
        self._dmcondname = dmcondname
        self.init_cvhf_direct(mol, intor, prescreen, qcondname)

    def __del__(self):
        libcvhf.CINTdel_optimizer(ctypes.byref(self._cintopt))
        libcvhf.CVHFdel_optimizer(ctypes.byref(self._this))

    def init_cvhf_direct(self, mol, intor, prescreen, qcondname):
        c_atm = numpy.array(mol._atm, dtype=numpy.int32)
        c_bas = numpy.array(mol._bas, dtype=numpy.int32)
        c_env = numpy.array(mol._env)
        natm = ctypes.c_int(c_atm.shape[0])
        nbas = ctypes.c_int(c_bas.shape[0])
        self._cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

#        libcvhf.CVHFnr_optimizer(ctypes.byref(self._this),
#                                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
#                                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
#                                 c_env.ctypes.data_as(ctypes.c_void_p))
        libcvhf.CVHFinit_optimizer(ctypes.byref(self._this),
                                   c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                                   c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                   c_env.ctypes.data_as(ctypes.c_void_p))
        self._this.contents.fprescreen = _fpointer(prescreen)

        if prescreen != 'CVHFnoscreen':
            fsetqcond = getattr(libcvhf, qcondname)
            fsetqcond(self._this,
                      c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                      c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                      c_env.ctypes.data_as(ctypes.c_void_p))

    @property
    def direct_scf_tol(self):
        return self._this.contents.direct_scf_cutoff
    @direct_scf_tol.setter
    def direct_scf_tol(self, v):
        self._this.contents.direct_scf_cutoff = v

    def set_dm_(self, dm, atm, bas, env):
        if self._dmcondname is not None:
            c_atm = numpy.array(atm, dtype=numpy.int32)
            c_bas = numpy.array(bas, dtype=numpy.int32)
            c_env = numpy.array(env)
            natm = ctypes.c_int(c_atm.shape[0])
            nbas = ctypes.c_int(c_bas.shape[0])
            if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
                n_dm = 1
            else:
                n_dm = len(dm)
            dm = numpy.ascontiguousarray(dm)
            fsetdm = getattr(libcvhf, self._dmcondname)
            fsetdm(self._this,
                   dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm),
                   c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                   c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                   c_env.ctypes.data_as(ctypes.c_void_p))

class _CVHFOpt(ctypes.Structure):
    _fields_ = [('nbas', ctypes.c_int),
                ('_padding', ctypes.c_int),
                ('direct_scf_cutoff', ctypes.c_double),
                ('q_cond', ctypes.c_void_p),
                ('dm_cond', ctypes.c_void_p),
                ('fprescreen', ctypes.c_void_p),
                ('r_vkscreen', ctypes.c_void_p)]

def make_cintopt(atm, bas, env, intor):
    c_atm = numpy.array(atm, dtype=numpy.int32, copy=False)
    c_bas = numpy.array(bas, dtype=numpy.int32, copy=False)
    c_env = numpy.array(env, copy=False)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])
    cintopt = ctypes.c_void_p()
    foptinit = getattr(libcvhf, intor+'_optimizer')
    foptinit(ctypes.byref(cintopt),
             c_atm.ctypes.data_as(ctypes.c_void_p), natm,
             c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
             c_env.ctypes.data_as(ctypes.c_void_p))
    return cintopt

################################################
# for general DM
# hermi = 0 : arbitary
# hermi = 1 : hermitian
# hermi = 2 : anti-hermitian
################################################
def incore(eri, dm, hermi=0):
    eri = numpy.ascontiguousarray(eri)
    dm = numpy.ascontiguousarray(dm)
    nao = dm.shape[0]
    vj = numpy.empty((nao,nao))
    vk = numpy.empty((nao,nao))
    npair = nao*(nao+1)//2
    if eri.ndim == 2 and npair*npair == eri.size: # 4-fold symmetry eri
        fdrv = getattr(libcvhf, 'CVHFnrs4_incore_drv')
        # 'ijkl,kl->ij'
        fvj = _fpointer('CVHFnrs4_kl_s2ij')
        # 'ijkl,il->jk'
        fvk = _fpointer('CVHFnrs4_il_s1jk')
        # or
        ## 'ijkl,ij->kl'
        #fvj = _fpointer('CVHFnrs4_ij_s2kl')
        ## 'ijkl,jk->il'
        #fvk = _fpointer('CVHFnrs4_jk_s1il')

        tridm = dm
    else: # 8-fold symmetry eri
        fdrv = getattr(libcvhf, 'CVHFnrs8_incore_drv')
        fvj = _fpointer('CVHFnrs8_tridm_vj')
        fvk = _fpointer('CVHFnrs8_jk_s2il')
        tridm = pyscf.lib.pack_tril(pyscf.lib.transpose_sum(dm))
        for i in range(nao):
            tridm[i*(i+1)//2+i] *= .5
    fdrv(eri.ctypes.data_as(ctypes.c_void_p),
         tridm.ctypes.data_as(ctypes.c_void_p),
         vj.ctypes.data_as(ctypes.c_void_p),
         dm.ctypes.data_as(ctypes.c_void_p),
         vk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nao), fvj, fvk)
    if hermi != 0:
        vj = pyscf.lib.hermi_triu(vj, hermi)
        vk = pyscf.lib.hermi_triu(vk, hermi)
    else:
        vj = pyscf.lib.hermi_triu(vj, 1)
    return vj, vk

# use cint2e_sph as cintor, CVHFnrs8_ij_s2kl, CVHFnrs8_jk_s2il as fjk to call
# direct_mapdm
def direct(dms, atm, bas, env, vhfopt=None, hermi=0):
    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        n_dm = 1
        nao = dms.shape[0]
        dms = (dms,)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
    npair = nao*(nao+1)//2
    tridm = numpy.empty((n_dm,nao*nao))
    for idm in range(n_dm):
        tridm[idm,:npair] = pyscf.lib.pack_tril(pyscf.lib.transpose_sum(dms[idm]))
        for i in range(nao):
            tridm[idm,i*(i+1)//2+i] *= .5

    if vhfopt is None:
        cintor = _fpointer('cint2e_sph')
        cintopt = make_cintopt(c_atm, c_bas, c_env, 'cint2e_sph')
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    funpack = _fpointer('CVHFunpack_nrblock2tril')
    fdot = _fpointer('CVHFfill_dot_nrs8')
    fvj = _fpointer('CVHFnrs8_tridm_vj')
    if hermi == 1:
        fvk = _fpointer('CVHFnrs8_jk_s2il')
    else:
        fvk = _fpointer('CVHFnrs8_jk_s1il')
    fjk = (ctypes.c_void_p*(2*n_dm))()
    dm1 = (ctypes.c_void_p*(2*n_dm))()
    for i in range(n_dm):
        dm1[i] = tridm[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = fvj
    for i in range(n_dm):
        assert(dms[i].flags.c_contiguous)
        dm1[n_dm+i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[n_dm+i] = fvk
    vjk = numpy.empty((2,n_dm,nao,nao))

    fdrv(cintor, fdot, funpack, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(n_dm*2), ctypes.c_int(1),
         cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if vhfopt is None:
        libcvhf.CINTdel_optimizer(ctypes.byref(cintopt))

    # vj must be symmetric
    for idm in range(n_dm):
        vjk[0,idm] = pyscf.lib.hermi_triu(vjk[0,idm], 1)
    if hermi != 0: # vk depends
        for idm in range(n_dm):
            vjk[1,idm] = pyscf.lib.hermi_triu(vjk[1,idm], hermi)
    if n_dm == 1:
        vjk = vjk.reshape(2,nao,nao)
    return vjk

# call all fjk for each dm, the return array has len(dms)*len(jkdescript)*ncomp components
# jkdescript: 'ij->s1kl', 'kl->s2ij', ...
def direct_mapdm(intor, intsymm, jkdescript,
                 dms, ncomp, atm, bas, env, vhfopt=None):
    assert(intsymm in ('s8', 's4', 's2ij', 's2kl', 's1',
                       'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        n_dm = 1
        nao = dms.shape[0]
        dms = (dms,)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
    if isinstance(jkdescript, str):
        njk = 1
        jkdescript = (jkdescript,)
    else:
        njk = len(jkdescript)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    dotsym = _INTSYMAP[intsymm]
    fdot = _fpointer('CVHFfill_dot_nr'+dotsym)

# nr_direct driver loop over kl pair. For each kl, funpack fills ij
    unpackname, unpackijas = _INTUNPACKMAP_NR[intsymm]
    funpack = _fpointer(unpackname)

    if isinstance(jkdescript, str):
        descr_sym = [[jkdescript.split('->')]]
    else:
        descr_sym = [x.split('->') for x in jkdescript]
# _swap_ik_jl because the implicit transposing in funpack.
# the resulting CVHFnr??_??_s??? can match the funpack function
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dm1 = (ctypes.c_void_p*(njk*n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFnr%s_%s_%s'%(unpackijas,
                                         _swap_ik_jl(dmsym),
                                         _swap_ik_jl(vsym).replace('kj','jk')))
        for j in range(n_dm):
            assert(dms[j].flags.c_contiguous)
            dm1[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.empty((njk,n_dm*ncomp,nao,nao))

    fdrv(cintor, fdot, funpack, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(njk*n_dm), ctypes.c_int(ncomp),
         cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if vhfopt is None:
        libcvhf.CINTdel_optimizer(ctypes.byref(cintopt))

    for i, (dmsym, vsym) in enumerate(descr_sym):
        if 's1il' == vsym: # which is computed as CVHFnr?_?_s1jk
            vjk[i] = vjk[i].transpose(0,2,1)
    if n_dm * ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    if njk == 1:
        vjk = vjk.reshape(vjk.shape[1:])
    return vjk

# for density matrices in dms, bind each dm to a jk operator
# jkdescript: 'ij->s1kl', 'kl->s2ij', ...
def direct_bindm(intor, intsymm, jkdescript,
                 dms, ncomp, atm, bas, env, vhfopt=None):
    assert(intsymm in ('s8', 's4', 's2ij', 's2kl', 's1',
                       'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        n_dm = 1
        nao = dms.shape[0]
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
    if isinstance(jkdescript, str):
        njk = 1
        jkdescript = (jkdescript,)
    else:
        njk = len(jkdescript)
    assert(njk == n_dm)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    dotsym = _INTSYMAP[intsymm]
    fdot = _fpointer('CVHFfill_dot_nr'+dotsym)

# nr_direct driver loop over kl pair. For each kl, funpack fills ij
    unpackname, unpackijas = _INTUNPACKMAP_NR[intsymm]
    funpack = _fpointer(unpackname)

    if isinstance(jkdescript, str):
        descr_sym = [[jkdescript.split('->')]]
    else:
        descr_sym = [x.split('->') for x in jkdescript]
# _swap_ik_jl because the implicit transposing in nr_direct driver
# swap ik, jl, then the resulting CVHFnr??_??_s??? can handle it
    fjk = (ctypes.c_void_p*(n_dm))()
    dm1 = (ctypes.c_void_p*(n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFnr%s_%s_%s'%(unpackijas,
                                         _swap_ik_jl(dmsym),
                                         _swap_ik_jl(vsym).replace('kj','jk')))
        assert(dms[i].flags.c_contiguous)
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    vjk = numpy.empty((njk,ncomp,nao,nao))

    fdrv(cintor, fdot, funpack, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(n_dm), ctypes.c_int(ncomp),
         cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if vhfopt is None:
        libcvhf.CINTdel_optimizer(ctypes.byref(cintopt))

    for i, (dmsym, vsym) in enumerate(descr_sym):
        if 's1il' == vsym: # which is computed as CVHFnr?_?_s1jk
            vjk[i] = vjk[i].transpose(0,2,1)
    if ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    if njk == 1:
        vjk = vjk.reshape(vjk.shape[1:])
    return vjk


# 8-fold permutation symmetry
def int2e_sph(atm, bas, env):
    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])
    libcvhf.CINTtot_cgto_spheric.restype = ctypes.c_int
    nao = libcvhf.CINTtot_cgto_spheric(c_bas.ctypes.data_as(ctypes.c_void_p), nbas)
    nao_pair = nao*(nao+1)//2
    eri = numpy.empty((nao_pair*(nao_pair+1)//2))
    libcvhf.int2e_sph_o5(eri.ctypes.data_as(ctypes.c_void_p),
                         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                         c_env.ctypes.data_as(ctypes.c_void_p))
    return eri


def incore_o2(eri, dm, hermi=1):
    '''use 4-fold symmetry for eri, ijkl=ijlk=jikl=jilk'''
    eri = numpy.ascontiguousarray(eri)
    dm = numpy.ascontiguousarray(dm)
    nao = dm.shape[0]
    dm0 = pyscf.lib.pack_tril(dm) * 2
    for i in range(nao):
        dm0[i*(i+1)//2+i] *= .5
    vj = pyscf.lib.unpack_tril(numpy.dot(eri, dm0))
    vk = numpy.zeros((nao,nao))
    libcvhf.CVHFnr_k(ctypes.c_int(nao),
                     eri.ctypes.data_as(ctypes.c_void_p),
                     dm.ctypes.data_as(ctypes.c_void_p),
                     vk.ctypes.data_as(ctypes.c_void_p))
    vk = pyscf.lib.hermi_triu(vk, hermi)
    return vj, vk



################################################################
# relativistic
def rdirect_mapdm(intor, intsymm, jkdescript,
                  dms, ncomp, atm, bas, env, vhfopt=None):
    assert(intsymm in ('s8', 's4', 's2ij', 's2kl', 's1',
                       'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        n_dm = 1
        nao = dms.shape[0]
        dms = (dms,)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
    if isinstance(jkdescript, str):
        njk = 1
        jkdescript = (jkdescript,)
    else:
        njk = len(jkdescript)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    dotsym = _INTSYMAP[intsymm]
    fdot = _fpointer('CVHFdot_r'+dotsym)

    unpackas = _INTUNPACKMAP_R[intsymm]
    if isinstance(jkdescript, str):
        descr_sym = [[jkdescript.split('->')]]
    else:
        descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dm1 = (ctypes.c_void_p*(njk*n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
# Unlike nr_direct driver, there are no funpack in r_direct driver. ik, jl
# should not be swapped
        f1 = _fpointer('CVHFr%s_%s_%s'%(unpackas, dmsym, vsym))
        for j in range(n_dm):
            assert(dms[j].flags.c_contiguous)
            dm1[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.empty((njk,n_dm*ncomp,nao,nao), dtype=numpy.complex)

    fdrv(cintor, fdot, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(njk*n_dm), ctypes.c_int(ncomp),
         cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if vhfopt is None:
        libcvhf.CINTdel_optimizer(ctypes.byref(cintopt))

    if n_dm * ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    if njk == 1:
        vjk = vjk.reshape(vjk.shape[1:])
    return vjk

# for density matrices in dms, bind each dm to a jk operator
def rdirect_bindm(intor, intsymm, jkdescript,
                  dms, ncomp, atm, bas, env, vhfopt=None):
    assert(intsymm in ('s8', 's4', 's2ij', 's2kl', 's1',
                       'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        n_dm = 1
        nao = dms.shape[0]
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
    if isinstance(jkdescript, str):
        njk = 1
        jkdescript = (jkdescript,)
    else:
        njk = len(jkdescript)
    assert(njk == n_dm)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    dotsym = _INTSYMAP[intsymm]
    fdot = _fpointer('CVHFdot_r'+dotsym)

    unpackas = _INTUNPACKMAP_R[intsymm]
    if isinstance(jkdescript, str):
        descr_sym = [[jkdescript.split('->')]]
    else:
        descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(n_dm))()
    dm1 = (ctypes.c_void_p*(n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
# Unlike nr_direct driver, there are no funpack in r_direct driver. ik, jl
# should not be swapped
        f1 = _fpointer('CVHFr%s_%s_%s'%(unpackas, dmsym, vsym))
        assert(dms[i].flags.c_contiguous)
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    vjk = numpy.empty((njk,ncomp,nao,nao), dtype=numpy.complex)

    fdrv(cintor, fdot, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(n_dm), ctypes.c_int(ncomp),
         cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if vhfopt is None:
        libcvhf.CINTdel_optimizer(ctypes.byref(cintopt))

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

_INTUNPACKMAP_NR = {
    's8  ': ('CVHFunpack_nrblock2tril'      , 's8'  ),
    's4  ': ('CVHFunpack_nrblock2trilu'     , 's2ij'),
    's2ij': ('CVHFunpack_nrblock2trilu'     , 's1'  ),
    's2kl': ('CVHFunpack_nrblock2rect'      , 's2ij'),
    's1'  : ('CVHFunpack_nrblock2rect'      , 's1'  ),
    'a4ij': ('CVHFunpack_nrblock2trilu_anti', 's2ij'),
    'a2ij': ('CVHFunpack_nrblock2trilu_anti', 's1'  ),
    'a4kl': ('CVHFunpack_nrblock2trilu'     , 's2ij'),
    'a2kl': ('CVHFunpack_nrblock2rect'      , 's2ij'),
}

def _swap_ik_jl(s):
    s = s.replace('i','@').replace('k','i').replace('@','k')
    s = s.replace('j','@').replace('l','j').replace('@','l')
    return s
