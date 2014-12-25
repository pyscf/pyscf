#!/usr/bin/env python

import os
import ctypes
import _ctypes
import numpy
import pyscf.lib

libcvhf = pyscf.lib.load_library('libcvhf')

class VHFOpt(object):
    def __init__(self, mol, intor, prescreen, qcondname, dmcondname):
        self._this = ctypes.POINTER(_CVHFOpt)()
        #print self._this.contents, expect ValueError: NULL pointer access
        self._intor = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, intor))
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

        foptinit = getattr(libcvhf, intor+'_optimizer')
        foptinit(ctypes.byref(self._cintopt),
                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                 c_env.ctypes.data_as(ctypes.c_void_p))

        libcvhf.CVHFinit_optimizer(ctypes.byref(self._this),
                                   c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                                   c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                   c_env.ctypes.data_as(ctypes.c_void_p))
        fsetqcond = getattr(libcvhf, qcondname)
        fsetqcond(self._this,
                  c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                  c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                  c_env.ctypes.data_as(ctypes.c_void_p))
        self._this.contents.fprescreen = \
                ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, prescreen))

    @property
    def direct_scf_threshold(self):
        return self._this.contents.direct_scf_cutoff
    @direct_scf_threshold.setter
    def direct_scf_threshold(self, v):
        self._this.contents.direct_scf_cutoff = v

    def set_dm_(self, dm, atm, bas, env):
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
        fvj = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs4_kl_s2ij'))
        # 'ijkl,il->jk'
        fvk = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs4_il_s1jk'))
        # or
        ## 'ijkl,ij->kl'
        #fvj = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs4_ij_s2kl'))
        ## 'ijkl,jk->il'
        #fvk = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs4_jk_s1il'))

        tridm = dm
    else: # 8-fold symmetry eri
        fdrv = getattr(libcvhf, 'CVHFnrs8_incore_drv')
        fvj = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs8_tridm_vj'))
        fvk = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs8_jk_s2il'))
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

# use cint2e_sph as intor, CVHFnrs8_ij_s2kl, CVHFnrs8_jk_s2il as fjk to call
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
        intor = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'cint2e_sph'))
        cintopt = ctypes.c_void_p()
        foptinit = getattr(libcvhf, 'cint2e_sph_optimizer')
        foptinit(ctypes.byref(cintopt),
                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                 c_env.ctypes.data_as(ctypes.c_void_p))
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    funpack = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle,
                                            'CVHFunpack_nrblock2tril'))
    fdot = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFfill_dot_nrs8'))
    fvj = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs8_tridm_vj'))
    fvk = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, 'CVHFnrs8_jk_s2il'))
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

    fdrv(intor, fdot, funpack, fjk, dm1,
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

# call all fjk for each dm, the return array has len(dms)*len(namefjk)*ncomp components
def direct_mapdm(cintor, cfdot, cunpack, namefjk,
                 dms, ncomp, atm, bas, env, vhfopt=None):
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
    if isinstance(namefjk, str):
        njk = 1
        namefjk = (namefjk,)
    else:
        njk = len(namefjk)

    if vhfopt is None:
        intor = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cintor))
        cintopt = ctypes.c_void_p()
        foptinit = getattr(libcvhf, cintor+'_optimizer')
        foptinit(ctypes.byref(cintopt),
                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                 c_env.ctypes.data_as(ctypes.c_void_p))
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    fdot = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cfdot))
    funpack = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cunpack))
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dm1 = (ctypes.c_void_p*(njk*n_dm))()
    for i, symb in enumerate(namefjk):
        f1 = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, symb))
        for j in range(n_dm):
            assert(dms[j].flags.c_contiguous)
            dm1[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.empty((njk,n_dm*ncomp,nao,nao))

    fdrv(intor, fdot, funpack, fjk, dm1,
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
def direct_bindm(cintor, cfdot, cunpack, namefjk,
                 dms, ncomp, atm, bas, env, vhfopt=None):
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
    if isinstance(namefjk, str):
        njk = 1
        namefjk = (namefjk,)
    else:
        njk = len(namefjk)
    assert(njk == n_dm)

    if vhfopt is None:
        intor = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cintor))
        cintopt = ctypes.c_void_p()
        foptinit = getattr(libcvhf, cintor+'_optimizer')
        foptinit(ctypes.byref(cintopt),
                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                 c_env.ctypes.data_as(ctypes.c_void_p))
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFnr_direct_drv')
    fdot = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cfdot))
    funpack = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cunpack))
    fjk = (ctypes.c_void_p*(n_dm))()
    dm1 = (ctypes.c_void_p*(n_dm))()
    for i, symb in enumerate(namefjk):
        f1 = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, symb))
        assert(dms[i].flags.c_contiguous)
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    vjk = numpy.empty((njk,ncomp,nao,nao))

    fdrv(intor, funpack, fdot, fjk, dm1,
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
def rdirect_mapdm(cintor, cfdot, namefjk,
                  dms, ncomp, atm, bas, env, vhfopt=None):
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
    if isinstance(namefjk, str):
        njk = 1
        namefjk = (namefjk,)
    else:
        njk = len(namefjk)

    if vhfopt is None:
        intor = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cintor))
        cintopt = ctypes.c_void_p()
        foptinit = getattr(libcvhf, cintor+'_optimizer')
        foptinit(ctypes.byref(cintopt),
                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                 c_env.ctypes.data_as(ctypes.c_void_p))
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    fdot = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cfdot))
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dm1 = (ctypes.c_void_p*(njk*n_dm))()
    for i, symb in enumerate(namefjk):
        f1 = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, symb))
        for j in range(n_dm):
            assert(dms[j].flags.c_contiguous)
            dm1[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.empty((njk,n_dm*ncomp,nao,nao), dtype=numpy.complex)

    fdrv(intor, fdot, fjk, dm1,
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
def rdirect_bindm(cintor, cfdot, namefjk,
                  dms, ncomp, atm, bas, env, vhfopt=None):
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
    if isinstance(namefjk, str):
        njk = 1
        namefjk = (namefjk,)
    else:
        njk = len(namefjk)
    assert(njk == n_dm)

    if vhfopt is None:
        intor = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cintor))
        cintopt = ctypes.c_void_p()
        foptinit = getattr(libcvhf, cintor+'_optimizer')
        foptinit(ctypes.byref(cintopt),
                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                 c_env.ctypes.data_as(ctypes.c_void_p))
        cvhfopt = ctypes.c_void_p()
    else:
        vhfopt.set_dm_(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    fdot = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, cfdot))
    fjk = (ctypes.c_void_p*(n_dm))()
    dm1 = (ctypes.c_void_p*(n_dm))()
    for i, symb in enumerate(namefjk):
        f1 = ctypes.c_void_p(_ctypes.dlsym(libcvhf._handle, symb))
        assert(dms[i].flags.c_contiguous)
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    vjk = numpy.empty((njk,ncomp,nao,nao), dtype=numpy.complex)

    fdrv(intor, fdot, fjk, dm1,
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

