#!/usr/bin/env python

import sys
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
        fvj = _fpointer('CVHFics4_kl_s2ij')
        # 'ijkl,il->jk'
        fvk = _fpointer('CVHFics4_il_s1jk')
        # or
        ## 'ijkl,ij->kl'
        #fvj = _fpointer('CVHFics4_ij_s2kl')
        ## 'ijkl,jk->il'
        #fvk = _fpointer('CVHFics4_jk_s1il')

        tridm = dm
    elif eri.ndim == 1 and npair*(npair+1)//2 == eri.size: # 8-fold symmetry eri
        fdrv = getattr(libcvhf, 'CVHFnrs8_incore_drv')
        fvj = _fpointer('CVHFics8_tridm_vj')
        if hermi == 1:
            fvk = _fpointer('CVHFics8_jk_s2il')
        else:
            fvk = _fpointer('CVHFics8_jk_s1il')
        tridm = pyscf.lib.pack_tril(pyscf.lib.transpose_sum(dm))
        for i in range(nao):
            tridm[i*(i+1)//2+i] *= .5
    else:
        raise RuntimeError('Array shape not consistent: DM %s, eri %s'
                           % (dm.shape, eri.shape))
    fdrv(eri.ctypes.data_as(ctypes.c_void_p),
         tridm.ctypes.data_as(ctypes.c_void_p),
         vj.ctypes.data_as(ctypes.c_void_p),
         dm.ctypes.data_as(ctypes.c_void_p),
         vk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nao), fvj, fvk)
    if hermi != 0:
        vj = pyscf.lib.hermi_triu_(vj, hermi)
        vk = pyscf.lib.hermi_triu_(vk, hermi)
    else:
        vj = pyscf.lib.hermi_triu_(vj, 1)
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
        dms = (numpy.asarray(dms, order='C'),)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
        dms = numpy.asarray(dms, order='C')

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
    fdot = _fpointer('CVHFdot_nrs8')
    fvj = _fpointer('CVHFnrs8_ji_s2kl')
    if hermi == 1:
        fvk = _fpointer('CVHFnrs8_li_s2kj')
    else:
        fvk = _fpointer('CVHFnrs8_li_s1kj')
    fjk = (ctypes.c_void_p*(2*n_dm))()
    dm1 = (ctypes.c_void_p*(2*n_dm))()
    for i in range(n_dm):
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = fvj
    for i in range(n_dm):
        assert(dms[i].flags.c_contiguous)
        dm1[n_dm+i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[n_dm+i] = fvk
    vjk = numpy.empty((2,n_dm,nao,nao))

    fdrv(cintor, fdot, fjk, dm1,
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
        vjk[0,idm] = pyscf.lib.hermi_triu_(vjk[0,idm], 1)
    if hermi != 0: # vk depends
        for idm in range(n_dm):
            vjk[1,idm] = pyscf.lib.hermi_triu_(vjk[1,idm], hermi)
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
        dms = (numpy.asarray(dms, order='C'),)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
        dms = numpy.asarray(dms, order='C')
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
    fdot = _fpointer('CVHFdot_nr'+dotsym)

    descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dm1 = (ctypes.c_void_p*(njk*n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        if dmsym in ('ij', 'kl', 'il', 'kj'):
            sys.stderr.write('not support DM description %s, transpose to %s\n' %
                             (dmsym, dmsym[::-1]))
            f1 = _fpointer('CVHFnr%s_%s_%s'%(intsymm, dmsym[::-1], vsym))
        else:
            f1 = _fpointer('CVHFnr%s_%s_%s'%(intsymm, dmsym, vsym))
        for j in range(n_dm):
            assert(dms[j].flags.c_contiguous)
            dm1[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.empty((njk,n_dm*ncomp,nao,nao))

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
        dms = (numpy.asarray(dms, order='C'),)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
        dms = numpy.asarray(dms, order='C')
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
    fdot = _fpointer('CVHFdot_nr'+dotsym)

    descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(n_dm))()
    dm1 = (ctypes.c_void_p*(n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        if dmsym in ('ij', 'kl', 'il', 'kj'):
            f1 = _fpointer('CVHFnr%s_%s_%s'%(intsymm, dmsym[::-1], vsym))
        else:
            f1 = _fpointer('CVHFnr%s_%s_%s'%(intsymm, dmsym, vsym))
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    vjk = numpy.empty((njk,ncomp,nao,nao))

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
    libcvhf.int2e_sph(eri.ctypes.data_as(ctypes.c_void_p),
                      c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                      c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                      c_env.ctypes.data_as(ctypes.c_void_p))
    return eri


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
        dms = (numpy.asarray(dms, order='C'),)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
        dms = numpy.asarray(dms, order='C')
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
        dms = (numpy.asarray(dms, order='C'),)
    else:
        n_dm = len(dms)
        nao = dms[0].shape[0]
        dms = numpy.asarray(dms, order='C')
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

