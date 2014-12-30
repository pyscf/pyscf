#!/usr/bin/env python

import os
import ctypes
import _ctypes
import numpy
import pyscf.lib
from pyscf.scf import _vhf

libao2mo = pyscf.lib.load_library('libao2mo')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle, name))

class AO2MOpt(object):
    def __init__(self, mol, intor,
                 prescreen='CVHFnoscreen', qcondname=None):
        self._this = ctypes.POINTER(_vhf._CVHFOpt)()
        #print self._this.contents, expect ValueError: NULL pointer access
        self._intor = _fpointer(intor)
        self._cintopt = _vhf.make_cintopt(mol._atm, mol._bas, mol._env, intor)

        c_atm = numpy.array(mol._atm, dtype=numpy.int32)
        c_bas = numpy.array(mol._bas, dtype=numpy.int32)
        c_env = numpy.array(mol._env)
        natm = ctypes.c_int(c_atm.shape[0])
        nbas = ctypes.c_int(c_bas.shape[0])

        libao2mo.CVHFinit_optimizer(ctypes.byref(self._this),
                                    c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                                    c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                    c_env.ctypes.data_as(ctypes.c_void_p))
        self._this.contents.fprescreen = _fpointer(prescreen)

        if prescreen != 'CVHFnoscreen':
            # for cint2e_sph, qcondname is 'CVHFsetnr_direct_scf'
            fsetqcond = getattr(libao2mo, qcondname)
            fsetqcond(self._this,
                      c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                      c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                      c_env.ctypes.data_as(ctypes.c_void_p))

    def __del__(self):
        libao2mo.CINTdel_optimizer(ctypes.byref(self._cintopt))
        libao2mo.CVHFdel_optimizer(ctypes.byref(self._this))

def _count_ij(icount, jcount, symm=0):
    if symm:
        assert(icount == jcount)
        return icount * (icount+1) / 2
    else:
        return icount*jcount

def _get_num_threads():
    libao2mo.omp_get_num_threads.restype = ctypes.c_int
    nthreads = libao2mo.omp_get_num_threads()
    return nthreads

# if vout is not None, transform AO to MO in-place
def nr_e1_(intor, mo_coeff, shape, sh_range, atm, bas, env,
           aosym='s1', mosym='s1', comp=1, ao2mopt=None, vout=None):
    assert(aosym in ('s4', 's2ij', 's2kl', 's1'))
    assert(mosym in ('s2', 's1'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    nao = mo_coeff.shape[0]
    i0, ic, j0, jc = shape
    ij_count = ic * jc

    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    ksh0, ksh1 = sh_range
    ANG_OF = 1
    NCTR_OF = 3
    fdim = lambda i: (c_bas[i,ANG_OF]*2+1)*c_bas[i,NCTR_OF]
    kstart = sum([fdim(ksh) for ksh in range(ksh0)])
    kend = kstart + sum([fdim(ksh) for ksh in range(ksh0, ksh1)])

    if aosym in ('s4', 's2kl'):
        klpair = kend*(kend+1)//2 - kstart*(kstart+1)//2
    else:
        klpair = (kend-kstart) * nao

    if aosym in ('s4', 's2ij'):
        if mosym == 's2':
            fmmm = _fpointer('AO2MOmmm_nr_s2_s2')
            assert(ic == jc)
            ij_count = ic * (ic+1) / 2
        elif ic <= jc:
            fmmm = _fpointer('AO2MOmmm_nr_s2_iltj')
        else:
            fmmm = _fpointer('AO2MOmmm_nr_s2_igtj')
    else:
        if ic <= jc:
            fmmm = _fpointer('AO2MOmmm_nr_s1_iltj')
        else:
            fmmm = _fpointer('AO2MOmmm_nr_s1_igtj')

    if vout is None:
        vout = numpy.empty((comp,klpair,ij_count))
    else:
        assert(vout.flags.c_contiguous)
        assert(vout.size > comp*klpair*ij_count)

    if ao2mopt is not None:
        cao2mopt = ao2mopt._this
        cintopt = ao2mopt._cintopt
        cintor = ao2mopt._intor
    else:
        cao2mopt = ctypes.c_void_p()
        cintor = _fpointer(intor)
        cintopt = _vhf.make_cintopt(atm, bas, env, intor)

    fdrv = getattr(libao2mo, 'AO2MOnr_e1_drv')
    ftrans = _fpointer('AO2MOtranse1_nr_' + aosym)
    fdrv(cintor, ftrans, fmmm,
         vout.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(ksh0), ctypes.c_int(ksh1-ksh0),
         ctypes.c_int(i0), ctypes.c_int(ic),
         ctypes.c_int(j0), ctypes.c_int(jc),
         ctypes.c_int(comp), cintopt, cao2mopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if ao2mopt is None:
        libao2mo.CINTdel_optimizer(ctypes.byref(cintopt))
    return vout

# if vout is not None, transform AO to MO in-place
def nr_e2_(eri, mo_coeff, shape, aosym='s1', mosym='s1', vout=None):
    assert(eri.flags.c_contiguous)
    assert(aosym in ('s4', 's2ij', 's2kl', 's1'))
    assert(mosym in ('s2', 's1'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    nao = mo_coeff.shape[0]
    k0, kc, l0, lc = shape
    kl_count = kc * lc

    if aosym == 's4' or aosym == 's2kl':
        if mosym == 's2':
            fmmm = _fpointer('AO2MOmmm_nr_s2_s2')
            assert(kc == lc)
            kl_count = kc * (kc+1) / 2
        elif kc <= lc:
            fmmm = _fpointer('AO2MOmmm_nr_s2_iltj')
        else:
            fmmm = _fpointer('AO2MOmmm_nr_s2_igtj')
    else:
        if kc <= lc:
            fmmm = _fpointer('AO2MOmmm_nr_s1_iltj')
        else:
            fmmm = _fpointer('AO2MOmmm_nr_s1_igtj')

    nrow = eri.shape[0]

    if vout is None:
        if aosym == 's4' or aosym == 's2kl':
            klnaopair = nao*(nao+1)//2
        else:
            klnaopair = nao**2
        if 0 and kl_count == klnaopair:
# we can reuse memory, but switch it off because it's so easy to cause mistake
            vout = eri
        else:
# memory cannot be reused unless nij == nao_pair, even nij < nao_pair.
# When OMP is used, data is not accessed sequentially, the transformed
# integrals can overwrite the AO integrals which are not used.
            vout = numpy.empty((nrow,kl_count))
    else:
        assert(vout.flags.c_contiguous)

    fdrv = getattr(libao2mo, 'AO2MOnr_e2_drv')
    ftrans = _fpointer('AO2MOtranse2_nr_' + aosym)
    fdrv(ftrans, fmmm,
         vout.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nrow), ctypes.c_int(nao),
         ctypes.c_int(k0), ctypes.c_int(kc),
         ctypes.c_int(l0), ctypes.c_int(lc))
    return vout

