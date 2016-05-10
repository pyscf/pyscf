#!/usr/bin/env python

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

        c_atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
        c_bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
        c_env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
        natm = ctypes.c_int(c_atm.shape[0])
        nbas = ctypes.c_int(c_bas.shape[0])
        self._cintopt = _vhf.make_cintopt(c_atm, c_bas, c_env, intor)

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


# if out is not None, transform AO to MO in-place
def nr_e1fill_(intor, sh_range, atm, bas, env,
               aosym='s1', comp=1, ao2mopt=None, out=None):
    assert(aosym in ('s4', 's2ij', 's2kl', 's1'))

    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    klsh0, klsh1, nkl = sh_range

    if '_cart' in intor:
        libao2mo.CINTtot_cgto_cart.restype = ctypes.c_int
        nao = libao2mo.CINTtot_cgto_cart(c_bas.ctypes.data_as(ctypes.c_void_p), nbas)
        cgto_in_shell = _fpointer('CINTcgto_cart')
    elif '_sph' in intor:
        libao2mo.CINTtot_cgto_spheric.restype = ctypes.c_int
        nao = libao2mo.CINTtot_cgto_spheric(c_bas.ctypes.data_as(ctypes.c_void_p), nbas)
        cgto_in_shell = _fpointer('CINTcgto_spheric')
    else:
        raise NotImplementedError('cint2e spinor AO integrals')

    if aosym in ('s4', 's2ij'):
        nao_pair = nao * (nao+1) // 2
    else:
        nao_pair = nao * nao
    if out is None:
        out = numpy.empty((comp,nkl,nao_pair))
    else:
        out = numpy.ndarray((comp,nkl,nao_pair), buffer=out)
    if out.size == 0:
        return out

    if ao2mopt is not None:
        cao2mopt = ao2mopt._this
        cintopt = ao2mopt._cintopt
        cintor = ao2mopt._intor
    else:
        cao2mopt = pyscf.lib.c_null_ptr()
        cintor = _fpointer(intor)
        cintopt = _vhf.make_cintopt(c_atm, c_bas, c_env, intor)

    fdrv = getattr(libao2mo, 'AO2MOnr_e1fill_drv')
    fill = _fpointer('AO2MOfill_nr_' + aosym)
    fdrv(cintor, cgto_in_shell, fill,
         out.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(klsh0), ctypes.c_int(klsh1-klsh0),
         ctypes.c_int(nkl), ctypes.c_int(comp),
         cintopt, cao2mopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if ao2mopt is None:
        libao2mo.CINTdel_optimizer(ctypes.byref(cintopt))
    return out

def nr_e1_(eri, mo_coeff, orbs_slice, aosym='s1', mosym='s1', out=None):
    assert(eri.flags.c_contiguous)
    assert(aosym in ('s4', 's2ij', 's2kl', 's1'))
    assert(mosym in ('s2', 's1'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    nao = mo_coeff.shape[0]
    i0, i1, j0, j1 = orbs_slice
    icount = i1 - i0
    jcount = j1 - j0
    ij_count = icount * jcount

    if aosym in ('s4', 's2ij'):
        if mosym == 's2':
            fmmm = _fpointer('AO2MOmmm_nr_s2_s2')
            assert(icount == jcount)
            ij_count = icount * (icount+1) // 2
        elif icount <= jcount:
            fmmm = _fpointer('AO2MOmmm_nr_s2_iltj')
        else:
            fmmm = _fpointer('AO2MOmmm_nr_s2_igtj')
    else:
        if icount <= jcount:
            fmmm = _fpointer('AO2MOmmm_nr_s1_iltj')
        else:
            fmmm = _fpointer('AO2MOmmm_nr_s1_igtj')

    nrow = eri.shape[0]

    if out is None:
        out = numpy.empty((nrow,ij_count))
    else:
        out = numpy.ndarray((nrow,ij_count), buffer=out)
    if out.size == 0:
        return out

    fdrv = getattr(libao2mo, 'AO2MOnr_e2_drv')
    pao_loc = ctypes.POINTER(ctypes.c_void_p)()
    c_nbas = ctypes.c_int(0)
    ftrans = _fpointer('AO2MOtranse1_nr_' + aosym)
    fdrv(ftrans, fmmm,
         out.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nrow), ctypes.c_int(nao),
         ctypes.c_int(i0), ctypes.c_int(icount),
         ctypes.c_int(j0), ctypes.c_int(jcount),
         pao_loc, c_nbas)
    return out

# if out is not None, transform AO to MO in-place
# ao_loc has nbas+1 elements, last element in ao_loc == nao
def nr_e2_(eri, mo_coeff, orbs_slice, aosym='s1', mosym='s1', out=None,
           ao_loc=None):
    assert(eri.flags.c_contiguous)
    assert(aosym in ('s4', 's2ij', 's2kl', 's2', 's1'))
    assert(mosym in ('s2', 's1'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    nao = mo_coeff.shape[0]
    k0, k1, l0, l1 = orbs_slice
    kc = k1 - k0
    lc = l1 - l0
    kl_count = kc * lc

    if aosym in ('s4', 's2', 's2kl'):
        if mosym == 's2':
            fmmm = _fpointer('AO2MOmmm_nr_s2_s2')
            assert(kc == lc)
            kl_count = kc * (kc+1) // 2
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

    if out is None:
        out = numpy.empty((nrow,kl_count))
    else:
        out = numpy.ndarray((nrow,kl_count), buffer=out)
    if out.size == 0:
        return out

    if ao_loc is None:
        pao_loc = ctypes.POINTER(ctypes.c_void_p)()
        c_nbas = ctypes.c_int(0)
        ftrans = _fpointer('AO2MOtranse2_nr_' + aosym)
    else:
        ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32)
        c_nbas = ctypes.c_int(ao_loc.shape[0]-1)
        pao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)
        ftrans = _fpointer('AO2MOsortranse2_nr_' + aosym)

    fdrv = getattr(libao2mo, 'AO2MOnr_e2_drv')
    fdrv(ftrans, fmmm,
         out.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nrow), ctypes.c_int(nao),
         ctypes.c_int(k0), ctypes.c_int(kc),
         ctypes.c_int(l0), ctypes.c_int(lc),
         pao_loc, c_nbas)
    return out


# if out is not None, transform AO to MO in-place
def r_e1_(intor, mo_coeff, orbs_slice, sh_range, atm, bas, env,
          tao, aosym='s1', comp=1, ao2mopt=None, out=None):
    assert(aosym in ('s4', 's2ij', 's2kl', 's1', 'a2ij', 'a2kl', 'a4ij',
                     'a4kl', 'a4'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    i0, i1, j0, j1 = orbs_slice
    icount = i1 - i0
    jcount = j1 - j0
    ij_count = icount * jcount

    c_atm = numpy.asarray(atm, dtype=numpy.int32)
    c_bas = numpy.asarray(bas, dtype=numpy.int32)
    c_env = numpy.asarray(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    klsh0, klsh1, nkl = sh_range

    if icount <= jcount:
        fmmm = _fpointer('AO2MOmmm_r_iltj')
    else:
        fmmm = _fpointer('AO2MOmmm_r_igtj')

    if out is None:
        out = numpy.empty((comp,nkl,ij_count), dtype=numpy.complex)
    else:
        out = numpy.ndarray((comp,nkl,nao_pair), dtype=numpy.complex,
                            buffer=out)
    if out.size == 0:
        return out

    if ao2mopt is not None:
        cao2mopt = ao2mopt._this
        cintopt = ao2mopt._cintopt
        cintor = ao2mopt._intor
    else:
        cao2mopt = pyscf.lib.c_null_ptr()
        cintor = _fpointer(intor)
        cintopt = _vhf.make_cintopt(c_atm, c_bas, c_env, intor)

    tao = numpy.asarray(tao, dtype=numpy.int32)

    fdrv = getattr(libao2mo, 'AO2MOr_e1_drv')
    fill = _fpointer('AO2MOfill_r_' + aosym)
    ftrans = _fpointer('AO2MOtranse1_r_' + aosym)
    fdrv(cintor, fill, ftrans, fmmm,
         out.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(klsh0), ctypes.c_int(klsh1-klsh0),
         ctypes.c_int(nkl),
         ctypes.c_int(i0), ctypes.c_int(icount),
         ctypes.c_int(j0), ctypes.c_int(jcount),
         ctypes.c_int(comp), cintopt, cao2mopt,
         tao.ctypes.data_as(ctypes.c_void_p),
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if ao2mopt is None:
        libao2mo.CINTdel_optimizer(ctypes.byref(cintopt))
    return out

# if out is not None, transform AO to MO in-place
# ao_loc has nbas+1 elements, last element in ao_loc == nao
def r_e2_(eri, mo_coeff, orbs_slice, tao, ao_loc, aosym='s1', out=None):
    assert(eri.flags.c_contiguous)
    assert(aosym in ('s4', 's2ij', 's2kl', 's1', 'a2ij', 'a2kl', 'a4ij',
                     'a4kl', 'a4'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    nao = mo_coeff.shape[0]
    k0, k1, l0, l1 = orbs_slice
    kc = k1 - k0
    lc = l1 - l0
    kl_count = kc * lc

    if kc <= lc:
        fmmm = _fpointer('AO2MOmmm_r_iltj')
    else:
        fmmm = _fpointer('AO2MOmmm_r_igtj')

    nrow = eri.shape[0]

    if out is None:
        out = numpy.empty((nrow,kl_count), dtype=numpy.complex)
    else:
        out = numpy.ndarray((nrow,kl_count), dtype=numpy.complex,
                            buffer=out)
    if out.size == 0:
        return out

    tao = numpy.asarray(tao, dtype=numpy.int32)
    ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32)
    c_nbas = ctypes.c_int(ao_loc.shape[0]-1)
    ftrans = _fpointer('AO2MOsortranse2_r_' + aosym)

    fdrv = getattr(libao2mo, 'AO2MOr_e2_drv')
    fdrv(ftrans, fmmm,
         out.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nrow), ctypes.c_int(nao),
         ctypes.c_int(k0), ctypes.c_int(kc),
         ctypes.c_int(l0), ctypes.c_int(lc),
         tao.ctypes.data_as(ctypes.c_void_p),
         ao_loc.ctypes.data_as(ctypes.c_void_p), c_nbas)
    return out


def _get_num_threads():
    libao2mo.omp_get_num_threads.restype = ctypes.c_int
    nthreads = libao2mo.omp_get_num_threads()
    return nthreads

# ij = i * (i+1) / 2 + j
def _extract_pair(ij):
    i = int(numpy.sqrt(2*ij+.25) - .5 + 1e-7)
    j = ij - i*(i+1)//2
    return i,j

