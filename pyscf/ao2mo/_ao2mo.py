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

import ctypes
import _ctypes
import numpy
from pyscf import lib
from pyscf.gto.moleintor import make_cintopt, make_loc, ascint3
from pyscf.scf import _vhf

libao2mo = lib.load_library('libao2mo')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle, name))

class AO2MOpt(object):
    def __init__(self, mol, intor, prescreen='CVHFnoscreen', qcondname=None):
        intor = ascint3(intor)
        self._this = ctypes.POINTER(_vhf._CVHFOpt)()
        #print self._this.contents, expect ValueError: NULL pointer access
        self._intor = intor

        c_atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
        c_bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
        c_env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
        natm = ctypes.c_int(c_atm.shape[0])
        nbas = ctypes.c_int(c_bas.shape[0])
        self._cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

        libao2mo.CVHFinit_optimizer(ctypes.byref(self._this),
                                    c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                                    c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                    c_env.ctypes.data_as(ctypes.c_void_p))
        self._this.contents.fprescreen = _fpointer(prescreen)

        if prescreen != 'CVHFnoscreen' and intor in ('int2e_sph', 'int2e_cart'):
            # for int2e_sph, qcondname is 'CVHFsetnr_direct_scf'
            ao_loc = make_loc(c_bas, intor)
            fsetqcond = getattr(libao2mo, qcondname)
            fsetqcond(self._this,
                      getattr(libao2mo, intor), self._cintopt,
                      ao_loc.ctypes.data_as(ctypes.c_void_p),
                      c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                      c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                      c_env.ctypes.data_as(ctypes.c_void_p))

    def __del__(self):
        libao2mo.CVHFdel_optimizer(ctypes.byref(self._this))


# if out is not None, transform AO to MO in-place
def nr_e1fill(intor, sh_range, atm, bas, env,
              aosym='s1', comp=1, ao2mopt=None, out=None):
    assert(aosym in ('s4', 's2ij', 's2kl', 's1'))
    intor = ascint3(intor)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])
    ao_loc = make_loc(bas, intor)
    nao = ao_loc[-1]

    klsh0, klsh1, nkl = sh_range

    if aosym in ('s4', 's2ij'):
        nao_pair = nao * (nao+1) // 2
    else:
        nao_pair = nao * nao
    out = numpy.ndarray((comp,nkl,nao_pair), buffer=out)
    if out.size == 0:
        return out

    if ao2mopt is not None:
        cao2mopt = ao2mopt._this
        cintopt = ao2mopt._cintopt
        intor = ao2mopt._intor
    else:
        cao2mopt = lib.c_null_ptr()
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
    cintor = _fpointer(intor)

    fdrv = getattr(libao2mo, 'AO2MOnr_e1fill_drv')
    fill = _fpointer('AO2MOfill_nr_' + aosym)
    fdrv(cintor, fill, out.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(klsh0), ctypes.c_int(klsh1-klsh0),
         ctypes.c_int(nkl), ctypes.c_int(comp),
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cao2mopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))
    return out

def nr_e1(eri, mo_coeff, orbs_slice, aosym='s1', mosym='s1', out=None):
    assert(eri.flags.c_contiguous)
    assert(aosym in ('s4', 's2ij', 's2kl', 's1'))
    assert(mosym in ('s2', 's1'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    assert(mo_coeff.dtype == numpy.double)
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
         (ctypes.c_int*4)(*orbs_slice), pao_loc, c_nbas)
    return out

# if out is not None, transform AO to MO in-place
# ao_loc has nbas+1 elements, last element in ao_loc == nao
def nr_e2(eri, mo_coeff, orbs_slice, aosym='s1', mosym='s1', out=None,
           ao_loc=None):
    assert(eri.flags.c_contiguous)
    assert(aosym in ('s4', 's2ij', 's2kl', 's2', 's1'))
    assert(mosym in ('s2', 's1'))
    mo_coeff = numpy.asfortranarray(mo_coeff)
    assert(mo_coeff.dtype == numpy.double)
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
         (ctypes.c_int*4)(*orbs_slice), pao_loc, c_nbas)
    return out


# if out is not None, transform AO to MO in-place
def r_e1(intor, mo_coeff, orbs_slice, sh_range, atm, bas, env,
         tao, aosym='s1', comp=1, ao2mopt=None, out=None):
    assert(aosym in ('s4', 's2ij', 's2kl', 's1', 'a2ij', 'a2kl', 'a4ij',
                     'a4kl', 'a4'))
    intor = ascint3(intor)
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

    out = numpy.ndarray((comp,nkl,ij_count), dtype=numpy.complex, buffer=out)
    if out.size == 0:
        return out

    if ao2mopt is not None:
        cao2mopt = ao2mopt._this
        cintopt = ao2mopt._cintopt
        intor = ao2mopt._intor
    else:
        cao2mopt = lib.c_null_ptr()
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
    cintor = _fpointer(intor)

    tao = numpy.asarray(tao, dtype=numpy.int32)
    ao_loc = make_loc(bas, 'spinor')

    fdrv = getattr(libao2mo, 'AO2MOr_e1_drv')
    fill = _fpointer('AO2MOfill_r_' + aosym)
    ftrans = _fpointer('AO2MOtranse1_r_' + aosym)
    fdrv(cintor, fill, ftrans, fmmm,
         out.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(klsh0), ctypes.c_int(klsh1-klsh0),
         ctypes.c_int(nkl), ctypes.c_int(comp),
         (ctypes.c_int*4)(*orbs_slice), tao.ctypes.data_as(ctypes.c_void_p),
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cao2mopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))
    return out

# if out is not None, transform AO to MO in-place
# ao_loc has nbas+1 elements, last element in ao_loc == nao
def r_e2(eri, mo_coeff, orbs_slice, tao, ao_loc, aosym='s1', out=None):
    assert(eri.flags.c_contiguous)
    assert(aosym in ('s4', 's2ij', 's2kl', 's1', 'a2ij', 'a2kl', 'a4ij',
                     'a4kl', 'a4'))
    mo_coeff = numpy.asarray(mo_coeff, dtype=numpy.complex128, order='F')
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
    out = numpy.ndarray((nrow,kl_count), dtype=numpy.complex128, buffer=out)
    if out.size == 0:
        return out

    tao = numpy.asarray(tao, dtype=numpy.int32)
    if ao_loc is None:
        c_ao_loc = ctypes.POINTER(ctypes.c_void_p)()
        c_nbas = ctypes.c_int(0)
        ftrans = _fpointer('AO2MOtranse2_r_' + aosym)
    else:
        ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32)
        c_ao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)
        c_nbas = ctypes.c_int(ao_loc.shape[0]-1)
        ftrans = _fpointer('AO2MOsortranse2_r_' + aosym)

    fdrv = getattr(libao2mo, 'AO2MOr_e2_drv')
    fdrv(ftrans, fmmm,
         out.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nrow), ctypes.c_int(nao),
         (ctypes.c_int*4)(*orbs_slice),
         tao.ctypes.data_as(ctypes.c_void_p), c_ao_loc, c_nbas)
    return out

