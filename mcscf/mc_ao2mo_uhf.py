#!/usr/bin/env python

import os, sys
import ctypes
import tempfile
import numpy
import h5py
import pyscf.lib
import pyscf.lib.numpy_helper
import pyscf.ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.mcscf import mc_ao2mo

libmcscf = pyscf.lib.load_library('libmcscf')

# least memory requirements:
#       ncore**2*(nmo-ncore)**2 + ncas**2*nmo**2*2 + nmo**3   words
# nmo  ncore  ncas  outcore  incore
# 200  40     16    2.4GB    5.3 GB (_eri 1.6GB )
# 250  50     16    4.9GB   12.0 GB (_eri 3.9GB )
# 300  60     16    9.0GB   23.7 GB (_eri 8.1GB )
# 400  80     16   24.6GB
# 500  100    16   54.8GB
# 600  120    16   107 GB


def trans_e1_incore(eri_ao, mo, ncore, ncas):
    nmo = mo[0].shape[1]
    nocc = (ncore[0] + ncas, ncore[1] + ncas)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril
    c_nmo = ctypes.c_int(nmo)

    erib = pyscf.ao2mo.incore.half_e1(eri_ao, (mo[1][:,:nocc[1]],mo[1]),
                                      compact=False)
    load_buf = lambda bufid: erib[bufid*nmo:bufid*nmo+nmo].copy()
    AAPP, AApp, APPA, tmp, IAPCV, APcv = \
            _trans_aapp_((mo[1],mo[0]), (ncore[1],ncore[0]), ncas, load_buf)
    jC_PP, jC_pp, kC_PP, ICVCV = \
            _trans_cvcv_((mo[1],mo[0]), (ncore[1],ncore[0]), ncas, load_buf)[:4]
    erib = tmp = None

    eria = pyscf.ao2mo.incore.half_e1(eri_ao, (mo[0][:,:nocc[0]],mo[0]),
                                      compact=False)
    load_buf = lambda bufid: eria[bufid*nmo:bufid*nmo+nmo].copy()
    aapp, aaPP, appa, apPA, Iapcv, apCV = \
            _trans_aapp_(mo, ncore, ncas, load_buf)
    jc_pp, jc_PP, kc_pp, Icvcv, cvCV = \
            _trans_cvcv_(mo, ncore, ncas, load_buf)

    jkcpp = jc_pp - kc_pp
    jkcPP = jC_PP - kC_PP
    return jkcpp, jkcPP, jC_pp, jc_PP, \
            aapp, aaPP, AApp, AAPP, appa, apPA, APPA, \
            Iapcv, IAPCV, apCV, APcv, Icvcv, ICVCV, cvCV


def trans_e1_outcore(casscf, mo, max_memory=None, ioblk_size=512, tmpdir=None,
                     verbose=0):
    mol = casscf.mol
    log = pyscf.lib.logger.Logger(casscf.stdout, verbose)
    ncore = casscf.ncore
    ncas = casscf.ncas
    nao, nmo = mo[0].shape
    nao_pair = nao*(nao+1)//2
    nocc = (ncore[0] + ncas, ncore[1] + ncas)

    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    pyscf.ao2mo.outcore.half_e1(mol, (mo[1][:,:nocc[1]],mo[1]), swapfile.name,
                                verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(bfn_id):
        buf = numpy.empty((nmo,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:nmo,col0:col1] = dat[bfn_id*nmo:(bfn_id+1)*nmo]
            col0 = col1
        return buf
    AAPP, AApp, APPA, tmp, IAPCV, APcv = \
            _trans_aapp_((mo[1],mo[0]), (ncore[1],ncore[0]), ncas, load_buf)
    jC_PP, jC_pp, kC_PP, ICVCV = \
            _trans_cvcv_((mo[1],mo[0]), (ncore[1],ncore[0]), ncas, load_buf)[:4]
    tmp = None
    fswap.close()

    ###########################

    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    pyscf.ao2mo.outcore.half_e1(mol, (mo[0][:,:nocc[0]],mo[0]), swapfile.name,
                                verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(bfn_id):
        buf = numpy.empty((nmo,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:nmo,col0:col1] = dat[bfn_id*nmo:(bfn_id+1)*nmo]
            col0 = col1
        return buf
    aapp, aaPP, appa, apPA, Iapcv, apCV = \
            _trans_aapp_(mo, ncore, ncas, load_buf)
    jc_pp, jc_PP, kc_pp, Icvcv, cvCV = \
            _trans_cvcv_(mo, ncore, ncas, load_buf)
    fswap.close()

    jkcpp = jc_pp - kc_pp
    jkcPP = jC_PP - kC_PP
    return jkcpp, jkcPP, jC_pp, jc_PP, \
            aapp, aaPP, AApp, AAPP, appa, apPA, APPA, \
            Iapcv, IAPCV, apCV, APcv, Icvcv, ICVCV, cvCV


def _trans_aapp_(mo, ncore, ncas, fload):
    nmo = mo[0].shape[1]
    nocc = (ncore[0] + ncas, ncore[1] + ncas)
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    klshape = (0, nmo, 0, nmo)

    japcv = numpy.empty((ncas,nmo,ncore[0],nmo-ncore[0]))
    aapp = numpy.empty((ncas,ncas,nmo,nmo))
    aaPP = numpy.empty((ncas,ncas,nmo,nmo))
    appa = numpy.empty((ncas,nmo,nmo,ncas))
    apPA = numpy.empty((ncas,nmo,nmo,ncas))
    apCV = numpy.empty((ncas,nmo,ncore[1],nmo-ncore[1]))
    ppp = numpy.empty((nmo,nmo,nmo))
    for i in range(ncas):
        buf = _ao2mo.nr_e2_(fload(ncore[0]+i), mo[0], klshape,
                            aosym='s4', mosym='s2')
        for j in range(nmo):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    ppp[j].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        aapp[i] = ppp[ncore[0]:nocc[0]]
        appa[i] = ppp[:,:,ncore[0]:nocc[0]]
        #japcp = avcp * 2 - acpv.transpose(0,2,1,3) - avcp.transpose(0,3,2,1)
        japcv[i] = ppp[:,:ncore[0],ncore[0]:] * 2 \
                 - ppp[:ncore[0],:,ncore[0]:].transpose(1,0,2) \
                 - ppp[ncore[0]:,:ncore[0],:].transpose(2,1,0)

        buf = _ao2mo.nr_e2_(fload(ncore[0]+i), mo[1], klshape,
                            aosym='s4', mosym='s2')
        for j in range(nmo):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    ppp[j].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        aaPP[i] = ppp[ncore[0]:nocc[0]]
        apPA[i] = ppp[:,:,ncore[1]:nocc[1]]
        apCV[i] = ppp[:,:ncore[1],ncore[1]:]

    return aapp, aaPP, appa, apPA, japcv, apCV

def _trans_cvcv_(mo, ncore, ncas, fload):
    nmo = mo[0].shape[1]
    nocc = (ncore[0] + ncas, ncore[1] + ncas)
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    jc_pp = numpy.empty((ncore[0],nmo,nmo))
    jc_PP = numpy.zeros((nmo,nmo))
    kc_pp = numpy.empty((ncore[0],nmo,nmo))
    jcvcv = numpy.zeros((ncore[0],nmo-ncore[0],ncore[0],nmo-ncore[0]))
    cvCV = numpy.empty((ncore[0],nmo-ncore[0],ncore[1],nmo-ncore[1]))
    vcp = numpy.empty((nmo-ncore[0],ncore[0],nmo))
    cpp = numpy.empty((ncore[0],nmo,nmo))
    for i in range(ncore[0]):
        buf = fload(i)
        klshape = (0, ncore[1], ncore[1], nmo-ncore[1])
        _ao2mo.nr_e2_(buf[ncore[0]:nmo], mo[1], klshape,
                      aosym='s4', mosym='s1', vout=cvCV[i])

        klshape = (0, nmo, 0, nmo)
        tmp = _ao2mo.nr_e2_(buf[i:i+1], mo[1], klshape, aosym='s4', mosym='s1')
        jc_PP += tmp.reshape(nmo,nmo)

        klshape = (0, ncore[0], 0, nmo)
        _ao2mo.nr_e2_(buf[ncore[0]:nmo], mo[0], klshape,
                      aosym='s4', mosym='s1', vout=vcp)
        kc_pp[i,ncore[0]:] = vcp[:,i]

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2_(buf[:ncore[0]], mo[0], klshape,
                      aosym='s4', mosym='s2', vout=buf[:ncore[0]])
        for j in range(ncore[0]):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    cpp[j].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        jc_pp[i] = cpp[i]
        kc_pp[i,:ncore[0]] = cpp[:,i]

        #jcvcv = cvcv * 2 - cvcv.transpose(2,1,0,3) - ccvv.transpose(0,2,1,3)
        jcvcv[i] = vcp[:,:,ncore[0]:] * 2 \
                 - vcp[:,:,ncore[0]:].transpose(2,1,0) \
                 - cpp[:,ncore[0]:,ncore[0]:].transpose(1,0,2)

    return jc_pp, jc_PP, kc_pp, jcvcv, cvCV



class _ERIS(object):
    def __init__(self, casscf, mo, method='incore'):
        mol = casscf.mol
        self.ncore = casscf.ncore
        self.ncas = casscf.ncas
        nmo = mo[0].shape[1]
        ncore = self.ncore
        ncas = self.ncas

        if method == 'outcore' \
           or _mem_usage(ncore, ncas, nmo)[0] + nmo**4*2/1e6 > casscf.max_memory*.9 \
           or casscf._scf._eri is None:
            self.jkcpp, self.jkcPP, self.jC_pp, self.jc_PP, \
            self.aapp, self.aaPP, self.AApp, self.AAPP, \
            self.appa, self.apPA, self.APPA, \
            self.Iapcv, self.IAPCV, self.apCV, self.APcv, \
            self.Icvcv, self.ICVCV, self.cvCV = \
                    trans_e1_outcore(casscf, mo, max_memory=casscf.max_memory,
                                     verbose=casscf.verbose)
        elif method == 'incore' and casscf._scf._eri is not None:
            self.jkcpp, self.jkcPP, self.jC_pp, self.jc_PP, \
            self.aapp, self.aaPP, self.AApp, self.AAPP, \
            self.appa, self.apPA, self.APPA, \
            self.Iapcv, self.IAPCV, self.apCV, self.APcv, \
            self.Icvcv, self.ICVCV, self.cvCV = \
                    trans_e1_incore(casscf._scf._eri, mo,
                                    casscf.ncore, casscf.ncas)
        else:
            raise KeyError('update ao2mo')

def _mem_usage(ncore, ncas, nmo):
    ncore = (ncore[0] + ncore[1]) * .5
    nvir = nmo - ncore
    outcore = (ncore**2*nvir**2*3 + ncas*nmo*ncore*nvir*4 + ncore*nmo**2*3 +
               ncas**2*nmo**2*7 + nmo**3*2) * 8/1e6
    incore = outcore + nmo**4/1e6 + ncore*nmo**3*4/1e6
    if outcore > 10000:
        sys.stderr.write('Be careful with the virtual memorty address space `ulimit -v`\n')
    return incore, outcore

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import ao2mo
    from pyscf.mcscf import mc1step_uhf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g',}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()

    mc = mc1step_uhf.CASSCF(m, 4, 4)
    mc.verbose = 4
    mo = m.mo_coeff

    eris0 = _ERIS(mc, mo, 'incore')
    eris1 = _ERIS(mc, mo, 'outcore')
    print('jkcpp', numpy.allclose(eris1.jkcpp, eris0.jkcpp))
    print('jkcPP', numpy.allclose(eris1.jkcPP, eris0.jkcPP))
    print('jC_pp', numpy.allclose(eris1.jC_pp, eris0.jC_pp))
    print('jc_PP', numpy.allclose(eris1.jc_PP, eris0.jc_PP))
    print('aapp ', numpy.allclose(eris1.aapp , eris0.aapp ))
    print('aaPP ', numpy.allclose(eris1.aaPP , eris0.aaPP ))
    print('AApp ', numpy.allclose(eris1.AApp , eris0.AApp ))
    print('AAPP ', numpy.allclose(eris1.AAPP , eris0.AAPP ))
    print('appa ', numpy.allclose(eris1.appa , eris0.appa ))
    print('apPA ', numpy.allclose(eris1.apPA , eris0.apPA ))
    print('APPA ', numpy.allclose(eris1.APPA , eris0.APPA ))
    print('cvCV ', numpy.allclose(eris1.cvCV , eris0.cvCV ))
    print('Icvcv', numpy.allclose(eris1.Icvcv, eris0.Icvcv))
    print('ICVCV', numpy.allclose(eris1.ICVCV, eris0.ICVCV))
    print('Iapcv', numpy.allclose(eris1.Iapcv, eris0.Iapcv))
    print('IAPCV', numpy.allclose(eris1.IAPCV, eris0.IAPCV))
    print('apCV ', numpy.allclose(eris1.apCV , eris0.apCV ))
    print('APcv ', numpy.allclose(eris1.APcv , eris0.APcv ))


    nmo = mo[0].shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = (ncas + ncore[0], ncas + ncore[1])
    eriaa = pyscf.ao2mo.incore.full(mc._scf._eri, mo[0])
    eriab = pyscf.ao2mo.incore.general(mc._scf._eri, (mo[0],mo[0],mo[1],mo[1]))
    eribb = pyscf.ao2mo.incore.full(mc._scf._eri, mo[1])
    eriaa = pyscf.ao2mo.restore(1, eriaa, nmo)
    eriab = pyscf.ao2mo.restore(1, eriab, nmo)
    eribb = pyscf.ao2mo.restore(1, eribb, nmo)
    jkcpp = numpy.einsum('iipq->ipq', eriaa[:ncore[0],:ncore[0],:,:]) \
          - numpy.einsum('ipqi->ipq', eriaa[:ncore[0],:,:,:ncore[0]])
    jkcPP = numpy.einsum('iipq->ipq', eribb[:ncore[1],:ncore[1],:,:]) \
          - numpy.einsum('ipqi->ipq', eribb[:ncore[1],:,:,:ncore[1]])
    jC_pp = numpy.einsum('pqii->pq', eriab[:,:,:ncore[1],:ncore[1]])
    jc_PP = numpy.einsum('iipq->pq', eriab[:ncore[0],:ncore[0],:,:])
    aapp = numpy.copy(eriaa[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
    aaPP = numpy.copy(eriab[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
    AApp = numpy.copy(eriab[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].transpose(2,3,0,1))
    AAPP = numpy.copy(eribb[ncore[1]:nocc[1],ncore[1]:nocc[1],:,:])
    appa = numpy.copy(eriaa[ncore[0]:nocc[0],:,:,ncore[0]:nocc[0]])
    apPA = numpy.copy(eriab[ncore[0]:nocc[0],:,:,ncore[1]:nocc[1]])
    APPA = numpy.copy(eribb[ncore[1]:nocc[1],:,:,ncore[1]:nocc[1]])

    cvCV = numpy.copy(eriab[:ncore[0],ncore[0]:,:ncore[1],ncore[1]:])
    Icvcv = eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:] * 2\
          - eriaa[:ncore[0],:ncore[0],ncore[0]:,ncore[0]:].transpose(0,3,1,2) \
          - eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:].transpose(0,3,2,1)
    ICVCV = eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:] * 2\
          - eribb[:ncore[1],:ncore[1],ncore[1]:,ncore[1]:].transpose(0,3,1,2) \
          - eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:].transpose(0,3,2,1)

    Iapcv = eriaa[ncore[0]:nocc[0],:,:ncore[0],ncore[0]:] * 2 \
          - eriaa[:,ncore[0]:,:ncore[0],ncore[0]:nocc[0]].transpose(3,0,2,1) \
          - eriaa[:,:ncore[0],ncore[0]:,ncore[0]:nocc[0]].transpose(3,0,1,2)
    IAPCV = eribb[ncore[1]:nocc[1],:,:ncore[1],ncore[1]:] * 2 \
          - eribb[:,ncore[1]:,:ncore[1],ncore[1]:nocc[1]].transpose(3,0,2,1) \
          - eribb[:,:ncore[1],ncore[1]:,ncore[1]:nocc[1]].transpose(3,0,1,2)
    apCV = numpy.copy(eriab[ncore[0]:nocc[0],:,:ncore[1],ncore[1]:])
    APcv = numpy.copy(eriab[:ncore[0],ncore[0]:,ncore[1]:nocc[1],:].transpose(2,3,0,1))

    print('jkcpp', numpy.allclose(jkcpp, eris0.jkcpp))
    print('jkcPP', numpy.allclose(jkcPP, eris0.jkcPP))
    print('jC_pp', numpy.allclose(jC_pp, eris0.jC_pp))
    print('jc_PP', numpy.allclose(jc_PP, eris0.jc_PP))
    print('aapp ', numpy.allclose(aapp , eris0.aapp ))
    print('aaPP ', numpy.allclose(aaPP , eris0.aaPP ))
    print('AApp ', numpy.allclose(AApp , eris0.AApp ))
    print('AAPP ', numpy.allclose(AAPP , eris0.AAPP ))
    print('appa ', numpy.allclose(appa , eris0.appa ))
    print('apPA ', numpy.allclose(apPA , eris0.apPA ))
    print('APPA ', numpy.allclose(APPA , eris0.APPA ))
    print('cvCV ', numpy.allclose(cvCV , eris0.cvCV ))
    print('Icvcv', numpy.allclose(Icvcv, eris0.Icvcv))
    print('ICVCV', numpy.allclose(ICVCV, eris0.ICVCV))
    print('Iapcv', numpy.allclose(Iapcv, eris0.Iapcv))
    print('IAPCV', numpy.allclose(IAPCV, eris0.IAPCV))
    print('apCV ', numpy.allclose(apCV , eris0.apCV ))
    print('APcv ', numpy.allclose(APcv , eris0.APcv ))

