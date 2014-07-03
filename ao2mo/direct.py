#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import os
import numpy
import h5py
import time

import pyscf.lib
import pyscf.lib.parameters as param
import pyscf.lib._ao2mo as _ao2mo

# default max_memory (MB) is lib.parameters.MEMORY_MAX
# default ioblk_size is 512 MB

def full(mol, mo_coeff, erifile, max_memory=None, ioblk_size=512, \
         dataname='eri_mo', verbose=None):
    wall0 = time.time()
    tcpu0 = time.clock()

    if verbose is None:
        verbose = mol.verbose
    log = pyscf.lib.logger.Logger(mol.fout, verbose)

    if mo_coeff.flags.c_contiguous:
        mo_coeff = numpy.array(mo_coeff, order='F')
    nao, nmo = mo_coeff.shape
    nao_pair = nao*(nao+1) / 2
    nmo_pair = nmo*(nmo+1) / 2

    if max_memory is None:
        max_memory = param.MEMORY_MAX * 1e6 / 8
    else:
        max_memory = max_memory * 1e6 / 8
    ioblk_size = max(ioblk_size*1e6/8, \
                     int(os.environ['OMP_NUM_THREADS'])*16*nmo_pair)

    feri = h5py.File(erifile, 'w')
    h5d_eri = feri.create_dataset(dataname, (nmo_pair,nmo_pair), 'f8')

    blklen = int(max_memory/nao_pair)
    log.debug('num. eri in MO repr. = %.8g, require disk %.8g MB', \
              float(nmo_pair)**2, float(nmo_pair)**2*8/1e6)
    log.debug('nao = %d, block len = %d, mem cache size %.8g MB', \
              nao, blklen, float(blklen)*nao_pair*8/1e6)
    def f_ijshape(block_id):
        n = block_id * blklen
        istart = _extract_pair_by_id(n)[0]
        icount = min(nmo, _extract_pair_by_id(n+blklen)[0]) - istart
        return (istart, icount, 0, istart+icount)
    klshape = (0, nmo, 0, nmo)

    num_block = _int_ceiling(nmo_pair, blklen)
    blkstart = 0
    for block_id in range(num_block):
        tcpui = 0
        tioi = 0
        tcpu1 = time.clock()

        ijshape = f_ijshape(block_id)
        log.debug('transform MO %d:%d, [%d/%d] step 1, trans e1', \
                  ijshape[0], ijshape[0]+ijshape[1], block_id+1, num_block)
        buf = _ao2mo.nr_e1_ao2mo(mo_coeff, ijshape, mol._atm, mol._bas, mol._env)
        tcpu1, dt = time.clock(), time.clock()-tcpu1
        log.debug('CPU time: %12.2f', dt)
        tcpui += dt

        nrow = buf.shape[0]
        ioblklen = min(nrow, int(ioblk_size/nmo_pair))
        log.debug('transform MO %d:%d, [%d/%d] step 2, len(buf) = %d, ' \
                  'len(ioblk) = %d', \
                  ijshape[0], ijshape[0]+ijshape[1], block_id+1, num_block, \
                  nrow, ioblklen)
        for ih, ib in enumerate(range(0, nrow, ioblklen)):
            ib1 = min(ib+ioblklen, nrow)
            pbuf = buf[ib:ib1]
            pbuf = _ao2mo.nr_e2_ao2mo(pbuf, mo_coeff, klshape)
            tcpu1, dt = time.clock(), time.clock()-tcpu1
            log.debug('                    step 2.(%d/%d) trans e2 ' \
                      'CPU time: %5.2f', \
                      ih+1, nrow/ioblklen+1, dt)
            tcpui += dt

            tio1 = time.time()
            h5d_eri[blkstart+ib:blkstart+ib1] = pbuf
            tcpu1, dt = time.clock(), time.clock()-tcpu1
            tcpui += dt
            tio1, dtio = time.time(), time.time()-tio1
            tioi += dtio
            log.debug('                    step 2.(%d/%d) dumping  ' \
                      'CPU time: %5.2f, I/O time: %5.2f', \
                      ih+1, nrow/ioblklen+1, dt, dtio)
        # release the memory held by buf, pbuf
        buf = None
        pbuf = None
        blkstart += nrow

        log.debug('transform [%d/%d] CPU time: %12.2f, I/O time: %12.2f', \
                  block_id+1, num_block, tcpui, tioi)
    feri.close()

    log.debug('AO->MO eri transformation, CPU time: %12.2f, ' \
              'Wall time: %12.2f', \
              time.clock()-tcpu0, time.time()-wall0)

def full_iofree(mol, mo_coeff, verbose=None):
    wall0 = time.time()
    tcpu0 = time.clock()

    if verbose is None:
        verbose = mol.verbose
    log = pyscf.lib.logger.Logger(mol.fout, verbose)

    mo_coeff = numpy.array(mo_coeff, order='F')
    nao, nmo = mo_coeff.shape
    nao_pair = nao*(nao+1) / 2
    nmo_pair = nmo*(nmo+1) / 2

    log.debug('num. eri in MO repr. = %.8g, require memory %.8g MB', \
              float(nmo_pair)**2, float(nmo_pair)**2*8/1e6)
    ijshape = klshape = (0, nmo, 0, nmo)

    tcpu1 = time.clock()
    log.debug('transform MO step 1, trans e1')
    buf = _ao2mo.nr_e1_ao2mo(mo_coeff, ijshape, mol._atm, mol._bas, mol._env)
    tcpu1, dt = time.clock(), time.clock()-tcpu1
    log.debug('CPU time: %12.2f', dt)

    log.debug('transform MO step 2, trans e2')
    buf = _ao2mo.nr_e2_ao2mo(buf, mo_coeff, klshape)
    tcpu1, dt = time.clock(), time.clock()-tcpu1
    log.debug('CPU time: %5.2f', dt)

    log.debug('AO->MO eri transformation, CPU time: %12.2f, ' \
              'Wall time: %12.2f', \
              time.clock()-tcpu0, time.time()-wall0)
    return buf

# n = i*(i+1)/2+j, n => (i,j)
def _extract_pair_by_id(n):
    i = int(numpy.sqrt(2*n+.25) - .5 + 1e-7)
    j = n - i*(i+1)/2
    return i,j

def _int_ceiling(n, m):
    return (n-1)/m + 1

#def gen_int2e_ao2mo(mol, mo_coeff):
#    eritmpfile = tempfile.mktemp('.h5')
#    full(mol, mo_coeff, eritmpfile, dataname='eri_mo', verbose=0)
#    feri = h5py.File(eritmpfile)
#    eri_mo = numpy.array(feri['eri_mo'])
#    os.remove(eritmpfile)
#    return eri_mo
def gen_int2e_ao2mo(mol, mo_coeff):
    return full_iofree(mol, mo_coeff, verbose=0)

#############################
# general AO to MO transformation takes four MO coefficients
# mo_coeffs = [i-mo,j-mo,k-mo,l-mo]
# The most efficient seq: dim(i-mo) <= dim(j-mo), dim(k-mo) <= dim(l-mo)
#                         dim(i-mo)*dim(j-mo) < dim(k-mo)*dim(l-mo)
# Low efficiency if dim(k-mo)*dim(l-mo) < dim(i-mo)*dim(j-mo)
#
# erifile is the MO integrals file dumped in HDF5 format, the dataset in that
# file uses the name given by dataname
#
def general(mol, mo_coeffs, erifile, max_memory=None, ioblk_size=512, \
            dataname='eri_mo', verbose=None, compact=True):
    wall0 = time.time()
    tcpu0 = time.clock()

    if verbose is None:
        verbose = mol.verbose
    log = pyscf.lib.logger.Logger(mol.fout, verbose)

    def iden_coeffs(mo1, mo2):
        return (id(mo1) == id(mo2)) \
                or (mo1.shape==mo2.shape and abs(mo1-mo2).sum()<1e-12)

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    klsame = compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    if ijsame:
        log.debug('i-mo == j-mo')
        nij_pair = nmoi*(nmoi+1) / 2
    else:
        nij_pair = nmoi*nmoj

    if klsame:
        log.debug('k-mo == l-mo')
        nkl_pair = nmok*(nmok+1) / 2
    else:
        nkl_pair = nmok*nmol

    if nij_pair > nkl_pair:
        log.warn('low efficiency for AO to MO trans!')

    if max_memory is None:
        max_memory = param.MEMORY_MAX * 1e6 / 8
    else:
        max_memory = max_memory * 1e6 / 8
    ioblk_size = max(ioblk_size*1e6/8, \
                     int(os.environ['OMP_NUM_THREADS'])*16*nkl_pair)

    feri = h5py.File(erifile, 'w')
    h5d_eri = feri.create_dataset(dataname, (nij_pair,nkl_pair), 'f8')

    nao = mo_coeffs[0].shape[0]
    nao_pair = nao*(nao+1) / 2
    blklen = int(max_memory/nao_pair)
    log.debug('num. eri in MO repr. = %.8g, require disk %.8g MB', \
              float(nij_pair)*nkl_pair, float(nij_pair)*nkl_pair*8/1e6)
    log.debug('nao = %d, block len = %d, mem cache size %.8g MB', \
              nao, blklen, float(blklen)*nao_pair*8/1e6)

    if ijsame:
        moji = numpy.array(mo_coeffs[0], order='F')
        def f_ijshape(block_id):
            n = block_id * blklen
            istart = _extract_pair_by_id(n)[0]
            icount = min(nmoi, _extract_pair_by_id(n+blklen)[0]) - istart
            return (istart, icount, 0, istart+icount)
    else:
        moji = numpy.array(numpy.hstack((mo_coeffs[1],mo_coeffs[0])), order='F')
        def f_ijshape(block_id):
            istart = block_id*blklen / nmoj
            icount = min(nmoi, (block_id*blklen+blklen)/nmoj) - istart
            return (nmoj+istart, icount, 0, nmoj)

    if klsame:
        molk = numpy.array(mo_coeffs[2], order='F')
        klshape = (0, nmok, 0, nmok)
    else:
        molk = numpy.array(numpy.hstack((mo_coeffs[3],mo_coeffs[2])), order='F')
        klshape = (nmol, nmok, 0, nmol)

    num_block = _int_ceiling(nij_pair, blklen)
    blkstart = 0
    for block_id in range(num_block):
        tcpui = 0
        tioi = 0
        tcpu1 = time.clock()

        ijshape = f_ijshape(block_id)
        log.debug('transform MO %d:%d, [%d/%d] step 1, trans e1', \
                  ijshape[0], ijshape[0]+ijshape[1], block_id+1, num_block)
        buf = _ao2mo.nr_e1_ao2mo(moji, ijshape, mol._atm, mol._bas, mol._env)
        tcpu1, dt = time.clock(), time.clock()-tcpu1
        log.debug('CPU time: %12.2f', dt)
        tcpui += dt

        nrow = buf.shape[0]
        ioblklen = min(nrow, int(ioblk_size/nkl_pair))
        log.debug('transform MO %d:%d, [%d/%d] step 2, len(buf) = %d, ' \
                  'len(ioblk) = %d', \
                  ijshape[0], ijshape[0]+ijshape[1], block_id+1, num_block, \
                  nrow, ioblklen)
        for ih, ib in enumerate(range(0, nrow, ioblklen)):
            ib1 = min(ib+ioblklen, nrow)
            pbuf = buf[ib:ib1]
            pbuf = _ao2mo.nr_e2_ao2mo(pbuf, molk, klshape)
            tcpu1, dt = time.clock(), time.clock()-tcpu1
            log.debug('                    step 2.(%d/%d) trans e2 ' \
                      'CPU time: %5.2f', \
                      ih+1, nrow/ioblklen+1, dt)
            tcpui += dt

            tio1 = time.time()
            h5d_eri[blkstart+ib:blkstart+ib1] = pbuf
            tcpu1, dt = time.clock(), time.clock()-tcpu1
            tcpui += dt
            tio1, dtio = time.time(), time.time()-tio1
            tioi += dtio
            log.debug('                    step 2.(%d/%d) dumping  ' \
                      'CPU time: %5.2f, I/O time: %5.2f', \
                      ih+1, nrow/ioblklen+1, dt, dtio)
        # release the memory held by buf, pbuf
        buf = None
        pbuf = None
        blkstart += nrow

        log.debug('transform [%d/%d] CPU time: %12.2f, I/O time: %12.2f', \
                  block_id+1, num_block, tcpui, tioi)
    feri.close()

    log.debug('AO->MO eri transformation, CPU time: %12.2f, ' \
              'Wall time: %12.2f', \
              time.clock()-tcpu0, time.time()-wall0)

def general_iofree(mol, mo_coeffs, verbose=None):
    wall0 = time.time()
    tcpu0 = time.clock()

    if verbose is None:
        verbose = mol.verbose
    log = pyscf.lib.logger.Logger(mol.fout, verbose)

    def iden_coeffs(mo1, mo2):
        return (id(mo1) == id(mo2)) \
                or (mo1.shape==mo2.shape and abs(mo1-mo2).sum()<1e-12)

    ijsame = iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    klsame = iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    if ijsame:
        log.debug('i-mo == j-mo')
        nij_pair = nmoi*(nmoi+1) / 2
    else:
        nij_pair = nmoi*nmoj

    if klsame:
        log.debug('k-mo == l-mo')
        nkl_pair = nmok*(nmok+1) / 2
    else:
        nkl_pair = nmok*nmol

    if nij_pair > nkl_pair:
        log.warn('low efficiency for AO to MO trans!')

    nao = mo_coeffs[0].shape[0]
    nao_pair = nao*(nao+1) / 2
    log.debug('num. eri in MO repr. = %.8g, require memory %.8g MB', \
              float(nij_pair)*nkl_pair,
              float(nij_pair)*max(nao_pair,nkl_pair)*8/1e6)

    if ijsame:
        moji = numpy.array(mo_coeffs[0], order='F')
        ijshape = (0, nmoi, 0, nmoi)
    else:
        moji = numpy.array(numpy.hstack((mo_coeffs[1],mo_coeffs[0])), order='F')
        ijshape = (nmoj, nmoi, 0, nmoj)

    if klsame:
        molk = numpy.array(mo_coeffs[2], order='F')
        klshape = (0, nmok, 0, nmok)
    else:
        molk = numpy.array(numpy.hstack((mo_coeffs[3],mo_coeffs[2])), order='F')
        klshape = (nmol, nmok, 0, nmol)

    tcpu1 = time.clock()
    log.debug('transform MO step 1, trans e1')
    buf = _ao2mo.nr_e1_ao2mo(moji, ijshape, mol._atm, mol._bas, mol._env)
    tcpu1, dt = time.clock(), time.clock()-tcpu1
    log.debug('CPU time: %12.2f', dt)

    log.debug('transform MO step 2, trans e2')
    buf = _ao2mo.nr_e2_ao2mo(buf, molk, klshape)
    tcpu1, dt = time.clock(), time.clock()-tcpu1
    log.debug('CPU time: %12.2f', dt)

    log.debug('AO->MO eri transformation, CPU time: %12.2f, ' \
              'Wall time: %12.2f', \
              time.clock()-tcpu0, time.time()-wall0)
    return buf

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_h2o'
    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()
    import time
    print time.clock()
    full(mol, rhf.mo_coeff, 'h2oeri.h5', 10, 5)
    print time.clock()
    from pyscf import incore
    eri0 = incore.full(rhf._eri, rhf.mo_coeff)
    feri = h5py.File('h2oeri.h5', 'r')
    print 'full', abs(eri0-feri['eri_mo']).sum()
    feri.close()

    print time.clock()
    c = rhf.mo_coeff
    general(mol, (c,c,c,c), 'h2oeri.h5', 10, 5)
    print time.clock()
    feri = h5py.File('h2oeri.h5', 'r')
    print 'general', abs(eri0-feri['eri_mo']).sum()
    feri.close()

    print time.clock()
    eri1 = full_iofree(mol, rhf.mo_coeff)
    print time.clock()
    print 'full_iofree', abs(eri0-eri1).sum()

    print time.clock()
    eri1 = general_iofree(mol, (c,c,c,c))
    print time.clock()
    print 'general_iofree', abs(eri0-eri1).sum()

    # set ijsame and klsame to False, then check
    #c = rhf.mo_coeff
    #n = c.shape[1]
    #eri1 = general_iofree(mol, (c,c,c,c))
    #eri1 = eri1.reshape(n,n,n,n)
    #eri1 = mo_eri_incore.gen_int2e_from_full_eri(eri1)
    #print 'general_iofree', abs(eri0-eri1).sum()
