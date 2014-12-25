#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import os
import random
import time
import numpy
import h5py
import pyscf.lib
import pyscf.lib.parameters as param
import pyscf.lib.logger as logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import direct

# default max_memory (MB) is 1600 MB, large cache cannot give better performance
# default ioblk_size is 512 MB

def full(mol, mo_coeff, erifile, max_memory=1500, ioblk_size=512, \
         dataname='eri_mo', verbose=None):
    time_0pass = (time.clock(), time.time())

    if verbose is None:
        verbose = mol.verbose
    log = logger.Logger(mol.stdout, verbose)

    mo_coeff = numpy.array(mo_coeff, order='F')
    nao, nmo = mo_coeff.shape
    nao_pair = nao*(nao+1) // 2
    nmo_pair = nmo*(nmo+1) // 2

    ish_ranges, e1_buflen, e2_buflen, ioblk_words = \
            info_swap_block(mol, max_memory, ioblk_size, nmo_pair, nmo_pair)

    swapfile = erifile + ('.swap%d' % random.randrange(1e8))
    fswap = h5py.File(swapfile)
    os.remove(swapfile)

    feri = h5py.File(erifile, 'w')
    h5d_eri = feri.create_dataset(dataname, (nmo_pair,nmo_pair), 'f8')

    log.debug('num. eri in MO repr. = %.8g, require disk %.8g + %.8g MB', \
              float(nmo_pair)**2, nmo_pair**2*8/1e6, nmo_pair*nao_pair*8/1e6)
    #log.debug('ish_ranges = %s', ish_ranges)
    log.debug('nao = %d, swap-block-shape ~ (%d,%d), mem cache size %.8g MB', \
              nao, e1_buflen, e2_buflen, \
              max(e1_buflen*nmo_pair,e2_buflen*nao_pair)*8/1e6)
    ijshape = klshape = (0, nmo, 0, nmo)

    # transform e1
    ti0 = log.timer('Initializing ao2mo.outcore.full', *time_0pass)
    for istep,sh_range in enumerate(ish_ranges):
        log.debug('step 1, AO %d:%d, [%d/%d], len(buf) = %d', \
                  sh_range[0], sh_range[1], istep+1, len(ish_ranges), \
                  sh_range[2])
        buf = _ao2mo.nr_e1range_(mo_coeff, sh_range, ijshape, \
                                 mol._atm, mol._bas, mol._env)
        ti2 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,len(ish_ranges)),
                        *ti0)
        #fg = fswap.create_group(str(istep))
        #tc0 = ti2
        for ic, col0 in enumerate(range(0, nmo_pair, e2_buflen)):
            col1 = min(col0+e2_buflen, nmo_pair)
            fswap['%d/%d'%(istep,ic)] = pyscf.lib.transpose(buf[:,col0:col1])
            #tc0 = log.timer('          transpose block %d'%ic, *tc0)
        ti0 = log.timer('transposing to disk', *ti2)
        # release the memory of buf before allocating temporary data
        buf = None

    time_1pass = log.timer('AO->MO eri transformation 1 pass', *time_0pass)

    ti0 = time_1pass
    buf = numpy.empty((e2_buflen, nao_pair))
    for istep, row0 in enumerate(range(0, nmo_pair, e2_buflen)):
        tioi = 0
        row1 = min(row0+e2_buflen, nmo_pair)
        nrow = row1 - row0
        ioblklen = min(nrow, int(ioblk_words/nmo_pair))
        log.debug('step 2, %d:%d, [%d/%d], len(buf) = %d, len(ioblk) = %d', \
                  row0, row1, istep+1, nmo_pair/e2_buflen+1, nrow, ioblklen)

        col0 = 0
        for ic, _ in enumerate(ish_ranges):
            dat = fswap['%d/%d'%(ic,istep)]
            col1 = col0 + dat.shape[1]
            buf[:nrow,col0:col1] = dat
            col0 = col1
        ti2 = log.timer('step 2 [%d/%d], fill buf' % (istep+1,nmo_pair/e2_buflen+1),
                        *ti0)
        tioi += ti2[1]-ti0[1]
        tc0 = ti2
        for ih, ib in enumerate(range(0, nrow, ioblklen)):
            ib1 = min(ib+ioblklen, nrow)
            pbuf = buf[ib:ib1]
            pbuf = _ao2mo.nr_e2_(pbuf, mo_coeff, klshape)
            tc1 = (time.clock(), time.time())

            h5d_eri[row0+ib:row0+ib1] = pbuf
            tc2 = (time.clock(), time.time())
            log.debug('          (%d/%d) trans CPU: %5.2f, ' \
                      'dumping CPU: %5.2f, I/O: %5.2f', \
                      ih+1, nrow/ioblklen+1, tc1[0]-tc0[0], \
                      tc2[0]-tc1[0], tc2[1]-tc1[1])
            tioi += tc2[1]-tc1[1]
            tc0 = tc2

        ti1 = (time.clock(), time.time())
        log.debug('step 2 [%d/%d] CPU time: %9.2f, I/O time: %9.2f', \
                  istep+1, nmo_pair/e2_buflen+1, ti1[0]-ti0[0], tioi)
        ti0 = ti1
    feri.close()
    fswap.close()

    log.timer('AO->MO eri transformation 2 pass', *time_1pass)
    log.timer('AO->MO eri transformation', *time_0pass)

def general(mol, mo_coeffs, erifile, max_memory=None, ioblk_size=512, \
            dataname='eri_mo', verbose=None, compact=True):
    time_0pass = (time.clock(), time.time())

    if verbose is None:
        verbose = mol.verbose
    log = logger.Logger(mol.stdout, verbose)

    def iden_coeffs(mo1, mo2):
        return (id(mo1) == id(mo2)) \
                or (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    klsame = compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    if ijsame:
        log.debug('i-mo == j-mo')
        nij_pair = nmoi*(nmoi+1) // 2
    else:
        nij_pair = nmoi*nmoj

    if klsame:
        log.debug('k-mo == l-mo')
        nkl_pair = nmok*(nmok+1) // 2
    else:
        nkl_pair = nmok*nmol

    if nij_pair > nkl_pair:
        log.warn('low efficiency for AO to MO trans!')

    ish_ranges, e1_buflen, e2_buflen, ioblk_words = \
            info_swap_block(mol, max_memory, ioblk_size, nij_pair, nkl_pair)

    swapfile = erifile + ('.swap%d' % random.randrange(1e8))
    fswap = h5py.File(swapfile)
    os.remove(swapfile)

    feri = h5py.File(erifile, 'w')
    h5d_eri = feri.create_dataset(dataname, (nij_pair,nkl_pair), 'f8')
    if nij_pair == 0 or nkl_pair == 0:
        feri.close()
        return None

    nao = mo_coeffs[0].shape[0]
    nao_pair = nao*(nao+1) // 2
    log.debug('num. eri in MO repr. = %.8g, require disk %.8g + %.8g MB', \
              float(nij_pair)*nkl_pair, nij_pair*nkl_pair*8/1e6, \
              nij_pair*nao_pair*8/1e6)
    #log.debug('ish_ranges = %s', ish_ranges)
    log.debug('nao = %d, swap-block-shape ~ (%d,%d), mem cache size %.8g MB', \
              nao, e1_buflen, e2_buflen, \
              max(e1_buflen*nij_pair,e2_buflen*nao_pair)*8/1e6)
    if ijsame:
        moji = numpy.array(mo_coeffs[0], order='F', copy=False)
        ijshape = (0, nmoi, 0, nmoi)
    else:
        moji = numpy.array(numpy.hstack((mo_coeffs[1],mo_coeffs[0])), \
                           order='F', copy=False)
        ijshape = (nmoj, nmoi, 0, nmoj)
    if klsame:
        molk = numpy.array(mo_coeffs[2], order='F', copy=False)
        klshape = (0, nmok, 0, nmok)
    else:
        molk = numpy.array(numpy.hstack((mo_coeffs[3],mo_coeffs[2])), \
                           order='F', copy=False)
        klshape = (nmol, nmok, 0, nmol)

    # transform e1
    ti0 = log.timer('Initializing ao2mo.outcore.general', *time_0pass)
    for istep,sh_range in enumerate(ish_ranges):
        log.debug('step 1, AO %d:%d, [%d/%d], len(buf) = %d', \
                  sh_range[0], sh_range[1], istep+1, len(ish_ranges), \
                  sh_range[2])
        buf = _ao2mo.nr_e1range_(moji, sh_range, ijshape, \
                                 mol._atm, mol._bas, mol._env)
        ti2 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,len(ish_ranges)),
                        *ti0)
        #tc0 = ti2
        for ic, col0 in enumerate(range(0, nij_pair, e2_buflen)):
            col1 = min(col0+e2_buflen, nij_pair)
            fswap['%d/%d'%(istep,ic)] = pyscf.lib.transpose(buf[:,col0:col1])
            #tc0 = log.timer('          transpose block %d'%ic, *tc0)
        ti0 = log.timer('transposing to disk', *ti2)
        # release the memory of buf before allocating temporary data
        buf = None

    time_1pass = log.timer('AO->MO eri transformation 1 pass', *time_0pass)

    ti0 = time_1pass
    buf = numpy.empty((e2_buflen, nao_pair))
    for istep, row0 in enumerate(range(0, nij_pair, e2_buflen)):
        tioi = 0
        row1 = min(row0+e2_buflen, nij_pair)
        nrow = row1 - row0
        ioblklen = min(nrow, int(ioblk_words/nkl_pair))
        log.debug('step 2, %d:%d, [%d/%d], len(buf) = %d, len(ioblk) = %d', \
                  row0, row1, istep+1, nij_pair/e2_buflen+1, nrow, ioblklen)

        col0 = 0
        for ic, _ in enumerate(ish_ranges):
            dat = fswap['%d/%d'%(ic,istep)]
            col1 = col0 + dat.shape[1]
            buf[:nrow,col0:col1] = dat
            col0 = col1
        ti2 = log.timer('step 2 [%d/%d], fill buf' % (istep+1,nij_pair/e2_buflen+1),
                        *ti0)
        tioi += ti2[1]-ti0[1]
        tc0 = ti2
        for ih, ib in enumerate(range(0, nrow, ioblklen)):
            ib1 = min(ib+ioblklen, nrow)
            pbuf = buf[ib:ib1]
            pbuf = _ao2mo.nr_e2_(pbuf, molk, klshape)
            tc1 = (time.clock(), time.time())

            h5d_eri[row0+ib:row0+ib1] = pbuf
            tc2 = (time.clock(), time.time())
            log.debug('          (%d/%d) trans CPU: %5.2f, ' \
                      'dumping CPU: %5.2f, I/O: %5.2f', \
                      ih+1, nrow/ioblklen+1, tc1[0]-tc0[0], \
                      tc2[0]-tc1[0], tc2[1]-tc1[1])
            tioi += tc2[1]-tc1[1]
            tc0 = tc2

        ti1 = (time.clock(), time.time())
        log.debug('step 2 [%d/%d] CPU time: %9.2f, I/O time: %9.2f', \
                  istep+1, nij_pair/e2_buflen+1, ti1[0]-ti0[0], tioi)
        ti0 = ti1
    feri.close()
    fswap.close()

    log.timer('AO->MO eri transformation 2 pass', *time_1pass)
    log.timer('AO->MO eri transformation', *time_0pass)

# decide the number of blocks needed for swap file
def info_swap_block(mol, max_memory, ioblk_size, nij_pair, nkl_pair):
    mem_words, ioblk_words = \
            direct._memory_and_ioblk_size(max_memory, ioblk_size,
                                          nij_pair, nkl_pair)
    nthreads = _ao2mo._get_num_threads()

    nao = mol.num_NR_cgto()
    nao_pair = nao*(nao+1) // 2

    # decided the buffer row and column sizes
    e1trans_buflen = min(int(mem_words/nij_pair), nao_pair)
    e2trans_buflen = min(int(mem_words/nao_pair/nthreads)*nthreads, nij_pair)
    # floating ioblk size
    if e1trans_buflen*e2trans_buflen > 1.2*ioblk_words:
        e2trans_buflen = int(1.2*ioblk_words/e1trans_buflen/nthreads)*nthreads

    ish_ranges = _info_sh_ranges(mol, e1trans_buflen)
    return ish_ranges, e1trans_buflen, e2trans_buflen, ioblk_words

def _info_sh_ranges(mol, buflen):
    # based on the row size of buffer, dynamic range of ishell for each buffer
    bas_dim = [(mol.angular_of_bas(i)*2+1)*(mol.nctr_of_bas(i)) \
               for i in range(mol.nbas)]
    ao_loc = [0]
    for i in bas_dim:
        ao_loc.append(ao_loc[-1]+i)
    nao = ao_loc[-1]
    nao_pair = nao*(nao+1) // 2
    ish_seg = [0] # save the starting shell of each buffer
    bufrows = []
    ij_start = 0
    for i in range(mol.nbas):
        ij_end = ao_loc[i+1]*(ao_loc[i+1]+1)//2
        if ij_end - ij_start > buflen:
            ish_seg.append(i) # put present shell to next segments
            bufrows.append(ao_loc[i]*(ao_loc[i]+1)//2-ij_start)
            ij_start = ao_loc[i]*(ao_loc[i]+1)//2
    ish_seg.append(mol.nbas)
    bufrows.append(nao_pair-ij_start)
    assert(sum(bufrows) == nao_pair)
    # for each buffer, ish_ranges record (start, end, bufrow)
    ish_ranges = list(zip(ish_seg[:-1], ish_seg[1:], bufrows))
    return ish_ranges


if __name__ == '__main__':
    import scf
    import gto
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_outcore'
    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()
    nao = mol.num_NR_cgto()
    npair = nao*(nao+1)//2
    print(info_swap_block(mol, 4 , 2, npair, npair))
    print(info_swap_block(mol, 20, 1, npair, npair))
    print(info_swap_block(mol, 20, 8, npair, npair))

    rhf = scf.RHF(mol)
    rhf.scf()
    import time
    print(time.clock())
    full(mol, rhf.mo_coeff, 'h2oeri.h5', 10, 5)
    print(time.clock())
    import incore
    eri0 = incore.full(rhf._eri, rhf.mo_coeff)
    feri = h5py.File('h2oeri.h5', 'r')
    print('full', abs(eri0-feri['eri_mo']).sum())
    feri.close()

    print(time.clock())
    c = rhf.mo_coeff
    general(mol, (c,c,c,c), 'h2oeri.h5', 10, 5)
    print(time.clock())
    feri = h5py.File('h2oeri.h5', 'r')
    print('general', abs(eri0-feri['eri_mo']).sum())
    feri.close()

    # set ijsame and klsame to False, then check
    c = rhf.mo_coeff
    n = c.shape[1]
    general(mol, (c,c,c,c), 'h2oeri.h5', 10, 5, compact=False)
    feri = h5py.File('h2oeri.h5', 'r')
    eri1 = numpy.array(feri['eri_mo']).reshape(n,n,n,n)
    import addons
    eri1 = addons.restore(4, eri1, n)
    print('general', abs(eri0-eri1).sum())
