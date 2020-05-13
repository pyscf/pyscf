#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Selected CI using Heat-Bath CI algorithm
(JCTC 2016, 12, 3674-3680)

Simple usage::

'''

import sys
import numpy
import time
import ctypes
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import direct_spin1

libhci = lib.load_library('libhci')

def contract_2e_ctypes(h1_h2, civec, norb, nelec, hdiag=None, **kwargs):
    h1, eri = h1_h2
    strs = civec._strs
    ndet = len(strs)
    if hdiag is None:
        hdiag = make_hdiag(h1, eri, strs, norb, nelec)
    ci1 = numpy.zeros_like(civec)

    h1 = numpy.asarray(h1, order='C')
    eri = numpy.asarray(eri, order='C')
    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    hdiag = numpy.asarray(hdiag, order='C')
    ci1 = numpy.asarray(ci1, order='C')

    libhci.contract_h_c(h1.ctypes.data_as(ctypes.c_void_p), 
                        eri.ctypes.data_as(ctypes.c_void_p), 
                        ctypes.c_int(norb), 
                        ctypes.c_int(nelec[0]), 
                        ctypes.c_int(nelec[1]), 
                        strs.ctypes.data_as(ctypes.c_void_p), 
                        civec.ctypes.data_as(ctypes.c_void_p), 
                        hdiag.ctypes.data_as(ctypes.c_void_p), 
                        ctypes.c_ulonglong(ndet), 
                        ci1.ctypes.data_as(ctypes.c_void_p))

    return ci1

def contract_2e(h1_h2, civec, norb, nelec, hdiag=None, **kwargs):
    h1, eri = h1_h2
    strs = civec._strs
    ndet = len(strs)
    if hdiag is None:
        hdiag = make_hdiag(h1, eri, strs, norb, nelec)
    ci1 = numpy.zeros_like(civec)

    eri = eri.reshape([norb]*4)

    for ip in range(ndet):
        for jp in range(ip):
            stria, strib = strs[ip].reshape(2,-1)
            strja, strjb = strs[jp].reshape(2,-1)
            desa, crea = str_diff(stria, strja)
            if len(desa) > 2:
                continue
            desb, creb = str_diff(strib, strjb)
            if len(desb) + len(desa) > 2:
                continue
            if len(desa) + len(desb) == 1:
# alpha->alpha
                if len(desb) == 0:
                    i,a = desa[0], crea[0]
                    occsa = str2orblst(stria, norb)[0]
                    occsb = str2orblst(strib, norb)[0]
                    fai = h1[a,i]
                    for k in occsa:
                        fai += eri[k,k,a,i] - eri[k,i,a,k]
                    for k in occsb:
                        fai += eri[k,k,a,i]
                    sign = cre_des_sign(a, i, stria)
                    ci1[jp] += sign * fai * civec[ip]
                    ci1[ip] += sign * fai * civec[jp]
# beta ->beta
                elif len(desa) == 0:
                    i,a = desb[0], creb[0]
                    occsa = str2orblst(stria, norb)[0]
                    occsb = str2orblst(strib, norb)[0]
                    fai = h1[a,i]
                    for k in occsb:
                        fai += eri[k,k,a,i] - eri[k,i,a,k]
                    for k in occsa:
                        fai += eri[k,k,a,i]
                    sign = cre_des_sign(a, i, strib)
                    ci1[jp] += sign * fai * civec[ip]
                    ci1[ip] += sign * fai * civec[jp]

            else:
# alpha,alpha->alpha,alpha
                if len(desb) == 0:
                    i,j = desa
                    a,b = crea
# 6 conditions for i,j,a,b
# --++, ++--, -+-+, +-+-, -++-, +--+ 
                    if a > j or i > b:
# condition --++, ++--
                        v = eri[a,j,b,i]-eri[a,i,b,j]
                        sign = cre_des_sign(b, i, stria)
                        sign*= cre_des_sign(a, j, stria)
                    else:
# condition -+-+, +-+-, -++-, +--+ 
                        v = eri[a,i,b,j]-eri[a,j,b,i]
                        sign = cre_des_sign(b, j, stria)
                        sign*= cre_des_sign(a, i, stria)
                    ci1[jp] += sign * v * civec[ip]
                    ci1[ip] += sign * v * civec[jp]
# beta ,beta ->beta ,beta
                elif len(desa) == 0:
                    i,j = desb
                    a,b = creb
                    if a > j or i > b:
                        v = eri[a,j,b,i]-eri[a,i,b,j]
                        sign = cre_des_sign(b, i, strib)
                        sign*= cre_des_sign(a, j, strib)
                    else:
                        v = eri[a,i,b,j]-eri[a,j,b,i]
                        sign = cre_des_sign(b, j, strib)
                        sign*= cre_des_sign(a, i, strib)
                    ci1[jp] += sign * v * civec[ip]
                    ci1[ip] += sign * v * civec[jp]
# alpha,beta ->alpha,beta
                else:
                    i,a = desa[0], crea[0]
                    j,b = desb[0], creb[0]
                    v = eri[a,i,b,j]
                    sign = cre_des_sign(a, i, stria)
                    sign*= cre_des_sign(b, j, strib)
                    ci1[jp] += sign * v * civec[ip]
                    ci1[ip] += sign * v * civec[jp]
        ci1[ip] += hdiag[ip] * civec[ip]

    return ci1

def spin_square(civec, norb, nelec):
    ss = numpy.dot(civec.T, contract_ss(civec, norb, nelec))
    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

def contract_ss(civec, norb, nelec):
    strs = civec._strs
    ndet = len(strs)
    ci1 = numpy.zeros_like(civec)

    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    ci1 = numpy.asarray(ci1, order='C')

    libhci.contract_ss_c(ctypes.c_int(norb), 
                        ctypes.c_int(nelec[0]), 
                        ctypes.c_int(nelec[1]), 
                        strs.ctypes.data_as(ctypes.c_void_p), 
                        civec.ctypes.data_as(ctypes.c_void_p), 
                        ctypes.c_ulonglong(ndet), 
                        ci1.ctypes.data_as(ctypes.c_void_p))

    return ci1

def make_hdiag(h1e, eri, strs, norb, nelec):
    eri = ao2mo.restore(1, eri, norb)
    diagj = numpy.einsum('iijj->ij',eri)
    diagk = numpy.einsum('ijji->ij',eri)

    ndet = len(strs)
    hdiag = numpy.zeros(ndet)
    for idet, (stra, strb) in enumerate(strs.reshape(ndet,2,-1)):
        aocc = str2orblst(stra, norb)[0]
        bocc = str2orblst(strb, norb)[0]
        e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
        e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
           + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
           - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
        hdiag[idet] = e1 + e2*.5
    return hdiag

def cre_des_sign(p, q, string):
    nset = len(string)
    pg, pb = p//64, p%64
    qg, qb = q//64, q%64

    if pg > qg:
        n1 = 0
        for i in range(nset-pg, nset-qg-1):
            n1 += bin(string[i]).count('1')
        n1 += bin(string[-1-pg] & numpy.uint64((1<<pb) - 1)).count('1')
        n1 += string[-1-qg] >> numpy.uint64(qb+1)
    elif pg < qg:
        n1 = 0
        for i in range(nset-qg, nset-pg-1):
            n1 += bin(string[i]).count('1')
        n1 += bin(string[-1-qg] & numpy.uint64((1<<qb) - 1)).count('1')
        n1 += string[-1-pg] >> numpy.uint64(pb+1)
    else:
        if p > q:
            mask = numpy.uint64((1 << pb) - (1 << (qb+1)))
        else:
            mask = numpy.uint64((1 << qb) - (1 << (pb+1)))
        n1 = bin(string[-1-pg]&mask).count('1')

    if n1 % 2:
        return -1
    else:
        return 1

def argunique(strs):
    def order(x, y):
        for i in range(y.size):
            if x[i] > y[i]:
                return 1
            elif y[i] > x[i]:
                return -1
        return 0
    def qsort_idx(idx):
        nstrs = len(idx)
        if nstrs <= 1:
            return idx
        else:
            ref = idx[-1]
            group_lt = []
            group_gt = []
            for i in idx[:-1]:
                c = order(strs[i], strs[ref])
                if c == -1:
                    group_lt.append(i)
                elif c == 1:
                    group_gt.append(i)
            return qsort_idx(group_lt) + [ref] + qsort_idx(group_gt)
    return qsort_idx(range(len(strs)))

def argunique_ctypes(strs):
    nstrs, nset = strs.shape

    sort_idx = numpy.empty(nstrs, dtype=numpy.uint64)

    strs = numpy.asarray(strs, order='C')
    sort_idx = numpy.asarray(sort_idx, order='C')

    nstrs_ = numpy.array([nstrs])

    libhci.argunique(strs.ctypes.data_as(ctypes.c_void_p), 
                     sort_idx.ctypes.data_as(ctypes.c_void_p), 
                     nstrs_.ctypes.data_as(ctypes.c_void_p), 
                     ctypes.c_int(nset))

    sort_idx = sort_idx[:nstrs_[0]]

    return sort_idx.tolist()

def str_diff(string0, string1):
    des_string0 = []
    cre_string0 = []
    nset = len(string0)
    off = 0
    for i in reversed(range(nset)):
        df = string0[i] ^ string1[i]
        des_string0.extend([x+off for x in find1(df & string0[i])])
        cre_string0.extend([x+off for x in find1(df & string1[i])])
        off += 64
    return des_string0, cre_string0

def excitation_level(string, nelec=None):
    nset = len(string)
    if nelec is None:
        nelec = 0
        for i in range(nset):
            nelec += bin(string[i]).count('1')

    g, b = nelec//64, nelec%64
    tn = nelec - bin(string[-1-g])[-b:].count('1')
    for s in string[nset-g:]:
        tn -= bin(s).count('1')
    return tn

def find1(s):
    return [i for i,x in enumerate(bin(s)[2:][::-1]) if x == '1']

def toggle_bit(s, place):
    g, b = place//64, place%64
    s[-1-g] ^= numpy.uint64(1<<b)
    return s

def select_strs_ctypes(myci, civec, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec):
    strs = civec._strs
    ndet = strs.shape[0]
    ndet, nset = strs.shape
    nset = nset // 2
    neleca, nelecb = nelec

    h1 = numpy.asarray(h1, order='C')  
    eri = numpy.asarray(eri, order='C')
    jk = numpy.asarray(jk, order='C')
    civec = numpy.asarray(civec, order='C')
    strs = numpy.asarray(strs, order='C')
    eri_sorted = numpy.asarray(eri_sorted, order='C')
    jk_sorted = numpy.asarray(jk_sorted, order='C')

    str_add = numpy.empty((0,strs.shape[1]), dtype=numpy.uint64)

    batch_size = max(1, 8 * 4 * neleca * nelecb * (norb-neleca) * (norb-nelecb))
    ndet_batch = int(myci.max_memory * 1024**2) // batch_size
    nbatches = ndet // ndet_batch + 1

    for i in range(nbatches):
        ndet_start = ndet_batch * i
        ndet_finish = min(ndet_batch * (i + 1), ndet)
        ndet_select_max = 4 * neleca * nelecb * (norb-neleca) * (norb-nelecb) * ndet_batch

        str_add_batch = numpy.empty((ndet_select_max, strs.shape[1]), dtype=numpy.uint64)
        n_str_add_batch = numpy.array([str_add_batch.shape[0]])

        str_add_batch = numpy.asarray(str_add_batch, order='C')

        libhci.select_strs(h1.ctypes.data_as(ctypes.c_void_p), 
                           eri.ctypes.data_as(ctypes.c_void_p), 
                           jk.ctypes.data_as(ctypes.c_void_p), 
                           eri_sorted.ctypes.data_as(ctypes.c_void_p), 
                           jk_sorted.ctypes.data_as(ctypes.c_void_p), 
                           ctypes.c_int(norb), 
                           ctypes.c_int(neleca), 
                           ctypes.c_int(nelecb), 
                           strs.ctypes.data_as(ctypes.c_void_p), 
                           civec.ctypes.data_as(ctypes.c_void_p), 
                           ctypes.c_ulonglong(ndet_start), 
                           ctypes.c_ulonglong(ndet_finish), 
                           ctypes.c_double(myci.select_cutoff),
                           str_add_batch.ctypes.data_as(ctypes.c_void_p),
                           n_str_add_batch.ctypes.data_as(ctypes.c_void_p))

        n_str_add_batch = n_str_add_batch[0]
        str_add_batch = str_add_batch[:n_str_add_batch]
        str_add = numpy.vstack((str_add, str_add_batch))

    str_add = numpy.asarray(str_add)
    return str_add

def enlarge_space(myci, civec, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec):
    if not isinstance(civec, (tuple, list)):
        civec = [civec]

    strs = civec[0]._strs

    nroots = len(civec)

    cidx = abs(civec[0]) > myci.ci_coeff_cutoff
    for p in range(1,nroots):
        cidx += abs(civec[p]) > myci.ci_coeff_cutoff

    strs = strs[cidx]

    ci_coeff = [as_SCIvector(c[cidx], strs) for c in civec]
 
    strs_new = strs.copy()

    for p in range(nroots):
        str_add = select_strs_ctypes(myci, ci_coeff[p], h1, eri, jk, eri_sorted, jk_sorted, norb, nelec)
        strs_new = numpy.vstack((strs, str_add))

    # Add strings together and remove duplicate strings
    tmp = numpy.ascontiguousarray(strs_new).view(numpy.dtype((numpy.void, strs_new.dtype.itemsize * strs_new.shape[1])))
    _, tmpidx = numpy.unique(tmp, return_index=True)

    new_ci = []
    for p in range(nroots):
        c = numpy.zeros(strs_new.shape[0])
        c[:ci_coeff[p].shape[0]] = ci_coeff[p]
        new_ci.append(c[tmpidx])

    strs_new = strs_new[tmpidx]

    return [as_SCIvector(ci, strs_new) for ci in new_ci]

def str2orblst(string, norb):
    occ = []
    vir = []
    nset = len(string)
    off = 0
    for k in reversed(range(nset)):
        s = string[k]
        occ.extend([x+off for x in find1(s)])
        for i in range(0, min(64, norb-off)): 
            if not (s & numpy.uint64(1<<i)):
                vir.append(i+off)
        off += 64
    return occ, vir

def orblst2str(lst, norb):
    nset = (norb+63) // 64
    string = numpy.zeros(nset, dtype=numpy.uint64)
    for i in lst:
        toggle_bit(string, i)
    return string

def kernel_float_space(myci, h1e, eri, norb, nelec, ci0=None,
                       tol=None, lindep=None, max_cycle=None, max_space=None,
                       nroots=None, davidson_only=None, max_iter=None,
                       max_memory=None, verbose=None, ecore=0, return_integrals=False, 
                       eri_sorted=None, jk=None, jk_sorted=None, **kwargs):
    if verbose is None:
        log = logger.Logger(myci.stdout, myci.verbose)
    elif isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(myci.stdout, verbose)
    if tol is None: tol = myci.conv_tol
    if lindep is None: lindep = myci.lindep
    if max_cycle is None: max_cycle = myci.max_cycle
    if max_space is None: max_space = myci.max_space
    if max_memory is None: max_memory = myci.max_memory
    if nroots is None: nroots = myci.nroots
    if max_iter is None: max_iter = myci.max_iter
    if myci.verbose >= logger.WARN:
        myci.check_sanity()

    log.info('\nStarting heat-bath CI algorithm...\n')
    log.info('Selection threshold:                  %8.5e',    myci.select_cutoff)
    log.info('CI coefficient cutoff:                %8.5e',    myci.ci_coeff_cutoff)
    log.info('Energy convergence tolerance:         %8.5e',    tol)
    log.info('Number of determinants tolerance:     %8.5e',    myci.conv_ndet_tol)
    log.info('Number of electrons:                  %s',       nelec)
    log.info('Number of orbitals:                   %3d',      norb)
    log.info('Number of roots:                      %3d',    nroots)

    nelec = direct_spin1._unpack_nelec(nelec, myci.spin)
    eri = ao2mo.restore(1, eri, norb)

    # Avoid resorting the integrals by storing them in memory
    eri = eri.ravel()

    if eri_sorted is None and jk is None and jk_sorted is None:
        log.debug("\nSorting two-electron integrals...")
        t_start = time.time()
        eri_sorted = abs(eri).argsort()[::-1]
        jk = eri.reshape([norb]*4)
        jk = jk - jk.transpose(2,1,0,3)
        jk = jk.ravel()
        jk_sorted = abs(jk).argsort()[::-1]
        t_current = time.time() - t_start
        log.debug('Timing for sorting the integrals: %10.3f', t_current)

    # Initial guess
    if ci0 is None:
        hf_str = numpy.hstack([orblst2str(range(nelec[0]), norb), orblst2str(range(nelec[1]), norb)]).reshape(1,-1)
        ci0 = [as_SCIvector(numpy.ones(1), hf_str)]
    else:
        assert(nroots == len(ci0))

    ci0 = myci.enlarge_space(ci0, h1e, eri, jk, eri_sorted, jk_sorted, norb, nelec)

    def hop(c):
        hc = myci.contract_2e((h1e, eri), as_SCIvector(c, ci_strs), norb, nelec, hdiag)
        return hc.ravel()
    precond = lambda x, e, *args: x/(hdiag-e+myci.level_shift)

    e_last = 0
    float_tol = 3e-4
    conv = False
    for icycle in range(max_iter):
        ci_strs = ci0[0]._strs
        float_tol = max(float_tol*.3, tol*1e2)
        log.info('\nMacroiteration %d', icycle)
        log.info('Number of CI configurations: %d', ci_strs.shape[0])
        hdiag = myci.make_hdiag(h1e, eri, ci_strs, norb, nelec)
        t_start = time.time()
        e, ci0 = myci.eig(hop, ci0, precond, tol=float_tol, lindep=lindep,
                          max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                          max_memory=max_memory, verbose=log, **kwargs)
        if not isinstance(ci0, (tuple, list)):
            ci0 = [ci0]
            e = [e]
        t_current = time.time() - t_start
        log.debug('Timing for solving the eigenvalue problem: %10.3f', t_current)
        ci0 = [as_SCIvector(c, ci_strs) for c in ci0]
        de, e_last = min(e)-e_last, min(e)
        log.info('Cycle %d  E = %s  dE = %.8g', icycle, numpy.array(e)+ecore, de)

        if abs(de) < tol*1e3:
            conv = True
            break

        last_ci0_size = float(len(ci_strs))
        t_start = time.time()
        ci0 = myci.enlarge_space(ci0, h1e, eri, jk, eri_sorted, jk_sorted, norb, nelec)
        t_current = time.time() - t_start
        log.debug('Timing for selecting configurations: %10.3f', t_current)
        if (((1 - myci.conv_ndet_tol) < len(ci0[0]._strs)/last_ci0_size < (1 + myci.conv_ndet_tol))):
            conv = True
            break

    ci_strs = ci0[0]._strs
    log.info('\nExtra CI in the final selected space')
    log.info('Number of CI configurations: %d', ci_strs.shape[0])
    hdiag = myci.make_hdiag(h1e, eri, ci_strs, norb, nelec)
    e, c = myci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                    max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                    max_memory=max_memory, verbose=log, **kwargs)
    if not isinstance(c, (tuple, list)):
        c = [c]
        e = [e]
    log.info('\nSelected CI  E = %s', numpy.array(e)+ecore)

    if (return_integrals):
        return (numpy.array(e)+ecore), [as_SCIvector(ci, ci_strs) for ci in c], eri_sorted, jk, jk_sorted
    else:
        return (numpy.array(e)+ecore), [as_SCIvector(ci, ci_strs) for ci in c]

def fix_spin(myci, shift=.2, ss=None, **kwargs):
    r'''If Selected CI solver cannot stick on spin eigenfunction, modify the solver by
    adding a shift on spin square operator

    .. math::

        (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

    Args:
        myci : An instance of :class:`SelectedCI`

    Kwargs:
        shift : float
            Level shift for states which have different spin
        ss : number
            S^2 expection value == s*(s+1)

    Returns
            A modified Selected CI object based on myci.
    '''
    if 'ss_value' in kwargs:
        sys.stderr.write('fix_spin_: kwarg "ss_value" will be removed in future release. '
                         'It was replaced by "ss"\n')
        ss_value = kwargs['ss_value']
    else:
        ss_value = ss

    def contract_2e(h1_h2, civec, norb, nelec, hdiag=None, **kwargs):
        if isinstance(nelec, (int, numpy.number)):
            sz = (nelec % 2) * .5
        else:
            sz = abs(nelec[0]-nelec[1]) * .5
        if ss_value is None:
            ss = sz*(sz+1)
        else:
            ss = ss_value

        h1, eri = h1_h2
        strs = civec._strs
        ndet = len(strs)
        if hdiag is None:
            hdiag = make_hdiag(h1, eri, strs, norb, nelec)
        ci1 = numpy.zeros_like(civec)
        ci2 = numpy.zeros_like(civec)

        h1 = numpy.asarray(h1, order='C')
        eri = numpy.asarray(eri, order='C')
        strs = numpy.asarray(strs, order='C')
        civec = numpy.asarray(civec, order='C')
        hdiag = numpy.asarray(hdiag, order='C')
        ci1 = numpy.asarray(ci1, order='C')
        ci2 = numpy.asarray(ci2, order='C')

        libhci.contract_h_c_ss_c(h1.ctypes.data_as(ctypes.c_void_p), 
                                 eri.ctypes.data_as(ctypes.c_void_p), 
                                 ctypes.c_int(norb), 
                                 ctypes.c_int(nelec[0]), 
                                 ctypes.c_int(nelec[1]), 
                                 strs.ctypes.data_as(ctypes.c_void_p), 
                                 civec.ctypes.data_as(ctypes.c_void_p), 
                                 hdiag.ctypes.data_as(ctypes.c_void_p), 
                                 ctypes.c_ulonglong(ndet), 
                                 ci1.ctypes.data_as(ctypes.c_void_p),
                                 ci2.ctypes.data_as(ctypes.c_void_p))

        if ss < sz*(sz+1)+.1:
# (S^2-ss)|Psi> to shift state other than the lowest state
            ci2 -= ss * civec
        else:
# (S^2-ss)^2|Psi> to shift states except the given spin.
# It still relies on the quality of initial guess
            tmp = ci2.copy()
            tmp -= ss * civec
            ci2 = -ss * tmp
            ci2 += myci.contract_ss(as_SCIvector_if_not(tmp, strs), norb, nelec)
            tmp = None
        ci2 *= shift
        ci1 += ci2

        return as_SCIvector_if_not(ci1, strs)

    myci.contract_2e = contract_2e
    return myci

def to_fci(civec, norb, nelec, root=0):
    assert(norb <= 64)
    neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = len(civec[root])
    fcivec = numpy.zeros((na,nb))
    for idet, (stra, strb) in enumerate(civec[root]._strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        fcivec[ka,kb] = civec[root][idet]
    return fcivec

def from_fci(fcivec, ci_strs, norb, nelec):
    neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    fcivec = fcivec.reshape(na,nb)
    #ta = [excitation_level(s, neleca) for s in strsa.reshape(-1,1)]
    #tb = [excitation_level(s, nelecb) for s in strsb.reshape(-1,1)]
    ndet = len(ci_strs)
    civec = numpy.zeros(ndet)
    for idet, (stra, strb) in enumerate(ci_strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        civec[idet] = fcivec[ka,kb]
    return as_SCIvector(civec, ci_strs)

def make_rdm12s(civec, norb, nelec):
    '''Spin orbital 1- and 2-particle reduced density matrices (aa, bb, aaaa, aabb, bbbb)
    '''
    strs = civec._strs
    ndet = len(strs)
    rdm1a = numpy.zeros(norb*norb)
    rdm1b = numpy.zeros(norb*norb)
    rdm2aa = numpy.zeros(norb*norb*norb*norb)
    rdm2ab = numpy.zeros(norb*norb*norb*norb)
    rdm2bb = numpy.zeros(norb*norb*norb*norb)

    civec = numpy.asarray(civec, order='C')
    strs = numpy.asarray(strs, order='C')
    rdm1a  = numpy.asarray(rdm1a, order='C')
    rdm1b  = numpy.asarray(rdm1b, order='C')
    rdm2aa = numpy.asarray(rdm2aa, order='C')
    rdm2ab = numpy.asarray(rdm2ab, order='C')
    rdm2bb = numpy.asarray(rdm2bb, order='C')

    # Compute 1- and 2-RDMs
    libhci.compute_rdm12s(ctypes.c_int(norb), 
                          ctypes.c_int(nelec[0]), 
                          ctypes.c_int(nelec[1]), 
                          strs.ctypes.data_as(ctypes.c_void_p), 
                          civec.ctypes.data_as(ctypes.c_void_p), 
                          ctypes.c_ulonglong(ndet), 
                          rdm1a.ctypes.data_as(ctypes.c_void_p),
                          rdm1b.ctypes.data_as(ctypes.c_void_p),
                          rdm2aa.ctypes.data_as(ctypes.c_void_p),
                          rdm2ab.ctypes.data_as(ctypes.c_void_p),
                          rdm2bb.ctypes.data_as(ctypes.c_void_p))

    rdm1a = rdm1a.reshape([norb]*2)
    rdm1b = rdm1b.reshape([norb]*2)
    rdm2aa = rdm2aa.reshape([norb]*4)
    rdm2ab = rdm2ab.reshape([norb]*4)
    rdm2bb = rdm2bb.reshape([norb]*4)

    # Sort 2-RDM into chemists' notation: <p_1 q_2|r_1 s_2> -> (p_1 r_1| q_2 s_2)
    rdm2aa = rdm2aa.transpose(0,2,1,3)
    rdm2ab = rdm2ab.transpose(0,2,1,3)
    rdm2bb = rdm2bb.transpose(0,2,1,3)

    return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb)

class SelectedCI(direct_spin1.FCISolver):
    def __init__(self, mol=None):
        direct_spin1.FCISolver.__init__(self, mol)
        self.ci_coeff_cutoff = .5e-3
        self.select_cutoff = .5e-3
        self.conv_tol = 1e-9
        self.conv_ndet_tol = 0.001
        self.nroots = 1
        self.max_iter = 10
        # Maximum memory in MB for storing lists of selected strings
        self.max_memory = 1000

##################################################
# don't modify the following attributes, they are not input options
        #self.converged = False
        #self.ci = None
        self._strs = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        direct_spin1.FCISolver.dump_flags(self, verbose)
        logger.info(self, 'ci_coeff_cutoff %g', self.ci_coeff_cutoff)
        logger.info(self, 'select_cutoff   %g', self.select_cutoff)

    # define absorb_h1e for compatibility to other FCI solver
    def absorb_h1e(h1, eri, *args, **kwargs):
        return (h1, eri)

    def contract_2e(self, h1_h2, civec, norb, nelec, hdiag=None, **kwargs):
        if getattr(civec, '_strs', None) is not None:
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)
        return contract_2e_ctypes(h1_h2, civec, norb, nelec, hdiag, **kwargs)
#        return contract_2e(h1_h2, civec, norb, nelec, hdiag, **kwargs)

    def contract_ss(self, civec, norb, nelec):
        if getattr(civec, '_strs', None) is not None:
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)
        return contract_ss(civec, norb, nelec)

    def spin_square(self, civec, norb, nelec):
        if getattr(civec, '_strs', None) is not None:
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)
        return spin_square(civec, norb, nelec)

    def make_hdiag(self, h1e, eri, strs, norb, nelec):
        return make_hdiag(h1e, eri, strs, norb, nelec)

    def to_fci(self, civec, norb, nelec):

        if getattr(civec, '_strs', None) is not None:
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)

        return to_fci(civec, norb, nelec)

    def make_rdm12s(self, civec, norb, nelec):

        if getattr(civec, '_strs', None) is not None:
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)

        return make_rdm12s(civec, norb, nelec)

    enlarge_space = enlarge_space
    kernel = kernel_float_space

SCI = SelectedCI


class _SCIvector(numpy.ndarray):
    def __array_finalize__(self, obj):
        self._strs = getattr(obj, '_strs', None)

def as_SCIvector(civec, ci_strs):
    civec = civec.view(_SCIvector)
    civec._strs = ci_strs
    return civec

def as_SCIvector_if_not(civec, ci_strs):
    if getattr(civec, '_strs', None) is None:
        civec = as_SCIvector(civec, ci_strs)
    return civec


if __name__ == '__main__':
    numpy.random.seed(3)
    strs = (numpy.random.random((14,3)) * 4).astype(numpy.uint64)
    print(strs)
    print(argunique(strs))

    norb = 6
    nelec = 3,3
    hf_str = numpy.hstack([orblst2str(range(nelec[0]), norb),
                           orblst2str(range(nelec[1]), norb)]).reshape(1,-1)
    numpy.random.seed(3)
    h1 = numpy.random.random([norb]*2)**4 * 1e-2
    h1 = h1 + h1.T
    eri = numpy.random.random([norb]*4)**4 * 1e-2
    eri = eri + eri.transpose(0,1,3,2)
    eri = eri + eri.transpose(1,0,2,3)
    eri = eri + eri.transpose(2,3,0,1)
    eri_sorted = abs(eri).argsort()[::-1]
    jk = eri.reshape([norb]*4)
    jk = jk - jk.transpose(2,1,0,3)
    jk = jk.ravel()
    jk_sorted = abs(jk).argsort()[::-1]
    ci1 = [as_SCIvector(numpy.ones(1), hf_str)]

    myci = SelectedCI()
    myci.select_cutoff = .001
    myci.ci_coeff_cutoff = .001

    ci2 = enlarge_space(myci, ci1, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec)
    print(len(ci2[0]))

    ci2 = enlarge_space(myci, ci1, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec)
    numpy.random.seed(1)
    ci3 = numpy.random.random(ci2[0].size)
    ci3 *= 1./numpy.linalg.norm(ci3)
    ci3 = [ci3]
    ci3 = enlarge_space(myci, ci2, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec)

    efci = direct_spin1.kernel(h1, eri, norb, nelec, verbose=5)[0]

    ci4 = contract_2e_ctypes((h1, eri), ci3[0], norb, nelec)

    fci3 = to_fci(ci3, norb, nelec)
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
    fci4 = direct_spin1.contract_2e(h2e, fci3, norb, nelec)
    fci4 = from_fci(fci4, ci3[0]._strs, norb, nelec)
    print(abs(ci4-fci4).sum())

    e = myci.kernel(h1, eri, norb, nelec, verbose=5)[0]
    print(e, efci)
