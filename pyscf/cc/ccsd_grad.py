#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import numpy
from pyscf import lib
from functools import reduce
from pyscf.lib import logger
from pyscf import gto
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_rdm
from pyscf.scf import rhf_grad
from pyscf.scf import cphf


def IX_intermediates(mycc, t1, t2, l1, l2, eris=None, d1=None, d2=None):
    if eris is None:
        eris = mycc.ao2mo()
    if d1 is None:
        d1 = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    if d2 is None:
        fd2intermediate = lib.H5TmpFile()
        d2 = ccsd_rdm.gamma2_outcore(mycc, t1, t2, l1, l2, fd2intermediate)
    dvovo, dvvvv, doooo, dvvoo, dvoov, dvvov, dvovv, dvooo = d2

    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    nvir_pair = nvir * (nvir+1) //2
    fswap = lib.H5TmpFile()

    d_vovo = _cp(dvovo) + _cp(dvoov).transpose(0,1,3,2)
    d_vovo = lib.transpose_sum(d_vovo.reshape(nov,nov)).reshape(nvir,nocc,nvir,nocc)
    fswap['d_voov'] = d_vovo.transpose(0,1,3,2)
    d_vovo = None

    d_vvoo = _cp(dvvoo)
    d_vvoo = d_vvoo + d_vvoo.transpose(1,0,3,2)
    fswap['d_vvoo'] = d_vvoo
    d_woo = d_vvoo[numpy.tril_indices(nvir)]
    for i in range(1, nvir):
        off = i*(i+1)//2
        d_woo[off:off+i] += d_vvoo[:i,i]
    fswap['d_woo'] = d_woo
    d_vvoo = None

# Note Ioo, Ivv are not hermitian
    Ioo = numpy.zeros((nocc,nocc))
    Ivv = numpy.zeros((nvir,nvir))
    Ivo = numpy.zeros((nvir,nocc))
    Xvo = numpy.zeros((nvir,nocc))

    eris_oooo = _cp(eris.oooo)
    eris_vooo = _cp(eris.vooo)
    d_oooo = _cp(doooo)
    d_oooo = _cp(d_oooo + d_oooo.transpose(1,0,2,3))
    Ioo += lib.einsum('jmlk,imlk->ij', d_oooo, eris_oooo) * 2
    Xvo += lib.einsum('aklj,iklj->ai', eris_vooo, d_oooo) * 2
    Xvo += numpy.einsum('kj,aikj->ai', doo, eris_vooo) * 4
    Xvo -= numpy.einsum('kj,ajik->ai', doo+doo.T, eris_vooo)

    d_vooo = _cp(dvooo)
    Ivv += lib.einsum('bkij,akij->ab', d_vooo, eris_vooo)
    Ivo += lib.einsum('akjl,ikjl->ai', d_vooo, eris_oooo)
    eris_oooo = eris_vooo = d_oooo = d_vooo = None

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc*nvir**2 * 4
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/unit)))
    log.debug1('IX_intermediates pass 1: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)/blksize))
    fswap.create_dataset('eris_wvo', (nvir_pair,nvir,nocc), 'f8',
                         chunks=(nvir_pair,blksize,nocc))
    fswap.create_dataset('d_wvo', (nvir_pair,nvir,nocc), 'f8',
                         chunks=(nvir_pair,blksize,nocc))
    dvv1 = dvv + dvv.T

    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_vooo = _cp(eris.vooo[p0:p1])
        eris_vvoo = _cp(eris.vvoo[p0:p1])

        d_vooo = _cp(dvooo[p0:p1])
        Ioo += lib.einsum('ajkl,aikl->ij', d_vooo, eris_vooo)
        Xvo += lib.einsum('bikj,bakj->ai', d_vooo, eris_vvoo)
        d_vooo = d_vooo + d_vooo.transpose(0,1,3,2)
        eris_voov = _cp(eris.voov[p0:p1])
        Ioo += lib.einsum('aklj,akli->ij', d_vooo, eris_vooo)
        Xvo += lib.einsum('bkji,bkja->ai', d_vooo, eris_voov)

        d_vvoo = _cp(fswap['d_vvoo'][p0:p1])
        Ioo += lib.einsum('bakj,baki->ij', d_vvoo, eris_vvoo)
        Ivv += lib.einsum('cbij,caij->ab', d_vvoo, eris_vvoo)
        Ivo += lib.einsum('bakj,bikj->ai', d_vvoo, eris_vooo)
        d_vvoo = None

        d_voov = _cp(fswap['d_voov'][p0:p1])
        Ivo += lib.einsum('bjka,bjki->ai', d_voov, eris_vooo)
        Ioo += lib.einsum('ajkb,aikb->ij', d_voov, eris_voov)
        Ivv += lib.einsum('cjib,cjia->ab', d_voov, eris_voov)
        eris_vvoo = eris_vooo = eris_voov = None

        # tril part of (d_vovv + d_vovv.transpose(0,1,3,2))
        d_vovv = _cp(dvovv[p0:p1])
        c_vovv = _ccsd.precontract(d_vovv.reshape(-1,nvir,nvir))
        fswap['d_wvo'][:,p0:p1] = c_vovv.reshape(p1-p0,nocc,nvir_pair).transpose(2,0,1)
        c_vovv = None

        eris_vox = _cp(eris.vovv[p0:p1])
        fswap['eris_wvo'][:,p0:p1] = eris_vox.reshape(p1-p0,nocc,nvir_pair).transpose(2,0,1)

        d_vovv = d_vovv + d_vovv.transpose(0,1,3,2)
        Ivo += lib.einsum('cjab,cjib->ai', d_vovv, _cp(eris.voov[p0:p1]))

        eris_vovv = lib.unpack_tril(eris_vox.reshape(-1,nvir_pair))
        eris_vox = None
        eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)
        Ivv += lib.einsum('cidb,cida->ab', d_vovv, eris_vovv)
        Xvo[p0:p1] += numpy.einsum('cb,aicb->ai', dvv, eris_vovv) * 4
        Xvo -= numpy.einsum('cb,ciba->ai', dvv1[p0:p1], eris_vovv)
        Xvo += lib.einsum('bjic,bjca->ai', d_voov, eris_vovv)
        d_vovv = d_voov = eris_vovv = None

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc*nvir**2 + nvir**3*2.5
    blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/unit))
    log.debug1('IX_intermediates pass 2: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)/blksize))
    for p0, p1 in lib.prange(0, nvir, blksize):
        off0 = p0*(p0+1)//2
        off1 = p1*(p1+1)//2
        d_vvvv = _cp(dvvvv[off0:off1]) * 4
        for i in range(p0, p1):
            d_vvvv[i*(i+1)//2+i-off0] *= .5
        d_vvvv = lib.unpack_tril(d_vvvv)
        eris_vvvv = lib.unpack_tril(_cp(eris.vvvv[off0:off1]))
        Ivv += lib.einsum('xcb,xca->ab', d_vvvv, eris_vvvv) * 2

        d_vvvo = _cp(fswap['d_wvo'][off0:off1])
        Xvo += lib.einsum('xci,xca->ai', d_vvvo, eris_vvvv)
        eris_vvvv = None

        buf = _cp(eris.vvoo[p0:p1,:p1])
        eris_vvoo = numpy.empty((off1-off0,nocc,nocc))
        q1 = 0
        for i in range(p1-p0):
            q0, q1 = q1, q1 + p0+i+1
            eris_vvoo[q0:q1] = buf[i,:p0+i+1]
        Ivo += lib.einsum('xaj,xji->ai', d_vvvo, eris_vvoo)
        buf = eris_vvoo = None

        eris_vvvo = _cp(fswap['eris_wvo'][off0:off1])
        Ivo += lib.einsum('xca,xci->ai', d_vvvv, eris_vvvo) * 2
        d_vvvv = None
        Ioo += lib.einsum('xcj,xci->ij', d_vvvo, eris_vvvo)
        Ivv += lib.einsum('xai,xbi->ab', d_vvvo, eris_vvvo)

        d_vvoo = _cp(fswap['d_woo'][off0:off1])
        Xvo += lib.einsum('xij,xaj->ai', d_vvoo, eris_vvvo)
        eris_vvvo = d_vvvo = d_vvoo = None

    Ioo *= -1
    Ivv *= -1
    Ivo *= -1
    Xvo += Ivo
    return Ioo, Ivv, Ivo, Xvo


def response_dm1(mycc, t1, t2, l1, l2, eris=None, IX=None):
    if eris is None:
        eris = mycc.ao2mo()
    if IX is None:
        Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris)
    else:
        Ioo, Ivv, Ivo, Xvo = IX
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc**2*nvir)))
    def fvind(x):
        x = x.reshape(Xvo.shape)
        if eris is None:
            mo_coeff = mycc.mo_coeff
            dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
            dm = (dm + dm.T) * 2
            v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, mycc._scf.get_veff(mol, dm),
                                   mo_coeff[:,:nocc]))
        else:
            v = numpy.zeros((nvir,nocc))
            for p0, p1 in lib.prange(0, nvir, blksize):
                eris_voov = _cp(eris.voov[p0:p1])
                v[p0:p1] += numpy.einsum('aijb,bj->ai', eris_voov, x) * 4
                v -= numpy.einsum('bija,bj->ai', eris_voov, x[p0:p1])
                eris_voov = None
                v -= numpy.einsum('baij,bj->ai', _cp(eris.vvoo[p0:p1]), x[p0:p1])
        return v
    mo_energy = eris.fock.diagonal()
    mo_occ = numpy.zeros_like(mo_energy)
    mo_occ[:nocc] = 2
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1


#
# Note: only works with canonical orbitals
# Non-canonical formula refers to JCP, 95, 2639
#
def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
           mf_grad=None, verbose=logger.INFO):
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if eris is None: eris = mycc.ao2mo()
    if mf_grad is None:
        mf_grad = rhf_grad.Gradients(mycc._scf)

    log = logger.new_logger(mycc, verbose)
    time0 = time.clock(), time.time()
    mol = mycc.mol
    if mycc.frozen is not 0:
        raise NotImplementedError('frozen orbital ccsd_grad')
    moidx = ccsd.get_moidx(mycc)
    mo_coeff = mycc.mo_coeff[:,moidx]  #FIXME: ensure mycc.mo_coeff is canonical orbital
    mo_energy = eris.fock.diagonal()
    nocc, nvir = t1.shape
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    log.debug('Build ccsd rdm1 intermediates')
    d1 = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    time1 = log.timer('rdm1 intermediates', *time0)

    log.debug('Build ccsd rdm2 intermediates')
    fd2intermediate = lib.H5TmpFile()
    d2 = ccsd_rdm.gamma2_outcore(mycc, t1, t2, l1, l2, fd2intermediate)
    time1 = log.timer('rdm2 intermediates', *time1)
    log.debug('Build ccsd response_rdm1')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris, d1, d2)
    time1 = log.timer('response_rdm1 intermediates', *time1)

    dm1mo = response_dm1(mycc, t1, t2, l1, l2, eris, (Ioo, Ivv, Ivo, Xvo))
    dm1mo[:nocc,:nocc] = doo + doo.T
    dm1mo[nocc:,nocc:] = dvv + dvv.T
    dm1ao = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    im1 = numpy.zeros_like(dm1mo)
    im1[:nocc,:nocc] = Ioo
    im1[nocc:,nocc:] = Ivv
    im1[nocc:,:nocc] = Ivo
    im1[:nocc,nocc:] = Ivo.T
    im1 = reduce(numpy.dot, (mo_coeff, im1, mo_coeff.T))
    time1 = log.timer('response_rdm1', *time1)

    log.debug('symmetrized rdm2 and MO->AO transformation')
# Roughly, dm2*2 is computed. *2 in _rdm2_mo2ao, *2 in _load_block_tril
    fdm2 = lib.H5TmpFile()
    dm1_with_hf = dm1mo.copy()
    for i in range(nocc):  # HF 2pdm ~ 4(ij)(kl)-2(il)(jk), diagonal+1 because of 4*dm2
        dm1_with_hf[i,i] += 1
    _rdm2_mo2ao(mycc, d2, dm1_with_hf, mo_coeff, fdm2)
    time1 = log.timer('MO->AO transformation', *time1)
    fd2intermediate = None

#TODO: pass hf_grad object to compute h1 and s1
    log.debug('h1 and JK1')
    h1 = mf_grad.get_hcore(mol)
    s1 = mf_grad.get_ovlp(mol)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf4sij = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1ao+dm1ao.T), p1))
    time1 = log.timer('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    hf_dm1 = mycc._scf.make_rdm1(mycc._scf.mo_coeff, mycc._scf.mo_occ)
    dm1ao += hf_dm1
    zeta += mf_grad.make_rdm1e(mycc._scf.mo_energy, mycc._scf.mo_coeff,
                               mycc._scf.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*1e6/8/(nao**3*2.5)))
    ioblksize = fdm2['dm2'].chunks[1]
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k]  = numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1])
# h[1] \dot DM, *2 for +c.c.,  contribute to f1
        h1ao = mf_grad._grad_rinv(mol, ia)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm1ao)
        de[k] += numpy.einsum('xji,ij->x', h1ao, dm1ao)
# -s[1]*e \dot DM,  contribute to f1
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
        de[k] -= numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1])
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf4sij[p0:p1]) * 2

# 2e AO integrals dot 2pdm
        ip0 = p0
        for b0, b1, nf in shell_prange(mol, shl0, shl1, blksize):
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=(b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas))
            eri1 = eri1.reshape(3,nf,nao,-1)
            dm2buf = numpy.empty((nf,nao,nao_pair))
            for i0, i1 in lib.prange(0, nao_pair, ioblksize):
                _load_block_tril(fdm2['dm2'], ip0, ip0+nf, i0, i1, dm2buf[:,:,i0:i1])
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
            eri1 = dm2buf = None
            ip0 += nf
        log.debug('grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer('grad of atom %d'%ia, *time1)

    de += rhf_grad.grad_nuc(mol)
    log.note('--------------- CCSD gradients ---------------')
    log.note('           x                y                z')
    for k, ia in enumerate(atmlst):
        log.note('%d %s  %15.9f  %15.9f  %15.9f', ia, mol.atom_symbol(ia), *de[k])
    log.note('----------------------------------------------')
    log.timer('CCSD gradients', *time0)
    fdm2 = None
    return de


def as_scanner(cc):
    '''Generating a scanner/solver for CCSD PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CCSD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CCSD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, cc
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> cc_scanner = cc.CCSD(scf.RHF(mol)).as_scanner()
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    import copy
    logger.info(cc, 'Set nuclear gradients of %s as a scanner', cc.__class__)
    cc = copy.copy(cc)
    cc._scf = cc._scf.as_scanner()
    def solver(mol):
        mf_scanner = cc._scf
        mf_scanner(mol)
        cc.mol = mol
        cc.mo_coeff = mf_scanner.mo_coeff
        cc.mo_occ = mf_scanner.mo_occ
        eris = cc.ao2mo(cc.mo_coeff)
        mf_grad = cc._scf.nuc_grad_method()
        cc.kernel(cc.t1, cc.t2, eris=eris)
        cc.solve_lambda(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris)
        de = kernel(cc, cc.t1, cc.t2, cc.l1, cc.l2, eris=eris, mf_grad=mf_grad)
        return cc.e_tot, de
    return solver


def shell_prange(mol, start, stop, blksize):
    nao = 0
    ib0 = start
    for ib in range(start, stop):
        now = (mol.bas_angular(ib)*2+1) * mol.bas_nctr(ib)
        nao += now
        if nao > blksize and nao > now:
            yield (ib0, ib, nao-now)
            ib0 = ib
            nao = now
    yield (ib0, stop, nao)

def _rdm2_mo2ao(mycc, d2, dm1, mo_coeff, fsave=None):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    time1 = time.clock(), time.time()
    if fsave is None:
        incore = True
        fsave = lib.H5TmpFile()
    else:
        incore = False
    dvovo, dvvvv, doooo, dvvoo, dvoov, dvvov, dvovv, dvooo = d2

    nvir, nocc = dvovo.shape[:2]
    nov = nocc * nvir
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    nvir_pair = nvir * (nvir+1) //2
    mo_coeff = numpy.asarray(mo_coeff, order='F')
    def _trans(vin, orbs_slice, out=None):
        nrow = vin.shape[0]
        if out is None:
            out = numpy.empty((nrow,nao_pair))
        fdrv = getattr(_ccsd.libcc, 'AO2MOnr_e2_drv')
        pao_loc = ctypes.POINTER(ctypes.c_void_p)()
        fdrv(_ccsd.libcc.AO2MOtranse2_nr_s1, _ccsd.libcc.CCmmm_transpose_sum,
             out.ctypes.data_as(ctypes.c_void_p),
             vin.ctypes.data_as(ctypes.c_void_p),
             mo_coeff.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(nrow), ctypes.c_int(nao),
             (ctypes.c_int*4)(*orbs_slice), pao_loc, ctypes.c_int(0))
        return out

    fswap = lib.H5TmpFile()
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*1e6/8/(nao_pair+nmo**2))
    blksize = min(nvir_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset('v', (nao_pair,nvir_pair), 'f8', chunks=(nao_pair,blksize))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        fswap['v'][:,p0:p1] = _trans(lib.unpack_tril(_cp(dvvvv[p0:p1])),
                                     (nocc,nmo,nocc,nmo)).T
    time1 = log.timer_debug1('_rdm2_mo2ao pass 1', *time1)

# transform dm2_ij to get lower triangular (dm2+dm2.transpose(0,1,3,2))
    blksize = int(max_memory*1e6/8/(nao_pair+nmo**2))
    blksize = min(nao_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset('o', (nmo,nocc,nao_pair), 'f8', chunks=(nmo,nocc,blksize))
    buf1 = numpy.zeros((nocc,nocc,nmo,nmo))
    buf1[:,:,:nocc,:nocc] = doooo
    buf1[:,:,nocc:,nocc:] = _cp(dvvoo).transpose(2,3,0,1)
    for i in range(nocc):
        buf1[i,i,:,:] += dm1
        buf1[:,i,i,:] -= dm1[:nocc] * .5
    buf1 = _trans(buf1.reshape(nocc**2,-1), (0,nmo,0,nmo))
    fswap['o'][:nocc] = buf1.reshape(nocc,nocc,nao_pair)
    for p0, p1 in lib.prange(nocc, nmo, nocc):
        buf1 = numpy.zeros((p1-p0,nocc,nmo,nmo))
        buf1[:,:,:nocc,:nocc] = dvooo[p0-nocc:p1-nocc]
        buf1[:,:,:nocc,nocc:] = dvoov[p0-nocc:p1-nocc]
        buf1[:,:,nocc:,:nocc] = dvovo[p0-nocc:p1-nocc]
        buf1[:,:,nocc:,nocc:] = dvovv[p0-nocc:p1-nocc]
        for i in range(nocc):
            buf1[:,i,i,:] -= dm1[p0:p1] * .5
        buf1 = _trans(buf1.reshape((p1-p0)*nocc,-1), (0,nmo,0,nmo))
        fswap['o'][p0:p1] = buf1.reshape(p1-p0,nocc,nao_pair)
    time1 = log.timer_debug1('_rdm2_mo2ao pass 2', *time1)

# transform dm2_kl then dm2 + dm2.transpose(2,3,0,1)
    gsave = fsave.create_dataset('dm2', (nao_pair,nao_pair), 'f8', chunks=(nao_pair,blksize))
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = numpy.zeros((p1-p0,nmo,nmo))
        buf1[:,nocc:,nocc:] = lib.unpack_tril(_cp(fswap['v'][p0:p1]))
        buf1[:,:,:nocc] = fswap['o'][:,:,p0:p1].transpose(2,0,1)
        buf2 = _trans(buf1, (0,nmo,0,nmo))
        ic = 0
        idx = diagidx[diagidx<p1]
        if p0 > 0:
            buf1 = _cp(gsave[:p0,p0:p1])
            buf1[:p0,:p1-p0] += buf2[:p1-p0,:p0].T
            buf2[:p1-p0,:p0] = buf1[:p0,:p1-p0].T
            buf1[:,idx[p0<=idx]-p0] *= .5
            gsave[:p0,p0:p1] = buf1
        lib.transpose_sum(buf2[:,p0:p1], inplace=True)
        buf2[:,idx] *= .5
        for i0, i1 in lib.prange(0, nao_pair, blksize):
            gsave[p0:p1,i0:i1] = buf2[:,i0:i1]
    time1 = log.timer_debug1('_rdm2_mo2ao pass 3', *time1)
    if incore:
        return fsave['dm2'].value
    else:
        return fsave

#
# .
# . .
# ----+             -----------
# ----|-+       =>  -----------
# . . | | .
# . . | | . .
#
def _load_block_tril(dat, row0, row1, col0, col1, out=None):
    shape = dat.shape
    nd = int(numpy.sqrt(shape[0]*2))
    if out is None:
        out = numpy.empty((row1-row0,nd,col1-col0)+shape[2:])
    dat1 = dat[row0*(row0+1)//2:row1*(row1+1)//2,col0:col1]
    p1 = 0
    for i in range(row0, row1):
        p0, p1 = p1, p1 + i+1
        out[i-row0,:i+1] = dat1[p0:p1]
        for j in range(row0, i):
            out[j-row0,i] = out[i-row0,j]
    for i in range(row1, nd):
        i2 = i*(i+1)//2
        out[:,i] = dat[i2+row0:i2+row1,col0:col1]
    return out

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import grad

    mol = gto.M()
    mf = scf.RHF(mol)

    mycc = ccsd.CCSD(mf)

    numpy.random.seed(2)
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
    eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
    fock0 = numpy.random.random((nmo,nmo))
    fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*20
    t1 = numpy.random.random((nocc,nvir))
    t2 = numpy.random.random((nocc,nocc,nvir,nvir))
    t2 = t2 + t2.transpose(1,0,3,2)
    l1 = numpy.random.random((nocc,nvir))
    l2 = numpy.random.random((nocc,nocc,nvir,nvir))
    l2 = l2 + l2.transpose(1,0,3,2)

    h1 = fock0 - (numpy.einsum('kkpq->pq', eri0[:nocc,:nocc])*2
                - numpy.einsum('pkkq->pq', eri0[:,:nocc,:nocc]))
    eris = lambda:None
    idx = numpy.tril_indices(nvir)
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.vooo = eri0[nocc:,:nocc,:nocc,:nocc].copy()
    eris.voov = eri0[nocc:,:nocc,:nocc,nocc:].copy()
    eris.vvoo = eri0[nocc:,nocc:,:nocc,:nocc].copy()
    eris.vovv = eri0[nocc:,:nocc,nocc:,nocc:]
    eris.vovv = eris.vovv[:,:,idx[0],idx[1]].copy()
    eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:]
    eris.vvvv = eris.vvvv[idx[0],idx[1]][:,idx[0],idx[1]].copy()
    eris.fock = fock0

    print('-----------------------------------')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris)
    numpy.random.seed(1)
    h1 = numpy.random.random((nmo,nmo))
    h1 = h1 + h1.T
    print(numpy.einsum('ij,ij', h1[:nocc,:nocc], Ioo) - 2613213.0346526774)
    print(numpy.einsum('ab,ab', h1[nocc:,nocc:], Ivv) - 6873038.9907923322)
    print(numpy.einsum('ai,ai', h1[nocc:,:nocc], Ivo) - 4353360.4241635408)
    print(numpy.einsum('ai,ai', h1[nocc:,:nocc], Xvo) - 203575.42337558540)
    dm1 = response_dm1(mycc, t1, t2, l1, l2, eris)
    print(numpy.einsum('pq,pq', h1[nocc:,:nocc], dm1[nocc:,:nocc])--486.638981725713393)

    fd2intermediate = lib.H5TmpFile()
    d2 = ccsd_rdm.gamma2_outcore(mycc, t1, t2, l1, l2, fd2intermediate)
    dm1 = numpy.zeros((nmo,nmo))
    mo_coeff = numpy.random.random((nmo,nmo)) - .5
    dm2 = _rdm2_mo2ao(mycc, d2, dm1, mo_coeff)
    print(lib.finger(dm2) - -2279.6732000822421)

    print('-----------------------------------')
    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    ehf = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.max_memory = 1
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
    print('gcc')
    print(g1)
#[[ 0   0                1.00950925e-02]
# [ 0   2.28063426e-02  -5.04754623e-03]
# [ 0  -2.28063426e-02  -5.04754623e-03]]

    lib.parameters.BOHR = 1
    r = 1.76#.748
    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % r,
        basis = '631g')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()
    ghf = grad.RHF(mf).grad()
    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
    print('gcc')
    print(g1)
#[[ 0.          0.         -0.07080036]
# [ 0.          0.          0.07080036]]
