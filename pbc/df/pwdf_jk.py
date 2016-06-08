#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
JK with analytic Fourier transformation
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools


def get_j_kpts(pwdf, cell, dm_kpts, hermi=1, vhfopt_or_mf=None,
               kpts=numpy.zeros((1,3)), kpt_band=None, jkops=None):
    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = (time.clock(), time.time())
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
    ngs = numpy.prod(pwdf.gs*2+1)
    vR = numpy.zeros(ngs)
    vI = numpy.zeros(ngs)
    for k, kpt in enumerate(kpts):
        kptii = numpy.asarray((kpt,kpt))
        dm = dm_kpts[k]

        p0 = 0
        for pqkR, pqkI, coulG in pwdf.pw_loop(cell, pwdf.gs, kptii, max_memory):
            nG = len(coulG)
            rhoR = numpy.dot(dm.ravel(), pqkR.reshape(nao**2,-1))
            rhoI = numpy.dot(dm.ravel(), pqkI.reshape(nao**2,-1))
            vR[p0:p0+nG] += rhoR * coulG
            vI[p0:p0+nG] += rhoI * coulG
            p0 += nG
        pqkR = pqkI = coulG = None
    t1 = log.timer('get_j pass 1 to compute J(G)', *t1)

    if kpt_band is None:
        kpts_lst = kpts
    else:
        kpts_lst = numpy.reshape(kpt_band, (-1,3))

    vj_kpts = []
    for k, kpt in enumerate(kpts_lst):
        kptii = numpy.asarray((kpt,kpt))
        vjR = 0
        vjI = 0
        p0 = 0
        for pqkR, pqkI, coulG in pwdf.pw_loop(cell, pwdf.gs, kptii, max_memory):
            nG = len(coulG)
            vjR += numpy.dot(pqkR, vR[p0:p0+nG])
            vjR += numpy.dot(pqkI, vI[p0:p0+nG])
            if abs(kpt).sum() > 1e-9:  # if not gamma point
                vjI += numpy.dot(pqkI, vR[p0:p0+nG])
                vjI -= numpy.dot(pqkR, vI[p0:p0+nG])
            p0 += nG
        pqkR = pqkI = coulG = None

        if abs(kpt).sum() < 1e-9:  # gamma point
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj_kpts.append(vj.reshape(nao,nao))
    t1 = log.timer('get_j pass 2', *t1)

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        return vj_kpts[0]
    else:
        return vj_kpts

def get_k_kpts(pwdf, cell, dm_kpts, hermi=1, vhfopt_or_mf=None,
               kpts=numpy.zeros((1,3)), kpt_band=None):
    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = (time.clock(), time.time())
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    def make_k1k2(kpti, kptj, dm):
        kptij = numpy.asarray((kpti,kptj))
        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kptj-kpti, True, pwdf,
                                      pwdf.gs) / cell.vol
        else:
            vkcoulG = tools.get_coulG(cell, kptj-kpti, True, vhfopt_or_mf,
                                      pwdf.gs) / cell.vol
        vjR = 0
        vjI = 0
        p0 = 0
        for pqkR, pqkI, coulG in pwdf.pw_loop(cell, pwdf.gs, kptij, max_memory):
            if vhfopt_or_mf is not None:
                nG = len(coulG)
                coulG = vkcoulG[p0:p0+nG]
                p0 += nG
            coulG = numpy.sqrt(coulG)
            pqkR *= coulG
            pqkI *= coulG
            pqk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            rqk = numpy.dot(dm, pqk.reshape(nao,-1)).reshape(nao**2,-1)
            pqk = pqk.conj()
            if gamma_point:
                vk += numpy.einsum('jiL,jlL->il', pqk, rqk).real
            else:
                vk += numpy.einsum('jiL,jlL->il', pqk, rqk)
        pqkR = LkR = pqkI = LkI = coulG = None
        return vk

    if kpt_band is None:
        kpts_lst = kpts
    else:
        kpts_lst = numpy.reshape(kpt_band, (-1,3))

    vk_kpts = [0] * len(kpts_lst)
    for k1, kpti in enumerate(kpts_lst):
        for k2, kptj in enumerate(kpts):
            vk_kpts[k1] += make_k1k2(kpti, kptj, dm_kpts[k2])

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        return vk_kpts[0]
    else:
        return vk_kpts


##################################################
#
# Single k-point
#
##################################################

def get_j_kpt(pwdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(pwdf, cell, dm, hermi, vhfopt, kpt, True, False)[0]

def get_k_kpt(pwdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(pwdf, cell, dm, hermi, vhfopt, kpt, True, False)[1]

def get_jk(pwdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True):
    '''JK for given k-point'''
    if kpt_band is not None and abs(kpt-kpt_band).sum() > 1e-9:
        vj = vk = None
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(xdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)[0]
        if with_j:
            vj = get_j_kpts(xdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)[0]
        return vj, vk

    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = (time.clock(), time.time())
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    kptii = numpy.asarray((kpt,kpt))
    gamma_point = abs(kpt).sum() < 1e-9
    vj = vk = None

    if with_k:
        vk = 0
        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, pwdf,
                                      pwdf.gs) / cell.vol
        else:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, vhfopt_or_mf,
                                      pwdf.gs) / cell.vol
    vjR = 0
    vjI = 0
    p0 = 0
    max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
    for pqkR, pqkI, coulG in pwdf.pw_loop(cell, pwdf.gs, kptii, max_memory):
        if with_j:
            rhoR = numpy.dot(dm.ravel(), pqkR.reshape(nao**2,-1))
            rhoI = numpy.dot(dm.ravel(), pqkI.reshape(nao**2,-1))
            rhoR *= coulG
            rhoI *= coulG
            vjR += numpy.dot(pqkR, rhoR)
            vjR += numpy.dot(pqkI, rhoI)
            if not gamma_point:
                vjI += numpy.dot(pqkI, rhoR)
                vjI -= numpy.dot(pqkR, rhoI)

        if with_k:
            if vhfopt_or_mf is not None:
                nG = len(coulG)
                coulG = vkcoulG[p0:p0+nG]
                p0 += nG
            coulG = numpy.sqrt(coulG)
            pqkR *= coulG
            pqkI *= coulG
            pqk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            rqk = numpy.dot(dm, pqk.reshape(nao,-1)).reshape(nao,nao,-1)
            pqk =(pqkR+pqkI*1j).reshape(nao,nao,-1)
            if gamma_point:
                vk += numpy.einsum('jiG,jlG->il', pqk, rqk).real
            else:
                vk += numpy.einsum('jiG,jlG->il', pqk, rqk)
    pqkR = pqkI = coulG = None

    if with_j:
        if gamma_point:
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj = vj.reshape(nao,nao)
    return vj, vk


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import pwdf

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.h = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    df = pwdf.PWDF(cell)
    df.gs = (15,)*3
    dm = pscf.RHF(cell).get_init_guess()
    vj, vk = df.get_jk(cell, dm)
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.384051732669329')
    df.analytic_ft = True
    #print(numpy.einsum('ij,ji->', vj, dm), 'ref=5.3766911667862516')
    #print(numpy.einsum('ij,ji->', vk, dm), 'ref=8.2255177602309022')
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.447018516011319')

