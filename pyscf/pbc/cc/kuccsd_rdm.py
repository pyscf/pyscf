#!/usr/bin/env python
# Copyright 2017-2021 The PySCF Developers. All Rights Reserved.
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

import numpy
from pyscf import lib
import itertools

#einsum = numpy.einsum
einsum = lib.einsum

#TODO: optimize memory use

def _gamma1_intermediates(cc, t1, t2, l1=None, l2=None):
    if l1 is None:
        l1 = [amp.conj() for amp in t1]
    if l2 is None:
        l2 = [amp.conj() for amp in t2]
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nkpts, nocca, nvira = t1a.shape
    _, noccb, nvirb = t1b.shape

    kconserv = cc.khelper.kconserv

    dooa  = -einsum('xie,xje->xij', l1a, t1a)
    dooa -=  einsum('xyzimef,xyzjmef->xij', l2ab, t2ab)
    dooa -=  einsum('xyzimef,xyzjmef->xij', l2aa, t2aa) * .5
    doob  = -einsum('xie,xje->xij', l1b, t1b)
    doob -=  einsum('yxzmief,yxzmjef->xij', l2ab, t2ab)
    doob -=  einsum('xyzimef,xyzjmef->xij', l2bb, t2bb) * .5

    dvva  = einsum('xma,xmb->xab', t1a, l1a)
    dvva += einsum('xyzmnae,xyzmnbe->zab', t2ab, l2ab)
    dvva += einsum('xyzmnae,xyzmnbe->zab', t2aa, l2aa) * .5
    dvvb  = einsum('xma,xmb->xab', t1b, l1b)

    for km, kn, ke in itertools.product(range(nkpts),repeat=3):
        ka = kconserv[km,ke,kn]
        dvvb[ka] += einsum('mnea,mneb->ab', t2ab[km,kn,ke], l2ab[km,kn,ke])
    dvvb += einsum('xyzmnae,xyzmnbe->zab', t2bb, l2bb) * .5

    xt1a  = einsum('xyzmnef,xyzinef->xmi', l2aa, t2aa) * .5
    xt1a += einsum('xyzmnef,xyzinef->xmi', l2ab, t2ab)
    xt2a  = einsum('xyzmnaf,xyzmnef->zae', t2aa, l2aa) * .5
    xt2a += einsum('xyzmnaf,xyzmnef->zae', t2ab, l2ab)
    xt2a += einsum('xma,xme->xae', t1a, l1a)

    dvoa = t1a.copy().transpose(0,2,1)
    for ka, km in itertools.product(range(nkpts),repeat=2):
        dvoa[ka] += einsum('imae, me->ai', t2aa[ka,km,ka], l1a[km])
        dvoa[ka] += einsum('imae, me->ai', t2ab[ka,km,ka], l1b[km])
    dvoa -= einsum('xmi,xma->xai', xt1a, t1a)
    dvoa -= einsum('xie,xae->xai', t1a, xt2a)

    xt1b  = einsum('xyzmnef,xyzinef->xmi', l2bb, t2bb) * .5
    xt1b += einsum('xyznmef,xyznief->ymi', l2ab, t2ab)
    xt2b  = einsum('xyzmnaf,xyzmnef->zae', t2bb, l2bb) * .5
    for km, kn, kf in itertools.product(range(nkpts),repeat=3):
        ka = kconserv[km,kf,kn]
        xt2b[ka] += einsum('mnfa,mnfe->ae', t2ab[km,kn,kf], l2ab[km,kn,kf])
    xt2b += einsum('xma,xme->xae', t1b, l1b)

    dvob = t1b.copy().transpose(0,2,1)
    for ka, km in itertools.product(range(nkpts),repeat=2):
        dvob[ka] += einsum('imae,me->ai', t2bb[ka,km,ka], l1b[km])
        dvob[ka] += einsum('miea,me->ai', t2ab[km,ka,km], l1a[km])
    dvob -= einsum('xmi, xma->xai',xt1b, t1b)
    dvob -= einsum('xie,xae->xai', t1b, xt2b)

    dova = l1a
    dovb = l1b

    return ((dooa, doob), (dova, dovb), (dvoa, dvob), (dvva, dvvb))


def make_rdm1(mycc, t1, t2, l1=None, l2=None, ao_repr=False):
    r'''
    One-particle spin density matrices dm1a, dm1b in MO basis (the
    occupied-virtual blocks due to the orbital response contribution are not
    included).

    dm1a[p,q] = <q_alpha^\dagger p_alpha>
    dm1b[p,q] = <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)

def _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False):
    doo, dOO = d1[0]
    dov, dOV = d1[1]
    dvo, dVO = d1[2]
    dvv, dVV = d1[3]
    nkpts, nocca, nvira = dov.shape
    _, noccb, nvirb = dOV.shape
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    dtype = numpy.result_type(doo, dOO, dov, dOV, dvo, dVO, dvv, dVV)

    dm1a = numpy.empty((nkpts,nmoa,nmoa), dtype=dtype)
    dm1a[:,:nocca,:nocca] = doo + doo.conj().transpose(0,2,1)
    dm1a[:,:nocca,nocca:] = dov + dvo.conj().transpose(0,2,1)
    dm1a[:,nocca:,:nocca] = dm1a[:,:nocca,nocca:].conj().transpose(0,2,1)
    dm1a[:,nocca:,nocca:] = dvv + dvv.conj().transpose(0,2,1)
    dm1a *= .5
    for k in range(nkpts):
        dm1a[k][numpy.diag_indices(nocca)] +=1

    dm1b = numpy.empty((nkpts,nmob,nmob), dtype=dtype)
    dm1b[:,:noccb,:noccb] = dOO + dOO.conj().transpose(0,2,1)
    dm1b[:,:noccb,noccb:] = dOV + dVO.conj().transpose(0,2,1)
    dm1b[:,noccb:,:noccb] = dm1b[:,:noccb,noccb:].conj().transpose(0,2,1)
    dm1b[:,noccb:,noccb:] = dVV + dVV.conj().transpose(0,2,1)
    dm1b *= .5
    for k in range(nkpts):
        dm1b[k][numpy.diag_indices(noccb)] +=1

    if with_frozen and mycc.frozen is not None:
        raise NotImplementedError
        _, nmoa = mycc.mo_occ[0].size
        _, nmob = mycc.mo_occ[1].size
        nocca = numpy.count_nonzero(mycc.mo_occ[0] > 0)
        noccb = numpy.count_nonzero(mycc.mo_occ[1] > 0)
        rdm1a = numpy.zeros((nkpts,nmoa,nmoa), dtype=dm1a.dtype)
        rdm1b = numpy.zeros((nkpts,nmob,nmob), dtype=dm1b.dtype)
        rdm1a[:,numpy.diag_indices(nocca)] = 1
        rdm1b[:,numpy.diag_indices(noccb)] = 1
        moidx = mycc.get_frozen_mask()
        moidxa = numpy.where(moidx[0])[0]
        moidxb = numpy.where(moidx[1])[0]
        rdm1a[moidxa[:,None],moidxa] = dm1a
        rdm1b[moidxb[:,None],moidxb] = dm1b
        dm1a = rdm1a
        dm1b = rdm1b

    if ao_repr:
        mo_a, mo_b = mycc.mo_coeff
        dm1a = lib.einsum('xpi,xij,xqj->xpq', mo_a, dm1a, mo_a.conj())
        dm1b = lib.einsum('xpi,xij,xqj->xpq', mo_b, dm1b, mo_b.conj())
    return dm1a, dm1b



if __name__ == '__main__':
    import numpy as np
    from pyscf.pbc import gto
    from pyscf.pbc import scf
    from pyscf import lib
    from pyscf.pbc.cc import KUCCSD

    a0 = 3.0
    vac = 200
    nmp = [1,1,1]
    dis = 10
    vec = [[ 3.0/2.0*a0*dis,np.sqrt(3.0)/2.0*a0*dis,  0],
           [-3.0/2.0*a0*dis,np.sqrt(3.0)/2.0*a0*dis,  0],
           [          0,                  0,vac]]
    bas = 'cc-pvdz'
    pos = [['H',(-a0/2.0,0,0)],
           ['H',( a0/2.0,0,0)]]

    cell = gto.M(unit='B',a=vec,atom=pos,basis=bas,verbose=4)
    cell.precision = 1e-11

    kpts = cell.make_kpts(nmp)
    kmf = scf.KUHF(cell,kpts,exxdiv=None).density_fit()
    kmf.max_cycle=250

    dm = kmf.get_init_guess()
    aoind = cell.aoslice_by_atom()
    idx0, idx1 = aoind
    dm[0,:,idx0[2]:idx0[3], idx0[2]:idx0[3]] = 0
    dm[0,:,idx1[2]:idx1[3], idx1[2]:idx1[3]] = dm[0,:,idx1[2]:idx1[3], idx1[2]:idx1[3]] * 2
    dm[1,:,idx0[2]:idx0[3], idx0[2]:idx0[3]] = dm[1,:,idx0[2]:idx0[3], idx0[2]:idx0[3]] * 2
    dm[1,:,idx1[2]:idx1[3], idx1[2]:idx1[3]] = 0

    ehf = kmf.kernel(dm)

    kcc = KUCCSD(kmf)
    ecc, t1, t2 = kcc.kernel()

    dm1a,dm1b = make_rdm1(kcc, t1, t2, l1=None, l2=None)

    ((dooa, doob), (dova, dovb), (dvoa, dvob), (dvva, dvvb)) = _gamma1_intermediates(kcc, t1, t2, l1=t1, l2=t2)

    print(np.linalg.norm(dooa))
    print(np.linalg.norm(doob))
    print(np.linalg.norm(dova))
    print(np.linalg.norm(dovb))
    print(np.linalg.norm(dvoa))
    print(np.linalg.norm(dvob))
    print(np.linalg.norm(dvva))
    print(np.linalg.norm(dvvb))

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    print(np.linalg.norm(t1a))
    print(np.linalg.norm(t1b))
    print(np.linalg.norm(t2aa))
    print(np.linalg.norm(t2ab))
    print(np.linalg.norm(t2bb))


    #print(kmf.spin_square())
    print(np.linalg.norm(dm1a))
    print(np.linalg.norm(dm1b))

    print(np.linalg.norm(dm1a - dm1b))
    print(dm1a[0])
    print(dm1b[0])
