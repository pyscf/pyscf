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
#

'''
UMP2 analytical nuclear gradients
'''

import time
import numpy
from pyscf import lib
from functools import reduce
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.ao2mo import _ao2mo
from pyscf.mp import ump2
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import mp2 as mp2_grad


def grad_elec(mp_grad, t2, atmlst=None, verbose=logger.INFO):
    mp = mp_grad.base
    log = logger.new_logger(mp, verbose)
    time0 = time.clock(), time.time()

    log.debug('Build ump2 rdm1 intermediates')
    d1 = ump2._gamma1_intermediates(mp, t2)
    time1 = log.timer_debug1('rdm1 intermediates', *time0)
    log.debug('Build ump2 rdm2 intermediates')

    mol = mp_grad.mol
    with_frozen = not ((mp.frozen is None)
                       or (isinstance(mp.frozen, (int, numpy.integer)) and mp.frozen == 0)
                       or (len(mp.frozen) == 0))
    moidx = mp.get_frozen_mask()
    OA_a, VA_a, OF_a, VF_a = mp2_grad._index_frozen_active(moidx[0], mp.mo_occ[0])
    OA_b, VA_b, OF_b, VF_b = mp2_grad._index_frozen_active(moidx[1], mp.mo_occ[1])
    orboa = mp.mo_coeff[0][:,OA_a]
    orbva = mp.mo_coeff[0][:,VA_a]
    orbob = mp.mo_coeff[1][:,OA_b]
    orbvb = mp.mo_coeff[1][:,VA_b]
    nao, nocca = orboa.shape
    nvira = orbva.shape[1]
    noccb = orbob.shape[1]
    nvirb = orbvb.shape[1]

# Partially transform MP2 density matrix and hold it in memory
# The rest transformation are applied during the contraction to ERI integrals
    t2aa, t2ab, t2bb = t2
    part_dm2aa = _ao2mo.nr_e2(t2aa.reshape(nocca**2,nvira**2),
                              numpy.asarray(orbva.T, order='F'), (0,nao,0,nao),
                              's1', 's1').reshape(nocca,nocca,nao,nao)
    part_dm2bb = _ao2mo.nr_e2(t2bb.reshape(noccb**2,nvirb**2),
                              numpy.asarray(orbvb.T, order='F'), (0,nao,0,nao),
                              's1', 's1').reshape(noccb,noccb,nao,nao)
    part_dm2ab = lib.einsum('ijab,pa,qb->ipqj', t2ab, orbva, orbvb)
    part_dm2aa = (part_dm2aa.transpose(0,2,3,1) -
                  part_dm2aa.transpose(0,3,2,1)) * .5
    part_dm2bb = (part_dm2bb.transpose(0,2,3,1) -
                  part_dm2bb.transpose(0,3,2,1)) * .5

    hf_dm1a, hf_dm1b = mp._scf.make_rdm1(mp.mo_coeff, mp.mo_occ)
    hf_dm1 = hf_dm1a + hf_dm1b

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    de = numpy.zeros((len(atmlst),3))
    Imata = numpy.zeros((nao,nao))
    Imatb = numpy.zeros((nao,nao))
    fdm2 = lib.H5TmpFile()
    vhf1 = fdm2.create_dataset('vhf1', (len(atmlst),2,3,nao,nao), 'f8')

# 2e AO integrals dot 2pdm
    max_memory = max(0, mp.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf = numpy.zeros((2,3,nao,nao))
        for b0, b1, nf in mp2_grad._shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2bufa = lib.einsum('pi,iqrj->pqrj', orboa[ip0:ip1], part_dm2aa)
            dm2bufa+= lib.einsum('qi,iprj->pqrj', orboa, part_dm2aa[:,ip0:ip1])
            dm2bufa = lib.einsum('pqrj,sj->pqrs', dm2bufa, orboa)
            tmp = lib.einsum('pi,iqrj->pqrj', orboa[ip0:ip1], part_dm2ab)
            tmp+= lib.einsum('qi,iprj->pqrj', orboa, part_dm2ab[:,ip0:ip1])
            dm2bufa+= lib.einsum('pqrj,sj->pqrs', tmp, orbob)
            tmp = None
            dm2bufa = dm2bufa + dm2bufa.transpose(0,1,3,2)
            dm2bufa = lib.pack_tril(dm2bufa.reshape(-1,nao,nao)).reshape(nf,nao,-1)
            dm2bufa[:,:,diagidx] *= .5

            dm2bufb = lib.einsum('pi,iqrj->pqrj', orbob[ip0:ip1], part_dm2bb)
            dm2bufb+= lib.einsum('qi,iprj->pqrj', orbob, part_dm2bb[:,ip0:ip1])
            dm2bufb = lib.einsum('pqrj,sj->pqrs', dm2bufb, orbob)
            tmp = lib.einsum('iqrj,sj->iqrs', part_dm2ab, orbob[ip0:ip1])
            tmp+= lib.einsum('iqrj,sj->iqsr', part_dm2ab[:,:,ip0:ip1], orbob)
            dm2bufb+= lib.einsum('pi,iqrs->srpq', orboa, tmp)
            tmp = None
            dm2bufb = dm2bufb + dm2bufb.transpose(0,1,3,2)
            dm2bufb = lib.pack_tril(dm2bufb.reshape(-1,nao,nao)).reshape(nf,nao,-1)
            dm2bufb[:,:,diagidx] *= .5

            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imata += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2bufa)
            Imatb += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2bufb)
            eri0 = None

            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,nf,nao,-1)
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2bufa) * 2
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2bufb) * 2
            dm2bufa = dm2bufb = None
# HF part
            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape(nf*nao,-1))
                eri1tmp = eri1tmp.reshape(nf,nao,nao,nao)
                vhf[:,i] += numpy.einsum('ijkl,ij->kl', eri1tmp, hf_dm1[ip0:ip1])
                vhf[0,i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1a[ip0:ip1])
                vhf[1,i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1b[ip0:ip1])
                vhf[:,i,ip0:ip1] += numpy.einsum('ijkl,kl->ij', eri1tmp, hf_dm1)
                vhf[0,i,ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1a)
                vhf[1,i,ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1b)
            eri1 = eri1tmp = None
        vhf1[k] = vhf
        log.debug('2e-part grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer_debug1('2e-part grad of atom %d'%ia, *time1)

# Recompute nocc, nvir to include the frozen orbitals and make contraction for
# the 1-particle quantities, see also the kernel function in uccsd_grad module.
    mo_a, mo_b = mp.mo_coeff
    mo_ea, mo_eb = mp._scf.mo_energy
    nao, nmoa = mo_a.shape
    nmob = mo_b.shape[1]
    nocca = numpy.count_nonzero(mp.mo_occ[0] > 0)
    noccb = numpy.count_nonzero(mp.mo_occ[1] > 0)
    s0 = mp._scf.get_ovlp()
    Imata = reduce(numpy.dot, (mo_a.T, Imata, s0, mo_a)) * -1
    Imatb = reduce(numpy.dot, (mo_b.T, Imatb, s0, mo_b)) * -1

    dm1a = numpy.zeros((nmoa,nmoa))
    dm1b = numpy.zeros((nmob,nmob))
    doo, dOO = d1[0]
    dvv, dVV = d1[1]
    if with_frozen:
        dco = Imata[OF_a[:,None],OA_a] / (mo_ea[OF_a,None] - mo_ea[OA_a])
        dfv = Imata[VF_a[:,None],VA_a] / (mo_ea[VF_a,None] - mo_ea[VA_a])
        dm1a[OA_a[:,None],OA_a] = (doo + doo.T) * .5
        dm1a[OF_a[:,None],OA_a] = dco
        dm1a[OA_a[:,None],OF_a] = dco.T
        dm1a[VA_a[:,None],VA_a] = (dvv + dvv.T) * .5
        dm1a[VF_a[:,None],VA_a] = dfv
        dm1a[VA_a[:,None],VF_a] = dfv.T
        dco = Imatb[OF_b[:,None],OA_b] / (mo_eb[OF_b,None] - mo_eb[OA_b])
        dfv = Imatb[VF_b[:,None],VA_b] / (mo_eb[VF_b,None] - mo_eb[VA_b])
        dm1b[OA_b[:,None],OA_b] = (dOO + dOO.T) * .5
        dm1b[OF_b[:,None],OA_b] = dco
        dm1b[OA_b[:,None],OF_b] = dco.T
        dm1b[VA_b[:,None],VA_b] = (dVV + dVV.T) * .5
        dm1b[VF_b[:,None],VA_b] = dfv
        dm1b[VA_b[:,None],VF_b] = dfv.T
    else:
        dm1a[:nocca,:nocca] = (doo + doo.T) * .5
        dm1a[nocca:,nocca:] = (dvv + dvv.T) * .5
        dm1b[:noccb,:noccb] = (dOO + dOO.T) * .5
        dm1b[noccb:,noccb:] = (dVV + dVV.T) * .5

    dm1 = (reduce(numpy.dot, (mo_a, dm1a, mo_a.T)),
           reduce(numpy.dot, (mo_b, dm1b, mo_b.T)))
    vhf = mp._scf.get_veff(mp.mol, dm1)
    Xvo = reduce(numpy.dot, (mo_a[:,nocca:].T, vhf[0], mo_a[:,:nocca]))
    XVO = reduce(numpy.dot, (mo_b[:,noccb:].T, vhf[1], mo_b[:,:noccb]))
    Xvo+= Imata[:nocca,nocca:].T - Imata[nocca:,:nocca]
    XVO+= Imatb[:noccb,noccb:].T - Imatb[noccb:,:noccb]

    dm1_resp = _response_dm1(mp, (Xvo,XVO))
    dm1a += dm1_resp[0]
    dm1b += dm1_resp[1]
    time1 = log.timer_debug1('response_rdm1 intermediates', *time1)

    Imata[nocca:,:nocca] = Imata[:nocca,nocca:].T
    Imatb[noccb:,:noccb] = Imatb[:noccb,noccb:].T
    im1 = reduce(numpy.dot, (mo_a, Imata, mo_a.T))
    im1+= reduce(numpy.dot, (mo_b, Imatb, mo_b.T))
    time1 = log.timer_debug1('response_rdm1', *time1)

    log.debug('h1 and JK1')
    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = mp_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    zeta = (mo_ea[:,None] + mo_ea) * .5
    zeta[nocca:,:nocca] = mo_ea[:nocca]
    zeta[:nocca,nocca:] = mo_ea[:nocca].reshape(-1,1)
    zeta_a = reduce(numpy.dot, (mo_a, zeta*dm1a, mo_a.T))
    zeta = (mo_eb[:,None] + mo_eb) * .5
    zeta[noccb:,:noccb] = mo_eb[:noccb]
    zeta[:noccb,noccb:] = mo_eb[:noccb].reshape(-1,1)
    zeta_b = reduce(numpy.dot, (mo_b, zeta*dm1b, mo_b.T))

    dm1 = (reduce(numpy.dot, (mo_a, dm1a, mo_a.T)),
           reduce(numpy.dot, (mo_b, dm1b, mo_b.T)))
    vhf_s1occ = mp._scf.get_veff(mol, (dm1[0]+dm1[0].T, dm1[1]+dm1[1].T))
    p1a = numpy.dot(mo_a[:,:nocca], mo_a[:,:nocca].T)
    p1b = numpy.dot(mo_b[:,:noccb], mo_b[:,:noccb].T)
    vhf_s1occ = (reduce(numpy.dot, (p1a, vhf_s1occ[0], p1a)) +
                 reduce(numpy.dot, (p1b, vhf_s1occ[1], p1b))) * .5
    time1 = log.timer_debug1('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    dm1pa = hf_dm1a + dm1[0]*2
    dm1pb = hf_dm1b + dm1[1]*2
    dm1 = dm1[0] + dm1[1] + hf_dm1
    zeta_a += rhf_grad.make_rdm1e(mo_ea, mo_a, mp.mo_occ[0])
    zeta_b += rhf_grad.make_rdm1e(mo_eb, mo_b, mp.mo_occ[1])
    zeta = zeta_a + zeta_b

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1])
# h[1] \dot DM, contribute to f1
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, dm1)
# -s[1]*e \dot DM,  contribute to f1
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
        de[k] -= numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1])
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', vhf1[k,0], dm1pa)
        de[k] -= numpy.einsum('xij,ij->x', vhf1[k,1], dm1pb)

    log.timer('%s gradients' % mp.__class__.__name__, *time0)
    return de

def _response_dm1(mp, Xvo):
    Xvo, XVO = Xvo
    nvira, nocca = Xvo.shape
    nvirb, noccb = XVO.shape
    nmoa = nocca + nvira
    nmob = noccb + nvirb
    nova = nocca * nvira
    mo_energy = mp._scf.mo_energy
    mo_occ = mp.mo_occ
    mo_a, mo_b = mp.mo_coeff
    def fvind(x):
        x1a = x[0,:nova].reshape(Xvo.shape)
        x1b = x[0,nova:].reshape(XVO.shape)
        dm1a = reduce(numpy.dot, (mo_a[:,nocca:], x1a, mo_a[:,:nocca].T))
        dm1b = reduce(numpy.dot, (mo_b[:,noccb:], x1b, mo_b[:,:noccb].T))
        va, vb = mp._scf.get_veff(mp.mol, (dm1a+dm1a.T, dm1b+dm1b.T))
        va = reduce(numpy.dot, (mo_a[:,nocca:].T, va, mo_a[:,:nocca]))
        vb = reduce(numpy.dot, (mo_b[:,noccb:].T, vb, mo_b[:,:noccb]))
        return numpy.hstack((va.ravel(), vb.ravel()))
    dvo = ucphf.solve(fvind, mo_energy, mo_occ, (Xvo,XVO), max_cycle=30)[0]
    dm1a = numpy.zeros((nmoa,nmoa))
    dm1a[nocca:,:nocca] = dvo[0]
    dm1a[:nocca,nocca:] = dvo[0].T
    dm1b = numpy.zeros((nmob,nmob))
    dm1b[noccb:,:noccb] = dvo[1]
    dm1b[:noccb,noccb:] = dvo[1].T
    return dm1a, dm1b


class Gradients(mp2_grad.Gradients):
    grad_elec = grad_elec

Grad = Gradients

ump2.UMP2.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g',
        spin = 2,
    )
    mf = scf.UHF(mol).run()
    mp = ump2.UMP2(mf).run()
    g1 = mp.Gradients().kernel()
# O     0.0000000000    -0.0000000000     0.1436990190
# H    -0.0000000000     0.1097329294    -0.0718495095
# H    -0.0000000000    -0.1097329294    -0.0718495095
    print(lib.finger(g1) - -0.22418090721297307)

    print('-----------------------------------')
    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g',
        spin = 2,
    )
    mf = scf.UHF(mol).run()
    mp = ump2.UMP2(mf)
    mp.frozen = [0,1,10,11,12]
    mp.max_memory = 1
    mp.kernel()
    g1 = Gradients(mp).kernel()
# O    -0.0000000000    -0.0000000000     0.1454782514
# H     0.0000000000     0.1092558730    -0.0727391257
# H    -0.0000000000    -0.1092558730    -0.0727391257
    print(lib.finger(g1) - -0.22437276158813313)

