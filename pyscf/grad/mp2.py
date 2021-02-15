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
MP2 analytical nuclear gradients
'''

import time
import numpy
from pyscf import lib
from functools import reduce
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.scf import cphf
from pyscf.mp import mp2
from pyscf.ao2mo import _ao2mo


def grad_elec(mp_grad, t2, atmlst=None, verbose=logger.INFO):
    mp = mp_grad.base
    log = logger.new_logger(mp, verbose)
    time0 = time.clock(), time.time()

    log.debug('Build mp2 rdm1 intermediates')
    d1 = mp2._gamma1_intermediates(mp, t2)
    doo, dvv = d1
    time1 = log.timer_debug1('rdm1 intermediates', *time0)

# Set nocc, nvir for half-transformation of 2pdm.  Frozen orbitals are exculded.
# nocc, nvir should be updated to include the frozen orbitals when proceeding
# the 1-particle quantities later.
    mol = mp_grad.mol
    with_frozen = not ((mp.frozen is None)
                       or (isinstance(mp.frozen, (int, numpy.integer)) and mp.frozen == 0)
                       or (len(mp.frozen) == 0))
    OA, VA, OF, VF = _index_frozen_active(mp.get_frozen_mask(), mp.mo_occ)
    orbo = mp.mo_coeff[:,OA]
    orbv = mp.mo_coeff[:,VA]
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]

# Partially transform MP2 density matrix and hold it in memory
# The rest transformation are applied during the contraction to ERI integrals
    part_dm2 = _ao2mo.nr_e2(t2.reshape(nocc**2,nvir**2),
                            numpy.asarray(orbv.T, order='F'), (0,nao,0,nao),
                            's1', 's1').reshape(nocc,nocc,nao,nao)
    part_dm2 = (part_dm2.transpose(0,2,3,1) * 4 -
                part_dm2.transpose(0,3,2,1) * 2)

    hf_dm1 = mp._scf.make_rdm1(mp.mo_coeff, mp.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    de = numpy.zeros((len(atmlst),3))
    Imat = numpy.zeros((nao,nao))
    fdm2 = lib.H5TmpFile()
    vhf1 = fdm2.create_dataset('vhf1', (len(atmlst),3,nao,nao), 'f8')

# 2e AO integrals dot 2pdm
    max_memory = max(0, mp.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf = numpy.zeros((3,nao,nao))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum('pi,iqrj->pqrj', orbo[ip0:ip1], part_dm2)
            dm2buf+= lib.einsum('qi,iprj->pqrj', orbo, part_dm2[:,ip0:ip1])
            dm2buf = lib.einsum('pqrj,sj->pqrs', dm2buf, orbo)
            dm2buf = dm2buf + dm2buf.transpose(0,1,3,2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1,nao,nao)).reshape(nf,nao,-1)
            dm2buf[:,:,diagidx] *= .5

            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2buf)
            eri0 = None

            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,nf,nao,-1)
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
            dm2buf = None
# HF part
            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i]).reshape(nf*nao,-1)
                eri1tmp = eri1tmp.reshape(nf,nao,nao,nao)
                vhf[i] += numpy.einsum('ijkl,ij->kl', eri1tmp, hf_dm1[ip0:ip1])
                vhf[i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1[ip0:ip1]) * .5
                vhf[i,ip0:ip1] += numpy.einsum('ijkl,kl->ij', eri1tmp, hf_dm1)
                vhf[i,ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1) * .5
            eri1 = eri1tmp = None
        vhf1[k] = vhf
        log.debug('2e-part grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer_debug1('2e-part grad of atom %d'%ia, *time1)

# Recompute nocc, nvir to include the frozen orbitals and make contraction for
# the 1-particle quantities, see also the kernel function in ccsd_grad module.
    mo_coeff = mp.mo_coeff
    mo_energy = mp._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(mp.mo_occ > 0)
    Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mp._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = numpy.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mp._scf.get_veff(mp.mol, dm1) * 2
    Xvo = reduce(numpy.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(mp, Xvo)
    time1 = log.timer_debug1('response_rdm1 intermediates', *time1)

    Imat[nocc:,:nocc] = Imat[:nocc,nocc:].T
    im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))
    time1 = log.timer_debug1('response_rdm1', *time1)

    log.debug('h1 and JK1')
    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = mp_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf_s1occ = reduce(numpy.dot, (p1, mp._scf.get_veff(mol, dm1+dm1.T), p1))
    time1 = log.timer_debug1('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    dm1p = hf_dm1 + dm1*2
    dm1 += hf_dm1
    zeta += rhf_grad.make_rdm1e(mo_energy, mo_coeff, mp.mo_occ)

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
        de[k] -= numpy.einsum('xij,ij->x', vhf1[k], dm1p)

    log.timer('%s gradients' % mp.__class__.__name__, *time0)
    return de


def as_scanner(grad_mp):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total MP2 energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    MP2 and the underlying SCF objects (max_memory etc) are automatically
    applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

    >>> from pyscf import gto, scf, mp
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> mp2_scanner = mp.MP2(scf.RHF(mol)).nuc_grad_method().as_scanner()
    >>> e_tot, grad = mp2_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
    >>> e_tot, grad = mp2_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(grad_mp, lib.GradScanner):
        return grad_mp

    logger.info(grad_mp, 'Create scanner for %s', grad_mp.__class__)

    class MP2_GradScanner(grad_mp.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            mp_scanner = self.base
            mp_scanner(mol, with_t2=True)
            de = self.kernel(mp_scanner.t2)
            return mp_scanner.e_tot, de
        @property
        def converged(self):
            return self.base._scf.converged
    return MP2_GradScanner(grad_mp)


def _shell_prange(mol, start, stop, blksize):
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

def _response_dm1(mp, Xvo):
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    mo_energy = mp._scf.mo_energy
    mo_occ = mp.mo_occ
    mo_coeff = mp.mo_coeff
    def fvind(x):
        x = x.reshape(Xvo.shape)
        dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
        v = mp._scf.get_veff(mp.mol, dm + dm.T)
        v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, v, mo_coeff[:,:nocc]))
        return v * 2
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1

def _index_frozen_active(frozen_mask, mo_occ):
    OA = numpy.where(( frozen_mask) & (mo_occ> 0))[0] # occupied active orbitals
    OF = numpy.where((~frozen_mask) & (mo_occ> 0))[0] # occupied frozen orbitals
    VA = numpy.where(( frozen_mask) & (mo_occ==0))[0] # virtual active orbitals
    VF = numpy.where((~frozen_mask) & (mo_occ==0))[0] # virtual frozen orbitals
    return OA, VA, OF, VF

class Gradients(rhf_grad.GradientsBasics):

    grad_elec = grad_elec

    def kernel(self, t2=None, atmlst=None, verbose=None):
        log = logger.new_logger(self, verbose)
        if t2 is None: t2 = self.base.t2
        if t2 is None: t2 = self.base.kernel()
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        de = self.grad_elec(t2, atmlst, verbose=log)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    # Calling the underlying SCF nuclear gradients because it may be modified
    # by external modules (e.g. QM/MM, solvent)
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    as_scanner = as_scanner

Grad = Gradients

# Inject to RMP2 class
mp2.MP2.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mp = mp2.MP2(mf).run()
    g1 = Gradients(mp).kernel()
# O    -0.0000000000    -0.0000000000     0.0089211366
# H     0.0000000000     0.0222745046    -0.0044605683
# H     0.0000000000    -0.0222745046    -0.0044605683
    print(lib.finger(g1) - -0.035681171529705444)

    mcs = mp.as_scanner()
    mol.set_geom_([
            ["O" , (0. , 0.     , 0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e1 = mcs(mol)
    mol.set_geom_([
            ["O" , (0. , 0.     ,-0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e2 = mcs(mol)
    print(g1[0,2], (e1-e2)/0.002*lib.param.BOHR)

    print('-----------------------------------')
    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mp = mp2.MP2(mf)
    mp.frozen = [0,1,10,11,12]
    mp.max_memory = 1
    mp.kernel()
    g1 = Gradients(mp).kernel()
# O    -0.0000000000    -0.0000000000     0.0037319667
# H    -0.0000000000    -0.0897959298    -0.0018659834
# H     0.0000000000     0.0897959298    -0.0018659834
    print(lib.finger(g1) - 0.12458103614793946)

    mcs = mp.as_scanner()
    mol.set_geom_([
            ["O" , (0. , 0.     , 0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e1 = mcs(mol)
    mol.set_geom_([
            ["O" , (0. , 0.     ,-0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e2 = mcs(mol)
    print(g1[0,2], (e1-e2)/0.002*lib.param.BOHR)

    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 1.76',
        basis = '631g',
        unit='Bohr')
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mp = mp2.MP2(mf)
    mp.kernel()
    g1 = mp.Gradients().kernel()
# H     0.0000000000     0.0000000000    -0.0800309688
# H     0.0000000000     0.0000000000     0.0800309688

