# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

'''
GMP2 in spin-orbital form
E(MP2) = 1/4 <ij||ab><ab||ij>/(ei+ej-ea-eb)
'''

import copy
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.mp import mp2
from pyscf import scf
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_gmp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    #moidx = mp.get_frozen_mask()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.oovv.dtype)
    else:
        t2 = None

    emp2 = 0
    for i in range(nocc):
        gi = numpy.asarray(eris.oovv[i]).reshape(nocc,nvir,nvir)
        t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        emp2 += numpy.einsum('jab,jab', t2i, gi) * .25
        if with_t2:
            t2[i] = t2i

    return emp2.real, t2

def energy(mp, t2, eris):
    '''MP2 energy'''
    eris_oovv = numpy.array(eris.oovv)
    e = 0.25*numpy.einsum('ijab,ijab', t2, eris_oovv)
    if abs(e.imag) > 1e-4:
        logger.warn(mp, 'Non-zero imaginary part found in GMP2 energy %s', e)
    return e.real

def update_amps(mp, t2, eris):
    '''Update non-canonical MP2 amplitudes'''
    #assert (isinstance(eris, _PhysicistsERIs))
    nocc, nvir = t2.shape[1:3]
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mp.level_shift

    foo = fock[:nocc,:nocc] - numpy.diag(mo_e_o)
    fvv = fock[nocc:,nocc:] - numpy.diag(mo_e_v)
    t2new  = lib.einsum('ijac,bc->ijab', t2, fvv)
    t2new -= lib.einsum('ki,kjab->ijab', foo, t2)
    t2new = t2new + t2new.transpose(1,0,3,2)
    t2new += numpy.asarray(eris.oovv).conj()

    eia = mo_e_o[:,None] - mo_e_v
    t2new /= lib.direct_sum('ia,jb->ijab', eia, eia)
    return t2new


def make_rdm1(mp, t2=None, ao_repr=False):
    r'''
    One-particle density matrix in the molecular spin-orbital representation
    (the occupied-virtual blocks from the orbital response contribution are
    not included).

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    from pyscf.cc import gccsd_rdm
    if t2 is None: t2 = mp.t2
    doo, dvv = _gamma1_intermediates(mp, t2)
    nocc, nvir = t2.shape[1:3]
    dov = numpy.zeros((nocc,nvir))
    d1 = doo, dov, dov.T, dvv
    return gccsd_rdm._make_rdm1(mp, d1, with_frozen=True, ao_repr=ao_repr)

def _gamma1_intermediates(mp, t2):
    doo = lib.einsum('imef,jmef->ij', t2.conj(), t2) *-.5
    dvv = lib.einsum('mnea,mneb->ab', t2, t2.conj()) * .5
    return doo, dvv

# spin-orbital rdm2 in Chemist's notation
def make_rdm2(mp, t2=None, ao_repr=False):
    r'''
    Two-particle density matrix in the molecular spin-orbital representation

    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    where p,q,r,s are spin-orbitals. p,q correspond to one particle and r,s
    correspond to another particle.  The contraction between ERIs (in
    Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if t2 is None: t2 = mp.t2
    nmo0 = mp.nmo
    nocc = nocc0 = mp.nocc

    if mp.frozen is None:
        dm2 = numpy.zeros((nmo0,nmo0,nmo0,nmo0), dtype=t2.dtype) # Chemist's notation
        #dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,2,1,3) * .5 - t2.transpose(0,3,1,2) * .5
        # using t2.transpose(0,2,1,3) == -t2.transpose(0,3,1,2)
        dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,2,1,3)
        dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()
    else:
        nmo0 = mp.mo_occ.size
        nocc0 = numpy.count_nonzero(mp.mo_occ > 0)
        moidx = mp.get_frozen_mask()
        oidx = numpy.where(moidx & (mp.mo_occ > 0))[0]
        vidx = numpy.where(moidx & (mp.mo_occ ==0))[0]

        dm2 = numpy.zeros((nmo0,nmo0,nmo0,nmo0), dtype=t2.dtype) # Chemist's notation
        dm2[oidx[:,None,None,None],vidx[:,None,None],oidx[:,None],vidx] = \
                t2.transpose(0,2,1,3)
        dm2[nocc0:,:nocc0,nocc0:,:nocc0] = \
                dm2[:nocc0,nocc0:,:nocc0,nocc0:].transpose(1,0,3,2).conj()

    dm1 = make_rdm1(mp, t2)
    dm1[numpy.diag_indices(nocc0)] -= 1

    # Be careful with convention of dm1 and dm2
    #   dm1[q,p] = <p^\dagger q>
    #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
    #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
    # When adding dm1 contribution, dm1 subscripts need to be flipped
    for i in range(nocc0):
        dm2[i,i,:,:] += dm1.T
        dm2[:,:,i,i] += dm1.T
        dm2[:,i,i,:] -= dm1.T
        dm2[i,:,:,i] -= dm1

    for i in range(nocc0):
        for j in range(nocc0):
            dm2[i,i,j,j] += 1
            dm2[i,j,j,i] -= 1

    if ao_repr:
        from pyscf.cc import ccsd_rdm
        dm2 = ccsd_rdm._rdm2_mo2ao(dm2, mp.mo_coeff)
    return dm2


class GMP2(mp2.MP2):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        assert (isinstance(mf, scf.ghf.GHF))
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        mem_incore = nocc**2*nvir**2*3 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff, verbose=self.verbose)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            return _make_eris_outcore(self, mo_coeff, self.verbose)

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfgmp2
        mymp = dfgmp2.DFGMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = copy.copy(mymp.with_df)
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    energy = energy
    update_amps = update_amps
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

MP2 = GMP2

scf.ghf.GHF.MP2 = lib.class_as_method(MP2)


#TODO: Merge this _PhysicistsERIs class with gccsd._PhysicistsERIs class
class _PhysicistsERIs:
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.orbspin = None
        self.oovv = None

    def _common_init_(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')
        mp_mo_coeff = mo_coeff

        self.mol = mp.mol

        mo_idx = mp.get_frozen_mask()
        if getattr(mo_coeff, 'orbspin', None) is not None:
            self.orbspin = mo_coeff.orbspin[mo_idx]
            mo_coeff = lib.tag_array(mo_coeff[:,mo_idx], orbspin=self.orbspin)
        else:
            orbspin = scf.ghf.guess_orbspin(mo_coeff)
            mo_coeff = mo_coeff[:,mo_idx]
            if not numpy.any(orbspin == -1):
                self.orbspin = orbspin[mo_idx]
                mo_coeff = lib.tag_array(mo_coeff, orbspin=self.orbspin)
        self.mo_coeff = mo_coeff

        if mp_mo_coeff is mp._scf.mo_coeff and mp._scf.converged:
            self.mo_energy = mp._scf.mo_energy[mo_idx]
            self.fock = numpy.diag(self.mo_energy)
        else:
            dm = mp._scf.make_rdm1(mp_mo_coeff, mp.mo_occ)
            vhf = mp._scf.get_veff(mp.mol, dm)
            fockao = mp._scf.get_fock(vhf=vhf, dm=dm)
            self.fock = self.mo_coeff.conj().T.dot(fockao).dot(self.mo_coeff)
            self.mo_energy = self.fock.diagonal().real

def _make_eris_incore(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    eris = _PhysicistsERIs()
    eris._common_init_(mp, mo_coeff)

    nocc = mp.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    orbspin = eris.orbspin

    if callable(ao2mofn):
        orbo = eris.mo_coeff[:,:nocc]
        orbv = eris.mo_coeff[:,nocc:]
        if orbspin is not None:
            orbo = lib.tag_array(orbo, orbspin=orbspin[:nocc])
            orbv = lib.tag_array(orbv, orbspin=orbspin[nocc:])
        eri = ao2mofn((orbo,orbv,orbo,orbv)).reshape(nocc,nvir,nocc,nvir)
    else:
        orboa = eris.mo_coeff[:nao//2,:nocc]
        orbob = eris.mo_coeff[nao//2:,:nocc]
        orbva = eris.mo_coeff[:nao//2,nocc:]
        orbvb = eris.mo_coeff[nao//2:,nocc:]
        if orbspin is None:
            eri  = ao2mo.kernel(mp._scf._eri, (orboa,orbva,orboa,orbva))
            eri += ao2mo.kernel(mp._scf._eri, (orbob,orbvb,orbob,orbvb))
            eri1 = ao2mo.kernel(mp._scf._eri, (orboa,orbva,orbob,orbvb))
            eri += eri1
            eri += eri1.T
            eri = eri.reshape(nocc,nvir,nocc,nvir)
        else:
            co = orboa + orbob
            cv = orbva + orbvb
            eri = ao2mo.kernel(mp._scf._eri, (co,cv,co,cv)).reshape(nocc,nvir,nocc,nvir)
            sym_forbid = (orbspin[:nocc,None] != orbspin[nocc:])
            eri[sym_forbid,:,:] = 0
            eri[:,:,sym_forbid] = 0

    eris.oovv = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    return eris

def _make_eris_outcore(mp, mo_coeff=None, verbose=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mp.stdout, mp.verbose)
    eris = _PhysicistsERIs()
    eris._common_init_(mp, mo_coeff)

    nocc = mp.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    assert (eris.mo_coeff.dtype == numpy.double)
    orboa = eris.mo_coeff[:nao//2,:nocc]
    orbob = eris.mo_coeff[nao//2:,:nocc]
    orbva = eris.mo_coeff[:nao//2,nocc:]
    orbvb = eris.mo_coeff[nao//2:,nocc:]
    orbspin = eris.orbspin

    feri = eris.feri = lib.H5TmpFile()
    dtype = numpy.result_type(eris.mo_coeff).char
    eris.oovv = feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), dtype)

    if orbspin is None:
        max_memory = mp.max_memory-lib.current_memory()[0]
        blksize = min(nocc, max(2, int(max_memory*1e6/8/(nocc*nvir**2*2))))
        max_memory = max(2000, max_memory)

        fswap = lib.H5TmpFile()
        ao2mo.kernel(mp.mol, (orboa,orbva,orboa,orbva), fswap, 'aaaa',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mp.mol, (orboa,orbva,orbob,orbvb), fswap, 'aabb',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mp.mol, (orbob,orbvb,orboa,orbva), fswap, 'bbaa',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mp.mol, (orbob,orbvb,orbob,orbvb), fswap, 'bbbb',
                     max_memory=max_memory, verbose=log)

        for p0, p1 in lib.prange(0, nocc, blksize):
            tmp  = numpy.asarray(fswap['aaaa'][p0*nvir:p1*nvir])
            tmp += numpy.asarray(fswap['aabb'][p0*nvir:p1*nvir])
            tmp += numpy.asarray(fswap['bbaa'][p0*nvir:p1*nvir])
            tmp += numpy.asarray(fswap['bbbb'][p0*nvir:p1*nvir])
            tmp = tmp.reshape(p1-p0,nvir,nocc,nvir)
            eris.oovv[p0:p1] = tmp.transpose(0,2,1,3) - tmp.transpose(0,2,3,1)

    else:  # with orbspin
        orbo = orboa + orbob
        orbv = orbva + orbvb

        max_memory = mp.max_memory-lib.current_memory()[0]
        blksize = min(nocc, max(2, int(max_memory*1e6/8/(nocc*nvir**2*2))))
        max_memory = max(2000, max_memory)

        fswap = lib.H5TmpFile()
        ao2mo.kernel(mp.mol, (orbo,orbv,orbo,orbv), fswap,
                     max_memory=max_memory, verbose=log)
        sym_forbid = orbspin[:nocc,None] != orbspin[nocc:]

        for p0, p1 in lib.prange(0, nocc, blksize):
            tmp = numpy.asarray(fswap['eri_mo'][p0*nvir:p1*nvir])
            tmp = tmp.reshape(p1-p0,nvir,nocc,nvir)
            tmp[sym_forbid[p0:p1]] = 0
            tmp[:,:,sym_forbid] = 0
            eris.oovv[p0:p1] = tmp.transpose(0,2,1,3) - tmp.transpose(0,2,3,1)

    cput0 = log.timer_debug1('transforming oovv', *cput0)
    return eris

del (WITH_T2)


if __name__ == '__main__':
    from functools import reduce
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    frozen = [0,1,2,3]
    pt = GMP2(mf, frozen=frozen)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.345306881488508)

    pt.max_memory = 1
    emp2, t2 = pt.kernel()
    print(emp2 - -0.345306881488508)

    dm1 = pt.make_rdm1(t2)
    dm2 = pt.make_rdm2(t2)
    nao = mol.nao_nr()
    mo_a = mf.mo_coeff[:nao]
    mo_b = mf.mo_coeff[nao:]
    nmo = mo_a.shape[1]
    eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
    orbspin = mf.mo_coeff.orbspin
    sym_forbid = (orbspin[:,None] != orbspin)
    eri[sym_forbid,:,:] = 0
    eri[:,:,sym_forbid] = 0
    hcore = scf.RHF(mol).get_hcore()
    h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
    h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - pt.e_tot)

    mf = scf.UHF(mol).run(max_cycle=1)
    mf = scf.addons.convert_to_ghf(mf)
    pt = GMP2(mf)
    print(pt.kernel()[0] - -0.371240143556976)
