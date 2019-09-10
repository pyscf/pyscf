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
# Ref:
# J. Chem. Phys. 117, 7433
#

from functools import reduce
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.scf import _vhf
from pyscf.grad import rhf as rhf_grad
from pyscf.scf import cphf

#
# LR-TDHF TDA gradients
#

def kernel(tdgrad, z, atmlst=None, mf_grad=None, max_memory=2000,
           verbose=logger.INFO):
    mol = tdgrad.mol
    mf = tdgrad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    #eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
    z = z[0].reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    def fvind(x):
        dm = numpy.einsum('pi,xij,qj->xpq', orbv, x, orbo)
        v_ao = mf.get_veff(mol, (dm+dm.transpose(0,2,1)))*2
        return numpy.einsum('pi,xpq,qj->xij', orbv, v_ao, orbo).reshape(3,-1)

    h1 = rhf_grad.get_hcore(mol)
    s1 = rhf_grad.get_ovlp(mol)

    eri1 = -mol.intor('int2e_ip1', aosym='s1', comp=3)
    eri1 = eri1.reshape(3,nao,nao,nao,nao)
    eri0 = ao2mo.kernel(mol, mo_coeff)
    eri0 = ao2mo.restore(1, eri0, nmo).reshape(nmo,nmo,nmo,nmo)
    g = eri0 * 2 - eri0.transpose(0,3,2,1)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[nocc:]

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        h1ao = h1ao + h1ao.transpose(0,2,1)
        h1mo = numpy.einsum('pi,xpq,qj->xij', mo_coeff, h1ao, mo_coeff)
        s1mo = numpy.einsum('pi,xpq,qj->xij', mo_coeff[p0:p1], s1[:,p0:p1], mo_coeff)
        s1mo = s1mo + s1mo.transpose(0,2,1)

        f1 = h1mo - numpy.einsum('xpq,pq->xpq', s1mo, zeta)
        f1-= numpy.einsum('klpq,xlk->xpq', g[:nocc,:nocc], s1mo[:,:nocc,:nocc])

        eri1a = eri1.copy()
        eri1a[:,:p0] = 0
        eri1a[:,p1:] = 0
        eri1a = eri1a + eri1a.transpose(0,2,1,3,4)
        eri1a = eri1a + eri1a.transpose(0,3,4,1,2)
        g1 = numpy.einsum('xpjkl,pi->xijkl', eri1a, mo_coeff)
        g1 = numpy.einsum('xipkl,pj->xijkl', g1, mo_coeff)
        g1 = numpy.einsum('xijpl,pk->xijkl', g1, mo_coeff)
        g1 = numpy.einsum('xijkp,pl->xijkl', g1, mo_coeff)
        g1 = g1 * 2 - g1.transpose(0,1,4,3,2)
        f1 += numpy.einsum('xkkpq->xpq', g1[:,:nocc,:nocc])
        f1ai = f1[:,nocc:,:nocc].copy()

        c1 = s1mo * -.5
        c1vo = cphf.solve(fvind, mo_energy, mo_occ, f1ai, max_cycle=50)[0]
        c1[:,nocc:,:nocc] = c1vo
        c1[:,:nocc,nocc:] = -(s1mo[:,nocc:,:nocc]+c1vo).transpose(0,2,1)
        f1 += numpy.einsum('kapq,xak->xpq', g[:nocc,nocc:], c1vo)
        f1 += numpy.einsum('akpq,xak->xpq', g[nocc:,:nocc], c1vo)

        e1  = numpy.einsum('xaijb,ai,bj->x', g1[:,nocc:,:nocc,:nocc,nocc:], z, z)
        e1 += numpy.einsum('xab,ai,bi->x', f1[:,nocc:,nocc:], z, z)
        e1 -= numpy.einsum('xij,ai,aj->x', f1[:,:nocc,:nocc], z, z)

        g1  = numpy.einsum('pjkl,xpi->xijkl', g, c1)
        g1 += numpy.einsum('ipkl,xpj->xijkl', g, c1)
        g1 += numpy.einsum('ijpl,xpk->xijkl', g, c1)
        g1 += numpy.einsum('ijkp,xpl->xijkl', g, c1)
        e1 += numpy.einsum('xaijb,ai,bj->x', g1[:,nocc:,:nocc,:nocc,nocc:], z, z)

        de[k] = e1

    return de


#class Gradients(rhf_grad.Gradients):
class Gradients(object):
    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.chkfile = td.chkfile
        self.base = td

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** LR %s gradients for %s ********',
                 self.base.__class__, self.base._scf.__class__)
        log.info('\n')

    def kernel(self, z):
        return kernel(self, z)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import dft
    from pyscf import tddft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.)], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol)
    mf.scf()
    td = tddft.rhf.TDA(mf)
    td.nstates = 3
    e, z = td.kernel()
    #print(e[0] + mf.e_tot)
    tdg = Gradients(td)
    hfg = rhf_grad.Gradients(mf)
    g1 = tdg.kernel(z[0])
    g2 = hfg.kernel()
    print(g1)# + g2
# 0  0  0.3021705380000239

