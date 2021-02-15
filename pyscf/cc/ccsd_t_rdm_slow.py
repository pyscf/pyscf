#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import numpy
from pyscf import lib
from pyscf.cc import ccsd_rdm

def _gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None, for_grad=False):
    doo, dov, dvo, dvv = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None: eris = mycc.ao2mo()
    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovov)

    mo_e = eris.mo_energy
    eia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)

    w =(numpy.einsum('iafb,kjcf->ijkabc', eris_ovvv.conj(), t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo.conj(), t2)) / d3
    v =(numpy.einsum('iajb,kc->ijkabc', eris_ovov.conj(), t1)
      + numpy.einsum('ck,ijab->ijkabc', eris.fock[nocc:,:nocc], t2)) / d3
    w = p6(w)
    v = p6(v)
    wv = w + v * .5
    rw = r6(w)
    goo = numpy.einsum('iklabc,jklabc->ij', wv.conj(), rw)
    gvv = numpy.einsum('ijkacd,ijkbcd->ab', wv, rw.conj())

    if not for_grad:
# t3 amplitudes in CCSD(T) is computed non-iteratively. The off-diagonal
# blocks of fock matrix does not contribute to CCSD(T) energy. To make Tr(H,D)
# consistent to the CCSD(T) total energy, the density matrix off-diagonal
# parts are excluded.
        doo[numpy.diag_indices(nocc)] -= goo.diagonal() * .5
        dvv[numpy.diag_indices(nvir)] += gvv.diagonal() * .5

    else:
# The off-diagonal blocks of fock matrix have small contributions to analytical
# nuclear gradients.
        doo -= goo * .5
        dvv += gvv * .5

    dvo += numpy.einsum('ijab,ijkabc->ck', t2.conj(), rw) * .5
    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
def _gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None,
                          compress_vvvv=False):
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            ccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
    if eris is None: eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovov)

    mo_e = eris.mo_energy
    eia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)

    w =(numpy.einsum('iafb,kjcf->ijkabc', eris_ovvv.conj(), t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo.conj(), t2)) / d3
    v =(numpy.einsum('iajb,kc->ijkabc', eris_ovov.conj(), t1)
      + numpy.einsum('ck,ijab->ijkabc', eris.fock[nocc:,:nocc], t2)) / d3
    w = p6(w)
    v = p6(v)
    rw = r6(w)
    rwv = r6(w * 2 + v * .5)

    dovov += numpy.einsum('kc,ijkabc->iajb', t1, rw.conj()) * .5
    dooov -= numpy.einsum('mkbc,ijkabc->jmia', t2, rwv.conj())
    # Note "dovvv +=" also changes the value of dvvov
    dovvv += numpy.einsum('kjcf,ijkabc->iafb', t2, rwv.conj())
    dvvov = dovvv.transpose(2,3,0,1)

    if compress_vvvv:
        nvir = mycc.nmo - mycc.nocc
        idx = numpy.tril_indices(nvir)
        vidx = idx[0] * nvir + idx[1]
        dvvvv = dvvvv + dvvvv.transpose(1,0,2,3)
        dvvvv = dvvvv + dvvvv.transpose(0,1,3,2)
        dvvvv = lib.take_2d(dvvvv.reshape(nvir**2,nvir**2), vidx, vidx)
        dvvvv *= .25

    return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov

def _gamma2_outcore(mycc, t1, t2, l1, l2, eris, h5fobj, compress_vvvv=False):
    return _gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv)

def make_rdm1(mycc, t1, t2, l1, l2, eris=None, ao_repr=False):
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    return ccsd_rdm._make_rdm1(mycc, d1, True, ao_repr=ao_repr)

# rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, eris=None):
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    return ccsd_rdm._make_rdm2(mycc, d1, d2, True, True)


def p6(t):
    t1 = t + t.transpose(0,2,1,3,5,4)
    return t1 + t1.transpose(1,0,2,4,3,5) + t1.transpose(1,2,0,4,5,3)
def r6(w):
    return (4 * w + w.transpose(0,1,2,4,5,3) + w.transpose(0,1,2,5,3,4)
            - 2 * w.transpose(0,1,2,5,4,3) - 2 * w.transpose(0,1,2,3,5,4)
            - 2 * w.transpose(0,1,2,4,3,5))


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf.cc import ccsd_t_slow as ccsd_t
    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
    from pyscf import ao2mo

    mol = gto.Mole()
    #mol.verbose = 5
    #mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.957 , .587)],
        [1 , (0.2,  .757 , .487)]]

    #mol.basis = 'ccpvdz'
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-1
    mf.scf()
    mcc = ccsd.CCSD(mf)
    mcc.conv_tol = 1e-14
    ecc, t1, t2 = mcc.kernel()
    eris = mcc.ao2mo()
    e3ref = ccsd_t.kernel(mcc, eris, t1, t2)
    l1, l2 = ccsd_t_lambda.kernel(mcc, eris, t1, t2)[1:]
    print(ecc, e3ref)

    eri_mo = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False)
    nmo = mf.mo_coeff.shape[1]
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    dm1 = make_rdm1(mcc, t1, t2, l1, l2, eris=eris)
    dm2 = make_rdm2(mcc, t1, t2, l1, l2, eris=eris)
    print(lib.finger(dm1) - 1.289951975176953 )
    print(lib.finger(dm2) - 6.6184784979411164)
    h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    e3 =(numpy.einsum('ij,ji->', h1, dm1)
       + numpy.einsum('ijkl,ijkl->', eri_mo, dm2)*.5 + mf.mol.energy_nuc())
    print(e3ref, e3-(mf.e_tot+ecc))

