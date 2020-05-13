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
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin1

def contract_2e(eri, civec_strs, norb, nelec, link_index=None):
    '''Compute E_{pq}E_{rs}|CI>'''
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    eri = ao2mo.restore(1, eri, norb)
    h_ps = numpy.einsum('pqqs->ps', eri)
    ci_coeff, ci_strs = civec_strs
    strsa, strsb = ci_strs
    strsa = numpy.asarray(strsa)
    strsb = numpy.asarray(strsb)

    if link_index is None:
        cd_indexa = cre_des_linkstr(strsa, norb, neleca)
        dd_indexa = des_des_linkstr(strsa, norb, neleca)
        cd_indexb = cre_des_linkstr(strsb, norb, nelecb)
        dd_indexb = des_des_linkstr(strsb, norb, nelecb)
    else:
        cd_indexa, dd_indexa, cd_indexb, dd_indexb = link_index
    ma = len(dd_indexa)
    mb = len(dd_indexb)
    na = len(strsa)
    nb = len(strsb)

    # Adding Eq (*) below to fcinew because contract_2e function computes the
    # contraction  "E_{pq}E_{rs} V_{pqrs} |CI>" (~ p^+ q r^+ s |CI>) while
    # the actual contraction for (aa|aa) and (bb|bb) part is
    # "p^+ r^+ s q V_{pqrs} |CI>". To make (aa|aa) and (bb|bb) code reproduce
    # "p^+ q r^+ s |CI>", we employ the identity
    #    p^+ q r^+ s = p^+ r^+ s q  +  delta(qr) p^+ s
    # the second term is the source of Eq (*)

    fcivec = ci_coeff.reshape(na,nb)
    fcinew = numpy.zeros_like(fcivec)
    # (bb|aa)
    t1 = numpy.zeros((norb,norb,na,nb))
    for str1, tab in enumerate(cd_indexa):
        for a, i, str0, sign in tab:
            if a >= 0:
                t1[a,i,str1] += sign * fcivec[str0]
    fcinew += numpy.einsum('ps,psab->ab', h_ps, t1)  # (*)
    t1 = numpy.dot(eri.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    t1 = t1.reshape(norb,norb,na,nb)
    for str1, tab in enumerate(cd_indexb):
        for a, i, str0, sign in tab:
            if a >= 0:
                fcinew[:,str0] += sign * t1[a,i,:,str1]

    # (aa|bb)
    t1 = numpy.zeros((norb,norb,na,nb))
    for str1, tab in enumerate(cd_indexb):
        for a, i, str0, sign in tab:
            if a >= 0:
                t1[a,i,:,str1] += sign * fcivec[:,str0]
    fcinew += numpy.einsum('ps,psab->ab', h_ps, t1)  # (*)
    t1 = numpy.dot(eri.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    t1 = t1.reshape(norb,norb,na,nb)
    for str1, tab in enumerate(cd_indexa):
        for a, i, str0, sign in tab:
            if a >= 0:
                fcinew[str0] += sign * t1[a,i,str1]

    eri1 = eri.transpose(0,2,1,3)
    # (aa|aa)
    t1 = numpy.zeros((norb,norb,ma,nb))
    for str1, tab in enumerate(dd_indexa):
        for i, j, str0, sign in tab:
            if i >= 0:
                t1[i,j,str1] += sign * fcivec[str0]
    t1 = numpy.dot(eri1.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    t1 = t1.reshape(norb,norb,ma,nb)
    for str1, tab in enumerate(dd_indexa):
        for i, j, str0, sign in tab:
            if i >= 0:
                fcinew[str0] += sign * t1[i,j,str1]

    # (bb|bb)
    t1 = numpy.zeros((norb,norb,na,mb))
    for str1, tab in enumerate(dd_indexb):
        for i, j, str0, sign in tab:
            if i >= 0:
                t1[i,j,:,str1] += sign * fcivec[:,str0]
    t1 = numpy.dot(eri1.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    t1 = t1.reshape(norb,norb,na,mb)
    for str1, tab in enumerate(dd_indexb):
        for i, j, str0, sign in tab:
            if i >= 0:
                fcinew[:,str0] += sign * t1[i,j,:,str1]
    return fcinew.reshape(ci_coeff.shape)

def enlarge_space(myci, civec_strs, eri, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    ci_coeff, ci_strs = civec_strs
    strsa, strsb = ci_strs
    na = len(strsa)
    nb = len(strsb)
    ci_coeff = ci_coeff.reshape(na,nb)
    eri = ao2mo.restore(1, eri, norb)
    eri_pq_max = abs(eri.reshape(norb**2,-1)).max(axis=1).reshape(norb,norb)
    civec_a_max = abs(ci_coeff).max(axis=1)
    civec_b_max = abs(ci_coeff).max(axis=0)

    aidx = civec_a_max > myci.ci_coeff_cutoff
    bidx = civec_b_max > myci.ci_coeff_cutoff
    ci_coeff = ci_coeff[aidx][:,bidx]
    civec_a_max = civec_a_max[aidx]
    civec_b_max = civec_b_max[bidx]
    strsa = numpy.asarray(strsa)[aidx]
    strsb = numpy.asarray(strsb)[bidx]

    def select_strs(civec_max, strs, nelec):
        strs_add = []
        for ia, str0 in enumerate(strs):
            occ = []
            vir = []
            for i in range(norb):
                if str0 & (1<<i):
                    occ.append(i)
                else:
                    vir.append(i)
            ca = civec_max[ia]
            for i1, i in enumerate(occ):
                for a1, a in enumerate(vir):
                    if eri_pq_max[a,i]*ca > myci.select_cutoff:
                        str1 = str0 ^ (1<<i) | (1<<a)
                        strs_add.append(str1)

                        if i < nelec and a >= nelec:
                            for j in occ[:i1]:
                                for b in vir[a1+1:]:
                                    if abs(eri[a,i,b,j])*ca > myci.select_cutoff:
                                        strs_add.append(str1 ^ (1<<j) | (1<<b))
        strs_add = sorted(set(strs_add) - set(strs))
        return numpy.asarray(strs_add, dtype=int)

    strsa_add = select_strs(civec_a_max, strsa, neleca)
    strsb_add = select_strs(civec_b_max, strsb, nelecb)
    strsa = numpy.append(strsa, strsa_add)
    strsb = numpy.append(strsb, strsb_add)
    aidx = numpy.argsort(strsa)
    bidx = numpy.argsort(strsb)
    ci_strs = (strsa[aidx], strsb[bidx])

    aidx = numpy.where(aidx < ci_coeff.shape[0])[0]
    bidx = numpy.where(bidx < ci_coeff.shape[1])[0]
    ci_coeff1 = numpy.zeros((len(strsa),len(strsb)))
    lib.takebak_2d(ci_coeff1, ci_coeff, aidx, bidx)
    return ci_coeff1, ci_strs

def cre_des_linkstr(strs, norb, nelec):
    addrs = dict(zip(strs, range(len(strs))))
    nvir = norb - nelec
    link_index = numpy.zeros((len(addrs),nelec+nelec*nvir,4), dtype=int)
    link_index[:,:,0] = -1
    for i0, str1 in enumerate(strs):
        occ = []
        vir = []
        for i in range(norb):
            if str1 & (1<<i):
                occ.append(i)
            else:
                vir.append(i)
        k = 0
        for i in occ:
            link_index[i0,k] = (i, i, i0, 1)
            k += 1
        for a in vir:
            for i in occ:
                str0 = str1 ^ (1<<i) | (1<<a)
                if str0 in addrs:
                    # [cre, des, targetddress, parity]
                    link_index[i0,k] = (a, i, addrs[str0], cistring.cre_des_sign(a, i, str1))
                    k += 1
    return link_index

def des_des_linkstr(strs, norb, nelec):
    inter = []
    for str0 in strs:
        occ = [i for i in range(norb) if str0 & (1<<i)]
        for i1, i in enumerate(occ):
            for j in occ[:i1]:
                inter.append(str0 ^ (1<<i) ^ (1<<j))
    inter = sorted(set(inter))
    addrs = dict(zip(strs, range(len(strs))))

    nvir = norb - nelec + 2
    link_index = numpy.zeros((len(inter),nvir*nvir,4), dtype=int)
    link_index[:,:,0] = -1
    for i1, str1 in enumerate(inter):
        vir = [i for i in range(norb) if not str1 & (1<<i)]
        k = 0
        for i in vir:
            for j in vir:
                str0 = str1 | (1<<i) | (1<<j)
                if i != j and str0 in addrs:
                    # from intermediate str1, create i, create j -> str0
                    # (str1 = des_i des_j str0)
                    # [cre_j, cre_i, targetddress, parity]
                    sign = cistring.cre_sign(i, str1)
                    sign*= cistring.cre_sign(j, str1|(1<<i))
                    link_index[i1,k] = (i, j, addrs[str0], sign)
                    k += 1
    return link_index


def make_hdiag(h1e, g2e, ci_strs, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa, strsb = ci_strs
    strsa = numpy.asarray(strsa)
    strsb = numpy.asarray(strsb)
    occslista = [[i for i in range(norb) if str0 & (1<<i)] for str0 in strsa]
    occslistb = [[i for i in range(norb) if str0 & (1<<i)] for str0 in strsb]

    g2e = ao2mo.restore(1, g2e, norb)
    diagj = numpy.einsum('iijj->ij',g2e)
    diagk = numpy.einsum('ijji->ij',g2e)
    hdiag = []
    for aocc in occslista:
        for bocc in occslistb:
            e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag.append(e1 + e2*.5)
    return numpy.array(hdiag)

def kernel(h1e, eri, norb, nelec, ecore=0, verbose=logger.NOTE):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    namax = cistring.num_strings(norb, neleca)
    nbmax = cistring.num_strings(norb, nelecb)

    myci = SelectedCI()

    strsa = [int('1'*neleca, 2)]
    strsb = [int('1'*nelecb, 2)]
    ci_strs = (strsa, strsb)
    ci0 = numpy.ones((1,1))
    ci0, ci_strs = enlarge_space(myci, (ci0, ci_strs), h2e, norb, nelec)

    def all_linkstr_index(ci_strs):
        cd_indexa = cre_des_linkstr(ci_strs[0], norb, neleca)
        dd_indexa = des_des_linkstr(ci_strs[0], norb, neleca)
        cd_indexb = cre_des_linkstr(ci_strs[1], norb, nelecb)
        dd_indexb = des_des_linkstr(ci_strs[1], norb, nelecb)
        return cd_indexa, dd_indexa, cd_indexb, dd_indexb

    def hop(c):
        hc = contract_2e(h2e, (c, ci_strs), norb, nelec, link_index)
        return hc.reshape(-1)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)

    e_last = 0
    tol = 1e-2
    conv = False
    for icycle in range(norb):
        tol = max(tol*1e-2, myci.float_tol)
        link_index = all_linkstr_index(ci_strs)
        hdiag = make_hdiag(h1e, eri, ci_strs, norb, nelec)
        e, ci0 = lib.davidson(hop, ci0.reshape(-1), precond, tol=tol,
                              verbose=verbose)
        print('icycle %d  ci.shape %s  E = %.15g' %
              (icycle, (len(ci_strs[0]), len(ci_strs[1])), e))
        if ci0.shape == (namax,nbmax) or abs(e-e_last) < myci.float_tol*10:
            conv = True
            break
        ci1, ci_strs = enlarge_space(myci, (ci0, ci_strs), h2e, norb, nelec)
        if ci1.size < ci0.size*1.02:
            conv = True
            break
        e_last = e
        ci0 = ci1

    link_index = all_linkstr_index(ci_strs)
    hdiag = make_hdiag(h1e, eri, ci_strs, norb, nelec)
    e, ci0 = lib.davidson(hop, ci0.reshape(-1), precond, tol=myci.conv_tol,
                          verbose=verbose)

    na = len(ci_strs[0])
    nb = len(ci_strs[1])
    return e+ecore, (ci0.reshape(na,nb), ci_strs)

# dm_pq = <|p^+ q|>
def make_rdm1(civec_strs, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    ci_coeff, ci_strs = civec_strs
    strsa, strsb = ci_strs
    strsa = numpy.asarray(strsa)
    strsb = numpy.asarray(strsb)

    cd_indexa = cre_des_linkstr(strsa, norb, neleca)
    cd_indexb = cre_des_linkstr(strsb, norb, nelecb)
    na = len(strsa)
    nb = len(strsb)

    fcivec = ci_coeff.reshape(na,nb)
    rdm1 = numpy.zeros((norb,norb))
    for str1, tab in enumerate(cd_indexa):
        for a, i, str0, sign in tab:
            if a >= 0:
                rdm1[a,i] += sign * numpy.dot(fcivec[str1], fcivec[str0])

    for str1, tab in enumerate(cd_indexb):
        for a, i, str0, sign in tab:
            if a >= 0:
                rdm1[a,i] += sign * numpy.dot(fcivec[:,str1], fcivec[:,str0])
    return rdm1.T

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm2(civec_strs, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    ci_coeff, ci_strs = civec_strs
    strsa, strsb = ci_strs
    strsa = numpy.asarray(strsa)
    strsb = numpy.asarray(strsb)

    cd_indexa = cre_des_linkstr(strsa, norb, neleca)
    dd_indexa = des_des_linkstr(strsa, norb, neleca)
    cd_indexb = cre_des_linkstr(strsb, norb, nelecb)
    dd_indexb = des_des_linkstr(strsb, norb, nelecb)
    ma = len(dd_indexa)
    mb = len(dd_indexb)
    na = len(strsa)
    nb = len(strsb)

    fcivec = ci_coeff.reshape(na,nb)
    rdm2 = numpy.zeros((norb,norb,norb,norb))
    # (bb|aa) and (aa|bb)
    t1a = numpy.zeros((norb,norb,na,nb))
    t1b = numpy.zeros((norb,norb,na,nb))
    for str1, tab in enumerate(cd_indexa):
        for a, i, str0, sign in tab:
            if a >= 0:
                t1a[a,i,str1] += sign * fcivec[str0]
    for str1, tab in enumerate(cd_indexb):
        for a, i, str0, sign in tab:
            if a >= 0:
                t1b[a,i,:,str1] += sign * fcivec[:,str0]
    tmp = numpy.dot(t1a.reshape(norb**2,-1), t1b.reshape(norb**2,-1).T)
    tmp = tmp.reshape([norb]*4).transpose(1,0,2,3)
    rdm2 += tmp
    rdm2 += tmp.transpose(2,3,0,1)

    # (aa|aa)
    t1a = numpy.zeros((norb,norb,ma,nb))
    for str1, tab in enumerate(dd_indexa):
        for i, j, str0, sign in tab:
            if i >= 0:
                t1a[i,j,str1] += sign * fcivec[str0]
    tmp = numpy.dot(t1a.reshape(norb**2,-1), t1a.reshape(norb**2,-1).T)
    rdm2 -= tmp.reshape([norb]*4).transpose(0,3,1,2)

    # (bb|bb)
    t1b = numpy.zeros((norb,norb,na,mb))
    for str1, tab in enumerate(dd_indexb):
        for i, j, str0, sign in tab:
            if i >= 0:
                t1b[i,j,:,str1] += sign * fcivec[:,str0]
    tmp = numpy.dot(t1b.reshape(norb**2,-1), t1b.reshape(norb**2,-1).T)
    rdm2 -= tmp.reshape([norb]*4).transpose(0,3,1,2)
    return rdm2


class SelectedCI:
    def __init__(self):
        self.ci_coeff_cutoff = 1e-3
        self.select_cutoff = 1e-3
        self.float_tol = 1e-6
        self.conv_tol = 1e-10


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
        ['H', ( 1., 2.    , 3.   )],
        ['H', ( 1., 2.    , 4.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1, c1 = kernel(h1e, eri, norb, nelec)
    e2, c2 = direct_spin1.kernel(h1e, eri, norb, nelec)
    print(e1, e1 - -11.894559902235565, 'diff to FCI', e1-e2)

    print(c1[0].shape, c2.shape)
    dm1_1 = make_rdm1(c1, norb, nelec)
    dm1_2 = direct_spin1.make_rdm1(c2, norb, nelec)
    print(abs(dm1_1 - dm1_2).sum())
    dm2_1 = make_rdm2(c1, norb, nelec)
    dm2_2 = direct_spin1.make_rdm12(c2, norb, nelec)[1]
    print(abs(dm2_1 - dm2_2).sum())
