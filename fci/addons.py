#!/usr/bin/env python

import numpy
from pyscf.fci import cistring
from pyscf import symm

def large_ci(ci, norb, nelec, tol=.1):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec//2
    else:
        neleca, nelecb = nelec
    idx = numpy.argwhere(abs(ci) > tol)
    res = []
    for i,j in idx:
        res.append((ci[i,j],
                    bin(cistring.addr2str(norb, neleca, i)),
                    bin(cistring.addr2str(norb, nelecb, j))))
    return res

def initguess_triplet(norb, nelec, binstring):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec//2
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    addr = cistring.str2addr(norb, neleca, int(binstring,2))
    ci0 = numpy.zeros((na,nb))
    ci0[addr,0] = numpy.sqrt(.5)
    ci0[0,addr] =-numpy.sqrt(.5)
    return ci0

def symm_initguess(norb, nelec, orbsym, wfnsym=0, irrep_nelec=None):
    '''CI wavefunction initial guess which matches the given symmetry.
    Based on RHF/ROHF orbitals
    '''
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    orbsym = numpy.asarray(orbsym)
    if not isinstance(orbsym[0], numpy.integer):
        raise RuntimeError('TODO: convert irrep symbol to irrep id')

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci1 = numpy.zeros((na,nb))

    orbleft = numpy.ones(norb, dtype=bool)
    stra = numpy.zeros(norb, dtype=bool)
    strb = numpy.zeros(norb, dtype=bool)
    if irrep_nelec is not None:
        for k,n in irrep_nelec.items():
            orbleft[orbsym==k] = False
            if isinstance(n, (int, numpy.integer)):
                idx = numpy.where(orbsym==k)[0][:n//2]
                stra[idx] = True
                strb[idx] = True
            else:
                na, nb = n
                stra[numpy.where(orbsym==k)[0][:na]] = True
                strb[numpy.where(orbsym==k)[0][:nb]] = True
                if (na-nb)%2:
                    wfnsym ^= k

    orbleft = numpy.where(orbleft)[0]
    neleca_left = neleca - stra.sum()
    nelecb_left = nelecb - strb.sum()
    spin = neleca_left - nelecb_left
    assert(neleca_left >= 0)
    assert(nelecb_left >= 0)
    assert(spin >= 0)

# assume "nelecb_left" doubly occupied orbitals
    if spin == 0:
        assert(wfnsym == 0)
        socclst = []
    else:
        socclst = orbleft[symm.route(wfnsym, spin, orbsym[orbleft])]
        if len(socclst) != spin:
            raise RuntimeError('No occ pattern found for wfnsym %s' % wfnsym)
    docclst = numpy.zeros(norb, dtype=bool)
    docclst[orbleft] = True
    docclst[socclst] = False
    docclst = numpy.where(docclst)[0][:nelecb_left]

    stra[docclst] = True
    strb[docclst] = True
    stra[socclst] = True
    stra = ''.join([str(int(i)) for i in stra])[::-1]
    strb = ''.join([str(int(i)) for i in strb])[::-1]
    #print stra, strb
    addra = cistring.str2addr(norb, neleca, int(stra,2))
    addrb = cistring.str2addr(norb, nelecb, int(strb,2))
    ci1[addra,addrb] = 1
    return ci1

def symmetrize_wfn(ci, norb, nelec, orbsym, wfnsym=0):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa = numpy.asarray(cistring.gen_strings4orblist(range(norb), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(norb), nelecb))
    airreps = numpy.zeros(strsa.size, dtype=numpy.int32)
    birreps = numpy.zeros(strsb.size, dtype=numpy.int32)
    for i in range(norb):
        airreps[numpy.bitwise_and(strsa, 1<<i) > 0] ^= orbsym[i]
        birreps[numpy.bitwise_and(strsb, 1<<i) > 0] ^= orbsym[i]
    #print(airreps)
    #print(birreps)
    mask = (numpy.bitwise_xor(airreps.reshape(-1,1), birreps) == wfnsym)
    ci1 = numpy.zeros_like(ci)
    ci1[mask] = ci[mask]
    return ci1


# construct (N-1)-electron wavefunction by removing an alpha electron from
# N-electron wavefunction:
# |N-1> = a_p |N>
def des_a(ci0, norb, nelec, ap_id):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec

    des_index = cistring.gen_des_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca-1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]))

    entry_has_ap = (des_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3]
    #print(addr_ci0)
    #print(addr_ci1)
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

# construct (N-1)-electron wavefunction by removing a beta electron from
# N-electron wavefunction:
def des_b(ci0, norb, nelec, ap_id):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec
    des_index = cistring.gen_des_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb-1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1))

    entry_has_ap = (des_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3]
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1

# construct (N+1)-electron wavefunction by adding an alpha electron to
# N-electron wavefunction:
# |N+1> = a_p^+ |N>
def cre_a(ci0, norb, nelec, ap_id):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec
    cre_index = cistring.gen_cre_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca+1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]))

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3]
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

# construct (N+1)-electron wavefunction by adding a beta electron to
# N-electron wavefunction:
def cre_b(ci0, norb, nelec, ap_id):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec
    cre_index = cistring.gen_cre_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb-1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1))

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3]
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1


def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    from pyscf.fci import direct_spin1
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = direct_spin1.contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))


def reorder(ci, nelec, orbidxa, orbidxb=None):
    '''reorder the CI coefficients wrt the reordering of orbitals (The relation
    of the reordered orbitals and original orbitals is  new = old[idx]).  Eg.
    orbidx = [2,0,1] to map   old orbital  a b c  ->   new orbital  c a b
    old-strings   0b011, 0b101, 0b110 ==  (1,2), (1,3), (2,3)
    orb-strings   (3,1), (3,2), (1,2) ==  0B101, 0B110, 0B011    <= by gen_strings4orblist
    then argsort to translate the string representation to the address
    [2(=0B011), 0(=0B101), 1(=0B110)]
    '''
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec
    if orbidxb is None:
        orbidxb = orbidxa
    guide_stringsa = cistring.gen_strings4orblist(orbidxa, neleca)
    guide_stringsb = cistring.gen_strings4orblist(orbidxb, nelecb)
    old_det_idxa = numpy.argsort(guide_stringsa)
    old_det_idxb = numpy.argsort(guide_stringsb)
    return ci.take(old_det_idxa, axis=0).take(old_det_idxb, axis=1)

def overlap(string1, string2, norb, s=None):
    '''Determinants overlap on non-orthogonal one-particle basis'''
    if s is None:  # orthogonal basis with s_ij = delta_ij
        return string1 == string2
    else:
        if isinstance(string1, str):
            nelec = string1.count('1')
            string1 = int(string1, 2)
        else:
            nelec = bin(string1).count('1')
        if isinstance(string2, str):
            assert(string2.count('1') == nelec)
            string2 = int(string2, 2)
        else:
            assert(bin(string2).count('1') == nelec)
        idx1 = [i for i in range(norb) if (1<<i & string1)]
        idx2 = [i for i in range(norb) if (1<<i & string2)]
        s1 = numpy.take(numpy.take(s, idx1, axis=0), idx2, axis=1)
        return numpy.linalg.det(s1)


if __name__ == '__main__':
    a4 = 10*numpy.arange(4)[:,None]
    a6 = 10*numpy.arange(6)[:,None]
    b4 = numpy.arange(4)
    b6 = numpy.arange(6)
    print([bin(i) for i in cistring.gen_strings4orblist(range(4), 3)])
    print([bin(i) for i in cistring.gen_strings4orblist(range(4), 2)])
    print(des_a(a4+b4, 4, 6, 0))
    print(des_a(a4+b4, 4, 6, 1))
    print(des_a(a4+b4, 4, 6, 2))
    print(des_a(a4+b4, 4, 6, 3))
    print('-------------')
    print(des_b(a6+b4, 4, (2,3), 0))
    print(des_b(a6+b4, 4, (2,3), 1))
    print(des_b(a6+b4, 4, (2,3), 2))
    print(des_b(a6+b4, 4, (2,3), 3))
    print('-------------')
    print(cre_a(a6+b4, 4, (2,3), 0))
    print(cre_a(a6+b4, 4, (2,3), 1))
    print(cre_a(a6+b4, 4, (2,3), 2))
    print(cre_a(a6+b4, 4, (2,3), 3))
    print('-------------')
    print(cre_b(a6+b6, 4, 4, 0))
    print(cre_b(a6+b6, 4, 4, 1))
    print(cre_b(a6+b6, 4, 4, 2))
    print(cre_b(a6+b6, 4, 4, 3))

    symm_initguess(6, (4,3), [0,1,0,0,3,0], wfnsym=1, irrep_nelec=None)
    symm_initguess(6, (4,3), [0,1,0,0,3,0], wfnsym=0, irrep_nelec={0:[3,2],3:2})

    norb = 6
    nelec = neleca, nelecb = 4,3
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci = numpy.ones((na,nb))
    print(symmetrize_wfn(ci, norb, nelec, [0,6,0,3,5,2], 2))
    s1 = numpy.random.seed(1)
    s1 = numpy.random.random((6,6))
    s1 = s1 + s1.T
    print(overlap(int('0b10011',2), int('0b011010',2), 6, s1) - -0.273996425116)
