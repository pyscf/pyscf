#!/usr/bin/env python

import numpy
from pyscf.fci import cistring

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
    old-strings   0b011, 0b101, 0b110
              ==  (1,2), (1,3), (2,3)
    orb-strings   (3,1), (3,2), (1,2)
              ==  0B101, 0B110, 0B011    <= by gen_strings4orblist
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
