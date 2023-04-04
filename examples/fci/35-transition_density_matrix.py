#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Transition density matrix between singlet and triplet states
'''

import numpy as np
from pyscf import gto
from pyscf import fci
from pyscf.fci import cistring

# <T|i_alpha^+ j_beta|S>
def make_rdm1_t2s(bra, ket, norb, nelec_ket):
    neleca, nelecb = nelec = nelec_ket
    ades_index = cistring.gen_des_str_index(range(norb), neleca+1)
    bdes_index = cistring.gen_des_str_index(range(norb), nelecb)
    na_bra = cistring.num_strings(norb, neleca+1)
    nb_bra = cistring.num_strings(norb, nelecb-1)
    na_ket = cistring.num_strings(norb, neleca)
    nb_ket = cistring.num_strings(norb, nelecb)
    assert bra.shape == (na_bra, nb_bra)
    assert ket.shape == (na_ket, nb_ket)

    t1bra = np.zeros((na_ket,nb_bra,norb))
    t1ket = np.zeros((na_ket,nb_bra,norb))
    for str0, tab in enumerate(bdes_index):
        for _, i, str1, sign in tab:
            t1ket[:,str1,i] += sign * ket[:,str0]
    for str0, tab in enumerate(ades_index):
        for _, i, str1, sign in tab:
            t1bra[str1,:,i] += sign * bra[str0,:]
    dm1 = np.einsum('abp,abq->pq', t1bra, t1ket)
    return dm1

# <S|i_beta^+ j_alpha|T>
def make_rdm1_s2t(bra, ket, norb, nelec_ket):
    '''Inefficient version. A check for make_rdm1_t2s'''
    neleca, nelecb = nelec = nelec_ket
    ades_index = cistring.gen_des_str_index(range(norb), neleca)
    bcre_index = cistring.gen_cre_str_index(range(norb), nelecb)
    na_bra = cistring.num_strings(norb, neleca-1)
    nb_bra = cistring.num_strings(norb, nelecb+1)
    na_ket = cistring.num_strings(norb, neleca)
    nb_ket = cistring.num_strings(norb, nelecb)
    assert bra.shape == (na_bra, nb_bra)
    assert ket.shape == (na_ket, nb_ket)

    t1ket = np.zeros((na_bra,nb_ket,norb))
    for str0, tab in enumerate(ades_index):
        for _, i, str1, sign in tab:
            t1ket[str1,:,i] += sign * ket[str0]

    t1bra = np.zeros((na_bra,nb_bra,norb,norb))
    for str0, tab in enumerate(bcre_index):
        for a, _, str1, sign in tab:
            t1bra[:,str1,a] += sign * t1ket[:,str0]
    dm1 = np.einsum('ab,abpq->pq', bra, t1bra)
    return dm1


if __name__ == '__main__':
    mol = gto.M(
        atom = '''
        Be 0   0  0
        H  0 -.9 .3
        H  0  .9 .3
        ''',
        basis = 'sto-3g'
    )
    mf = mol.RHF().run()
    neleca, nelecb = mol.nelec
    norb = mf.mo_coeff.shape[1]

    np.set_printoptions(4, linewidth=150)
    cisolver = fci.FCI(mf)
    e_s, wfn_s = cisolver.kernel()

    cisolver.spin = 2
    e_t, wfn_t = cisolver.kernel()
    print(f'Singlet state energy = {e_s}, Triplet state energy = {e_t}')

    dm_st = make_rdm1_s2t(wfn_s, wfn_t, norb, (neleca+1, nelecb-1))
    dm_ts = make_rdm1_t2s(wfn_t, wfn_s, norb, (neleca, nelecb))
    print(abs(dm_st - dm_ts.T).max())
