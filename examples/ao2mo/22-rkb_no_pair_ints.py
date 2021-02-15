#!/usr/bin/env python

'''
Integral transformation to compute 2-electron integrals for no-pair
Dirac-Coulomb Hamiltonian. The molecular orbitals are based on RKB basis.
'''

import h5py
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.ao2mo import r_outcore

mol = gto.M(
    atom = '''
    O   0.   0.       0.
    H   0.   -0.757   0.587
    H   0.   0.757    0.587
    ''',
    basis = 'ccpvdz',
)

mf = scf.DHF(mol).run()

def no_pair_ovov(mol, mo_coeff, erifile):
    '''
    2-electron integrals ( o v | o v ) for no-pair Hamiltonian under RKB basis
    '''
    c = lib.param.LIGHT_SPEED
    n4c, nmo = mo_coeff.shape
    n2c = n4c // 2
    nNeg = nmo // 2
    nocc = mol.nelectron
    nvir = nmo // 2 - nocc
    mo_pos_l = mo_coeff[:n2c,nNeg:]
    mo_pos_s = mo_coeff[n2c:,nNeg:] * (.5/c)
    Lo = mo_pos_l[:,:nocc]
    So = mo_pos_s[:,:nocc]
    Lv = mo_pos_l[:,nocc:]
    Sv = mo_pos_s[:,nocc:]

    dataname = 'dhf_ovov'
    def run(mos, intor):
        r_outcore.general(mol, mos, erifile,
                          dataname='tmp', intor=intor)
        blksize = 400
        nij = mos[0].shape[1] * mos[1].shape[1]
        with h5py.File(erifile) as feri:
            for i0, i1 in lib.prange(0, nij, blksize):
                buf = feri[dataname][i0:i1]
                buf += feri['tmp'][i0:i1]
                feri[dataname][i0:i1] = buf

    r_outcore.general(mol, (Lo, Lv, Lo, Lv), erifile,
                      dataname=dataname, intor='int2e_spinor')
    run((So, Sv, So, Sv), 'int2e_spsp1spsp2_spinor')
    run((So, Sv, Lo, Lv), 'int2e_spsp1_spinor'     )
    run((Lo, Lv, So, Sv), 'int2e_spsp2_spinor'     )

no_pair_ovov(mol, mf.mo_coeff, 'dhf_ovov.h5')
with h5py.File('dhf_ovov.h5', 'r') as f:
    nocc = mol.nelectron
    nvir = mol.nao * 2 - nocc
    print('Number of DHF occupied orbitals %s' % nocc)
    print('Number of DHF virtual orbitals in positive states %s' % nvir)
    print('No-pair MO integrals (ov|ov) have shape %s' % str(f['dhf_ovov'].shape))


