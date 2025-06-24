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
import tempfile
import os

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

    # Remove the file if it already exists to ensure a clean start
    if os.path.exists(erifile):
        os.remove(erifile)

    with h5py.File(erifile, 'w') as feri:
        # Initial write with the first set of integrals, creating the dataset
        r_outcore.general(mol, (Lo, Lv, Lo, Lv), feri,
                          dataname=dataname, intor='int2e_spinor')

        def run_and_add(mol, mos, erifile, dataname_main, intor):
            # Use a temporary file for the intermediate integrals
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmpfile:
                tmp_erifile = tmpfile.name

            try:
                # Write intermediate integrals to the temporary file
                r_outcore.general(mol, mos, tmp_erifile,
                                  dataname='tmp', intor=intor)

                # Read the temporary data and add it to the main dataset
                with h5py.File(erifile, 'a') as main_feri, h5py.File(tmp_erifile, 'r') as tmp_feri:
                     main_dataset = main_feri[dataname_main]
                     tmp_dataset = tmp_feri['tmp']
                     main_dataset[:] += tmp_dataset[:]

            finally:
                # Clean up the temporary file
                os.remove(tmp_erifile)


        # Subsequent writes that update the dataset by adding to it
        run_and_add(mol, (So, Sv, So, Sv), erifile, dataname, 'int2e_spsp1spsp2_spinor')
        run_and_add(mol, (So, Sv, Lo, Lv), erifile, dataname, 'int2e_spsp1_spinor'     )
        run_and_add(mol, (Lo, Lv, So, Sv), erifile, dataname, 'int2e_spsp2_spinor'     )


no_pair_ovov(mol, mf.mo_coeff, 'dhf_ovov.h5')
with h5py.File('dhf_ovov.h5', 'r') as f:
    nocc = mol.nelectron
    nvir = nmo // 2 - nocc
    print('Number of DHF occupied orbitals %s' % nocc)
    print('Number of DHF virtual orbitals in positive states %s' % nvir)
    print('No-pair MO integrals (ov|ov) have shape %s' % str(f['dhf_ovov'].shape))
