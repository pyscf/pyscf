#!/usr/bin/env python
#
# Contributor(s):
# - Wirawan Purwanto <wirawan0@gmail.com>
# - Qiming Sun <osirpt.sun@gmail.com>
#
#

import numpy
import h5py


def dump_mcscf(mol, chkfile, e_tot, mo_coeff, ncore, ncas,
               mo_occ=None, mo_energy=None,
               e_cas=None, ci_vector=None):
    """Dumps MCSCF/CASSCF calculation to checkpoint file.
    """
    if h5py.is_hdf5(chkfile):
        fh5 = h5py.File(chkfile)
        if 'mcscf' in fh5:
            del(fh5['mcscf'])
    else:
        fh5 = h5py.File(chkfile, 'w')
    if 'mol' not in fh5:
        fh5['mol'] = format(mol.pack())

    fh5['mcscf/mo_coeff'] = mo_coeff
    def store(key, val):
        if val is not None: fh5[key] = val
    store('mcscf/e_tot', e_tot)
    store('mcscf/e_cas', e_cas)
    store('mcscf/ci', ci)
    store('mcscf/ncore', ncore)
    store('mcscf/ncas', ncas)
    store('mcscf/mo_occ', mo_occ)
    store('mcscf/mo_energy', mo_energy)
    fh5.close()


