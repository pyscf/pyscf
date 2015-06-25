#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Contributor(s):
# - Wirawan Purwanto <wirawan0@gmail.com>
#

import numpy
import h5py


def dump_mcscf(mol, chkfile, mo_coeff,
               mcscf_energy=None, e_cas=None,
               ci_vector=None,
               iter_micro_tot=None, iter_macro=None,
               converged=None, mo_occ=None
              ):
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
    if mo_occ is not None:
        fh5['mcscf/mo_occ'] = mo_occ
    def store(key, val):
        if val is not None: fh5[key] = val
    store('mcscf/mcscf_energy', mcscf_energy)
    store('mcscf/e_cas', e_cas)
    store('mcscf/ci_vector', ci_vector)
    store('mcscf/iter_macro', iter_macro)
    store('mcscf/iter_micro_tot', iter_micro_tot)
    store('mcscf/converged', converged)
    fh5.close()


