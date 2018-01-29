#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

def convert_to_gcisd(myci):
    from pyscf import scf
    from pyscf.ci import gcisd
    if isinstance(myci, gcisd.GCISD):
        return myci

    mf = scf.addons.convert_to_ghf(myci._scf)
    gci = gcisd.GCISD(mf)
    assert(myci._nocc is None)
    assert(myci._nmo is None)
    gci.__dict__.update(myci.__dict__)
    gci._scf = mf
    gci.mo_coeff = mf.mo_coeff
    gci.mo_occ = mf.mo_occ
    if isinstance(myci.frozen, (int, np.integer)):
        gci.frozen = myci.frozen * 2
    else:
        raise NotImplementedError
    gci.ci = gcisd.from_rcisdvec(myci.ci, myci.nocc, mf.mo_coeff.orbspin)
    return gci
