#!/usr/bin/env python

"""Solve a two-site electron-phonon model with the direct_ep FCI solver.

The model combines electron hopping, on-site Hubbard repulsion, local
electron-phonon coupling, and a phonon Hamiltonian. The electron numbers
are fixed to one alpha and one beta electron.
"""

import numpy

# A Hubbard-Holstein model also includes quantized phonons. direct_ep uses
# a maximum phonon occupation per site; increase it to check convergence.
from pyscf.fci import direct_ep
nsite, nelec, nphonon = 2, (1, 1), 3
hopping = numpy.array([[0., -1.], [-1., 0.]])
hphonon = numpy.array([[1.1, .1], [.1, 1.1]])
u, coupling = 1.5, .5
e_ep, ci_ep = direct_ep.kernel(hopping, u, coupling, hphonon,
                              nsite, nelec, nphonon, tol=1e-10)
electron_dm1, electron_dm2 = direct_ep.make_rdm12e(ci_ep, nsite, nelec)
phonon_dm1 = direct_ep.make_rdm1p(ci_ep, nsite, nelec, nphonon)
print('Electron-phonon energy', e_ep)
print('Electron and phonon density matrices', electron_dm1, phonon_dm1)
