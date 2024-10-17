#!/usr/bin/env python
#
# Author: Susi Lehtola <susi.lehtola@gmail.com>
#

from pyscf import gto, symm
import basis_set_exchange

'''
Interface with the Basis Set Exchange's Python backend.
https://github.com/MolSSI-BSE/basis_set_exchange

This example illustrates how basis sets can be directly obtained from
a local copy of the Basis Set Exchange.
'''

# PySCF has a native interface to the Basis Set Exchange, which is
# used as a fall-through if the basis set was not found within PySCF
# itself
mol = gto.M(
    atom = '''
N          0.6683566134        0.2004327755        0.0000000000
H          0.9668193796       -0.3441960976        0.8071193402
H          0.9668193796       -0.3441960976       -0.8071193402
F         -0.7347916126       -0.0467759204        0.0000000000
''',
    basis = 'HGBSP1-7',
    verbose = 4
)


# One can also use the Basis Set Exchange in a more direct fashion
mol = gto.M(
    atom = '''
N          0.6683566134        0.2004327755        0.0000000000
H          0.9668193796       -0.3441960976        0.8071193402
H          0.9668193796       -0.3441960976       -0.8071193402
F         -0.7347916126       -0.0467759204        0.0000000000
''',
    basis = {
        # The BSE returns the basis in NWChem format, which is then
        # parsed in PySCF as usual. It is also easy to mix various
        # basis sets, although this should only be done by experts.
        'H' : gto.load(basis_set_exchange.api.get_basis('sto-2g', elements='H', fmt='nwchem'), 'H'),
        'N' : gto.load(basis_set_exchange.api.get_basis('svp (dunning-hay)', elements='N', fmt='nwchem'), 'N'),
        'F' : gto.load(basis_set_exchange.api.get_basis('2zapa-nr', elements='F', fmt='nwchem'), 'F'),
    },
    verbose = 4
)


mol = gto.M(
    atom = '''
N          0.6683566134        0.2004327755        0.0000000000
H          0.9668193796       -0.3441960976        0.8071193402
H          0.9668193796       -0.3441960976       -0.8071193402
F         -0.7347916126       -0.0467759204        0.0000000000
''',
    basis = {
        # A more usual use case: use a similar family basis set on all
        # atoms, but use a more diffuse basis set on the heavy atoms
        # than on the hydrogens. This example uses aug-cc-pVDZ on the
        # hydrogens, and d-aug-cc-pVDZ on the heavy atoms.
        'H' : gto.load(basis_set_exchange.api.get_basis('aug-cc-pVDZ', augment_diffuse=0, elements='H', fmt='nwchem'), 'H'),
        'N' : gto.load(basis_set_exchange.api.get_basis('aug-cc-pVDZ', augment_diffuse=1, elements='N', fmt='nwchem'), 'N'),
        'F' : gto.load(basis_set_exchange.api.get_basis('aug-cc-pVDZ', augment_diffuse=1, elements='F', fmt='nwchem'), 'F'),
    },
    verbose = 4
)


mol = gto.M(
    atom = '''
N          0.6683566134        0.2004327755        0.0000000000
H          0.9668193796       -0.3441960976        0.8071193402
H          0.9668193796       -0.3441960976       -0.8071193402
F         -0.7347916126       -0.0467759204        0.0000000000
''',
    basis = {
        # This example prepares a minimal basis from cc-pVTZ by
        # removing all the free primitives in the basis set, leaving
        # only the fully contracted functions.
        'H' : gto.load(basis_set_exchange.api.get_basis('cc-pVTZ', remove_free_primitives=True, elements='H', fmt='nwchem'), 'H'),
        'N' : gto.load(basis_set_exchange.api.get_basis('cc-pVTZ', remove_free_primitives=True, elements='N', fmt='nwchem'), 'N'),
        'F' : gto.load(basis_set_exchange.api.get_basis('cc-pVTZ', remove_free_primitives=True, elements='F', fmt='nwchem'), 'F'),
    },
    verbose = 4
)
