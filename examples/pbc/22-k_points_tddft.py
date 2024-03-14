#!/usr/bin/env python
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

import numpy as np
from pyscf.pbc import gto, scf, tdscf
from pyscf import scf as molscf
from pyscf import lib


'''
TDSCF with k-point sampling
'''
atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
a = '''
1.7850000000 1.7850000000 0.0000000000
0.0000000000 1.7850000000 1.7850000000
1.7850000000 0.0000000000 1.7850000000
'''
basis = '''
C  S
    9.031436   -1.960629e-02
    3.821255   -1.291762e-01
    0.473725    5.822572e-01
C  P
    4.353457    8.730943e-02
    1.266307    2.797034e-01
    0.398715    5.024424e-01
''' # a trimmed DZ basis for fast test
pseudo = 'gth-hf-rev'
cell = gto.M(atom=atom, basis=basis, a=a, pseudo=pseudo).set(verbose=3)
kmesh = [2,1,1]
kpts = cell.make_kpts(kmesh)
nkpts = len(kpts)
mf = scf.KRHF(cell, kpts).rs_density_fit().run()

log = lib.logger.new_logger(mf)

''' k-point TDSCF solutions can have non-zero momentum transfer between particle and hole.
    This can be controlled by `td.kshift_lst`. By default, kshift_lst = [0] and only the
    zero-momentum transfer solution (i.e., 'vertical' in k-space) will be solved, as
    demonstrated in the example below.
'''
td = mf.TDA().set(nstates=5).run()
log.note('RHF-TDA:')
for kshift,es in zip(td.kshift_lst,td.e):
    log.note('kshift = %d  Eex = %s', kshift, ' '.join([f'{e:.3f}' for e in es*27.2114]))

''' If GDF/RSDF is used as the density fitting method (as in this example), solutions
    with non-zero particle-hole momentum-transfer solution is also possible. The example
    below demonstrates how to calculate solutions with all possible kshift.

    NOTE: if FFTDF is used, pyscf will set `kshift_lst` to [0].
'''
td = mf.TDHF().set(nstates=5, kshift_lst=list(range(nkpts))).run()
log.note('RHF-TDHF:')
for kshift,es in zip(td.kshift_lst,td.e):
    log.note('kshift = %d  Eex = %s', kshift, ' '.join([f'{e:.3f}' for e in es*27.2114]))


''' TDHF at a single k-point compared to molecular TDSCF
'''
atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
'''
a = np.eye(3) * 20  # big box to match isolated molecule
basis = 'sto-3g'
auxbasis = 'weigend'
cell = gto.M(atom=atom, basis=basis, a=a).set(verbose=3)
mol = cell.to_mol()

log = lib.logger.new_logger(cell)

xc = 'b3lyp'
# pbc
mf = scf.RKS(cell).set(xc=xc).rs_density_fit(auxbasis=auxbasis).run()
pbctda = mf.TDA().run()
pbctd = mf.TDDFT().run()
# mol
molmf =  molscf.RKS(cell).set(xc=xc).density_fit(auxbasis=auxbasis).run()
moltda = molmf.TDA().run()
moltd = molmf.TDDFT().run()

_format = lambda e: ' '.join([f'{x*27.2114:.3f}' for x in e])
log.note('PBC TDA  : %s', _format(pbctda.e))
log.note('Mol TDA  : %s', _format(moltda.e))
log.note('PBC TDDFT: %s', _format(pbctd.e))
log.note('Mol TDDFT: %s', _format(moltd.e))
