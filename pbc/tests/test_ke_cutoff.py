import numpy as np

from pyscf import gto
from pyscf.dft import rks

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import rks as pbcrks


def test_ke_cutoff(pseudo=None):

    # The periodic calculation
    eke_cut = []
    eno_cut = []
    max_ke = []
    Ls = [5, 10, 15, 20, 25, 30, 40, 50]
    for L in Ls:

        cell = pbcgto.Cell()
        cell.unit = 'B'
        cell.a = np.diag([L,L,L])
        cell.gs = np.array([20,20,20])
        cell.nimgs = [0,0,0]

        cell.atom = [['He', (L/2.,L/2.,L/2.)]]
        cell.basis = { 'He': [[0, (0.8, 1.0)],
                              [0, (1.0, 1.0)],
                              [0, (1.2, 1.0)]] }

        cell.pseudo = pseudo

        cell.ke_cutoff = 10
        cell.build()
        mf = pbcrks.RKS(cell)

        max_ke.append(np.max(0.5*np.einsum('gi,gi->g', cell.Gv, cell.Gv)))

        eke_cut.append(mf.scf())

        cell.ke_cutoff = None
        cell.build()
        mf = pbcrks.RKS(cell)

        eno_cut.append(mf.scf())

    # The basic idea is that for a fixed Ke cutoff, the
    # basis functions do not change too much even when
    # the box volume is being changed. So, one should
    # find that the energy dependence with box size is smaller
    # when a KE cutoff is employed.

    for i, L in enumerate(Ls):
        print "Ke Cutoff, L: %d, %f, %f" % (L, eke_cut[i], max_ke[i])

    # Ke Cutoff, L: 5, -2.468773, 947.482023
    # Ke Cutoff, L: 10, -2.466350, 236.870506
    # Ke Cutoff, L: 15, -2.465358, 105.275780
    # Ke Cutoff, L: 20, -2.462961, 59.217626
    # Ke Cutoff, L: 25, -2.421159, 37.899281
    # Ke Cutoff, L: 30, -2.263560, 26.318945
    # Ke Cutoff, L: 40, -2.278470, 14.804407
    # Ke Cutoff, L: 50, -3.386092, 9.474820

    for i, L in enumerate(Ls):
        print "No Cutoff, L: %d, %f, %f" % (L, eno_cut[i], max_ke[i])

    # No Cutoff, L: 5, -2.610023, 947.482023
    # No Cutoff, L: 10, -2.603423, 236.870506
    # No Cutoff, L: 15, -2.601986, 105.275780
    # No Cutoff, L: 20, -2.570535, 59.217626
    # No Cutoff, L: 25, -2.447315, 37.899281
    # No Cutoff, L: 30, -2.262098, 26.318945
    # No Cutoff, L: 40, -2.275831, 14.804407
    # No Cutoff, L: 50, -3.386092, 9.474820
