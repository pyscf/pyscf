import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc import scf,cc
from pyscf     import cc as mol_cc
from pyscf.pbc.tools.pbc import super_cell

# generating the cell

def test_kuccsd_supercell_vs_kpts_high_cost():
    cell = gto.M(
        unit = 'B',
        a = [[ 0.,          3.37013733,  3.37013733],
             [ 3.37013733,  0.,          3.37013733],
             [ 3.37013733,  3.37013733,  0.        ]],
        mesh = [13]*3,
        atom = '''He 0 0 0
                  He 1.68506866 1.68506866 1.68506866''',
        basis = [[0, (1., 1.)], [0, (.5, 1.)]],
        verbose = 0,
    )

    nmp = [3,3,1]

    # treating supercell at gamma point
    supcell = super_cell(cell,nmp)

    gmf  = scf.UHF(supcell,exxdiv=None)
    ehf  = gmf.kernel()
    gcc  = cc.UCCSD(gmf)
    ecc, t1, t2 = gcc.kernel()
    print('UHF energy (supercell) %f \n' % (float(ehf)/numpy.prod(nmp)))
    print('UCCSD correlation energy (supercell) %f \n' % (float(ecc)/numpy.prod(nmp)))
    assert abs(ehf / 9 - -4.343308413289) < 1e-7
    assert abs(ecc / 9 - -0.009470753047083676) < 1e-6

    # treating mesh of k points

    kpts  = cell.make_kpts(nmp)
    kpts -= kpts[0]
    kmf   = scf.KUHF(cell,kpts,exxdiv=None)
    ehf   = kmf.kernel()
    kcc   = cc.KUCCSD(kmf)
    ecc, t1, t2 = kcc.kernel()

    print('UHF energy (kpts) %f \n' % ehf)
    print('UCCSD correlation energy (kpts) %f \n' % ecc)
    assert abs(ehf - -4.343308413289) < 1e-7
    assert abs(ecc - -0.009470753047083676) < 1e-6
