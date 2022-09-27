import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc import scf,cc
from pyscf     import cc as mol_cc
from pyscf.pbc.tools.pbc import super_cell
import pyscf

def test_kuccsd_openshell():
    cell = gto.M(
        unit = 'B',
        a = [[ 0.,          6.74027466,  6.74027466],
             [ 6.74027466,  0.,          6.74027466],
             [ 6.74027466,  6.74027466,  0.        ]],
        mesh = [13]*3,
        atom = '''H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733''',
        basis = [[0, (1., 1.)], [0, (.5, 1.)]],
        verbose = 1,
        charge = 0,
        spin = 1,
    )

    nmp = [3,1,1]
    # cell spin multiplied by nkpts
    cell.spin = cell.spin*3

    # treating 3*1*1 supercell at gamma point
    supcell = super_cell(cell,nmp)
    umf  = scf.UHF(supcell,exxdiv=None)
    umf.conv_tol = 1e-11
    ehf  = umf.kernel()

    ucc  = cc.UCCSD(umf)
    ucc.conv_tol = 1e-12
    ecc, t1, t2 = ucc.kernel()
    print('UHF energy (supercell) %.9f \n' % (float(ehf)/3.))
    print('UCCSD correlation energy (supercell) %.9f \n' % (float(ecc)/3.))
    assert abs(ehf / 3 - -1.003789445) < 1e-7
    assert abs(ecc / 3 - -0.029056542) < 1e-6

    # kpts calculations
    kpts  = cell.make_kpts(nmp)
    kpts -= kpts[0]
    kmf   = scf.KUHF(cell,kpts,exxdiv=None)
    kmf.conv_tol = 1e-11
    ehf   = kmf.kernel()
    kcc   = cc.KUCCSD(kmf)
    kcc.conv_tol = 1e-12
    ecc, t1, t2 = kcc.kernel()
    print('UHF energy (kpts) %.9f \n' % (float(ehf)))
    print('UCCSD correlation energy (kpts) %.9f \n' % (float(ecc)))
    assert abs(ehf - -1.003789445) < 1e-7
    assert abs(ecc - -0.029056542) < 1e-6
