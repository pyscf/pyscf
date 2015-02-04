import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import ao2mo
from pyscf import fci

def run(i, dm0, mo0):
    x = i
    y = (2.54 - 0.46 * x)
    x = x * 0.529177249
    y = y * 0.529177249
    mol = gto.M(
        verbose = 5,
        output = 'out-%2.1f'%i,
        atom = [
            ['Be',( 0., 0.    , 0.   )],
            ['H', ( x, -y    , 0.    )],
            ['H', ( x,  y    , 0.    )],],
        basis = '6-311G',
        symmetry = True)

    mf = scf.RHF(mol)
    ehf = mf.scf(dm0)
    mf.analyze()

    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fcisolver.davidson_only = True # force the CI solver stick on (A1)^2(B1)^0 configuration
    if mo0 is not None:
        mo0 = mcscf.project_init_guess(mc, mo0)

    emc = mc.mc1step(mo0)[0]
    mc.analyze()

    print('%2.1f bohr, HF energy: %12.8f, CASSCF energy: %12.8f' % (i, ehf, emc))
    return mf, mc

dm0 = mo0 = None
for i in reversed(numpy.arange(1.0, 4.1, .1)):
    mf, mc = run(i, dm0, mo0)
    dm0 = mf.make_rdm1()
    mo_coeff = mc.mo_coeff

