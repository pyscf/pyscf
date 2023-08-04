import numpy
from pyscf import gto, scf, mcscf

'''
Scan BeH2 molecule symmetric dissociation curve

Note the CI wave function might change symmetry in the scanning.  Adjust
fcisolver parameters to maintain the right symmetry.
'''

def run(i, dm0, mo0, ci0):
    x = i
    y = (2.54 - 0.46 * x)
    x = x * 0.529177249
    y = y * 0.529177249
    mol = gto.M(
        verbose = 0,
        atom = [
            ['Be',( 0., 0.    , 0.   )],
            ['H', ( x, -y    , 0.    )],
            ['H', ( x,  y    , 0.    )],],
        basis = '6-311G',
        symmetry = True)

    mf = scf.RHF(mol)
    ehf = mf.scf(dm0)

    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fcisolver.davidson_only = True # force the CI solver stick on (A1)^2(B1)^0 configuration
    if mo0 is not None:
        mo0 = mcscf.project_init_guess(mc, mo0)

    emc = mc.mc1step(mo0, ci0)[0]

    print('%2.1f bohr, HF energy: %12.8f, CASSCF energy: %12.8f' % (i, ehf, emc))
    return mf, mc

dm0 = mo0 = ci = None
for i in reversed(numpy.arange(1.0, 4.1, .1)):
    mf, mc = run(i, dm0, mo0, ci)
    dm0 = mf.make_rdm1()
    mo_coeff = mc.mo_coeff
    ci = mc.ci
