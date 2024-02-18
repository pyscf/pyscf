import pyscf
import dftd4.pyscf as disp


xyz = """ H    0    0    -1.403
Br   0    0    0.040"""

mol = pyscf.M(
    atom=xyz,
    basis="vDZP",
    ecp="vDZP",
    charge=0,
    spin=0,
    symmetry=True,
    verbose=3,
)

mf = pyscf.scf.KS(mol)
mf.xc = "wb97x_v"

mf.nlc = None
mf._numint.libxc.is_nlc = lambda x: False

mf.disp = "d4"

energy = mf.kernel()
