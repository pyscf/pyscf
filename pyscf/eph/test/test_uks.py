from pyscf import dft, gto
from pyscf.eph import eph_fd, uks
import numpy as np

if __name__ == '__main__':
    mol = gto.M()
    mol.atom = '''O 0.000000000000 0.000000002577 0.868557119905
                  H 0.000000000000 -1.456050381698 2.152719488376
                  H 0.000000000000 1.456050379121 2.152719486067'''

    mol.unit = 'Bohr'
    mol.basis = 'sto3g'
    mol.verbose=4
    mol.build() # this is a pre-computed relaxed geometry

    mf = dft.UKS(mol)
    mf.grids.level=6
    mf.grids.build()
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-16
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    print("Force on the atoms/au:")
    print(grad)
    assert(abs(grad).max()<1e-5)
    mat, omega = eph_fd.kernel(mf)
    matmo, _ = eph_fd.kernel(mf, mo_rep=True)

    myeph = uks.EPH(mf)
    eph, _ = myeph.kernel()
    ephmo, _ = myeph.kernel(mo_rep=True)
    print("***Testing on UKS***")
    for i in range(len(omega)):
        print("Mode %i, AO"%i,min(np.linalg.norm(eph[:,i]-mat[:,i]), np.linalg.norm(eph[:,i]+mat[:,i])))
        print("Mode %i, AO"%i, min(abs(eph[:,i]-mat[:,i]).max(), abs(eph[:,i]+mat[:,i]).max()))
        print("Mode %i, MO"%i,min(np.linalg.norm(ephmo[:,i]-matmo[:,i]), np.linalg.norm(ephmo[:,i]+matmo[:,i])))
        print("Mode %i, MO"%i, min(abs(ephmo[:,i]-matmo[:,i]).max(), abs(ephmo[:,i]+matmo[:,i]).max()))
