from pyscf import scf, gto
from pyscf.eph import eph_fd, rhf
import numpy as np

if __name__ == '__main__':
    mol = gto.M()
    mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
                ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
                ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]

    mol.unit = 'Bohr'
    mol.basis = 'sto3g'
    mol.verbose=4
    mol.build() # this is a pre-computed relaxed geometry

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-16
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    print("Force on the atoms/au:")
    print(grad)
    assert(abs(grad).max()<1e-5)
    mat, omega = eph_fd.kernel(mf)
    matmo, _ = eph_fd.kernel(mf, mo_rep=True)

    myeph = rhf.EPH(mf)
    eph, _ = myeph.kernel()
    ephmo, _ = myeph.kernel(mo_rep=True)
    print("***Testing on RHF***")
    for i in range(len(omega)):
        print("Mode %i, AO"%i,min(np.linalg.norm(eph[i]-mat[i]), np.linalg.norm(eph[i]+mat[i])))
        print("Mode %i, AO"%i, min(abs(eph[i]-mat[i]).max(), abs(eph[i]+mat[i]).max()))
        print("Mode %i, MO"%i,min(np.linalg.norm(ephmo[i]-matmo[i]), np.linalg.norm(ephmo[i]+matmo[i])))
        print("Mode %i, MO"%i, min(abs(ephmo[i]-matmo[i]).max(), abs(ephmo[i]+matmo[i]).max()))
