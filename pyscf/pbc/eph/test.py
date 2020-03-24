from pyscf.pbc import gto, dft, scf
from pyscf.pbc.eph import eph_fd

if __name__ == '__main__':
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([1,1,1])
    mf = dft.KRKS(cell, kpts)
    mf.xc = 'b3lyp'
    mf.exxdiv =  None
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-7
    mf.kernel()

    mat, omega = eph_fd.kernel(mf, disp=1e-4)
    print("|Mat|_{max}",abs(mat).max())
