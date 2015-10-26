import numpy as np

def fft(f, gs):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    Re: MH (3.25), we assume Ns := ngs = 2*gs+1

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of :func:`cartesian_prod`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.
    
    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    ngs = 2*gs+1
    f3d = np.reshape(f, ngs)
    g3d = np.fft.fftn(f3d)
    return np.ravel(g3d)

def ifft(g, gs):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `span3`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.
    
    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    ngs = 2*gs+1
    g3d = np.reshape(g, ngs)
    f3d = np.fft.ifftn(g3d)
    return np.ravel(f3d)
    
def get_coulG(cell):
    '''Calculate the Coulomb kernel 4*pi/G^2 for all G-vectors (0 for G=0).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coulG : (ngs,) ndarray
            The Coulomb kernel.

    '''
    Gv = cell.Gv
    absG2 = np.einsum('ij,ij->j',np.conj(Gv),Gv)
    with np.errstate(divide='ignore'):
        coulG = 4*np.pi/absG2
    coulG[0] = 0.

    return coulG

