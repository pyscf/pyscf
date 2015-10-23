import numpy as np

def span3(*xs):
    '''Generate integer coordinates for each three-dimensional grid point.

    Args:
        *xs : length-3 tuple of np.arange() arrays
            The integer coordinates along each direction.

    Returns:
         (3, ngx*ngy*ngz) ndarray
            The integer coordinates for each grid point.

    Examples:

    >>> span3(np.array([2,3,2]))
    array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
           [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

    '''
    c = np.empty([3]+[len(x) for x in xs])
    c[0,:,:,:] = np.asarray(xs[0]).reshape(-1,1,1)
    c[1,:,:,:] = np.asarray(xs[1]).reshape(1,-1,1)
    c[2,:,:,:] = np.asarray(xs[2]).reshape(1,1,-1)
    return c.reshape(3,-1)

def fft(f, gs):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    Re: MH (3.25), we assume Ns := ngs = 2*gs+1

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of `span3`.
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

