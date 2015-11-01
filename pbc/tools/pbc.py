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
    ngs = 2*np.asarray(gs)+1
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
    ngs = 2*np.asarray(gs)+1
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
    absG2 = np.einsum('ij,ij->j',np.conj(cell.Gv),cell.Gv)
    with np.errstate(divide='ignore'):
        coulG = 4*np.pi/absG2
    coulG[0] = 0.

    return coulG


def replicate_cell(cell, ncopy):
    '''Create an ncopy[0] x ncopy[1] x ncopy[2] repeat
    of the input cell
    Args:
        cell : instance of :class: 'Cell'
        ncopy : (3, ) array

    Returns:
        repcell : :class: 'Cell'
    '''

    repcell = cell.copy()
    repatom = []
    for Lx in range(ncopy[0]):
        for Ly in range(ncopy[1]):
            for Lz in range(ncopy[2]):
                for atom, coord in cell._atom:
                    L = np.dot(cell._h, [Lx, Ly, Lz])
                    repatom.append([atom, coord + L])

    repcell.atom = repatom
    repcell.unit = 'B'
    repcell.h = np.dot(cell._h, np.diag(ncopy))
    repcell.build(False, False)

    return repcell

def get_KLMN(kpts, Gv):
    '''Given array KLMN where for gs indices, K, L, M, 
    KLMN[K,L,M] gives index of N that satifies
    momentum conservation

       G(K) - G(L) = G(M) - G(N)

    This is used for symmetry e.g. in integrals of the form

    [\phi*[K](1) \phi[L](1) | \phi*[M](2) \phi[N](2)]

    is zero unless N satisfies the above.
    '''
    nkpts = kpts.shape[0]
    KLMN = np.zeros([nkpts,nkpts,nkpts], np.int)

    for K, GvK in enumerate(Gv):
        for L, GvL in enumerate(Gv):
            for M, GvM in enumerate(Gv):
                GvN = GvM + GvL - GvK
                KLMN[K, L, M] = np.where(np.logical_and(Gv < GvN + 1.e-12,
                                                        Gv > GvN - 1.e-12))[0][0]

    return KLMN

