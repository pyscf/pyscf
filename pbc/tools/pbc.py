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
    absG2 = np.einsum('gi,gi->g', cell.Gv, cell.Gv)
    with np.errstate(divide='ignore'):
        coulG = 4*np.pi/absG2
    coulG[0] = 0.

    return coulG

def get_lattice_Ls(cell, nimgs):
    '''Get the (Cartesian, unitful) lattice translation vectors for nearby images.'''
    Ts = [[i,j,k] for i in range(-nimgs[0],nimgs[0]+1)
                  for j in range(-nimgs[1],nimgs[1]+1)
                  for k in range(-nimgs[2],nimgs[2]+1)
                  if i**2+j**2+k**2 <= 1./3*np.dot(nimgs,nimgs)]
    Ts = np.array(Ts)
    Ls = np.dot(cell._h, Ts.T).T
    return Ls

def super_cell(cell, nimgs):
    '''Create an nimgs[0] x nimgs[1] x nimgs[2] supercell of the input cell

    Args:
        cell : instance of :class:`Cell`
        nimgs : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    Ls = get_lattice_Ls(cell, nimgs)
    supcell = cell.copy()
    supcell.atom = []
    for L in Ls:
        atom1 = []
        for ia in range(cell.natm):
            atom1.append([cell._atom[ia][0], cell._atom[ia][1]+L])
        supcell.atom.extend(atom1)
    supcell.unit = 'B'
    supcell.h = np.dot(cell._h, np.diag(nimgs))
    supcell.build(False, False, verbose=0)
    return supcell

def get_KLMN(kpts):
    '''Get array KLMN where for gs indices, K, L, M,
    KLMN[K,L,M] gives index of N that satifies
    momentum conservation

       G(K) - G(L) = G(M) - G(N)

    This is used for symmetry e.g. integrals of the form

    [\phi*[K](1) \phi[L](1) | \phi*[M](2) \phi[N](2)]

    are zero unless N satisfies the above.
    '''
    nkpts = kpts.shape[0]
    KLMN = np.zeros([nkpts,nkpts,nkpts], np.int)

    for K, kvK in enumerate(kpts):
        for L, kvL in enumerate(kpts):
            for M, kvM in enumerate(kpts):
                kvN = kvM + kvL - kvK
                KLMN[K, L, M] = np.where(np.logical_and(kpts < kvN + 1.e-12,
                                              kpts > kvN - 1.e-12))[0][0]

    return KLMN

def cutoff_to_gs(h, cutoff):
    '''
    Convert KE cutoff to #grid points (gs variable)

        uses KE = k^2 / 2, where k_max ~ \pi / grid_spacing

    Args: 
        h : (3,3) ndarray
            The unit cell lattice vectors, a "three-column" array [a1|a2|a3], in Bohr
        cutoff : float
            KE energy cutoff in a.u.

    Returns: 
        gs : (3,) array
    '''
    grid_spacing = np.pi / np.sqrt(2 * cutoff)

    print grid_spacing
    print h

    h0 = np.linalg.norm(h[:,0])
    h1 = np.linalg.norm(h[:,1])
    h2 = np.linalg.norm(h[:,2])

    print h0, h1, h2
    # number of grid points is 2gs+1 (~ 2 gs) along each direction 
    gs = np.ceil([h0 / (2*grid_spacing), 
                  h1 / (2*grid_spacing), 
                  h2 / (2*grid_spacing)])
    return gs
