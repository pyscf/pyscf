import numpy as np
import scipy.linalg
from pyscf import lib

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


def fftk(f, gs, r, k):
    '''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*np.exp(-1j*np.dot(k,r.T)), gs)


def ifftk(g, gs, r, k):
    '''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, gs) * np.exp(1j*np.dot(k,r.T))


def get_coulG(cell, k=np.zeros(3), exx=False, mf=None):
    '''Calculate the Coulomb kernel for all G-vectors, handling G=0 and exchange.

    Args:
        cell : instance of :class:`Cell`
        k : (3,) ndarray
        exx : bool
            Whether this is an exchange matrix element.
        mf : instance of :class:`SCF`

    Returns:
        coulG : (ngs,) ndarray
            The Coulomb kernel.

    '''
    kG = k + cell.Gv

    ############################################################################
    # Here we 'wrap around' the high frequency k+G vectors into their lower
    # frequency counterparts.  Important if you want the gamma point and k-point
    # answers to agree
    ############################################################################
    boxEdge       = np.dot(2.*np.pi*np.diag(cell.gs+0.5),np.linalg.inv(cell._h))
    reducedCoords = np.dot(kG,np.linalg.inv(boxEdge))
    factor        = np.trunc(reducedCoords)
    kG -= 2.*np.dot(np.sign(factor),boxEdge)
    ############################################################################
    # End of 'wrap around'
    ############################################################################


    absG2 = np.einsum('gi,gi->g', kG, kG)

    try:
        kpts = mf.kpts
    except AttributeError:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if exx is False or mf.exxdiv is None:
        with np.errstate(divide='ignore'):
            coulG = 4*np.pi/absG2
        if np.linalg.norm(k) < 1e-8:
            coulG[0] = 0.
    elif mf.exxdiv == 'vcut_sph':
        Rc = (3*Nk*cell.vol/(4*np.pi))**(1./3)
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.cos(np.sqrt(absG2)*Rc))
        if np.linalg.norm(k) < 1e-8:
            coulG[0] = 4*np.pi*0.5*Rc**2
    elif mf.exxdiv == 'ewald':
        with np.errstate(divide='ignore'):
            coulG = 4*np.pi/absG2
        if np.linalg.norm(k) < 1e-8:
            coulG[0] = Nk*cell.vol*madelung(cell, kpts)
    elif mf.exxdiv == 'vcut_ws':
        if mf.exx_built == False:
            mf.precompute_exx()
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.exp(-absG2/(4*mf.exx_alpha**2))) + 0j
        if np.linalg.norm(k) < 1e-8:
            coulG[0] = np.pi / mf.exx_alpha**2
        # Index k+cell.Gv into the precomputed vq and add on
        gxyz = np.round(np.dot(kG, mf.exx_kcell.h)/(2*np.pi)).astype(int)
        ngs = 2*mf.exx_kcell.gs+1
        gxyz = (gxyz + ngs)%(ngs)
        qidx = (gxyz[:,0]*ngs[1] + gxyz[:,1])*ngs[2] + gxyz[:,2]
        #qidx = [np.linalg.norm(mf.exx_q-kGi,axis=1).argmin() for kGi in kG]
        maxqv = abs(mf.exx_q).max(axis=0)
        is_lt_maxqv = (abs(kG) <= maxqv).all(axis=1)
        coulG += mf.exx_vq[qidx] * is_lt_maxqv

    return coulG


def madelung(cell, kpts):
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc.scf.hf import ewald

    Nk = get_monkhorst_pack_size(cell, kpts)
    ecell = pbcgto.Cell()
    ecell.atom = 'H 0. 0. 0.'
    ecell.spin = 1
    ecell.gs = cell.gs
    ecell.precision = 1e-16
    ecell.unit = 'B'
    ecell.h = cell._h * Nk
    ecell.build(False,False)
    return -2*ewald(ecell, ecell.ew_eta, ecell.ew_cut)


def get_monkhorst_pack_size(cell, ckpts):
    kpts = np.dot(ckpts, cell._h.T) / (2*np.pi)
    import ase.dft.kpoints
    Nk, eoff = ase.dft.kpoints.get_monkhorst_pack_size_and_offset(kpts)
    #Nk = np.array([len(np.unique(ki)) for ki in kpts.T])
    return Nk


def f_aux(cell, q):
    a = cell._h.T
    b = 2*np.pi*scipy.linalg.inv(cell._h)
    denom = 4 * np.dot(b*np.sin(a*q/2.), b*np.sin(a*q/2.)) \
          + 2 * np.dot(b*np.sin(a*q), np.roll(b,1,axis=0)*np.sin(np.roll(a,1,axis=0)*q ))
    return 1./(2*np.pi)**2 * 1./denom


def get_lattice_Ls(cell, nimgs):
    '''Get the (Cartesian, unitful) lattice translation vectors for nearby images.'''
    Ts = [[i,j,k] for i in range(-nimgs[0],nimgs[0]+1)
                  for j in range(-nimgs[1],nimgs[1]+1)
                  for k in range(-nimgs[2],nimgs[2]+1)
                  if i**2+j**2+k**2 <= 1./3*np.dot(nimgs,nimgs)]
    Ts = np.array(Ts)
    Ls = np.dot(cell._h, Ts.T).T
    return Ls


def super_cell(cell, ncopy):
    '''Create an ncopy[0] x ncopy[1] x ncopy[2] supercell of the input cell

    Args:
        cell : instance of :class:`Cell`
        ncopy : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    supcell = cell.copy()
    supcell.atom = []
    for Lx in range(ncopy[0]):
        for Ly in range(ncopy[1]):
            for Lz in range(ncopy[2]):
                # Using cell._atom guarantees coord is in Bohr
                for atom, coord in cell._atom:
                    L = np.dot(cell._h, [Lx, Ly, Lz])
                    supcell.atom.append([atom, coord + L])
    supcell.unit = 'B'
    supcell.h = np.dot(cell._h, np.diag(ncopy))
    supcell.gs = np.array([ncopy[0]*cell.gs[0] + (ncopy[0]-1)//2,
                           ncopy[1]*cell.gs[1] + (ncopy[1]-1)//2,
                           ncopy[2]*cell.gs[2] + (ncopy[2]-1)//2])
    supcell.build(False, False)
    return supcell


def get_KLMN(cell,kpts):
    '''Get array KLMN where for gs indices, K, L, M,
    KLMN[K,L,M] gives index of N that satifies
    momentum conservation

       G(K) - G(L) = - G(M) + G(N)

    This is used for symmetry e.g. integrals of the form

    [\phi*[K](1) \phi[L](1) | \phi*[M](2) \phi[N](2)]

    are zero unless N satisfies the above.
    '''
    nkpts = kpts.shape[0]
    KLMN = np.zeros([nkpts,nkpts,nkpts], np.int)
    kvecs = 2*np.pi*scipy.linalg.inv(cell._h)

    for K, kvK in enumerate(kpts):
        for L, kvL in enumerate(kpts):
            for M, kvM in enumerate(kpts):
                # Here we find where kvN = kvM + kvL - kvK (mod K)
                temp = range(-1,2)
                xyz = lib.cartesian_prod((temp,temp,temp))
                found = 0
                kvMLK = kvK - kvL + kvM
                kvN = kvMLK
                for ishift in xrange(len(xyz)):
                    kvN = kvMLK + np.dot(xyz[ishift],kvecs)
                    finder = np.where(np.logical_and(kpts < kvN + 1.e-12,
                                                     kpts > kvN - 1.e-12).sum(axis=1)==3)
                    # You want to make sure the k-point is the same in all 3 indices as kvN
                    if len(finder[0]) > 0:
                        KLMN[K, L, M] = finder[0][0]
                        found = 1
                        break

                if found == 0:
                    print "get_klmn is screwing up again!"
                    print kvMLK
                    sys.exit()
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

    #print grid_spacing
    #print h

    h0 = np.linalg.norm(h[:,0])
    h1 = np.linalg.norm(h[:,1])
    h2 = np.linalg.norm(h[:,2])

    #print h0, h1, h2
    # number of grid points is 2gs+1 (~ 2 gs) along each direction
    gs = np.ceil([h0 / (2*grid_spacing),
                  h1 / (2*grid_spacing),
                  h2 / (2*grid_spacing)])
    return gs.astype(int)

