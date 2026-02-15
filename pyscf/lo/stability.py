import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.scf.stability import STAB_NROOTS, STAB_TOL, dump_status


def stability_newton(mlo, verbose=None, return_status=False, nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mlo, verbose)
    g, hop, hdiag = mlo.gen_g_hop()

    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd

    x0 = numpy.zeros_like(g)
    mask = abs(g) > 1e-10
    x0[mask] = 1. / hdiag[mask]
    # add a few preconditioned random vectors
    x0 = numpy.vstack((x0, precond(numpy.random.rand(10, x0.size), 0, None)))
    e, v = lib.davidson(hop, x0, precond, tol=tol, verbose=log.verbose-1, nroots=nroots)
    log.info('stability: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mlo.__class__.__name__}', 'internal')
    if stable:
        mo = mlo.mo_coeff
    else:
        u = mlo.extract_rotation(v)
        mo = mlo.rotate_orb(u)
    if return_status:
        return mo, stable
    else:
        return mo


def pipek_stability_jacobi(mlo, verbose=None, return_status=False):
    ''' Check Jacobi-sweep stability for PM localization.
    '''
    log = logger.new_logger(mlo, verbose)
    exponent = mlo.exponent

    tril_ijdx = numpy.tril_indices(mlo.norb, k=-1)
    tril_idx, tril_jdx = tril_ijdx
    npair = tril_idx.size

    thetapool = numpy.asarray([1, 2, 3]) * 0.25 * numpy.pi

    def update_rotation_local_(u, theta, i, j):
        ui = u[:, i].copy()
        uj = u[:, j].copy()
        c = numpy.cos(theta)
        s = numpy.sin(theta)
        u[:, i] = ui * c + uj * s
        u[:, j] = -ui * s + uj * c

    u = mlo.identity_rotation()
    stable = True

    while True:
        mo_coeff = mlo.rotate_orb(u)
        Pij = mlo.atomic_pops(mlo.mol, mo_coeff).real
        Qi = lib.einsum('xii->xi', Pij)
        Qi_exp = Qi**exponent
        Lij = (Qi_exp[:, tril_idx] + Qi_exp[:, tril_jdx]).sum(axis=0)

        dLij = numpy.zeros_like(Lij)
        thetas = numpy.zeros_like(Lij)

        mem_avail = mlo.mol.max_memory - lib.current_memory()[0]
        natm = Pij.shape[0]
        blkpair = max(1, min(npair, int(numpy.floor(mem_avail*0.5 / (5*natm*8/1e6)))))

        # Loop over theta candidates and update best (theta, dL) for each pair
        for theta in thetapool:
            c = numpy.cos(theta)
            s = numpy.sin(theta)
            c2 = c * c
            s2 = s * s
            cs2 = 2.0 * c * s

            # loop over pairs to save memory
            for p0,p1 in lib.prange(0, npair, blkpair):
                ps = slice(p0,p1)

                ii = tril_idx[ps]
                jj = tril_jdx[ps]

                # Population after rotations
                Qi_i = Qi[:, ii]
                Qi_j = Qi[:, jj]
                Pij_ij = Pij[:, ii, jj]
                Qitild = Qi_i * c2 + Qi_j * s2 + cs2 * Pij_ij
                Qjtild = Qi_i * s2 + Qi_j * c2 - cs2 * Pij_ij
                Qi_i = Qi_j = Pij_ij = None

                # Population change
                dL_blk = (Qitild**exponent + Qjtild**exponent).sum(axis=0) - Lij[ps]
                Qitild = Qjtild = None

                # Find theta that increases the PM objective
                mask = dL_blk > (dLij[ps] + mlo.conv_tol)
                if numpy.any(mask):
                    dLij[ps][mask] = dL_blk[mask]
                    thetas[ps][mask] = theta

        idxs = numpy.where(dLij > mlo.conv_tol)[0]
        if idxs.size == 0:
            break

        # Sort idxs in decreasing order
        idxs = idxs[numpy.argsort(dLij[idxs])[::-1]]

        # Remove overlapping pairs using a greedy algorithm
        stable = False
        done = numpy.zeros(mlo.norb, dtype=bool)

        for idx in idxs:
            i, j = tril_idx[idx], tril_jdx[idx]
            if done[i] or done[j]:
                continue
            done[i] = done[j] = True

            theta = thetas[idx]
            log.info('Rotating orbital pair (%d,%d) by %.2f Pi. delta_f= %.14g',
                     i, j, theta/numpy.pi, dLij[idx])

            update_rotation_local_(u, theta, i, j)

    if stable:
        log.info(f'{mlo.__class__.__name__} is stable in the Jacobi stability analysis')
        mo_coeff = mlo.mo_coeff
    else:
        mo_coeff = mlo.rotate_orb(u)

    return (mo_coeff, stable) if return_status else mo_coeff
