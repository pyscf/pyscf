import numpy as np
import scipy
import scipy.linalg

__all__ = ["localize_ao"]

def localize_ao(mol, coeff, centers, power=1):
    """coeff : initial coefficients"""
    nao = coeff.shape[-1]
    s = mol.intor_symmetric('int1e_ovlp')
    assert np.allclose(np.linalg.multi_dot((coeff.T, s, coeff))-np.eye(nao), 0)

    atom_coords = mol.atom_coords() # In Bohr
    atom_centers = np.asarray([atom_coords[i] for i in centers])
    #print("Atom centers:")
    #print(atom_centers)

    # TODO: PBC
    rmat_ao = mol.intor_symmetric("int1e_r")     # In Bohr!
    r2mat_ao = mol.intor_symmetric('int1e_r2')
    #rmat = np.einsum("xab,ai,bj->xij", rmat_ao, coeff, coeff)
    rmat = np.einsum("xpq,pa,qb->abx", rmat_ao, coeff, coeff)
    assert np.allclose(rmat, rmat.transpose(1,0,2))
    #r2mat = np.einsum("xab,ai,bj->xij", r2mat_ao, coeff, coeff)
    r2mat = np.einsum("pq,pa,qb->ab", r2mat_ao, coeff, coeff)
    R2 = np.einsum("ix,ix->i", atom_centers, atom_centers)

    def _unpack_values(values):
        """Unpack into skew-symmetric matrix"""
        size = int(np.sqrt(2*values.size)) + 1
        idx = np.tril_indices(size, -1)
        unpacked = np.zeros((size, size), dtype=values.dtype)
        unpacked[idx] = values
        unpacked = (unpacked - unpacked.T)
        return unpacked

    def _pack_values(values, assert_symmetry=True):
        assert (np.allclose(values, -values.T) or not assert_symmetry)
        size = values.shape[-1]
        idx = np.tril_indices(size, -1)
        packed = values[idx]
        return packed

    iteration = 0
    fval = None

    def objective_function(values, full=True):
        nonlocal iteration, fval
        iteration += 1
        s = _unpack_values(values)
        u = scipy.linalg.expm(s)
        #c = np.dot(coeff, u)
        #d = np.einsum("ai,bi,xab->xi", c, c, dipole)
        d = np.einsum("ai,bi,abx->ix", u, u, rmat)
        fi = -2*np.einsum("ix,ix->i", atom_centers, d)
        if power > 1:
            assert full
        if full:
            # These contributions are invarient under unitary transformations and can be ignored
            #fi += (R2 + np.einsum("ai,bi,ab->i", c, c, r2mat))
            d2 = np.einsum("ai,bi,ab->i", u, u, r2mat)
            fi += (R2 + d2)

        fval = np.sum(fi**power)
        if iteration % 1000 == 0:
            print("Iteration %4d: value = %.5e" % (iteration, fval))
        return fval

    def gradient(values):
        raise NotImplementedError()
        # TODO
        s = _unpack_values(values)
        u = scipy.linalg.expm(s)
        #u2 = np.eye(s.shape[-1])
        #rmat2 = np.einsum("ai,bj,abx->ijx", u, u, rmat)
        #assert np.allclose(u, u2)
        #c = np.dot(coeff, u)
        #grad = (np.einsum("abx,bi,ix->ai", rmat, u, atom_centers)
        #      + np.einsum("bax,bi,ix->ai", rmat, u, atom_centers))
        #grad = 2*(np.einsum("ai,abx,bj,jx->ij", u, rmat, u, atom_centers)
        #        + np.einsum("ai,bax,bj,jx->ij", u, rmat, u, atom_centers))
        grad = 4 * np.einsum("xbz,by,yz->xy", rmat, u, atom_centers).T
        #grad = 4 * np.einsum("xbz,by,yz->xy", rmat2, u, atom_centers).T
        #grad = 4 * np.einsum("xbz,by,yz->xy", rmat2, u2, atom_centers).T
        #print(grad)
        #grad += np.einsum("iy,xbz,bi,iz->xy", s, rmat, u, atom_centers)
        #grad += np.einsum("ax,abz,by,yz->xy", s, rmat, u, atom_centers)
        grad[np.diag_indices(nao)] = 0
        np.set_printoptions(linewidth=300)
        print(grad)
        print(np.linalg.norm(grad - grad.T))

        #grad = (grad - grad.T)
        grad = _pack_values(grad, assert_symmetry=True)
        #grad = _pack_values(grad, assert_symmetry=False)
        return grad

    #np.random.seed(0)
    s0 = _pack_values(np.zeros((nao, nao)))
    #s0 += 1e-6*(np.random.rand(*s0.shape)-0.5)
    #s0 += 1e-2*(np.random.rand(*s0.shape)-0.5)
    #s0 += 2*(np.random.rand(*s0.shape)-0.5)

    #import numdifftools as nd
    #m = 2
    #a = np.random.rand(m, m)
    ##a = np.eye(m)
    #s = np.random.rand(m, m)
    #u = scipy.linalg.expm(s)
    ##fval = lambda x : np.trace(np.dot(a, scipy.linalg.expm(x)))
    #def fval(x):
    #    x = x.reshape(m, m)
    #    return np.trace(np.dot(a, scipy.linalg.expm(x)))
    #grad = nd.Gradient(fval)
    ##grad = nd.Jacobian(fval)
    #s0 = np.random.rand(m, m)
    #drv = grad(s0).reshape(m, m)
    #print(drv)

    #def grad(x):
    #    x = x.reshape(m, m)
    #    u = scipy.linalg.expm(x)
    #    #return np.dot(u.T, a)
    #    g = np.dot(a, u.T)
    #    g = (g + g.T)/2
    #    return g
    #    #return u.T

    #drv = grad(s0)
    #print(drv)


    #g2 = gradient(s0)
    #print(g2.shape)
    #print(g2)

    #grad = nd.Gradient(objective_function)
    #drv = grad(s0)
    #print("Gradient")
    #print(drv.shape)
    #print(drv)
    #print(np.linalg.norm(g2-drv))

    #1/0

    fval = objective_function(s0)
    print("Initial value = %.5e" % (fval))
    res = scipy.optimize.minimize(objective_function, x0=s0, args=(True,))
    s = _unpack_values(res.x)
    u = scipy.linalg.expm(s)
    coeff_loc = np.dot(coeff, u)
    print("Final value = %.5e" % (fval))

    def get_localization(coeff):
        dip = np.einsum("ai,bi,xab->xi", coeff, coeff, rmat_ao)
        omega = -2*np.einsum("ix,xi->i", atom_centers, dip)
        omega += np.einsum("ai,bi,ab->i", coeff, coeff, r2mat_ao)
        omega += np.einsum("ix,ix->i", atom_centers, atom_centers)
        omega = np.sqrt(omega)
        return omega

    print("Initial localizations:\n%r" % get_localization(coeff))
    print("Final localizations:\n%r" % get_localization(coeff_loc))

    return coeff_loc



if __name__ == "__main__":
    import pyscf
    import pyscf.gto
    import pyscf.scf
    import pyscf.lo

    #distance = 0.74
    #distance = 1.0
    distance = 0.5

    #for d in [1.0, 2.0]:
    for d in [0.0]:

        atom = "H 0 0 %f ; H 0 0 %f" % (d, d+distance)
        basis = "cc-pVDZ"
        mol = pyscf.gto.M(atom=atom, basis=basis)

        s = mol.intor_symmetric('int1e_ovlp')
        coeff_lowdin = pyscf.lo.vec_lowdin(np.eye(s.shape[-1]), s)

        #centers = [mol.bas_coord(i) for i in range(100)]
        centers = [l[0] for l in mol.ao_labels(None)]
        coeff_loc = localize_ao(mol, coeff_lowdin, centers)
        #coeff_loc = localize_ao(mol, coeff_lowdin, centers, power=2)

        1/0

        from util import create_orbital_file
        #create_orbital_file(mol, "orbitals-in", coeffs)
        #create_orbital_file(mol, "orbitals-out", coeffs_u)
        create_orbital_file(mol, "orbitals-in", coeff_lowdin, filetype="cube")
        create_orbital_file(mol, "orbitals-out", coeff_loc, filetype="cube")

        coeff_loc = localize_ao(mol, coeff_lowdin, centers, power=2)
        create_orbital_file(mol, "orbitals-2-out", coeff_loc, filetype="cube")


    1/0


    dipol = mol.intor_symmetric("int1e_r", comp=3)
    print(dipol * 0.52)
    print(dipol.shape)

    hf = pyscf.scf.RHF(mol)
    hf.kernel()

    dipol_mo = np.einsum("xab,ai,bj->xij", dipol, hf.mo_coeff, hf.mo_coeff)
    print(dipol_mo)
    print(dipol_mo * 0.52)

