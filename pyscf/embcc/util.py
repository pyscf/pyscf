import numpy as np
import scipy
import scipy.optimize

__all__ = [
        "eigassign",
        "eigreorder_logging",
        ]

def eigassign(e1, v1, e2, v2, b=None, cost_matrix="e^2/v", return_cost=False):
    """
    Parameters
    ----------
    b : ndarray
        If set, eigenvalues and eigenvectors belong to a generalized eigenvalue problem of the form Av=Bve.
    cost_matrix : str
        Defines the way to calculate the cost matrix.
    """

    if e1.shape != e2.shape:
        raise ValueError("e1=%r with shape=%r and e2=%r with shape=%r are not compatible." % (e1, e1.shape, e2, e2.shape))
    if v1.shape != v2.shape:
        raise ValueError("v1=%r with shape=%r and v2=%r with shape=%r are not compatible." % (v1, v1.shape, v2, v2.shape))
    if e1.shape[0] != v1.shape[-1]:
        raise ValueError("e1=%r with shape=%r and v1=%r with shape=%r are not compatible." % (e1, e1.shape, v1, v1.shape))
    if e2.shape[0] != v2.shape[-1]:
        raise ValueError("e2=%r with shape=%r and v2=%r with shape=%r are not compatible." % (e2, e2.shape, v2, v2.shape))


    assert np.allclose(e1.imag, 0)
    assert np.allclose(e2.imag, 0)
    assert np.allclose(v1.imag, 0)
    assert np.allclose(v2.imag, 0)

    # Distance function
    if b is None:
        vmat = np.abs(np.dot(v1.T, v2))
    else:
        vmat = np.abs(np.linalg.multi_dot((v1.T, b, v2)))
    emat = np.abs(np.subtract.outer(e1, e2))

    # Original formulation
    if cost_matrix == "(1-v)*e":
        dist = (1-vmat) * emat
    elif cost_matrix == "(1-v)":
        dist = (1-vmat)
    elif cost_matrix == "v/e":
        dist = -vmat / (emat + 1e-14)
    elif cost_matrix == "e/v":
        dist = emat / (vmat + 1e-14)
    # This performed best in tests
    elif cost_matrix == "e^2/v":
        #dist = emat**2 / (vmat + 1e-14)
        dist = emat**2 / np.fmax(vmat, 1e-14)
    elif cost_matrix == "e^2/v**2":
        dist = emat**2 / (vmat + 1e-14)**2
    elif cost_matrix == "e/sqrt(v)":
        dist = emat / np.sqrt(vmat + 1e-14)
    else:
        raise ValueError("Unknown cost_matrix: %s" % cost_matrix)

    row, col = scipy.optimize.linear_sum_assignment(dist)
    # The col indices are the new sorting
    sort = col
    if return_cost:
        cost = dist[row,col].sum()
        return sort, cost
    else:
        return sort

def eigreorder_logging(e, reorder, log):
    for i, j in enumerate(reorder):
        # No reordering
        if i == j:
            continue
        # Swap between two eigenvalues
        elif reorder[j] == i:
            if i < j:
                log("Reordering eigenvalues %3d <-> %3d : %+6.3g <-> %+6.3g", j, i, e[j], e[i])
        # General reordering
        else:
            log("Reordering eigenvalues %3d --> %3d : %+6.3g --> %+6.3g", j, i, e[j], e[i])


def _test():

    n = 30
    N = 100

    nums = [20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 5000]
    #nums = [20, 30, 40, 50, 60]#, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000]

    result = np.zeros((N, len(nums), 3))

    for i in range(N):

        np.random.seed(i)
        A = 1*(2*np.random.rand(n,n)-1)
        B = 1*(2*np.random.rand(n,n)-1)
        C = 1*(2*np.random.rand(n,n)-1)

        #def get_intermediate(t):
        #    t2 = 3*t
        #    if 0.0 <= t2 < 1.0:
        #        M1, M2 = A, B
        #    elif 1.0 <= t2 < 2.0:
        #        M1, M2 = B, C
        #    elif 2.0 <= t2 <= 3.0:
        #        M1, M2 = C, A

        #    I = (1-t)*M1 + t*M2
        #    return I

        def get_intermediate(t):
            t = 3*t
            I = ((t-1)*(t-2)*A
                   + t*(t-2)*(t-3)*B
                   + t*(t-1)*(t-3)*C)
            return I

        assert np.allclose(get_intermediate(0.0), get_intermediate(1.0))

        def run(t_array, cost_matrix):
            e_ref, v_ref = None, None
            for t in t_array:
                I = get_intermediate(t)
                e, v = np.linalg.eigh(I)
                if e_ref is not None:
                    sort = eigassign(e_ref, v_ref, e, v, cost_matrix=cost_matrix)
                    e = e[sort]
                    v = v[:,sort]
                e_ref = e
                v_ref = v
            return sort

        sort_exact = np.asarray(list(range(n)))
        for j, num in enumerate(nums):
            t_array = np.linspace(0, 1, num)

            sort = run(t_array, "e/v")
            result[i,j,0] = np.sum(sort != sort_exact)

            sort = run(t_array, "e**2/v")
            result[i,j,1] = np.sum(sort != sort_exact)

            sort = run(t_array, "v*e")
            result[i,j,2] = np.sum(sort != sort_exact)


            #with open("results.txt", "a") as f:
            #    f.write("%3d  %.6g  %.6g  %.6g  %.6g\n" % (n, d_ve, d_vsqrte, d_vde, d_evv))

    mean = np.mean(result, axis=0)
    std = np.std(result, axis=0)

    for j, num in enumerate(nums):
        with open("results.txt", "a") as f:
            fmt = "%3d"+6*"  %.6g"+"\n"
            f.write(fmt % (num, mean[j,0], std[j,0], mean[j,1], std[j,1], mean[j,2], std[j,2]))


if __name__ == "__main__":

    _test()
    1/0

    #for i in range(100):
    #    a = np.random.rand(10,10)
    #    row, col = scipy.optimize.linear_sum_assignment(a)
    #    rowt, colt = scipy.optimize.linear_sum_assignment(a.T)
    #    print(row, rowt)
    #    print(col, colt)
    #    sort = np.argsort(colt)
    #
    #    assert np.allclose(row, rowt)
    #    #assert np.allclose(col, colt)
    #    assert np.allclose(col, sort)
    #    
    #
    #1/0
    
    
    #def make_test_matrix(t):
    #    return np.array([
    #        [1,     2*t+1 , t**2 ,   t**3],
    #        [2*t+1, 2-t   , t**2 , 1-t**3],
    #        [t**2 , t**2  , 3-2*t,   t**2],
    #        [t**3 , 1-t**3, t**2 ,  4-3*t]])
    
    
    #ts = np.linspace(-1, 1, 11)
    
    #a1 = make_test_matrix(0.3)
    #a2 = make_test_matrix(0.4)
    #
    #a1 = np.eye(3)
    #a2 = np.eye(3)
    #a2[:,1], a2[:,2] =  a1[:,2], a1[:,1]
    #print(a2)
    
    m = 100
    np.random.seed(0)
    a = np.random.rand(m, m)
    e, v = np.linalg.eigh(a)
    #
    e1, v1 = e, v
    e2, v2 = e.copy(), v.copy()
    sort = np.random.permutation(m)
    print(sort)
    e2 = e1[sort]
    v2 = v1[:,sort]
    
    
    #e1, v1 = np.linalg.eigh(a1)
    #e2, v2 = np.linalg.eigh(a2)
    
    print(e1)
    print(e2)
    
    es, vs, cost = eigassign(e1, v1, e2, v2, return_cost=True)
    
    print(cost)
    print(es)
    print(np.allclose(es, e1))
    print(np.allclose(vs, v1))

    es, vs, cost = eigassign(e1, v1, e2, v2, cost_matrix="new", return_cost=True)
    
    print(cost)
    print(es)
    print(np.allclose(es, e1))
    print(np.allclose(vs, v1))

