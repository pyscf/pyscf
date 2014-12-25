import numpy

from pyscf import lib
from pyscf.scf import diis

class DIIS(diis.DIIS):
    def update(self, x):
        self.push_vec(x)

        nd = self.get_num_diis_vec()
        if nd <= self.min_space:
            return x

        H = numpy.ones((nd+1,nd+1), x.dtype)
        H[0,0] = 0
        G = numpy.zeros(nd+1, x.dtype)
        G[0] = 1
        for i in range(nd):
            #dti = self.get_err_vec(i)
            dti = x - self.get_vec(i-1)
            for j in range(i+1):
                #dtj = self.get_err_vec(j)
                dtj = x - self.get_vec(j-1)
                H[i+1,j+1] = numpy.dot(numpy.array(dti).ravel(), \
                                       numpy.array(dtj).ravel())
                H[j+1,i+1] = H[i+1,j+1].conj()

        try:
            c = numpy.linalg.solve(H, G)
        except numpy.linalg.linalg.LinAlgError:
            lib.logger.warn(self, 'singularity in diis')
            #c = pyscf.lib.solve_lineq_by_SVD(H, G)
            ## damp diagonal elements to avoid singularity
            #for i in range(H.shape[0]):
            #    H[i,i] = H[i,i] + 1e-9
            #c = numpy.linalg.solve(H, G)
            for i in range(1,nd):
                H[i,i] = H[i,i] + 1e-11
            c = numpy.linalg.solve(H, G)
            #c = numpy.linalg.solve(H[:nd,:nd], G[:nd])
        lib.logger.debug1(self, 'diis-c %s', c)

        x = numpy.zeros_like(x)
        for i, ci in enumerate(c[1:]):
            x += self.get_vec(i) * ci
        return x
