import copy

import numpy as np
import scipy
import scipy.optimize

from matplotlib import pyplot as plt



class BasisSet:

    def __init__(self, exp, coeff):
        self.exp = exp
        self.coeff = coeff

    def discard_diffuse(self, ndiscard=1):
        self.exp = self.exp[:-ndiscard]
        self.coeff = self.coeff[:-ndiscard]

    def radial_func(self, r):
        f = np.sum(self.coeff[:,np.newaxis] * np.exp(np.outer(-self.exp, r**2)), axis=0)
        return f

    def fit_to_basis(self, basis, fixed_exp=1):
        exp = self.exp
        exp_fixed = self.exp[:fixed_exp]
        coeff = self.coeff

        maxr = 10.0
        grid = np.linspace(0, maxr, num=200)

        rad0 = basis.radial_func(grid)
        rad1 = self.radial_func(grid)

        fig, axes = plt.subplots(ncols=2)

        axes[0].plot(grid, rad0, label="original")
        axes[1].plot(grid, grid*rad0, label="original")
        axes[0].plot(grid, rad1, label="truncated")
        axes[1].plot(grid, grid*rad1, label="truncated")

        def objective_func(norm):
            diff = (rad0 - norm*rad1)
            return np.linalg.norm(diff)

        res = scipy.optimize.minimize(objective_func, x0=1)
        norm = res.x
        self.coeff *= norm
        rad2 = self.radial_func(grid)
        axes[0].plot(grid, rad2, label="normalized")
        axes[1].plot(grid, grid*rad2, label="normalized")

        nexpopt = len(exp)-fixed_exp

        #maxrfit = 1/exp[-1]
        maxrfit = np.sqrt(np.log(2) / exp[-1])

        print(maxrfit)
        weight = (grid < maxrfit)

        def objective_func(params):
            e, c = np.split(params, [nexpopt])
            e = np.hstack((exp_fixed, e))
            rad = BasisSet(e, c).radial_func(grid)
            diff = (rad - rad0) * weight
            return np.linalg.norm(diff)

        x0=np.hstack((exp[fixed_exp:], coeff))
        res = scipy.optimize.minimize(objective_func, x0=x0)
        e_opt, c_opt = np.split(res.x, [nexpopt])
        print(exp)
        e_opt = np.hstack((exp_fixed, e_opt))
        print(e_opt)
        basis_opt = BasisSet(e_opt, c_opt)
        rad_opt = basis_opt.radial_func(grid)

        axes[0].plot(grid, rad_opt, label="optimized")
        axes[1].plot(grid, grid*rad_opt, label="optimized")
        axes[0].legend()
        plt.show()




if __name__ == "__main__":

    #e = np.asarray([8.3744350009, 1.8058681460, 0.4852528328, 0.1658236932])
    #c = np.asarray([-0.0283380461, -0.1333810052, -0.3995676063, -0.5531027541])
    data = np.asarray([
        5.3685662937,   0.0974901974,   #0.0000000000   0.0000000000  -0.0510969367   0.0000000000   0.0000000000
        1.9830691554,   0.1041996677,   #0.0000000000   0.0000000000  -0.1693035193   0.0000000000   0.0000000000
        0.6978346167,  -0.3645093878,   #0.0000000000   0.0000000000  -0.3579933930   0.0000000000   0.0000000000
        0.2430968816,  -0.6336931464,   #1.0000000000   0.0000000000  -0.4327616531   1.0000000000   0.0000000000
        0.0812865018,  -0.1676727564,   #0.0000000000   1.0000000000  -0.2457672757   0.0000000000   1.0000000000
        ])
    e = data[::2].copy()
    c = data[1::2].copy()

    basis = BasisSet(e, c)
    basis2 = copy.copy(basis)
    basis2.discard_diffuse()
    basis2.fit_to_basis(basis)


