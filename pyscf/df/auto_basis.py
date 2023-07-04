'''
JCTC, 13, 554
'''

f = [20, 4.0, 4.0, 3.5, 2.5, 2.0, 2.0]
beta_big = [1.8, 2.0, 2.2, 2.2, 2.2, 2.3, 3.0, 3.0]
beta_small = 1.8

def auto_basis(mol, l_inc=1):
    l_val = highest_l_in_occupied_shell
    l_max = max(mol._bas[:,0])
    l_max_aux = max(l_val*2, l_max+l_inc)

    a_min_aux = a_min[:,None] + a_min
    a_max_aux = a_min[:,None] + a_min
    r_exp = _gaussian_int(a, l)
    a_max_eff = 2 * k(l)**2 / (np.pi * r_exp)
    a_max_eff_aux = a_max_eff[:,None] + a_max_eff
    a_max_aux = ([max(f[l] * a_max_eff_aux, a_max_aux) for l in range(2*l_val+1)] +
                 a_max_aux[2*l_val+1:])

    beta = list(beta_big)
    beta[:2*l_val] = beta_small

    ns = np.log(a_max_aux/a_min_aux) / beta
