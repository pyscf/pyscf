import numpy as np
import itertools

np.random.seed(0)

n = 8
m = 2
nfrag = 4
A = np.s_[0*m:1*m]
B = np.s_[1*m:2*m]
C = np.s_[2*m:3*m]
D = np.s_[3*m:4*m]
spaces = [A, B, C, D]
indices = [np.arange(n)[space] for space in spaces]

inverses = [~np.isin(np.arange(n), idx) for idx in indices]
print(indices)
print(inverses)

def same_space(i, j):
    for idx in indices:
        if i in idx and j in idx:
            return True
    return False

# T1

fock = np.random.rand(n,n)
T1 = np.full((n, n), 1.0)

res = np.einsum("ij,ij->", T1, fock)
res2 = 0.0
for space in spaces:
    res2 += np.einsum("ij,ij->", T1[space], fock[space])

assert np.isclose(res, res2)

# Alternative (better?)
res2 = 0.0
for idx, space in enumerate(spaces):
    inverse = inverses[idx]
    res2 += np.einsum("ij,ij->", T1[space,space], fock[space,space])
    res2 += 0.5*np.einsum("ij,ij->", T1[space,inverse], fock[space,inverse])
    res2 += 0.5*np.einsum("ij,ij->", T1[inverse,space], fock[inverse,space])

assert np.isclose(res, res2)

# t2

eri = np.random.rand(n, n, n, n)
T2 = np.full((n, n, n, n), 1.0)

res = np.einsum("ijkl,ijkl", T2, eri)

print(res)


# Most democratic (currently)
res1 = 0.0
for space in spaces:
    res1 += np.einsum("ijkl,ijkl", T2[space], eri[space])
assert np.isclose(res, res1)

res2 = 0.0
# occ
for idx, space in enumerate(spaces):
    inverse = inverses[idx]

    res2 += np.einsum("ijkl,ijkl", T2[space,space], eri[space,space])
    res2 += 0.5*np.einsum("ijkl,ijkl", T2[space,inverse], eri[space,inverse])
    res2 += 0.5*np.einsum("ijkl,ijkl", T2[inverse,space], eri[inverse,space])
assert np.isclose(res, res2)

# Most authoritarion
res3 = 0.0
for idx, space in enumerate(spaces):
    print("Space %d" % idx)
    inv = inverses[idx]
    # 4 in space
    #res3 += np.einsum("ijkl,ijkl", T2[space,space,space,space], eri[space,space,space,space])
    ## 3 in space
    #res3 += 1*np.einsum("ijkl,ijkl", T2[inv,space,space,space], eri[inv,space,space,space])
    #res3 += 1*np.einsum("ijkl,ijkl", T2[space,inv,space,space], eri[space,inv,space,space])
    #res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,inv,space], eri[space,space,inv,space])
    #res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,space,inv], eri[space,space,space,inv])


    # Combination
    res3 += 1*np.einsum("ijkl,ijkl", T2[:,space,space,space], eri[:,space,space,space])
    res3 += 1*np.einsum("ijkl,ijkl", T2[space,inv,space,space], eri[space,inv,space,space])
    res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,inv,space], eri[space,space,inv,space])
    res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,space,inv], eri[space,space,space,inv])

    # 2 in space (tricky!)
    for idx2, space2 in enumerate(spaces):
        if idx == idx2:
            continue
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space2,space2,space,space], eri[space2,space2,space,space])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space2,space,space2,space], eri[space2,space,space2,space])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space2,space,space,space2], eri[space2,space,space,space2])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space,space2,space2,space], eri[space,space2,space2,space])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space,space2,space,space2], eri[space,space2,space,space2])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space,space,space2,space2], eri[space,space,space2,space2])

        for idx3, space3 in enumerate(spaces):
            if idx == idx3:
                continue
            if idx2 == idx3:
                continue
            res3 += np.einsum("ijkl,ijkl", T2[space2,space3,space,space], eri[space2,space3,space,space])
            res3 += np.einsum("ijkl,ijkl", T2[space2,space,space3,space], eri[space2,space,space3,space])
            res3 += np.einsum("ijkl,ijkl", T2[space2,space,space,space3], eri[space2,space,space,space3])
            res3 += np.einsum("ijkl,ijkl", T2[space,space2,space3,space], eri[space,space2,space3,space])
            res3 += np.einsum("ijkl,ijkl", T2[space,space2,space,space3], eri[space,space2,space,space3])
            res3 += np.einsum("ijkl,ijkl", T2[space,space,space2,space3], eri[space,space,space2,space3])

            for idx4, space4 in enumerate(spaces):
                if idx == idx4:
                    continue
                if idx2 == idx4:
                    continue
                if idx3 == idx4:
                    continue
                res3 += np.einsum("ijkl,ijkl", T2[space,space2,space3,space4], eri[space,space2,space3,space4])
print(res3)
assert np.isclose(res, res3)

#def check_3equal(i, j, k, l):
#    return (i == j == k) or (i == j == l) or (i == k == l) or (j == k == l)
#
#def check_2equal_2equal(i, j, k, l):
#    return ((i == j) and (k == l)) or ((i == k) and (j == l)) or ((i == l) and (j == k))
#
#def check_2equal(i, j, k, l):
#    return (i == j) or (i == k) or (i == l) or (j == k) or (j == l) or (k == l)
#
#def check_all_different(i, j, k, l):
#    return (i != j) and (i != k) and (i != l) and (j != k) and (j != l) and (k != l)

def check_4equal(i, j, k, l):
    return (i == j == k == l)

def check_3equal(i, j, k, l):
    return (i == j == k) or (i == j == l) or (i == k == l)

def check_2equal_2equal(i, j, k, l):
    return ((i == j) and (k == l)) or ((i == k) and (j == l)) or ((i == l) and (j == k))

def check_2equal(i, j, k, l):
    return (i == j) or (i == k) or (i == l)

def check_all_different(i, j, k, l):
    return (i != j) and (i != k) and (i != l) and (j != k) and (j != l) and (k != l)


res4 = 0.0
#for i, s1 in enumerate(spaces):
#    for j, s2 in enumerate(spaces):
#        for k, s3 in enumerate(spaces):
#            for l, s4 in enumerate(spaces):
#                #if check_3equal(i, j, k, l):
#                #    fac = 1
#                #elif check_2equal_2equal(i, j, k, l):
#                #    fac = 0.5
#                #elif check_2equal(i, j, k, l):
#                #    fac = 1
#                #elif check_all_different(i, j, k, l):
#                #    fac = 0.25
#                #else:
#                #    continue
#                res4 += fac*np.einsum("ijkl,ijkl", T2[s1,s2,s3,s4], eri[s1,s2,s3,s4])

#def count(i, j, k, l):
    #return sum([i == j, i == k, i == l])
    #for 

# Fragment loop
energies = np.zeros((nfrag,))

for i, s1 in enumerate(spaces):
    for j, s2 in enumerate(spaces):
        for k, s3 in enumerate(spaces):
            for l, s4 in enumerate(spaces):
                c = count(i,j,k,l)
                if c == 3:
                    res4 += np.einsum("ijkl,ijkl", T2[s1,s2,s3,s4], eri[s1,s2,s3,s4])




                if check_4equal(i, j, k, l):
                    res4 += np.einsum("ijkl,ijkl", T2[s1,s2,s3,s4], eri[s1,s2,s3,s4])
                if check_3equal(i, j, k, l):
                    res4 += np.einsum("ijkl,ijkl", T2[s1,s2,s3,s4], eri[s1,s2,s3,s4])


print(res4)
assert np.isclose(res, res4)

1/0



if False:
    print("Space %d" % idx)
    inv = inverses[idx]
    # 4 in space
    #res3 += np.einsum("ijkl,ijkl", T2[space,space,space,space], eri[space,space,space,space])
    ## 3 in space
    #res3 += 1*np.einsum("ijkl,ijkl", T2[inv,space,space,space], eri[inv,space,space,space])
    #res3 += 1*np.einsum("ijkl,ijkl", T2[space,inv,space,space], eri[space,inv,space,space])
    #res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,inv,space], eri[space,space,inv,space])
    #res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,space,inv], eri[space,space,space,inv])


    # Combination
    res3 += 1*np.einsum("ijkl,ijkl", T2[:,space,space,space], eri[:,space,space,space])
    res3 += 1*np.einsum("ijkl,ijkl", T2[space,inv,space,space], eri[space,inv,space,space])
    res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,inv,space], eri[space,space,inv,space])
    res3 += 1*np.einsum("ijkl,ijkl", T2[space,space,space,inv], eri[space,space,space,inv])

    # 2 in space (tricky!)
    for idx2, space2 in enumerate(spaces):
        if idx == idx2:
            continue
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space2,space2,space,space], eri[space2,space2,space,space])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space2,space,space2,space], eri[space2,space,space2,space])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space2,space,space,space2], eri[space2,space,space,space2])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space,space2,space2,space], eri[space,space2,space2,space])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space,space2,space,space2], eri[space,space2,space,space2])
        res3 += 0.5*np.einsum("ijkl,ijkl", T2[space,space,space2,space2], eri[space,space,space2,space2])

        for idx3, space3 in enumerate(spaces):
            if idx == idx3:
                continue
            if idx2 == idx3:
                continue
            res3 += np.einsum("ijkl,ijkl", T2[space2,space3,space,space], eri[space2,space3,space,space])
            res3 += np.einsum("ijkl,ijkl", T2[space2,space,space3,space], eri[space2,space,space3,space])
            res3 += np.einsum("ijkl,ijkl", T2[space2,space,space,space3], eri[space2,space,space,space3])
            res3 += np.einsum("ijkl,ijkl", T2[space,space2,space3,space], eri[space,space2,space3,space])
            res3 += np.einsum("ijkl,ijkl", T2[space,space2,space,space3], eri[space,space2,space,space3])
            res3 += np.einsum("ijkl,ijkl", T2[space,space,space2,space3], eri[space,space,space2,space3])

            for idx4, space4 in enumerate(spaces):
                if idx == idx4:
                    continue
                if idx2 == idx4:
                    continue
                if idx3 == idx4:
                    continue
                res3 += np.einsum("ijkl,ijkl", T2[space,space2,space3,space4], eri[space,space2,space3,space4])






    # 1
    #res3 += 0.25**np.einsum("ijkl,ijkl", T2[space,inv][:,:,inv][:,:,:,inv], eri[space,inv][:,:,inv][:,:,:,inv])
    #res3 += 0.25**np.einsum("ijkl,ijkl", T2[inv,space][:,:,inv][:,:,:,inv], eri[inv,space][:,:,inv][:,:,:,inv])
    #res3 += 0.25**np.einsum("ijkl,ijkl", T2[inv][:,inv][:,:,space,inv], eri[inv][:,inv][:,:,space,inv])
    #res3 += 0.25**np.einsum("ijkl,ijkl", T2[inv][:,inv][:,:,inv,space], eri[inv][:,inv][:,:,inv,space])


    #f2 = 1
    #for i in range(n):
    #    for j in range(n):
    #        if i in indices[idx] or j in indices[idx]:
    #            continue
    #        if same_space(i, j):
    #            res2 += 0.5*np.einsum("kl,kl", T2[i,j,space,space], eri[i,j,space,space])
    #            res2 += 0.5*np.einsum("jl,jl", T2[i,space,j,space], eri[i,space,j,space])
    #            res2 += 0.5*np.einsum("jk,jk", T2[i,space,space,j], eri[i,space,space,j])
    #            res2 += 0.5*np.einsum("il,il", T2[space,i,j,space], eri[space,i,j,space])
    #            res2 += 0.5*np.einsum("ik,ik", T2[space,i,space,j], eri[space,i,space,j])
    #            res2 += 0.5*np.einsum("ij,ij", T2[space,space,i,j], eri[space,space,i,j])
    #        else:
    #            res2 += np.einsum("kl,kl", T2[i,j,space,space], eri[i,j,space,space])
    #            res2 += np.einsum("jl,jl", T2[i,space,j,space], eri[i,space,j,space])
    #            res2 += np.einsum("jk,jk", T2[i,space,space,j], eri[i,space,space,j])
    #            res2 += np.einsum("il,il", T2[space,i,j,space], eri[space,i,j,space])
    #            res2 += np.einsum("ik,ik", T2[space,i,space,j], eri[space,i,space,j])
    #            res2 += np.einsum("ij,ij", T2[space,space,i,j], eri[space,space,i,j])
    #for jdx, space2 in enumerate(spaces):
    #    pass


    ##res2 += f2*np.einsum("ijkl,ijkl", T[:,:,space,space], eri[:,:,space,space])
    ##res2 += f2*np.einsum("ijkl,ijkl", T[:,space,:,space], eri[:,space,:,space])
    ##res2 += f2*np.einsum("ijkl,ijkl", T[:,space,space,:], eri[:,space,space,:])
    ##res2 += f2*np.einsum("ijkl,ijkl", T[space,:,:,space], eri[space,:,:,space])
    ##res2 += f2*np.einsum("ijkl,ijkl", T[space,:,space,:], eri[space,:,space,:])
    ##res2 += f2*np.einsum("ijkl,ijkl", T[space,space,:,:], eri[space,space,:,:])
    ### 1 in space (tricky!)
    ##f1 = 0.5
    ##res2 -= f1*np.einsum("ijkl,ijkl", T[:,:,:,space], eri[:,:,:,space])
    ##res2 -= f1*np.einsum("ijkl,ijkl", T[:,:,space,:], eri[:,:,space,:])
    ##res2 -= f1*np.einsum("ijkl,ijkl", T[:,space,:,:], eri[:,space,:,:])
    ##res2 -= f1*np.einsum("ijkl,ijkl", T[space,:,:,:], eri[space,:,:,:])

print(res3)
assert np.isclose(res, res3)


