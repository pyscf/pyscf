# TESTING IF MATRIX EXPONENTIAL LIE MAP HOLDS FOR GAMMA SPACE
import numpy
import math
import numpy.random
import numpy.linalg
import scipy.linalg


dimension = 5
occupancy = 2

matrix = numpy.random.rand(dimension, dimension)

def checkMatrixDerivative(matrix):
    numberOfVariables = int(0.5 * len(matrix) * (len(matrix)-1))
    basis = numpy.zeros((numberOfVariables, len(matrix), len(matrix)))
    counter = 0
    for i in range(1, len(matrix)):
        for j in range(0, i):
            basis[counter][i][j] = 1
            basis[counter][j][i] = -1

            counter += 1

    counter = 0
    for i in range(1, len(matrix)):
        for j in range(0, i):
            # I, J is wiggled component
            supposedDeriv = numpy.zeros(matrix.shape)
            fact = 1.0

            for k in range(300):
                contr = numpy.zeros(matrix.shape)
                contr2 = numpy.zeros(matrix.shape)


                for p in range(k):
                    contr += (numpy.linalg.matrix_power(matrix, p) @ basis[counter] @ numpy.linalg.matrix_power(matrix, k-p-1)) * 100**(-k)
                    #if k < 16:
                    #    contr2 += numpy.linalg.matrix_power(matrix, p) @ basis[counter] @ numpy.linalg.matrix_power(matrix, k-p-1)



                if k == 0:
                    fact = 1.0
                else:
                    fact *= k * 0.01
                

                print(fact)
                print("Norm:   " + str(numpy.linalg.norm(contr / fact)))
                #if k < 16:
                #    print("Norm 2: " + str(numpy.linalg.norm(contr2 / math.factorial(k))))
                supposedDeriv += contr / fact

            #print("Test:")
            #print(supposedDerivPot)

            epsilon = 10**-7
            #realDerivP = numpy.linalg.matrix_power(scipy.linalg.expm((matrix[i][j] + epsilon) / float(m) * basis[counter]), m)
            #realDerivM = numpy.linalg.matrix_power(scipy.linalg.expm((matrix[i][j] - epsilon) / float(m) * basis[counter]), m)
            realDerivP = scipy.linalg.expm(matrix + epsilon * basis[counter])
            realDerivM = scipy.linalg.expm(matrix - epsilon * basis[counter])


            realDeriv = (realDerivP - realDerivM) / (2 * epsilon)
            print("Real:")
            print(realDeriv)
            print("Supposed:")
            print(supposedDeriv)
            print("Difference:")
            print(numpy.linalg.norm(realDeriv-supposedDeriv))
            print()
            counter += 1

def checkRoutine2(matrix, dmatrix):
    expmatrix = scipy.linalg.expm(matrix)
    expmatrixinv = scipy.linalg.expm(-matrix)

    gammatest = expmatrixinv @ dmatrix @ expmatrix

    #print(gammatest @ gammatest - gammatest)


def getNumericalDerivative(matrix, dmatrix, epsilon):
    m1 = scipy.linalg.expm(epsilon * matrix) @ dmatrix @ scipy.linalg.expm(-epsilon * matrix)
    m2 = scipy.linalg.expm(-epsilon * matrix) @ dmatrix @ scipy.linalg.expm(epsilon * matrix)
    return (m2 - m1) / (2 * epsilon)



for i in range(dimension):
    matrix[i][i] = 0

for i in range(1, dimension):
    for j in range(i-1, dimension):
        matrix[j][i] = - matrix[i][j]


dmatrix = numpy.zeros((dimension, dimension))
for i in range(occupancy):
    dmatrix[i][i] = 1

#checkRoutine(20, matrix, dmatrix)
#checkRoutine2(matrix, dmatrix)
checkMatrixDerivative(matrix)
