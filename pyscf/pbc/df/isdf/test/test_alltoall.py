from mpi4py import MPI
import numpy
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a_size = 2
senddata = (rank+1)*numpy.arange(size*a_size, dtype=int).reshape(a_size, size)
recvdata = numpy.empty(size*a_size, dtype=int)
comm.Alltoall(senddata.T.copy(), recvdata)

print("process %s sending %s receiving %s " % (rank, senddata, recvdata))

# mpiexec -n 10 python mpialltoall.py

def matrix_all2all_Row2Col(comm, nRow, nCol, Mat):
    assert nRow % size == 0
    assert nCol % size == 0
    sendbuf = Mat.copy()
    assert sendbuf.shape == (nRow, nCol//size)
    recvbuf = numpy.empty((Mat.size,), dtype=Mat.dtype)
    comm.Alltoall(sendbuf, recvbuf)
    Mat_packed = numpy.empty((nRow//size, nCol), dtype=Mat.dtype)
    for i in range(size):
        Mat_packed[:, nCol//size * i: nCol//size * (i+1)] = recvbuf[recvbuf.size//size * i: recvbuf.size//size * (i+1)].reshape(nRow//size, -1)
    return Mat_packed

def matrix_all2all_Col2Row(comm, nRow, nCol, Mat):
    assert nRow % size == 0
    assert nCol % size == 0
    sendbuf = numpy.empty((Mat.size,), dtype=Mat.dtype)
    for i in range(size):
        sendbuf[sendbuf.size//size * i: sendbuf.size//size * (i+1)] = Mat[:, nCol//size * i: nCol//size * (i+1)].ravel()[:]    
    recvbuf = numpy.empty((Mat.size,), dtype=Mat.dtype)
    comm.Alltoall(sendbuf, recvbuf)
    return recvbuf.reshape(nRow, nCol//size)

matrix_test = np.random.rand(10 * size, 8)

mat1 = matrix_all2all_Row2Col(comm, matrix_test.shape[0], matrix_test.shape[1]*size, matrix_test)
print("mat1 = ", mat1.shape)

mat2 = matrix_all2all_Col2Row(comm, matrix_test.shape[0], matrix_test.shape[1]*size, mat1)

assert np.allclose(matrix_test, mat2)
