import h5py
import numpy as np
import time
from pyscf import lib

# create data

NROW = 200
NCOL = 500000

data = np.random.rand(NROW, NCOL)

data_read = np.zeros((NROW, NCOL))

# different chunsizes

chunksizes = [(10, 10000),  (14, 10000),  (21, 10000),  (40, 10000),  (50, 10000), 
              (10, 100000), (14, 100000), (21, 100000), (40, 100000), (50, 100000),
              (10, 150000), (14, 150000), (21, 150000), (40, 150000), (50, 150000),
              (10, 500000), (14, 500000), (21, 500000), (40, 500000), (50, 500000)]

# write data with h5py

for chunksize in chunksizes:
    with h5py.File(f'h5py_test_{chunksize[0]}_{chunksize[1]}.h5', 'w') as f:
        dset = f.create_dataset("data", data=data, chunks=chunksize)
        start_time = time.time()
        for i in range(4):
            dset.write_direct(data)
            # dset[:] = data
        print(f"h5py Write whole data - Chunksize {chunksize}: {time.time() - start_time} seconds")
        
        # test write with different chunksize

        for chunksize2 in chunksizes:
            start_time = time.time()
            for i in range(4):
                for row0, row1 in lib.prange(0, NROW, chunksize2[0]):
                    for col0, col1 in lib.prange(0, NCOL, chunksize2[1]):
                        source_sel = np.s_[row0:row1, col0:col1]
                        dset.write_direct(data, source_sel=source_sel, dest_sel=source_sel)
            print(f"h5py Write - Chunksize {chunksize2}: {time.time() - start_time} seconds")
            # verify the correctness 
            f["data"].read_direct(data_read)
            assert np.allclose(data, data_read)

# read chunk

for chunksize in chunksizes:
    with h5py.File(f'h5py_test_{chunksize[0]}_{chunksize[1]}.h5', 'r') as f:
        # f["data"].write_direct(data)
        start_time = time.time()
        for i in range(4):
            f["data"].read_direct(data_read)
        print(f"h5py Read whole data - Chunksize {chunksize}: {time.time() - start_time} seconds")

        # assert np.allclose(data, data_read)

        # test read with different chunksize

        for chunksize2 in chunksizes:
            data_read_tmp = np.zeros((chunksize2[0], chunksize2[1]))
            dest_sel = np.s_[:chunksize2[0], :chunksize2[1]]
            start_time = time.time()
            for i in range(4):
                for row0, row1 in lib.prange(0, NROW, chunksize2[0]):
                    for col0, col1 in lib.prange(0, NCOL, chunksize2[1]):
                        # create buffer for reading
                        data_read_tmp2 = np.ndarray((row1-row0, col1-col0), dtype=data.dtype, buffer=data_read_tmp, order='C')
                        dest_sel      = np.s_[:, :]
                        source_sel = np.s_[row0:row1, col0:col1]
                        f["data"].read_direct(data_read_tmp2, source_sel=source_sel, dest_sel=dest_sel)
            print(f"h5py Read - Chunksize {chunksize2}: {time.time() - start_time} seconds")
