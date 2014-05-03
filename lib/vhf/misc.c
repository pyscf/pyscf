/*
 *
 */

#include <string.h>

void CVHFunpack(int n, double *vec, double *mat)
{
        int i, j;
        for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, vec++) {
                        mat[i*n+j] = *vec;
                        mat[j*n+i] = *vec;
                }
        }
}


void extract_row_from_tri(double *row, unsigned int row_id, unsigned int ndim,
                          double *tri)
{
        unsigned long idx;
        unsigned int i;
        idx = (unsigned long)row_id * (row_id + 1) / 2;
        memcpy(row, tri+idx, sizeof(double)*row_id);
        for (i = row_id; i < ndim; i++) {
                idx += i;
                row[i] = tri[idx];
        }
}

