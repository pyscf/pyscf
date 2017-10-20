/*
 * csr matrix structure 
 */
typedef struct scsr_matrix
{
  float *data; // size nnz
  int *RowPtr; // size RowPtrSize
  int *ColInd; // size: nnz
  int nnz, RowPtrSize;
  int m, n; // shape full matrix
  cusparseMatDescr_t descr;

} scsr_matrix;

extern "C" scsr_matrix init_sparse_matrix_csr_gpu_float(float *csrValA, int *csrRowPtrA, 
    int *csrColIndA, int m, int n, int nnz, int RowPtrSize);
extern "C" void free_csr_matrix_gpu(scsr_matrix csr);

extern "C" void init_tddft_iter_gpu(float *X4, int norbs_in, float *ksn2e,
    float *ksn2f, int nfermi_in, int nprod_in, int vstart_in,
    float *cc_da_vals, int *cc_da_rowPtr, int *cc_da_col_ind,
    int *cc_da_shape, int cc_da_nnz, int cc_da_indptr_size,
    float *v_dab_vals, int *v_dab_rowPtr, int *v_dab_col_ind,
    int *v_dab_shape, int v_dab_nnz, int v_dab_indptr_size);

extern "C" void free_device();
extern "C" void apply_rf0_device(float *v_ext_real, float *v_ext_imag, float *temp);
