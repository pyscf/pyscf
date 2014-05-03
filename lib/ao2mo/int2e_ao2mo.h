/*
 * File: int2e_ao2mo.h
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#define DATASET_ERI     "eri_mo"

void init_eri_file(char filename[], int nao, int nmo);
void save_page_eri(char filename[], double *page, int page_cols,
                   int row_count, int col_count,
                   int row_start, int col_start);
void load_page_eri(char finlename[], double *page, int rows, int cols,
                   int row_start, int col_start);

void load_all_eri_mo(char filename[], double *page, const int nmo);


long uppertri_index(int i, int j, int n);
long lowertri_index(int i, int j);
#define LOWERTRI_INDEX(I,J)     ((I) > (J) ? ((I)*((I)+1)/2+(J)) : ((J)*((J)+1)/2+(I)))

void pack_uppertri(double *mat, int ndim, double *v, int incv);
void pack_lowertri(double *mat, int ndim, double *v, int incv);
void unpack_uppertri(double *mat, int ndim, double *v, int incv);
void unpack_lowertri(double *mat, int ndim, double *v, int incv);
void extract_shells_by_id(int id, int *i, int*j);

void set_mem_size(long n);

void int2e_ao2mo(char filename[], double *mo_coeff, const int nmo,
                 const int *atm, const int natm,
                 const int *bas, const int nbas, const double *env);

void int2e_ao2mo_o1(char filename[], double *mo_coeff, const int nmo,
                    const int *atm, const int natm,
                    const int *bas, const int nbas, const double *env);

void int2e_ao2mo_o2(char filename[], double *mo_coeff, const int nmo,
                    const int *atm, const int natm,
                    const int *bas, const int nbas, const double *env);

void int2e_sph_o3(double *eri, const int *atm, const int natm,
                  const int *bas, const int nbas, const double *env);
