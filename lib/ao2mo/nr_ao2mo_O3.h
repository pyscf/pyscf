/*
 * File: nr_ao2mo_O3.h
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

void trans_e2_tri_o0(double *vout, double *vin, double *mo_coeff, int nao,
                     int i_start, int i_count, int j_start, int j_count);
void trans_e2_tri_o1(double *vout, double *vin, double *mo_coeff, int nao,
                     int i_start, int i_count, int j_start, int j_count);
void trans_e2_tri_o2(double *vout, double *vin, double *mo_coeff, int nao,
                     int i_start, int i_count, int j_start, int j_count);

void nr_e1_ao2mo_o0(double *eri, double *mo_coeff,
                    int i_start, int i_count, int j_start, int j_count,
                    const int *atm, const int natm,
                    const int *bas, const int nbas, const double *env);

void nr_e1_ao2mo_o1(double *eri, double *mo_coeff,
                    int i_start, int i_count, int j_start, int j_count,
                    const int *atm, const int natm,
                    const int *bas, const int nbas, const double *env);

void nr_e1_ao2mo_o2(double *eri, double *mo_coeff,
                    int i_start, int i_count, int j_start, int j_count,
                    const int *atm, const int natm,
                    const int *bas, const int nbas, const double *env);

void nr_e2_ao2mo_o0(const int nrow, double *vout, double *vin,
                    double *mo_coeff, const int nao,
                    int i_start, int i_count, int j_start, int j_count);

void nr_e2_ao2mo_o1(const int nrow, double *vout, double *vin,
                    double *mo_coeff, const int nao,
                    int i_start, int i_count, int j_start, int j_count);

void nr_e2_ao2mo_o2(const int nrow, double *vout, double *vin,
                    double *mo_coeff, const int nao,
                    int i_start, int i_count, int j_start, int j_count);

