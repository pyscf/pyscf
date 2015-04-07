/*
 *
 */

void AO2MOnr_incore8f_acc(double *vout, double *eri, double *mo_coeff,
                          int row_start, int row_count, int nao,
                          int i_start, int i_count, int j_start, int j_count,
                          void (*ftrans)());
void AO2MOnr_incore4f_acc(double *vout, double *eri, double *mo_coeff,
                          int row_start, int row_count, int nao,
                          int i_start, int i_count, int j_start, int j_count,
                          void (*ftrans)());

void AO2MOnr_e1incore_drv(double *eri_mo, double *eri_ao, double *mo_coeff,
                          void (*facc)(), void (*ftrans)(), int nao,
                          int i_start, int i_count, int j_start, int j_count);

