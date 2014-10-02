/*
 *
 */

struct _AO2MOEnvs {
        int natm;
        int nbas;
        const int *atm;
        const int *bas;
        const double *env;
        int nao;
        int ish_start;
        int ish_count;
        int bra_start;
        int bra_count;
        int ket_start;
        int ket_count;
        int *ao_loc;
        int *idx_tri;
};

void AO2MOdtrilmm_o1(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c);
void AO2MOdtrilmm_o2(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c);
void AO2MOdtriumm_o1(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c);
void AO2MOdtriumm_o2(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c);

void AO2MOnr_tri_e2_o0(double *vout, double *vin, double *mo_coeff, int nao,
                       int i_start, int i_count, int j_start, int j_count);
void AO2MOnr_tri_e2_o1(double *vout, double *vin, double *mo_coeff, int nao,
                       int i_start, int i_count, int j_start, int j_count);
void AO2MOnr_tri_e2_o2(double *vout, double *vin, double *mo_coeff, int nao,
                       int i_start, int i_count, int j_start, int j_count);

void AO2MOnr_e1_o0(double *eri, double *mo_coeff,
                   int i_start, int i_count, int j_start, int j_count,
                   int *atm, int natm, int *bas, int nbas, double *env);

void AO2MOnr_e1_o1(double *eri, double *mo_coeff,
                   int i_start, int i_count, int j_start, int j_count,
                   int *atm, int natm, int *bas, int nbas, double *env);

void AO2MOnr_e1_o2(double *eri, double *mo_coeff,
                   int i_start, int i_count, int j_start, int j_count,
                   int *atm, int natm, int *bas, int nbas, double *env);

void AO2MOnr_e2_o0(const int nrow, double *vout, double *vin,
                   double *mo_coeff, const int nao,
                   int i_start, int i_count, int j_start, int j_count);

void AO2MOnr_e2_o1(const int nrow, double *vout, double *vin,
                   double *mo_coeff, const int nao,
                   int i_start, int i_count, int j_start, int j_count);

void AO2MOnr_e2_o2(const int nrow, double *vout, double *vin,
                   double *mo_coeff, const int nao,
                   int i_start, int i_count, int j_start, int j_count);

int AO2MOcount_ij(int i_start, int i_count, int j_start, int j_count);

void AO2MOnr_e1_drv(double *eri, double *mo_coeff,
                    void (*ftransform_kl)(), void (*ftrans_e1)(),
                    int ijshl_start, int ijshl_count,
                    int k_start, int k_count, int l_start, int l_count,
                    int *atm, int natm, int *bas, int nbas, double *env);

void AO2MOnr_e2_drv(const int nrow, double *vout, double *vin,
                    double *mo_coeff, const int nao,
                    void (*const ftrans_e2)(),
                    int i_start, int i_count, int j_start, int j_count);
