__global__ void normalize_energy_gpu(float *ksn2e, float *ksn2f, double omega_re, double omega_im, 
    float *nm2v_re, float *nm2v_im, int nfermi, int nprod, int nvirt, int vstart);
__global__ void padding_nm2v( float *nm2v_re, float *nm2v_im, int nfermi, int norbs, int nvirt, int vstart);

extern "C" void init_tddft_iter_gpu(float *X4, int norbs_in, float *ksn2e,
    float *ksn2f, int nfermi_in, int nprod_in, int vstart_in);

extern "C" void apply_rf0_device(float *sab_real, float *sab_imag, double comega_re, 
    double comega_im, int *block_size, int *grid_size);

extern "C" void free_device();

extern "C" void memcpy_sab_host2device(float *sab, int Async);
extern "C" void memcpy_sab_device2host(float *sab, int Async);

extern "C" void calc_nb2v_from_sab(int reim);
extern "C" void get_nm2v_real();
extern "C" void get_nm2v_imag();
extern "C" void calc_nb2v_from_nm2v_real();
extern "C" void calc_nb2v_from_nm2v_imag();
extern "C" void get_sab(int reim);
extern "C" void div_eigenenergy_gpu(double omega_re, double omega_im, 
    int *block_size, int *grid_size);
