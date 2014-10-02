/*
 *
 */

/* japcp = apcp * 4
 *       - acpp.transpose(0,2,1,3)
 *       - apcp.transpose(0,3,2,1)
 */
// transform last three indices
void MCSCFinplace_apcp(double *japcp, double *ppp,
                       int ncore, int ncas, int nmo)
{
        int j, k, l;
        int d1 = nmo;
        int d2 = nmo * nmo;
        int o1 = nmo;
        double *apcp;

        for (j = 0; j < nmo; j++) {
                apcp = ppp + j * d2;
                for (k = 0; k < ncore; k++) {
                for (l = 0; l < nmo; l++) {
                        japcp[k*o1+l] = apcp[k*d1+l] * 4
                                - ppp[j*d1+k*d2+l] - ppp[j+k*d1+l*d2];
                } }
                japcp += ncore * nmo;
        }
}

/* jcvcp += cvcp * 4
 *        - cvcp.transpose(2,1,0,3)
 *        - ccvp.transpose(0,2,1,3)
 */
// for given index i, transform last three indices
void MCSCFinplace_cvcp(double *jcvcp, double *vcp, double *cpp, int i,
                       int ncore, int ncas, int nmo)
{
        int nvir = nmo - ncore;
        int j, k, l, n;
        int d1 = nmo;
        int d2 = ncore * nmo;
        int dd = nmo * nmo;
        unsigned long d3 = nvir * ncore * nmo;
        double *pixxx = jcvcp + d3 * i;
        double *pxxix = jcvcp + d1 * i;
        double *cvp = cpp + ncore * d1;

        for (n = 0, j = 0; j < nvir; j++) {
                for (k = 0; k < ncore; k++) {
                for (l = 0; l < nmo; l++, n++) {
                        pixxx[k*d1+l] += vcp[n] * 4 - cvp[j*d1+k*dd+l];
                        pxxix[j*d2+d3*k+l] -= vcp[j*d2+k*d1+l];
                } }
                pixxx += d2;
        }
}

