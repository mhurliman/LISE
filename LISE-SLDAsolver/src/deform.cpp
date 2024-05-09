// for license information, see the accompanying LICENSE file
/*
compute the deformation parameters beta and gamma
gamma not computed yet
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "potentials.h"

void deform(const double* rho_p, const double* rho_n, int nxyz, const Lattice_arrays* latt_coords, double dxyz, FILE* fout)
{
    double* rho = AllocateInit<double>(nxyz, [&](int i) { return rho_p[i] + rho_n[i]; });

    double xcm, ycm, zcm;
    double num = center_dist(rho, nxyz, latt_coords, &xcm, &ycm, &zcm);

    double r2 = 0; 
    double q30 = 0;
    double q[3][3]{};

    for (int i = 0; i < nxyz; ++i)
    {
        double xa = latt_coords->xa[i];
        double ya = latt_coords->ya[i];
        double za = latt_coords->za[i];
        
        double _q30 = za * (2 * za * za - 3 * xa * xa - 3 * ya * ya) * sqrt(7 / PI) / 4.0;
        xa = xa * xa - 2.0 * xa * latt_coords->wx[i] * xcm + xcm * xcm;
        ya = ya * ya - 2.0 * ya * latt_coords->wy[i] * ycm + ycm * ycm;
        za = za * za - 2.0 * za * latt_coords->wz[i] * zcm + zcm * zcm;

        double xa1 = latt_coords->wx[i] * latt_coords->xa[i] - xcm;
        double ya1 = latt_coords->wy[i] * latt_coords->ya[i] - ycm;
        double za1 = latt_coords->wz[i] * latt_coords->za[i] - zcm;

        r2 += (xa + ya + za) * rho[i];
        q[0][0] += (2.0 * xa - ya - za) * rho[i];
        q[1][1] += (-xa + 2.0 * ya - za) * rho[i];
        q[2][2] += (-xa - ya + 2.0 * za) * rho[i];
        q[0][1] += (3.0 * xa1 * ya1 * rho[i]);
        q[0][2] += (3.0 * xa1 * za1 * rho[i]);
        q[1][2] += (3.0 * ya1 * za1 * rho[i]);

        q30 += _q30 * rho[i];
    }
    
    Free(rho);
    
    q[1][0] = q[0][1];
    q[2][0] = q[0][2];
    q[2][1] = q[1][2];

    printf("qxx = %g fm^2  qyy = %g fm^2   qzz = %g fm^2\n", q[0][0] * dxyz, q[1][1] * dxyz, q[2][2] * dxyz);
    printf("xcm = %g fm  ycm = %g fm   zcm = %g fm \n\n", xcm, ycm, zcm);
    printf("q30 = %g fm^3\n", q30 * dxyz);

    fprintf(fout, "qxx = %g   qyy = %g    qzz = %g \n", q[0][0] * dxyz, q[1][1] * dxyz, q[2][2] * dxyz);
    fprintf(fout, "xcm = %g fm  ycm = %g fm   zcm = %g fm \n\n", xcm, ycm, zcm);
    fprintf(fout, "q30 = %g b^3\n", q30 * dxyz / 1000.0);

    double beta = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            beta += q[i][j] * q[j][i];
        }
    }

    beta = 5.0 * sqrt(beta / 216.0) / r2;
    printf(" beta= %g \n", beta);
    fprintf(fout, " beta= %g \n", beta);

    beta = sqrt(PI / 5.0) * q[2][2] / r2;
    printf(" beta_2 = %9.6f \n", beta);
    fprintf(fout, " beta_2 = %9.6f \n", beta);
}

double q2av(double* rho_p, double* rho_n, double* qzz, const int nxyz, const double np, const double nn) 
{
    double sum = 0;
    int i;
    double sum_p = 0, sum_n = 0;

    for (i = 0;i < nxyz;i++) 
    {
        sum_p += rho_p[i];
        sum_n += rho_n[i];
    }

    sum = 0.;
    for (i = 0;i < nxyz;i++)
        sum += (rho_p[i] + rho_n[i]) * qzz[i];

    return sum;
}

double cm_coord(double* rho_p, double* rho_n, double* qzz, const int nxyz, const double np, const double nn) 
{
    double sum = 0;
    int i;
    double sum_p = 0, sum_n = 0;

    for (i = 0;i < nxyz;i++) 
    {
        sum_p += rho_p[i];
        sum_n += rho_n[i];
    }
    sum_p /= np;
    sum_n /= nn;

    sum = 0.;
    for (i = 0;i < nxyz;i++)
        sum += (rho_p[i] / sum_p + rho_n[i] / sum_n) * qzz[i];

    return(sum / (nn + np));
}
