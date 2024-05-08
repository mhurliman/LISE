// for license information, see the accompanying LICENSE file
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <mpi.h>

#include "vars_nuclear.h"

void i2xyz(const int i, int* ix, int* iy, int* iz, const int ny, const int nz);

void gradient_real(double* f, const int n, double* g_x, double* g_y, double* g_z, const int nstart, const int nstop, const MPI_Comm comm, double complex* d1_x, double complex* d1_y, double complex* d1_z, const int nx, const int ny, const int nz, const int ired)
{
    /*
    ired = 1: reduction
         = anything else : no reduction
    */
    double* gr_x, * gr_y, * gr_z;
    assert(gr_x = malloc(n * sizeof(double)));
    assert(gr_y = malloc(n * sizeof(double)));
    assert(gr_z = malloc(n * sizeof(double)));

    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    for (int i = 0; i < n; i++)
    {
        gr_x[i] = 0;
        gr_y[i] = 0;
        gr_z[i] = 0;
    }

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        double sum = 0;
#pragma omp parallel for default(shared) private(ix2) reduction(+:sum) 
        for (int ix2 = 0; ix2 < nx; ix2++)
            sum += creal(d1_x[ix1 - ix2 + nx1] * f[iz1 + nz * (iy1 + ny * ix2)]);

        gr_x[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iz2) reduction(+:sum) 
        for (int iz2 = 0; iz2 < nz; iz2++)
            sum += creal(d1_z[iz1 - iz2 + nz1] * f[iz2 + nz * (iy1 + ny * ix1)]);

        gr_z[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iy2) reduction(+:sum) 
        for (int iy2 = 0; iy2 < ny; iy2++)
            sum += creal(d1_y[iy1 - iy2 + ny1] * f[iz1 + nz * (iy2 + ny * ix1)]);

        gr_y[i] += sum;
    }

    if (ired == 1) /* all reduce */
    {
        MPI_Allreduce(gr_x, g_x, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gr_y, g_y, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gr_z, g_z, n, MPI_DOUBLE, MPI_SUM, comm);
    }
    else 
    {
#pragma omp parallel for default(shared) private(i)
        for (int i = 0; i < n; i++)
        {
            g_x[i] = gr_x[i];
            g_y[i] = gr_y[i];
            g_z[i] = gr_z[i];
        }
    }

    free(gr_x); 
    free(gr_y); 
    free(gr_z);
}

void gradient_real_orig(double* f, const int n, double* g_x, double* g_y, double* g_z, const int nstart, const int nstop, const MPI_Comm comm, double complex* d1_x, double complex* d1_y, double complex* d1_z, const int nx, const int ny, const int nz, const int ired)
{
    /*
    ired = 1: reduction
         = anything else : no reduction
    */
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;
    
    double* gr_x, * gr_y, * gr_z;
    assert(gr_x = malloc(n * sizeof(double)));
    assert(gr_y = malloc(n * sizeof(double)));
    assert(gr_z = malloc(n * sizeof(double)));

    for (int i = 0; i < n; i++)
    {
        gr_x[i] = 0;
        gr_y[i] = 0;
        gr_z[i] = 0;
    }

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        for (int j = 0; j < n; j++)
        {
            int ix2, iy2, iz2;
            i2xyz(j, &ix2, &iy2, &iz2, ny, nz);

            if (iy1 == iy2 && iz1 == iz2)
                gr_x[i] += creal(d1_x[ix1 - ix2 + nx1] * f[j]);

            if (iy1 == iy2 && ix1 == ix2)
                gr_z[i] += creal(d1_z[iz1 - iz2 + nz1] * f[j]);

            if (ix1 == ix2 && iz1 == iz2)
                gr_y[i] += creal(d1_y[iy1 - iy2 + ny1] * f[j]);
        }
    }

    if (ired == 1) /* all reduce */
    {
        MPI_Allreduce(gr_x, g_x, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gr_y, g_y, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gr_z, g_z, n, MPI_DOUBLE, MPI_SUM, comm);
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            g_x[i] = gr_x[i];
            g_y[i] = gr_y[i];
            g_z[i] = gr_z[i];
        }
    }

    free(gr_x); 
    free(gr_y); 
    free(gr_z);
}

void gradient(double complex* f, const int n, double complex* g_x, double complex* g_y, double complex* g_z, const int nstart, const int nstop, const MPI_Comm comm, double complex* d1_x, double complex* d1_y, double complex* d1_z, const int nx, const int ny, const int nz, const int ired)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

#pragma omp parallel for default(shared) private(i)
    for (int i = 0; i < n; i++)
    {
        g_x[i] = 0;
        g_y[i] = 0;
        g_z[i] = 0;
    }

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        double complex sum = 0;
#pragma omp parallel for default(shared) private(ix2) reduction(+:sum) 
        for (int ix2 = 0; ix2 < nx; ix2++)
            sum += d1_x[ix1 - ix2 + nx1] * f[iz1 + nz * (iy1 + ny * ix2)];

        g_x[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iz2) reduction(+:sum) 
        for (int iz2 = 0; iz2 < nz; iz2++)
            sum += d1_z[iz1 - iz2 + nz1] * f[iz2 + nz * (iy1 + ny * ix1)];

        g_z[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iy2) reduction(+:sum) 
        for (int iy2 = 0; iy2 < ny; iy2++)
            sum += d1_y[iy1 - iy2 + ny1] * f[iz1 + nz * (iy2 + ny * ix1)];

        g_y[i] += sum;
    }

    if (ired == 1)
    {
        double complex* gx, * gy, * gz;
        assert(gx = malloc(n * sizeof(double)));
        assert(gy = malloc(n * sizeof(double)));
        assert(gz = malloc(n * sizeof(double)));

        for (int i = 0; i < n; i++)
        {
            gx[i] = g_x[i];
            gy[i] = g_y[i];
            gz[i] = g_z[i];
        }
        
        MPI_Allreduce(gx, g_x, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gy, g_y, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gz, g_z, n, MPI_DOUBLE, MPI_SUM, comm);

        free(gx); 
        free(gy); 
        free(gz);
    }
}

void gradient_ud(double complex* f, const int n, double complex* g_x, double complex* g_y, double complex* g_z, double complex* g_xd, double complex* g_yd, double complex* g_zd, const int nstart, const int nstop, const MPI_Comm comm, double complex* d1_x, double complex* d1_y, double complex* d1_z, const int nx, const int ny, const int nz)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;
    
    for (int i = 0; i < nstop - nstart; i++)
    {
        g_x[i] = 0;
        g_y[i] = 0;
        g_z[i] = 0;
        g_xd[i] = 0;
        g_yd[i] = 0;
        g_zd[i] = 0;
    }

    for (int i = 0; i < nstop - nstart; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i + nstart, &ix1, &iy1, &iz1, ny, nz);

        double complex sum1 = 0;
        double complex sum2 = 0;

#pragma omp parallel for default(shared) private(ix2) reduction(+:sum1,sum2) 
        for (int ix2 = 0; ix2 < nx; ix2++)
        {
            sum1 += d1_x[ix1 - ix2 + nx1] * f[iz1 + nz * (iy1 + ny * ix2)];
            sum2 += d1_x[ix1 - ix2 + nx1] * f[iz1 + nz * (iy1 + ny * ix2) + n];
        }

        g_x[i] += sum1;
        g_xd[i] += sum2;

        sum1 = 0;
        sum2 = 0;

#pragma omp parallel for default(shared) private(iz2) reduction(+:sum1,sum2) 
        for (int iz2 = 0; iz2 < nz; iz2++)
        {
            sum1 += d1_z[iz1 - iz2 + nz1] * f[iz2 + nz * (iy1 + ny * ix1)];
            sum2 += d1_z[iz1 - iz2 + nz1] * f[iz2 + nz * (iy1 + ny * ix1) + n];
        }

        g_z[i] += sum1;
        g_zd[i] += sum2;

        sum1 = 0;
        sum2 = 0;

#pragma omp parallel for default(shared) private(iy2) reduction(+:sum1,sum2) 
        for (int iy2 = 0; iy2 < ny; iy2++)
        {
            sum1 += d1_y[iy1 - iy2 + ny1] * f[iz1 + nz * (iy2 + ny * ix1)];
            sum2 += d1_y[iy1 - iy2 + ny1] * f[iz1 + nz * (iy2 + ny * ix1) + n];
        }

        g_y[i] += sum1;
        g_yd[i] += sum2;
    }
}

void gradient_orig(double complex* f, const int n, double complex* g_x, double complex* g_y, double complex* g_z, const int nstart, const int nstop, const MPI_Comm comm, double complex* d1_x, double complex* d1_y, double complex* d1_z, const int nx, const int ny, const int nz, const int ired)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    for (int i = 0; i < n; i++)
    {
        g_x[i] = 0;
        g_y[i] = 0;
        g_z[i] = 0;
    }

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        for (int j = 0; j < n; j++)
        {
            int ix2, iy2, iz2;
            i2xyz(j, &ix2, &iy2, &iz2, ny, nz);
            
            if (iy1 == iy2 && iz1 == iz2)
                g_x[i] += d1_x[ix1 - ix2 + nx1] * f[j];

            if (iy1 == iy2 && ix1 == ix2)
                g_z[i] += d1_z[iz1 - iz2 + nz1] * f[j];

            if (ix1 == ix2 && iz1 == iz2)
                g_y[i] += d1_y[iy1 - iy2 + ny1] * f[j];
        }
    }

    if (ired == 1)
    {
        double complex* gx, * gy, * gz;
        assert(gx = malloc(n * sizeof(double)));
        assert(gy = malloc(n * sizeof(double)));
        assert(gz = malloc(n * sizeof(double)));

        for (int i = 0; i < n; i++)
        {
            gx[i] = g_x[i];
            gy[i] = g_y[i];
            gz[i] = g_z[i];
        }

        MPI_Allreduce(gx, g_x, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gy, g_y, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gz, g_z, n, MPI_DOUBLE, MPI_SUM, comm);

        free(gx); 
        free(gy); 
        free(gz);
    }
}

void laplacean_complex(
    const double complex* f, 
    int n, 
    double complex* lapf, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    int nx, int ny, int nz, 
    int ired
)
{
    for (int i = 0; i < n; i++)
    {
        lapf[i] = 0;
    }

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        for (int ix2 = 0; ix2 < nx; ix2++)
            lapf[i] -= (k1d_x[abs(ix1 - ix2)] * f[iz1 + nz * (iy1 + ny * ix2)]);

        for (int iz2 = 0; iz2 < nz; iz2++)
            lapf[i] -= (k1d_z[abs(iz1 - iz2)] * f[iz2 + nz * (iy1 + ny * ix1)]);

        for (int iy2 = 0; iy2 < ny; iy2++)
            lapf[i] -= (k1d_y[abs(iy1 - iy2)] * f[iz1 + nz * (iy2 + ny * ix1)]);
    }

    if (ired == 1)
    {
        double complex* tmp;
        assert(tmp = malloc(n * sizeof(double complex)));

        for (int i = 0; i < n; i++)
            tmp[i] = lapf[i];

        MPI_Allreduce(tmp, lapf, n, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
        free(tmp);
    }
}


void laplacean(
    const double* f, 
    int n, 
    double* lapf, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    int nx, int ny, int nz, 
    int ired
)
{
    memset(lapf, 0, n * sizeof(lapf[0]));

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        for (int ix2 = 0; ix2 < nx; ix2++)
            lapf[i] -= (k1d_x[abs(ix1 - ix2)] * f[iz1 + nz * (iy1 + ny * ix2)]);

        for (int iz2 = 0; iz2 < nz; iz2++)
            lapf[i] -= (k1d_z[abs(iz1 - iz2)] * f[iz2 + nz * (iy1 + ny * ix1)]);

        for (int iy2 = 0; iy2 < ny; iy2++)
            lapf[i] -= (k1d_y[abs(iy1 - iy2)] * f[iz1 + nz * (iy2 + ny * ix1)]);
    }

    if (ired == 1)
    {
        double* tmp;
        assert(tmp = malloc(n * sizeof(double)));

        memcpy(tmp, lapf, n * sizeof(tmp[0]));

        MPI_Allreduce(tmp, lapf, n, MPI_DOUBLE, MPI_SUM, comm);
        free(tmp);
    }
}

void diverg(
    const double* fx, const double* fy, const double* fz, 
    double* divf, 
    int n,  int nstart,  int nstop, 
    MPI_Comm comm, 
    const double complex* d1_x, const double complex* d1_y, const double complex* d1_z, 
    int nx, int ny, int nz
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    double* divf_r;
    assert(divf_r = malloc(n * sizeof(double)));
    memset(divf_r, 0, n * sizeof(divf_r[0]));

    for (int i = 0; i < n; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        for (int j = nstart; j < nstop; j++)
        {
            int ix2, iy2, iz2;
            i2xyz(j, &ix2, &iy2, &iz2, ny, nz);

            if (iy1 == iy2 && iz1 == iz2)
                divf_r[i] += creal(d1_x[ix1 - ix2 + nx1] * fx[j]);

            if (iy1 == iy2 && ix1 == ix2)
                divf_r[i] += creal(d1_z[iz1 - iz2 + nz1] * fz[j]);

            if (ix1 == ix2 && iz1 == iz2)
                divf_r[i] += creal(d1_y[iy1 - iy2 + ny1] * fy[j]);
        }
    }

    MPI_Allreduce(divf_r, divf, n, MPI_DOUBLE, MPI_SUM, comm);
    free(divf_r);
}

void match_lattices(
    Lattice_arrays* latt3, 
    int nx, int ny, int nz, 
    int nx3, int ny3, int nz3, 
    FFtransf_vars* fftrans, 
    double Lc
) 
{
    double fpi = 4.0 * acos(-1.0) * 197.3269631 / 137.035999679;  /* 4 * pi * e2 */

    int nxyz = nx * ny * nz;
    int nxyz3 = fftrans->nxyz3;

    assert(fftrans->i_l2s = malloc(nxyz3 * sizeof(int)));
    assert(fftrans->fc = malloc(nxyz3 * sizeof(double)));
    assert(fftrans->i_s2l = malloc(nxyz * sizeof(int)));

    for (int ix3 = 0; ix3 < nx; ix3++) 
    {
        int ix = ix3;

        for (int iy3 = 0; iy3 < ny; iy3++) 
        {
            int iy = iy3;

            for (int iz3 = 0; iz3 < nz; iz3++) 
            {
                int iz = iz3;

                fftrans->i_s2l[iz + nz * (iy + ny * ix)] = iz3 + nz3 * (iy3 + ny3 * ix3);
                fftrans->i_l2s[iz3 + nz3 * (iy3 + ny3 * ix3)] = iz + nz * (iy + ny * ix);
            }
        }
    }

    fftrans->fc[0] = fpi * .5 * Lc * Lc;

    for (int i = 1; i < nxyz3; i++)
    {
        fftrans->fc[i] = fpi * (1.0 - cos(sqrt(latt3->kin[i]) * Lc)) / latt3->kin[i];
    }

    free(latt3->kx); latt3->kx = NULL;
    free(latt3->ky); latt3->ky = NULL;
    free(latt3->kz); latt3->kz = NULL;
    free(latt3->xa); latt3->xa = NULL;
    free(latt3->ya); latt3->ya = NULL;
    free(latt3->za); latt3->za = NULL;
    free(latt3->kin); latt3->kin = NULL;
}
