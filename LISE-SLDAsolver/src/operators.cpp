// for license information, see the accompanying LICENSE file
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <mpi.h>

#include "common.h"

void gradient_real(
    const double* f, 
    int n, 
    double* g_x, double* g_y, double* g_z, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int ired
)
{
    /*
    ired = 1: reduction
         = anything else : no reduction
    */
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    double* gr_x = AllocateZeroed<double>(n);
    double* gr_y = AllocateZeroed<double>(n);
    double* gr_z = AllocateZeroed<double>(n);

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        double sum = 0;
#pragma omp parallel for default(shared) private(ix2) reduction(+:sum) 
        for (int ix2 = 0; ix2 < nx; ix2++)
        {
            sum += std::real(d1_x[ix1 - ix2 + nx1] * f[iz1 + nz * (iy1 + ny * ix2)]);
        }
        gr_x[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iz2) reduction(+:sum) 
        for (int iz2 = 0; iz2 < nz; iz2++)
        {
            sum += std::real(d1_z[iz1 - iz2 + nz1] * f[iz2 + nz * (iy1 + ny * ix1)]);
        }
        gr_z[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iy2) reduction(+:sum) 
        for (int iy2 = 0; iy2 < ny; iy2++)
        {
            sum += std::real(d1_y[iy1 - iy2 + ny1] * f[iz1 + nz * (iy2 + ny * ix1)]);
        }
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
        CopyMemory(g_x, n, gr_x);
        CopyMemory(g_y, n, gr_y);
        CopyMemory(g_z, n, gr_z);
    }

    Free(gr_x); 
    Free(gr_y); 
    Free(gr_z);
}

void gradient_real_orig(
    const double* f, 
    int n, 
    double* g_x, double* g_y, double* g_z, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int ired
)
{
    /*
    ired = 1: reduction
         = anything else : no reduction
    */
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;
    
    double* gr_x = AllocateZeroed<double>(n);
    double* gr_y = AllocateZeroed<double>(n);
    double* gr_z = AllocateZeroed<double>(n);

    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        for (int j = 0; j < n; j++)
        {
            int ix2, iy2, iz2;
            i2xyz(j, &ix2, &iy2, &iz2, ny, nz);

            if (iy1 == iy2 && iz1 == iz2)
                gr_x[i] += std::real(d1_x[ix1 - ix2 + nx1] * f[j]);

            if (iy1 == iy2 && ix1 == ix2)
                gr_z[i] += std::real(d1_z[iz1 - iz2 + nz1] * f[j]);

            if (ix1 == ix2 && iz1 == iz2)
                gr_y[i] += std::real(d1_y[iy1 - iy2 + ny1] * f[j]);
        }
    }

    if (ired == 1) 
    {
        /* all reduce */
        MPI_Allreduce(gr_x, g_x, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gr_y, g_y, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gr_z, g_z, n, MPI_DOUBLE, MPI_SUM, comm);
    }
    else
    {
        CopyMemory(g_x, n, gr_x);
        CopyMemory(g_y, n, gr_y);
        CopyMemory(g_z, n, gr_z);
    }

    Free(gr_x); 
    Free(gr_y); 
    Free(gr_z);
}

void gradient(
    const complex* f, 
    int n, 
    complex* g_x, complex* g_y, complex* g_z, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int ired
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    ZeroMemory(g_x, n);
    ZeroMemory(g_y, n);
    ZeroMemory(g_z, n);

#pragma omp parallel for default(shared) private(i)
    for (int i = nstart; i < nstop; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        complex sum = 0;
#pragma omp parallel for default(shared) private(ix2) reduction(+:sum) 
        for (int ix2 = 0; ix2 < nx; ix2++)
        {
            sum += d1_x[ix1 - ix2 + nx1] * f[iz1 + nz * (iy1 + ny * ix2)];
        }
        g_x[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iz2) reduction(+:sum) 
        for (int iz2 = 0; iz2 < nz; iz2++)
        {
            sum += d1_z[iz1 - iz2 + nz1] * f[iz2 + nz * (iy1 + ny * ix1)];
        }
        g_z[i] += sum;

        sum = 0;
#pragma omp parallel for default(shared) private(iy2) reduction(+:sum) 
        for (int iy2 = 0; iy2 < ny; iy2++)
        {
            sum += d1_y[iy1 - iy2 + ny1] * f[iz1 + nz * (iy2 + ny * ix1)];
        }
        g_y[i] += sum;
    }

    if (ired == 1)
    {
        complex* gx = AllocateCopy<complex>(n, g_x);
        complex* gy = AllocateCopy<complex>(n, g_y);
        complex* gz = AllocateCopy<complex>(n, g_z);
        
        MPI_Allreduce(gx, g_x, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gy, g_y, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gz, g_z, n, MPI_DOUBLE, MPI_SUM, comm);

        Free(gx); 
        Free(gy); 
        Free(gz);
    }
}

void gradient_ud(
    const complex* f, 
    int n, 
    complex* g_x, complex* g_y, complex* g_z, 
    complex* g_xd, complex* g_yd, complex* g_zd, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz
)
{
    ZeroMemory(g_x, nstop - nstart);
    ZeroMemory(g_y, nstop - nstart);
    ZeroMemory(g_z, nstop - nstart);

    ZeroMemory(g_xd, nstop - nstart);
    ZeroMemory(g_yd, nstop - nstart);
    ZeroMemory(g_zd, nstop - nstart);

    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    for (int i = 0; i < nstop - nstart; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i + nstart, &ix1, &iy1, &iz1, ny, nz);

        complex sum1 = 0;
        complex sum2 = 0;

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

void gradient_orig(
    const complex* f, 
    int n, 
    complex* g_x, complex* g_y, complex* g_z, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int ired
)
{
    ZeroMemory(g_x, n);
    ZeroMemory(g_y, n);
    ZeroMemory(g_z, n);

    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

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
        complex* gx = AllocateCopy<complex>(n, g_x);
        complex* gy = AllocateCopy<complex>(n, g_y);
        complex* gz = AllocateCopy<complex>(n, g_z);

        MPI_Allreduce(gx, g_x, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gy, g_y, n, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(gz, g_z, n, MPI_DOUBLE, MPI_SUM, comm);

        Free(gx); 
        Free(gy); 
        Free(gz);
    }
}

void laplacean_complex(
    const complex* f, 
    int n, 
    complex* lapf, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    int nx, int ny, int nz, 
    int ired
)
{
    ZeroMemory(lapf, n);

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
        complex* tmp = AllocateCopy<complex>(n, lapf);
        MPI_Allreduce(tmp, lapf, n, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
        Free(tmp);
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
    ZeroMemory(lapf, n);

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
        double* tmp = AllocateCopy<double>(n, lapf);

        MPI_Allreduce(tmp, lapf, n, MPI_DOUBLE, MPI_SUM, comm);
        Free(tmp);
    }
}

void diverg(
    const double* fx, const double* fy, const double* fz, 
    double* divf, 
    int n,
    int nstart,  int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    double* divf_r = AllocateZeroed<double>(n);

    for (int i = 0; i < n; i++)
    {
        int ix1, iy1, iz1;
        i2xyz(i, &ix1, &iy1, &iz1, ny, nz);

        for (int j = nstart; j < nstop; j++)
        {
            int ix2, iy2, iz2;
            i2xyz(j, &ix2, &iy2, &iz2, ny, nz);

            if (iy1 == iy2 && iz1 == iz2)
                divf_r[i] += std::real(d1_x[ix1 - ix2 + nx1] * fx[j]);

            if (iy1 == iy2 && ix1 == ix2)
                divf_r[i] += std::real(d1_z[iz1 - iz2 + nz1] * fz[j]);

            if (ix1 == ix2 && iz1 == iz2)
                divf_r[i] += std::real(d1_y[iy1 - iy2 + ny1] * fy[j]);
        }
    }

    MPI_Allreduce(divf_r, divf, n, MPI_DOUBLE, MPI_SUM, comm);
    Free(divf_r);
}

void match_lattices(
    Lattice_arrays* latt3, 
    int nx, int ny, int nz, 
    int nx3, int ny3, int nz3, 
    FFtransf_vars* fftrans, 
    double Lc
) 
{
    double fpi = 4.0 * PI * 197.3269631 / 137.035999679;  /* 4 * pi * e2 */

    int nxyz = nx * ny * nz;
    int nxyz3 = fftrans->nxyz3;

    fftrans->i_l2s = Allocate<int>(nxyz3);
    fftrans->fc = Allocate<double>(nxyz3);
    fftrans->i_s2l = Allocate<int>(nxyz);

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

    fftrans->fc[0] = fpi * 0.5 * Lc * Lc;

    for (int i = 1; i < nxyz3; i++)
    {
        fftrans->fc[i] = fpi * (1.0 - cos(sqrt(latt3->kin[i]) * Lc)) / latt3->kin[i];
    }

    Free(latt3->kx);
    Free(latt3->ky);
    Free(latt3->kz);
    Free(latt3->xa);
    Free(latt3->ya);
    Free(latt3->za);
    Free(latt3->kin);
}
