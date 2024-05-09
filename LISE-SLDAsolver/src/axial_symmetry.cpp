// for license information, see the accompanying LICENSE file
/*
Axial_symmetry.c
This file contains util functions for axial-symmetry stuff.
author: Shi Jin <js1421@uw.edu>
Date: 12/31/15
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

#include "common.h"

#define EPS 1e-14

struct Points
{
    int ix;
    int iy;
    double norm1;  // |xx| + |yy|
    double norm2;  // sqrt(xx**2 + yy**2)
};

/* Using axial symmetry, get the index of pts with different distances to symmetry axis (z) */
int get_pts_tbc(int nx, int ny, int nz, double dx, double dy, Axial_symmetry* ax)
{
    // the array that stores the square (ix-nx/2)^2+(iy-ny/2)^2
    double* xx = Allocate<double>(nx);
    double* yy = Allocate<double>(ny);

    // malloc maximum possible space for buffers in the xy-plane
    Points* points_tbc = Allocate<Points>(nx * ny);
    int* indxy = Allocate<int>(nx * ny); // the array that stores the index of the points with different distances to the symmetry axis z

    for (int ix = 0; ix < nx; ix++) 
        xx[ix] = dx * ((double)ix - (double)nx / 2.0);

    for (int iy = 0; iy < ny; iy++) 
        yy[iy] = dy * ((double)iy - (double)ny / 2.0);

    // Search for points on xy-plane with different distances to the z-axis
    int npxy = 0;

    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            int flag = 0;
            double norm1 = fabs(xx[ix]) + fabs(yy[iy]);
            double norm2 = sqrt(Square(xx[ix]) + Square(yy[iy])); // the square of distance to the symmetry axis

            for (int i = 0; i < npxy; i++)
            {
                if ((fabs(points_tbc[i].norm1 - norm1) < EPS) && (fabs(points_tbc[i].norm2 - norm2) < EPS))
                {
                    flag = 1;
                    break;
                }
            }

            if (flag == 0)  // new distance, add into sq array
            {
                points_tbc[npxy].ix = ix;
                points_tbc[npxy].iy = iy;
                points_tbc[npxy].norm1 = norm1;
                points_tbc[npxy].norm2 = norm2;

                int ixy = ix * ny + iy;
                indxy[npxy] = ixy;

                npxy += 1;
            }
        }
    }

    ax->npts_xy = npxy;  // number of points need to be calculated in xy plane

    // now cast it into 3d system with different iz
    int j = 0;

    for (int i = 0; i < npxy; i++)
        for (int iz = 0; iz < nz; iz++, j++)
        {
            int ixyz = indxy[i] * nz + iz;
            ax->ind_xyz[j] = ixyz;
        }

    ax->npts_xyz = nz * npxy; // number of points need to be calculated in xy plane

    // now relates each pt in 3d with a pt in cylindrical coordinates

    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int iz = 0; iz < nz; iz++)
            {
                int ixyz = ix * ny * nz + iy * nz + iz;

                int flag = 0;
                double norm1 = fabs(xx[ix]) + fabs(yy[iy]);
                double norm2 = sqrt(Square(xx[ix]) + Square(yy[iy]));

                for (int i = 0; i < npxy; i++)
                {
                    if ((fabs(points_tbc[i].norm1 - norm1) < EPS) && (fabs(points_tbc[i].norm2 - norm2) < EPS))
                    {
                        int k = nz * i + iz;

                        ax->car2cyl[ixyz] = ax->ind_xyz[k];
                        flag = 1;
                        break;
                    }
                }

                if (flag == 0)
                {
                    fprintf(stderr, "ERROR: Fail to find pts in cylindrical coordinates");
                    return -1;
                }

            }
        }
    }

    Free(indxy);
    Free(points_tbc);
    Free(xx); 
    Free(yy);

    return 0;
}

/* routine that refill the densities in lattice with axial symmetry */
void axial_symmetry_densities(Densities* dens, const Axial_symmetry* ax, int nxyz, const Lattice_arrays* lattice_coords, FFtransf_vars* fftransf_vars, MPI_Comm comm, int iam)
{
    // refill the density: rho
    for (int i = 0; i < nxyz; i++) 
    {
        dens->rho[i] = dens->rho[ax->car2cyl[i]];
    }

    double* jjx1 = AllocateZeroed<double>(nxyz);
    double* jjy1 = AllocateZeroed<double>(nxyz);
    double* jjz1 = AllocateZeroed<double>(nxyz);

    double* tau1 = AllocateZeroed<double>(nxyz);
    complex* nu1 = AllocateZeroed<complex>(nxyz);

    double* jjx = Allocate<double>(nxyz);
    double* jjy = Allocate<double>(nxyz);
    double* jjz = Allocate<double>(nxyz);

    double* tau = Allocate<double>(nxyz);
    complex* nu = Allocate<complex>(nxyz);

    std::copy(dens->jjx, dens->jjx + dens->nstop - dens->nstart, jjx1 + dens->nstart);
    std::copy(dens->jjy, dens->jjy + dens->nstop - dens->nstart, jjy1 + dens->nstart);
    std::copy(dens->jjz, dens->jjz + dens->nstop - dens->nstart, jjz1 + dens->nstart);
    std::copy(dens->tau, dens->tau + dens->nstop - dens->nstart, tau1 + dens->nstart);
    std::copy(dens->nu, dens->nu + dens->nstop - dens->nstart, nu1 + dens->nstart);

    MPI_Allreduce(jjx1, jjx, nxyz, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(jjy1, jjy, nxyz, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(jjz1, jjz, nxyz, MPI_DOUBLE, MPI_SUM, comm);

    MPI_Allreduce(tau1, tau, nxyz, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(nu1, nu, nxyz, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);

    Free(nu1); 
    Free(tau1);

    // refile tau, nu
    for (int i = 0; i < nxyz; i++)
    {
        tau[i] = tau[ax->car2cyl[i]];
        nu[i] = nu[ax->car2cyl[i]];
    }

    // refill jjx, jjy, jjz:
    const double* xa = lattice_coords->xa;
    const double* ya = lattice_coords->ya;

    double* ff = Allocate<double>(nxyz);
    for (int i = 0; i < nxyz; i++)
    {
        double tmp = sqrt(Square(xa[i]) + Square(ya[i]));

        if (tmp > EPS)
        {
            ff[i] = jjx[i] * xa[i] / tmp + jjy[i] * ya[i] / tmp;
        }
        else
        {
            ff[i] = jjx[i];
        }
    }

    // fill in lattices
    for (int i = 0; i < nxyz; i++)
    {
        ff[i] = ff[ax->car2cyl[i]];
        jjz[i] = jjz[ax->car2cyl[i]];
    }

    // re-expand into cartesian
    for (int i = 0; i < nxyz; i++)
    {
        double tmp = sqrt(Square(xa[i]) + Square(ya[i]));

        if (tmp > EPS)
        {
            jjx[i] = ff[i] * xa[i] / tmp;
            jjy[i] = ff[i] * ya[i] / tmp;
        }
        else
        {
            jjx[i] = ff[i];
            jjy[i] = 0;
        }
    }

    int iwork = dens->nstop - dens->nstart;
    int nstart = dens->nstart;
    int nstop = dens->nstop;

    for (int i = 0; i < iwork; i++)
    {
        dens->tau[i] = tau[i + dens->nstart];
        dens->nu[i] = nu[i + dens->nstart];
        dens->jjx[i] = jjx[i + dens->nstart];
        dens->jjy[i] = jjy[i + dens->nstart];
        dens->jjz[i] = jjz[i + dens->nstart];
    }

    // calculate divjj
    // jjx ***********
    for (int i = 0; i < nxyz; i++)
        fftransf_vars->buff[i] = jjx[i];

    fftw_execute(fftransf_vars->plan_f);
    for (int i = 0; i < nxyz; i++)
        fftransf_vars->buff[i] *= lattice_coords->kx[i] * 1i / (double)nxyz;

    fftw_execute(fftransf_vars->plan_b);
    for (int i = 0; i < iwork; i++)
        dens->divjj[i] = std::real(fftransf_vars->buff[i + nstart]);

    // jjy *************
    for (int i = 0; i < nxyz; i++)
        fftransf_vars->buff[i] = jjy[i];

    fftw_execute(fftransf_vars->plan_f);
    for (int i = 0; i < nxyz; i++)
        fftransf_vars->buff[i] *= lattice_coords->ky[i] * 1i / (double)nxyz;

    fftw_execute(fftransf_vars->plan_b);
    for (int i = 0; i < iwork; i++)
        dens->divjj[i] += std::real(fftransf_vars->buff[i + nstart]);

    // jjz **************
    for (int i = 0; i < nxyz; i++)
        fftransf_vars->buff[i] = jjz[i];

    fftw_execute(fftransf_vars->plan_f);
    for (int i = 0; i < nxyz; i++)
        fftransf_vars->buff[i] *= lattice_coords->kz[i] * 1i / (double)nxyz;

    fftw_execute(fftransf_vars->plan_b);
    for (int i = 0; i < iwork; i++)
        dens->divjj[i] += std::real(fftransf_vars->buff[i + nstart]);

    Free(jjx); 
    Free(jjy); 
    Free(jjz);
    Free(jjx1); 
    Free(jjy1); 
    Free(jjz1);
    Free(nu); 
    Free(tau);
}

