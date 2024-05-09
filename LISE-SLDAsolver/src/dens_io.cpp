// for license information, see the accompanying LICENSE file
/*
   Used to save and read back densities
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include "common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mpi.h>
#include <string.h>
#include <fcntl.h>

void interpolate_lattice(
    const double* x, const double* y, const double* z, 
    int nx, int ny, int nz, 
    const double* x1, const double* y1, const double* z1, 
    int nx1, double ny1, double nz1, 
    double* dens, const double* dens1
) 
{
    // Trlinear interpolation: http://en.wikipedia.org/wiki/Trilinear_interpolation
    int nxyz = nx * ny * nz;
    /*
    int i, i1;
    int n;
    */
    for (int i = 0; i < nxyz; i++) 
    {
        int ix, iy, iz;
        i2xyz(i, &ix, &iy, &iz, ny, nz);
        int ix1 = -1;

        for (int n = 1; n < nx1; n++) 
        {
            if (x[ix] <= x1[n]) 
            {
                ix1 = n;
                break;
            }
        }

        int ix10, iy10, iz10;
        double xx1, yy1, zz1;
        if (ix1 < 0) 
        {
            ix1 = 0;
            ix10 = nx1 - 1;
            xx1 = fabs(x1[0]);
        }
        else 
        {
            ix10 = ix1 - 1;
            xx1 = x1[ix1];
        }

        int iy1 = -1;
        for (int n = 1; n < ny1; n++) 
        {
            if (y[iy] <= y1[n]) 
            {
                iy1 = n;
                break;
            }
        }

        if (iy1 < 0) 
        {
            iy1 = 0;
            iy10 = ny1 - 1;
            yy1 = fabs(y1[0]);
        }
        else 
        {
            iy10 = iy1 - 1;
            yy1 = y1[iy1];
        }

        int iz1 = -1;
        for (int n = 1; n < nz1; n++) 
        {
            if (z[iz] <= z1[n]) 
            {
                iz1 = n;
                break;
            }
        }

        if (iz1 < 0) 
        {
            iz1 = 0;
            iz10 = nz1 - 1;
            zz1 = fabs(z1[0]);
        }
        else 
        {
            iz10 = iz1 - 1;
            zz1 = z1[iz1];
        }

        //determined now where I am 
        int i000 = iz10 + nz1 * (iy10 + ny1 * ix10);
        int i100 = iz10 + nz1 * (iy10 + ny1 * ix1);
        int i010 = iz10 + nz1 * (iy1 + ny1 * ix10);
        int i001 = iz1 + nz1 * (iy10 + ny1 * ix10);
        int i110 = iz10 + nz1 * (iy1 + ny1 * ix1);
        int i101 = iz1 + nz1 * (iy10 + ny1 * ix1);
        int i011 = iz1 + nz1 * (iy1 + ny1 * ix10);
        int i111 = iz1 + nz1 * (iy1 + ny1 * ix1);

        double xd = (x[ix] - x1[ix10]) / (xx1 - x1[ix10]);
        double yd = (y[iy] - y1[iy10]) / (yy1 - y1[iy10]);
        double zd = (z[iz] - z1[iz10]) / (zz1 - z1[iz10]);

        double c00 = dens1[i000] * (1.0 - xd) + dens1[i100] * xd;
        double c10 = dens1[i010] * (1.0 - xd) + dens1[i110] * xd;
        double c01 = dens1[i001] * (1.0 - xd) + dens1[i101] * xd;
        double c11 = dens1[i011] * (1.0 - xd) + dens1[i111] * xd;

        double c0 = c00 * (1.0 - yd) + c10 * yd;
        double c1 = c01 * (1.0 - yd) + c11 * yd;

        dens[i] = c0 * (1.0 - zd) + c1 * zd;
    }
}

void interpolate_lattice_complex(
    const double* x, const double* y, const double* z, 
    int nx, int ny, int nz, 
    const double* x1, const double* y1, const double* z1, 
    int nx1, double ny1, double nz1, 
    complex* dens, const complex* dens1
) 
{
    // Trlinear interpolation: http://en.wikipedia.org/wiki/Trilinear_interpolation
    int i, i1;
    int nxyz = nx * ny * nz;
    int ix, iy, iz;
    int ix1, iy1, iz1;
    int n;
    int ix10, iy10, iz10;
    double xx1, yy1, zz1;

    for (i = 0;i < nxyz;i++) 
    {
        i2xyz(i, &ix, &iy, &iz, ny, nz);
        ix1 = -1;

        for (n = 1;n < nx1;n++) 
        {
            if (x[ix] <= x1[n]) 
            {
                ix1 = n;
                break;
            }
        }

        if (ix1 < 0)
        {
            ix1 = 0;
            ix10 = nx1 - 1;
            xx1 = fabs(x1[0]);
        }
        else 
        {
            ix10 = ix1 - 1;
            xx1 = x1[ix1];
        }

        iy1 = -1;
        for (n = 1;n < ny1;n++) 
        {
            if (y[iy] <= y1[n]) 
            {
                iy1 = n;
                break;
            }
        }

        if (iy1 < 0) 
        {
            iy1 = 0;
            iy10 = ny1 - 1;
            yy1 = fabs(y1[0]);
        }
        else 
        {
            iy10 = iy1 - 1;
            yy1 = y1[iy1];
        }

        iz1 = -1;
        for (n = 1;n < nz1;n++)
        {
            if (z[iz] <= z1[n])
            {
                iz1 = n;
                break;
            }
        }

        if (iz1 < 0) 
        {
            iz1 = 0;
            iz10 = nz1 - 1;
            zz1 = fabs(z1[0]);
        }
        else 
        {
            iz10 = iz1 - 1;
            zz1 = z1[iz1];
        }

        //determined now where I am 
        int i000 = iz10 + nz1 * (iy10 + ny1 * ix10);
        int i100 = iz10 + nz1 * (iy10 + ny1 * ix1);
        int i010 = iz10 + nz1 * (iy1 + ny1 * ix10);
        int i001 = iz1 + nz1 * (iy10 + ny1 * ix10);
        int i110 = iz10 + nz1 * (iy1 + ny1 * ix1);
        int i101 = iz1 + nz1 * (iy10 + ny1 * ix1);
        int i011 = iz1 + nz1 * (iy1 + ny1 * ix10);
        int i111 = iz1 + nz1 * (iy1 + ny1 * ix1);

        double xd = (x[ix] - x1[ix10]) / (xx1 - x1[ix10]);
        double yd = (y[iy] - y1[iy10]) / (yy1 - y1[iy10]);
        double zd = (z[iz] - z1[iz10]) / (zz1 - z1[iz10]);

        complex c00 = dens1[i000] * (1.0 - xd) + dens1[i100] * xd;
        complex c10 = dens1[i010] * (1.0 - xd) + dens1[i110] * xd;
        complex c01 = dens1[i001] * (1.0 - xd) + dens1[i101] * xd;
        complex c11 = dens1[i011] * (1.0 - xd) + dens1[i111] * xd;

        complex c0 = c00 * (1.0 - yd) + c10 * yd;
        complex c1 = c01 * (1.0 - yd) + c11 * yd;

        dens[i] = c0 * (1.0 - zd) + c1 * zd;
    }
}

int write_dens(
    const char* fn, 
    const Densities* dens, 
    MPI_Comm comm, 
    int iam, 
    int nx, int ny, int nz, 
    const double* amu, 
    double dx, double dy, double dz
)
{
    FILE* fd = nullptr;
    int iflag = EXIT_SUCCESS;

    if (iam == 0)
    {
        //if ( ( i = unlink( ( const char * ) fn ) ) != 0 ) fprintf( stderr , "Cannot unlink() FILE %s\n" , fn ) ;
        fd = fopen(fn, "wb");
        if (fd == nullptr)
        {
            fprintf(stderr, "error: cannot open FILE %s for WRITE\n", fn);
            iflag = EXIT_FAILURE;
        }
    }

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);
    MPI_Barrier(comm);

    if (iflag == EXIT_FAILURE)
        return iflag;

    int nxyz = nx * ny * nz;
    double* tau1 = AllocateZeroed<double>(nxyz);
    double* divjj1 = AllocateZeroed<double>(nxyz);
    complex* nu1 = AllocateZeroed<complex>(nxyz);

    double* tau = Allocate<double>(nxyz);
    double* divjj = Allocate<double>(nxyz);
    complex* nu = Allocate<complex>(nxyz);
    
    std::copy(dens->tau, dens->tau + dens->nstop - dens->nstart, tau1 + dens->nstart);
    std::copy(dens->divjj, dens->divjj + dens->nstop - dens->nstart, divjj1 + dens->nstart);
    std::copy(dens->nu, dens->nu + dens->nstop - dens->nstart, nu1 + dens->nstart);

    MPI_Reduce(tau1, tau, nxyz, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(divjj1, divjj, nxyz, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(nu1, nu, nxyz, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comm);

    Free(nu1); 
    Free(tau1); 
    Free(divjj1);

    if (iam == 0)
    {
        int nx_ny_nz[3] { nx, ny, nz };
        double dx_dy_dz[3] { dx, dy, dz };

        size_t bytes_written = 0;
        bytes_written += fwrite(nx_ny_nz, sizeof(nx_ny_nz), 1, fd);
        bytes_written += fwrite(dx_dy_dz, sizeof(dx_dy_dz), 1, fd);
        bytes_written += fwrite(dens->rho, sizeof(dens->rho[0]), nxyz, fd);
        bytes_written += fwrite(dens->tau, sizeof(dens->tau[0]), nxyz, fd);
        bytes_written += fwrite(dens->divjj, sizeof(dens->divjj[0]), nxyz, fd);
        bytes_written += fwrite(nu, sizeof(nu[0]), nxyz, fd);
        bytes_written += fwrite(amu, sizeof(amu[0]), 5, fd);

        size_t bytestowrite = 
            sizeof(nx_ny_nz) + sizeof(dx_dy_dz) + 
            (sizeof(dens->rho[0]) + sizeof(dens->tau[0]) + sizeof(dens->divjj[0]) + sizeof(nu)) * nxyz +
            sizeof(amu[0]) * 5;

        if (bytes_written != bytestowrite)
        {
            fprintf(stderr, "err: failed to WRITE %ld bytes\n", bytestowrite);
            iflag = EXIT_FAILURE;
        }
        
        fclose(fd);
    }

    Free(tau);
    Free(nu); 
    Free(divjj);

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);

    return iflag;
}

int read_constr(const char* fn, int nxyz, double* cc_lambda, int iam, MPI_Comm comm)
{
    FILE* fd = nullptr;

    int iflag = EXIT_SUCCESS;

    if (iam == 0)
    {
        /* Check to see if the file already exists, if so exit */
        fd = fopen(fn, "rb");
        if (fd == nullptr)
        {
            fprintf(stderr, "error: cannot open FILE %s for READ\n", fn);
            iflag = EXIT_FAILURE;
        }

    }

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);

    if (iflag == EXIT_FAILURE)
        return iflag;

    if (iam == 0)
    {
        size_t bytes_read = fread(cc_lambda, sizeof(cc_lambda[0]), 3, fd);
        size_t bytestoread = 3 * sizeof(double);

        if (bytes_read != bytestoread)
        {
            fprintf(stderr, "err: failed to READ %ld bytes\n", bytestoread);
            iflag = EXIT_FAILURE;
        }

        fclose(fd);
    }

    MPI_Bcast(cc_lambda, 3, MPI_DOUBLE, 0, comm);
    return iflag;
}

int read_dens(const char* fn, Densities* dens, MPI_Comm comm, int iam, int nx, int ny, int nz, double* amu, double dx, double dy, double dz, const char* filename)
{
    FILE* fd = nullptr;
    int iflag = EXIT_SUCCESS;

    if (iam == 0)
    {
        /* Check to see if the file already exists, if so exit */
        fd = fopen(fn, "rb");
        if (fd == nullptr)
        {
            fprintf(stderr, "error: cannot open FILE %s for READ\n", fn);
            iflag = EXIT_FAILURE;
        }
    }

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);
    if (iflag == EXIT_FAILURE)
        return iflag;

    int nxyz = nx * ny * nz;
    double* tau = AllocateZeroed<double>(nxyz);
    double* divjj = AllocateZeroed<double>(nxyz);
    complex* nu = AllocateZeroed<complex>(nxyz);

    ZeroMemory(dens->rho, nxyz);

    if (iam == 0)
    {
        size_t bytesread = 0;
        size_t bytestoread = 0;

        int nx1, ny1, nz1;
        bytesread += fread(&nx1, sizeof(nx1), 1, fd);
        bytesread += fread(&ny1, sizeof(ny1), 1, fd);
        bytesread += fread(&nz1, sizeof(nz1), 1, fd);

        printf("nx1=%d ny1=%d nz1=%d\n", nx1, ny1, nz1);

        int dx_dy_dz[3];
        bytesread += fread(&dx_dy_dz, sizeof(dx_dy_dz), 1, fd);

        if (nx != nx1 || ny != ny1 || nz != nz1)
        {
            int nxyz1 = nx1 * ny1 * nz1;
            double* buff = Allocate<double>(nxyz1);

            bytesread += fread(buff, sizeof(double), nxyz1, fd);
            if (nx >= nx1 && ny >= ny1 && nz >= nz1)
            {
                copy_lattice_arrays(buff, dens->rho, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
            }
            else if (nx <= nx1 && ny <= ny1 && nz <= nz1)
            {
                copy_lattice_arrays_l2s(buff, dens->rho, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
            }
            else 
            {
                fprintf(stderr, "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n");
                iflag = EXIT_FAILURE;
                fclose(fd);
            }

            bytesread += fread(buff, sizeof(double), nxyz1, fd);
            if (nx >= nx1 && ny >= ny1 && nz >= nz1)
            {
                copy_lattice_arrays(buff, tau, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
            }
            else if (nx <= nx1 && ny <= ny1 && nz <= nz1)
            {
                copy_lattice_arrays_l2s(buff, tau, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
            }
            else 
            {
                fprintf(stderr, "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n");
                iflag = EXIT_FAILURE;
                fclose(fd);
            }
            
            bytesread += fread(buff, sizeof(double), nxyz1, fd);
            if (nx >= nx1 && ny >= ny1 && nz >= nz1)
            {
                copy_lattice_arrays(buff, divjj, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
            }
            else if (nx <= nx1 && ny <= ny1 && nz <= nz1)
            {
                copy_lattice_arrays_l2s(buff, divjj, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
            }
            else 
            {
                fprintf(stderr, "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n");
                iflag = EXIT_FAILURE;
                fclose(fd);
            }
            Free(buff);

            complex* buffc = Allocate<complex>(nxyz1);
            bytesread += fread(buffc, sizeof(complex), nxyz1, fd);
            if (nx >= nx1 && ny >= ny1 && nz >= nz1)
            {
                copy_lattice_arrays(buffc, nu, sizeof(complex), nx1, ny1, nz1, nx, ny, nz);
            }
            else if (nx <= nx1 && ny <= ny1 && nz <= nz1)
            {
                copy_lattice_arrays_l2s(buffc, nu, sizeof(complex), nx1, ny1, nz1, nx, ny, nz);
            }
            else 
            {
                fprintf(stderr, "err: nx1 ny1 nz1 should be either larger or smaller than nx ny nz uniformly\n");
                iflag = EXIT_FAILURE;
                fclose(fd);
            }

            free(buffc);

            bytestoread += (3 * sizeof(double) + sizeof(complex)) * nxyz1;
        }
        else
        {
            bytesread += fread(dens->rho, sizeof(dens->rho[0]), nxyz, fd);
            bytesread += fread(dens->tau, sizeof(dens->tau[0]), nxyz, fd);
            bytesread += fread(dens->divjj, sizeof(dens->divjj[0]), nxyz, fd);
            bytesread += fread(dens->nu, sizeof(dens->nu[0]), nxyz, fd);
            
            bytestoread += (3 * sizeof(double) + sizeof(complex)) * nxyz;
        }

        bytesread += fread(amu, sizeof(double), 5, fd);

        if (bytesread != bytestoread)
        {
            fprintf(stderr, "err: failed to READ %ld bytes\n", bytestoread);
            iflag = EXIT_FAILURE;
        }

        bytestoread += 6 * sizeof(int) +

        fclose(fd);
    }

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);
    if (iflag == EXIT_FAILURE)
    {
        Free(tau); 
        Free(nu);

        return EXIT_FAILURE;
    }

    MPI_Bcast(dens->rho, nxyz, MPI_DOUBLE, 0, comm);
    MPI_Bcast(tau, nxyz, MPI_DOUBLE, 0, comm);
    MPI_Bcast(divjj, nxyz, MPI_DOUBLE, 0, comm);
    MPI_Bcast(nu, nxyz, MPI_DOUBLE_COMPLEX, 0, comm);
    MPI_Bcast(amu, 5, MPI_DOUBLE, 0, comm);


    for (int i = dens->nstart; i < dens->nstop; i++)
    {
        dens->tau[i - dens->nstart] = tau[i];
        dens->divjj[i - dens->nstart] = divjj[i];
        dens->nu[i - dens->nstart] = nu[i];
    }

    Free(tau); 
    Free(nu); 
    Free(divjj);

    return EXIT_SUCCESS;
}

int copy_lattice_arrays(void* bf1, void* bf, size_t siz, int nx1, int ny1, int nz1, int nx, int ny, int nz)
{
    int nx_start = (nx - nx1) / 2;
    int nx_stop  = (nx + nx1) / 2;
    int ny_start = (ny - ny1) / 2;
    int ny_stop  = (ny + ny1) / 2;
    int nz_start = (nz - nz1) / 2;
    int nz_stop  = (nz + nz1) / 2;

    for (int ix = 0; ix < nx1; ix++)
    {
        for (int iy = 0; iy < ny1; iy++)
        {
            for (int iz = 0; iz < nz1; iz++)
            {
                int ixyz = iz + nz_start + nz * (iy + ny_start + ny * (ix + nx_start));
                int ixyz1 = iz + nz1 * (iy + ny1 * ix);

                memcpy(static_cast<char*>(bf) + ixyz * siz, static_cast<char*>(bf1) + ixyz1 * siz, siz);
            }
        }
    }

    return EXIT_SUCCESS;
}

// used for problems that cut densities in larger lattice into smaller
int copy_lattice_arrays_l2s(void* bf1, void* bf, size_t siz, int nx1, int ny1, int nz1, int nx, int ny, int nz)
{
    int nx_start = (nx1 - nx) / 2;
    int nx_stop = (nx1 + nx) / 2;
    int ny_start = (ny1 - ny) / 2;
    int ny_stop = (ny1 + ny) / 2;
    int nz_start = (nz1 - nz) / 2;
    int nz_stop = (nz1 + nz) / 2;

    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int iz = 0; iz < nz; iz++)
            {
                int ixyz1 = iz + nz_start + nz1 * (iy + ny_start + ny1 * (ix + nx_start));
                int ixyz = iz + nz * (iy + ny * ix);

                memcpy(static_cast<char*>(bf) + ixyz * siz, static_cast<char*>(bf1) + ixyz1 * siz, siz);
            }
        }
    }

    return EXIT_SUCCESS;
}

int write_dens_txt(FILE* fd, const Densities* dens, MPI_Comm comm, int iam, int nx, int ny, int nz, const double* amu)
{
    int iflag = EXIT_SUCCESS;

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);
    MPI_Barrier(comm);

    if (iflag == EXIT_FAILURE)
        return iflag;

    int nxyz = nx * ny * nz;

    double* tau1 = AllocateZeroed<double>(nxyz);
    double* divjj1 = AllocateZeroed<double>(nxyz);
    complex* nu1 = AllocateZeroed<complex>(nxyz);

    std::copy(dens->tau, dens->tau + dens->nstop - dens->nstart, tau1 + dens->nstart);
    std::copy(dens->divjj, dens->divjj + dens->nstop - dens->nstart, divjj1 + dens->nstart);
    std::copy(dens->nu, dens->nu + dens->nstop - dens->nstart, nu1 + dens->nstart);

    double* tau = Allocate<double>(nxyz);
    double* divjj = Allocate<double>(nxyz);
    complex* nu = Allocate<complex>(nxyz);

    MPI_Reduce(tau1, tau, nxyz, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(divjj1, divjj, nxyz, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(nu1, nu, nxyz, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comm);

    Free(nu1); 
    Free(tau1); 
    Free(divjj1);
    Free(tau);
    Free(nu);
    Free(divjj);

    if (iam == 0)
    {
        for (int i = 0; i < nxyz; i++)
        {
            fprintf(fd, "rho[%d] = %.12le\n", i, dens->rho[i]);
        }

        for (int i = 0; i < nxyz; i++)
        {
            fprintf(fd, "nu[%d] = %.12le %12leI\n", i, std::real(nu[i]), std::imag(nu[i]));
        }

        for (int i = 0; i < nxyz; i++)
        {
            fprintf(fd, "tau[%d] = %.12le\n", i, tau[i]);
        }

        for (int i = 0; i < nxyz; i++)
        {
            fprintf(fd, "divjj[%d] = %.12le\n", i, divjj[i]);
        }
    }

    return iflag;

}

int write_qpe(const char* fn, double* lam, MPI_Comm comm, int iam, int nwf)
{
    FILE* fd = nullptr;
    int iflag = EXIT_SUCCESS;
    if (iam == 0)
    {   
        fd = fopen(fn, "wb");
        if (fd == nullptr)
        {
            fprintf(stderr, "error: cannot open FILE %s for WRITE\n", fn);
            iflag = EXIT_FAILURE;
        }
    }

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);
    MPI_Barrier(comm);

    if (iflag == EXIT_FAILURE)
        return iflag;

    if (iam == 0)
    {
        size_t bytes_written = fread(lam, sizeof(double), nwf, fd);
        size_t bytestowrite = nwf * sizeof(double);

        if (bytes_written != bytestowrite)
        {
            fprintf(stderr, "err: failed to WRITE %ld bytes\n", bytestowrite);
            iflag = EXIT_FAILURE;
        }

        fclose(fd);
    }

    return iflag;
}
