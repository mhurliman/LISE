// for license information, see the accompanying LICENSE file

/*
   Used to save and read back potentials
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mpi.h>
#include <string.h>

#include "common.h"

int write_pots(const char* fn, const double* pot_arrays, int nx, int ny, int nz, double dx, double dy, double dz, int ishift)
{
    if (unlink(fn) != 0)
    {
        fprintf(stderr, "Cannot unlink() FILE %s\n", fn);
    }

    FILE* fd = fopen(fn, "wb");
    if (fd == nullptr)
    {
        fprintf(stderr, "error: cannot open FILE %s for WRITE\n", fn);
        return EXIT_FAILURE;
    }

    long int bytestowrite = sizeof(int);

    int nxyz = nx * ny * nz;
    int nx_ny_nz[3] { nx, ny, nz };
    int dx_dy_dz[3] { dx, dy, dz };

    size_t bytes_written = 0;
    bytes_written += fwrite(nx_ny_nz, sizeof(nx_ny_nz), 1, fd);
    bytes_written += fwrite(dx_dy_dz, sizeof(dx_dy_dz), 1, fd);
    bytes_written += fwrite(pot_arrays + ishift, sizeof(double), 7 * nxyz + 4, fd);

    size_t bytes_to_write = sizeof(int) * 6 + sizeof(double) * (7 * nxyz + 4);

    if (bytes_written != bytes_to_write)
    {
        fprintf(stderr, "err: failed to WRITE %ld bytes\n", bytestowrite);
        fclose(fd);
        return EXIT_FAILURE;
    }

    fclose(fd);
    return EXIT_SUCCESS;
}

int read_pots(const char* fn, double* pot_arrays, int nx, int ny, int nz, double dx, double dy, double dz, int ishift)
{
    /* Check to see if the file already exists, if so exit */
    FILE* fd = fopen(fn, "rb");
    if (fd == nullptr)
    {
        fprintf(stderr, "error: cannot open FILE %s for READ\n", fn);
        return EXIT_FAILURE;
    }
    
    size_t bytes_read = 0;
    size_t bytestoread = 0;

    int nx_ny_nz[3];
    bytes_read += fread(nx_ny_nz, sizeof(nx_ny_nz), 1, fd);

    int nx1 = nx_ny_nz[0];
    int ny1 = nx_ny_nz[1];
    int nz1 = nx_ny_nz[2];

    printf("potentials: nx1=%d ny1=%d nz1=%d\n", nx1, ny1, nz1);

    double dx_dy_dz[3];
    bytes_read += fread(dx_dy_dz, sizeof(dx_dy_dz), 1, fd);

    double dx1 = dx_dy_dz[0];
    double dy1 = dx_dy_dz[1];
    double dz1 = dx_dy_dz[2];

    if (dx1 != dx || dy1 != dy || dz1 != dz)
    {
        printf("error: the lattice constants do not coincide \n new: %6.4f %6.4f %6.4f \n old: %6.4f %6.4f %6.4f \n", dx, dy, dz, dx1, dy1, dz1);
        fclose(fd);
        return EXIT_FAILURE;
    }

    int nxyz = nx * ny * nz;

    if (nx != nx1 || ny != ny1 || nz != nz1)
    {
        int nxyz1 = nx1 * ny1 * nz1;

        double* tmp = Allocate<double>(7 * nxyz1 + 4);
        bytes_read += fread(tmp, sizeof(double), 7 * nxyz1 + 4, fd);

        /* Coulomb should be added here */
        copy_lattice_arrays(tmp, pot_arrays + ishift, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
        copy_lattice_arrays(tmp + nxyz1, pot_arrays + ishift + nxyz, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
        copy_lattice_arrays(tmp + 2 * nxyz1, pot_arrays + ishift + 2 * nxyz, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
        copy_lattice_arrays(tmp + 3 * nxyz1, pot_arrays + ishift + 3 * nxyz, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
        copy_lattice_arrays(tmp + 4 * nxyz1, pot_arrays + ishift + 4 * nxyz, sizeof(double), nx1, ny1, nz1, nx, ny, nz);
        copy_lattice_arrays(tmp + 5 * nxyz1, pot_arrays + ishift + 5 * nxyz, sizeof(complex), nx1, ny1, nz1, nx, ny, nz);

        pot_arrays[7 * nxyz + ishift] = tmp[7 * nxyz];
        pot_arrays[7 * nxyz1 + ishift + 1] = tmp[7 * nxyz + 1];
        pot_arrays[7 * nxyz1 + ishift + 2] = tmp[7 * nxyz + 2];
        pot_arrays[7 * nxyz1 + ishift + 3] = tmp[7 * nxyz + 3];

        Free(tmp);

        bytestoread += (7 * nxyz1 + 4) * sizeof(double);
    }
    else
    {
        bytes_read += fread(pot_arrays + ishift, sizeof(double), 7 * nxyz + 4, fd);
        bytestoread += (7 * nxyz + 4) * sizeof(double);
    }

    bytestoread += 3 * sizeof(int) + 3 * sizeof(double);

    if (bytes_read != bytestoread)
    {
        fprintf(stderr, "err: failed to READ %ld bytes\n", bytestoread);
        fclose(fd);
        return EXIT_FAILURE;
    }

    fclose(fd);
    return EXIT_SUCCESS;
}
