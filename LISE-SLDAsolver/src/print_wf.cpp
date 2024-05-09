// for license information, see the accompanying LICENSE file

/* used to print the wave functions; will be replaced in the near future */
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include "common.h"

int print_wf(const char* fn, MPI_Comm comm, const double* lam, const complex* z, int ip, int nwf, int m_ip, int n_iq, int i_p, int i_q, int mb, int nb, int p_proc, int q_proc, int nx, int ny, int nz, double e_cut, double* occ, int icub)
{
    int nxyz = nx * ny * nz;
    int n = 4 * nxyz;
    int nhalf = n / 2;
    int ntot = nhalf + nwf;
    MPI_Barrier(comm);

    complex* vec = Allocate<complex>(n);
    complex* vec1 = Allocate<complex>(n);

    int iflag = EXIT_SUCCESS;

    FILE* fd = nullptr;
    if (ip == 0)
    {
        if (unlink(fn) != 0) 
        {
            fprintf(stderr, "Cannot unlink() FILE %s\n", fn);
        }

        fd = fopen(fn, "wb");
        if (fd == nullptr)
        {
            fprintf(stderr, "error: cannot open FILE %s for WRITE\n", fn);
            iflag = EXIT_FAILURE;
        }
    }

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);

    if (iflag == EXIT_FAILURE)
        return EXIT_FAILURE;

    for (int jj = nhalf; jj < ntot; jj++)
    {
        /* construct one vector at a time for positive eigenvalues */
        double f1 = sqrt(factor_ec(lam[jj], e_cut, icub));
        //      printf( "posix: gj[ %d ] f1[ %f ]\n" , jj , f1 ) ;

        ZeroMemory(vec1, n);

        for (int lj = 0; lj < n_iq; lj++)
        {
            int j = i_q * nb + (int)(floor((double)lj / nb)) * q_proc * nb + lj % nb;

            if (j == jj)
            {
                for (int li = 0; li < m_ip; li++)
                {
                    vec1[i_p * mb + (int)(floor((double)li / mb)) * p_proc * mb + li % mb] = z[lj * m_ip + li] * f1;
                }
            }
        }

        MPI_Reduce(vec1, vec, n, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comm);

        if (ip == 0)
        {
            size_t bytes_written = fwrite(vec, sizeof(complex), n, fd);
            size_t bytestowrite = n * sizeof(complex);

            if (bytestowrite != bytes_written)
            {
                fprintf(stderr, "err in print_wf: failed to WRITE %ld bytes\n", bytestowrite);
                iflag = EXIT_FAILURE;
                fclose(fd);
            }
        }

        MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);

        if (iflag == EXIT_FAILURE)
            return(EXIT_FAILURE);
    }

    if (ip == 0)
    {
        fclose(fd);
        printf("%d wave functions successfully written in %s\n", nwf, fn);
    }

    Free(vec); 
    Free(vec1);

    return EXIT_SUCCESS;
}

int print_wf2(const char* fn, MPI_Comm comm, const double* lam, const complex* z, int ip, int nwf, int m_ip, int n_iq, int i_p, int i_q, int mb, int nb, int p_proc, int q_proc, int nx, int ny, int nz, double e_cut, double* occ, int icub)
{
    int nxyz = nx * ny * nz;
    int n = 4 * nxyz;
    int nhalf = n / 2;

    complex* vec = Allocate<complex>(n);
    complex* vec1 = Allocate<complex>(n);

    int iflag = EXIT_SUCCESS;

    FILE* fd = nullptr;
    if (ip == 0)
    {
        if (unlink(fn) != 0)
            fprintf(stderr, "Cannot unlink() FILE %s\n", fn);

        fd = fopen(fn, "wb");
        if (fd == nullptr)
        {
            fprintf(stderr, "error: cannot open FILE %s for WRITE\n", fn);
            iflag = EXIT_FAILURE;
        }
    }

    MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);
    if (iflag == EXIT_FAILURE)
        return EXIT_FAILURE;

    int nwf1 = 0;

    for (int jj = nhalf; jj < n; jj++)
    {
        /* construct one vector at a time for positive eigenvalues , time reversal states constructed by hand */
        if (occ[jj - nhalf] < .9)
            continue;

        nwf1++;

        double f1 = sqrt(factor_ec(lam[jj], e_cut, icub));

        ZeroMemory(vec1, n);

        for (int lj = 0; lj < n_iq; lj++)
        {
            int j = i_q * nb + (int)(floor((double)lj / nb)) * q_proc * nb + lj % nb;

            if (j == jj)
            {
                for (int li = 0; li < m_ip; li++)
                {
                    vec1[i_p * mb + (int)(floor((double)li / mb)) * p_proc * mb + li % mb] = z[lj * m_ip + li] * f1;
                }
            }
        }

        MPI_Reduce(vec1, vec, n, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comm);

        if (ip == 0)
        {
            size_t bytes_written = fwrite(vec, sizeof(complex), n, fd);
            size_t bytestowrite = n * sizeof(complex);

            if (bytes_written != bytestowrite)
            {
                fprintf(stderr, "err in print_wf2: failed to WRITE %ld bytes\n", bytestowrite);
                iflag = EXIT_FAILURE;
                fclose(fd);
            }
        }

        MPI_Bcast(&iflag, 1, MPI_INT, 0, comm);

        if (iflag == EXIT_FAILURE)
            return EXIT_FAILURE;
    }

    if (ip == 0)
    {
        fclose(fd);
        printf("%d wave functions successfully written in %s\n", nwf1, fn);
    }

    Free(vec); 
    Free(vec1);

    return EXIT_SUCCESS;
}
