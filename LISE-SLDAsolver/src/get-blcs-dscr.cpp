// for license information, see the accompanying LICENSE file
/* kenneth.roche@pnl.gov */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>

#include "common.h"

void get_blcs_dscr(MPI_Comm commc, int m, int n, int mb, int nb, int p, int q, int* ip, int* iq, int* blcs_dscr, int* nip, int* niq)
{
    int i, MONE = -1, ZERO = 0;

    /* mpi process ids and sizeof(MPI_COMM_WORLD) */
    int iam, np;
    MPI_Comm_size(commc, &np);
    MPI_Comm_rank(commc, &iam);

    if (np != p * q)
    {
        if (iam == 0)
        {
            printf("[%d]error in grid parameters: np != p.q\n", iam); /* recall that np_ is the number of processes in commw */
        }
    }

    if (p * q == 1) /* small problem case :: calls LAPACK directly */
    {
        if (iam == 0)
        {
            printf("[%d]Single processor case\n\tcall to zgeev_() here\n...exiting\n", iam);
        }
    }

    for (i = 0; i < 9; i++)
    {
        blcs_dscr[i] = -1;
    }

    /* initialize the BLACS grid - a virtual rectangular grid */
    int iam_blacs, nprocs_blacs;
    Cblacs_pinfo(&iam_blacs, &nprocs_blacs);

    if (nprocs_blacs < 1)
    {
        Cblacs_setup(&iam_blacs, &nprocs_blacs);
    }

    int ictxt = Csys2blacs_handle(commc); /* translate MPI communicator to BLACS integer value */
    Cblacs_gridinit(&ictxt, "R", p, q); /* 'Row-Major' */
    Cblacs_gridinfo(ictxt, &p, &q, ip, iq); /* get (ip,iq) , the process (row,column) id */

    get_mem_req_blk_cyc(*ip, *iq, p, q, m, n, mb, nb, nip, niq);
    blcs_dscr[0] = 1;
    blcs_dscr[1] = ictxt;
    blcs_dscr[2] = m;
    blcs_dscr[3] = n;
    blcs_dscr[4] = mb;
    blcs_dscr[5] = nb;
    blcs_dscr[6] = 0; /* C vs F conventions */
    blcs_dscr[7] = 0; /* C vs F conventions */
    blcs_dscr[8] = *nip;

    MPI_Barrier(commc);
}
