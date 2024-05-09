// for license information, see the accompanying LICENSE file
/*
Translation of the fortran subroutine to create two groups
for protons and neutrons, respectively
*/
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "common.h"

int create_mpi_groups(MPI_Comm commw, MPI_Comm* gr_comm, int np, int* gr_np, int* gr_ip, MPI_Group* group_comm)
{
    MPI_Group world_group;
    MPI_Comm_group(commw, &world_group);

    *gr_np = np / 2;

    int* rankbuf = Allocate<int>(*gr_np);

    for (int i = 0; i < *gr_np; i++)
    {
        rankbuf[i] = i;
    }

    /* form group of processes for protons now */
    MPI_Group group_p;
    MPI_Comm gr_p;
    int ip_p;

    MPI_Group_incl(world_group, *gr_np, rankbuf, &group_p);
    MPI_Comm_create(commw, group_p, &gr_p);
    MPI_Group_rank(group_p, &ip_p);

    /* form the neutron group */
    for (int i = 0; i < *gr_np; i++)
    {
        rankbuf[i] += *gr_np;
    }

    /*  MPI_Group_difference( world_group , group_p , &group_n ) ; */
    MPI_Group group_n;
    MPI_Comm gr_n;
    int ip_n;

    MPI_Group_incl(world_group, *gr_np, rankbuf, &group_n);
    MPI_Comm_create(commw, group_n, &gr_n);
    MPI_Group_rank(group_n, &ip_n);

    int rank;
    MPI_Comm_rank(commw, &rank);

    int isospin;
    if (ip_p != MPI_UNDEFINED)
    {
        isospin = 1;
        *gr_comm = gr_p;
        *gr_ip = ip_p;
        
        if (ip_p != rank)
            printf(" the process will fail, the proton process does not have the expected group ip: %d != %d\n", rank, *gr_ip);

        *group_comm = group_p;
    }

    if (ip_n != MPI_UNDEFINED)
    {
        isospin = -1;
        *gr_comm = gr_n;
        *gr_ip = ip_n;

        if (ip_n + *gr_np != rank)
            printf(" the process will fail, the proton process does not have the expected group ip: %d != %d\n", rank, *gr_ip + *gr_np);

        *group_comm = group_n;
    }

    Free(rankbuf);

    return isospin;
}

void destroy_mpi_groups(MPI_Group* group_comm, MPI_Comm* gr_comm)
{
    MPI_Barrier(*gr_comm);

    MPI_Group_free(group_comm);
    MPI_Comm_free(gr_comm);
}
