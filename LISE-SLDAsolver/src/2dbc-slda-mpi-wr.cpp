// for license information, see the accompanying LICENSE file
//
// kenneth.roche@pnl.gov ; k8r@uw.edu
// initial version ... rochekj@ornl.gov 
// writes 2dbc decomposed matrix to global column vectors in a FILE
// ... using > MPI2 ROMIO semantics for portability
//
//grab a bag of routines ... not all needed here
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <mpi.h>

#include "common.h"

//for allocating arrays in the 2d-block cyclic space 
void get_blk_cyc(int ip, int npdim, int ma, int mblk, int* nele)
{ 
    //virtual process(0,0) owns the first element(s), rows are traversed first in the loop of this map - this is known as a column major ordering 
    int srcproc = 0;

    //virtual process(0,0) owns the first element(s) 
    int my_ele = (npdim + ip - srcproc) % npdim;
    int nele_ = ma / mblk;
    int np = (nele_ / npdim) * mblk;
    int extra = nele_ % npdim;

    if (my_ele < extra) 
        np += mblk;
    else if (my_ele == extra) 
        np += ma % mblk;

    *nele = np;
}

//for determining how many elements / data portion of the symbolic matrix the process at (ip,iq) of the pXq virtual rectangular process grid will manage in memory
void get_mem_req_blk_cyc_new(int ip, int iq, int np, int nq, int ma, int na, int mblk, int nblk, int* nip, int* niq)
{
    if (ip >= 0 && iq >= 0)
    {
        get_blk_cyc(ip, np, ma, mblk, nip); //get rows
        get_blk_cyc(iq, nq, na, nblk, niq); //get columns ...

#ifdef VERBOSE
        printf("ip,iq = %d,%d\t nip, niq = %d %d\n", ip, iq, *nip, *niq);
#endif
    }
    else
    {
        *nip = -1;
        *niq = -1;
    }
}

void bc_wr_mpi(const char* fn, MPI_Comm com, int p, int q, int ip, int iq, int blk, int jstrt, int jstp, int jstrd, int nxyz, complex* z)
{ 
    // i.e. ...  bc_wr_mpi( fnP, MPI_COMM_WORLD, p , q , ip , iq , mb , 0 , na , 1 , na , a );
    /*
    int i, j, k;
    int ixyz, ierr;
    int fd, niogrp;
    int igrp, * nq_grp, * nv_grp, * vgrd, * itmp, * ranks, rcnt, lcnt, indx;
    int irow, icol, gi, gj, li, lj, iqtmp;
    int nip, niq;
    complex* rz, * sz;
    MPI_Comm* comg;
    MPI_Group gw, * g;
    int* iamg, mygrp, gsz;
    MPI_Status mpi_st;
    MPI_File mpifd;
    MPI_Offset loffset;
    int iam, np, iflg;
    long int loff;
    */

    // working communicator and communicator internals 
    int iam, np;
    MPI_Comm_size(com, &np);
    MPI_Comm_rank(com, &iam);

    int code = 0;
    if (jstrt < 0 || jstrt >= nxyz || jstp < 0 || jstp > nxyz)
    { 
        // error 
        if (iam == 0)
            printf("error: invalid index range[b,e,s:%d,%d,%d]\n\t....exiting\n", jstrt, jstp, jstrd);
        
        code = -1;
        MPI_Abort(com, code);
    }

    if (np != p * q)
    { 
        // diagnostic to be safe 
        if (iam == 0)
            printf("error: grid dimensions do not match process count[p,q:%d,%d][np:%d]\n\t....exiting\n", p, q, np);

        code = -1;
        MPI_Abort(com, code);
    }
    
    int niogrp = q; // set the number of writers to be the number of processes in the column dimension of the virtual grid

    // going to get a communicator for each process column in the virtual process grid
    // the size of each group will be the number of process rows in the virtual process grid
    int* nq_grp = Allocate<int>(niogrp);
    if (nq_grp == nullptr)
    {
        code = -1;
        MPI_Abort(com, code);
    }

    int* nv_grp = Allocate<int>(niogrp);
    if (nv_grp == nullptr)
    {
        code = -1;
        MPI_Abort(com, code);
    } 
    
    /* the number of global columns from z in [jstrt,jstp;jstrd] to be written by a particular io group that belong to a particular io group */
    for (int igrp = 0; igrp < niogrp; igrp++)
    {
        nq_grp[igrp] = q / niogrp;

        if (igrp < q % niogrp) 
            nq_grp[igrp]++;
    }

    /* create mpi to blacs id map - will need this to construct communicators */
    int* itmp = nullptr;
    if (iam == 0)
    {
        itmp = Allocate<int>(np);
        if (itmp == nullptr)
        {
            code = -1;
            MPI_Abort(com, code);
        }
    }

    int* vgrd = Allocate<int>(np);
    if (vgrd == nullptr)
    {
        code = -1;
        MPI_Abort(com, code);
    }

    // NOTE - this for testing purposes! single column major index from ip, iq, p, q
    int test = ip + p * iq;
    MPI_Gather(&test, 1, MPI_INT, itmp, 1, MPI_INT, 0, com);

    if (iam == 0)
    {
        for (int i = 0; i < np; i++) 
        {
            vgrd[itmp[i]] = i;
        }
        free(itmp);
    }

    //note this is within the calling communicator
    MPI_Bcast(vgrd, np, MPI_INT, 0, com);
    
    // form unique io group communicators 
    MPI_Comm* comg = Allocate<MPI_Comm>(niogrp);
    if (comg == nullptr)
    {
        code = -1;
        MPI_Abort(com, code);
    }

    MPI_Group* g = Allocate<MPI_Group>(niogrp);
    if (g == nullptr)
    {
        code = -1;
        MPI_Abort(com, code);
    }

    // mpi id in io group communicator
    int* iamg = Allocate<int>(niogrp);
    if (iamg == nullptr)
    {
        code = -1;
        MPI_Abort(com, code);
    }

    // form a mpi group for com
    MPI_Group gw;
    MPI_Comm_group(com, &gw);

    int nq_grp_max = 0; // used here to find max( nq_grp() ) 
    int grp_max = 0; // ... will hold the group index of the io group containing largest number of virtual process columns 

    for (int igrp = 0; igrp < niogrp; igrp++)
    {
        if (nq_grp[igrp] > nq_grp_max)
        {
            nq_grp_max = nq_grp[igrp];
            grp_max = igrp;
        }
    }

    // process ranks from gw that belong to a particular io group 
    int* ranks = Allocate<int>(nq_grp[grp_max] * p);
    if (ranks == nullptr)
    {
        code = -1;
        MPI_Abort(com, code);
    }

    int mygrp;
    int lcnt = 0;
    int rcnt = 0;

    for (int igrp = 0; igrp < niogrp; igrp++)
    {
        nv_grp[igrp] = 0;
        rcnt += nq_grp[igrp];
        int indx = 0;

        for (int icol = lcnt; icol < rcnt; icol++)
        { 
            // substitute loop over iq in virtual grid constrained to the iogrp space 
            if (iq == icol) 
                mygrp = igrp;

            for (int gj = jstrt; gj < jstp; gj += jstrd)
            { 
                    // update the number of vectors to be written from igrp 
                int iqtmp = (int)floor((double)(gj / blk)) % q;

                if (icol == iqtmp) 
                    nv_grp[igrp]++;
            }
            for (int irow = 0; irow < p; irow++)
            { 
                // assign PEs by virtual grid id to list of processes to be included in io group igrp 
                ranks[indx] = vgrd[irow + p * icol];
                indx++;
            }
        }
        lcnt = rcnt;

        // form communicator for specific igrp */
        int gsz;
        MPI_Group_incl(gw, p * nq_grp[igrp], ranks, &g[igrp]);
        MPI_Group_size(g[igrp], &gsz);

        if (gsz != p * nq_grp[igrp])
        {
            code = -1;
            MPI_Abort(com, code);
        }

        MPI_Comm_create(com, g[igrp], &comg[igrp]);
        MPI_Group_rank(g[igrp], &iamg[igrp]);
    }
    Free(vgrd);

    // at this point each of the io groups has been formed using mpi semantics ...
    // now, determine offsets into FILE -each group will be unique 
    long int loff = 0L;
    for (int igrp = 0; igrp < mygrp; igrp++)
    {
        loff += (long)nv_grp[igrp] * nxyz * sizeof(complex); // type should be made general but fine for now
    }

    // int MPI_File_open( MPI_Comm comm, char *filename, int amode, MPI_Info info, MPI_File *mpi_fh );
    MPI_File mpifd;
    MPI_File_open(com, fn, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpifd);

    if (iamg[mygrp] != MPI_UNDEFINED)
    {
        // send / receive buffers - length of column vector of global array -NOTE: this should be bsiz for optimal performance
        complex* sz = Allocate<complex>(nxyz);
        complex* rz = Allocate<complex>(nxyz);

        int indx = 0; // now used to bump offset index into the lustre file 
        int iflg = -1;
        int lcnt = 0;
        int rcnt = 0;

        for (int igrp = 0; igrp < mygrp; igrp++) 
        {
            lcnt += nq_grp[igrp];
        }

        rcnt = lcnt + nq_grp[mygrp];
        int nip, niq;
        get_mem_req_blk_cyc_new(ip, iq, p, q, nxyz, nxyz, blk, blk, &nip, &niq);

        for (int gj = jstrt; gj < jstp; gj += jstrd)
        {
            int iqtmp = (int)floor((double)(gj / blk)) % q;

            for (int icol = lcnt; icol < rcnt; icol++)
            {
                if (icol == iqtmp)
                {
                    iflg = 1;
                }
            }

            if (iflg > 0)
            { 
                // ... the iogrp owns the column index gj 
                for (int i = 0; i < nxyz; i++) 
                {
                    rz[i] = sz[i] = 0;
                }

                if (iq == iqtmp) // ... iq owns gj and a copy needs to occur -all the other PEs contribute 0.,0. to the sum 
                {
                    int lj = (int)floor(floor((double)(gj / blk)) / (double)q) * blk + gj % blk;

                    for (int li = 0; li < nip; li++)
                    { 
                        // form contribution to single global vector 
                        int gi = ip * blk + (int)floor((double)(li / blk)) * p * blk + li % blk;
                        sz[gi] = z[li + lj * nip];
                    }
                }

                MPI_Reduce(sz, rz, nxyz, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comg[mygrp]);

                if (iamg[mygrp] == 0)
                {
                    MPI_Offset loffset = (MPI_Offset)(loff + indx * nxyz * sizeof(complex));
                    MPI_File_write_at(mpifd, loffset, (const void*)rz, 2 * nxyz, MPI_DOUBLE, MPI_STATUS_IGNORE);

                    printf("");
                    indx++;
                }
            }

            iflg = -1;
        }

        Free(sz); 
        Free(rz);
    }

    MPI_File_close(&mpifd);
    MPI_Barrier(com);

    Free(comg); 
    Free(g); 
    Free(iamg); 
    Free(nq_grp); 
    Free(nv_grp); 
    Free(ranks);
}
