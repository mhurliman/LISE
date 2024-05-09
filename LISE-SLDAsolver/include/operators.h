// for license information, see the accompanying LICENSE file
#pragma once

#include <complex.h>
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
);

void gradient_real_orig(
    const double* f, 
    int n, 
    double* g_x, double* g_y, double* g_z, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int ired
);

void gradient(
    const complex* f, 
    int n, 
    complex* g_x, complex* g_y, complex* g_z, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int ired
);

void gradient_ud(
    const complex* f, 
    int n, 
    complex* g_x, complex* g_y, complex* g_z, 
    complex* g_xd, complex* g_yd, complex* g_zd, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz
);

void gradient_orig(
    const complex* f, 
    int n, 
    complex* g_x, complex* g_y, complex* g_z, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int ired
);

void laplacean_complex(
    const complex* f, 
    int n, 
    complex* lapf, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    int nx, int ny, int nz, 
    int ired
);

void laplacean(
    const double* f, 
    int n, 
    double* lapf, 
    int nstart, int nstop, 
    MPI_Comm comm, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    int nx, int ny, int nz, 
    int ired
);

void diverg(
    const double* fx, const double* fy, const double* fz, 
    double* divf, 
    int n,  
    int nstart, int nstop, 
    MPI_Comm comm, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz
);

void match_lattices(
    Lattice_arrays* latt3, 
    int nx, int ny, int nz, 
    int nx3, int ny3, int nz3, 
    FFtransf_vars* fftrans, 
    double Lc
);
