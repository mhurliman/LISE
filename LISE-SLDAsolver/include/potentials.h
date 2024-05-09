// for license information, see the accompanying LICENSE file
#pragma once

#include "vars_nuclear.h"

// make_potentials.cpp
void allocate_pots(Potentials* pots, double hbar2m, double* pot_array, int ishift, int n);
double center_dist(const double* rho, int n, const Lattice_arrays* latt_coords, double* xc, double* yc, double* zc);
void coul_pot3(double* vcoul, const double* rho, int nstart, int nstop, int nxyz, double npart, FFtransf_vars* fftransf_vars, double dxyz);
int dens_func_params(int iforce, int ihfb, int isospin, Couplings* cc_edf, int ip, int icub, double alpha_pairing);
void get_u_re(MPI_Comm comm, Densities* dens_p, Densities* dens_n, Potentials* pots, Couplings* cc_edf, double hbar2m, int nstart, int nstop, int nxyz, int icoul, int isospin, const double* k1d_x, const double* k1d_y, const double* k1d_z, int nx, int ny, int nz, const Lattice_arrays* latt_coords, FFtransf_vars* fftransf_vars, double nprot, double dxyz);
void update_potentials(int icoul, int isospin, Potentials* pots, Densities* dens_p, Densities* dens_n, Densities* dens, Couplings* cc_edf, double e_cut, int nstart, int nstop, MPI_Comm comm, int nx, int ny, int nz, double hbar2m, complex* d1_x, complex* d1_y, complex* d1_z,double* k1d_x, double* k1d_y, double* k1d_z, const Lattice_arrays* latt_coords, FFtransf_vars* fftransf_vars, double nprot, double dxyz, int icub);
void mix_potentials(double* pot_array, double* pot_array_old, double alpha, int ishift, int n);
double center_dist(const double* rho, int n, const Lattice_arrays* latt_coords, double* xc, double* yc, double* zc);
double center_dist_pn(const double* rho_p, const double* rho_n, int n, const Lattice_arrays* latt_coords, double* xc, double* yc, double* zc);

// pots_io.cpp
int read_pots(const char* fn, double* pot_arrays, int nx, int ny, int nz, double dx, double dy, double dz, int ishift);
int write_pots(const char* fn, const double* pot_arrays, int nx, int ny, int nz, double dx, double dy, double dz, int ishift);

// external_pot.cpp
void external_pot(int iext, int n, int n_part, double hbo, double* v_ext, complex* delta_ext, double hbar2m, Lattice_arrays* lattice_coords, double rr, double rneck, double wneck, double z0, double v0);
void make_filter(double* filter, Lattice_arrays* latt_coords, int nxyz);
void external_so_m(const double* v_ext, double* wx, double* wy, double* wz, MPI_Comm gr_comm, complex* d1_x, complex* d1_y, complex* d1_z, int nx, int ny, int nz);