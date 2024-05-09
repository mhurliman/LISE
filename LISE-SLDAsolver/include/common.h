#pragma once

#include <assert.h>
#include <complex>

#include "vars_nuclear.h"
#include "memory.h"

using namespace std::literals::complex_literals;

using complex = std::complex<double>;

// ham_matrix.cpp
void i2xyz(const int i, int* ix, int* iy, int* iz, const int ny, const int nz);
void make_ham(complex* ham, const double* k1d_x, const double* k1d_y, const double* k1d_z, const complex* d1_x, const complex* d1_y, const complex* d1_z, const Potentials* pots, const Densities* dens, int nx, int ny, int nz, int m_ip, int n_iq, int i_p, int i_q, int mb, int nb, int p_proc, int q_proc, int nstart, int nstop, MPI_Comm comm);
void laplacean(const double* f, int n, double* lapf, int nstart, int nstop, MPI_Comm comm, const double* k1d_x, const double* k1d_y, const double* k1d_z, int nx, int ny, int nz, int ired);

// broyden_min.cpp
int broydenMod_min(double* v_in, double* v_in_old, double* f_old, double* b_bra, double* b_ket, double* d, double* v_out, int n, int* it_out, double alpha);

// create_destroy_groups
int create_mpi_groups(MPI_Comm commw, MPI_Comm* gr_comm, int np, int* gr_np, int* gr_ip, MPI_Group* group_comm);
void destroy_mpi_groups(MPI_Group* group_comm, MPI_Comm* gr_comm);

// axial_symmetry.cpp
int get_pts_tbc(int nx, int ny, int nz, double dx, double dy, Axial_symmetry* ax);
void axial_symmetry_densities(Densities* dens, const Axial_symmetry* ax, int nxyz, const Lattice_arrays* lattice_coords, FFtransf_vars* fftransf_vars, MPI_Comm comm, int iam);

// constr_dens.cpp
void allocate_dens(Densities* dens, int ip, int np, int nxyz);
void mem_share(Densities* dens, double* ex_array, int nxyz, int idim);
void make_coordinates(int nxyz, int nx, int ny, int nz, double dx, double dy, double dz, Lattice_arrays* lattice_vars);
void exch_nucl_dens(MPI_Comm commw, int ip, int gr_ip, int gr_np, int idim, double* ex_array_p, double* ex_array_n);
void generate_ke_1d(double* ke, int n, double a, int iopt_der);
void generate_der_1d(complex* der, int n, double a, int iopt_der);
double rescale_dens(Densities* dens, int nxyz, double npart, double dxyz, int iscale);
void compute_densities(const double* lam, const complex* z, int nxyz, int ip, MPI_Comm comm, Densities* dens, int m_ip, int n_iq, int i_p, int i_q, int mb, int nb, int p_proc, int q_proc, int nx, int ny, int nz, const complex* d1_x, const complex* d1_y, const complex* d1_z, const double* k1d_x, const double* k1d_y, const double* k1d_z, double e_max, int* nwf, double* occ, int icub);
double factor_ec(double e, double e_cut, int icub);

// dens_start.cpp
void dens_startTheta(double A_mass, double npart, int nxyz, Densities* dens, const Lattice_arrays* lattice_coords, double dxyz, double lx, double ly, double lz, int ideform);

// dens_io.cpp
int read_dens(const char* fn, Densities* dens, MPI_Comm comm, int iam, int nx, int ny, int nz, double* amu, double dx, double dy, double dz, const char* filename);
int copy_lattice_arrays(void* bf1, void* bf, size_t siz, int nx1, int ny1, int nz1, int nx, int ny, int nz);
int read_constr(const char* fn, int nxyz, double* cc_lambda, int iam, MPI_Comm comm);
int write_dens_txt(FILE* fd, const Densities* dens, MPI_Comm comm, int iam, int nx, int ny, int nz, const double* amu);
int write_qpe(const char* fn, double* lam, MPI_Comm comm, int iam, int nwf);
int write_dens(const char* fn, const Densities* dens, MPI_Comm comm, int iam, int nx, int ny, int nz, const double* amu, double dx, double dy, double dz);

// deform.cpp
void deform(const double* rho_p, const double* rho_n, int nxyz, const Lattice_arrays* latt_coords, double dxyz, FILE* fout);

// get-mem-req-blk-cyc.cpp
void get_mem_req_blk_cyc(int ip, int iq, int np, int nq, int ma, int na, int mblk, int nblk, int *nip, int *niq);

// get-blcs-dscr.cpp
void get_blcs_dscr(MPI_Comm commc, int m, int n, int mb, int nb, int p, int q, int* ip, int* iq, int* blcs_dscr, int* nip, int* niq);

// system_energy.cpp
void system_energy(const Couplings* cc_edf, int icoul, const Densities* dens, const Densities* dens_p, const Densities* dens_n, int isospin, complex* delta, int nstart, int nstop, int ip, int gr_ip, MPI_Comm comm, MPI_Comm gr_comm,double* k1d_x, double* k1d_y, double* k1d_z, int nx, int ny, int nz, double hbar2m, double dxyz, const Lattice_arrays* latt_coords, FFtransf_vars* fftransf_vars, double nprot,double nneut, FILE* out);

// 2dbc-slda-mpi-wr.cpp
void bc_wr_mpi(const char* fn, MPI_Comm com, int p, int q, int ip, int iq, int blk, int jstrt, int jstp, int jstrd, int nxyz, complex* z);

// print_wf.cpp
int print_wf2(const char* fn, MPI_Comm comm, const double* lam, const complex* z, int ip, int nwf, int m_ip, int n_iq, int i_p, int i_q, int mb, int nb, int p_proc, int q_proc, int nx, int ny, int nz, double e_cut, double* occ, int icub);


int     opterr = 1,             /* if error message should be printed */
optind = 1,             /* index into parent argv vector */
optopt,                 /* character checked for validity */
optreset;               /* reset getopt */
char* optarg;                /* argument associated with option */

#define BADCH   (int)'?'
#define BADARG  (int)':'
#define EMSG    ""

/*
* getopt --
*      Parse argc/argv argument vector.
*/
int getopt(int nargc, char* const nargv[], const char* ostr);

// cblacs
void Cblacs_pinfo(int*, int*);
void Cblacs_setup(int*, int*);
void Cblacs_get(int, int, int*);
void Cblacs_gridinit(int*, char*, int, int);
void Cblacs_gridinfo(int, int*, int*, int*, int*);
void Cblacs_exit(int);
void Cfree_blacs_system_handle(int blacshandle);
int Csys2blacs_handle(MPI_Comm comm);

template <typename T>
T Square(T x) { return x * x; }
