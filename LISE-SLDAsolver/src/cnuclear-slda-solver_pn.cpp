// for license information, see the accompanying LICENSE file

/* Main program */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <complex.h>
#include <assert.h>

#include "common.h"
#include "potentials.h"
#include "operators.h"

int parse_input_file(char* file_name);
int readcmd(int argc, char* argv[], int ip);
void print_help(const int ip, char** argv);

//Summit, use IBM's build of Netlib ScaLAPACK
void pzheevd(char*, char*, int*, complex*, int*, int*, int*, double*, complex*, int*, int*, int*, complex*, int*, double*, int*, int*, int*, int*);

//Intel MKL
//#include <mkl.h>
//#include <mkl_scalapack.h>
//#include <mkl_blacs.h>
//#include <mkl_pblas.h>
//void  pzheevd(const char* jobz, const char* uplo, const MKL_INT* n, const MKL_Complex16* a, const MKL_INT* ia, const MKL_INT* ja, const MKL_INT* desca, double* w, MKL_Complex16* z, const MKL_INT* iz, const MKL_INT* jz, const MKL_INT* descz, MKL_Complex16* work, const MKL_INT* lwork, double* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info);



metadata_t md =
{
  8, // nx
  8, // ny
  8, // nz
  1.0, // dx
  1.0, // dy
  1.0, // dz
  0, // broyden
  1, // coulomb
  100, // niter
  8, // nprot
  8, // nneut
  0, // iext
  1, // force
  1, // pairing
  0.25, // alpha_mixing
  200.0, // ecut
  //    1, // icub
      0, // imass
      0, // icm
      0, // irun
      1, // isymm
      0, // resc_dens
      0, // iprint_wf
      0, // deformation
      0.0, // q0 (quadrupole constraints)
      0.0, // v0 (strength of external field)
      0.0, // z0: reference point in fission process
      0.0, // wneck: parameter of twisted potential
      0.0, // rneck: parameter of twisted potential
      8, // p
      8, // q
      40, //mb
      40, // nb
      1e10, //ggp: above 1e9 value does not record
      1e10, //ggn
      0.0 // alpha_pairing: 0.0 for volume, 0.5 for mixed, 1.0 for surface
};

#define EXT_POT_ZERO 0
#define EXT_POT_HO 1
#define EXT_POT_WOODS_SAXON 2
#define EXT_POT_CONSTANT_PAIRING 3

int main(int argc, char** argv)
{
    
    int it, niter = 2, it_broy = 0;
    double e_cut = 75.0;
    MPI_Status istats;
    int tag1 = 120, tag2 = 200;
    int i_p, i_q, p_proc = -1, q_proc = -1, mb = 40, nb = 40, m_ip, n_iq;  /* used for decomposition */
    Potentials pots, *pots_ptr, pots_old;
    double v[3];
    double* v_in_old;
    double* f_old, *b_bra, *b_ket, *diag;
    int i_b;
    int hw;
    long long int rcy, rus, ucy, uus;
    FFtransf_vars fftransf_vars;
    Axial_symmetry ax;
    double xcm, ycm, zcm;
    double rcm[3];
    int descr_h[9], descr_z[9];
    int icoul = 1, ihfb = 1, imin = 0, iforce = 1, ider = 1; /* options for
                                              icoul = 0 no Coulomb, 1 Coulomb = default
                              ihfb = 0 no pairing , 1 pairing
                              imin = 0 simple mixing, 1 Broyden
                          iforce = 0 no interaction, 1 = SLy4 ( default ) , 2 = SkM* , 11 = Sly4 + surface pairing
                          */
    int ideform = 0;  /* for initial densities, the deformation ideform = 0 sherical, ideform = 2 axially symmetric , ideform = 3 triaxial */
    int iext = 0; /* different external potential choices */
    int irun = 0; /* 0 = run with spherical density, 1 = read densities */
    int icub = 1; /* 0 = use spherical cutoff, 1 = use cubic cutoff*/
    int imass = 0; /* 0 = same masses for protons and neutrons, 1 = different masses */
    int icm = 0; /* 0 = no center of mass correction, 1 = center of mass correction */
    int irsc = 0;
    int iprint_wf = -1; /* -1 no wfs saved , 0 all wfs saved , 1 Z proton wfs saved, 2 N neutron wfs saved */
    double ggp = 1e10, ggn = 1e10; // pairing coupling constants
    double alpha_pairing = 0.0; // pairing mixing parameter: 0 volume, 0.5 mixed, 1.0 volume.
    int m_broy, ishift, m_keep = 7;
    double* npart;
    int i, ii, j, na, info, Ione = 1, itb;
    
    complex s1;
    complex* work, tw[2];
    double* rwork, tw_[2];

    //complex * rwork , tw_[2] ;
    double* amu, const_amu = 4.e-2, c_q2 = 1.e-5;
    double mass_p = 938.272013;
    double mass_n = 939.565346;
    double hbarc = 197.3269631;
    double hbar2m = Square(hbarc) / (mass_p + mass_n);
    double hbo = 2.0, alpha_mix = 0.25, rone = 1.0;
    double err;

    int idim;
    int nwf, nwf_n;
    char fn[FILENAME_MAX] = "densities_p_0.cwr", iso_label[] = "_p";
    char fn_[FILENAME_MAX]; /* for testing the different versions */
    FILE* fd_dns;
    int option_index = 0;
    double xconstr[4], z0 = 0., y0 = 0., asym = 0., v0 = 0., wneck = 100., rneck = 100.;
    double xlagmul[4];
    double Lx = -1.0, Ly = -1.0, Lz = -1.0;
    int ierr;

    setbuf(stdout, NULL);

    // Initialize MPI w/ proton & neutron groups
    MPI_Init(&argc, &argv);

    int ip, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &ip);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Group group_comm;
    MPI_Comm gr_comm;
    int gr_ip, gr_np;
    int isospin = create_mpi_groups(MPI_COMM_WORLD, &gr_comm, np, &gr_np, &gr_ip, &group_comm);

    if (ip == 0)
    {
        i = readcmd(argc, argv, ip);
        if (i == -1)
        {
            printf("TERMINATING! NO INPUT FILE.\n");

            ierr = -1;
            MPI_Abort(MPI_COMM_WORLD, ierr);
            return EXIT_FAILURE;
        }

        // Read input file
        // Info from file is loaded into metadata structure
        printf("READING INPUT: `%s`\n", argv[i]);

        j = parse_input_file(argv[i]);
        if (j == 0)
        {
            printf("PROBLEM WITH INPUT FILE: `%s`.\n", argv[i]);

            ierr = -1;
            MPI_Abort(MPI_COMM_WORLD, ierr);
            return EXIT_FAILURE;
        }
    }

    // Broadcast input parameters
    MPI_Bcast(&md, sizeof(md), MPI_BYTE, 0, MPI_COMM_WORLD);

    int nx = md.nx; 
    int ny = md.ny; 
    int nz = md.nz;
    double dx = md.dx; 
    double dy = md.dy; 
    double dz = md.dz;

    if (md.broyden == 1) 
        imin = 1;  // broyden minimization will be turned on

    if (md.coulomb == 0) 
        icoul = 0; // no coulomb potential

    double nprot = md.nprot;
    double nneut = md.nneut;
    int niter = md.niter;

    icm = md.icm;
    if (icm == 1)
        hbar2m = hbar2m * (1.0 - 1.0 / (nneut + nprot));

    imass = md.imass;

    if (imass == 1) 
    {
        hbar2m = 0.5 * Square(hbarc);

        if (isospin == 1)
            hbar2m /= mass_p;
        else
            hbar2m /= mass_n;
    }

    iext = md.iext;

    iforce = md.force;

    if (md.pairing == 0) ihfb = 0;

    alpha_mix = md.alpha_mix;
    e_cut = md.ecut;   // energy cutoff

    if (icub == 0)
    {
        // Spherical cut-off
        e_cut = hbar2m * PI * PI / dx / dx;
    }

    irun = md.irun;   // choice of start of density

    if (md.resc_dens == 1) 
    {
        irsc = 1;
    }

    iprint_wf = md.iprint_wf;   // choice of printing wfs

    ideform = md.deformation;
    xconstr[0] = md.q0; // quadruple constraint
    z0 = md.z0;  // reference point of distribution

    wneck = md.wneck;
    rneck = md.rneck;

    v0 = md.v0;

    // cyclic distribution in scalapack
    p_proc = md.p;
    q_proc = md.q;
    mb = md.mb;
    nb = md.nb;

    // pairing coupling constants
    ggp = md.ggp;
    ggn = md.ggn;
    // pairing mixing parameter.
    alpha_pairing = md.alpha_pairing;

    if (nx < 0 || ny < 0 || nz < 0 || nprot < 0. || nneut < 0.)
    {
        if (ip == 0)
        {
            fprintf(stdout, "nx=%d ny=%d nz=%d \n", nx, ny, nz);
            fprintf(stdout, "Lx=%f Ly=%f Lz=%f \n", Lx, Ly, Lz);
            fprintf(stdout, "nprot=%f nneut=%f\n", nprot, nneut);
            fprintf(stdout, "required parameters not provided \n\n ");
        }

        print_help(ip, argv);

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (p_proc < 0 && q_proc < 0)
    {
        p_proc = (int)sqrt(gr_np);
        q_proc = p_proc; /* square grids, although not required */
    }
    else
    {
        if (p_proc < 0)
            p_proc = gr_np / q_proc;

        if (q_proc < 0)
            q_proc = gr_np / p_proc;

        if (gr_ip == 0)
            fprintf(stdout, "%d x %d grid \n", p_proc, q_proc);
    }

    int nxyz = nx * ny * nz;
    int Lx = nx * dx;
    int Ly = ny * dy;
    int Lz = nz * dz;
    double dxyz = dx * dy * dz;
    double sdxyz = 1.0 / sqrt(dxyz);

    pots.nxyz = nxyz;
    pots_old.nxyz = nxyz;

    int isymm = md.isymm;

    if (ip == 0)
    {
        fprintf(stdout, " *********************************** \n");

        fprintf(stdout, " * Welcome to the world of wonders *\n * brought to you by the magic of SLDA *\n ");
        fprintf(stdout, " *********************************** \n\n\n");
        fprintf(stdout, " You have requested a calculation with the following parameters: \n");
        fprintf(stdout, " nx=%d ny=%d nz=%d dx=%g dy=%g dz=%g\n", nx, ny, nz, dx, dy, dz);
        fprintf(stdout, " Lx=%g fm Ly=%g fm Lz=%g fm\n", Lx, Ly, Lz);
        fprintf(stdout, " Z=%f N=%f\n\n", nprot, nneut);
    }

    ax.ind_xyz = Allocate<int>(nxyz);
    ax.car2cyl = Allocate<int>(nxyz);

    if (isymm == 1)
    {
        if ((ierr = get_pts_tbc(nx, ny, nz, dx, dy, &ax)) != 0)
        {
            fprintf(stdout, "ip[%d], ERROR IN CONSTRUCTING CYLINDRICAL COORDINATES, EXITTING ....\n", ip);

            MPI_Abort(MPI_COMM_WORLD, ierr);
            return(EXIT_FAILURE);
        }
        else
        {
            if (ip == 0)
            {
                fprintf(stdout, "%d pts need to be calculated in xy-plane, %d totally in full space\n", ax.npts_xy, ax.npts_xyz);
            }

        }
    }
    else
        // calculate full lattice 
    {
        ax.npts_xy = nx * ny;

        ax.npts_xyz = nxyz;

        for (int i = 0; i < nxyz; i++)
        {
            ax.ind_xyz[i] = i;
            ax.car2cyl[i] = i;
        }
        if (ip == 0)
        {
            fprintf(stdout, "%d pts need to be calculated in xy-plane, %d totally in full space\n", ax.npts_xy, ax.npts_xyz);
        }

    }

    if (imin == 0)
    {
        m_broy = 7 * nxyz + 5;
        ishift = 0;

        if (ip == 0)
        {
            fprintf(stdout, "Linear mixing will be performed \n");
        }
    }
    else
    {
        m_broy = 2 * (7 * nxyz + 5);

        if (ip == 0)
        {
            fprintf(stdout, "Broyden mixing will be performed\n");
            f_old = Allocate<double>(m_broy);
            b_bra = Allocate<double>((niter - 1) * m_broy);
            b_ket = Allocate<double>((niter - 1) * m_broy);
            diag = Allocate<double>(niter - 1);
            v_in_old = Allocate<double>(m_broy);
        }

        ishift = (7 * nxyz + 5) * (1 - isospin) / 2;
    }

    if (ip == 0)
    {
        fprintf(stdout, "alpha_mix = %f \n", alpha_mix);
    }

    int root_p = 0;
    int root_n = gr_np;
    
    double* pot_array = AllocateZeroed<double>(m_broy);
    double* pot_array_old = AllocateZeroed<double>(m_broy);

    allocate_pots(&pots, hbar2m, pot_array, ishift, m_broy);
    allocate_pots(&pots_old, hbar2m, pot_array_old, ishift, m_broy);

    *pots.amu = -12.5;
    *pots_old.amu = -12.5;

    memset(pots.lam, 0, 4 * sizeof(pots.lam[0]));
    memset(xconstr, 0, 4 * sizeof(xconstr[0]));

    Densities dens_p, dens_n;
    allocate_dens(&dens_p, gr_ip, gr_np, nxyz);
    allocate_dens(&dens_n, gr_ip, gr_np, nxyz);

    Densities* dens = isospin == 1 ? &dens_p : &dens_n;

    if (dens->nstop - dens->nstart > 0)
    {
        idim = 2.0 * (dens->nstop - dens->nstart) + nxyz;
    }
    else
    {
        idim = nxyz;
    }

    double* ex_array_n, * ex_array_p;
    double* ex_array_p = Allocate<double>(idim);
    double* ex_array_n = Allocate<double>(idim);

    mem_share(&dens_p, ex_array_p, nxyz, idim);
    mem_share(&dens_n, ex_array_n, nxyz, idim);

    if (isospin == 1)
    {
        dens->nu = dens_p.nu;
        npart = &nprot;
        amu = pots.amu;

        if (dens->nstart - dens->nstop > 0)
        {
            free(dens_n.nu);
        }
    }
    else
    {
        icoul = 0;

        dens->nu = dens_n.nu;
        npart = &nneut;
        amu = pots.amu;

        if (dens->nstart - dens->nstop > 0)
        {
            free(dens_p.nu);
        }

        sprintf(iso_label, "_n");
    }

    double dx_ = dx;
    int n3 = nx;
    if (n3 < ny) 
    {
        n3 = ny;
        dx_ = dy;
    }

    if (n3 < nz) 
    {
        n3 = nz;
        dx_ = dz;
    }

    // Build the x and k lattice arrays
    int nx3 = 3 * n3;
    int ny3 = 3 * n3;
    int nz3 = 3 * n3;
    int nxyz3 = nx3 * ny3 * nz3;;

    Lattice_arrays lattice_coords, lattice_coords3;
    make_coordinates(nxyz, nx, ny, nz, dx, dy, dz, &lattice_coords);
    make_coordinates(nxyz3, nx3, ny3, nz3, dx, dy, dz, &lattice_coords3);

    // Create the FFT plans and buffers
    FFtransf_vars fftransf_vars;
    fftransf_vars.nxyz3 = nxyz3;

    double Lc = sqrt(Square(n3 * dx_) + Square(n3 * dx_) + Square(n3 * dx_));
    match_lattices(&lattice_coords3, nx, ny, nz, nx3, ny3, nz3, &fftransf_vars, Lc);

    fftransf_vars.buff = Allocate<complex>(nxyz);
    fftransf_vars.buff3 = Allocate<complex>(nxyz3);

    fftransf_vars.plan_f = fftw_plan_dft_3d(nx, ny, nz, (fftw_complex*)fftransf_vars.buff, (fftw_complex*)fftransf_vars.buff, FFTW_FORWARD, FFTW_ESTIMATE);
    fftransf_vars.plan_b = fftw_plan_dft_3d(nx, ny, nz, (fftw_complex*)fftransf_vars.buff, (fftw_complex*)fftransf_vars.buff, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftransf_vars.plan_f3 = fftw_plan_dft_3d(nx3, ny3, nz3, (fftw_complex*)fftransf_vars.buff3, (fftw_complex*)fftransf_vars.buff3, FFTW_FORWARD, FFTW_ESTIMATE);
    fftransf_vars.plan_b3 = fftw_plan_dft_3d(nx3, ny3, nz3, (fftw_complex*)fftransf_vars.buff3, (fftw_complex*)fftransf_vars.buff3, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftransf_vars.filter = Allocate<double>(nxyz);

    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int iz = 0; iz < nz; iz++)
            {
                if (ix == nx / 2 || iy == ny / 2 || iz == nz / 2)
                {
                    fftransf_vars.filter[iz + nz * (iy + ny * ix)] = 0;
                }
                else
                {
                    fftransf_vars.filter[iz + nz * (iy + ny * ix)] = 1.0;
                }
            }
        }
    }

    /* constructing the phase factor in densities */
    assert(dens->phases = (complex*) malloc(nxyz * sizeof(complex)));

    for (int i = 0; i < nxyz; i++)
    {
        double xa = lattice_coords.xa[i];
        double ya = lattice_coords.ya[i];

        double rr = sqrt(Square(xa) + Square(ya));

        if (rr < 1e-14)  // at center of vortex (vortex line) the pairing field should vanish
        {
            dens->phases[i] = 0;
        }
        else
        {
            double phi = asin(ya / rr);

            if (phi > 1e-14)
            {
                if (xa < 0)
                    phi = PI - phi;
            }
            else if (phi < -1e-14)
            {
                if (xa < 0)
                    phi = -PI - phi;
            }
            else
            {
                if (xa < 0)
                    phi = -PI;
            }

            dens->phases[i] = std::exp(phi * 1i);
        }
    }
    
    /*************************************************************/
    if (icoul == 1) /* declare variables for fourier transforms needed for Coulomb */
    {
        if (gr_ip == 0)
        {
            fprintf(stdout, "Coulomb interaction between protons included \n");
        }
    }

    switch (irun)
    {
    case 0:
        dens_startTheta(nprot + nneut, *npart, nxyz, dens, &lattice_coords, dxyz, nx * dx, ny * dy, nz * dz, ideform);
        exch_nucl_dens(MPI_COMM_WORLD, ip, gr_ip, gr_np, idim, ex_array_p, ex_array_n);

        if (ip == 0) 
        {
            fprintf(stdout, "initial densities set\n");

            double* rho_t = AllocateInit<double>(nxyz, [&](int i) { return dens_p.rho[i] + dens_n.rho[i]; });

            center_dist(rho_t, nxyz, &lattice_coords, &xcm, &ycm, &zcm);

            fprintf(stdout, "Initial: xcm=%f ycm=%f zcm=%f\n", xcm, ycm, zcm);
            Free(rho_t);
        }
        break;

    case 1:
        sprintf(fn, "dens%s.cwr", iso_label);
        sprintf(fn_, "interpDens%s.cwr", iso_label);

        if (gr_ip == 0)
        {
            fprintf(stdout, "Reading densities from %s\n", fn);
        }

        if (read_dens(fn, dens, gr_comm, gr_ip, nx, ny, nz, amu, dx, dy, dz, fn_) == EXIT_FAILURE)
        {
            if (gr_ip == 0)
            {
                fprintf(stdout, " \n *** \n Could not read the densities \n Exiting \n");
            }
            
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        else
        {
            MPI_Barrier(gr_comm);
            if (gr_ip == 0)
            {
                fprintf(stdout, "The densities were successfully read by proc %d\n", ip);
            }
        }

        exch_nucl_dens(MPI_COMM_WORLD, ip, gr_ip, gr_np, idim, ex_array_p, ex_array_n);
        break;

    case 2:
        if (gr_ip == 0)
        {
            sprintf(fn, "pots%s.cwr", iso_label);
            i = read_pots(fn, pot_array, nx, ny, nz, dx, dy, dz, ishift);
            ii = i;
        }

        MPI_Bcast(&i, 1, MPI_INT, root_p, MPI_COMM_WORLD);
        if (i == EXIT_FAILURE)
        {
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        MPI_Bcast(&ii, 1, MPI_INT, root_n, MPI_COMM_WORLD);
        if (ii == EXIT_FAILURE)
        {
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        if (imin == 0)
        {
            MPI_Bcast(pot_array, 7 * nxyz + 5, MPI_DOUBLE, 0, gr_comm);
        }
        else
        {
            MPI_Bcast(pot_array + 7 * nxyz + 5, 7 * nxyz + 5, MPI_DOUBLE, root_n, MPI_COMM_WORLD);
            MPI_Bcast(pot_array, 7 * nxyz + 5, MPI_DOUBLE, root_p, MPI_COMM_WORLD);
        }
        break;
    }

    external_pot(iext, nxyz, (int)(nprot + nneut), hbo, pots.v_ext, pots.delta_ext, hbar2m, &lattice_coords, .35 * nx * dx, rneck, wneck, z0, v0);

    // using HFBTHO constraint field
    if (iext == 70)
    {
        double cc_lambda[3];
        double* constr;
        double* filter;
        assert(constr = (double*)malloc(3 * nxyz * sizeof(double)));
        assert(filter = (double*)malloc(3 * nxyz * sizeof(double)));

        read_constr("constr.cwr", nxyz, cc_lambda, gr_ip, gr_comm);
        for (int i = 0; i < nxyz; i++) 
            filter[i] = 1.;

        make_filter(filter, &lattice_coords, nxyz);

        for (int i = 0;i < nxyz;i++)
        {
            double xa = lattice_coords.xa[i];
            double ya = lattice_coords.ya[i];
            double za = lattice_coords.za[i];

            constr[i] = za / 10;  // dipole
            constr[i + nxyz] = (2 * za * za - xa * xa - ya * ya) / 100.0;  // quadrupole
            constr[i + 2 * nxyz] = za * (2 * za * za - 3 * xa * xa - 3 * ya * ya) * sqrt(7 / PI) / 4 / 1000.0; // oxatupole
        }

        ZeroMemory(pots.v_ext, nxyz);

        for (int j = 0; j < 3; j++)
        {
            for (int i = 0; i < nxyz; i++)
            {
                pots.v_ext[i] += -1.0 * (constr[i + j * nxyz] * cc_lambda[j]) * filter[i];
            }
        }
        
        // add a Wood-Saxon filter

        Free(constr); 
        Free(filter);
    }

#ifdef CONSTRCALC
    make_constraint(pots.v_constraint, lattice_coords.xa, lattice_coords.ya, lattice_coords.za, nxyz, y0, z0, asym, lattice_coords.wx, lattice_coords.wy, lattice_coords.wz, v0);
#ifdef CONSTR_Q0
    c_q2 = 0.05 / xconstr[0];
    pots.lam2[0] = c_q2 * (q2av(dens_p.rho, dens_n.rho, pots.v_constraint, nxyz, nprot, nneut) * dxyz - xconstr[0]) + pots.lam[0];
#endif

    center_dist_pn(dens_p.rho, dens_n.rho, nxyz, &lattice_coords, rcm, rcm + 1, rcm + 2);

    for (j = 1; j < 4; j++) 
    {
        pots.lam2[j] = 3.e-1 * (rcm[j - 1] - xconstr[j]) + pots.lam[j];
    }
#endif

    Couplings cc_edf;
    dens_func_params(iforce, ihfb, isospin, &cc_edf, ip, icub, alpha_pairing);

    if (ggp < 1e9) 
    {
        cc_edf.gg_p = ggp;

        if (isospin == 1) 
            cc_edf.gg = ggp;
    }

    if (ggn < 1e9) 
    {
        cc_edf.gg_n = ggn;

        if (isospin == -1) 
            cc_edf.gg = ggn;
    }

    if (ip == 0)
    {
        fprintf(stdout, " ** Pairing parameters ** \n proton strength = %f neutron strength = %f \n", cc_edf.gg_p, cc_edf.gg_n);
    }
    
    /* need to set the grid here */
    int na = 4 * nxyz;
    get_blcs_dscr(gr_comm, na, na, mb, nb, p_proc, q_proc, &i_p, &i_q, descr_h, &m_ip, &n_iq);

    complex* z_eig, *ham; /* the Hamiltonian to be diagonalized, cyclicly decomposed */
    complex* ham = Allocate<complex>(m_ip * n_iq);
    complex* z_eig = Allocate<complex>(m_ip * n_iq);

    double* lam, *lam_old;
    double* lam = Allocate<double>(na);
    double* lam_old = Allocate<double>(2 * nxyz);

    memset(lam_old, 0, 2 * nxyz * sizeof(lam_old[0]));

    /* allocate some extra arrays to store the 1D KE and Derivatives */
    double* k1d_x = Allocate<double>(nx);
    double* k1d_y = Allocate<double>(ny);
    double* k1d_z = Allocate<double>(nz);

    generate_ke_1d(k1d_x, nx, dx, ider);
    generate_ke_1d(k1d_y, ny, dy, ider);
    generate_ke_1d(k1d_z, nz, dz, ider);

    complex* d1_x = Allocate<complex>(2 * nx - 1);
    complex* d1_y = Allocate<complex>(2 * ny - 1);
    complex* d1_z = Allocate<complex>(2 * nz - 1);

    generate_der_1d(d1_x, nx, dx, ider);
    generate_der_1d(d1_y, ny, dy, ider);
    generate_der_1d(d1_z, nz, dz, ider);

    if (iext == 5 && irun < 2)
    {
        external_so_m(pots.v_ext, pots.wx, pots.wy, pots.wz, gr_comm, d1_x, d1_y, d1_z, nx, ny, nz);
    }

    double* occ = Allocate<double>(2 * nxyz);

    if (iforce != 0 && irun < 2)
    {
        get_u_re(gr_comm, &dens_p, &dens_n, &pots, &cc_edf, hbar2m, dens->nstart, dens->nstop, nxyz, icoul, isospin, k1d_x, k1d_y, k1d_z, nx, ny, nz, &lattice_coords, &fftransf_vars, nprot, dxyz);
        update_potentials(icoul, isospin, &pots, &dens_p, &dens_n, dens, &cc_edf, e_cut, dens->nstart, dens->nstop, gr_comm, nx, ny, nz, hbar2m, d1_x, d1_y, d1_z, k1d_x, k1d_y, k1d_z, &lattice_coords, &fftransf_vars, nprot, dxyz, icub);
    }

#ifdef USEPOTENT
    if (gr_ip == 0)
    {
        sprintf(fn, "pots%s.cwr", iso_label);
        i = read_pots(fn, pot_array, nx, ny, nz, dx, dy, dz, ishift);
        ii = i;
    }

    MPI_Bcast(&i, 1, MPI_INT, root_p, MPI_COMM_WORLD);
    if (i == EXIT_FAILURE)
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Bcast(&ii, 1, MPI_INT, root_n, MPI_COMM_WORLD);
    if (ii == EXIT_FAILURE)
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (imin == 0)
        MPI_Bcast(pot_array, 7 * nxyz + 5, MPI_DOUBLE, 0, gr_comm);
    else
    {
        MPI_Bcast(pot_array + 7 * nxyz + 5, 7 * nxyz + 5, MPI_DOUBLE, root_n, MPI_COMM_WORLD);
        MPI_Bcast(pot_array, 7 * nxyz + 5, MPI_DOUBLE, root_p, MPI_COMM_WORLD);
    }

#endif
    if (imin != 0)
    {
        if (ip == root_n)
        {
            MPI_Send(pot_array + m_broy / 2, m_broy / 2, MPI_DOUBLE, root_p, tag1, MPI_COMM_WORLD);
        }

        if (ip == root_p)
        {
            MPI_Recv(pot_array + m_broy / 2, m_broy / 2, MPI_DOUBLE, root_n, tag1, MPI_COMM_WORLD, &istats);
        }
    }

    mix_potentials(pot_array, pot_array_old, rone, 0, m_broy);

    double xpart = rescale_dens(dens, nxyz, *npart, dxyz, irsc);

    double xxpart = *npart * xpart;
    const_amu = 25.0 / *npart;

    FILE* file_out = NULL;
    if (gr_ip == 0)
    {
        sprintf(fn, "dens%s_start_info.txt", iso_label);

        file_out = fopen(fn, "w");

        if (isospin == 1)
            deform(dens_p.rho, dens_n.rho, nxyz, &lattice_coords, dxyz, file_out);

        fprintf(stdout, "N%s = %13.9f\n", iso_label, xxpart);
        fprintf(stdout, "mu%s = %f\n", iso_label, *amu);
    }

    if (irun == 1)
    {
        system_energy(&cc_edf, icoul, dens, &dens_p, &dens_n, isospin, pots.delta, dens->nstart, dens->nstop, ip, gr_ip, MPI_COMM_WORLD, gr_comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, hbar2m, dxyz, &lattice_coords, &fftransf_vars, nprot, nneut, file_out);
    }

    complex* delta_old;
    assert(delta_old = (complex*) malloc(nxyz * sizeof(complex)));

    for (int i = 0; i < nxyz; i++) 
    {
        delta_old[i] = pots.delta[i];
    }

    if (write_dens_txt(file_out, dens, gr_comm, gr_ip, nx, ny, nz, amu) != EXIT_SUCCESS)
    {
        fprintf(stdout, "Error, could not write densities from input file\n");

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (gr_ip == 0)
    {
        fclose(file_out);
    }

    if (gr_ip == 0)
    {
        sprintf(fn, "pots%s_start.txt", iso_label);

        file_out = fopen(fn, "w");

        for (int i = 0;i < nxyz;i++)
            fprintf(file_out, "delta[%d] = %.12le %12leI\n", i, std::real(pots.delta[i]), std::imag(pots.delta[i]));

        fclose(file_out);
    }

    // Main SCF iteration loop
    for (int it = 0; it < niter; it++)
    {
        make_ham(ham, k1d_x, k1d_y, k1d_z, d1_x, d1_y, d1_z, &pots, dens, nx, ny, nz, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc, dens->nstart, dens->nstop, gr_comm);

        /* scalapack diag */
        int lwork = -1; /* probe the system for information */
        int lrwork = -1;
        int liwork = 7 * na + 8 * n_iq + 2;
        int info;

        int* iwork = Allocate<int>(liwork);

        // gcc
        // pzheevd_( "V" , "L" , &na , ham , &Ione , &Ione , descr_h , lam , z_eig , &Ione , &Ione , descr_h , tw , &lwork , tw_ , &lrwork , iwork , &liwork , &info ) ;
        // ibm xl
        pzheevd("V", "L", &na, ham, &Ione, &Ione, descr_h, lam, z_eig, &Ione, &Ione, descr_h, tw, &lwork, tw_, &lrwork, iwork, &liwork, &info);
        liwork = iwork[0];
        Free(iwork);

        int* iwork = Allocate<int>(liwork);

        lwork = (int)std::real(tw[0]);
        complex* work = Allocate<complex>(lwork);

        lrwork = (int)std::real(tw_[0]);
        double* rwork = Allocate<double>(lrwork);
        
        // gcc
        // pzheevd_( "V" , "L" , &na , ham , &Ione , &Ione , descr_h , lam , z_eig , &Ione , &Ione , descr_h , work , &lwork , rwork , &lrwork , iwork , &liwork , &info ) ;
        // ibm xl
        pzheevd("V", "L", &na, ham, &Ione, &Ione, descr_h, lam, z_eig, &Ione, &Ione, descr_h, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);

        Free(iwork); 
        Free(work); 
        Free(rwork);

        for (int ii = 0; ii < m_ip * n_iq; ii++)
        {
            z_eig[ii] = sdxyz * z_eig[ii];
        }
        
        compute_densities(lam, z_eig, nxyz, gr_ip, gr_comm, dens, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc, nx, ny, nz, d1_x, d1_y, d1_z, k1d_x, k1d_y, k1d_z, e_cut, &nwf, occ, icub);

        if (isymm == 1)
        {
            axial_symmetry_densities(dens, &ax, nxyz, &lattice_coords, &fftransf_vars, gr_comm, gr_ip);
        }
        
        for (int i = 0; i < nwf; i++)
        {
            occ[i] = occ[i] * dxyz;
        }

        xpart = rescale_dens(dens, nxyz, *npart, dxyz, irsc);
        exch_nucl_dens(MPI_COMM_WORLD, ip, gr_ip, gr_np, idim, ex_array_p, ex_array_n);
        sprintf(fn, "dens%s_%1d.cwr", iso_label, it % 2);

        if (write_dens(fn, dens, gr_comm, gr_ip, nx, ny, nz, amu, dx, dy, dz) != EXIT_SUCCESS)
        {
            fprintf(stdout, "Error, could not save densities in iteration %d of %d", it + 1, niter);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        update_potentials(icoul, isospin, &pots, &dens_p, &dens_n, dens, &cc_edf, e_cut, dens->nstart, dens->nstop, gr_comm, nx, ny, nz, hbar2m, d1_x, d1_y, d1_z, k1d_x, k1d_y, k1d_z, &lattice_coords, &fftransf_vars, nprot, dxyz, icub);

        if (gr_ip == 0)
        {
            err = 0;

            for (int i = 0; i < nxyz; i++)
            {
                err += std::abs(pots.delta[i] - delta_old[i]);
            }

            fprintf(stdout, "err of pairing_gap%s: %.12le\n", iso_label, err);
        }

        xxpart = *npart * xpart;
        *amu -= const_amu * (xxpart - *npart);

#ifdef CONSTRCALC
#ifdef CONSTR_Q0
        pots.lam[0] += c_q2 * (q2av(dens_p.rho, dens_n.rho, pots.v_constraint, nxyz, nprot, nneut) * dxyz - xconstr[0]);
#endif
        center_dist_pn(dens_p.rho, dens_n.rho, nxyz, &lattice_coords, rcm, rcm + 1, rcm + 2);

        for (int j = 1; j < 4;j++) 
        {
            pots.lam[j] += 3.e-1 * (rcm[j - 1] - xconstr[j]);
        }

        if (ip == 0) 
        {
            fprintf(stdout, "D12=%f fm\n", distMass(dens_p.rho, dens_n.rho, nxyz, 0., lattice_coords.za, lattice_coords.wz, dxyz));

            int j = 0;
            fprintf(stdout, "lam[%d]=%f <O[%d]>=%e %f \n", j, pots.lam[j], j, dxyz * q2av(dens_p.rho, dens_n.rho, pots.v_constraint + j * nxyz, nxyz, nprot, nneut), xconstr[j]);
            
            for (j = 1; j < 4;j++)
                fprintf(stdout, "lam[%d]=%f <O[%d]>=%e %f \n", j, pots.lam[j], j, rcm[j - 1], xconstr[j]);
        }
#endif
        if (ip == 0) 
        {
            sprintf(fn, "dens%s_%1d.dat", iso_label, it % 2);

            fd_dns = fopen(fn, "w");
            for (int i = 0; i < nxyz; i++) 
            {
                if (lattice_coords.xa[i] == 0 && lattice_coords.ya[i] == 0)
                {
                    fprintf(fd_dns, "%f %e\n", lattice_coords.za[i], dens_n.rho[i] + dens_p.rho[i]);
                }
            }

            fclose(fd_dns);
            sprintf(fn, "dens_%1d.dat", it % 2);

            fd_dns = fopen(fn, "w");
            for (int i = 0; i < nxyz; i++) 
            {
                if (lattice_coords.ya[i] == 0.)
                {
                    fprintf(fd_dns, "%f %f %e\n", lattice_coords.xa[i], lattice_coords.za[i], dens_n.rho[i] + dens_p.rho[i]);
                }
            }

            fclose(fd_dns);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (imin == 0)
        {
            mix_potentials(pot_array, pot_array_old, alpha_mix, ishift, m_broy);
        }
        else
        {
            if (ip == root_n) 
            {
                MPI_Send(pot_array + m_broy / 2, m_broy / 2, MPI_DOUBLE, root_p, tag1, MPI_COMM_WORLD);
            }

            if (ip == root_p)
            {
                MPI_Recv(pot_array + m_broy / 2, m_broy / 2, MPI_DOUBLE, root_n, tag1, MPI_COMM_WORLD, &istats);
                i_b = broydenMod_min(pot_array_old, v_in_old, f_old, b_bra, b_ket, diag, pot_array, m_broy, &it_broy, 1. - alpha_mix);
            }

            MPI_Bcast(&i_b, 1, MPI_INT, root_p, MPI_COMM_WORLD);
            MPI_Bcast(pot_array, m_broy, MPI_DOUBLE, root_p, MPI_COMM_WORLD);
        }

#ifdef CONSTRCALC
#ifdef CONSTR_Q0
        pots.lam2[0] = c_q2 * (dxyz * q2av(dens_p.rho, dens_n.rho, pots.v_constraint, nxyz, nprot, nneut) - xconstr[0]) + pots.lam[0];
#endif
        for (j = 1; j < 4;j++) {
            pots.lam2[j] = 3.e-1 * (rcm[j - 1] - xconstr[j]) + pots.lam[j];
        }
#endif

        if (gr_ip == 0)
        {
            sprintf(fn, "out%s.txt", iso_label);
            file_out = fopen(fn, "w");
            fprintf(file_out, "Iteration number %d, mu%s=%12.6f\n", it + 1, iso_label, *amu);
            fprintf(file_out, "N%s = %13.9f\n", iso_label, xpart * (double)*npart);
            fprintf(file_out, "Nwf%s = %d\n", iso_label, nwf);

            if (isospin == 1)
            {
                deform(dens_p.rho, dens_n.rho, nxyz, &lattice_coords, dxyz, file_out);
            }
        }

        system_energy(&cc_edf, icoul, dens, &dens_p, &dens_n, isospin, pots.delta, dens->nstart, dens->nstop, ip, gr_ip, MPI_COMM_WORLD, gr_comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, hbar2m, dxyz, &lattice_coords, &fftransf_vars, nprot, nneut, file_out);

        if (gr_ip == 0)
        {
            fprintf(stdout, "\n Iteration number %d, mu%s=%12.6f\n", it + 1, iso_label, *amu);
            fprintf(stdout, "N%s = %13.9f\n", iso_label, xpart * (double)*npart);
            fprintf(stdout, "Nwf%s = %d\n", iso_label, nwf);
            fprintf(file_out, "      E_qp        Occ       log( | Eqp-Eqp_old | ) \n");

            err = 0;

            for (int i = 2 * nxyz; i < 2 * nxyz + nwf; ++i)
            {
                fprintf(file_out, " %12.8f    %8.6f    %10.6f\n", lam[i], occ[i - 2 * nxyz], log10(fabs(lam[i] - lam_old[i - 2 * nxyz])));
                err += Square(lam[i] - lam_old[i - 2 * nxyz]);
                lam_old[i - 2 * nxyz] = lam[i];
            }

            err = sqrt(err);

            fprintf(stdout, "err%s = %13.9f\n", iso_label, err);
            fprintf(file_out, "err%s = %13.9f\n", iso_label, err);
            printf(" \n");
            
            fclose(file_out);
            sprintf(fn, "pots%s_%1d.cwr", iso_label, it % 2);

            i = write_pots(fn, pot_array, nx, ny, nz, dx, dy, dz, ishift);
        }
    }

    if (icoul == 1)
    {
        fftw_destroy_plan(fftransf_vars.plan_f);
        fftw_destroy_plan(fftransf_vars.plan_b);
        Free(fftransf_vars.buff);
    }

    Free(ham);
    Free(lam_old);
    Free(k1d_x);
    Free(k1d_y);
    Free(k1d_z);
    Free(d1_x); 
    Free(d1_y); 
    Free(d1_z);
    Free(ex_array_n); 
    Free(ex_array_p);

    /* write the info for the TD code */
    double amu_n;
    if (ip == root_n)
    {
        nwf_n = nwf;
        amu_n = *amu;
    }

    MPI_Bcast(&nwf_n, 1, MPI_INT, root_n, MPI_COMM_WORLD);
    MPI_Bcast(&amu_n, 1, MPI_DOUBLE, root_n, MPI_COMM_WORLD);

    if (ip == 0)
    {
        if (imin == 1)
        {
            free(f_old); free(b_bra); free(b_ket); free(diag); free(v_in_old);
        }
        if (iprint_wf == 1)
            nwf = (int)nprot;
        else if (iprint_wf == 2)
            nwf_n = nneut;
        else if (iprint_wf == 3) {
            nwf = (int)nprot;
            nwf_n = (int)nneut;
        }

        sprintf(fn, "info.slda_solver");
        file_out = fopen(fn, "wb");
        fwrite(&nwf, sizeof(int), 1, file_out);
        fwrite(&nwf_n, sizeof(int), 1, file_out);
        fwrite(amu, sizeof(double), 1, file_out);
        fwrite(&amu_n, sizeof(double), 1, file_out);
        fwrite(&dx, sizeof(double), 1, file_out);
        fwrite(&dy, sizeof(double), 1, file_out);
        fwrite(&dz, sizeof(double), 1, file_out);
        fwrite(&nx, sizeof(int), 1, file_out);
        fwrite(&ny, sizeof(int), 1, file_out);
        fwrite(&nz, sizeof(int), 1, file_out);
        fwrite(&e_cut, sizeof(double), 1, file_out);
#ifdef CONSTRCALC
        fwrite(pots.lam2, sizeof(double), 4, file_out);
#endif
        fclose(file_out);
    }

    free(pot_array); 
    free(pot_array_old);

    double* qpe;
    assert(qpe = (double*)malloc(nwf * sizeof(double)));

    if (iprint_wf > -1)
    {
        sprintf(fn, "wf%s.cwr", iso_label);
        sprintf(fn_, "%s.lstr", fn);
        //      printf( "iam[ %d ] fn[ %s ] fn_[ %s ]\n" , gr_ip , fn , fn_ ) ;

        if (iprint_wf == 0)
        {
            if (gr_ip == 0) printf("entering MPI only write\n");
            bc_wr_mpi(fn, gr_comm, p_proc, q_proc, i_p, i_q, nb, 2 * nxyz, 4 * nxyz, 1, 4 * nxyz, z_eig);
        }
        else
            if (iprint_wf == 1) 
            {
                if (isospin == 1)
                    print_wf2(fn, gr_comm, lam, z_eig, gr_ip, nwf, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc, nx, ny, nz, e_cut, occ, icub);
                else {
                    if (gr_ip == 0) printf("entering MPI-only write\n");
                    bc_wr_mpi(fn, gr_comm, p_proc, q_proc, i_p, i_q, nb, 2 * nxyz, 4 * nxyz, 1, 4 * nxyz, z_eig);
                }
            }

            else if (iprint_wf == 2) 
            {
                if (isospin == -1)
                    print_wf2(fn, gr_comm, lam, z_eig, gr_ip, nwf, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc, nx, ny, nz, e_cut, occ, icub);
                else {
                    if (gr_ip == 0) printf("entering MPI-only write\n");
                    bc_wr_mpi(fn, gr_comm, p_proc, q_proc, i_p, i_q, nb, 2 * nxyz, 4 * nxyz, 1, 4 * nxyz, z_eig);
                }
            }
            else if (iprint_wf == 3)
            {
                print_wf2(fn, gr_comm, lam, z_eig, gr_ip, nwf, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc, nx, ny, nz, e_cut, occ, icub);

            }
    }
    else if (ip == 0)
        printf("wave functions not saved, iprint=%d\n", iprint_wf);


    // save positive qpe
    sprintf(fn, "qpe%s.cwr", iso_label);
    if (write_qpe(fn, qpe, gr_comm, gr_ip, nwf) != EXIT_SUCCESS)

    {

        fprintf(stdout, "Error, could not save qpe at the end of iterations. \n");

        MPI_Finalize();

        return(EXIT_FAILURE);

    }

    Free(z_eig);
    Free(lam); 
    Free(occ);

    destroy_mpi_groups(&group_comm, &gr_comm);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

int minim_int(const int i1, const int i2)
{
    if (i1 < i2)
        return i1;
    else
        return i2;
}

void print_help(const int ip, char** argv)
{
    if (ip == 0)
    {
        printf(" usage: %s --options \n", argv[0]);
        printf("       --nx Nx                  Nx **required** the number of points on the x direction\n");
        printf("       --ny Ny                  Ny **required** the number of points on the y direction\n");
        printf("       --nz Nz                  Nz **required** the number of points on the z direction\n");
        printf("       --nprot np or -Z np       number of protons **required** \n");
        printf("       --nneut nn or -N nn       number of neutrons **required** \n");
        printf("       --Lx Lx                  Lx: x-lattice length **required** \n");
        printf("       --Ly Ly                  Ly: y-lattice length **required** \n");
        printf("       --Lz Lz                  Lz: z-lattice length **required** \n");
        printf("       --nopairing              pairing coupling set to 0; default with pairing; note that the wfs will still have u and v components \n");
        printf("       --broyden or -b          Broyden mixing will be performed\n");
        printf("       --alpha_mix alpha or -a alpha \n");
        printf("                                the mixing parameter; default alpha=0.25 \n");
        printf("       --nocoulomb              Coulomb interaction between protons set to zero; by default, Coulomb is included \n");
        printf("       --niter n                 number of iterations\n");
        printf("       --irun n                  restart option: n = 0 start with guess for densities (default), n = 1 restart reading previously computed densities , n=2 read previously computed potentials \n");
        printf("       --iprint_wf n             printing options for the wave functions: n=-1 (default) no priting, n=0 all wfs printed, n=1 only Z proton wave functions saved, n=2 only N neutron wave functions saved \n");
        printf("       --hw hbaromega            option for a HO potential hbaromega=2 MeV by default\n");
        printf("       --ecut e_cut              the energy cut in MeV, default e_cut = 75 MeV  \n");
        printf("       --pproc p_proc            number of processors on the grid, default sqrt( total number of processors / 2 )  \n");
        printf("       --qproc q_proc            number of processors on the grid, default sqrt( total number of processors / 2 )  \n");
        printf("       --mb mb                   size of the block (mb=nb), default mb=40 \n");
        printf("       --nb nb                   size of the block (mb=nb), default nb=40 \n");
        printf("       --hbo hw                  hw , default hw=2 MeV\n");
        printf("       --resc_dens               if set, the densities will be scalled to the correct number of particles; nonscalling is the default \n");
        printf("       --deformation n           if n = 0 (default) spherical densities , n = 2 axially symmetrix densities , n = triaxial densities \n");
        printf("                                 this is meaningful only if irun = 0, otherwize will be ignored \n");
    }
}

int parse_input_file(char* file_name)
{
    FILE* fp;
    fp = fopen(file_name, "r");
    if (fp == NULL)
        return 0;

    int i;

    char s[MAX_REC_LEN];
    char tag[MAX_REC_LEN];
    char ptag[MAX_REC_LEN];

    while (fgets(s, MAX_REC_LEN, fp) != NULL)
    {
        // Read first element of line
        tag[0] = '#'; tag[1] = '\0';
        sscanf(s, "%s %*s", tag);

        // Loop over known tags;
        if (strcmp(tag, "#") == 0)
            continue;
        else if (strcmp(tag, "nx") == 0)
            sscanf(s, "%s %d %*s", tag, &md.nx);

        else if (strcmp(tag, "ny") == 0)
            sscanf(s, "%s %d %*s", tag, &md.ny);

        else if (strcmp(tag, "nz") == 0)
            sscanf(s, "%s %d %*s", tag, &md.nz);

        else if (strcmp(tag, "dx") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.dx);

        else if (strcmp(tag, "dy") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.dy);

        else if (strcmp(tag, "dz") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.dz);

        else if (strcmp(tag, "broyden") == 0)
            sscanf(s, "%s %d %*s", tag, &md.broyden);

        else if (strcmp(tag, "Z") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.nprot);

        else if (strcmp(tag, "N") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.nneut);

        else if (strcmp(tag, "niter") == 0)
            sscanf(s, "%s %d %*s", tag, &md.niter);

        else if (strcmp(tag, "iext") == 0)
            sscanf(s, "%s %d %*s", tag, &md.iext);

        else if (strcmp(tag, "force") == 0)
            sscanf(s, "%s %d %*s", tag, &md.force);

        else if (strcmp(tag, "pairing") == 0)
            sscanf(s, "%s %d %*s", tag, &md.pairing);

        else if (strcmp(tag, "print_wf") == 0)
            sscanf(s, "%s %d %*s", tag, &md.iprint_wf);

        else if (strcmp(tag, "alpha_mix") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.alpha_mix);

        else if (strcmp(tag, "ecut") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.ecut);
        //	else if (strcmp (tag,"icub") == 0)
        //	  sscanf (s,"%s %d %*s",tag,&md.icub);
        else if (strcmp(tag, "imass") == 0)
            sscanf(s, "%s %d %*s", tag, &md.imass);

        else if (strcmp(tag, "icm") == 0)
            sscanf(s, "%s %d %*s", tag, &md.icm);

        else if (strcmp(tag, "irun") == 0)
            sscanf(s, "%s %d %*s", tag, &md.irun);

        else if (strcmp(tag, "isymm") == 0)
            sscanf(s, "%s %d %*s", tag, &md.isymm);

        else if (strcmp(tag, "deform") == 0)
            sscanf(s, "%s %d %*s", tag, &md.deformation);

        else if (strcmp(tag, "q0") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.q0);

        else if (strcmp(tag, "v0") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.v0);

        else if (strcmp(tag, "z0") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.z0);

        else if (strcmp(tag, "wneck") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.wneck);

        else if (strcmp(tag, "rneck") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.rneck);

        else if (strcmp(tag, "resc_dens") == 0)
            sscanf(s, "%s %d %*s", tag, &md.resc_dens);

        else if (strcmp(tag, "p") == 0)
            sscanf(s, "%s %d %*s", tag, &md.p);

        else if (strcmp(tag, "q") == 0)
            sscanf(s, "%s %d %*s", tag, &md.q);

        else if (strcmp(tag, "mb") == 0)
            sscanf(s, "%s %d %*s", tag, &md.mb);

        else if (strcmp(tag, "nb") == 0)
            sscanf(s, "%s %d %*s", tag, &md.nb);

        else if (strcmp(tag, "ggp") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.ggp);

        else if (strcmp(tag, "ggn") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.ggn);
        else if (strcmp(tag, "alpha_pairing") == 0)
            sscanf(s, "%s %lf %*s", tag, &md.alpha_pairing);
    }

    fclose(fp);
    return 1;
}

int readcmd(int argc, char* argv[], int ip)
{
    static const char* optString = "h";
    int opt = 0;

    do
    {
        opt = getopt(argc, argv, optString);
        switch (opt)
        {
        case 'h':
            print_help(ip, argv);
            break;

        default:
            break;
        }
    } while (opt != -1);

    if (optind < argc)
        return optind;
    else
        return -1;
}
