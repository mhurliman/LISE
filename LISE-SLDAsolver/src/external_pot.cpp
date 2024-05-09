
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>

#include "common.h"
#include "operators.h"

void external_pot(int iext, int n, int n_part, double hbo, double* v_ext, complex* delta_ext, double hbar2m, Lattice_arrays* lattice_coords, double rr, double rneck, double wneck, double z0, double v0)
{
    /* iext: in, choice for external potentials */
    /* n : in , Nxyz */
    /* N_particles: in , total number of particles, needed for some options */
    /* hbo: in , real parameter, hbar omega for the HO choice */
    /* v_ext: out external potential */
    /* delta_ext: out , external pairing field */
    double hw;

    /* iext = 0 : no external field */
    if (iext == 1)  /* HO */
    {
        hw = 0.25 * hbo * hbo / hbar2m;

        for (int i = 0; i < n; i++)
        {
            v_ext[i] = hw * (Square(lattice_coords->xa[i]) + Square(lattice_coords->ya[i]) + Square(lattice_coords->za[i]));
        }
    }

    if (iext == 2) /* External Woods-Saxon potential, proportional external pairing */
    {
        hw = n_part;
        hw = 1.2 * pow(hw, 1. / 3.);
        for (int i = 0; i < n; i++)
        {
            v_ext[i] = -50.0 / (1. + exp(sqrt(Square(lattice_coords->xa[i]) + Square(lattice_coords->ya[i]) + Square(lattice_coords->za[i]))));
            delta_ext[i] = -hbo * v_ext[i];
        }
    }

    if (iext == 3)  /* Constant pairing */
    {
        for (int i = 0; i < n; i++)
        {
            delta_ext[i] = hbo;
        }
    }

    if (iext == 4 || iext == 5)
    {
        for (int i = 0; i < n; i++)
        {
            double r1 = sqrt(Square(lattice_coords->xa[i]) + Square(lattice_coords->ya[i]) + Square(lattice_coords->za[i] + 7.5));
            double r2 = sqrt(Square(lattice_coords->xa[i]) + Square(lattice_coords->ya[i]) + Square(lattice_coords->za[i] - 7.5));

            v_ext[i] = -50.0 / (1.0 + exp(-2.0) * cosh(r1)) - 50.0 / (1.0 + exp(-2.0) * cosh(r2));

            if (iext == 5)
                delta_ext[i] = 0.05 * v_ext[i];
        }
    }

    if (iext == 6)
    {
        hw = 0.25 * hbo * hbo / hbar2m;
        
        for (int i = 0; i < n; i++)
        {
            v_ext[i] = hw * (Square(lattice_coords->xa[i]) + Square(lattice_coords->ya[i]) + Square(lattice_coords->za[i]));
            delta_ext[i] = 0.1 * hbo;
        }
    }

    if (iext == 7)  /* HO */
    {
        hw = 0.25 * hbo * hbo / hbar2m;

        for (int i = 0; i < n; i++)
            v_ext[i] = hw * (Square(lattice_coords->xa[i]) + Square(lattice_coords->ya[i]) + 0.5 * Square(lattice_coords->za[i]));
    }

    if (iext == 8)  /* HO */
    {
        hw = 0.25 * hbo * hbo / hbar2m;

        for (int i = 0; i < n; i++)
        {
            if (fabs(lattice_coords->xa[i]) > rr)
                v_ext[i] = hw * Square(fabs(lattice_coords->xa[i]) - rr);

            if (fabs(lattice_coords->ya[i]) > rr)
                v_ext[i] += hw * Square(fabs(lattice_coords->ya[i]) - rr);

            if (fabs(lattice_coords->za[i]) > rr)
                v_ext[i] += 0.5 * hw * Square(fabs(lattice_coords->za[i]) - rr);
        }
    }

    if (iext == 9) 
    {
        hw = n_part;
        hw = 1.2 * pow(hw, 1. / 3.);

        for (int i = 0; i < n; i++) 
        {
            delta_ext[i] = hbo / (1.0 + exp((sqrt(Square(lattice_coords->xa[i]) + Square(lattice_coords->ya[i]) + Square(lattice_coords->za[i])) - hw) / 2.0));
        }
    }

    if (iext == 60) 
    {
        double aa = 0.65;
        double w = wneck;
        double amp = (1. + exp((z0 - w) / aa)) * (1. + exp(-(z0 + w) / aa));
        double kk = 1. + exp(-rneck / aa);

        for (int i = 0; i < n; i++)
        {
            double rho = sqrt(lattice_coords->xa[i] * lattice_coords->xa[i] + lattice_coords->ya[i] * lattice_coords->ya[i]);
            double z = lattice_coords->za[i] - z0;
            
            v_ext[i] = amp * v0 * (1.0 - kk / ((1.0 + exp((rho - rneck) / aa)))) / ((1.0 + exp(-(z + w) / aa)) * (1.0 + exp((z - w) / aa)));
        }
    }

    if (iext == 61)
    {
        for (int i = 0; i < n; i++)
        {
            double z = lattice_coords->za[i];
            v_ext[i] = v0 * z * z * z;
        }
    }
}

void external_so_m(const double* v_ext, double* wx, double* wy, double* wz, MPI_Comm gr_comm, complex* d1_x, complex* d1_y, complex* d1_z, int nx, int ny, int nz)
{
    double mass = 939.565346;
    double hbarc = 197.3269631;

    int nxyz = nx * ny * nz;
    double lambda = 2.5 * Square(0.5 * hbarc / mass); /* half the strength */
    gradient_real(v_ext, nxyz, wx, wy, wz, 0, nxyz, gr_comm, d1_x, d1_y, d1_z, nx, ny, nz, 0);

    for (int i = 0; i < nxyz; i++)
    {
        wx[i] *= lambda;
        wy[i] *= lambda;
        wz[i] *= lambda;
    }
}

// make filter for constraining field
void make_filter(double* filter, Lattice_arrays* latt_coords, int nxyz)
{
    double beta = 1.5; 
    double aa = 1.0; 
    double R0 = 10.0;
    
    for (int i = 0; i < nxyz; i++)
    {
        double xa = latt_coords->xa[i];
        double ya = latt_coords->ya[i];
        double za = latt_coords->za[i];

        double rr = sqrt(Square(xa) + Square(ya) + Square(za / beta));
        filter[i] = 1.0 / (1.0 + exp((rr - R0) / aa));
    }
}
