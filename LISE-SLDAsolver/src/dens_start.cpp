// for license information, see the accompanying LICENSE file
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"

void dens_startTheta(double A_mass, double npart, int nxyz, Densities* dens, const Lattice_arrays* lattice_coords, double dxyz, double lx, double ly, double lz, int ideform)
{
    /* one option to start the code with a normal/anomalous spherical density */
    double a = 1.8;

    double r0 = 1.12 * pow(A_mass, 0.33333334);
    double rz = r0;

    double rx, ry;
    switch (ideform)
    {
    case 0:
        rx = r0;
        ry = r0;
        break;

    case 2:
        rx = 0.8 * r0;
        ry = rx;
        break;

    case -2:
        rx = 1.2 * r0;
        ry = rx;
        break;

    case 3:
        rz = 1.2 * r0;
        ry = r0;
        rx = 0.8 * r0;
        break;

    default:
        printf("this option does not exist, switching to spherical densities\n");
        rx = rz;
        ry = rz;
        break;
    }

    for (int i = 0; i < nxyz; i++)
    {
        double xa = (fabs(lattice_coords->xa[i]) - rx) / a;
        double ya = (fabs(lattice_coords->ya[i]) - ry) / a;
        double za = (fabs(lattice_coords->za[i]) - rz) / a;

        dens->rho[i] = 0.08 * .125 * (1. - erf(xa)) * (1. - erf(ya)) * (1. - erf(za));
    }

    for (int i = dens->nstart; i < dens->nstop; i++) 
    {
        dens->nu[i - dens->nstart] = 0.02 * dens->rho[i];
    }
}


void dens_gauss(int A_mass, int npart, int nxyz, Densities* dens, const Lattice_arrays* lattice_coords, double dxyz, double lx, double ly, double lz, int ideform)
{
    /* one option to start the code with a normal/anomalous spherical density */
    double r0 = lx / 5.0 /* 1. * pow( ( double ) A_mass , 1./3. ) */;

    double r01, r02;
    if (ideform == 2 || ideform == 3)
    {
        r01 = 1.4 * r0;

        if (ideform == 3)
            r02 = 0.8 * r0;
        else
            r02 = r0;
    }
    else
    {
        r01 = r0;
        r02 = r0;
    }

    double apart = 0;
    for (int i = 0; i < nxyz; i++)
    {
        double xa = lattice_coords->xa[i];
        double ya = lattice_coords->ya[i];
        double za = lattice_coords->za[i];

        dens->rho[i] = 0.08 * (exp(-pow(xa / r0, 2.)) + exp(-pow((xa - lx) / r0, 2.)) + exp(-pow((xa + lx) / r0, 2.)) + exp(-pow((xa - 2. * lx) / r0, 2.)) + exp(-pow((xa + 2. * lx) / r0, 2.)) + exp(-pow((xa - 3. * lx) / r0, 2.)) + exp(-pow((xa + 3. * lx) / r0, 2.)))
            * (exp(-pow(ya / r02, 2.)) + exp(-pow((ya - ly) / r02, 2.)) + exp(-pow((ya + ly) / r02, 2.)) + exp(-pow((ya - 2. * ly) / r02, 2.)) + exp(-pow((ya + 2. * ly) / r02, 2.)) + exp(-pow((ya - 3. * ly) / r02, 2.)) + exp(-pow((ya + 3. * ly) / r02, 2.)))
            * (exp(-pow(za / r01, 2.)) + exp(-pow((za - lz) / r01, 2.)) + exp(-pow((za + lz) / r01, 2.)) + exp(-pow((za - 2. * lz) / r01, 2.)) + exp(-pow((za + 2. * lz) / r01, 2.)) + exp(-pow((za - 3. * lz) / r01, 2.)) + exp(-pow((za + 3. * lz) / r01, 2.)));
        
        apart += dens->rho[i];
    }

    apart = apart * dxyz / ((double)npart);

    for (int i = 0; i < nxyz; ++i) 
    {
        dens->rho[i] *= 0.08;
    }

    for (int i = dens->nstart; i < dens->nstop; i++) 
    {
        dens->nu[i - dens->nstart] = 0.1 * dens->rho[i];
    }
}
