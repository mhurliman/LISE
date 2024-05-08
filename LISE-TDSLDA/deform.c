// for license information, see the accompanying LICENSE file
/*
compute the deformation parameters beta and gamma
gamma not computed yet
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vars.h"
#include <mpi.h>
#include "tdslda_func.h"

#define PI 3.141592653589793238462643383279502884197

double deform(double* rho, int nxyz, Lattice_arrays* latt_coords, double dxyz)
{
    double q[3][3];

    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
            q[i][j] = 0.;

    double xcm, ycm, zcm;
    double num = center_dist(rho, nxyz, latt_coords, &xcm, &ycm, &zcm);
    double r2 = 0;

    for (int i = 0; i < nxyz; ++i)
    {
        double xa = latt_coords->xa[i] - xcm;
        double ya = latt_coords->ya[i] - ycm;
        double za = latt_coords->za[i] - zcm;

        r2 += (xa * xa + ya * ya + za * za) * rho[i];
        q[2][2] += ((-xa * xa - ya * ya + 2. * za * za) * rho[i]);
    }

    double beta = sqrt(PI / 5.) * q[2][2] / r2;
    return beta;
}
