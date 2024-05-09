// for license information, see the accompanying LICENSE file

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <mpi.h>

#include "common.h"
#include "operators.h"

void allocate_dens(Densities* dens, int ip, int np, int nxyz)
{
    divide_work(nxyz, ip, np, &dens->nstart, &dens->nstop);

    int iwork = dens->nstop - dens->nstart;
    if (iwork > 0)
    {
        dens->jjx = Allocate<double>(iwork);
        dens->jjy = Allocate<double>(iwork);
        dens->jjz = Allocate<double>(iwork);

        dens->tau = AllocateZeroed<double>(iwork);
        dens->nu = AllocateZeroed<complex>(iwork);
        dens->divjj = AllocateZeroed<double>(iwork);
    }

    dens->rho = AllocateZeroed<double>(nxyz);
}

void mem_share(Densities* dens, double* ex_array, int nxyz, int idim)
{
    memset(ex_array, 0, idim * sizeof(ex_array[0]));

    dens->rho = ex_array;

    if (idim > nxyz) 
    {
        dens->divjj = ex_array + nxyz;
        dens->tau = ex_array + (nxyz + dens->nstop - dens->nstart);
    }
}

/* function to generate the 1D kinetic energy */
void generate_ke_1d(double* ke, int n, double a, int iopt_der)
{
    double pi2 = pow(PI / a, 2);
    double xn = (double)n;
    double xn2 = xn * xn;
    double constant = 2.0 * pi2 / xn2;

    if (iopt_der == 0)
        *ke = pi2 * (1.0 - 1.0 / xn2) / 3.0;   /* Baye */
    else
        *ke = pi2 * (1.0 + 2.0 / xn2) / 3.0;

    double isign = -1.0;
    for (int i = 1; i < n; i++)
    {
        double angle = (double)i * PI / xn;
        ke[i] = isign * constant / (Square(sin(angle)));

        if (iopt_der == 0)
        {
            ke[i] = ke[i] * cos(angle); /* Baye */
        }

        isign = -isign;
    }
}

/* Generate the 1D derivative */
void generate_der_1d(complex* der, int n, double a, int iopt_der)
{
    double c1 = 0;  /* remove the highest component for c1=0 */

    double xn = (double)n;
    int n1 = n - 1;
    double c = PI / xn;

    /* Note: the order in the derivative array is: -( N - 1 ) cooresponding to der[0],
       -(N-1) + 1 corresponds to der[1] , and so forth */
    if (iopt_der == 0)
        der[n1] = 0;
    else
        der[n1] = -c * c1 * 1i / a;

    double isign = -1.0 / a;

    for (int i = 1; i < n; i++)
    {
        double angle = i * c;

        if (iopt_der == 0)
            der[n1 + i] = isign * c / sin(angle);
        else
            der[n1 + i] = isign * c * (1.0 / tan(angle) - c1 * 1i);

        isign = -isign;
        der[n1 - i] = -conj(der[n1 + i]);
    }
}

void make_coordinates(int nxyz, int nx, int ny, int nz, double dx, double dy, double dz, Lattice_arrays* lattice_vars)
{
    lattice_vars->wx = Allocate<int>(nxyz);
    lattice_vars->wy = Allocate<int>(nxyz);
    lattice_vars->wz = Allocate<int>(nxyz);

    lattice_vars->xa = Allocate<double>(nxyz);
    lattice_vars->ya = Allocate<double>(nxyz);
    lattice_vars->za = Allocate<double>(nxyz);
    lattice_vars->kx = Allocate<double>(nxyz);
    lattice_vars->ky = Allocate<double>(nxyz);
    lattice_vars->kz = Allocate<double>(nxyz);
    lattice_vars->kin = Allocate<double>(nxyz);

    double* xx = AllocateFilled<double>(nx, 1.0);
    double* yy = AllocateFilled<double>(ny, 1.0);
    double* zz = AllocateFilled<double>(nz, 1.0);

    xx[nx / 2] = 0;
    yy[ny / 2] = 0;
    zz[nz / 2] = 0;

    grid3(xx, yy, zz, nx, ny, nz, lattice_vars->xa, lattice_vars->ya, lattice_vars->za);

    for (int i = 0; i < nxyz; i++) 
    {
        lattice_vars->wx[i] = static_cast<double>(lattice_vars->xa[i]);
        lattice_vars->wy[i] = static_cast<double>(lattice_vars->ya[i]);
        lattice_vars->wz[i] = static_cast<double>(lattice_vars->za[i]);
    }

    double xn = static_cast<double>(nx) / 2.0;
    for (int i = 0; i < nx; i++)
        xx[i] = (static_cast<double>(i) - xn) * dx;

    xn = ((double)ny) / 2.0;
    for (int i = 0; i < ny; i++)
        yy[i] = (static_cast<double>(i) - xn) * dy;

    xn = ((double)nz) / 2.0;
    for (int i = 0; i < nz; i++)
        zz[i] = (static_cast<double>(i) - xn) * dz;

    grid3(xx, yy, zz, nx, ny, nz, lattice_vars->xa, lattice_vars->ya, lattice_vars->za);

    /* set also the momentum space */
    xn = PI / (static_cast<double>(nx) * dx);
    for (int i = 0; i < nx / 2; i++)
    {
        xx[i] = (double)(2 * i) * xn;
        xx[i + nx / 2] = (double)(2 * i - nx) * xn;
    }

    xn = PI / (static_cast<double>(ny) * dy);
    for (int i = 0; i < ny / 2; i++)
    {
        yy[i] = static_cast<double>(2 * i) * xn;
        yy[i + ny / 2] = static_cast<double>(2 * i - ny) * xn;
    }

    xn = PI / static_cast<double>(nz * dz);
    for (int i = 0; i < nz / 2; i++)
    {
        zz[i] = static_cast<double>(2 * i) * xn;
        zz[i + nz / 2] = static_cast<double>(2 * i - nz) * xn;
    }

    grid3(xx, yy, zz, nx, ny, nz, lattice_vars->kx, lattice_vars->ky, lattice_vars->kz);

    for (int i = 0; i < nxyz; i++)
    {
        lattice_vars->kin[i] = Square(lattice_vars->kx[i]) + Square(lattice_vars->ky[i]) + Square(lattice_vars->kz[i]);
    }

    Free(xx);
    Free(yy);
    Free(zz);
}

void compute_densities(
    const double* lam, 
    const complex* z, 
    int nxyz, 
    int ip, 
    MPI_Comm comm, 
    Densities* dens, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc, 
    int nx, int ny, 
    int nz, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    double e_max, 
    int* nwf, 
    double* occ, 
    int icub
)
{
    /*
     * Computes the particle, anomalous and current densities from the eigenvectors z
     */
    int nstart = dens->nstart;
    int nstop = dens->nstop;
    int iwork = nstop - nstart;

    int n = 4 * nxyz;
    int nn = 3 * nxyz;
    int nhalf = 2 * nxyz;

    // laplacian of up and down components 
    complex* lapl_u = Allocate<complex>(nxyz); // Asserting the laplacean elements.
    complex* lapl_d = Allocate<complex>(nxyz);

    complex* dx_d, *dy_d, *dz_d, *dx_u, *dy_u, *dz_u;
    if (iwork > 0) 
    {
        dx_d = Allocate<complex>(iwork);
        dy_d = Allocate<complex>(iwork);
        dz_d = Allocate<complex>(iwork);

        dx_u = Allocate<complex>(iwork);
        dy_u = Allocate<complex>(iwork);
        dz_u = Allocate<complex>(iwork);
    }

    double* occ1 = AllocateZeroed<double>(nhalf);
    double* rho = AllocateZeroed<double>(nxyz);

    complex* vec = Allocate<complex>(nn);
    complex* vec1 = Allocate<complex>(n);

    ZeroMemory(dens->tau, iwork);
    ZeroMemory(dens->divjj, iwork);
    ZeroMemory(dens->nu, iwork);

    *nwf = 0;

    double f1 = 1.0;
    for (int jj = nhalf; jj < n; jj++)
    {
        /* construct one vector at a time for positive eigenvalues */
        f1 = sqrt(factor_ec(lam[jj], e_max, icub));

        if (f1 < 1e-6)
            break;

        memset(vec1, 0, n * sizeof(vec1[0]));

#pragma omp parallel for default(shared) private(li,ii,lj,j) 
        for (int lj = 0; lj < n_iq; lj++)
        {
            int j = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;

            if (j == jj)
            {
                for (int li = 0; li < m_ip; li++)
                {
                    int ii = i_p * mb + (li / mb) * p_proc * mb + li % mb;
                    vec1[ii] = z[lj * m_ip + li] * f1;
                }
            }
        }

        MPI_Allreduce(vec1 + nxyz, vec, nn, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
        gradient_ud(vec + nxyz, nxyz, dx_u, dy_u, dz_u, dx_d, dy_d, dz_d, nstart, nstop, comm, d1_x, d1_y, d1_z, nx, ny, nz);

        // Calculating Laplacians.
        laplacean_complex(vec + nxyz, nxyz, lapl_u, nstart, nstop, comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, 0);
        laplacean_complex(vec + 2 * nxyz, nxyz, lapl_d, nstart, nstop, comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, 0);

#pragma omp parallel for default(shared) private(i,iu,id,ii) 
        for (int i = nstart; i < nstop; i++)
        {
            int ii = i - nstart;
            int iu = i + nxyz;
            int id = iu + nxyz;

            occ1[*nwf] += (Square(std::abs(vec[iu])) + Square(std::abs(vec[id])));
            rho[i] += (Square(std::abs(vec[iu])) + Square(std::abs(vec[id])));

            dens->tau[ii] -= std::real(conj(vec[iu]) * lapl_u[i] + conj(vec[id]) * lapl_d[i]);
            dens->nu[ii] -= (conj(vec[i]) * vec[iu]);
            dens->divjj[ii] -= (std::imag(dy_u[ii] * conj(dx_u[ii]) - dy_d[ii] * conj(dx_d[ii]) + dz_u[ii] * conj(dy_d[ii]) - dy_u[ii] * conj(dz_d[ii])) + std::real(dx_d[ii] * conj(dz_u[ii]) - dx_u[ii] * conj(dz_d[ii])));
        }

        (*nwf)++;
    }

    free(vec); 
    free(vec1);

    if (iwork > 0) 
    { 
        free(dx_d); 
        free(dy_d); 
        free(dz_d); 
        free(dx_u); 
        free(dy_u); 
        free(dz_u); 
    }

    MPI_Allreduce(occ1, occ, nhalf, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(rho, dens->rho, nxyz, MPI_DOUBLE, MPI_SUM, comm);

    free(lapl_u); 
    free(lapl_d);
    free(occ1);

    laplacean(dens->rho, nxyz, rho, 0, nxyz, comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, 0); // Now buffer rho contains lap rho.

    for (int i = 0; i < nstop - nstart; i++)
    {
        dens->tau[i] += 0.5 * rho[i + nstart];
        dens->divjj[i] *= 2.0;
    }

    free(rho);
}

void compute_densities_finitetemp(
    const double* lam, 
    const complex* z, 
    int nxyz, 
    int ip, 
    MPI_Comm comm, 
    Densities* dens, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc, 
    int nx, int ny, int nz, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    double e_max, 
    double temp, 
    int* nwf, 
    double* occ,
    int icub
)
{
    /*
     * Computes the particle, anomalous and current densities from the eigenvectors z in finite temperature case
     */
    int nstart = dens->nstart;
    int nstop = dens->nstop;
    int iwork = nstop - nstart;
    int n = 4 * nxyz;
    int nn = 3 * nxyz;
    int nhalf = 2 * nxyz;

    complex* dx_d, *dy_d, *dz_d, *dx_u, *dy_u, *dz_u;
    if (iwork > 0) 
    {
        dx_d = Allocate<complex>(iwork);
        dy_d = Allocate<complex>(iwork);
        dz_d = Allocate<complex>(iwork);

        dx_u = Allocate<complex>(iwork);
        dy_u = Allocate<complex>(iwork);
        dz_u = Allocate<complex>(iwork);
    }
    

    double* occ1 = AllocateZeroed<double>(nhalf);
    double* rho = AllocateZeroed<double>(nxyz);

    complex* vec = Allocate<complex>(nn);
    complex* vec1 = Allocate<complex>(n);

    ZeroMemory(dens->tau, iwork);
    ZeroMemory(dens->divjj, iwork);
    ZeroMemory(dens->nu, iwork);

    *nwf = 0;

    for (int jj = nhalf; jj < n; jj++)
    {
        /* construct one vector at a time for positive eigenvalues */
        double f1 = sqrt(factor_ec(lam[jj], e_max, icub));

        if (f1 < 1e-6)
        {
            break;
        }

        double ft = 1.0 / (1.0 + exp(-1.0 * lam[jj] / temp));

        memset(vec1, 0, n * sizeof(vec1[0]));

#pragma omp parallel for default(shared) private(li,ii,lj,j) 
        for (int lj = 0; lj < n_iq; lj++)
        {
            int j = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;

            if (j == jj)
                for (int li = 0; li < m_ip; li++)
                {
                    int ii = i_p * mb + (li / mb) * p_proc * mb + li % mb;
                    vec1[ii] = z[lj * m_ip + li] * f1;
                }
        }

        // finite temperature part. I
        MPI_Allreduce(vec1 + nxyz, vec, nn, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
        gradient_ud(vec + nxyz, nxyz, dx_u, dy_u, dz_u, dx_d, dy_d, dz_d, nstart, nstop, comm, d1_x, d1_y, d1_z, nx, ny, nz);

#pragma omp parallel for default(shared) private(i,iu,id,ii) 
        for (int i = nstart; i < nstop; i++)
        {
            int ii = i - nstart;
            int iu = i + nxyz;  // v - up component
            int id = iu + nxyz;   // v - down component 

            occ1[*nwf] += (Square(std::abs(vec[iu])) + Square(std::abs(vec[id]))) * ft;
            rho[i] += (Square(std::abs(vec[iu])) + Square(std::abs(vec[id]))) * ft;

            dens->tau[ii] += (Square(std::abs(dx_d[ii])) + Square(std::abs(dy_d[ii])) + Square(std::abs(dz_d[ii])) + Square(std::abs(dx_u[ii])) + Square(std::abs(dy_u[ii])) + Square(std::abs(dz_u[ii]))) * ft;
            dens->nu[ii] -= (conj(vec[i]) * vec[iu]) * (2 * ft - 1);
            dens->divjj[ii] -= (std::imag(dy_u[ii] * conj(dx_u[ii]) - dy_d[ii] * conj(dx_d[ii]) + dz_u[ii] * conj(dy_d[ii]) - dy_u[ii] * conj(dz_d[ii])) + std::real(dx_d[ii] * conj(dz_u[ii]) - dx_u[ii] * conj(dz_d[ii]))) * ft;
        }

        // finite temperature part. II
        MPI_Allreduce(vec1, vec, nn, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
        gradient_ud(vec, nxyz, dx_u, dy_u, dz_u, dx_d, dy_d, dz_d, nstart, nstop, comm, d1_x, d1_y, d1_z, nx, ny, nz);

#pragma omp parallel for default(shared) private(i,iu,id,ii) 
        for (int i = nstart; i < nstop; i++)
        {
            int ii = i - nstart;
            int iu = i;  // u - up component
            int id = iu + nxyz;   // u - down component 
            occ1[*nwf] += (Square(std::abs(vec[iu])) + Square(std::abs(vec[id]))) * (1 - ft);
            rho[i] += (Square(std::abs(vec[iu])) + Square(std::abs(vec[id]))) * (1 - ft);

            dens->tau[ii] += (Square(std::abs(dx_d[ii])) + Square(std::abs(dy_d[ii])) + Square(std::abs(dz_d[ii])) + Square(std::abs(dx_u[ii])) + Square(std::abs(dy_u[ii])) + Square(std::abs(dz_u[ii]))) * (1 - ft);
            dens->divjj[ii] -= (std::imag(dy_u[ii] * conj(dx_u[ii]) - dy_d[ii] * conj(dx_d[ii]) + dz_u[ii] * conj(dy_d[ii]) - dy_u[ii] * conj(dz_d[ii])) + std::real(dx_d[ii] * conj(dz_u[ii]) - dx_u[ii] * conj(dz_d[ii]))) * (1 - ft);
        }

        (*nwf)++;
    }

    free(vec); 
    free(vec1);

    if (iwork > 0) 
    { 
        free(dx_d); 
        free(dy_d); 
        free(dz_d); 
        free(dx_u); 
        free(dy_u); 
        free(dz_u); 
    }

    MPI_Allreduce(occ1, occ, nhalf, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(rho, dens->rho, nxyz, MPI_DOUBLE, MPI_SUM, comm);

    free(rho); 
    free(occ1);

    for (int i = 0; i < nstop - nstart; i++)
    {
        dens->divjj[i] *= 2.0;
    }
}

double factor_ec(double e, double e_cut, int icub)
{
    double e1 = (e - e_cut) / 0.25;

    double fact;
    if (e1 > 40.0)
    {
        fact = 0;
    }
    else if (e1 < -40.0)
    {
        fact = 1.0;
    }
    else
    {
        fact = 1.0 / (1.0 + exp(e1));
    }

    if (icub == 0) 
    {
        // Spherical cutoff.
        fact = e > e_cut ? 0 : 1.0;
    }
    else if (icub == 1) 
    {
        // for cubic-cutoff
        fact = 1.0; 
    }

    return fact;
}

void shift_coords(
    complex* vec, 
    int n, 
    FFtransf_vars* fftransf_vars, 
    const Lattice_arrays* latt_coords, 
    double xcm, double ycm, double zcm
)
{
    for (int i = 0; i < n; i++)
    {
        fftransf_vars->buff[i] = vec[i];
    }

    fftw_execute(fftransf_vars->plan_f);
    for (int i = 0; i < n; i++)
    {
        double xarg = latt_coords->kx[i] * xcm + latt_coords->ky[i] * ycm + latt_coords->kz[i] * zcm;
        fftransf_vars->buff[i] *= (cos(xarg) + 1i * sin(xarg));
    }

    fftw_execute(fftransf_vars->plan_b);

    double xarg = 1.0 / n;
    for (int i = 0; i < n; i++)
    {
        vec[i] = fftransf_vars->buff[i] * xarg;
    }
}

double rescale_dens(Densities* dens, int nxyz, double npart, double dxyz, int iscale)
{
    double xpart = 0;
    for (int i = 0; i < nxyz; i++)
    {
        xpart += dens->rho[i];
    }

    xpart = xpart * dxyz / npart;

    if (iscale == 0)
    {
        return xpart;
    }

    for (int i = 0; i < nxyz; i++)
    {
        dens->rho[i] = dens->rho[i] / xpart;
    }

    return xpart;
}

void exch_nucl_dens(
    MPI_Comm commw, 
    int ip, 
    int gr_ip, int gr_np, 
    int idim, 
    double* ex_array_p, double* ex_array_n
)
{
    const int tag1 = 100, tag2 = 300;

    MPI_Status istats;
    if (ip == gr_ip)
    {
        MPI_Send(ex_array_p, idim, MPI_DOUBLE, gr_ip + gr_np, tag1, commw);
    }

    if (ip == gr_ip + gr_np)
    {
        MPI_Recv(ex_array_p, idim, MPI_DOUBLE, gr_ip, tag1, commw, &istats);
    }

    if (ip == gr_ip + gr_np)
    {
        MPI_Send(ex_array_n, idim, MPI_DOUBLE, gr_ip, tag2, commw);
    }

    if (ip == gr_ip)
    {
        MPI_Recv(ex_array_n, idim, MPI_DOUBLE, gr_ip + gr_np, tag2, commw, &istats);
    }
}

void grid3(
    const double* xx, const double* yy, const double* zz, 
    int nx, int ny, int nz, 
    double* X, double* Y, double* Z
)
{
    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int iz = 0; iz < nz; iz++)
            {
                int ixyz = iz + nz * (iy + ny * ix);

                X[ixyz] = xx[ix];
                Y[ixyz] = yy[iy];
                Z[ixyz] = zz[iz];
            }
        }
    }
}

void divide_work(int n, int ip, int np, int* nstart, int* nstop)
{
    const int nav = n / np;
    const int nspill = n - np * nav;

    *nstop = 0;

    for (int i = 0; i <= ip; i++)
    {
        *nstart = *nstop;
        *nstop = *nstart + nav;

        if (i < nspill)
        {
            (*nstop)++;
        }
    }
}

void array_rescale(double* a, int n, double alpha)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = alpha * a[i];
    }
}

void cm_initial(
    const double* lam, 
    const complex* z, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc, 
    int nx, int ny, int nz, 
    double e_max, 
    const Lattice_arrays* latt_coords, 
    double* xcm, double* ycm, double* zcm, 
    MPI_Comm comm, 
    int icub
)
{
    int nxyz = nx * ny * nz;

    double xcm1 = 0;
    double ycm1 = 0;
    double zcm1 = 0;
    double xpart = 0;

    int n = 4 * nxyz;
    int nhalf = 2 * nxyz;

    for (int jj = nhalf; jj < n; jj++)
    {
        /* construct one vector at a time for positive eigenvalues */
        double f1 = sqrt(factor_ec(lam[jj], e_max, icub));
        if (f1 < 1e-6)
        {
            break;
        }

        for (int lj = 0; lj < n_iq; lj++)
        {
            int j = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;

            if (j == jj)
            {
                for (int li = 0; li < m_ip; li++)
                {
                    double wf = pow(std::abs(z[lj * m_ip + li]) * f1, 2.);
                    int i = (i_p * mb + (li / mb) * p_proc * mb + li % mb);

                    if (i > nhalf - 1)
                    {
                        i = i % nxyz;

                        xcm1 += (latt_coords->xa[i] * wf);
                        ycm1 += (latt_coords->ya[i] * wf);
                        zcm1 += (latt_coords->za[i] * wf);

                        xpart += wf;
                    }
                }
            }
        }
    }

    MPI_Allreduce(&xpart, xcm, 1, MPI_DOUBLE, MPI_SUM, comm);
    xpart = *xcm;

    MPI_Allreduce(&xcm1, xcm, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&ycm1, ycm, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&zcm1, zcm, 1, MPI_DOUBLE, MPI_SUM, comm);

    *xcm /= xpart;
    *ycm /= xpart;
    *zcm /= xpart;
}

void change_mu_eq_sp(double* amu, double* lam, double* occ, const int nwf, const double npart)
{
    double* e_eq = Allocate<double>(nwf);
    double* del2 = Allocate<double>(nwf);

    for (int i = 0; i < nwf; ++i)
    {
        e_eq[i] = lam[i] * (1.0 - 2.0 * occ[i]) + *amu;
        del2[i] = pow(lam[i], 20.) - pow(e_eq[i] - *amu, 2.);
    }

    for (int k = 0; k < 51; ++k)
    {
        double xpart = (double)nwf;
        double dndmu = 0;

        for (int i = 0; i < nwf; ++i)
        {
            double e_qp = sqrt(pow(e_eq[i] - *amu, 2.0) + del2[i]);
            xpart -= (e_eq[i] - *amu) / e_qp;
            dndmu += del2[i] / pow(e_qp, 3.0);
        }

        xpart = xpart / 2.0;
        if (fabs(xpart - npart) < 1e-12)
        {
            *amu -= 6.0 * (xpart - npart) / npart;
            break;
        }

        dndmu = 1.0 / dndmu;
        dndmu = std::min(0.8 * dndmu, 3.0);

        *amu += dndmu * (npart - xpart);
    }

    free(del2); 
    free(e_eq);
}

double distMass(
    const double* rho_p, const double* rho_n, 
    double n, 
    double z0, 
    const double* za, 
    const int* wz, 
    double dxyz
)
{
    double sum1 = 0;
    double sum2 = 0;
    double sum1_ = 0;
    double sum2_ = 0;

    for (int i = 0; i < n; i++) 
    {
        double rho = rho_p[i] + rho_n[i];

        if (za[i] < z0) 
        {
            sum1_ += rho;
            sum1 += (z0 - za[i] * wz[i]) * rho;
        }
        else if (za[i] > z0) 
        {
            sum2_ += rho;
            sum2 += (za[i] * wz[i] - z0) * rho;
        }
        else 
        {
            sum1_ += .5 * rho;
            sum2_ += .5 * rho;
        }
    }

    printf("Mass1=%f Mass2=%f\n", sum1_ * dxyz, sum2_ * dxyz);

    return sum2 / sum2_ + sum1 / sum1_;
}
