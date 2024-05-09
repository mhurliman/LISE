// for license information, see the accompanying LICENSE file

/* includes all the routines necessary to construct the Hamiltonian */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <inttypes.h>
#include <assert.h>
#include <mpi.h>

#include "common.h"

/* computes the occupation numbers = sum over the v's for each eigenvector */
void occ_numbers(
    const complex* z, 
    double* occ, 
    int nxyz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc, 
    MPI_Comm gr_comm, int gr_ip)
{
    double* occbuff = Allocate<double>(2 * nxyz);

    memset(occbuff, 0, 2 * nxyz * sizeof(occbuff[0]));

    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb - 2 * nxyz;
        if (gj < 0)
            continue;

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;
            if (gi / nxyz < 2)
                continue;

            int i = lj * m_ip + li;
            occbuff[gj] += Square(std::abs(z[i]));
        }
    }

    MPI_Allreduce(occbuff, occ, 2 * nxyz, MPI_DOUBLE, MPI_SUM, gr_comm);
    free(occbuff);
}

void make_ke_loc(
    complex* ham, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    const double* mass_eff, 
    const double* u_loc, 
    int nx, int ny, int nz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc
)
{
    int n = m_ip * n_iq;
    int nxyz = nx * ny * nz;

    /* the hamiltonian is zeroed here */
    memset(ham, 0, n * sizeof(ham[0]));

    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;

        double dsign;
        if (gj < 2 * nxyz)
            dsign = 0.5;
        else
            dsign = -0.5;

        int j_xyz = gj % nxyz;
        
        int ix_j, iy_j, iz_j;
        i2xyz(j_xyz, &ix_j, &iy_j, &iz_j, ny, nz);

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;

            if (gi / nxyz == gj / nxyz)
            {
                int i_xyz = gi % nxyz;
                int ix_i, iy_i, iz_i;
                i2xyz(i_xyz, &ix_i, &iy_i, &iz_i, ny, nz);

                int i = lj * m_ip + li;

                if (gi == gj)
                    ham[i] += 2.0 * dsign * u_loc[i_xyz];

                if (iy_i == iy_j && iz_i == iz_j)
                    ham[i] += dsign * k1d_x[abs(ix_i - ix_j)] * (mass_eff[i_xyz] + mass_eff[j_xyz]);

                if (ix_i == ix_j && iz_i == iz_j)
                    ham[i] += dsign * k1d_y[abs(iy_i - iy_j)] * (mass_eff[i_xyz] + mass_eff[j_xyz]);

                if (iy_i == iy_j && ix_i == ix_j)
                    ham[i] += dsign * k1d_z[abs(iz_i - iz_j)] * (mass_eff[i_xyz] + mass_eff[j_xyz]);
            }
        }
    }
}

void make_pairing(
    complex* ham, 
    int nxyz, 
    const complex* delta, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc,  int q_proc
)
{
    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;
        int j_xyz = gj % nxyz;

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;
            if (j_xyz != gi % nxyz)
                continue;

            int i = lj * m_ip + li;
            if (gi / nxyz == 0 && gj / nxyz == 3)
                ham[i] = delta[j_xyz];

            if (gi / nxyz == 1 && gj / nxyz == 2)
                ham[i] = -delta[j_xyz];

            if (gi / nxyz == 2 && gj / nxyz == 1)
                ham[i] = -conj(delta[j_xyz]); /* conjugate here */

            if (gi / nxyz == 3 && gj / nxyz == 0)
                ham[i] = conj(delta[j_xyz]); /* conjugate here */
        }
    }
}

/* Calculates the SO contribution between u_up and u_up and u_up and u_dn */
void so_contributions_1(
    complex* ham, 
    double* wx, double* wy, double* wz, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;
    int nxyz = nx * ny * nz;

    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;
        if (gj > 2 * nxyz - 1)
            continue;

        int jj = gj % nxyz;
        int ix2, iy2, iz2;   
        i2xyz(jj, &ix2, &iy2, &iz2, ny, nz);

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;
            if (gi / nxyz != 0)
                continue;

            int ii = gi;
            int ix1, iy1, iz1;
            i2xyz(ii, &ix1, &iy1, &iz1, ny, nz);

            int i = lj * m_ip + li;

            if (gj / nxyz == 0)
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] -= 1i * (wy[ii] + wy[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] += 1i * (wx[ii] + wx[jj]) * d1_y[iy1 - iy2 + ny1];
            }
            else
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] += (wz[ii] + wz[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] -= 1i * (wz[ii] + wz[jj]) * d1_y[iy1 - iy2 + ny1];

                if (ix1 == ix2 && iy1 == iy2)
                    ham[i] -= (wx[ii] + wx[jj] - 1i * (wy[ii] + wy[jj])) * d1_z[iz1 - iz2 + nz1];
            }
        }
    }
}

/* Calculates the SO contribution between u_dn and u_up and u_dn and u_dn */
void so_contributions_2(
    complex* ham, 
    const double* wx, const double* wy, const double* wz, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;
    int nxyz = nx * ny * nz;

    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;
        if (gj > 2 * nxyz - 1)
            continue;

        int jj = gj % nxyz;
        int ix2, iy2, iz2;
        i2xyz(jj, &ix2, &iy2, &iz2, ny, nz);

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;
            if (gi / nxyz != 1)
                continue;

            int ii = gi % nxyz;
            int ix1, iy1, iz1;
            i2xyz(ii, &ix1, &iy1, &iz1, ny, nz);

            int i = lj * m_ip + li;

            if (gj / nxyz == 1)
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] += 1i * (wy[ii] + wy[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] -= 1i * (wx[ii] + wx[jj]) * d1_y[iy1 - iy2 + ny1];
            }
            else
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] -= (wz[ii] + wz[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] -= 1i * (wz[ii] + wz[jj]) * d1_y[iy1 - iy2 + ny1];

                if (ix1 == ix2 && iy1 == iy2)
                    ham[i] += (wx[ii] + wx[jj] + 1i * (wy[ii] + wy[jj])) * d1_z[iz1 - iz2 + nz1];
            }
        }
    }
}

/* Calculates the SO contribution between v_up and v_up and v_up and v_dn */
void so_contributions_3(
    complex* ham, 
    const double* wx, const double* wy, const double* wz, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;
    int nxyz = nx * ny * nz;

    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;
        if (gj < 2 * nxyz)
            continue;

        int jj = gj % nxyz;
        int ix2, iy2, iz2;
        i2xyz(jj, &ix2, &iy2, &iz2, ny, nz);

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;
            if (gi / nxyz != 2)
                continue;

            int ii = gi % nxyz;
            int ix1, iy1, iz1;
            i2xyz(ii, &ix1, &iy1, &iz1, ny, nz);

            int i = lj * m_ip + li;

            if (gj / nxyz == 2)
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] -= 1i * (wy[ii] + wy[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] += 1i * (wx[ii] + wx[jj]) * d1_y[iy1 - iy2 + ny1];
            }
            else
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] -= (wz[ii] + wz[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] -= 1i * (wz[ii] + wz[jj]) * d1_y[iy1 - iy2 + ny1];

                if (ix1 == ix2 && iy1 == iy2)
                    ham[i] += (wx[ii] + wx[jj] + 1i * (wy[ii] + wy[jj])) * d1_z[iz1 - iz2 + nz1];
            }
        }
    }
}

/* Calculates the SO contribution between v_up and v_dn and v_dn and v_dn */
void so_contributions_4(
    complex* ham, 
    const double* wx, const double* wy, const double* wz, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    int nx, int ny, int nz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    int n = m_ip * n_iq;
    int nxyz = nx * ny * nz;

    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;

        if (gj < 2 * nxyz)
            continue;

        int jj = gj % nxyz;
        int ix2, iy2, iz2;
        i2xyz(jj, &ix2, &iy2, &iz2, ny, nz);

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;

            if (gi / nxyz != 3)
                continue;

            int ii = gi % nxyz;
            int ix1, iy1, iz1;
            i2xyz(ii, &ix1, &iy1, &iz1, ny, nz);

            int i = lj * m_ip + li;

            if (gj / nxyz == 3)
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] += 1i * (wy[ii] + wy[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] -= 1i * (wx[ii] + wx[jj]) * d1_y[iy1 - iy2 + ny1];
            }
            else
            {
                if (iy1 == iy2 && iz1 == iz2)
                    ham[i] += (wz[ii] + wz[jj]) * d1_x[ix1 - ix2 + nx1];

                if (ix1 == ix2 && iz1 == iz2)
                    ham[i] -= 1i * (wz[ii] + wz[jj]) * d1_y[iy1 - iy2 + ny1];

                if (ix1 == ix2 && iy1 == iy2)
                    ham[i] -= (wx[ii] + wx[jj] - 1i * (wy[ii] + wy[jj])) * d1_z[iz1 - iz2 + nz1];
            }
        }
    }
}

void i2xyz(int i, int* ix, int* iy, int* iz, int ny, int nz)
{
    *iz = i % nz;
    *iy = ((i - *iz) / nz) % ny;
    *ix = (i - *iz - nz * *iy) / (nz * ny);

    if (*iz + nz * (*iy + ny * *ix) != i)
    {
        printf(" wrong mapping %d != %d ", *iz + nz * (*iy + ny * *ix), i);
    }
}

void make_p_term(
    complex* ham, 
    complex* d1_x, complex* d1_y, complex* d1_z, 
    double Vx, double Vy, double Vz, int nx, 
    int ny, int nz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc
)
{
    int nx1 = nx - 1;
    int ny1 = ny - 1;
    int nz1 = nz - 1;

    int nxyz = nx * ny * nz;

    for (int lj = 0; lj < n_iq; lj++)
    {
        int gj = i_q * nb + (lj / nb) * q_proc * nb + lj % nb;
        int j_xyz = gj % nxyz;

        int ix_j, iy_j, iz_j;
        i2xyz(j_xyz, &ix_j, &iy_j, &iz_j, ny, nz);

        for (int li = 0; li < m_ip; li++)
        {
            int gi = i_p * mb + (li / mb) * p_proc * mb + li % mb;
            if (gi / nxyz == gj / nxyz)
            {
                int i_xyz = gi % nxyz;
                
                int ix_i, iy_i, iz_i;
                i2xyz(i_xyz, &ix_i, &iy_i, &iz_i, ny, nz);
                
                int i = lj * m_ip + li;

                if (iy_i == iy_j && iz_i == iz_j)
                    ham[i] += 1i * d1_x[ix_i - ix_j + nx1] * Vx;

                if (ix_i == ix_j && iz_i == iz_j)
                    ham[i] += 1i * d1_y[iy_i - iy_j + ny1] * Vy;

                if (iy_i == iy_j && ix_i == ix_j)
                    ham[i] += 1i * d1_z[iz_i - iz_j + nz1] * Vz;
            }
        }
    }
}

/* constructs the full Hamiltonian matrix */
void make_ham(
    complex* ham, 
    const double* k1d_x, const double* k1d_y, const double* k1d_z, 
    const complex* d1_x, const complex* d1_y, const complex* d1_z, 
    const Potentials* pots, const Densities* dens, 
    int nx, int ny, int nz, 
    int m_ip, int n_iq, 
    int i_p, int i_q, 
    int mb, int nb, 
    int p_proc, int q_proc, 
    int nstart, int nstop, 
    MPI_Comm comm
)
{
#ifdef CONSTR_Q0
    int ii0 = 0;
#else
    int ii0 = 1;
#endif

    double* u_loc = Allocate<double>(pots->nxyz);
    laplacean(pots->mass_eff, pots->nxyz, u_loc, nstart, nstop, comm, k1d_x, k1d_y, k1d_z, nx, ny, nz, 1);

    for (int i = 0; i < pots->nxyz; i++)
    {
        u_loc[i] = 0.5 * u_loc[i] + pots->u_re[i] - *(pots->amu);

#ifdef CONSTRCALC
        for (int ii = ii0; ii < 4; ii++)
            u_loc[i] += pots->lam2[ii] * pots->v_constraint[i + ii * pots->nxyz];
#endif
    }

    make_ke_loc(ham, k1d_x, k1d_y, k1d_z, pots->mass_eff, u_loc, nx, ny, nz, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc);

    complex* delta2 = Allocate<complex>(pots->nxyz);
    for (int i = 0; i < pots->nxyz; i++)
    {
        delta2[i] = pots->delta[i] + pots->delta_ext[i];
    }

    make_pairing(ham, pots->nxyz, delta2, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc);

    so_contributions_1(ham, pots->wx, pots->wy, pots->wz, d1_x, d1_y, d1_z, nx, ny, nz, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc);
    so_contributions_2(ham, pots->wx, pots->wy, pots->wz, d1_x, d1_y, d1_z, nx, ny, nz, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc);
    so_contributions_3(ham, pots->wx, pots->wy, pots->wz, d1_x, d1_y, d1_z, nx, ny, nz, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc);
    so_contributions_4(ham, pots->wx, pots->wy, pots->wz, d1_x, d1_y, d1_z, nx, ny, nz, m_ip, n_iq, i_p, i_q, mb, nb, p_proc, q_proc);

    free(u_loc);
    free(delta2);
}
