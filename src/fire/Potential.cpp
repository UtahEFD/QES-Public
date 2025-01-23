/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file Potential.cpp
 * @brief This function calculates the fire induced winds based on heat release and plume merging
 */
#include "Fire.h"


void Fire ::potential(WINDSGeneralData *WGD)
{

    auto start = std::chrono::high_resolution_clock::now();// Start recording execution time
    float g = 9.81;
    float rhoAir = 1.125;
    float C_pa = 1150;
    float T_a = 293.15;

    float ur, uz;
    float u_p = 0;      ///< u velocity from potential field in target cell
    float v_p = 0;      ///< v velocity from potential field in target cell
    float w_p = 0;      ///< w velocity from potential field in target cell
    float alpha_e = 0.09;///< entrainment constant (Kaye & Linden 2004)
    float lambda_mix = 1 / alpha_e * sqrt(25.0 / 132.0);///< nondimensional plume mixing height
    float U_c;      ///<characteristic velocity 
    float L_c;      ///<characteristic length 

    // dr and dz, assume linear spacing between
    float drStar = rStar[1] - rStar[0];
    float dzStar = zStar[1] - zStar[0];

    // reset potential fields
    std::fill(Pot_u.begin(), Pot_u.end(), 0);
    std::fill(Pot_v.begin(), Pot_v.end(), 0);
    std::fill(Pot_w.begin(), Pot_w.end(), 0);

    // set z_mix to terrain height
    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny - 1; j++) {
            int id = i + j * (nx - 1);
            z_mix[id] = WGD->terrain_id[id];
        }
    }
    /**
     * Calculate Potential field based on heat release
     * Baum and McCaffrey plume model
     **/

    // loop through burning cells to get heat release

    //float H0 = 0;///< heat release
    int cent = 0;
    int icent = 0;
    int jcent = 0;
    int counter = 0;
    int firei;///< index center of merged fire sources (i)
    int firej;///< index center of merged fire sources (j)
    int firek;///< index average height of terrain of merged fires (k)

    for (int ii = 0; ii < nx - 1; ii++) {
        for (int jj = 0; jj < ny - 1; jj++) {
            int id = ii + jj * (nx - 1);
            if (burn_flag[id] == 1) {
                counter += 1;
                icent += ii;
                jcent += jj;
                H += H0[id];
            }
        }
    }

    if (H != 0) {
        float kmax = 0;///< plume mixing height
        int XIDX;
        int YIDX;
        int ZIDX = 0;
        int filt = 0;
        float k_fire = 0;///< terrain index for plume merge
        float k_fire_old = 0;
        float mixIDX = 0;
        float mixIDX_old;

        while (filt < nx - 1) {
            filt = pow(2.0, ZIDX);
            counter = 0;
            ZIDX += 1;
            z_mix_old = z_mix;
            XIDX = 0;
            while (XIDX < nx - 1 - filt) {
                YIDX = 0;
                while (YIDX < ny - 1 - filt) {
                    H = 0;
                    k_fire = 0;
                    k_fire_old = 0;
                    icent = 0;
                    jcent = 0;
                    counter = 0;
                    for (int ii = XIDX; ii < XIDX + filt; ii++) {
                        for (int jj = YIDX; jj < YIDX + filt; jj++) {
                            int id = ii + jj * (nx - 1);
                            if (burn_flag[id] == 1) {
                                icent += ii;
                                jcent += jj;
                                H += H0[id];
                                k_fire += WGD->terrain[id];
                                k_fire_old += z_mix_old[id];
                                counter += 1;
                            }
                        }
                    }

                    if (H != 0) {
                        firei = icent / counter;
                        firej = jcent / counter;
                        U_c = pow(g * g * H / rhoAir / T_a / C_pa, 1.0 / 5.0);
                        L_c = pow(H / rhoAir / C_pa / T_a / pow(g, 1.0 / 2.0), 2.0 / 5.0);
                        firek = k_fire / counter;
                        mixIDX_old = floor(k_fire_old / counter);
                        mixIDX = (lambda_mix * dx * filt)/dz + k_fire;
                        for (int ii = XIDX; ii < XIDX + filt; ii++) {
                            for (int jj = YIDX; jj < YIDX + filt; jj++) {
                                int id = ii + jj * (nx - 1);
                                z_mix[id] = mixIDX;
                            }
                        }
                        kmax = nz - 3 > mixIDX ? mixIDX : nz - 3;
                        // Loop through vertical levels
                        for (int kpot = mixIDX_old; kpot < kmax; kpot++) {
                            // Calculate virtual origin
                            float z_v = dx * filt * 0.124 / alpha_e + firek;///< virtual origin for merged plumes
                            float z_k = (kpot + z_v) * dz / L_c;///< non-dim vertical distance between fire cell and target cell k
                            if (z_k < 0) {
                                z_k = 0;
                            }
                            float zeta = 0;
                            float x1 = 0;
                            // Loop through horizontal domain
                            for (int ipot = 0; ipot < nx - 1; ipot++) {
                                for (int jpot = 0; jpot < ny - 1; jpot++) {
                                    float deltaX = (ipot - firei) * dx / L_c;///< non-dim distance between fire cell and target cell k in x direction
                                    float deltaY = (jpot - firej) * dy / L_c;///< non-dim distance between fire cell and target cell k in y direction
                                    float h_k = sqrt(deltaX * deltaX + deltaY * deltaY);///< non-dim radial distance from fire cell and target cell k in horizontal
                                    u_p = 0;
                                    v_p = 0;
                                    w_p = 0;
                                    // if radius = 0
                                    if (h_k < 0.00001 && z_k < 60) {
                                        float zMinIdx = floor(z_k / dzStar);
                                        float zMaxIdx = ceil(z_k / dzStar);
                                        ur = 0.0;
                                        uz = u_z[zMinIdx * pot_r];
                                        u_p = U_c * ur;
                                        v_p = U_c * ur;
                                        w_p = U_c * uz;
                                    }
                                    // if in potential field lookup, r*(h_k) < 30 and z*(z_k) < 60
                                    else if (z_k < 60 && h_k < 30) {
                                        // indices for lookup
                                        float rMinIdx = floor(h_k / drStar);
                                        float rMaxIdx = ceil(h_k / drStar);
                                        float zMinIdx = floor(z_k / dzStar);
                                        float zMaxIdx = ceil(z_k / dzStar);
                                        ur = u_r[rMinIdx + zMinIdx * pot_r];
                                        uz = u_z[rMinIdx + zMinIdx * pot_r];
                                        u_p = U_c * ur * deltaX / h_k;
                                        v_p = U_c * ur * deltaY / h_k;
                                        w_p = U_c * uz;
                                    } else {
                                        zeta = sqrt(h_k * h_k + z_k * z_k);
                                        x1 = (1 + cos(atan(h_k / z_k))) / 2.0;
                                        // lookup indices for G(x) and G'(x) - spans 0.5 to 1.0
                                        int gMinIdx = floor(pot_G * (x1 - .5) / .5);
                                        int gMaxIdx = ceil(pot_G * (x1 - .5) / .5);
                                        // values for G and G'
                                        float g_x = 0.5 * (G[gMinIdx] + G[gMaxIdx]);
                                        float gprime_x = 0.5 * (Gprime[gMinIdx] + Gprime[gMaxIdx]);
                                        ur = h_k / (2 * PI * pow(h_k * h_k + z_k * z_k, (3 / 2.0))) + pow(zeta, (-1 / 3.0)) * ((5 / 6.0) * (1 - 2 * x1) / sqrt(x1 * (1 - x1)) * g_x - sqrt(x1 * (1 - x1)) * gprime_x);
                                        uz = z_k / (2 * PI * pow(h_k * h_k + z_k * z_k, (3 / 2.0))) + pow(zeta, (-1 / 3.0)) * ((5 / 3.0) * g_x + (1 - 2 * x1) / 2.0 * gprime_x);
                                        if (h_k != 0) {
                                            u_p = U_c * ur * deltaX / h_k;
                                            v_p = U_c * ur * deltaY / h_k;
                                        }
                                        w_p = U_c * uz;
                                    }
                                    // modify potential fields
		                            int cellCentPot = ipot + jpot * (nx - 1) + (kpot) * (nx - 1) * (ny - 1);
                                    Pot_u[cellCentPot] += u_p;
                                    Pot_v[cellCentPot] += v_p;
                                    Pot_w[cellCentPot] += w_p;
                                }
                            }
                        }
                    }
                    YIDX += filt;
                }//while YIDX
                XIDX += filt;
            }//while XIDX
        }//k
    }//H0!=0
    // Modify u,v,w in solver - superimpose Potential field onto velocity field (interpolate from potential cell centered values)
    for (int kadd = 1; kadd < nz - 2; kadd++) {
        for (int jadd = 1; jadd < ny - 1; jadd++) {
            for (int iadd = 1; iadd < nx - 1; iadd++) {
                int cell_face = iadd + jadd * nx + (kadd - 1) * nx * ny;
                int cell_cent = iadd + jadd * (nx - 1) + (kadd - 1) * (nx - 1) * (ny - 1);
                if (WGD->icellflag[cell_cent] == 12 or WGD->icellflag[cell_cent] == 1) {
                    WGD->u0[cell_face] = WGD->u0[cell_face] + 0.5 * (Pot_u[cell_cent] + Pot_u[cell_cent + 1]);
                    WGD->v0[cell_face] = WGD->v0[cell_face] + 0.5 * (Pot_v[cell_cent] + Pot_v[cell_cent + (nx - 1)]);
                    WGD->w0[cell_face] = WGD->w0[cell_face] + 0.5 * (Pot_w[cell_cent] + Pot_w[cell_cent + (nx - 1) * (ny - 1)]);
                }
            }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time

    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "[QES-Fire] Pot\t Elapsed time: " << elapsed.count() << " s\n";// Print out elapsed execution time
}
