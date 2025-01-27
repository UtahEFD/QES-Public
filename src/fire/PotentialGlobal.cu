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
 * @file PotentialGlobal.cu
 * @brief This function calculates the fire induced winds based on heat release and plume merging using CUDA Global Memory
 */

#include "PotentialGlobal.h"

#define BLOCKSIZE 1024

using namespace std::chrono;
using namespace std;
using std::ofstream;
using std::ifstream;
using std::istringstream;
using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;
using std::to_string;

#define cudaCheck(x) _cudaCheck(x, #x, __FILE__, __LINE__)

template<typename T>
void _cudaCheck(T e, const char *func, const char *call, const int line)
{
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Fire-Induced Potential kernal
__global__ 
void PotGlob(
    int nx, int ny, int nz, int filt, 
    float firei, float firej,
    int mixIDX_old, int kmax,
    float* d_u_r, float* d_u_z, float* d_G, float* d_Gprime,
    float dx, float dy, float dz, float dzStar, float drStar, 
    int pot_r, int pot_G, float z_v, float U_c, float L_c,
    float* d_Pot_u, float* d_Pot_v, float* d_Pot_w
) {
    int ipot = blockDim.x * blockIdx.x + threadIdx.x;
    int jpot = blockDim.y * blockIdx.y + threadIdx.y;
    int kpot = blockDim.z * blockIdx.z + threadIdx.z;

    float d_u_p = 0.0;
    float d_v_p = 0.0;
    float d_w_p = 0.0;
    float ur = 0.0;
    float uz = 0.0;

    if (ipot >= nx - 1 || jpot >= ny - 1) return;
    if (kpot > mixIDX_old && kpot <= kmax){

    float z_k = (kpot*dz - z_v) / L_c;
    if (z_k < 0) {
        z_k = 0;
    }

    float deltaX = (ipot - firei) * dx / L_c;///< non-dim distance between fire cell and target cell k in x direction
    float deltaY = (jpot - firej) * dy / L_c;///< non-dim distance between fire cell and target cell k in y direction
    float h_k = sqrtf(deltaX * deltaX + deltaY * deltaY);///< non-dim radial distance from fire cell and target cell k in horizontal
    
    // if radius = 0
    if (h_k < 0.00001 && z_k < 60) {
        int zMinIdx = floor(z_k / dzStar);
        int zMaxIdx = ceil(z_k / dzStar);
        ur = 0.0;
        uz = d_u_z[zMinIdx * pot_r];
        d_u_p = U_c * ur;
        d_v_p = U_c * ur;
        d_w_p = U_c * uz;
    }
    // if in potential field lookup, r*(h_k) < 30 and z*(z_k) < 60
    else if (z_k < 60 && h_k < 30) {
        // indices for lookup
        int rMinIdx = floor(h_k / drStar);
        int rMaxIdx = ceil(h_k / drStar);
        int zMinIdx = floor(z_k / dzStar);
        int zMaxIdx = ceil(z_k / dzStar);
        ur = d_u_r[rMinIdx + zMinIdx * pot_r];
        uz = d_u_z[rMinIdx + zMinIdx * pot_r];
        d_u_p = U_c * ur * deltaX / h_k;
        d_v_p = U_c * ur * deltaY / h_k;
        d_w_p = U_c * uz;
    } 
    else {
        float zeta = sqrtf(h_k * h_k + z_k * z_k);
        //float x1 = (1 + cos(atan(h_k / z_k))) / 2.0; 
        float x1 = (1 + (1 / sqrtf((h_k / z_k) * (h_k / z_k) + 1))) / 2.0;
        // lookup indices for G(x) and G'(x) - spans 0.5 to 1.0
        int gMinIdx = floor(pot_G * (x1 - .5) / .5);
        int gMaxIdx = ceil(pot_G * (x1 - .5) / .5);
        // values for G and G'
        float g_x = 0.5 * (d_G[gMinIdx] + d_G[gMaxIdx]);
        float gprime_x = 0.5 * (d_Gprime[gMinIdx] + d_Gprime[gMaxIdx]);
        ur = h_k / (2 * M_PI * powf(h_k * h_k + z_k * z_k, (3 / 2.0))) + 
            powf(zeta, (-1 / 3.0)) * ((5 / 6.0) * (1 - 2 * x1) / sqrtf(x1 * (1 - x1)) * g_x - sqrtf(x1 * (1 - x1)) * gprime_x);
        uz = z_k / (2 * M_PI * powf(h_k * h_k + z_k * z_k, (3 / 2.0))) + 
            powf(zeta, (-1 / 3.0)) * ((5 / 3.0) * g_x + (1 - 2 * x1) / 2.0 * gprime_x);
        if (h_k != 0) {
            d_u_p = U_c * ur * deltaX / h_k;
            d_v_p = U_c * ur * deltaY / h_k;
        }
        d_w_p = U_c * uz;
    }
    // modify potential fields
	int cellCentPot = ipot + jpot * (nx - 1) + (kpot) * (nx - 1) * (ny - 1);
    
    atomicAdd(&d_Pot_u[cellCentPot], d_u_p);
    atomicAdd(&d_Pot_v[cellCentPot], d_v_p);
    atomicAdd(&d_Pot_w[cellCentPot], d_w_p); 

    }
}

// Main function
void Fire ::potentialGlobal(WINDSGeneralData *WGD)
{
    auto start = std::chrono::high_resolution_clock::now();// Start recording execution time
    const int gridSize = (nx - 1) * (ny - 1) * (nz - 1);    ///< 3D grid size

    float g = 9.81;
    float rhoAir = 1.125;
    float C_pa = 1150;
    float T_a = 293.15;
    float alpha_e = 0.09;///< entrainment constant (Kaye & Linden 2004)
    float lambda_mix = 1 / alpha_e * sqrt(25.0 / 132.0);///< nondimensional plume mixing height
    float U_c;      ///<characteristic velocity 
    float L_c;      ///<characteristic length 

    // dr and dz, assume linear spacing between
    float drStar = rStar[1] - rStar[0];
    float dzStar = zStar[1] - zStar[0];

    std::fill(Pot_u.begin(), Pot_u.end(), 0);
    std::fill(Pot_v.begin(), Pot_v.end(), 0);
    std::fill(Pot_w.begin(), Pot_w.end(), 0);

    // allocate and copy potential velocity variables
    cudaMalloc((void **)&d_u_r, pot_r*pot_z * sizeof(float));
    cudaMemcpy(d_u_r, u_r.data(), pot_r*pot_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_u_z, pot_r*pot_z * sizeof(float));
    cudaMemcpy(d_u_z, u_z.data(), pot_r*pot_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_G, pot_G * sizeof(float));
    cudaMemcpy(d_G, G.data(), pot_G * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_Gprime, pot_G * sizeof(float));
    cudaMemcpy(d_Gprime, Gprime.data(), pot_G * sizeof(float), cudaMemcpyHostToDevice);
    
    // allocate and initialize potential velocity on device
    cudaMalloc((void **)&d_Pot_u, gridSize * sizeof(float));
    cudaMalloc((void **)&d_Pot_v, gridSize * sizeof(float));
    cudaMalloc((void **)&d_Pot_w, gridSize * sizeof(float));
    cudaMemcpy(d_Pot_u, Pot_u.data(), gridSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pot_v, Pot_v.data(), gridSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pot_w, Pot_w.data(), gridSize * sizeof(float), cudaMemcpyHostToDevice);
    
    /**
     * Calculate Potential field based on heat release
     * Baum and McCaffrey plume model
     **/

    // set z_mix to terrain height
    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny - 1; j++) {
            int id = i + j * (nx - 1);
            z_mix[id] = WGD->terrain_id[id];
        }
    }

    int icent = 0;
    int jcent = 0;
    int counter = 0;
    int firei;///< index center of merged fire sources (i)
    int firej;///< index center of merged fire sources (j)
    int firek;///< index average height of terrain of merged fires (k)

    int kmax = 0;///< plume mixing height
    int XIDX;
    int YIDX;
    int ZIDX = 0;
    int filt = 0;
    float k_fire = 0;///< terrain index for plume merge
    float k_fire_old = 0;
    float mixIDX = 0;
    int mixIDX_old;

    while (filt < nx - 1) {
        filt = pow(2.0, ZIDX);
        //counter = filt*filt;
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
                    mixIDX = ceil((lambda_mix * dx * filt)/dz + k_fire);
                    for (int ii = XIDX; ii < XIDX + filt; ii++) {
                        for (int jj = YIDX; jj < YIDX + filt; jj++) {
                            int id = ii + jj * (nx - 1);
                            z_mix[id] = mixIDX;
                        }
                    }
                    kmax = nz - 3 > mixIDX ? mixIDX : nz - 3;

                    // Calculate virtual origin
                    float z_v = -0.2869 * dx * filt * lambda_mix + firek;
                    
                    dim3 threadsPerBlock(32,32,1);
                    dim3 numBlocks(ceil(WGD->nx/16), ceil(WGD->ny/16), ceil(WGD->nz)/1);
                    
                    // calculate fire induced winds
                    PotGlob<<<numBlocks, threadsPerBlock>>>(
                        nx, ny, nz, filt,  
                        firei, firej, 
                        mixIDX_old, kmax,
                        d_u_r, d_u_z, d_G, d_Gprime,
                        dx, dy, dz, dzStar, drStar, 
                        pot_r, pot_G, z_v, U_c, L_c,
                        d_Pot_u, d_Pot_v, d_Pot_w
                    );  
                    cudaCheck(cudaGetLastError());  
                }
                YIDX += filt;
            } 
            XIDX += filt;
        }   
    }
  
    // Copy potential velocity from device to host
    cudaMemcpy(Pot_u.data(), d_Pot_u, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pot_v.data(), d_Pot_v, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pot_w.data(), d_Pot_w, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Modify u,v,w in solver - superimpose Potential field onto velocity field (interpolate from potential cell centered values)
    for (int iadd = 1; iadd < nx - 1; iadd++) {
        for (int jadd = 1; jadd < ny - 1; jadd++) {
            for (int kadd = 1; kadd < nz - 2; kadd++) {
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
    
    // Free memory
    cudaFree(d_Pot_u);
    cudaFree(d_Pot_v);
    cudaFree(d_Pot_w);
    cudaFree(d_u_r);
    cudaFree(d_u_z);
    cudaFree(d_G);
    cudaFree(d_Gprime);

    auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "[QES-Fire] Pot\t Elapsed time: " << elapsed.count() << " s\n";// Print out elapsed execution time
}