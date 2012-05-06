/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb.
* Source: Adapted from diffusion.f90 in QUICurbv5.? Fortran code.
*/

#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "../cpp/celltypes.h"
#include "quicutil/velocities.h"

extern "C" 
void showError(char const* loc);

extern "C" 
void cudaMax(float* d_array, int size, float* d_max);

extern "C" 
void cudaTurbulenceModel
(
	float* d_visc, 
	QUIC::celltypes d_typs, QUIC::velocities d_vels,
	float dx, float dy, float dz
);

namespace QUIC 
{

	// Coalescing this kernel isn't going to be fun.
	__global__ void k_diffusion
	(
		velocities d_vels,
		float* d_visc, float* d_max_of_visc,
		float dx, float dy, float dz
	) 
	{
		int nx = d_vels.dim.x - 1;
		int ny = d_vels.dim.y - 1;
		//int nz = d_vels.dim.z - 1;
	
		int grid_slice   = (nx+1) * (ny+1);
		int grid_row     = nx + 1;
		int stick_length = blockDim.x;

		int bidr = (int) blockIdx.x / (nx / stick_length);	
		int bidc =  blockIdx.x % (nx / stick_length);

		int gI = (blockIdx.y + 1) * grid_slice + bidr * grid_row + bidc * stick_length + threadIdx.x;

		if(bidr == 0) {return;} //Don't do first row. Last row left out.
		if(gI % grid_row == 0) {return;} //Don't do first col. Last column left out.

		float dxi = 1.f / dx;
		float dyi = 1.f / dy;
		float dzi = 1.f / dz;

		float* d_uo = d_vels.u;
		float* d_vo = d_vels.v;
		float* d_wo = d_vels.w;

		float c_uo = d_uo[gI];
		float c_vo = d_vo[gI];
		float c_wo = d_wo[gI];

		int gI_pi = gI + 1;
		int gI_mi = gI - 1;
		int gI_pj = gI + grid_row;
		int gI_mj = gI - grid_row;
		int gI_pk = gI + grid_slice;
		int gI_mk = gI - grid_slice;

		int gI_pi_mj = gI_pi - grid_row;
		int gI_mi_pj = gI_mi + grid_row;
		int gI_pi_mk = gI_pi - grid_slice;
		int gI_pj_mk = gI_pj - grid_slice;
		int gI_mi_pk = gI_mi + grid_slice;
		int gI_mj_pk = gI_mj + grid_slice;

		// X Momentum
		float Tuuip = 2.f*dxi*(d_uo[gI_pi] - c_uo       );
		float Tuuim = 2.f*dxi*(c_uo        - d_uo[gI_mi]);
		float Tuvjp =     dyi*(d_uo[gI_pj]	- c_uo       ) + dxi*(d_vo[gI_pi   ] - c_vo);
		float Tuvjm =     dyi*(c_uo        - d_uo[gI_mj]) + dxi*(d_vo[gI_pi_mj] - d_vo[gI_mj]);
		float Tuwkp =     dzi*(d_uo[gI_pk] - c_uo       ) + dxi*(d_wo[gI_pi   ] - c_wo);
		float Tuwkm =     dzi*(c_uo        - d_uo[gI_mk]) + dxi*(d_wo[gI_pi_mk] - d_wo[gI_mk]);
		
		float Fxd = d_visc[gI]*(dxi*(Tuuip - Tuuim) + dyi*(Tuvjp - Tuvjm) + dzi*(Tuwkp - Tuwkm));


		// Y Momentum
		float Tvuip =     dxi*(d_vo[gI_pi] - c_vo       ) + dyi*(d_uo[gI_pj   ] - c_uo       );
		float Tvuim =     dxi*(c_vo        - d_vo[gI_mi]) + dyi*(d_uo[gI_mi_pj] - d_uo[gI_mi]);
		float Tvvjp = 2.f*dyi*(d_vo[gI_pj] - c_vo       );
		float Tvvjm = 2.f*dyi*(c_vo        - d_vo[gI_mj]);
		float Tvwkp =     dzi*(d_vo[gI_pk] - c_vo       ) + dyi*(d_wo[gI_pj   ] - c_wo       );
		float Tvwkm =     dzi*(c_vo        - d_vo[gI_mk]) + dyi*(d_wo[gI_pj_mk] - d_wo[gI_mk]);
		
		float Fyd = d_visc[gI]*(dxi*(Tvuip - Tvuim) + dyi*(Tvvjp - Tvvjm) + dzi*(Tvwkp - Tvwkm));


		// Z Momentum
		float Twuip =     dxi*(d_wo[gI_pi] - c_wo       ) + dzi*(d_uo[gI_pk   ] - c_uo       );
		float Twuim =     dxi*(c_wo        - d_wo[gI_mi]) + dzi*(d_uo[gI_mi_pk] - d_uo[gI_mi]);
		float Twvjp =     dyi*(d_wo[gI_pj] - c_wo       ) + dzi*(d_vo[gI_pk   ] - c_vo       );
		float Twvjm =     dyi*(c_wo        - d_wo[gI_mj]) + dzi*(d_vo[gI_mj_pk] - d_vo[gI_mj]);
		float Twwkp = 2.f*dzi*(d_wo[gI_pk] - c_wo       );
		float Twwkm = 2.f*dzi*(c_wo        - d_wo[gI_mk]);
		
		float Fzd = d_visc[gI]*(dxi*(Twuip - Twuim) + dyi*(Twvjp - Twvjm) + dzi*(Twwkp - Twwkm));


		// Update velocity with diffusive fluxes
		float min = dx; 
		if(dy < min) min = dy;
		if(dz < min) min = dz;

		float dt = .25f * (min * min);
		if(*d_max_of_visc != 0.f) {dt /= *d_max_of_visc;}

		d_uo[gI] = c_uo + dt * Fxd;
		d_vo[gI] = c_vo + dt * Fyd;
		d_wo[gI] = c_wo + dt * Fzd;
		
		d_uo = d_vo = d_wo = 0; // Undangle the pointers.
	}

	// Take the velocities and apply the diffusion model.
	extern "C"
	void cudaDiffusion
	(
		velocities d_vels, celltypes d_typs,
		float* d_visc, float dx, float dy, float dz
	) 
	{
		int nx = d_typs.dim.x;
		int ny = d_typs.dim.y;
		int nz = d_typs.dim.z;
	
		int grid_domain = d_vels.dim.x*d_vels.dim.y*d_vels.dim.z;

		// Turbulence subroutine
		// Where viscosities get set.
		cudaTurbulenceModel(d_visc, d_typs, d_vels, dx, dy, dz);

		//Diffusion
		// Needs to change to improve performace. Get largest multiple of 64 that divides threads.
		//Want this to be a multiple of 64.
		int threads = (nx > 512) ? 64 : 64 ;

		int blocks_per_slice = (nx * ny) / (threads);
		int slices = nz - 1;

		dim3 dimGrid(blocks_per_slice, slices);
		dim3 dimBlock(threads);

		float* d_max_of_visc; cudaMalloc((void**) &d_max_of_visc, sizeof(float));
		cudaMax(d_visc, grid_domain, d_max_of_visc);

		k_diffusion<<< dimGrid, dimBlock >>>(d_vels, d_visc, d_max_of_visc, dx, dy, dz); 
		showError("Diffusion");
	}
}

#endif

