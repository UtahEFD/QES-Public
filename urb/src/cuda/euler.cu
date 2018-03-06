/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb.
* Source: Adapted from euler.f90 in QUICurbv5.? Fortran code.
*/

#ifndef EULER_H
#define EULER_H

#include <stdio.h>

#include "../cpp/celltypes.h"
#include "quicloader/velocities.h"

extern "C" void showError(char const* loc);

namespace QUIC 
{

	__global__ 
	void k_euler
	(
		velocities d_vels, float* d_p1, 
		float ovalpha1, float ovalpha2, 
		float dx_inv, float dy_inv, float dz_inv, 
		unsigned int size, unsigned int offst
	) 
	{
		unsigned int fbI = offst + (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x;
		unsigned int cI  = fbI + threadIdx.x;
	
		// From second slice to end of d_p1...
		if(cI < size)
		{
			int cell_row = d_vels.dim.x - 1;
			int cell_slc = cell_row*(d_vels.dim.y - 1);
			int grid_row = d_vels.dim.x;
		
			unsigned int gI = cI + int(cI / cell_row) + grid_row*int(cI / cell_slc);

			float d_p1_cI = d_p1[cI];
	
			d_vels.u[gI] += ovalpha1*dx_inv*(d_p1_cI - d_p1[cI - 1]);
			d_vels.v[gI] += ovalpha1*dy_inv*(d_p1_cI - d_p1[cI - cell_row]);
			d_vels.w[gI] += ovalpha2*dz_inv*(d_p1_cI - d_p1[cI - cell_slc]);
		}
	}

	// Device version of building->interior, I believe. This is a common operation
	// and could be pushed elsewhere.
	__global__ 
	void k_clear_buildings
	(
		velocities d_vels, celltypes d_typs, 
		unsigned int size, unsigned int offst
	)
	{
		unsigned int fbI = offst + (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x;
		unsigned int cI  = fbI + threadIdx.x;
	
		if(cI < size)
		{
			int cell_row = d_typs.dim.x;
			int cell_slc = d_typs.dim.x*d_typs.dim.y;
			int grid_row = d_vels.dim.x;
			int grid_slc = d_vels.dim.x*d_vels.dim.y;

			unsigned int gI    = cI + int(cI / cell_row) + grid_row*int(cI / cell_slc);
			unsigned int gI_pi = gI + 1;
			unsigned int gI_pj = gI + grid_row;
			unsigned int gI_pk = gI + grid_slc;

			if(d_typs.c[cI] == SOLID) 
			{
				d_vels.u[gI]    = d_vels.v[gI]    = d_vels.w[gI]    = 0.f;
				d_vels.u[gI_pi] = d_vels.v[gI_pj] = d_vels.w[gI_pk] = 0.f;
			}
		}
	}

	// Calculates velocites in d_vels from lagrangian in d_p1
	extern "C"
	void cudaEuler
	(
		velocities d_vels, celltypes d_typs,
		float* d_p1, float alpha1, float alpha2, 
		float dx, float dy, float dz 
	) 
	{
		// Get max block x dim.
		int device_num = 0;
		cudaGetDevice(&device_num);
		struct cudaDeviceProp dprops;
		cudaGetDeviceProperties(&dprops, device_num);
		int mx_grid_x = dprops.maxGridSize[0];
	
		// Find block size.
		unsigned int offst = d_typs.dim.x*d_typs.dim.y;
		unsigned int size  = d_typs.dim.x*d_typs.dim.y*d_typs.dim.z - offst;
		
		int thrds = 256;
		int blcks = (size / thrds) + 1;
		
		dim3 g_dim(blcks);
		dim3 b_dim(thrds);

		// Avoid recalculation
		float ovalpha1 = 1.f / (2.f*alpha1*alpha1);
		float ovalpha2 = 1.f / (2.f*alpha2*alpha2);

		float dx_inv = 1.f / dx;
		float dy_inv = 1.f / dy;
		float dz_inv = 1.f / dz;


		if(blcks > mx_grid_x) // New style
		{
			int grid_rws = blcks / mx_grid_x;
			    g_dim    = dim3(mx_grid_x, grid_rws);
		
			// \\todo put these 1 / dx, 1 / dy, 1 / alphas into constant memory?
			k_euler<<< g_dim, b_dim >>>
			(
				d_vels, d_p1, ovalpha1, ovalpha2, 
				dx_inv, dy_inv, dz_inv, size, offst
			);
			showError("Euler (big/full)");

			k_clear_buildings<<< g_dim, b_dim >>>(d_vels, d_typs, size, offst); 
			showError("ClearBuildings (big/full)");

			offst += grid_rws*mx_grid_x*thrds;
			g_dim  = dim3(blcks - grid_rws*mx_grid_x);
		}

		k_euler<<< g_dim, b_dim >>>
		(
			d_vels, d_p1, ovalpha1, ovalpha2, 
			dx_inv, dy_inv, dz_inv, size, offst
		);
		showError("Euler");
		
		k_clear_buildings<<< g_dim, b_dim >>>(d_vels, d_typs, size, offst); 
		showError("ClearBuildings (big/full)");
	}
}

#endif
