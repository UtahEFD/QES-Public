/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb
* Source: Adapted from sor3d.f90, the denominators portion (now in 
*         denominators.f90).
*/

#ifndef COMP_DENOMS_H
#define COMP_DENOMS_H

#include "quic/boundaryMatrices.h"

extern "C" void showError(char const* loc);

namespace QUIC
{
	__global__ void k_comp_denoms
	(
		boundaryMatrices d_bndrs, 
		float omegarelax, float A, float B,
		unsigned int size, unsigned int offst
	) 
	{
		unsigned int blck_cnt = blockIdx.y*gridDim.x + blockIdx.x;
	  unsigned int cI       = offst + blck_cnt*blockDim.x + threadIdx.x;

    float o, p, q; float dmmy;
    
    decodeBoundary
    (
    	d_bndrs.cmprssd[cI], 
    	dmmy, dmmy, dmmy, dmmy, dmmy, dmmy, 
    	o, p, q
    );

		if(cI < size)
		{
		  d_bndrs.denoms[cI] = omegarelax / (2.f*(o + A*p + B*q));
		}
	}

	extern "C"
	void cudaCompDenoms(boundaryMatrices d_bndrs, float omegarelax, float A, float B) 
	{		
		// Get max block x dim.
		int device_num = 0;
		cudaGetDevice(&device_num);
		struct cudaDeviceProp dprops;
		cudaGetDeviceProperties(&dprops, device_num);
		int mx_grid_x = dprops.maxGridSize[0];
	
		// Find block size.
		int size  = d_bndrs.dim.x*d_bndrs.dim.y*d_bndrs.dim.z;
		int thrds = 256;
		int blcks = (size / thrds) + 1;
		int offst = 0;

		dim3 g_dim(blcks);
		dim3 b_dim(thrds);

		if(blcks > mx_grid_x) // New style
		{
			int grid_rws = blcks / mx_grid_x;
			    g_dim    = dim3(mx_grid_x, grid_rws);
		
			// Big / Full
			k_comp_denoms<<< g_dim, b_dim >>>(d_bndrs, omegarelax, A, B, size, offst); 
			showError("Comp Denoms (big/full)");
			
			offst = grid_rws*mx_grid_x*thrds;
			g_dim = dim3(blcks - grid_rws*mx_grid_x);
		}

		// Calculates comp denom that's been allocated.
		k_comp_denoms<<< g_dim, b_dim >>>(d_bndrs, omegarelax, A, B, size, offst);
		showError("Comp Denoms");
  }
}

#endif
