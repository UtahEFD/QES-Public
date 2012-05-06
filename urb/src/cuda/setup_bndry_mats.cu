/**
*	Author: Josh Clark <clark617@d.umn.edu> 
* Reason: CUDA-nizing QUICurb init / setup routines.
* Source: Adapted from boundary matrices setup routine by originally located in 
*         urbSetup.
*/

#ifndef SETUP_BNDRY_MATS_H
#define SETUP_BNDRY_MATS_H

#include "cuda/showerror.cu"
#include "../cpp/boundaryMatrices.h"

namespace QUIC
{
	
	__global__ void k_setup_bndry_mats
	(
		boundaryMatrices d_bndrs,
		celltypes d_typs,
		unsigned int size, unsigned int offst
	)
	{
		unsigned int blck_cnt = blockIdx.y*gridDim.x + blockIdx.x;
	  unsigned int cI       = offst + blck_cnt*blockDim.x + threadIdx.x;
	
    if(cI < size)
		{
		  determineBoundaryCell(d_bndrs.cmprssd[cI], d_typs, cI);
		}
	}
	
	extern "C"
	void cudaSetupBndryMats(boundaryMatrices d_bndrs, celltypes d_typs) 
	{
				// Get max block x dim.
		int device_num = 0;
		cudaGetDevice(&device_num);
		struct cudaDeviceProp dprops;
		cudaGetDeviceProperties(&dprops, device_num);
		unsigned mx_grid_x = dprops.maxGridSize[0];
	
		// Find block size.
		unsigned int size  = d_bndrs.dim.x*d_bndrs.dim.y*d_bndrs.dim.z;
		unsigned int thrds = 256;
		unsigned int blcks = (size / thrds) + 1;
		unsigned int offst = 0;

		dim3 g_dim(blcks);
		dim3 b_dim(thrds);

		if(blcks > mx_grid_x) // New style
		{
			int grid_rws = blcks / mx_grid_x;
			    g_dim    = dim3(mx_grid_x, grid_rws);
		
			// Big / Full
			k_setup_bndry_mats<<< g_dim, b_dim >>>(d_bndrs, d_typs, size, offst); 
			showError("Comp Denoms (big/full)");
			
			offst += grid_rws*mx_grid_x*thrds;
			g_dim  = dim3(blcks - grid_rws*mx_grid_x);
		}

		// Calculates comp denom that's been allocated.
		k_setup_bndry_mats<<< g_dim, b_dim >>>(d_bndrs, d_typs, size, offst);
		showError("Comp Denoms");
	}
	
}

#endif
