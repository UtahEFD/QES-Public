/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA kernel to for SOR iteration scheme. 
* Source: Adapted from sor3d.f90 in QUICurbv5.? (Fortran)
*/

#ifndef ITERATION_RED_BLACK_COMP_H
#define ITERATION_RED_BLACK_COMP_H

#include <stdio.h>
#include "quic/boundaryMatrices.h"

extern "C" void showError(char const* loc);

namespace QUIC 
{

	__global__ void k_cmprssdRBSOR_v2
	(
		float* d_p1, float* d_p2, boundaryMatrices d_bndrs, float* d_r, 
		float omegarelax, float one_less_omegarelax, float A, float B, float dx, 
		int offst, int pass
	)
	{
		extern __shared__ float data[];

		int stick_length = blockDim.x;
		
		// Find absolute index.
		int tidx     = threadIdx.x;
		int blck_cnt = blockIdx.y*gridDim.x + blockIdx.x; // How many blocks in

		unsigned int fbI = offst + blck_cnt*stick_length; // First index in block.
		unsigned int cI  = fbI + tidx; // Absolute index
		
		// For stencil / masking
		int cmprssd = d_bndrs.cmprssd[cI];
		
		// Get pointers to the shared data space.
		float* fStick = (float*)&data[0*stick_length]; //front
		float* bStick = (float*)&data[1*stick_length]; //back
		float* uStick = (float*)&data[2*stick_length]; //up
		float* dStick = (float*)&data[3*stick_length]; //down

		float* lValue = (float*)&data[4*stick_length + 0]; //Far left value
		float* cStick = (float*)&data[4*stick_length + 1]; //center
		float* rValue = (float*)&data[5*stick_length + 1]; //Far right value
		
		// Get row and slice size of domain.
		int row_size = d_bndrs.dim.x;
		int slc_size = d_bndrs.dim.x*d_bndrs.dim.y;
		// Find absolute indices for neighbors needed in SOR calculation.
		int bI  = cI - row_size;		
		int fI  = cI + row_size;
		int dI  = cI - slc_size;
		int uI  = cI + slc_size;

		//Load the data
		cStick[tidx] = d_p1[cI];
		bStick[tidx] = d_p1[bI];
		fStick[tidx] = d_p1[fI];
		dStick[tidx] = d_p1[dI];		
		uStick[tidx] = d_p1[uI];

		// Only have one thread do the dirty work on the ends. Otherwise what are 
		// probably bank conflicts occur and this version is slower than previous.
		// Accesses out of bounds are handled by starting past the bottom slice and 
		// having the device arrays padded when allocated.
		//
		// Even faster than having two seperate threads. (Prolly the extra if).
		if(tidx == 0)
		{
			int lVI = fbI - 1; // value on the left end of cStick

			// Half of the uncoalesced global accesses.
			// Never a problem if first slice skipped.
			//lValue[0] = (cI > 0) ? d_p1[lVI] : 0. ; 
			lValue[0] = d_p1[lVI];

			int rVI = fbI + stick_length; // value on the right end of cStick			

			// Half of the uncoalesced global accesses.
			// Never a problem since 127 extra elements always added.
			//rValue[0] = (cI < slc_size*d_bndrs.dim.z) ? d_p1[rVI] : 0. ;
			rValue[0] = d_p1[rVI];
		}
		
		__syncthreads();


		// Don't do anything if first/last slice/row/col.
		// Don't do anything if not cell's pass.
		if(decodePassMask(cmprssd) ^ pass && !decodeDomainMask(cmprssd))
		{
			// Decompress the boundry conditions
			float e, f, g, h, m, n, dmmy;
			decodeBoundary(cmprssd, e, f, g, h, m, n, dmmy, dmmy, dmmy);

			//Do the SOR calculation.
			cStick[tidx] =  d_bndrs.denoms[cI]
											*
											(
														(e*cStick[tidx + 1] + f*cStick[tidx - 1])
												+ A*(g*fStick[tidx]     + h*bStick[tidx])
												+ B*(m*uStick[tidx]     + n*dStick[tidx]) 
												- dx*dx*d_r[cI] 
											)
											+
											one_less_omegarelax * cStick[tidx];

			d_p1[cI] = cStick[tidx];
		}
	}

	// Assumes that d_p1, d_p2, d_bndrs and d_r are each linear arrays that are
	// divisible by 128. Padding should have been applied at the ends during 
	// allocation; otherwise, seg faults will occur.
	extern "C"
	void cudaIterCmprssdRBSOR
	(
	  float* d_p1, float* d_p2,
	  boundaryMatrices d_bndrs, float* d_r, 
	  float omegarelax, float one_less_omegarelax, 
	  float A, float B, float dx
	)
	{
		// Get max block x dim.
		int device_num = 0;
		cudaGetDevice(&device_num);
		struct cudaDeviceProp dprops;
		cudaGetDeviceProperties(&dprops, device_num);
		int mx_grid_x = dprops.maxGridSize[0];
	
		// Find block size.
		// Shave off a bit by skipping the first and last slice.
		int slc_size = d_bndrs.dim.x*d_bndrs.dim.y;
		int size  = slc_size*(d_bndrs.dim.z - 2); // Don't top or bottom slice.
		int thrds = 128;
		int blcks = (size / thrds) + 1; // Don't miss the extras. Okay because of padding.
		int offst = slc_size;
		
		// Basic block dimensions and shared Memory
		dim3 b_dim(thrds);
		dim3 g_dim(blcks);
		int sMem = (thrds*5 + 2)*sizeof(float);
		
		if(blcks > mx_grid_x) // New style
		{
			int grid_rws = blcks / mx_grid_x;
			    g_dim    = dim3(mx_grid_x, grid_rws);
		
			// Big / Full
			k_cmprssdRBSOR_v2 <<< g_dim, b_dim, sMem >>> 
			(
				d_p1, d_p2, d_bndrs, d_r, 
				omegarelax, one_less_omegarelax, A, B, dx, 
				offst, 0
			); 
			showError("Iteration_comp_p0 (big/full)");
		
			k_cmprssdRBSOR_v2 <<< g_dim, b_dim, sMem >>> 
			(
				d_p1, d_p2, d_bndrs, d_r, 
				omegarelax, one_less_omegarelax, A, B, dx, 
				offst, 1
			); 
			showError("Iteration_comp_p1  (big/full)");
			
			offst += grid_rws*mx_grid_x*thrds;
			g_dim  = dim3(blcks - grid_rws*mx_grid_x);
		}

		k_cmprssdRBSOR_v2 <<< g_dim, b_dim, sMem >>> 
		(
			d_p1, d_p2, d_bndrs, d_r, 
			omegarelax, one_less_omegarelax, A, B, dx, 
			offst, 0
		); 
		showError("Iteration_comp_p0");
		
		k_cmprssdRBSOR_v2 <<< g_dim, b_dim, sMem >>> 
		(
			d_p1, d_p2, d_bndrs, d_r, 
			omegarelax, one_less_omegarelax, A, B, dx, 
			offst, 1
		); 
		showError("Iteration_comp_p1");	
	}
}

#endif

