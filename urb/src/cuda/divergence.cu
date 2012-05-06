/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA-nizing QUICurb
* Source: Adapted from divergence.f90 calculations from QUICurb.
*/

#ifndef DIVERGENCE_H
#define DIVERGENCE_H 1

#include "quicutil/velocities.h"

extern "C" void showError(char const* loc);

namespace QUIC 
{
	/**
	 * Divergence Kernel
	 *
	 * This kernel calculates the divergence matrix r.
	 *
	 * @param d_r device pointer to divergence matrix r (OUTPUT)
	 * @param d_vels struct holding initial velocity matrices.
	 * @param alpha1 @todo doc: what's alpha1?
	 * @param dx_inv 1 over the size of cells in the x-dimension
	 * @param dy_inv 1 over the size of cells in the y-dimension
	 * @param dz_inv 1 over the size of cells in the z-dimension
	 */
	__global__ void k_divergence
	(
		float* d_r, velocities d_vels, float alpha1, 
		float dx_inv, float dy_inv, float dz_inv,
		unsigned int slc_size, unsigned int size, unsigned int offst
	) 
	{
		int nx         = d_vels.dim.x - 1;
		int grid_row   = d_vels.dim.x;
		int grid_slice = d_vels.dim.x*d_vels.dim.y;
		
		int fbI = offst + (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x;
		int cI  = fbI + threadIdx.x;
		int gI  = cI + int(cI / nx) + grid_row*int(cI / slc_size);

		if(cI < size) // Don't access memory outside d_r range.
		{
			d_r[cI] = (-2.f*alpha1*alpha1) 
								* 
								(
									dx_inv*(d_vels.u[gI + 1]          - d_vels.u[gI]) 
								+ dy_inv*(d_vels.v[gI + grid_row]   - d_vels.v[gI]) 
								+ dz_inv*(d_vels.w[gI + grid_slice] - d_vels.w[gI])
								);
		}
	}


	/**
	 * Divergence Wrapper
	 * 
	 * Wraps up k_divergence.
	 *
	 * @note Shared memory is allocated.
 	 *
	 * @param d_r device pointer to divergence matrix r (OUTPUT)
	 * @param d_vels struct holding initial velocity matrices.
	 * @param alpha1 @todo doc: what's alpha1?
	 * @param dx the size of cells in the x-dimension
	 * @param dy the size of cells in the y-dimension
	 * @param dz the size of cells in the z-dimension
	 */
	extern "C"
	void cudaDivergence
	(
		float* d_r, velocities d_vels, 
		float alpha1, 
		float dx, float dy, float dz
	) 
	{
		// Get max block x dim.
		int device_num = 0;
		cudaGetDevice(&device_num);
		struct cudaDeviceProp dprops;
		cudaGetDeviceProperties(&dprops, device_num);
		int mx_grid_x = dprops.maxGridSize[0];
	
		// Initial velocities should be what starts in d_vels (per solve).
		int nx = d_vels.dim.x - 1;
		int ny = d_vels.dim.y - 1;
		int nz = d_vels.dim.z - 1;
	
		// Find block size.
		unsigned int slc_size = nx*ny;
		unsigned int size     = slc_size*nz;
		unsigned int offst    = 0;		
		
		int thrds = 128;
		int blcks = (size / thrds) + 1;

		dim3 g_dim(blcks);
		dim3 b_dim(thrds);

		// Save some operations.
		float dx_inv = 1.f / dx;
		float dy_inv = 1.f / dy;
		float dz_inv = 1.f / dz;

		if(blcks > mx_grid_x) // New style
		{
			int grid_rws = blcks / mx_grid_x;
			    g_dim    = dim3(mx_grid_x, grid_rws);
		
			// Big / Full
			k_divergence<<< g_dim, b_dim >>>
			(
				d_r, d_vels, alpha1, 
				dx_inv, dy_inv, dz_inv,
				slc_size, size, offst
			);
			showError("Divergence (big/full)");
			
			offst += grid_rws*mx_grid_x*thrds;
			g_dim  = dim3(blcks - grid_rws*mx_grid_x);
		}

		// Calculates comp denom that's been allocated.
		k_divergence<<< g_dim, b_dim >>>
		(
			d_r, d_vels, alpha1, 
			dx_inv, dy_inv, dz_inv,
			slc_size, size, offst
		);
		showError("Divergence");
	}
}

#endif

