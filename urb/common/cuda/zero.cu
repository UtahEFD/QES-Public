/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA kernel to zero a matrix - common functionality.
* Limits: Can "zero" linear device memory upto segments of length:
*         maxGridSize.x * (maxGridSize.y - 1) * 512
*         65535*65534*1 = 4294836225
*                  2^32 = 4294967296 (4byte unsigned int or ?size_t?)
*
*					Limit is 4294836225 elements.
*/

#ifndef ZERO_H
#define ZERO_H

#include <stdio.h>

extern "C" void showError(char* loc);

namespace QUIC 
{

	// Not thoroughly tested, but appears to work.
	__global__ void k_zero(float* M, size_t size, float value) 
	{
		// Indexing
		size_t cI = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

		if(cI < size) {M[cI] = value;}
	}

	extern "C"
	void cudaZero(float* M, size_t size, float value) 
	{
		int device_num = 0;
		cudaGetDevice(&device_num);
		struct cudaDeviceProp dprops;
		cudaGetDeviceProperties(&dprops, device_num);
		
		int mx_grid_x = dprops.maxGridSize[0];
		//int mx_grid_y = dprops.maxGridSize[1];
	
		int thrds = dprops.maxThreadsDim[0] / 2;
		int blcks = (size / thrds) + 1;
		int offst = 0;

		dim3 b_dim(thrds);
		dim3 g_dim(blcks);
		
		if(blcks > mx_grid_x) // New style
		{
			int grid_rws = blcks / mx_grid_x;
			    g_dim    = dim3(mx_grid_x, grid_rws);
			
			// Big / Full
			k_zero<<< g_dim, b_dim >>>(M, size, value); showError("Zero (big/full)");
			
			offst += grid_rws*mx_grid_x*thrds;
			g_dim  = blcks - grid_rws*mx_grid_x;
		}
		
		k_zero<<< g_dim, b_dim >>>(&M[offst], size, value);	showError("Zero");
	}
}

#endif

