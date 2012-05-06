/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA kernel to find the absolute difference of two matrices on the 
*         device. Used in urbCUDA as part of QUICurbCUDA.
* Remark: Sum kernel is faster than this one, which seems strange.
*/

#ifndef ABS_DIFF_H
#define ABS_DIFF_H

extern "C" void showError(char* loc);

namespace QUIC
{
	/** Error Calculation Kernel
	 *
	 * This kernel works entry by entry, finding the absolute value of the 
	 * difference of values in d_p1 and values in d_p2, and places the result 
	 * into d_err. 
	 *
	 * @param d_err the device pointer to what will hold the absolute difference 
	 *        of d_p1 and d_p2.
	 * @param d_p1 the device pointer to one of the two matrices to have their 
	 *        difference found.
	 * @param d_p2 the other one.
	 * @param size the size of each matrix
	 * @param offst the offst into each matrix
	 *
	 */
	__global__ 
	void k_abs_diff
	(
		float* d_err, 
		float* d_p1, float* d_p2, 
		unsigned int size, unsigned int offst
	)
	{
		int cI = offst + (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

		if(cI < size) {d_err[cI] = fabs(d_p1[cI] - d_p2[cI]);}
	}


	/** 
	 * Error Calculation Wrapper
	 * 
	 * Wraps up k_absolute_difference.
	 *
	 * @param d_err the device pointer to what will hold the absolute difference 
	 *        of d_p1 and d_p2.
	 * @param d_p1 the device pointer to one of the two matrices to have their 
	 *        difference found.
	 * @param d_p2 the other one.
	 * @param size the size of each matrix for this operation.
	 */
	extern "C"
	void cudaAbsDiff
	(
		float* d_err, 
		float* d_p1, float* d_p2, 
		unsigned int size
	) 
	{
		int device_num = 0;
		cudaGetDevice(&device_num);
		struct cudaDeviceProp dprops;
		cudaGetDeviceProperties(&dprops, device_num);
		
		int mx_grid_x = dprops.maxGridSize[0];
		//int mx_grid_y = dprops.maxGridSize[1];

		int thrds = 256;
		int blcks = (size / thrds) + 1;
		int offst = 0;

		dim3 b_dim(thrds);
		dim3 g_dim(blcks);

		if(blcks > mx_grid_x) // New style
		{
			int grid_rws = blcks / mx_grid_x;
			    g_dim    = dim3(mx_grid_x, grid_rws);
		
			k_abs_diff<<< g_dim, b_dim >>>(d_err, d_p1, d_p2, size, offst);
			showError("Absolute Difference (big/full)");
			
			offst += grid_rws*mx_grid_x*thrds;
			g_dim  = blcks - grid_rws*mx_grid_x;
		}

		k_abs_diff<<< g_dim, b_dim >>>(d_err, d_p1, d_p2, size, offst);
		showError("Absolute Difference");
	}
}

#endif

