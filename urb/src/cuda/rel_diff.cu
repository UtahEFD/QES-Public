/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Common functionality using CUDA to find relative difference of two
*         matrices. Used as part of the matrix_viewer for QUICurbCUDA.
* Remark: Needs to be extended to handle more elements.
*/

#ifndef REL_DIFF_H
#define REL_DIFF_H

extern "C" void showError(char* loc);

namespace QUIC
{

	/** Percentage Error Calculation Kernel
	 *
	 * This kernel works entry by entry, finding the relative difference between
	 * d_p1 and d_p2, where E_r = fabs(d_p1 - d_p2) / d_p2. The result is placed 
	 * into d_rltv_dff. 
	 *
	 * @param d_rltv_dff the device pointer to what will hold the relative 
	          difference of d_p1 and d_p2.
	 * @param d_p1 the device pointer to one of the two matrices to have their 
	          difference found.
	 * @param d_p2 the other one.
	 * @param size the number of elements in both d_p1 and d_p2.
	 *
	 */
	__global__ void k_rel_diff
	(
		float* d_rltv_dff, 
		float* d_p1, float* d_p2, int size, 
		float tol
	)
	{
		// Indexing
		int cI = blockIdx.x * blockDim.x + threadIdx.x;

		float x_t = d_p2[cI];
		float x_a = d_p1[cI];

		if(cI < size)
		{
			d_rltv_dff[cI] = (fabs(x_t - x_a) < tol) ? 0.f : fabs(x_a - x_t) / x_t ;
		}
	}


	/** 
	 * Percentage Error Calculation Wrapper
	 * 
	 * Wraps up k_rel_diff.
	 *
	 * @param d_rltv_dff the device pointer to what will hold the relative 
	          difference of d_p1 and d_p2.
	 * @param d_p1 the device pointer to one of the two matrices to have their 
	          difference found.
	 * @param d_p2 the other one.
	 * @param size the number of elements in both d_p1 and d_p2.
	 */
	extern "C"
	void cudaRelDiff
	(
		float* d_rltv_dff, 
		float* d_p1, float* d_p2, int size, 
		float tol
	)
	{
		int threads = 512;
		int blocks = (int) (size / threads);

		k_rel_diff<<< dim3(blocks + 1), dim3(threads) >>>
		(
			d_rltv_dff, 
			d_p1, d_p2, size, 
			tol
		); 
		showError("Relative Difference");
	}
}

#endif

