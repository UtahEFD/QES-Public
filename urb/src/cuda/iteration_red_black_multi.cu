/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Attempting to use multiple devices to iterate faster.
* Remark: Uses an older version of the iteration kernel, not up-to-date with
*         the kernel in revision ~340.
*/

#ifndef ITERATION_RED_BLACK_MULTI_H
#define ITERATION_RED_BLACK_MULTI_H

extern "C" void showError(char const* loc);

namespace QUIC 
{

	/**
	 * SOR Iteration Kernel (v2)
	 *
	 * @note Current working and fastest kernel
	 * 
	 * This kernel is the guts behind the Red-Black SOR scheme
	 * used to solve for the lagrange multipliers.
	 *
	 * @param d_e a device pointer to boundry matrix e
	 * @param d_f a device pointer to boundry matrix f
	 * @param d_g a device pointer to boundry matrix g
	 * @param d_h a device pointer to boundry matrix h
	 * @param d_m a device pointer to boundry matrix m
	 * @param d_n a device pointer to boundry matrix n
	 * @param d_r a device pointer to divergence matrix r
	 * @param one_less_omegarelax 1 - over-relaxation scheme coefficient
	 * @param d_p1 a device pointer to lagrange multipliers matrix (INPUT & OUTPUT)
	 * @param d_p2 a device pointer to the old lagrange multipliers matrix (NO USED)
	 * @param nx the number of cells in the x-dimension
	 * @param ny the number of cells in the y-dimension
	 * @param nz the number of cells in the z-dimension
	 * @param pass the current pass for Red-Black, odd or even distinguishes
	 */
	__global__ void k_iter_red_black_multi
	(
		float* d_e, float* d_f, float* d_g, float* d_h, float* d_m, float* d_n, 
		float* d_r, float one_less_omegarelax, 
		float* d_p1, float* d_p2, 
		int nx, int ny, int pass
	)
	{
		extern __shared__ float data[];

		int stick_length = blockDim.x;

		float* fStick = (float*)&data[0 * stick_length]; //front
		float* bStick = (float*)&data[1 * stick_length]; //back
		float* uStick = (float*)&data[2 * stick_length]; //up
		float* dStick = (float*)&data[3 * stick_length]; //down

		float* lValue = (float*)&data[4 * stick_length + 0]; //Far left value
		float* cStick = (float*)&data[4 * stick_length + 1]; //center
		float* rValue = (float*)&data[5 * stick_length + 1]; //Far right value

		int bidr = (int) blockIdx.x / (nx / stick_length);	
		int bidc =       blockIdx.x % (nx / stick_length);


		if(bidr == 0 || bidr == ny - 1) {return;} //Don't do any calculations for top and bottom row, because they are part of the boundry.

		int tidx = threadIdx.x;
		int bidx = blockIdx.x;
		int bidy = blockIdx.y;
		int slice_size = nx * ny;

		int offset = bidx * (stick_length);
		int cI = offset + tidx;

		int fI = cI + nx;
		int bI = cI - nx;
		int uI = cI + slice_size;
		int dI = cI - slice_size;

		//Load the data
		cStick[tidx] = d_p1[cI];
		fStick[tidx] = d_p1[fI];
		bStick[tidx] = d_p1[bI];
		uStick[tidx] = d_p1[uI];
		dStick[tidx] = d_p1[dI];

		//New RB SOR //Switch every row, slice and pass...
		if( (pass + bidy + bidr + tidx) & 1) {return;}

		int rVI = offset + stick_length; //value on the right end of cStick
		int lVI = offset - 1;   		 //value on the left end of cStick

/*
 *!!!! Could probably get away with loading rValue and lValue no matter what.
 *!!!! worst case they are from another slice and don't get used.
 *!!!! then only two checks. One for if cI is in 0 col and one for 
 *!!!! if cI is in nx - 1 col.
 *!!!! Testing removes uncoalesced memory transfers. These if statments make things faster!?!
 *!!!! Seems as though if statements make things faster.
 */

		if(bidc == 0 && bidc == (nx / stick_length) - 1) 
		{
			//Don't try to set the values for rValue and lValue.

			//Don't try to calculate the boundries.
			if(tidx == 0) 				    {return;}
			else if(tidx == stick_length - 1) {return;}
		}
		//This block includes the left boundry. Do something different.
		else if(bidc == 0) 
		{
			rValue[0] = d_p1[rVI];

			//Don't do any calculations for left boundry; although, we used it to load data.
			if(tidx == 0) {return;}
		} 

		//This block includes the right boundry. Do something different.
		else if(bidc == (nx / stick_length) - 1) 
		{
			lValue[0] = d_p1[lVI];

			//Don't do any calculations for right boundry; although, we used it to load data.
			if(tidx == stick_length - 1) {return;}
		}
		//This block shouldn't include any boundries. Load an extra element on each side of the cStick.
		else 
		{
			lValue[0] = d_p1[lVI];
			rValue[0] = d_p1[rVI];
		}

		__syncthreads();

		//Do the SOR calculation.

		cStick[tidx] = (
						   (d_e[cI] * cStick[tidx + 1] + d_f[cI] * cStick[tidx - 1])
						 + (d_g[cI] * fStick[tidx]     + d_h[cI] * bStick[tidx])
						 + (d_m[cI] * uStick[tidx]     + d_n[cI] * dStick[tidx]) 
						 -  d_r[cI] 
						)
						+
						one_less_omegarelax * cStick[tidx];

		//Write only the values in cStick that have been modified.
		d_p1[cI] = cStick[tidx];
	}



	/**
	 * SOR Iteration Wrapper
	 *
	 * Wraps up k_iter_red_black_v2. This wrapper calls the kernel twice,
	 * once for the 'red' pass and once for the 'black' pass. One call of
	 * this wrapper equates to one iteration of Red-Black SOR scheme on
	 * lagrange matrix d_p1.
	 * 
	 * @note Shared memory allocated.
	 *
	 * @param d_e a device pointer to boundry matrix e
	 * @param d_f a device pointer to boundry matrix f
	 * @param d_g a device pointer to boundry matrix g
	 * @param d_h a device pointer to boundry matrix h
	 * @param d_m a device pointer to boundry matrix m
	 * @param d_n a device pointer to boundry matrix n
	 * @param d_r a device pointer to divergence matrix r
	 * @param one_less_omegarelax 1 - over-relaxation scheme coefficient
	 * @param d_p1 a device pointer to lagrange multipliers matrix (INPUT & OUTPUT)
	 * @param d_p2 a device pointer to the old lagrange multipliers matrix (NO USED)
	 * @param nx the number of cells in the x-dimension
	 * @param ny the number of cells in the y-dimension
	 * @param nz the number of cells in the z-dimension
	 */
	extern "C"
	void cudaIterRBMulti
	(
		float* d_e, float* d_f, float* d_g, float* d_h, float* d_m, float* d_n, 
		float* d_r, float one_less_omegarelax, 
		float* d_p1, float* d_p2, 
		int nx, int ny, int group_size, int threads
	)
	{
		dim3 dimGrid(group_size);
		dim3 dimBlock(threads);
		int sharedMem = (threads * 5 + 2) * sizeof(float);

		k_iter_red_black_multi<<< dimGrid, dimBlock, sharedMem >>> 
		(
			d_e, d_f, d_g, d_h, d_m, d_n, 
			d_r, one_less_omegarelax, 
			d_p1, d_p2, 
			nx, ny, 0
		); showError("Iteration_multi_p0");
		k_iter_red_black_multi<<< dimGrid, dimBlock, sharedMem >>> 
		(
			d_e, d_f, d_g, d_h, d_m, d_n, 
			d_r, one_less_omegarelax, 
			d_p1, d_p2, 
			nx, ny, 1
		); showError("Iteration_multi_p1");
	}
}

#endif
