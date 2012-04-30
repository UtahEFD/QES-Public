/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA kernel to find the max of a matrix - common functionality.
*         Used in urbViewer and urbCUDA (for diffusion) as part of QUICurbCUDA.
* Remark: Needs to be extended to handle more elements.
*/

#ifndef MAX_H
#define MAX_H

#include <stdio.h>

extern "C" void showError(char* loc);
extern "C" void cudaZero(float* d_abse, size_t nz, float value);

namespace QUIC 
{
	
	__global__ void k_max(float* d_array, float* d_maxs) 
	{
		extern __shared__ float data[];

		int section = blockDim.x;
		int chunk   = section * 4;

		// Address Data
		float* s_1 = (float*)&data[0 * section];
		float* s_2 = (float*)&data[1 * section];
		float* s_3 = (float*)&data[2 * section];
		float* s_4 = (float*)&data[3 * section];
	
		int tidx = threadIdx.x;
		int s1I = blockIdx.x * chunk + 0 * section + tidx;
		int s2I = blockIdx.x * chunk + 1 * section + tidx;
		int s3I = blockIdx.x * chunk + 2 * section + tidx;
		int s4I = blockIdx.x * chunk + 3 * section + tidx;

		// Load Data
		s_1[tidx] = d_array[s1I];
		s_2[tidx] = d_array[s2I];
		s_3[tidx] = d_array[s3I];
		s_4[tidx] = d_array[s4I];

		__syncthreads();

		float max;

		while(section >= 1) 
		{
			if(tidx < section) 
			{
				//Mini Reduce
				max = s_1[tidx];
				if(s_2[tidx] > max) {max = s_2[tidx];}
				if(s_3[tidx] > max) {max = s_3[tidx];}
				if(s_4[tidx] > max) {max = s_4[tidx];}
				s_1[tidx] = max;
			}

			section >>= 2;

			// Change where the section pointers are.
			s_1 = (float*)&data[0 * section];
			s_2 = (float*)&data[1 * section];
			s_3 = (float*)&data[2 * section];
			s_4 = (float*)&data[3 * section];

			__syncthreads();
		}

		if(tidx == 0) {d_maxs[blockIdx.x] = s_1[0];}
	}

	__global__ void k_find_simple_max(float* d_array, int size, float* d_max) 
	{
		float cur_max = d_array[0];
		for(int i = 0; i < size; i++) 
		{
			if(d_array[i] > cur_max) {cur_max = d_array[i];}
		}
		*d_max = cur_max;
	}

	extern "C"
	void cudaMax(float* d_array, int size, float* d_max) 
	{

		// \\ todo fix this better. When a device manager class is had, use it
		//    to get the proper value...
		int max_blocks = /*?*/ 65535; /*?*/ //From device info...
		// Gives a total of 65535 * 1024 sized array that max can be found of.

		int threads = 256; 
		int chunk   = threads*4; //Must be power of 4 => threads a power of 4.		
		// Find largest power of 4 size
		int blocks     = int(size / chunk);
		int left_overs = size - blocks*chunk;
		int sharedMem  = chunk*sizeof(float);

		// Make sure the size if doable...
		if(size > max_blocks * chunk) 
		{
			printf
			(
				"cudaMax cannot max arrays larger than %d elements.", 
				max_blocks * chunk
			);
			return;
		}

		float* d_maxs; cudaMalloc((void**) &d_maxs, (blocks + 1) * sizeof(float));
		cudaZero(d_maxs, blocks + 1, 0.f);

		if(blocks > 0) 
		{
			k_max<<< dim3(blocks), dim3(threads), sharedMem >>>(d_array, d_maxs); 
			showError("Max");
		}
		
		//Reduce the blocks if enough. 
		if(blocks >= chunk) 
		{
			cudaMax(d_maxs, blocks, &d_maxs[0]);
			blocks = 1;
		}

		if(left_overs) // Then eat them!!
		{
			k_find_simple_max<<< dim3(1), dim3(1) >>>
			(
				&d_array[size - left_overs], 
				left_overs, 
				&d_maxs[blocks]
			); 
			showError("Simple Max (left_overs)");
		}

		k_find_simple_max<<< dim3(1), dim3(1) >>>
		(
			d_maxs, 
			blocks + 1, 
			&d_maxs[0]
		); 
		showError("Simple Max");

		cudaMemcpy(d_max, &d_maxs[0], sizeof(float), cudaMemcpyDeviceToDevice);		
		cudaFree(d_maxs);
	}
}

#endif
