/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA kernel to find the sum of a matrix - common functionality.
*         Used in urbCUDA for finding abse of p1 - p2.
* Limits: Can sum a maximum of ~67107840 elements.
* Remark: Old summing kernel.
*/

#ifndef SUM_H
#define SUM_H

extern "C" void showError(char* loc);
extern "C" void cudaZero(float* d_abse, size_t nz, float value);

namespace QUIC
{
	__global__ void k_sum(float* d_array, float* d_sums) 
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
		int baseI = blockIdx.x*chunk + tidx;
		int s1I = baseI + 0*section;
		int s2I = baseI + 1*section;
		int s3I = baseI + 2*section;
		int s4I = baseI + 3*section;

		// Load Data
		s_1[tidx] = d_array[s1I];
		s_2[tidx] = d_array[s2I];
		s_3[tidx] = d_array[s3I];
		s_4[tidx] = d_array[s4I];

		__syncthreads();

		while(section >= 1) 
		{
			
			if(tidx < section) 
			{
				//Mini Reduce
				s_1[tidx] = s_1[tidx] + s_2[tidx] + s_3[tidx] + s_4[tidx];
			}

			section >>= 2;

			// Change where the section pointers are.
			s_1 = (float*)&data[0 * section];
			s_2 = (float*)&data[1 * section];
			s_3 = (float*)&data[2 * section];
			s_4 = (float*)&data[3 * section];

			__syncthreads();
		}

		if(tidx == 0) {d_sums[blockIdx.x] = s_1[0];}
	}

	__global__ void k_find_simple_sum(float* d_array, int size, float* d_sum) 
	{
		float sum = 0.f;
		for(int i = 0; i < size; i++) {sum += d_array[i];}
		*d_sum = sum;
	}

	/**
	* Check for okay size in datamodule.cpp checkParametersQ
	* Looks like it could be expanded if ever needed. Use the
	* Y dimension for the blocks.
	*/
	extern "C"
	void cudaSum(float* d_array, int size, float* d_sum) 
	{
		int threads = 256; //Must be power of 4.
		int chunk   = threads*4; //Must be power of 4 => threads a power of 4.		
		// Find largest power of 4 size
		int blocks     = int( size / chunk);
		int left_overs = size - blocks*chunk;
		int sharedMem  = chunk*sizeof(float);

		float* d_sums; cudaMalloc((void**) &d_sums, (blocks + 1)*sizeof(float));
		cudaZero(d_sums, blocks + 1, 0.f);

		k_sum<<< dim3(blocks), dim3(threads), sharedMem >>>(d_array, d_sums); 
		showError("sum");
		
		//Reduce the blocks if enough.
		if(blocks >= chunk) 
		{
			cudaSum(d_sums, blocks, &d_sums[0]);
			blocks = 1;
		}

		if(left_overs) // Then eat them!!
		{
			k_find_simple_sum<<< dim3(1), dim3(1) >>>
			(
				&d_array[size - left_overs], 
				left_overs, 
				&d_sums[blocks]
			); 
			showError("Simple Sum (left_overs)");
		}

		k_find_simple_sum<<< dim3(1), dim3(1) >>>
		(
			d_sums, 
			blocks + 1, 
			&d_sums[0]
		);
		showError("Simple Sum");

		cudaMemcpy(d_sum, &d_sums[0], sizeof(float), cudaMemcpyDeviceToDevice);		
		cudaFree(d_sums);
	}
}

#endif
