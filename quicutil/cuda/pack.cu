/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA kernel to pack elements from 4 matrices into a OpenGL buffer for
*         accessing with RGBA in a shader. Used in urbViewer as part of 
*         QUICurbCUDA to view velocities (from euler).
* Remark: Needs to be extended to handle more elements.
*/

#ifndef PACK_H
#define PACK_H

extern "C" void showError(char* loc);

namespace QUIC 
{

	__global__ void k_pack
	(
		float* d_packee, 
		float* d_per1, float* d_per2, float* d_per3, float* d_per4, 
		int size
	)
	{
		extern __shared__ float data[];

		int stck_lngth = blockDim.x;

		float* pcker1 = (float*) &data[stck_lngth * 0];
		float* pcker2 = (float*) &data[stck_lngth * 1];
		float* pcker3 = (float*) &data[stck_lngth * 2];
		float* pcker4 = (float*) &data[stck_lngth * 3];
		float* packee = (float*) &data[stck_lngth * 4];

		// Indexing
		int tidx = threadIdx.x;
		int cI = blockIdx.x * blockDim.x + threadIdx.x;

		// Packing offset
		int pOS = blockIdx.x * blockDim.x * 4;

		// Load data
		pcker1[tidx] = d_per1[cI];
		pcker2[tidx] = d_per2[cI];
		pcker3[tidx] = d_per3[cI];
		pcker4[tidx] = d_per4[cI];

		__syncthreads();

		// Interleave data
		packee[tidx * 4 + 0] = pcker1[tidx];
		packee[tidx * 4 + 1] = pcker2[tidx];
		packee[tidx * 4 + 2] = pcker3[tidx];
		packee[tidx * 4 + 3] = pcker4[tidx];

		__syncthreads();

		// Store data in new matrix
		d_packee[pOS + stck_lngth * 0 + tidx] = packee[stck_lngth * 0 + tidx];
		d_packee[pOS + stck_lngth * 1 + tidx] = packee[stck_lngth * 1 + tidx];
		d_packee[pOS + stck_lngth * 2 + tidx] = packee[stck_lngth * 2 + tidx];
		d_packee[pOS + stck_lngth * 3 + tidx] = packee[stck_lngth * 3 + tidx];
	}

	extern "C"
	void cudaPack
	(
		float* d_packee, 
		float* d_per1, float* d_per2, float* d_per3, float* d_per4, 
		int size
	) 
	{
		int threads = 256;
		int blocks = (int) size / threads;
		int sharedMem = (threads * 4 + threads * 4) * sizeof(float);

		k_pack<<< dim3(blocks), dim3(threads), sharedMem >>>
		(
			d_packee, 
			d_per1, d_per2, d_per3, d_per4, 
			size
		); 
		showError("Pack");

		int left_overs = size - blocks * threads;

		if(left_overs) 
		{
			sharedMem = (left_overs * 4 + left_overs * 4) * sizeof(float);
			k_pack<<< dim3(1), dim3(left_overs), sharedMem >>>
			(
				d_packee, 
				d_per1, d_per2, d_per3, d_per4, 
				left_overs
			);
			showError("Pack");
		}
	}
}

#endif

