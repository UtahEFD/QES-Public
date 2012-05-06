/**
* Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: CUDA kernel to find the min of a matrix - common functionality.
*         Used in urbViewer as part of QUICurbCUDA.
* Remark: Needs to extended to handle more elements.
*/

#ifndef MAP_MATRIX_TO_INTERVAL_H
#define MAP_MATRIX_TO_INTERVAL_H

extern "C" void showError(char* loc);
extern "C" void cudaMax(float* d_array, int size, float* d_max);
extern "C" void cudaMin(float* d_array, int size, float* d_min);

namespace QUIC 
{

	// Maps the given matrix to the given interval.
	__global__ void k_map_matrix_to_interval
	(
		float* d_mat, int size, 
		float a, float b, 
		float* d_min, float* d_max
	)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;

		if(x < size) 
		{
			float m = *d_min;
			float M = *d_max;		

			if(m == M) {m = 0.f; M = 1.f;}
			if(a == b) {b = a + 1.f;}

			d_mat[x] = (d_mat[x] - m) * (b - a) / (M - m) + a;
		}
	}

	// Be nice to wrap the next three up....
	extern "C"
	void cudaMapMatrixToIntervalWithDeviceLimits
	(
		float* d_mat, int size, 
		float a, float b, 
		float* d_min, float* d_max
	) 
	{
		int threads = 512;
		int blocks = (int) (size / threads);
	
		//float min; 	cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
		//float max;	cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

		//std::cout << "Min = " << min << std::endl;
		//std::cout << "Max = " << max << std::endl;
		//std::cout << "Interval: [" << a << ", " << b << "]" << std::endl;

		k_map_matrix_to_interval<<< dim3(blocks + 1), dim3(threads) >>>
		(
			d_mat, size, 
			a, b, 
			d_min, d_max
		); 
		showError("Map Matrix to Interval");
	}

	extern "C"
	void cudaMapMatrixToInterval(float* d_mat, int size, float a, float b) 
	{
		float* d_min; cudaMalloc((void**) &d_min, sizeof(float));
		float* d_max; cudaMalloc((void**) &d_max, sizeof(float));

		cudaMin(d_mat, size, d_min);
		cudaMax(d_mat, size, d_max);

		cudaMapMatrixToIntervalWithDeviceLimits(d_mat, size, a, b, d_min, d_max);
	}
}

#endif

