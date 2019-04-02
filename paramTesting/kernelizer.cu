#include "kernelizer.h"

template<typename T>
void _cudaCheck(T e, const char* func, const char* call, const int line){
    if(e != cudaSuccess){
        printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

__global__ void init(int*flagsGPU, int x, int y);
__global__ void setDataAs(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int x, int y, float dx, float dy);
__global__ void assignDataAs(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int idx, int x, int y, float dx, float dy);

//__global__ void setDataAsR(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, dim3 dimGridCells, dim3 dimBlockCells, int x, int y, float dx, float dy);
//__global__ void assignDataAsR(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int idx, int x, int y, float dx, float dy);
//__global__ void setDataAsT(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int x, int y, float dx, float dy);

void doTheGPU(vector<DataA> datAs)
{
	auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time


	cudaDeviceProp* dev1 = new cudaDeviceProp();
	cudaGetDeviceProperties(dev1,0);

	int * flags, *flagsGPU;

	flags = new int[X * Y];
	flagsGPU = new int [X * Y]; 

	cudaMalloc( (void**)&flagsGPU, X * Y * sizeof(float) );

	int maxThreadsPerBlock = dev1->maxThreadsPerBlock;

	unsigned long maxGridX = dev1->maxGridSize[0];
	unsigned long maxGridY = dev1->maxGridSize[1];
	unsigned long maxGridZ = dev1->maxGridSize[2];

	if (X * Y > maxThreadsPerBlock * maxGridZ * maxGridY * maxGridX)
	{
		printf("Error: not enough room on GPU to run this size\n");
		return;
	}

	unsigned long tX, tY, bX, bY, bZ;
	tX = 32;
	tY = ((X * Y) / 32) + 1;

	if (tY * tX > (unsigned long)maxThreadsPerBlock)
	{
		unsigned long total = tY * tX;
		tY = maxThreadsPerBlock / tX;
		total = (total / (tY * tX)) + 1;
		if (total > maxGridX)
		{
			printf("Error: too big for now. need to fix\n");
			return;
		}
		else
		{
			bX = total;
			bY = bZ = 1;
		}
	} 
	else
		bX = bY = bZ = 1;

	dim3 dimGridCells(bX, bY, bZ);
	dim3 dimBlockCells(tX, tY, 1);




	init<<<dimGridCells, dimBlockCells>>>(flagsGPU, X, Y);

//DATA A
	


	vector<float> datAsF;
	vector<int> datAsI;

	float* datAsFGPU;
	int* datAsIGPU;

	cudaMalloc( (void**)&datAsFGPU, (4 * datAs.size()) * sizeof(float) );
	cudaMalloc( (void**)&datAsIGPU, (1 + 3 * datAs.size()) * sizeof(int));

	datAsI.push_back(datAs.size());

	for (unsigned int i = 0; i < datAs.size(); i++)
	{
		datAs[i].appendFloatsToBuffer(datAsF);
		datAs[i].appendIntsToBuffer(datAsI);
	}

	cudaMemcpy(datAsFGPU, datAsF.data(), (4 * datAs.size()) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(datAsIGPU, datAsI.data(), ( 1 + 3 * datAs.size()) * sizeof(int), cudaMemcpyHostToDevice);

	setDataAs<<<dimGridCells, dimBlockCells>>>(flagsGPU, datAsFGPU, datAsIGPU, X, Y, DIMX, DIMY);


	cudaCheck(cudaGetLastError()); 

	cudaMemcpy(flags, flagsGPU, X * Y * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(flagsGPU);
	cudaFree(datAsFGPU);
	cudaFree(datAsIGPU);

	auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time
    std::chrono::duration<double> elapsed = finish - start;
    printf("Elapsed time:%lf\n", elapsed.count());   // Print out elapsed execution time    
/*
	for (int i = 0; i < X; i++)
	{
		for (int j = 0; j < Y; j++)
			printf("%d ", flags[i * Y + j]);
		printf("\n");
	}//*/
	return;
}



__global__ void init(int*flagsGPU, int x, int y)
{
	int blockSize = blockDim.x * blockDim.y * blockDim.z;

	//this is not efficient, but it is clear. Speed isn't the goal here
	int idx = threadIdx.x + 
				threadIdx.y * blockDim.x +
				threadIdx.z * blockDim.x * blockDim.y +
				blockIdx.x * blockSize +
				blockIdx.y * blockSize * gridDim.x +
				blockIdx.z * blockSize * gridDim.x * gridDim.y;

	if (idx >= x * y)
		return;

	flagsGPU[idx] = 0;
}

//set up dynamic parallelism to diversify more. Could be a fast solution
__global__ void setDataAs(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int x, int y, float dx, float dy)
{
	int blockSize = blockDim.x * blockDim.y * blockDim.z;

	//this is not efficient, but it is clear. Speed isn't the goal here
	int idx = threadIdx.x + 
				threadIdx.y * blockDim.x +
				threadIdx.z * blockDim.x * blockDim.y +
				blockIdx.x * blockSize +
				blockIdx.y * blockSize * gridDim.x +
				blockIdx.z * blockSize * gridDim.x * gridDim.y;

	if (idx >= x * y)
		return;

	for (int building = 0; building < datAsIGPU[0]; building++)
	{
		int bF = building * 4, bI = building * 3;
		int tX = idx / x;
		int tY = idx % x;

		int inner, outter;

		inner = ((tX >= (int)(datAsFGPU[bF] / dx)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx)) &&
							(tY >= (int)(datAsFGPU[bF + 1] / dy)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy)));

		outter = (( ((tX == (int)(datAsFGPU[bF] / dx - 1)) || (tX == (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
						&&  ((tY >= (int)(datAsFGPU[bF + 1] / dy - 1)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) || 
						( ((tX >= (int)(datAsFGPU[bF] / dx - 1)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
						&&  ((tY == (int)(datAsFGPU[bF + 1] / dy - 1)) || (tY == (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) );

		flagsGPU[idx] =  outter * datAsIGPU[1 + bI + 2] + inner * datAsIGPU[1 + bI + 1] + (outter + inner == 0) * flagsGPU[idx];	
	}//*/

//	assignDataAs<<<1,flagsGPU[0]>>>(flagsGPU, datAsFGPU, datAsIGPU, idx, x, y, dx, dy);
}



__global__ void assignDataAs(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int idx, int x, int y, float dx, float dy)
{
	int bF = threadIdx.x * 4, bI = threadIdx.x * 3;
	int tX = idx / x;
	int tY = idx % x;

	int inner, outter;

	inner = ((tX >= (int)(datAsFGPU[bF] / dx)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx)) &&
						(tY >= (int)(datAsFGPU[bF + 1] / dy)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy)));

	outter = (( ((tX == (int)(datAsFGPU[bF] / dx - 1)) || (tX == (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
					&&  ((tY >= (int)(datAsFGPU[bF + 1] / dy - 1)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) || 
					( ((tX >= (int)(datAsFGPU[bF] / dx - 1)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
					&&  ((tY == (int)(datAsFGPU[bF + 1] / dy - 1)) || (tY == (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) );

	flagsGPU[idx] =  outter * datAsIGPU[1 + bI + 2] + inner * datAsIGPU[1 + bI + 1] + (outter + inner == 0) * flagsGPU[idx];

}
/*

__global__ void setDataAsR(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, dim3 dimGridCells, dim3 dimBlockCells, int x, int y, float dx, float dy)
{
	int idx = threadIdx.x + threadIdx.y * blockDim.x;
	if (idx >= datAsIGPU[0])
		return;

	printf("%d\n", idx);
	assignDataAsR<<<dimGridCells, dimBlockCells>>>(flagsGPU, datAsFGPU, datAsIGPU, idx, x, y, dx, dy);
}


__global__ void assignDataAsR(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int idx, int x, int y, float dx, float dy)
{
	int blockSize = blockDim.x * blockDim.y * blockDim.z;

	int idxT = threadIdx.x + 
				threadIdx.y * blockDim.x +
				threadIdx.z * blockDim.x * blockDim.y +
				blockIdx.x * blockSize +
				blockIdx.y * blockSize * gridDim.x +
				blockIdx.z * blockSize * gridDim.x * gridDim.y;

	int bF = idx * 4, bI = idx * 3;
	int tX = idxT / x;
	int tY = idxT % x;

	int inner, outter;

	inner = ((tX >= (int)(datAsFGPU[bF] / dx)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx)) &&
						(tY >= (int)(datAsFGPU[bF + 1] / dy)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy)));

	outter = (( ((tX == (int)(datAsFGPU[bF] / dx - 1)) || (tX == (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
					&&  ((tY >= (int)(datAsFGPU[bF + 1] / dy - 1)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) || 
					( ((tX >= (int)(datAsFGPU[bF] / dx - 1)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
					&&  ((tY == (int)(datAsFGPU[bF + 1] / dy - 1)) || (tY == (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) );

	flagsGPU[idxT] =  outter * datAsIGPU[1 + bI + 2] + inner * datAsIGPU[1 + bI + 1] + (outter + inner == 0) * flagsGPU[idx];
}

__global__ void setDataAsT(int* flagsGPU, float* datAsFGPU, int* datAsIGPU, int x, int y, float dx, float dy)
{
	int blockSize = blockDim.x * blockDim.y * blockDim.z;

	//this is not efficient, but it is clear. Speed isn't the goal here
	int idx = threadIdx.x + 
				threadIdx.y * blockDim.x +
				threadIdx.z * blockDim.x * blockDim.y +
				blockIdx.x * blockSize +
				blockIdx.y * blockSize * gridDim.x;
	int building = blockIdx.z;

	if (idx >= x * y)
		return;

	int bF = building * 4, bI = building * 3;
	int tX = idx / x;
	int tY = idx % x;

	int inner, outter;

	inner = ((tX >= (int)(datAsFGPU[bF] / dx)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx)) &&
						(tY >= (int)(datAsFGPU[bF + 1] / dy)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy)));

	outter = (( ((tX == (int)(datAsFGPU[bF] / dx - 1)) || (tX == (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
					&&  ((tY >= (int)(datAsFGPU[bF + 1] / dy - 1)) && (tY <= (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) || 
					( ((tX >= (int)(datAsFGPU[bF] / dx - 1)) && (tX <= (int)(datAsFGPU[bF] / dx + datAsFGPU[bF + 2] / dx + 1)))
					&&  ((tY == (int)(datAsFGPU[bF + 1] / dy - 1)) || (tY == (int)(datAsFGPU[bF + 1] / dy + datAsFGPU[bF + 3] / dy + 1))) ) );

	flagsGPU[idx] =  outter * datAsIGPU[1 + bI + 2] + inner * datAsIGPU[1 + bI + 1] + (outter + inner == 0) * flagsGPU[idx];	
}*/