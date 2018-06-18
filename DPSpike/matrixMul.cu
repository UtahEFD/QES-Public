/*
* This program randomly generates two matricies of size SxS and then
* uses a GPU to multply them together. The generated matricies as 
* well as the resulting product of the two are then output to the 
* terminal.
*
* author: Zach Patterson
*/
#include <stdio.h>
#include <stdlib.h>

#define S 20 //Width/Height of the matrix
#define STOTAL (S * S) //number of all elements
#define BX 5 //length of the grid of blocks
#define BY 2 //height of the grid of blocks
#define TX 5 //length of the block of threads
#define TY 4 //height of the block of threads
#define TZ 2 //depth of the block of threads
#define THREADS (TX * TY * TZ) //Number of threads in one block

/*
* This function is run on each thread of the GPU.
* It identifies what row and column of Md and Nd 
* respectively will be used in the dot product, and
* will place the result of that dot product into the
* Pd matrix.
* Md - primary matrix
* Nd - secondary matrix
* Pd - result matrix
*/
__global__ void matrixMul( const long* Md, const long* Nd, long* Pd)
{

	//the index of the element of the result matrix to be computed
	int threadNum = THREADS * (blockIdx.x + blockIdx.y * BX) +
					TX * (threadIdx.y + threadIdx.z * TY) + threadIdx.x;

	long sum = 0; //running sum of the dot product
	int row = (threadNum / S) * S; //truncates down to multiple of S
	int col = threadNum % S; //gets offset of column
	int k; //index of loops

	//run dot product
	for (k = 0; k < S; k++)
		sum += Md[row + k] * Nd[k * S + col];

	//place dot product into element spot
	Pd[threadNum] = sum;
}


/*
* This function multiplies matricies by allocating 
* memory space to the GPU and then creats a grid 
* of threads to individually handle each element 
* of the resulting matrix.
* M - primary matrix
* N - secondary matrix
* P - result matrix
*/
void multiplyMatricies( const long* M, const long* N, long* P)
{
	int size = S * S * sizeof(long);
	long * Md = (long *)malloc(sizeof(long) * STOTAL); //copy of primary matrix on GPU
	long * Nd = (long *)malloc(sizeof(long) * STOTAL); //copy of secondary matrix on GPU
	long * Pd = (long *)malloc(sizeof(long) * STOTAL); //copy of result matrix on GPU

	//Set sizes of grid and blcoks
	dim3 dimGrid(BX, BY, 1);
	dim3 dimBlock(TX, TY, TZ);

	//Allocate space
	cudaMalloc( (void**)&Md, size);
	cudaMalloc( (void**)&Nd, size);
	cudaMalloc( (void**)&Pd, size);

	//Copy values to GPU memory
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

	//call threads
	matrixMul<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

	//Place copy of result matrix into result matrix
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

	//Free GPU space
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}






/*
----------------------------------------------------------
-------------------GENERATED OUTPUT-----------------------
----------------------------------------------------------

csdev01:~% ./matrixMul
Primary Matrix
8, 7, 7, 7, 8, 5, 5, 5, 4, 6, 5, 2, 9, 5, 3, 3, 5, 0, 2, 4
3, 4, 3, 4, 4, 6, 2, 2, 4, 7, 8, 2, 6, 6, 0, 3, 1, 5, 8, 3
2, 3, 2, 0, 5, 8, 2, 9, 5, 1, 7, 2, 2, 6, 0, 5, 3, 1, 8, 5
1, 0, 8, 6, 6, 0, 1, 5, 3, 6, 5, 1, 3, 7, 6, 8, 0, 4, 8, 4
7, 0, 1, 4, 1, 3, 5, 8, 0, 4, 0, 7, 4, 1, 6, 1, 3, 7, 2, 3
4, 4, 4, 2, 0, 3, 2, 4, 7, 4, 6, 9, 2, 9, 0, 1, 1, 7, 2, 3
9, 6, 1, 3, 0, 4, 3, 1, 4, 3, 6, 0, 4, 7, 1, 0, 1, 7, 9, 5
8, 2, 6, 1, 6, 6, 3, 2, 8, 4, 4, 2, 3, 1, 2, 6, 1, 3, 3, 0
1, 6, 5, 4, 2, 5, 8, 5, 0, 6, 9, 2, 4, 0, 2, 1, 0, 2, 0, 1
7, 2, 5, 7, 8, 0, 9, 9, 0, 2, 3, 2, 4, 0, 1, 1, 9, 6, 3, 4
0, 2, 8, 4, 5, 8, 2, 5, 5, 5, 1, 8, 1, 1, 1, 9, 3, 2, 7, 7
5, 5, 7, 8, 2, 2, 1, 9, 9, 9, 0, 4, 2, 1, 7, 5, 5, 2, 7, 9
5, 5, 9, 4, 7, 3, 4, 6, 9, 9, 8, 1, 1, 1, 5, 7, 6, 2, 9, 6
2, 9, 9, 6, 4, 7, 3, 0, 3, 4, 6, 0, 7, 2, 0, 7, 9, 2, 4, 1
7, 5, 8, 9, 3, 1, 8, 5, 9, 5, 7, 4, 3, 8, 7, 0, 0, 0, 4, 4
3, 8, 3, 2, 3, 4, 4, 0, 8, 9, 9, 4, 6, 2, 6, 0, 7, 5, 6, 5
7, 5, 2, 5, 5, 0, 5, 6, 3, 6, 8, 0, 8, 0, 5, 9, 6, 8, 9, 3
0, 0, 2, 1, 8, 6, 7, 6, 4, 5, 3, 3, 0, 2, 0, 7, 3, 2, 8, 4
5, 7, 7, 3, 3, 2, 4, 7, 1, 9, 8, 6, 9, 1, 6, 8, 6, 7, 2, 8
0, 1, 7, 4, 0, 2, 3, 4, 1, 2, 0, 9, 9, 3, 0, 8, 1, 6, 1, 0


Secondary Matrix
4, 6, 9, 9, 5, 1, 2, 2, 3, 6, 9, 1, 4, 4, 9, 2, 1, 2, 7, 1
9, 0, 1, 7, 1, 1, 0, 1, 7, 0, 3, 3, 2, 0, 0, 5, 7, 6, 9, 2
1, 6, 5, 8, 7, 3, 0, 8, 5, 7, 4, 0, 9, 0, 1, 5, 0, 7, 1, 5
6, 5, 9, 3, 7, 3, 4, 3, 3, 6, 3, 0, 4, 3, 4, 9, 9, 9, 0, 7
5, 1, 8, 7, 2, 8, 6, 1, 5, 8, 3, 2, 4, 4, 8, 6, 4, 6, 1, 5
8, 1, 2, 5, 7, 2, 3, 8, 8, 3, 6, 1, 6, 6, 3, 6, 7, 8, 2, 2
8, 3, 2, 0, 7, 8, 5, 7, 1, 4, 7, 9, 9, 6, 2, 2, 6, 4, 0, 1
1, 3, 3, 0, 4, 2, 6, 8, 1, 9, 7, 7, 9, 4, 8, 0, 0, 8, 4, 2
7, 4, 0, 3, 9, 8, 9, 1, 8, 2, 4, 0, 8, 9, 1, 5, 8, 5, 6, 2
6, 5, 9, 7, 5, 9, 2, 3, 2, 5, 6, 5, 4, 6, 3, 0, 3, 6, 5, 3
6, 9, 3, 1, 6, 0, 0, 8, 8, 8, 6, 3, 6, 4, 7, 1, 1, 4, 6, 4
5, 7, 2, 4, 5, 4, 3, 4, 2, 3, 8, 5, 9, 1, 8, 2, 2, 5, 0, 9
4, 8, 4, 8, 3, 9, 8, 8, 8, 6, 6, 6, 3, 8, 2, 2, 3, 5, 6, 8
9, 8, 6, 0, 7, 4, 8, 6, 3, 8, 4, 4, 4, 8, 8, 2, 6, 1, 8, 0
8, 3, 7, 3, 4, 1, 3, 7, 7, 0, 4, 7, 7, 9, 9, 5, 5, 0, 9, 7
7, 3, 6, 2, 0, 1, 1, 9, 8, 7, 9, 1, 4, 2, 8, 1, 0, 0, 7, 2
1, 4, 5, 2, 4, 4, 6, 3, 0, 9, 8, 7, 5, 7, 5, 4, 4, 1, 6, 6
5, 8, 8, 7, 9, 8, 6, 8, 6, 9, 6, 1, 4, 1, 6, 2, 9, 7, 6, 8
0, 3, 5, 1, 8, 6, 5, 6, 0, 3, 8, 1, 8, 7, 5, 2, 7, 5, 0, 5
0, 6, 0, 9, 3, 0, 0, 2, 5, 7, 2, 8, 8, 1, 1, 0, 1, 1, 0, 4


Result Matrix
504, 453, 489, 481, 479, 414, 380, 478, 459, 561, 541, 352, 546, 457, 457, 336, 386, 473, 420, 390
414, 406, 390, 352, 451, 368, 317, 433, 385, 456, 456, 237, 444, 381, 375, 230, 364, 402, 332, 323
349, 325, 289, 259, 393, 273, 308, 422, 344, 446, 444, 257, 476, 362, 391, 198, 289, 350, 296, 250
382, 404, 455, 323, 428, 340, 309, 470, 370, 492, 443, 247, 488, 370, 443, 238, 311, 362, 327, 347
307, 333, 353, 297, 358, 283, 276, 370, 244, 379, 420, 293, 416, 296, 387, 174, 264, 316, 262, 319
398, 419, 311, 306, 442, 311, 304, 379, 334, 416, 419, 237, 448, 309, 370, 192, 319, 352, 327, 295
371, 384, 352, 339, 438, 299, 289, 367, 332, 397, 422, 220, 401, 348, 345, 199, 365, 329, 342, 267
375, 311, 349, 341, 379, 312, 270, 363, 369, 379, 424, 170, 413, 328, 352, 231, 276, 334, 308, 255
355, 290, 270, 264, 324, 251, 184, 369, 304, 337, 347, 251, 374, 251, 261, 187, 242, 350, 243, 244
324, 375, 430, 355, 416, 365, 340, 407, 274, 543, 483, 338, 491, 334, 418, 253, 315, 404, 274, 362
368, 349, 354, 367, 423, 330, 282, 444, 373, 456, 483, 257, 546, 321, 383, 258, 310, 410, 244, 356
423, 426, 464, 449, 504, 382, 357, 451, 408, 511, 537, 349, 619, 437, 444, 305, 390, 458, 388, 412
483, 459, 504, 456, 549, 431, 366, 522, 476, 586, 600, 342, 656, 482, 495, 321, 403, 482, 429, 409
429, 365, 395, 392, 409, 340, 288, 447, 426, 475, 477, 243, 442, 357, 331, 314, 360, 411, 369, 350
522, 475, 449, 385, 553, 386, 373, 466, 422, 489, 501, 337, 605, 467, 452, 325, 416, 449, 393, 361
484, 449, 407, 432, 503, 423, 350, 439, 450, 461, 514, 351, 536, 463, 394, 282, 429, 418, 432, 412
462, 470, 529, 424, 484, 431, 381, 545, 451, 585, 612, 348, 544, 455, 519, 265, 398, 434, 461, 434
329, 272, 312, 245, 366, 333, 289, 383, 280, 412, 429, 254, 459, 331, 357, 189, 283, 335, 208, 254
507, 544, 516, 529, 495, 420, 347, 595, 515, 632, 631, 441, 621, 427, 524, 270, 358, 474, 489, 491
298, 337, 290, 271, 307, 291, 252, 400, 288, 366, 382, 203, 361, 220, 298, 167, 217, 309, 228, 316
*/