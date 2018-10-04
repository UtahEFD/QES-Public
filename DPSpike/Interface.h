#pragma once
#include <cstdlib>
#include <cstdio>


#define S 20 //Width/Height of the matrix
#define STOTAL (S * S) //number of all elements
#define BX 5 //length of the grid of blocks
#define BY 2 //height of the grid of blocks
#define TX 5 //length of the block of threads
#define TY 4 //height of the block of threads
#define TZ 2 //depth of the block of threads
#define THREADS (TX * TY * TZ) //Number of threads in one block

class Interface
{
public:
	static void printMatrix(long* matrix);
	void multiplyMatricies_Wrapper( const long* M, const long* N, long* P);

};