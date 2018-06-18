#include "Interface.h"
#include <cstdio>
#include <cstdlib>
#include <time.h>



int main()
{
	//check to see if threads meet elements in a matrix
	if (S * S != BX * BY * THREADS)
	{
		printf("Thread count does not match the number of\n");
		printf("of elements in the matrix\n");
		return 1;
	}

	long* M = new long[STOTAL]; //primary matrix
	long* N = new long[STOTAL]; //secondary matrix
	long* P = new long[STOTAL]; //result matrix
	time_t t; //current time
	int i; //index of loops

	//set seed for rand()
	srand((unsigned) time(&t));

	//generate two random matricies of size S
	for (i = 0; i < S * S; i++)
	{
		M[i] = (long)(rand() % 10);
		N[i] = (long)(rand() % 10);
	}

	Interface inter;
	//Multiply matricies
	inter.multiplyMatricies_Wrapper(M, N, P);

		//print results
	printf("Primary Matrix\n");
	Interface::printMatrix(M);
	printf("\n\nSecondary Matrix\n");
	Interface::printMatrix(N);
	printf("\n\nResult Matrix\n");
	Interface::printMatrix(P);

	return 0;
}