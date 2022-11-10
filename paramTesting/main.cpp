#include "DataA.h"
#include "DataB.h"
#include "kernelizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>


using std::vector;

bool isInt(char* c)
{
	int count = 0;
	while (c[count] != 0)
		if (c[count] >= '0' && c[count] <= '9')
			count++;
		else
			return false;

	return count != 0;
}

int main(int argc, char* args[])
{
	
	srand(time(NULL));
	int numDataA;

	if (argc != 2)
	{
		printf("Error, must supply exactly 1 parameter\n");
		return -1;
	}
	else if ( !isInt(args[1]) )
	{
		printf("Error, parameter 1 must be an int\n");
		return -2;	
	}

	numDataA = atoi(args[1]);

	vector<DataA> datAs;

	for (int i = 0; i < numDataA; i++)
		datAs.push_back( DataA(X, Y, DIMX, DIMY) );

	doTheGPU(datAs);




	return 0;
}