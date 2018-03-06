/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Display CUDA errors when they occur.
* Remark: cutil.h may have something like this.
*/

#ifndef SHOWERROR
#define SHOWERROR 1

#include <stdio.h>

#define SHOW_CUDA_ERRORS 1

extern "C"
void showError(char const* loc) 
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) 
	{
		printf("CUDA Error in %s: %s.\n", loc, cudaGetErrorString(err)); 
		fflush(stdout);
	}		
}

#endif
