#include "test_DTEHeightField.h"

#include <string>
#include <cstdio>
#include <cstdlib>

int main()
{

	std::string results;
	/******************
	 * DTEHeightField * 
	 ******************/
	printf("starting DTEHeightField tests...\n");
	test_DTEHeightField DTEHF_test;
	results = DTEHF_test.mainTest();
	if (results == "")
	{
		printf("DTEHeightField: Success!\n");
	}
	else
	{
		printf("DTEHeightField: Failure\n%s\n", results.c_str());
		exit(EXIT_FAILURE);
	}


	printf("All tests pass!\n");
	exit(EXIT_SUCCESS);
	return 0;
}