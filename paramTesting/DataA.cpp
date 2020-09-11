#include "DataA.h"



void DataA::appendFloatsToBuffer(std::vector<float>& vec)
{
	vec.push_back(x);
	vec.push_back(y);
	vec.push_back(length);
	vec.push_back(width);
}

void DataA::appendIntsToBuffer(std::vector<int>& vec)
{
	vec.push_back(id);
	vec.push_back(solidVal);
	vec.push_back(outterVal);
}