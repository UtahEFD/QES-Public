#pragma once

#include "ParseInterface.h"

template <class T>
class Vector3 : public ParseInterface
{
protected:
	std::vector<T> values;
public:
	
	virtual void parseValues()
	{
		parseTaglessValues<T>(values);
	}

	int& operator[](const int i)
	{
		return values[i%3];
	}
};