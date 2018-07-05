#pragma once

#include "ParseInterface.h"

template <class T>
class Vector3 : public ParseInterface
{
protected:
	std::vector<T> values;
public:
	
	Vector3()
	{
		for (int i = 0; i < 3; i++)
			values[i] = 0;	
	}

	template <typename X> Vector3(const Vector3<X>& newV)
	{
		for (int i = 0; i < 3; i++)
			values[i] = newV[i];
	}

	virtual void parseValues()
	{
		parseTaglessValues<T>(values);
	}

	T& operator[](const int i)
	{
		return values[i%3];
	}
};