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
		values.push_back( (0) );
		values.push_back( (0) );
		values.push_back( (0) );
	}

/*	template <typename X> Vector3(const Vector3<X>& newV)
	{
		for (int i = 0; i < 3; i++)
			values[i] = newV[i];
	}
//*/

	template <typename X> Vector3(const X a, const X b, const X c)
	{
		values.push_back(a);
		values.push_back(b);
		values.push_back(c);
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