#pragma once

#include "ParseInterface.h"
<<<<<<< HEAD
=======
#include <type_traits>

#define FLOATS_ARE_EQUAL(x,y) ( (x) - (y) > -0.0000001 && (x) - (y) < 0.0000001 )
>>>>>>> origin/doxygenAdd

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
		values.clear();
		parseTaglessValues<T>(values);
	}

	T& operator[](const int i)
	{
		return values[i%3];
	}

<<<<<<< HEAD
	bool operator==(Vector3<T>& v)
	{
		return v.values[0] == values[0] && v.values[1] == values[1] && v.values[2] == values[2];
=======
	bool operator==(const Vector3<T>& v)
	{
		if (std::is_same<T,float>::value || std::is_same<T,double>::value)
			return FLOATS_ARE_EQUAL(values[0], v.values[0]) && FLOATS_ARE_EQUAL(values[1], v.values[1]) && FLOATS_ARE_EQUAL(values[2], v.values[2]);
		else
			return v.values[0] == values[0] && v.values[1] == values[1] && v.values[2] == values[2];
>>>>>>> origin/doxygenAdd
	}
};