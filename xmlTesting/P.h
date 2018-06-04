#pragma once

#include "ParseInterface.h"

#include <iostream>

class P : public ParseInterface
{
public:

	virtual void output() = 0;
};

class P1 : public P
{
public:
	std::string str;
	int x;

	P1()
	{
		str = "";
		x = 0;
	}

	void parseValues()
	{
		parsePrimative<std::string>(true, str, "name");
		parsePrimative<int>(true, x, "xVal");
	}

	virtual void output()
	{
		std::cout << "P1\n" << str << std::endl << x << std::endl;  
	}
};

class P2 : public P
{
public:
	std::string str;
	char c;
	float f;
	float q;

	P2()
	{
		str = "";
		c = 0;
		f = 0.0f;
		q = 0.0f;
	}

	void parseValues()
	{
		parsePrimative<std::string>(true, str, "name");
		parsePrimative<char>(true, c, "cVal");
		parsePrimative<float>(true, f, "fVal");
		parsePrimative<float>(true, q, "qVal");
	}

	virtual void output()
	{
		std::cout << "P2\n" << str << std::endl << c << std::endl << f << std::endl;  
	}
};