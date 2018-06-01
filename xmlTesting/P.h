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
		parsePrimative<std::string>(str, "name");
		parsePrimative<int>(x, "xVal");
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

	P2()
	{
		str = "";
		c = 0;
		f = 0.0f;
	}

	void parseValues()
	{
		parsePrimative<std::string>(str, "name");
		parsePrimative<char>(c, "cVal");
		parsePrimative<float>(f, "fVal");
	}

	virtual void output()
	{
		std::cout << "P2\n" << str << std::endl << c << std::endl << f << std::endl;  
	}
};