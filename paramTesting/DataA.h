#pragma once
#include <stdlib.h>
#include <vector>

class DataA
{
	public:
		const int id = 1;
		float x, y;
		float length, width;
		int solidVal, outterVal;

		DataA(int xN, int yN, float dX, float dY)
		{
			x = rand() % (int)( (xN - 5) * dX );
			x += dX;
			y = rand() % (int)( (yN - 5) * dY );
			y += dY;
			width = rand() % (int)( dX * 4 );
			width += dX;
			length = rand() % (int)( dY * 4 );
			length += dY;
			solidVal = rand () % 3;
			solidVal += 2;
			outterVal = solidVal - 1;
		}

		void appendFloatsToBuffer(std::vector<float>& vec);
		void appendIntsToBuffer(std::vector<int>& vec);
};