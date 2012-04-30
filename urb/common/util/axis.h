/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Provide more data in a visualizer.
* Remark: At 60% of where it should be.
*/

#ifndef INC_AXIS_H
#define INC_AXIS_H

#include "normal.h"

class axis
{

	private:
		float strt;
		float fnsh;
		float ncrmnt;

		float r;
		float g;
		float b;
		float a;

		float lngth;
		point lctn;
		normal rnttn;
		normal r1;
		normal r2;

	public:
		axis(float start, float finish, float increment, point loc, normal ori);
		~axis();

		void glDraw();
		void setColor(float r, float g, float b);
		void setColor(float r, float g, float b, float a);
		void setLength(float len);
		void setLocation(point loc);
		void setOrientation(normal ori);
};


#endif
