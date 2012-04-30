/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Encapsulate the idea of a normal.
*/

#ifndef INC_NORMAL_H
#define INC_NORMAL_H

#include <math.h>
#include "point.h"

class normal : public point
{
	public:
        
		normal(void);
		normal(point const& p);
		normal(normal const& n);
		normal(float const& _x, float const& _y, float const& _z);
       
		float getX() const;
		float getY() const;
		float getZ() const;

		normal operator -() const;
		normal& operator =(normal const& n);
		normal& operator =(point const& p);

		normal operator /(normal const& m) const;
		normal operator /(point const& p) const;

		static void normalize(point& p);
		static void normalize(float& x, float& y, float& z);

  protected:
    
    point::x;
    point::y;
    point::z;
};

//normal operator /(point const& p, normal const& m);

#endif

