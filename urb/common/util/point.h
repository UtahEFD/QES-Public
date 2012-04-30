/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Collect the idea of a point and provide operator functionality
*         for clearly code later.
*/

#ifndef INC_POINT_H
#define INC_POINT_H

#include <cmath>

class point 
{        
  public:

    float x;
    float y;
    float z;

  //Constructors
    point(void);
		point(point const& p);
    point(float const& _x, float const& _y, float const& _z);

  //Methods
		point operator -() const;

    point& operator +=(point const& p);
    point& operator +=(float const& s);

    point& operator -=(point const& p);
    point& operator -=(float const& s);

		point& operator =(point const& p);

    point operator +(point const& q) const;
    point operator +(float const& s) const;

    point operator -(point const& q) const;
    point operator -(float const& s) const;

    float operator *(point const& q) const;
    point operator *(float const& s) const;

    point operator /(point const& q) const;
    point operator /(float const& s) const;

    bool operator <(point const& q) const;
    bool operator >(point const& q) const;

		bool operator==(point const& q) const;
		bool operator!=(point const& q) const;
};

point operator *(float const& s, point const& q);
point operator -(float const& s, point const& q);
point operator +(float const& s, point const& q);
float mag(point const& p);

#endif

