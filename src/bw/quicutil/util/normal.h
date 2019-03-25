/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Encapsulate the idea of a normal.
*/

#ifndef INC_NORMAL_H
#define INC_NORMAL_H

#include <math.h>
#include "point.h"

namespace sivelab
{
  class normal
  {
    public:
      static const normal UNIT;
      static const normal X_POS;
      static const normal Y_POS;
      static const normal Z_POS;
    
		  static void normalize(point& p);
		  static void normalize(double& x, double& y, double& z);

	  public:
		  normal(void);
		  normal(point const& p);
		  normal(normal const& n);
		  normal(double const& x, double const& y, double const& z);
         
		  double getX() const;
		  double getY() const;
		  double getZ() const;

      operator point() const;

		  normal operator-() const;

		  normal operator/(normal const& m) const;
		  normal operator/(point const& p) const;

      point operator*(float const& s) const;

    protected:
      double x;
      double y;
      double z;
  };
}

#endif

