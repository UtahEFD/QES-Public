#include "normal.h"

namespace sivelab
{
  const normal normal::UNIT  = normal(1., 1., 1.);
  const normal normal::X_POS = normal(1., 0., 0.);
  const normal normal::Y_POS = normal(0., 1., 0.);
  const normal normal::Z_POS = normal(0., 0., 1.);

  void normal::normalize(point& p) 
  {
    normal::normalize(p.x, p.y, p.z);
  }

  void normal::normalize(double& x, double& y, double& z)
  {
    double vecLength = sqrt(x*x + y*y + z*z);
    if (vecLength == 0.)
    {
      x = 0.;
      y = 0.;
      z = 1.;
    }
    else
    {
      x /= vecLength;
      y /= vecLength;
      z /= vecLength;
    }
  }

  normal::normal(void)
  : x(0.),
    y(0.),
    z(1.)
  {}

  normal::normal(normal const& n)
  : x(n.x),
    y(n.y),
    z(n.z) 
  {}

  normal::normal(point const& p)
  : x(p.x), 
    y(p.y), 
    z(p.z)
  {
    normal::normalize(x, y, z);
  }

  normal::normal(double const& newX, double const& newY, double const& newZ) 
  : x(newX),
    y(newY),
    z(newZ)
  {
    // Defaulting of normal to UP.
    normal::normalize(x, y, z);
  }

  double normal::getX() const {return x;}
  double normal::getY() const {return y;}
  double normal::getZ() const {return z;}

  normal::operator point() const 
  {
    return point(x, y, z);
  }

  normal normal::operator -() const 
  {
    return normal(-x, -y, -z);
  }

  normal normal::operator /(normal const& m) const
  {
    return normal(y*m.z - z*m.y, z*m.x - x*m.z, x*m.y - y*m.x);
  }

  normal normal::operator /(point const& p) const
  {
    return normal(y*p.z - z*p.y, z*p.x - x*p.z, x*p.y - y*p.x);
  }

  normal operator /(point const& p, normal const& m)
  {
    return m.operator/(p);
  }
  
  point normal::operator*(float const& s) const
  {
    return point(x * s, y * s, z * s);
  }
}

