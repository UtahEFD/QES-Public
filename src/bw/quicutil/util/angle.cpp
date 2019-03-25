#include "angle.h"

namespace sivelab
{
  // Basic ENG angles. (Engineering)
  const angle angle::E_000 = angle(  0.f, DEG);
  const angle angle::E_010 = angle( 10.f, DEG);
  const angle angle::E_015 = angle( 15.f, DEG);
  const angle angle::E_030 = angle( 30.f, DEG);
  const angle angle::E_045 = angle( 45.f, DEG);
  const angle angle::E_090 = angle( 90.f, DEG);
  const angle angle::E_135 = angle(135.f, DEG);
  const angle angle::E_180 = angle(180.f, DEG);
  const angle angle::E_270 = angle(270.f, DEG);
  const angle angle::E_360 = angle(360.f, DEG);
		
  // Basic MET angles. (Meteorological)
  const angle angle::M_000 = angle(  0.f, DEG, MET);
  const angle angle::M_010 = angle( 10.f, DEG, MET);
  const angle angle::M_015 = angle( 15.f, DEG, MET);
  const angle angle::M_030 = angle( 30.f, DEG, MET);
  const angle angle::M_045 = angle( 45.f, DEG, MET);
  const angle angle::M_090 = angle( 90.f, DEG, MET);
  const angle angle::M_135 = angle(135.f, DEG, MET);
  const angle angle::M_180 = angle(180.f, DEG, MET);
  const angle angle::M_270 = angle(270.f, DEG, MET);
  const angle angle::M_360 = angle(360.f, DEG, MET);

  angle::angle()
  {
    // Engineering system assumed with wind from the east.
    sstm = ENG;
    rads = R_000;		
  }

  angle::angle(Direction const& d, System const& s)
  {
    sstm = s;
    this->setDirection(d, sstm);
  }

  angle::angle(float _angle_size, Unit const& u, System const& s)
  {	
    sstm = s;
    if(u == DEG) 
      {
	this->setDegrees(_angle_size, sstm);
      }
    else 
      {
	this->setRadians(_angle_size, sstm);
      }
  }

  angle::~angle() {}

  System angle::system() const {return sstm;}

  float angle::contain(float radians, System const& s)
  {
    if(s == ENG) {radians += R_180;}
		
    radians = fmod(radians, R_360);
		
    if(radians < R_000) {radians += R_360;}
    if(radians > R_360) {radians -= R_360;}
		
    if(s == ENG) {radians -= R_180;}

    return radians;
  }
		
  float angle::radians(System const& s) const
  {
    if(s == MET)
      {
	// Then convert back to meteorological
	if(-R_090 <= rads && rads <= -R_180)
	  {
	    return contain(-(rads + R_090), MET);
	  }
	else
	  {
	    return contain(R_270 - rads, MET);
	  }
      }
    else
      {
	return rads; // should already be contained.
      }
  }

  void angle::setRadians(float _radians, System const& s)
  {
    if(s == MET)
      {
	_radians = contain(_radians, MET);
			
	// If meteorological, then convert to engineering.
	if(R_000 <= _radians && _radians <= R_090)
	  {
	    rads = -(_radians + R_090);
	  }
	else
	  {
	    rads = R_270 - _radians;
	  }
      }
    else
      {
	rads = _radians;
      }

    rads = contain(rads, ENG);
  }
		
  float angle::degrees(System const& s) const 
  {
    return this->radians(s) * TO_DEGS;
  }

  void angle::setDegrees(float _degrees, System const& s)
  {
    this->setRadians(_degrees * TO_RADS, s);
  }

  Direction angle::direction(System const& s) const
  {
    float tol = R_011; // Splits circle into 16 pieces. 11.25 degrees.
	
    Direction eng_dir = NOTSUBORDINAL;
	
    if(isN(rads, tol)) {eng_dir = NORTH;} // ENG North
    if(isS(rads, tol)) {eng_dir = SOUTH;} // ENG South
    if(isE(rads, tol)) {eng_dir = EAST;}  // ENG East
    if(isW(rads, tol)) {eng_dir = WEST;}  // ENG West  
		
    if(isNE(rads, tol)) {eng_dir = NORTHEAST;} // ENG NE
    if(isSE(rads, tol)) {eng_dir = SOUTHEAST;} // ENG SE
    if(isSW(rads, tol)) {eng_dir = SOUTHWEST;} // ENG SW
    if(isNW(rads, tol)) {eng_dir = NORTHWEST;} // ENG NW

    if(isNNE(rads, tol)) {eng_dir = NORTHNORTHEAST;} // ENG NNE
    if(isENE(rads, tol)) {eng_dir = EASTNORTHEAST;}  // ENG ENE
    if(isESE(rads, tol)) {eng_dir = EASTSOUTHEAST;}  // ENG ESE
    if(isSSE(rads, tol)) {eng_dir = SOUTHSOUTHEAST;} // ENG SSE
    if(isSSW(rads, tol)) {eng_dir = SOUTHSOUTHWEST;} // ENG SSW
    if(isWSW(rads, tol)) {eng_dir = WESTSOUTHWEST;}  // ENG WSW
    if(isWNW(rads, tol)) {eng_dir = WESTNORTHWEST;}  // ENG WNW
    if(isNNW(rads, tol)) {eng_dir = NORTHNORTHWEST;} // ENG NNW
		
    if(eng_dir == NOTSUBORDINAL)
      {
	std::cout 
	  << 
	  "angle::getDirection() - you shouldn't get here" 
	  << 
	  std::endl;
	std::cout << "rads = " << rads << std::endl;
	std::cout << "returning 'NOTSUBORDINAL'." << std::endl;
      }
		
    return (s == MET) ? opposite(eng_dir) : eng_dir ;
  }


  void angle::setDirection(Direction const& d, System const& s)
  {
    // if meteorological, then translate
    Direction eng_dir = (s == MET) ? opposite(d) : d ;
	
    switch(eng_dir)
      {
	case NORTH:          rads =  R_090; break;
	case NORTHNORTHEAST: rads =  R_068; break;
	case NORTHEAST:      rads =  R_045; break;
	case EASTNORTHEAST:  rads =  R_023; break;
		  
	case EAST:           rads =  R_000; break;
	case EASTSOUTHEAST:  rads = -R_023; break;
	case SOUTHEAST:      rads = -R_045; break;
	case SOUTHSOUTHEAST: rads = -R_068; break;
		  
	case SOUTH:          rads = -R_090; break;
	case SOUTHSOUTHWEST: rads = -R_113; break;
	case SOUTHWEST:      rads = -R_135; break;
	case WESTSOUTHWEST:  rads = -R_158; break;
		  
	case WEST:           rads = -R_180; break;
	case WESTNORTHWEST:  rads =  R_158; break;
	case NORTHWEST:      rads =  R_135; break;
	case NORTHNORTHWEST: rads =  R_113; break;
		  
	default:             rads = R_000; 
      }
  }
	
  bool angle::inDirQ
  (
   Direction const& d, 
   angle const& tolerance, 
   System const& s
   ) const
  {
    float tol = fabs(tolerance.radians(sstm));
  	
    Direction eng_dir = (s == MET) ? opposite(d) : d ;
  
    switch(eng_dir)
      {
	case NORTH:          return isN  (rads, tol);
	case NORTHNORTHEAST: return isNNE(rads, tol);
	case NORTHEAST:      return isNE (rads, tol);
	case EASTNORTHEAST:  return isENE(rads, tol);
		  
	case EAST:           return isE  (rads, tol);
	case EASTSOUTHEAST:  return isESE(rads, tol);
	case SOUTHEAST:      return isSE (rads, tol);
	case SOUTHSOUTHEAST: return isSSE(rads, tol);
		  
	case SOUTH:          return isS  (rads, tol);
	case SOUTHSOUTHWEST: return isSSW(rads, tol);
	case SOUTHWEST:      return isSW (rads, tol);
	case WESTSOUTHWEST:  return isWSW(rads, tol);
		  
	case WEST:           return isW  (rads, tol);
	case WESTNORTHWEST:  return isWNW(rads, tol);
	case NORTHWEST:      return isNW (rads, tol);
	case NORTHNORTHWEST: return isNNW(rads, tol);
		  
	default: return false;
      }
  }
  /*
    normal angle::directionAsNormal(System const& s) const
    {
    float radians = this->radians(s);
  
    return normal(::cos(radians), ::sin(radians), 0.);
    }
  */
  bool angle::isPoleQ(angle const& tolerance) const
  {
    float tol = fabs(tolerance.rads);
		
    if(tol > R_090) {return true;}
		
    if(isN(rads, tol)) {return true;} // ENG North
    if(isS(rads, tol)) {return true;} // ENG South
		
    return false;
  }

  bool angle::isCardinalQ(angle const& tolerance) const
  {

    float tol = fabs(tolerance.rads);

    if(tol > R_045) {return true;}
		
    if(isE(rads, tol)) {return true;} // ENG East
    if(isW(rads, tol)) {return true;} // ENG West  
		
    return isPoleQ(tol);
  }

  bool angle::isOrdinalQ(angle const& tolerance) const
  {
    // Northwest in ENG
    float tol = fabs(tolerance.rads);
		
    if(tol > R_023) {return true;}
		
    if(isNE(rads, tol)) {return true;} // ENG NE
    if(isSE(rads, tol)) {return true;} // ENG SE
    if(isSW(rads, tol)) {return true;} // ENG SW
    if(isNW(rads, tol)) {return true;} // ENG NW
		
    return isCardinalQ(tol) || isPoleQ(tol); 
  }

  bool angle::isSubOrdinalQ(angle const& tolerance) const
  {
    float tol = fabs(tolerance.rads);
	
    if(tol > R_011) {return true;}

    if(isNNE(rads, tol)) {return true;} // ENG NNE
    if(isENE(rads, tol)) {return true;} // ENG ENE
    if(isESE(rads, tol)) {return true;} // ENG ESE
    if(isSSE(rads, tol)) {return true;} // ENG SSE
    if(isSSW(rads, tol)) {return true;} // ENG SSW
    if(isWSW(rads, tol)) {return true;} // ENG WSW
    if(isWNW(rads, tol)) {return true;} // ENG WNW
    if(isNNW(rads, tol)) {return true;} // ENG NNW
	
    return isOrdinalQ(tol) || isCardinalQ(tol) || isPoleQ(tol);
  }

  angle angle::operator-() const 
  {
    return angle(-this->radians(sstm), RAD, sstm);
  }

  angle& angle::operator=(angle const& a)
  {
    rads = a.rads;
    sstm = a.sstm;
    return *this;
  }

  angle  angle::operator+ (angle const& a) const 
  {		
    return angle(radians(sstm) + a.radians(a.sstm), RAD, sstm);		
  }

  angle& angle::operator+=(angle const& a) 
  {
    this->setRadians(this->radians(sstm) + a.radians(a.sstm), sstm);
    return *this;
  }

  angle angle::operator-(angle const& a) const 
  {
    return angle(this->radians(sstm) - a.radians(a.sstm), RAD, sstm);
  }

  angle& angle::operator-=(angle const& a) 
  {
    this->setRadians(this->radians(sstm) - a.radians(a.sstm), sstm);
    return *this;
  }

  float angle::operator*(angle const& a) const
  {
    float x1 = sin(this->radians(sstm));
    float y1 = cos(this->radians(sstm));
	
    float x2 = sin(a.radians(a.sstm));
    float y2 = cos(a.radians(a.sstm));
	
    return x1*x2 + y1*y2;
  }

  angle angle::operator*(float const& f) const 
  {
    return angle(this->radians(sstm) * f, RAD, sstm);
  }
  angle angle::operator*(int   const& f) const 
  {
    return angle(this->radians(sstm) * f, RAD, sstm);
  }

  angle& angle::operator*=(float const& f) 
  {
    this->setRadians(this->radians(sstm)*f, sstm);
    return *this;
  }
  angle& angle::operator*=(int   const& f) 
  {
    this->setRadians(this->radians(sstm)*f, sstm);
    return *this;
  }

  angle angle::operator/(float const& f) const 
  {
    return angle((f == 0.) ? 0. : this->radians(sstm) / f, RAD, sstm );
  }
  angle angle::operator/(int   const& f) const 
  {
    return angle((f == 0 ) ? 0  : this->radians(sstm) / f, RAD, sstm );
  }

  angle& angle::operator/=(float const& f) 
  {
    this->setRadians(this->radians(sstm) / (f == 0.) ? 1. : f, sstm); 
    return *this;
  }
  angle& angle::operator/=(int   const& f) 
  {
    this->setRadians(this->radians(sstm) / (f == 0 ) ? 1  : f, sstm);
    return *this;
  }

  bool angle::operator> (angle const& a) const 
  {
    if(this->sstm == a.sstm)
      {
	return this->radians(this->sstm) > a.radians(a.sstm);
      }
    else
      {
	return false;
      }
  }
  bool angle::operator>=(angle const& a) const 
  {
    if(sstm == a.sstm)
      {
	return radians(sstm) >= a.radians(a.sstm);
      }
    else
      {
	return false;
      }
  }

  bool angle::operator< (angle const& a) const 
  {
    if(this->sstm == a.sstm)
      {
	return this->radians(this->sstm) < a.radians(a.sstm);
      }
    else
      {
	return false;
      }
  }
  bool angle::operator<=(angle const& a) const
  {
    if(this->sstm == a.sstm)
      {
	return this->radians(this->sstm) <= a.radians(a.sstm);
      }
    else
      {
	return false;
      }
  }

  bool angle::operator==(angle const& a) const 
  {
    return rads == a.rads;
  }

  angle operator*(float const& a, angle const& b) {return b*a;}
  angle operator*(int   const& a, angle const& b) {return b*a;}
	
  ///////////////
  // protected //
  ///////////////

  Direction angle::opposite(Direction const& d)
  {
    switch(d)
      {
	case NORTH:     		 return SOUTH;
	case NORTHNORTHEAST: return SOUTHSOUTHWEST;
	case EASTNORTHEAST:  return WESTSOUTHWEST;
	case NORTHEAST:			 return SOUTHWEST;
			  
	case EAST:      		 return WEST;
	case SOUTHEAST: 		 return NORTHWEST;
	case EASTSOUTHEAST:  return WESTNORTHWEST;
	case SOUTHSOUTHEAST: return NORTHNORTHWEST;
	 		  
	case SOUTH:     		 return NORTH;
	case SOUTHWEST: 		 return NORTHEAST;
	case SOUTHSOUTHWEST: return NORTHNORTHEAST;
	case WESTSOUTHWEST:  return EASTNORTHEAST;

	case WEST:      		 return EAST;
	case NORTHWEST: 		 return SOUTHEAST;
	case WESTNORTHWEST:  return EASTSOUTHEAST;
	case NORTHNORTHWEST: return SOUTHSOUTHEAST;
	 		  
	default: 						 return NOTSUBORDINAL; // 0
      }
  }

  bool angle::isN(float const& a, float const& tol)
  {
    // ENG North
    return ( R_090 - tol <= a && a <= R_090 + tol) ? true : false ;
  }
  bool angle::isNNE(float const& a, float const& tol)
  {
    // ENG NNE
    return ( R_068 - tol <  a && a <  R_068 + tol) ? true : false ;
  }
  bool angle::isNE(float const& a, float const& tol)
  {
    // ENG NE
    return ( R_045 - tol <= a && a <= R_045 + tol) ? true : false ;
  }
  bool angle::isENE(float const& a, float const& tol)
  {
    // ENG ENE
    return ( R_023 - tol <  a && a <  R_023 + tol) ? true : false ;
  }
	
  bool angle::isE(float const& a, float const& tol)
  {
    // ENG East
    return ( R_000 - tol <= a && a <=  R_000 + tol) ? true : false;
  }
  bool angle::isESE(float const& a, float const& tol)
  {
    // ENG ESE
    return (-R_023 - tol <  a && a <  -R_023 + tol) ? true : false ;
  }
  bool angle::isSE(float const& a, float const& tol)
  {
    // ENG SE
    return (-R_045 - tol <= a && a <= -R_045 + tol) ? true : false ;
  }
  bool angle::isSSE(float const& a, float const& tol)
  {
    // ENG SSE
    return (-R_068 - tol <  a && a <  -R_068 + tol) ? true : false ; 
  }
	
  bool angle::isS(float const& a, float const& tol)
  {
    // ENG South
    return (-R_090 - tol <= a && a <= -R_090 + tol) ? true : false ;
  }
  bool angle::isSSW(float const& a, float const& tol)
  {
    // ENG SSW
    return (-R_113 - tol <  a && a <  -R_113 + tol) ? true : false ; 
  }
  bool angle::isSW(float const& a, float const& tol)
  {
    // ENG SW
    return (-R_135 - tol <= a && a <= -R_135 + tol) ? true : false ;
  }
  bool angle::isWSW(float const& a, float const& tol)
  {
    // ENG WSW
    return (-R_158 - tol <  a && a <  -R_158 + tol) ? true : false ;
  }
	
  bool angle::isW(float const& a, float const& tol)
  {
    // ENG West  
    return ( R_180 - tol <= a || a <= -R_180 + tol) ? true : false ;
  }
  bool angle::isWNW(float const& a, float const& tol)
  {
    // ENG WNW
    return ( R_158 - tol <  a && a <   R_158 + tol) ? true : false ;
  }
  bool angle::isNW(float const& a, float const& tol)
  {
    // ENG NW
    return ( R_135 - tol <= a && a <=  R_135 + tol) ? true : false ;
  }
  bool angle::isNNW(float const& a, float const& tol)
  {
    // ENG NNW
    return ( R_113 - tol <  a && a <   R_113 + tol) ? true : false ;
  }
	

  // Be sure to scope away from this class.
  float cos(angle const& a)	
  {
    return (a.system() == ENG) ? ::cos(a.radians()) : ::cos(a.radians(MET)) ;
  }
  float sin(angle const& a)	
  {
    return (a.system() == ENG) ? ::sin(a.radians()) : ::sin(a.radians(MET)) ;
  }
  float tan(angle const& a)	
  {
    return (a.system() == ENG) ? ::tan(a.radians()) : ::tan(a.radians(MET)) ;
  }


  /*
    std::istream& operator>>(std::istream& input, angle a)
    {
    input >> a.rads;
    a.rads = angle::contain(a.rads);
    return input;
    }
  */

  std::ostream& operator<<(std::ostream& output, angle const& a)
  {
    output << ((a.sstm == ENG) ? a.degrees() : a.degrees(MET)) << " ";
    output << ((a.sstm == ENG) ? "ENG"       : "MET"         ) << " ";
    output << std::flush;
    return output;
  }


}
