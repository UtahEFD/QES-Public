/*
 * Author: Andrew Larson <lars2865@d.umn.edu>
 * Reason: Part of the QUIC system. This class puts meteorological and 
 * 				engineering angles together in one class so that angles can be
 *					specified in either system and requested in either system as
 *					degrees or radians.
 */

#ifndef ANGLE
#define ANGLE

#include <cmath>
#include <iostream> 

#include "normal.h"

namespace sivelab
{
  // Note that the labels are rounded to the nearest degree.
  // Ex. R_023 is really 22.5 degrees in radians.
  // May want to rename these as 'AR_XXX'.
  // Common angles (in radians) that will be used by at least this class.
  static const float R_000 =  .000*M_PI;
  static const float R_023 =  .125*M_PI;
  static const float R_045 =  .250*M_PI;
  static const float R_068 =  .375*M_PI;
  static const float R_090 =  .500*M_PI;
  static const float R_113 =  .625*M_PI;
  static const float R_135 =  .750*M_PI;
  static const float R_158 =  .875*M_PI;
  static const float R_180 = 1.000*M_PI;

  static const float R_270 = 1.500*M_PI;
  static const float R_360 = 2.000*M_PI;
  static const float R_011 = .0625*M_PI;

  static const float TO_RADS = M_PI / 180.;
  static const float TO_DEGS = 180. / M_PI;

  // By default an angle should be specified in terms
  // of radians in the ENGINEERING system.
  // This needs to be integrated into
  // the existing QUIC code.
  // To specify something in the METEOROLOGICAL
  // system in degrees is should look something like:

  // angle new_angle = angle(0, DEG, MET);
  // That's wind from due north.

  // angle new_angle = angle(0);
  // That's wind goind due east.

  // 180. should not be a possible angle since 180. = -180. in ENGINEERING
  // Does this cause problems?
  // Watch that boundary. Look there when problems are found with this class.

  // Direction the wind comes from or goes to, depending on the system used.
  // Directions could be pushed to another file.
  enum Direction
    {	
      NORTH, 
      NORTHNORTHEAST,
      NORTHEAST, 
      EASTNORTHEAST,
		
      EAST,
      EASTSOUTHEAST,		  
      SOUTHEAST, 
      SOUTHSOUTHEAST,
		
      SOUTH, 
      SOUTHSOUTHWEST,
      SOUTHWEST, 
      WESTSOUTHWEST,
		
      WEST,
      WESTNORTHWEST,
      NORTHWEST,
      NORTHNORTHWEST,
		
      NOTSUBORDINAL
    };

  enum Unit   {RAD, DEG}; // Radians or Degrees
  enum System {ENG, MET}; // Engineering or Meteorological

  /*
   * The angle class stores angles internally as radians in the engineering
   * system, which is limited to the interval [-PI, PI) with 0 being the
   * default value for an angle.
   *
   * Angles can be specified or requested as either degrees or radians and in
   * either the engineering system or the meteorological system, which is
   * limited to the interval [0, 360). When specifying angles using directions
   * it is important to note that a direction in the meteorological system
   * means the wind is coming from that direction. In constrast, in the 
   * engineering system the direction specifies which way the wind is going
   * to. For instance, ENG EAST = MET WEST.
   *
   * Default angle should be EAST in ENG. (Zero)
   *
   * Be careful comparing meteorological to engineering. This wouldn't work well
   * due to the nature of the internal representation. They both do no have a 
   * common "zero" from which to start. Even meteorological to meteorological 
   * may cause problems due to the internal representation, i.e. 
   * 80 DEG MET !< 100 DEG MET, since internally 80 DEG MET = 170 DEG ENG and
   * 100 DEG MET = -170 DEG ENG, which would make 100 DEG MET < 80 DEG MET, which
   * is not what you want. Maybe comparisons are not needed. Use directionQ 
   * instead...
   */
  class angle
  {
  public:
    // Basic ENG angles. (Engineering)
    static const angle E_000;
    static const angle E_010;
    static const angle E_015;
    static const angle E_030;
    static const angle E_045;
    static const angle E_090;
    static const angle E_135;
    static const angle E_180;
    static const angle E_270;
    static const angle E_360;
		
    // Basic MET angles. (Meteorological)
    static const angle M_000;
    static const angle M_010;
    static const angle M_015;
    static const angle M_030;
    static const angle M_045;
    static const angle M_090;
    static const angle M_135;
    static const angle M_180;
    static const angle M_270;
    static const angle M_360;
	
  public:
		
    angle();
    angle(Direction const& d,	System const& s = ENG);
    angle
    (
     float _angle_size, 
     Unit const& u = RAD, 
     System const& s = ENG
     );
    virtual ~angle();

    System system() const;
    // Maybe want a way to set the system...
    // void setSystem(System const& s);

    // Get or set the radians in a specified system.		
    float radians(System const& s = ENG) const;
    void setRadians(float, System const& s = ENG);
		
    // Get or set the degrees in a specified system.
    float degrees(System const& s = ENG) const;
    void setDegrees(float, System const& s = ENG);

    // Get or set the angle in a specified system.
    // the closest subordinal direction is returned.
    Direction direction(System const& s = ENG) const;
    void setDirection(Direction const& d, System const& s = ENG);		  
		  
    // Query whether or not an angle is in a particular direction.
    bool inDirQ
    (
     Direction const& d,
     angle const& tol = angle(R_011),		  	 
     System const& s = ENG
     ) const;
		  
    //normal directionAsNormal(System const& s = ENG) const;
		  
    // Test whether an angle is within tolerance of a 
    // particular direction set.
    bool isPoleQ(angle const& tol) const;
    bool isCardinalQ(angle const& tol) const;
    bool isOrdinalQ(angle const& tol) const;
    bool isSubOrdinalQ(angle const& tol) const;

    angle  operator-() const;
    angle& operator=(angle const&);
		
    angle  operator+ (angle const&) const;
    angle& operator+=(angle const&);		

    angle  operator- (angle const&) const;
    angle& operator-=(angle const&);

    float  operator*(angle const&) const; // dot product
		
    angle operator*(float const&) const;
    angle operator*(int   const&) const;
		
    angle& operator*=(float const&);
    angle& operator*=(int   const&);
		
    angle operator/(float const&) const;
    angle operator/(int   const&) const;
		
    angle& operator/=(float const&);
    angle& operator/=(int   const&);
		
    bool operator> (angle const&) const;
    bool operator>=(angle const&) const;
		
    bool operator< (angle const&) const;	
    bool operator<=(angle const&) const;
		
    bool operator==(angle const&) const;
		
    // Decide on a default?
    //friend std::istream& operator>>(std::istream&, angle);
    friend std::ostream& operator<<(std::ostream&, angle const&);

  protected:
	
    float rads;
    System sstm;
		  
    static float contain(float a, System const& s = ENG);
    static Direction opposite(Direction const& d);
			
    // Givin an angle and tolerance, these functions indicate whether
    // the wind is a certain direction in the engineering system.
    // For internal use only.
    static bool isN  (float const& a, float const& tol);
    static bool isNNE(float const& a, float const& tol);
    static bool isNE (float const& a, float const& tol);
    static bool isENE(float const& a, float const& tol);
			
    static bool isE  (float const& a, float const& tol);
    static bool isESE(float const& a, float const& tol);
    static bool isSE (float const& a, float const& tol);
    static bool isSSE(float const& a, float const& tol);
			
    static bool isS  (float const& a, float const& tol);
    static bool isSSW(float const& a, float const& tol);
    static bool isSW (float const& a, float const& tol);
    static bool isWSW(float const& a, float const& tol);
			
    static bool isW  (float const& a, float const& tol);
    static bool isWNW(float const& a, float const& tol);
    static bool isNW (float const& a, float const& tol);
    static bool isNNW(float const& a, float const& tol);
			
			
  };

  angle operator*(float const&, angle const&);
  angle operator*(int   const&, angle const&);

  // Decide on a default?
  //std::istream& operator>>(std::istream&, angle);
  std::ostream& operator<<(std::ostream&, angle const&);

  float cos(angle const&);
  float sin(angle const&);
  float tan(angle const&);

}

#endif
