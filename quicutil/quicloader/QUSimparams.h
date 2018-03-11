#ifndef __QUICDATAFILE_QUSIMPARAMS_H__
#define __QUICDATAFILE_QUSIMPARAMS_H__ 1

#include <cstdlib>
#include <fstream>
#include "QUICDataFile.h"
#include "legacyFileParser.h"

// //////////////////////////////////////////////////////////////////
// 
// Class for holding the QU_simparams.inp file
// 
// //////////////////////////////////////////////////////////////////
class quSimParams : public quicDataFile
{
public:
  quSimParams();
  ~quSimParams() {}

  quSimParams(const quSimParams& other)
    {
      std::cerr<<"Copy constructor called"<<std::endl;
      *this = other;

    }

  //overloaded assignment
  quSimParams& operator= (const quSimParams& other);

  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);
  void build_map();
	
  // Check for discovery and default if necessary.		
  int nx;
  int ny;
  int nz;

  float dx;
  float dy;
  float dz;

  int vstretch;

  float start_time;
  float time_incr;
  int num_time_steps;
  int day_of_year;
		
  int utc_conversion;

  int roof_type;
  int upwind_type;
  int canyon_type;
  int intersection_flag;
		
  int max_iterations;
  int residual_reduction;
  int diffusion_flag;
  int diffusion_step;
		
  int domain_rotation;
  float utmx;
  float utmy;
		
  int utm_zone;
  int quic_cfd_type;
  int wake_type;

  int explosive_building_damage;
  int building_array_flag;   // added by 5.72

private:
};

#endif // #define __QUICDATAFILE_QUSIMPARAMS_H__
