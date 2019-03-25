#ifndef __QUICDATAFILE_QPPARAMS_H__
#define __QUICDATAFILE_QPPARAMS_H__ 1

#include <fstream>
#include "QUICDataFile.h"
#include "legacyFileParser.h"
#include <cassert>
///removing enum so as to support language_mapping 
// //////////////////////////////////////////////////////////////////
// 
// Class for holding the QP_Params.inp file
// 
// //////////////////////////////////////////////////////////////////
class qpParams : public quicDataFile
{
public:
  enum SourceType {
    BASIC = 1,
   DENSEGAS = 2,
  DISTPARTSIZE = 3,
    EXPLOSIVE = 4,
   ERADSOURCE = 5,
  BIOSLURRY = 6,
  TWOPHASE = 7,
    EXFILTRATION = 8    
  };
  
  qpParams()
  : quicDataFile(),
    isiteflag(0), 
    iindoorflag(0), 
    inextgridflag(1),
    westernEdge(0), 
    southernEdge(0),
    z0(0.0f), 
    rcl(0.0f), 
    boundaryLayerHeight(0.0),
    nonLocalMixing(1), 
    useCFDTurbulence(false), 
    numParticles(0)
  {}
  
  ~qpParams() {}
	 //copy constructor
  qpParams(const qpParams& other)
    {
      std::cerr<<"Copy constructor called"<<std::endl;
      *this = other;

    }

  //overloaded assignment
  qpParams& operator= (const qpParams& other);


 // SourceType sourceFlag;
  int sourceFlag;  ///replaced with those values   
	 ///  BASIC = 1,
	 ///   DENSEGAS = 2,
	 ///   DISTPARTSIZE = 3,
	 ///   EXPLOSIVE = 4,
	 ///   ERADSOURCE = 5,
	 ///   BIOSLURRY = 6,
	 ///   TWOPHASE = 7,
	   /// EXFILTRATION = 8    

  short isiteflag;   // !normal QUIC (isitefl=0) or sensor siting (=1) mode
  bool iindoorflag; // !indoor calculations turned off (=0) or turned on (=1)
  short inextgridflag;      // !1 - inner grid, 2 - outer grid
  float westernEdge;  // !Location of western edge of inner grid relative to outer grid (m)
  float southernEdge; // !Location of southern edge of inner relative to outer grid (m)
  float z0;  // wallSurfRoughness;    // !Wall Surface Roughness Length (m)
  float rcl;  // !Reciprocal Monin-Obukhov length(1/m)
  float boundaryLayerHeight;  // !Boundary Layer height (m)
  bool nonLocalMixing;        // !use 1 to enable non-local mixing
  bool useCFDTurbulence;        // !use 1 to enable use of QUIC-CFD turbulence
  int numParticles;          // !number of particles released over entire simulation
  short particleDistFlag;     // !Number of particle distribution flag (1 = by mass, 2 = by source)
  bool particleSplitFlag;     // !Particle splitting flag
  bool particleRecyclingFlag; // !Particle recycling flag
  int partNumIncreaseFactor;  // !Total particle number increase factor
  short numParticleSplit;     // !Number of particles a particle is split into
  double partSplittingDosage; // !Particle splitting target dose (gs/m^3)
  float taylorMicroscaleMin;  // !Enable Taylor microscale lower limit to sub-time steps
  int randomNumberSeed;  // !Random number seed
  double timeStep;       // !time step (s)
  double duration;       // !duration (s)
  double concAvgTime;   // !concentration averaging time (s)
  double concStartTime; // !starting time for concentration averaging (s)
  double partOutputPeriod; // !particle output period (s)
  float nbx;  // !in x direction, # of collecting boxes (concentration grid cells) 
  float nby;  // !in y direction, # of collecting boxes (concentration grid cells) 
  float nbz;  // !in z direction, # of collecting boxes (concentration grid cells) 
  float xbl;  // !lower limits for collecting boxes in x in meters
  float xbu;  // !upper limits for collecting boxes in x direction in meters
  float ybl;  // !lower limits for collecting boxes in y in meters
  float ybu;  // !upper limits for collecting boxes in y direction in meters
  float zbl;  // !lower limits for collecting boxes in z in meters
  float zbu;  // !upper limits for collecting boxes in z direction in meters

  bool readQUICFile(const std::string &filename);
  bool writeQUICFile(const std::string &filename);
  void build_map();    
private:
};

#endif // #define __QUICDATA_QPPARAMS_H__ 1
